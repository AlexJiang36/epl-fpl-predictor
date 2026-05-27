import os
import pandas as pd
from sqlalchemy import create_engine, text


def export_features_v2(start_gw: int, end_gw: int, out_csv: str) -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            'DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"'
        )

    engine = create_engine(db_url)

    with engine.begin() as conn:
        # Base player GW rows (final exported grain stays player_id x gw)
        # Pull rows only for the requested export range.
        df = pd.read_sql(
            text(
                '''
                SELECT
                  s.player_id,
                  s.gw,
                  s.minutes,
                  s.goals_scored,
                  s.assists,
                  s.clean_sheets,
                  s.total_points,
                  p.position,
                  p.now_cost,
                  p.status,
                  p.team_id
                FROM player_gw_stats s
                JOIN players p ON p.id = s.player_id
                WHERE s.gw BETWEEN :start_gw AND :end_gw
                ORDER BY s.player_id, s.gw
                '''
            ),
            conn,
            params={"start_gw": start_gw, "end_gw": end_gw},
        )

        if df.empty:
            raise RuntimeError(f"No rows found in player_gw_stats for gw range [{start_gw}, {end_gw}]")

        # Team-fixture rows.
        # IMPORTANT: pull all fixtures from gw 1..end_gw (not only start_gw..end_gw),
        # so "before_gw" standings / rolling context for start_gw are computed correctly.
        team_fixture = pd.read_sql(
            text(
                '''
                WITH team_fixture_rows AS (
                    SELECT
                        f.gw,
                        f.home_team_id AS team_id,
                        f.away_team_id AS opponent_team_id,
                        1.0 AS is_home,
                        f.home_score AS goals_for,
                        f.away_score AS goals_against
                    FROM fixtures f
                    WHERE f.gw BETWEEN 1 AND :end_gw
                      AND f.finished = TRUE

                    UNION ALL

                    SELECT
                        f.gw,
                        f.away_team_id AS team_id,
                        f.home_team_id AS opponent_team_id,
                        0.0 AS is_home,
                        f.away_score AS goals_for,
                        f.home_score AS goals_against
                    FROM fixtures f
                    WHERE f.gw BETWEEN 1 AND :end_gw
                      AND f.finished = TRUE
                )
                SELECT
                    gw,
                    team_id,
                    opponent_team_id,
                    is_home,
                    goals_for,
                    goals_against
                FROM team_fixture_rows
                ORDER BY team_id, gw, opponent_team_id
                '''
            ),
            conn,
            params={"end_gw": end_gw},
        )

    # ------------------------------------------------------
    # Player rolling features (v0 + v2 minute stability)
    # ------------------------------------------------------
    g = df.groupby("player_id", group_keys=False)

    # Lag
    df["pts_last1"] = g["total_points"].shift(1)
    df["mins_last1"] = g["minutes"].shift(1)

    # Rolling means
    for w in (3, 5, 8):
        df[f"pts_roll{w}_mean"] = g["total_points"].shift(1).rolling(w, min_periods=1).mean()
    for w in (3, 5):
        df[f"mins_roll{w}_mean"] = g["minutes"].shift(1).rolling(w, min_periods=1).mean()

    # Nonzero minutes rate (last 5, shifted)
    df["mins_roll5_nonzero_rate"] = (
        g["minutes"]
        .shift(1)
        .rolling(5, min_periods=1)
        .apply(lambda x: (pd.Series(x) > 0).mean())
    )

    # Event rolling sums
    df["g_roll5_sum"] = g["goals_scored"].shift(1).rolling(5, min_periods=1).sum()
    df["a_roll5_sum"] = g["assists"].shift(1).rolling(5, min_periods=1).sum()
    df["cs_roll5_sum"] = g["clean_sheets"].shift(1).rolling(5, min_periods=1).sum()

    # Day54 minute-stability features
    df["recent_avg_minutes"] = g["minutes"].shift(1).rolling(5, min_periods=1).mean()
    df["recent_mins_60_plus_count"] = (
        g["minutes"]
        .shift(1)
        .rolling(5, min_periods=1)
        .apply(lambda x: (pd.Series(x) >= 60).sum())
    )

    # Static transforms
    df["now_cost_m"] = df["now_cost"] / 10.0

    # ------------------------------------------------------
    # Team/GW context features from finished fixtures
    # ------------------------------------------------------
    if not team_fixture.empty:
        team_fixture["points_gw_single"] = 0
        team_fixture.loc[team_fixture["goals_for"] > team_fixture["goals_against"], "points_gw_single"] = 3
        team_fixture.loc[team_fixture["goals_for"] == team_fixture["goals_against"], "points_gw_single"] = 1

        # Team x GW aggregates
        team_gw = (
            team_fixture.groupby(["team_id", "gw"], as_index=False)
            .agg(
                fixtures_in_gw=("opponent_team_id", "count"),
                home_ratio_gw=("is_home", "mean"),
                team_goals_for_gw=("goals_for", "mean"),
                team_goals_against_gw=("goals_against", "mean"),
                team_goals_for_sum_gw=("goals_for", "sum"),
                team_goals_against_sum_gw=("goals_against", "sum"),
                team_points_gw=("points_gw_single", "sum"),
            )
            .sort_values(["team_id", "gw"])
            .reset_index(drop=True)
        )

        tg = team_gw.groupby("team_id", group_keys=False)

        # Recent rolling strength features (all shifted by 1 to stay leakage-safe)
        team_gw["team_attack_strength_recent"] = (
            tg["team_goals_for_gw"].shift(1).rolling(5, min_periods=1).mean()
        )
        team_gw["team_defense_weakness_recent"] = (
            tg["team_goals_against_gw"].shift(1).rolling(5, min_periods=1).mean()
        )

        # Ranking / standings context BEFORE current gw
        team_gw["team_points_before_gw"] = tg["team_points_gw"].cumsum().shift(1)
        team_gw["team_matches_before_gw"] = tg["fixtures_in_gw"].cumsum().shift(1)
        team_gw["team_goal_diff_before_gw"] = (
            (tg["team_goals_for_sum_gw"].cumsum() - tg["team_goals_against_sum_gw"].cumsum()).shift(1)
        )

        team_gw["team_points_per_match_before_gw"] = (
            team_gw["team_points_before_gw"] / team_gw["team_matches_before_gw"]
        )
        team_gw["team_goal_diff_per_match_before_gw"] = (
            team_gw["team_goal_diff_before_gw"] / team_gw["team_matches_before_gw"]
        )

        # Rank within each gw using only before-gw standings metrics
        rank_frame = team_gw[
            [
                "team_id",
                "gw",
                "team_points_per_match_before_gw",
                "team_goal_diff_per_match_before_gw",
                "team_points_before_gw",
            ]
        ].copy()

        rank_frame["team_points_per_match_before_gw_rank_sort"] = rank_frame[
            "team_points_per_match_before_gw"
        ].fillna(-9999.0)
        rank_frame["team_goal_diff_per_match_before_gw_rank_sort"] = rank_frame[
            "team_goal_diff_per_match_before_gw"
        ].fillna(-9999.0)
        rank_frame["team_points_before_gw_rank_sort"] = rank_frame[
            "team_points_before_gw"
        ].fillna(-9999.0)

        rank_frame = rank_frame.sort_values(
            [
                "gw",
                "team_points_per_match_before_gw_rank_sort",
                "team_goal_diff_per_match_before_gw_rank_sort",
                "team_points_before_gw_rank_sort",
                "team_id",
            ],
            ascending=[True, False, False, False, True],
        ).reset_index(drop=True)

        rank_frame["team_rank_before_gw"] = rank_frame.groupby("gw").cumcount() + 1
        rank_frame = rank_frame[["team_id", "gw", "team_rank_before_gw"]]

        team_gw = team_gw.merge(rank_frame, on=["team_id", "gw"], how="left")

        # Opponent features from the opponent team's own before-gw / recent context
        opp_context = team_gw[
            [
                "team_id",
                "gw",
                "team_rank_before_gw",
                "team_points_per_match_before_gw",
                "team_goal_diff_per_match_before_gw",
                "team_attack_strength_recent",
                "team_defense_weakness_recent",
            ]
        ].rename(
            columns={
                "team_id": "opponent_team_id",
                "team_rank_before_gw": "opponent_rank_before_gw_single",
                "team_points_per_match_before_gw": "opponent_points_per_match_before_gw_single",
                "team_goal_diff_per_match_before_gw": "opponent_goal_diff_per_match_before_gw_single",
                "team_attack_strength_recent": "opponent_goals_scored_avg_recent_single",
                "team_defense_weakness_recent": "opponent_goals_conceded_avg_recent_single",
            }
        )

        team_fixture_v2 = team_fixture.merge(
            opp_context,
            on=["opponent_team_id", "gw"],
            how="left",
        )

        # Keep old single proxy for continuity
        team_fixture_v2["fixture_difficulty_proxy_single"] = 1.0 / (
            team_fixture_v2["opponent_goals_conceded_avg_recent_single"].fillna(1.0) + 0.5
        )

        # New clearer directional difficulty features
        # Higher value = more difficult
        team_fixture_v2["fixture_difficulty_for_attackers_single"] = 1.0 / (
            team_fixture_v2["opponent_goals_conceded_avg_recent_single"].fillna(1.0) + 0.5
        )
        team_fixture_v2["fixture_difficulty_for_defenders_single"] = (
            team_fixture_v2["opponent_goals_scored_avg_recent_single"].fillna(0.0)
        )

        team_gw_context = (
            team_fixture_v2.groupby(["team_id", "gw"], as_index=False)
            .agg(
                fixtures_in_gw=("opponent_team_id", "count"),
                home_ratio_gw=("is_home", "mean"),
                opponent_rank_before_gw=("opponent_rank_before_gw_single", "mean"),
                opponent_points_per_match_before_gw=("opponent_points_per_match_before_gw_single", "mean"),
                opponent_goal_diff_per_match_before_gw=("opponent_goal_diff_per_match_before_gw_single", "mean"),
                opponent_goals_scored_avg_recent=("opponent_goals_scored_avg_recent_single", "mean"),
                opponent_goals_conceded_avg_recent=("opponent_goals_conceded_avg_recent_single", "mean"),
                fixture_difficulty_proxy=("fixture_difficulty_proxy_single", "mean"),
                fixture_difficulty_for_attackers=("fixture_difficulty_for_attackers_single", "mean"),
                fixture_difficulty_for_defenders=("fixture_difficulty_for_defenders_single", "mean"),
            )
        )

        team_gw_context = team_gw_context.merge(
            team_gw[
                [
                    "team_id",
                    "gw",
                    "team_rank_before_gw",
                    "team_points_per_match_before_gw",
                    "team_goal_diff_per_match_before_gw",
                    "team_attack_strength_recent",
                    "team_defense_weakness_recent",
                ]
            ],
            on=["team_id", "gw"],
            how="left",
        )

        # Merge team/gw context onto player rows
        df = df.merge(
            team_gw_context,
            on=["team_id", "gw"],
            how="left",
        )
    else:
        df["fixtures_in_gw"] = pd.NA
        df["home_ratio_gw"] = pd.NA
        df["team_rank_before_gw"] = pd.NA
        df["opponent_rank_before_gw"] = pd.NA
        df["team_points_per_match_before_gw"] = pd.NA
        df["opponent_points_per_match_before_gw"] = pd.NA
        df["team_goal_diff_per_match_before_gw"] = pd.NA
        df["opponent_goal_diff_per_match_before_gw"] = pd.NA
        df["opponent_goals_scored_avg_recent"] = pd.NA
        df["opponent_goals_conceded_avg_recent"] = pd.NA
        df["fixture_difficulty_proxy"] = pd.NA
        df["fixture_difficulty_for_attackers"] = pd.NA
        df["fixture_difficulty_for_defenders"] = pd.NA
        df["team_attack_strength_recent"] = pd.NA
        df["team_defense_weakness_recent"] = pd.NA

    # Binary home/away signal for normal single-fixture GWs
    df["is_home_gw"] = (df["home_ratio_gw"].fillna(0.0) >= 0.5).astype(int)

    # Drop rows that cannot have leakage-safe lag features
    df = df.dropna(subset=["pts_last1", "mins_last1"]).reset_index(drop=True)

    # Note:
    # previous-season priors are intentionally NOT added yet in this v2.1 export,
    # because current source tables do not yet encode season explicitly.

    # One-hot encode categoricals (ridge-friendly)
    df = pd.get_dummies(df, columns=["position", "status"], drop_first=False)

    df.to_csv(out_csv, index=False)
    print(f"OK: wrote {len(df)} rows -> {out_csv}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start_gw", type=int, required=True)
    ap.add_argument("--end_gw", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    export_features_v2(args.start_gw, args.end_gw, args.out)


if __name__ == "__main__":
    main()
