import os
import pandas as pd
from sqlalchemy import create_engine, text

from ml.features.team_context import (
    build_team_context_from_team_fixture,
    build_team_fixture_context,
)


def _build_team_fixture_rows(fixtures: pd.DataFrame) -> pd.DataFrame:
    home = fixtures[
        [
            "gw",
            "kickoff_time",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
        ]
    ].copy()
    home = home.rename(
        columns={
            "home_team_id": "team_id",
            "away_team_id": "opponent_team_id",
            "home_score": "goals_for",
            "away_score": "goals_against",
        }
    )
    home["is_home"] = 1.0

    away = fixtures[
        [
            "gw",
            "kickoff_time",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
        ]
    ].copy()
    away = away.rename(
        columns={
            "away_team_id": "team_id",
            "home_team_id": "opponent_team_id",
            "away_score": "goals_for",
            "home_score": "goals_against",
        }
    )
    away["is_home"] = 0.0

    team_fixture = pd.concat([home, away], ignore_index=True)
    team_fixture = team_fixture[
        [
            "gw",
            "kickoff_time",
            "team_id",
            "opponent_team_id",
            "is_home",
            "goals_for",
            "goals_against",
        ]
    ].copy()

    team_fixture["kickoff_time"] = pd.to_datetime(team_fixture["kickoff_time"], utc=True)
    team_fixture = team_fixture.sort_values(["team_id", "kickoff_time"]).reset_index(drop=True)
    return team_fixture


def export_features_v2_1(start_gw: int, end_gw: int, out_csv: str) -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            'DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"'
        )

    engine = create_engine(db_url)

    with engine.begin() as conn:
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

        fixtures_history = pd.read_sql(
            text(
                '''
                SELECT
                  gw,
                  kickoff_time,
                  home_team_id,
                  away_team_id,
                  home_score,
                  away_score
                FROM fixtures
                WHERE gw BETWEEN 1 AND :end_gw
                  AND kickoff_time IS NOT NULL
                  AND finished = TRUE
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                ORDER BY kickoff_time ASC
                '''
            ),
            conn,
            params={"end_gw": end_gw},
        )

    g = df.groupby("player_id", group_keys=False)

    df["pts_last1"] = g["total_points"].shift(1)
    df["mins_last1"] = g["minutes"].shift(1)

    for w in (3, 5, 8):
        df[f"pts_roll{w}_mean"] = g["total_points"].shift(1).rolling(w, min_periods=1).mean()
    for w in (3, 5):
        df[f"mins_roll{w}_mean"] = g["minutes"].shift(1).rolling(w, min_periods=1).mean()

    df["mins_roll5_nonzero_rate"] = (
        g["minutes"]
        .shift(1)
        .rolling(5, min_periods=1)
        .apply(lambda x: (pd.Series(x) > 0).mean())
    )

    df["g_roll5_sum"] = g["goals_scored"].shift(1).rolling(5, min_periods=1).sum()
    df["a_roll5_sum"] = g["assists"].shift(1).rolling(5, min_periods=1).sum()
    df["cs_roll5_sum"] = g["clean_sheets"].shift(1).rolling(5, min_periods=1).sum()

    df["recent_avg_minutes"] = g["minutes"].shift(1).rolling(5, min_periods=1).mean()
    df["recent_mins_60_plus_count"] = (
        g["minutes"]
        .shift(1)
        .rolling(5, min_periods=1)
        .apply(lambda x: (pd.Series(x) >= 60).sum())
    )

    # Day56 intrinsic / availability proxies
    df["recent_zero_min_count"] = (
        g["minutes"]
        .shift(1)
        .rolling(5, min_periods=1)
        .apply(lambda x: (pd.Series(x) == 0).sum())
    )

    df["recent_90_plus_count"] = (
        g["minutes"]
        .shift(1)
        .rolling(5, min_periods=1)
        .apply(lambda x: (pd.Series(x) >= 90).sum())
    )

    df["recent_start_like_rate"] = (
        g["minutes"]
        .shift(1)
        .rolling(5, min_periods=1)
        .apply(lambda x: (pd.Series(x) >= 60).mean())
    )

    df["minutes_std_roll5"] = (
        g["minutes"]
        .shift(1)
        .rolling(5, min_periods=2)
        .std()
    )

    pts_roll3 = g["total_points"].shift(1).rolling(3, min_periods=1).mean()
    pts_roll5 = g["total_points"].shift(1).rolling(5, min_periods=1).mean()
    mins_roll3 = g["minutes"].shift(1).rolling(3, min_periods=1).mean()
    mins_roll5 = g["minutes"].shift(1).rolling(5, min_periods=1).mean()

    df["points_trend_last3_minus_last5"] = pts_roll3 - pts_roll5
    df["minutes_trend_last3_minus_last5"] = mins_roll3 - mins_roll5

    prev_zero = (g["minutes"].shift(1) == 0).fillna(False)
    prev_positive = (g["minutes"].shift(2) > 0).fillna(False)
    df["return_from_absence_flag"] = (prev_zero & prev_positive).astype(int)

    def _matches_since_return(values):
        vals = list(values)
        if not vals:
            return None
        for idx in range(len(vals) - 1, -1, -1):
            if vals[idx] == 0:
                return len(vals) - idx - 1
        return len(vals)

    df["matches_since_return"] = (
        g["minutes"]
        .shift(1)
        .rolling(8, min_periods=1)
        .apply(_matches_since_return)
    )

    df["minutes_drop_recent"] = mins_roll5 - mins_roll3

    df["now_cost_m"] = df["now_cost"] / 10.0

    if not fixtures_history.empty:
        team_fixture = _build_team_fixture_rows(fixtures_history)

        team_context = build_team_context_from_team_fixture(team_fixture)
        team_fixture_context = build_team_fixture_context(team_fixture)

        own_context_cols = [
            "team_id",
            "gw",
            "fixtures_in_gw",
            "home_ratio_gw",
            "team_rank_before_gw",
            "team_points_per_match_before_gw",
            "team_goal_diff_per_match_before_gw",
            "team_goals_scored_avg_recent",
            "team_goals_conceded_avg_recent",
            "team_attack_strength_recent",
            "team_defense_weakness_recent",
        ]

        team_fixture_context = team_fixture_context.merge(
            team_context[own_context_cols],
            on=["team_id", "gw"],
            how="left",
        )

        team_gw_context = (
            team_fixture_context.groupby(["team_id", "gw"], as_index=False)
            .agg(
                fixtures_in_gw=("fixtures_in_gw", "first"),
                home_ratio_gw=("home_ratio_gw", "first"),
                team_rank_before_gw=("team_rank_before_gw", "first"),
                team_points_per_match_before_gw=("team_points_per_match_before_gw", "first"),
                team_goal_diff_per_match_before_gw=("team_goal_diff_per_match_before_gw", "first"),
                team_goals_scored_avg_recent=("team_goals_scored_avg_recent", "first"),
                team_goals_conceded_avg_recent=("team_goals_conceded_avg_recent", "first"),
                team_attack_strength_recent=("team_attack_strength_recent", "first"),
                team_defense_weakness_recent=("team_defense_weakness_recent", "first"),
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
        df["team_goals_scored_avg_recent"] = pd.NA
        df["opponent_goals_scored_avg_recent"] = pd.NA
        df["team_goals_conceded_avg_recent"] = pd.NA
        df["opponent_goals_conceded_avg_recent"] = pd.NA
        df["fixture_difficulty_proxy"] = pd.NA
        df["fixture_difficulty_for_attackers"] = pd.NA
        df["fixture_difficulty_for_defenders"] = pd.NA
        df["team_attack_strength_recent"] = pd.NA
        df["team_defense_weakness_recent"] = pd.NA

    df["is_home_gw"] = (df["home_ratio_gw"].fillna(0.0) >= 0.5).astype(int)

    df = df.dropna(subset=["pts_last1", "mins_last1"]).reset_index(drop=True)

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

    export_features_v2_1(args.start_gw, args.end_gw, args.out)


if __name__ == "__main__":
    main()
