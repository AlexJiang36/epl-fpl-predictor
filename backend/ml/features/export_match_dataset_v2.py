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
            "fixture_id",
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
            "fixture_id",
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
            "fixture_id",
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
    team_fixture = team_fixture.sort_values(["team_id", "kickoff_time", "fixture_id"]).reset_index(drop=True)
    return team_fixture


def _add_rest_days(team_fixture_context: pd.DataFrame) -> pd.DataFrame:
    tfc = team_fixture_context.copy()
    tfc = tfc.sort_values(["team_id", "kickoff_time", "fixture_id"]).reset_index(drop=True)
    tfc["prev_kickoff_time"] = tfc.groupby("team_id")["kickoff_time"].shift(1)
    tfc["rest_days_before_fixture"] = (
        (tfc["kickoff_time"] - tfc["prev_kickoff_time"]).dt.total_seconds() / 86400.0
    )
    return tfc


def _compute_h2h_features(fixtures_history: pd.DataFrame, target_fixtures: pd.DataFrame, n_h2h: int) -> pd.DataFrame:
    hist = fixtures_history.copy()
    hist["kickoff_time"] = pd.to_datetime(hist["kickoff_time"], utc=True)
    hist = hist.sort_values("kickoff_time").reset_index(drop=True)

    rows = []
    for _, fx in target_fixtures.iterrows():
        fixture_id = int(fx["fixture_id"])
        before_ts = fx["kickoff_time"]
        home_id = int(fx["home_team_id"])
        away_id = int(fx["away_team_id"])

        mask = (
            (hist["kickoff_time"] < before_ts)
            & (
                (
                    (hist["home_team_id"] == home_id)
                    & (hist["away_team_id"] == away_id)
                )
                | (
                    (hist["home_team_id"] == away_id)
                    & (hist["away_team_id"] == home_id)
                )
            )
        )

        h2h = hist.loc[mask].sort_values("kickoff_time", ascending=False).head(n_h2h)

        if h2h.empty:
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "h2h_matches_count_recent": 0,
                    "h2h_home_points_avg_recent": None,
                    "h2h_home_goal_diff_avg_recent": None,
                }
            )
            continue

        pts = []
        goal_diffs = []

        for _, r in h2h.iterrows():
            hs = int(r["home_score"])
            a_s = int(r["away_score"])

            if int(r["home_team_id"]) == home_id:
                gf, ga = hs, a_s
            else:
                gf, ga = a_s, hs

            if gf > ga:
                pts.append(3.0)
            elif gf == ga:
                pts.append(1.0)
            else:
                pts.append(0.0)

            goal_diffs.append(float(gf - ga))

        rows.append(
            {
                "fixture_id": fixture_id,
                "h2h_matches_count_recent": len(h2h),
                "h2h_home_points_avg_recent": sum(pts) / len(pts),
                "h2h_home_goal_diff_avg_recent": sum(goal_diffs) / len(goal_diffs),
            }
        )

    return pd.DataFrame(rows)


def export_match_dataset_v2(start_gw: int, end_gw: int, n_form: int, out_csv: str, n_h2h: int = 3) -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError('DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"')

    engine = create_engine(db_url)

    with engine.begin() as conn:
        target_fixtures = pd.read_sql(
            text(
                '''
                SELECT
                  id AS fixture_id,
                  gw,
                  kickoff_time,
                  home_team_id,
                  away_team_id,
                  home_score,
                  away_score
                FROM fixtures
                WHERE gw BETWEEN :start_gw AND :end_gw
                  AND kickoff_time IS NOT NULL
                  AND finished = TRUE
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                ORDER BY kickoff_time ASC
                '''
            ),
            conn,
            params={"start_gw": start_gw, "end_gw": end_gw},
        )

        if target_fixtures.empty:
            raise RuntimeError(f"No finished fixtures found in gw range [{start_gw}, {end_gw}]")

        history_fixtures = pd.read_sql(
            text(
                '''
                SELECT
                  id AS fixture_id,
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

    target_fixtures["kickoff_time"] = pd.to_datetime(target_fixtures["kickoff_time"], utc=True)
    history_fixtures["kickoff_time"] = pd.to_datetime(history_fixtures["kickoff_time"], utc=True)

    team_fixture = _build_team_fixture_rows(history_fixtures)
    team_context = build_team_context_from_team_fixture(team_fixture)
    team_fixture_context = build_team_fixture_context(team_fixture)

    own_context_cols = [
        "team_id",
        "gw",
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

    team_fixture_context = _add_rest_days(team_fixture_context)

    # Keep only target export range after context is fully built from history
    tfc_target = team_fixture_context[
        (team_fixture_context["gw"] >= start_gw) & (team_fixture_context["gw"] <= end_gw)
    ].copy()

    home_rows = tfc_target[tfc_target["is_home"] == 1.0].copy()
    away_rows = tfc_target[tfc_target["is_home"] == 0.0].copy()

    home_rows = home_rows.rename(
        columns={
            "team_id": "home_team_id",
            "opponent_team_id": "away_team_id",
            "team_rank_before_gw": "home_team_rank_before_gw",
            "team_points_per_match_before_gw": "home_points_per_match_before_gw",
            "team_goal_diff_per_match_before_gw": "home_goal_diff_per_match_before_gw",
            "team_goals_scored_avg_recent": "home_goals_scored_avg_recent",
            "team_goals_conceded_avg_recent": "home_goals_conceded_avg_recent",
            "team_attack_strength_recent": "home_attack_strength_recent",
            "team_defense_weakness_recent": "home_defense_weakness_recent",
            "rest_days_before_fixture": "rest_days_home",
            "fixture_difficulty_proxy_single": "home_fixture_difficulty_proxy",
            "fixture_difficulty_for_attackers_single": "home_fixture_difficulty_for_attackers",
            "fixture_difficulty_for_defenders_single": "home_fixture_difficulty_for_defenders",
        }
    )

    away_rows = away_rows.rename(
        columns={
            "team_id": "away_team_id",
            "opponent_team_id": "home_team_id",
            "team_rank_before_gw": "away_team_rank_before_gw",
            "team_points_per_match_before_gw": "away_points_per_match_before_gw",
            "team_goal_diff_per_match_before_gw": "away_goal_diff_per_match_before_gw",
            "team_goals_scored_avg_recent": "away_goals_scored_avg_recent",
            "team_goals_conceded_avg_recent": "away_goals_conceded_avg_recent",
            "team_attack_strength_recent": "away_attack_strength_recent",
            "team_defense_weakness_recent": "away_defense_weakness_recent",
            "rest_days_before_fixture": "rest_days_away",
            "fixture_difficulty_proxy_single": "away_fixture_difficulty_proxy",
            "fixture_difficulty_for_attackers_single": "away_fixture_difficulty_for_attackers",
            "fixture_difficulty_for_defenders_single": "away_fixture_difficulty_for_defenders",
        }
    )

    keep_home = [
        "fixture_id",
        "home_team_id",
        "away_team_id",
        "home_team_rank_before_gw",
        "home_points_per_match_before_gw",
        "home_goal_diff_per_match_before_gw",
        "home_goals_scored_avg_recent",
        "home_goals_conceded_avg_recent",
        "home_attack_strength_recent",
        "home_defense_weakness_recent",
        "rest_days_home",
        "home_fixture_difficulty_proxy",
        "home_fixture_difficulty_for_attackers",
        "home_fixture_difficulty_for_defenders",
    ]
    keep_away = [
        "fixture_id",
        "home_team_id",
        "away_team_id",
        "away_team_rank_before_gw",
        "away_points_per_match_before_gw",
        "away_goal_diff_per_match_before_gw",
        "away_goals_scored_avg_recent",
        "away_goals_conceded_avg_recent",
        "away_attack_strength_recent",
        "away_defense_weakness_recent",
        "rest_days_away",
        "away_fixture_difficulty_proxy",
        "away_fixture_difficulty_for_attackers",
        "away_fixture_difficulty_for_defenders",
    ]

    home_rows = home_rows[keep_home].copy()
    away_rows = away_rows[keep_away].copy()

    out = target_fixtures.merge(
        home_rows,
        on=["fixture_id", "home_team_id", "away_team_id"],
        how="left",
    ).merge(
        away_rows,
        on=["fixture_id", "home_team_id", "away_team_id"],
        how="left",
    )

    # Derived comparison features
    out["strength_diff_recent"] = (
        out["home_attack_strength_recent"] - out["away_attack_strength_recent"]
    )
    out["points_per_match_diff_before_gw"] = (
        out["home_points_per_match_before_gw"] - out["away_points_per_match_before_gw"]
    )
    out["goal_diff_per_match_diff_before_gw"] = (
        out["home_goal_diff_per_match_before_gw"] - out["away_goal_diff_per_match_before_gw"]
    )
    out["rank_diff_before_gw"] = (
        out["away_team_rank_before_gw"] - out["home_team_rank_before_gw"]
    )

    # Goals foundation targets
    out["home_goals"] = out["home_score"]
    out["away_goals"] = out["away_score"]

    def make_label(row) -> str:
        hs = int(row["home_score"])
        aw = int(row["away_score"])
        if hs > aw:
            return "H"
        if hs < aw:
            return "A"
        return "D"

    out["result_label"] = out.apply(make_label, axis=1)

    # Limited H2H signals
    h2h = _compute_h2h_features(history_fixtures, target_fixtures, n_h2h=n_h2h)
    out = out.merge(h2h, on="fixture_id", how="left")

    cols = [
        "fixture_id",
        "gw",
        "kickoff_time",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "home_goals",
        "away_goals",
        "result_label",
        "home_team_rank_before_gw",
        "away_team_rank_before_gw",
        "home_points_per_match_before_gw",
        "away_points_per_match_before_gw",
        "home_goal_diff_per_match_before_gw",
        "away_goal_diff_per_match_before_gw",
        "home_goals_scored_avg_recent",
        "away_goals_scored_avg_recent",
        "home_goals_conceded_avg_recent",
        "away_goals_conceded_avg_recent",
        "home_attack_strength_recent",
        "away_attack_strength_recent",
        "home_defense_weakness_recent",
        "away_defense_weakness_recent",
        "rest_days_home",
        "rest_days_away",
        "home_fixture_difficulty_proxy",
        "away_fixture_difficulty_proxy",
        "home_fixture_difficulty_for_attackers",
        "away_fixture_difficulty_for_attackers",
        "home_fixture_difficulty_for_defenders",
        "away_fixture_difficulty_for_defenders",
        "strength_diff_recent",
        "points_per_match_diff_before_gw",
        "goal_diff_per_match_diff_before_gw",
        "rank_diff_before_gw",
        "h2h_matches_count_recent",
        "h2h_home_points_avg_recent",
        "h2h_home_goal_diff_avg_recent",
    ]

    out = out[cols].copy()
    out.to_csv(out_csv, index=False)
    print(f"OK: wrote {len(out)} rows -> {out_csv}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start_gw", type=int, required=True)
    ap.add_argument("--end_gw", type=int, required=True)
    ap.add_argument("--n_form", type=int, default=5)
    ap.add_argument("--n_h2h", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    export_match_dataset_v2(
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        n_form=args.n_form,
        out_csv=args.out,
        n_h2h=args.n_h2h,
    )


if __name__ == "__main__":
    main()
