from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a historical team season summary from historical staging tables. "
            "This is read-only and does not modify the database."
        )
    )
    parser.add_argument("--season", required=True, help="Historical season key, for example 2024_25.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--out-json", default=None, help="Optional metadata/report JSON path.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def table_exists(table_name: str) -> bool:
    db = SessionLocal()
    try:
        value = db.execute(
            text(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = :table_name
                """
            ),
            {"table_name": table_name},
        ).scalar()
        return int(value or 0) > 0
    finally:
        db.close()


def read_sql_dataframe(sql: str, params: Dict[str, Any]) -> pd.DataFrame:
    db = SessionLocal()
    try:
        return pd.read_sql(text(sql), db.bind, params=params)
    finally:
        db.close()


def safe_divide(numerator: Any, denominator: Any) -> Optional[float]:
    if denominator is None or pd.isna(denominator) or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def fixture_records_for_team_summary(fixtures: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for _, row in fixtures.iterrows():
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        home_score_int = int(home_score)
        away_score_int = int(away_score)

        if home_score_int > away_score_int:
            home_points = 3
            away_points = 0
            home_result = "W"
            away_result = "L"
        elif home_score_int < away_score_int:
            home_points = 0
            away_points = 3
            home_result = "L"
            away_result = "W"
        else:
            home_points = 1
            away_points = 1
            home_result = "D"
            away_result = "D"

        records.append(
            {
                "season": row["season"],
                "raw_team_id": row["raw_home_team_id"],
                "gw": row["gw"],
                "is_home": True,
                "goals_for": home_score_int,
                "goals_against": away_score_int,
                "points": home_points,
                "win": 1 if home_result == "W" else 0,
                "draw": 1 if home_result == "D" else 0,
                "loss": 1 if home_result == "L" else 0,
                "clean_sheet": 1 if away_score_int == 0 else 0,
            }
        )
        records.append(
            {
                "season": row["season"],
                "raw_team_id": row["raw_away_team_id"],
                "gw": row["gw"],
                "is_home": False,
                "goals_for": away_score_int,
                "goals_against": home_score_int,
                "points": away_points,
                "win": 1 if away_result == "W" else 0,
                "draw": 1 if away_result == "D" else 0,
                "loss": 1 if away_result == "L" else 0,
                "clean_sheet": 1 if home_score_int == 0 else 0,
            }
        )

    return pd.DataFrame(records)


def build_team_summary(season: str) -> pd.DataFrame:
    required_tables = ["historical_teams", "historical_fixtures"]
    missing_tables = [table_name for table_name in required_tables if not table_exists(table_name)]
    if missing_tables:
        raise RuntimeError("Missing required historical staging tables: %s" % missing_tables)

    teams = read_sql_dataframe(
        """
        SELECT
            season,
            raw_team_id,
            raw_team_name,
            raw_team_short_name,
            canonical_team_id,
            canonical_team_name,
            mapping_status,
            mapping_confidence,
            notes
        FROM historical_teams
        WHERE season = :season
        """,
        {"season": season},
    )

    fixtures = read_sql_dataframe(
        """
        SELECT
            season,
            raw_fixture_id,
            gw,
            raw_home_team_id,
            raw_away_team_id,
            finished,
            home_score,
            away_score
        FROM historical_fixtures
        WHERE season = :season
        """,
        {"season": season},
    )

    if teams.empty:
        raise RuntimeError("No historical_teams rows found for season=%s." % season)
    if fixtures.empty:
        raise RuntimeError("No historical_fixtures rows found for season=%s." % season)

    fixtures["gw"] = pd.to_numeric(fixtures["gw"], errors="coerce")
    fixtures["home_score"] = pd.to_numeric(fixtures["home_score"], errors="coerce")
    fixtures["away_score"] = pd.to_numeric(fixtures["away_score"], errors="coerce")

    scheduled_home = fixtures.groupby(["season", "raw_home_team_id"], dropna=False).size().reset_index(name="home_fixtures")
    scheduled_home = scheduled_home.rename(columns={"raw_home_team_id": "raw_team_id"})
    scheduled_away = fixtures.groupby(["season", "raw_away_team_id"], dropna=False).size().reset_index(name="away_fixtures")
    scheduled_away = scheduled_away.rename(columns={"raw_away_team_id": "raw_team_id"})
    scheduled = scheduled_home.merge(scheduled_away, on=["season", "raw_team_id"], how="outer")
    scheduled["home_fixtures"] = scheduled["home_fixtures"].fillna(0).astype(int)
    scheduled["away_fixtures"] = scheduled["away_fixtures"].fillna(0).astype(int)
    scheduled["scheduled_fixtures"] = scheduled["home_fixtures"] + scheduled["away_fixtures"]

    finished_fixtures = fixtures[(fixtures["home_score"].notna()) & (fixtures["away_score"].notna())].copy()
    team_match_records = fixture_records_for_team_summary(finished_fixtures)

    if team_match_records.empty:
        aggregated = pd.DataFrame(columns=["season", "raw_team_id"])
    else:
        aggregated = (
            team_match_records.groupby(["season", "raw_team_id"], dropna=False)
            .agg(
                matches=("gw", "count"),
                home_matches=("is_home", "sum"),
                goals_for=("goals_for", "sum"),
                goals_against=("goals_against", "sum"),
                points=("points", "sum"),
                wins=("win", "sum"),
                draws=("draw", "sum"),
                losses=("loss", "sum"),
                clean_sheets=("clean_sheet", "sum"),
            )
            .reset_index()
        )
        aggregated["away_matches"] = aggregated["matches"] - aggregated["home_matches"]
        aggregated["goal_difference"] = aggregated["goals_for"] - aggregated["goals_against"]

    summary = teams.merge(
        scheduled[["season", "raw_team_id", "scheduled_fixtures", "home_fixtures", "away_fixtures"]],
        on=["season", "raw_team_id"],
        how="left",
    )
    summary = summary.merge(aggregated, on=["season", "raw_team_id"], how="left")

    integer_columns = [
        "scheduled_fixtures",
        "home_fixtures",
        "away_fixtures",
        "matches",
        "home_matches",
        "away_matches",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "goal_difference",
        "points",
        "clean_sheets",
    ]
    for column in integer_columns:
        if column not in summary.columns:
            summary[column] = 0
        summary[column] = summary[column].fillna(0).astype(int)

    summary["points_per_match"] = summary.apply(lambda row: safe_divide(row["points"], row["matches"]), axis=1)
    summary["goals_for_per_match"] = summary.apply(lambda row: safe_divide(row["goals_for"], row["matches"]), axis=1)
    summary["goals_against_per_match"] = summary.apply(lambda row: safe_divide(row["goals_against"], row["matches"]), axis=1)
    summary["clean_sheet_rate"] = summary.apply(lambda row: safe_divide(row["clean_sheets"], row["matches"]), axis=1)

    for column in ["points_per_match", "goals_for_per_match", "goals_against_per_match", "clean_sheet_rate"]:
        summary[column] = summary[column].round(4)

    ordered_columns = [
        "season",
        "raw_team_id",
        "raw_team_name",
        "raw_team_short_name",
        "canonical_team_id",
        "canonical_team_name",
        "mapping_status",
        "mapping_confidence",
        "scheduled_fixtures",
        "matches",
        "home_matches",
        "away_matches",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "goal_difference",
        "points",
        "points_per_match",
        "goals_for_per_match",
        "goals_against_per_match",
        "clean_sheets",
        "clean_sheet_rate",
        "notes",
    ]
    existing_ordered_columns = [column for column in ordered_columns if column in summary.columns]
    summary = summary[existing_ordered_columns].sort_values(
        by=["points", "goal_difference", "goals_for", "raw_team_name"], ascending=[False, False, False, True]
    )

    return summary


def build_report(summary: pd.DataFrame, season: str, out_csv: str) -> Dict[str, Any]:
    unmapped_count = int(summary["canonical_team_id"].isna().sum()) if "canonical_team_id" in summary.columns else None
    report = {
        "created_at": utc_now(),
        "season": season,
        "out_csv": out_csv,
        "row_count": int(len(summary)),
        "unmapped_team_count": unmapped_count,
        "total_matches_counted_twice_by_team": int(summary["matches"].sum()) if "matches" in summary.columns else None,
        "total_points": int(summary["points"].sum()) if "points" in summary.columns else None,
        "top_teams_by_points": summary.head(10)[
            ["raw_team_id", "raw_team_name", "raw_team_short_name", "matches", "points", "points_per_match", "goal_difference"]
        ].to_dict(orient="records"),
        "notes": [
            "This exporter is read-only.",
            "Team identity remains historical/raw unless canonical_team_id has been mapped.",
            "matches counts completed fixtures with both scores present.",
            "total_matches_counted_twice_by_team should be 760 for a full 380-fixture EPL season.",
        ],
    }
    return report


def save_json(report: Dict[str, Any], out_json: Optional[str]) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("saved_report:", path)


def main() -> None:
    args = parse_args()
    summary = build_team_summary(args.season)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    report = build_report(summary, args.season, str(out_csv))
    save_json(report, args.out_json)

    print("=== Historical Team Season Summary ===")
    print("season:", args.season)
    print("row_count:", report["row_count"])
    print("unmapped_team_count:", report["unmapped_team_count"])
    print("total_matches_counted_twice_by_team:", report["total_matches_counted_twice_by_team"])
    print("total_points:", report["total_points"])
    print("saved_csv:", out_csv)


if __name__ == "__main__":
    main()
