from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text

from app.core.db import SessionLocal


TABLES_WITH_SEASON = [
    "gameweeks",
    "fixtures",
    "player_gw_stats",
    "predictions",
    "match_predictions",
]

DUPLICATE_CHECKS = [
    ("gameweeks", ["season", "gw"]),
    ("fixtures", ["season", "gw", "home_team_id", "away_team_id"]),
    ("player_gw_stats", ["season", "player_id", "gw"]),
    ("predictions", ["season", "player_id", "target_gw", "model_name"]),
    ("match_predictions", ["season", "fixture_id", "model_name"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a season-level data summary for imported historical seasons."
    )
    parser.add_argument("--season", required=True, help="Season key, for example 2024_25.")
    parser.add_argument("--out-json", default=None, help="Optional output JSON path.")
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


def get_table_columns(table_name: str) -> List[str]:
    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = :table_name
                ORDER BY ordinal_position
                """
            ),
            {"table_name": table_name},
        ).mappings().all()
        return [str(row["column_name"]) for row in rows]
    finally:
        db.close()


def count_rows(table_name: str, season: str) -> Optional[int]:
    if not table_exists(table_name):
        return None

    columns = get_table_columns(table_name)
    if "season" not in columns:
        return None

    db = SessionLocal()
    try:
        value = db.execute(
            text("SELECT COUNT(*) FROM %s WHERE season = :season" % table_name),
            {"season": season},
        ).scalar()
        return int(value or 0)
    finally:
        db.close()


def min_max_gw(table_name: str, season: str) -> Optional[Dict[str, Any]]:
    if not table_exists(table_name):
        return None

    columns = get_table_columns(table_name)
    if "season" not in columns or "gw" not in columns:
        return None

    db = SessionLocal()
    try:
        row = db.execute(
            text(
                """
                SELECT
                    MIN(gw) AS min_gw,
                    MAX(gw) AS max_gw,
                    COUNT(DISTINCT gw) AS distinct_gw_count
                FROM %s
                WHERE season = :season
                """ % table_name
            ),
            {"season": season},
        ).mappings().first()
        if row is None:
            return None
        return {
            "min_gw": row["min_gw"],
            "max_gw": row["max_gw"],
            "distinct_gw_count": int(row["distinct_gw_count"] or 0),
        }
    finally:
        db.close()


def duplicate_group_count(table_name: str, season: str, key_columns: Sequence[str]) -> Optional[int]:
    if not table_exists(table_name):
        return None

    columns = get_table_columns(table_name)
    for col in key_columns:
        if col not in columns:
            return None

    key_sql = ", ".join(key_columns)
    db = SessionLocal()
    try:
        value = db.execute(
            text(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT %s, COUNT(*) AS n
                    FROM %s
                    WHERE season = :season
                    GROUP BY %s
                    HAVING COUNT(*) > 1
                ) dup
                """ % (key_sql, table_name, key_sql)
            ),
            {"season": season},
        ).scalar()
        return int(value or 0)
    finally:
        db.close()


def player_stats_summary(season: str) -> Dict[str, Any]:
    if not table_exists("player_gw_stats"):
        return {"available": False}

    columns = get_table_columns("player_gw_stats")
    required = {"season", "player_id", "gw"}
    if not required.issubset(set(columns)):
        return {"available": False, "missing_columns": sorted(required - set(columns))}

    select_parts = [
        "COUNT(*) AS row_count",
        "COUNT(DISTINCT player_id) AS distinct_player_count",
        "COUNT(DISTINCT gw) AS distinct_gw_count",
        "MIN(gw) AS min_gw",
        "MAX(gw) AS max_gw",
    ]
    if "minutes" in columns:
        select_parts.append("SUM(COALESCE(minutes, 0)) AS total_minutes")
    if "total_points" in columns:
        select_parts.append("SUM(COALESCE(total_points, 0)) AS total_points")

    db = SessionLocal()
    try:
        row = db.execute(
            text(
                """
                SELECT %s
                FROM player_gw_stats
                WHERE season = :season
                """ % ", ".join(select_parts)
            ),
            {"season": season},
        ).mappings().first()
        if row is None:
            return {"available": True, "row_count": 0}
        return dict(row)
    finally:
        db.close()


def fixture_summary(season: str) -> Dict[str, Any]:
    if not table_exists("fixtures"):
        return {"available": False}

    columns = get_table_columns("fixtures")
    required = {"season", "gw"}
    if not required.issubset(set(columns)):
        return {"available": False, "missing_columns": sorted(required - set(columns))}

    select_parts = [
        "COUNT(*) AS row_count",
        "COUNT(DISTINCT gw) AS distinct_gw_count",
        "MIN(gw) AS min_gw",
        "MAX(gw) AS max_gw",
    ]
    if "finished" in columns:
        select_parts.append("SUM(CASE WHEN finished THEN 1 ELSE 0 END) AS finished_count")
    if "home_team_id" in columns and "away_team_id" in columns:
        select_parts.append(
            "COUNT(DISTINCT home_team_id) + COUNT(DISTINCT away_team_id) AS team_presence_count_proxy"
        )

    db = SessionLocal()
    try:
        row = db.execute(
            text(
                """
                SELECT %s
                FROM fixtures
                WHERE season = :season
                """ % ", ".join(select_parts)
            ),
            {"season": season},
        ).mappings().first()
        if row is None:
            return {"available": True, "row_count": 0}
        return dict(row)
    finally:
        db.close()


def build_report(season: str) -> Dict[str, Any]:
    row_counts = {
        table_name: count_rows(table_name, season)
        for table_name in TABLES_WITH_SEASON
    }

    gw_coverage = {
        table_name: min_max_gw(table_name, season)
        for table_name in ["gameweeks", "fixtures", "player_gw_stats"]
    }

    duplicate_results = []
    for table_name, key_columns in DUPLICATE_CHECKS:
        duplicate_results.append(
            {
                "table": table_name,
                "key_columns": list(key_columns),
                "duplicate_groups": duplicate_group_count(table_name, season, key_columns),
            }
        )

    report = {
        "created_at": utc_now(),
        "season": season,
        "row_counts": row_counts,
        "gw_coverage": gw_coverage,
        "player_gw_stats_summary": player_stats_summary(season),
        "fixture_summary": fixture_summary(season),
        "duplicate_results": duplicate_results,
        "ready_for_prior_builder": bool((row_counts.get("player_gw_stats") or 0) > 0),
        "ready_for_match_features": bool((row_counts.get("fixtures") or 0) > 0),
        "notes": [
            "This summary is read-only.",
            "ready_for_prior_builder requires player_gw_stats rows for the season.",
            "ready_for_match_features requires fixtures rows for the season.",
        ],
    }
    return report


def print_summary(report: Dict[str, Any]) -> None:
    print("=== Season Data Summary ===")
    print("season:", report["season"])
    print("ready_for_prior_builder:", report["ready_for_prior_builder"])
    print("ready_for_match_features:", report["ready_for_match_features"])
    print()

    print("Row counts:")
    for table_name, count in sorted(report["row_counts"].items()):
        print("- %s: %s" % (table_name, count))
    print()

    print("GW coverage:")
    for table_name, coverage in sorted(report["gw_coverage"].items()):
        print("- %s: %s" % (table_name, coverage))
    print()

    print("Duplicate checks:")
    for item in report["duplicate_results"]:
        print("- %s %s: %s" % (item["table"], item["key_columns"], item["duplicate_groups"]))
    print()


def save_report(report: Dict[str, Any], out_json: Optional[str]) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("saved_report:", path)


def main() -> None:
    args = parse_args()
    report = build_report(args.season)
    print_summary(report)
    save_report(report, args.out_json)


if __name__ == "__main__":
    main()
