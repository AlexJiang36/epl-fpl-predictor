from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text

from app.core.db import SessionLocal


STAGING_TABLES = [
    "historical_teams",
    "historical_players",
    "historical_fixtures",
    "historical_player_gw_stats",
]

DUPLICATE_CHECKS = [
    ("historical_teams", ["season", "raw_team_id"]),
    ("historical_players", ["season", "raw_player_id"]),
    ("historical_fixtures", ["season", "gw", "raw_home_team_id", "raw_away_team_id"]),
    ("historical_player_gw_stats", ["season", "raw_player_id", "gw"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a read-only summary of historical staging data for one season."
    )
    parser.add_argument("--season", required=True, help="Historical season key, for example 2024_25.")
    parser.add_argument("--out-json", default=None)
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


def count_rows(table_name: str, season: str) -> Optional[int]:
    if not table_exists(table_name):
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


def duplicate_group_count(table_name: str, season: str, key_columns: Sequence[str]) -> Optional[int]:
    if not table_exists(table_name):
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


def gw_coverage(table_name: str, season: str) -> Optional[Dict[str, Any]]:
    if not table_exists(table_name):
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


def mapping_counts(table_name: str, canonical_column: str, season: str) -> Dict[str, int]:
    if not table_exists(table_name):
        return {"mapped": 0, "unmapped": 0}

    db = SessionLocal()
    try:
        row = db.execute(
            text(
                """
                SELECT
                    SUM(CASE WHEN %s IS NULL THEN 0 ELSE 1 END) AS mapped,
                    SUM(CASE WHEN %s IS NULL THEN 1 ELSE 0 END) AS unmapped
                FROM %s
                WHERE season = :season
                """ % (canonical_column, canonical_column, table_name)
            ),
            {"season": season},
        ).mappings().first()
        return {
            "mapped": int((row or {}).get("mapped") or 0),
            "unmapped": int((row or {}).get("unmapped") or 0),
        }
    finally:
        db.close()


def aggregate_player_stats(season: str) -> Dict[str, Any]:
    if not table_exists("historical_player_gw_stats"):
        return {}

    db = SessionLocal()
    try:
        row = db.execute(
            text(
                """
                SELECT
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT raw_player_id) AS distinct_player_count,
                    COUNT(DISTINCT gw) AS distinct_gw_count,
                    SUM(minutes) AS total_minutes,
                    SUM(total_points) AS total_points
                FROM historical_player_gw_stats
                WHERE season = :season
                """
            ),
            {"season": season},
        ).mappings().first()
        return dict(row or {})
    finally:
        db.close()


def build_report(season: str) -> Dict[str, Any]:
    row_counts = {
        table_name: count_rows(table_name, season)
        for table_name in STAGING_TABLES
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
        "gw_coverage": {
            "historical_fixtures": gw_coverage("historical_fixtures", season),
            "historical_player_gw_stats": gw_coverage("historical_player_gw_stats", season),
        },
        "mapping_counts": {
            "teams": mapping_counts("historical_teams", "canonical_team_id", season),
            "players": mapping_counts("historical_players", "canonical_player_id", season),
        },
        "player_stats_summary": aggregate_player_stats(season),
        "duplicate_results": duplicate_results,
        "ready_for_staging_priors": bool((row_counts.get("historical_player_gw_stats") or 0) > 0),
        "ready_for_canonical_import": False,
        "notes": [
            "This summary is read-only.",
            "ready_for_staging_priors means historical raw player-GW rows exist.",
            "ready_for_canonical_import remains false until identity mapping is resolved.",
        ],
    }
    return report


def print_summary(report: Dict[str, Any]) -> None:
    print("=== Historical Staging Summary ===")
    print("season:", report["season"])
    print("ready_for_staging_priors:", report["ready_for_staging_priors"])
    print("ready_for_canonical_import:", report["ready_for_canonical_import"])
    print()

    print("Row counts:")
    for table_name, count in sorted(report["row_counts"].items()):
        print("- %s: %s" % (table_name, count))
    print()

    print("GW coverage:")
    for table_name, coverage in sorted(report["gw_coverage"].items()):
        print("- %s: %s" % (table_name, coverage))
    print()

    print("Mapping counts:")
    for label, counts in sorted(report["mapping_counts"].items()):
        print("- %s: %s" % (label, counts))
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
