from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal


CORE_TABLE_REQUIREMENTS = {
    "gameweeks": ["season", "gw"],
    "fixtures": ["season", "gw", "home_team_id", "away_team_id"],
    "player_gw_stats": ["season", "player_id", "gw", "minutes", "total_points"],
    "players": ["id", "fpl_player_id", "web_name", "team_id", "position"],
    "teams": ["id"],
}

SEASON_COUNT_TABLES = [
    "gameweeks",
    "fixtures",
    "player_gw_stats",
    "predictions",
    "match_predictions",
]

PLAYER_GW_COLUMN_ALIASES = {
    "player_key": ["player_id", "fpl_player_id", "element"],
    "gw": ["gw", "round", "event"],
    "minutes": ["minutes"],
    "total_points": ["total_points", "points"],
}

PLAYER_GW_OPTIONAL_ALIASES = {
    "goals_scored": ["goals_scored", "goals"],
    "assists": ["assists"],
    "clean_sheets": ["clean_sheets"],
    "bonus": ["bonus"],
    "value": ["value", "now_cost"],
    "was_home": ["was_home", "is_home"],
    "opponent_team": ["opponent_team", "opponent_team_id"],
}

FIXTURE_COLUMN_ALIASES = {
    "gw": ["gw", "round", "event"],
    "home_team": ["home_team_id", "team_h", "home_team"],
    "away_team": ["away_team_id", "team_a", "away_team"],
}

FIXTURE_RECOMMENDED_ALIASES = {
    "kickoff_time": ["kickoff_time", "kickoff", "kickoff_date"],
    "home_score": ["home_score", "team_h_score", "home_goals"],
    "away_score": ["away_score", "team_a_score", "away_goals"],
    "finished": ["finished", "complete", "is_finished"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run historical season import readiness checks. "
            "This script does not write to the database."
        )
    )
    parser.add_argument(
        "--season",
        type=str,
        required=True,
        help="Historical season to validate, for example 2024_25.",
    )
    parser.add_argument(
        "--player-gw-csv",
        type=str,
        default=None,
        help="Optional player gameweek stats CSV to validate before import.",
    )
    parser.add_argument(
        "--fixtures-csv",
        type=str,
        default=None,
        help="Optional fixtures/results CSV to validate before import.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to save the readiness report as JSON.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Number of sample rows to include from each provided CSV.",
    )
    return parser.parse_args()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_table_columns(table_name: str) -> List[str]:
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


def _table_exists(table_name: str) -> bool:
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


def _count_rows(table_name: str, season: Optional[str] = None) -> int:
    db = SessionLocal()
    try:
        columns = _get_table_columns(table_name)
        if season is not None and "season" in columns:
            value = db.execute(
                text(f"SELECT COUNT(*) FROM {table_name} WHERE season = :season"),
                {"season": season},
            ).scalar()
        else:
            value = db.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        return int(value or 0)
    finally:
        db.close()


def _season_counts(table_name: str) -> List[Dict[str, Any]]:
    if not _table_exists(table_name):
        return []

    columns = _get_table_columns(table_name)
    if "season" not in columns:
        return []

    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                f"""
                SELECT season, COUNT(*) AS n
                FROM {table_name}
                GROUP BY season
                ORDER BY season
                """
            )
        ).mappings().all()
        return [{"season": str(row["season"]), "row_count": int(row["n"] or 0)} for row in rows]
    finally:
        db.close()


def _duplicate_group_count(table_name: str, season: str, key_columns: Sequence[str]) -> Optional[int]:
    if not _table_exists(table_name):
        return None

    columns = _get_table_columns(table_name)
    for col in key_columns:
        if col not in columns:
            return None

    key_sql = ", ".join(key_columns)
    db = SessionLocal()
    try:
        value = db.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT {key_sql}, COUNT(*) AS n
                    FROM {table_name}
                    WHERE season = :season
                    GROUP BY {key_sql}
                    HAVING COUNT(*) > 1
                ) dup
                """
            ),
            {"season": season},
        ).scalar()
        return int(value or 0)
    finally:
        db.close()


def _resolve_alias(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lower_to_original = {col.lower(): col for col in columns}
    for alias in aliases:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]
    return None


def _validate_csv(
    *,
    csv_path: str,
    season: str,
    required_aliases: Dict[str, List[str]],
    optional_aliases: Dict[str, List[str]],
    duplicate_key_names: Sequence[str],
    sample_rows: int,
) -> Dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        return {
            "path": csv_path,
            "exists": False,
            "passed": False,
            "errors": [f"CSV file not found: {csv_path}"],
            "warnings": [],
        }

    df = pd.read_csv(path)
    columns = list(df.columns)

    errors: List[str] = []
    warnings: List[str] = []
    resolved_required: Dict[str, Optional[str]] = {}
    resolved_optional: Dict[str, Optional[str]] = {}

    for logical_name, aliases in required_aliases.items():
        resolved = _resolve_alias(columns, aliases)
        resolved_required[logical_name] = resolved
        if resolved is None:
            errors.append(
                "Missing required logical column %s. Accepted aliases: %s"
                % (logical_name, aliases)
            )

    for logical_name, aliases in optional_aliases.items():
        resolved_optional[logical_name] = _resolve_alias(columns, aliases)

    if "season" in columns:
        mismatched = df[df["season"].astype(str) != season]
        if not mismatched.empty:
            errors.append(
                "CSV contains a season column, but %s rows do not match --season=%s."
                % (len(mismatched), season)
            )
    else:
        warnings.append(
            "CSV has no season column. This is okay for dry-run, but importer must attach --season=%s."
            % season
        )

    null_counts: Dict[str, int] = {}
    for logical_name, col in resolved_required.items():
        if col is None:
            continue
        null_counts[logical_name] = int(df[col].isna().sum())
        if null_counts[logical_name] > 0:
            errors.append(
                "Required column %s (%s) has %s null values."
                % (logical_name, col, null_counts[logical_name])
            )

    duplicate_groups: Optional[int] = None
    duplicate_columns = []
    for logical_name in duplicate_key_names:
        col = resolved_required.get(logical_name)
        if col:
            duplicate_columns.append(col)

    if len(duplicate_columns) == len(duplicate_key_names):
        duplicate_groups = int(df.duplicated(subset=duplicate_columns).sum())
        if duplicate_groups > 0:
            errors.append(
                "CSV has %s duplicate rows for key columns %s."
                % (duplicate_groups, duplicate_columns)
            )

    sample = []
    if sample_rows > 0 and not df.empty:
        sample = df.head(sample_rows).where(pd.notna(df.head(sample_rows)), None).to_dict(orient="records")

    return {
        "path": csv_path,
        "exists": True,
        "passed": len(errors) == 0,
        "row_count": int(len(df)),
        "columns": columns,
        "resolved_required_columns": resolved_required,
        "resolved_optional_columns": resolved_optional,
        "required_null_counts": null_counts,
        "duplicate_key_columns": duplicate_columns,
        "duplicate_key_row_count": duplicate_groups,
        "sample_rows": sample,
        "errors": errors,
        "warnings": warnings,
    }


def build_report(
    *,
    season: str,
    player_gw_csv: Optional[str],
    fixtures_csv: Optional[str],
    sample_rows: int,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for table_name, required_columns in CORE_TABLE_REQUIREMENTS.items():
        exists = _table_exists(table_name)
        columns = _get_table_columns(table_name) if exists else []
        missing = [col for col in required_columns if col not in columns]
        checks.append(
            {
                "name": "table_contract_%s" % table_name,
                "passed": exists and not missing,
                "details": {
                    "table": table_name,
                    "exists": exists,
                    "required_columns": required_columns,
                    "missing_columns": missing,
                    "available_columns": columns,
                },
            }
        )

    season_counts = {
        table_name: _season_counts(table_name)
        for table_name in SEASON_COUNT_TABLES
        if _table_exists(table_name)
    }

    target_season_existing_rows = {
        table_name: _count_rows(table_name, season)
        for table_name in SEASON_COUNT_TABLES
        if _table_exists(table_name) and "season" in _get_table_columns(table_name)
    }

    if any(count > 0 for count in target_season_existing_rows.values()):
        warnings.append(
            "Target season %s already has rows in one or more tables. "
            "Future importer should require explicit overwrite/upsert behavior." % season
        )

    duplicate_checks = [
        ("gameweeks", ["season", "gw"]),
        ("player_gw_stats", ["season", "player_id", "gw"]),
        ("predictions", ["season", "player_id", "target_gw", "model_name"]),
        ("match_predictions", ["season", "fixture_id", "model_name"]),
    ]

    duplicate_results = []
    for table_name, key_columns in duplicate_checks:
        duplicate_count = _duplicate_group_count(table_name, season, key_columns)
        if duplicate_count is None:
            duplicate_results.append(
                {
                    "table": table_name,
                    "key_columns": key_columns,
                    "checked": False,
                    "duplicate_groups": None,
                }
            )
            continue

        duplicate_results.append(
            {
                "table": table_name,
                "key_columns": key_columns,
                "checked": True,
                "duplicate_groups": duplicate_count,
            }
        )
        checks.append(
            {
                "name": "no_duplicate_%s_keys_for_target_season" % table_name,
                "passed": duplicate_count == 0,
                "details": {
                    "table": table_name,
                    "season": season,
                    "key_columns": key_columns,
                    "duplicate_groups": duplicate_count,
                },
            }
        )

    csv_reports: Dict[str, Optional[Dict[str, Any]]] = {
        "player_gw_csv": None,
        "fixtures_csv": None,
    }

    if player_gw_csv:
        player_report = _validate_csv(
            csv_path=player_gw_csv,
            season=season,
            required_aliases=PLAYER_GW_COLUMN_ALIASES,
            optional_aliases=PLAYER_GW_OPTIONAL_ALIASES,
            duplicate_key_names=["player_key", "gw"],
            sample_rows=sample_rows,
        )
        csv_reports["player_gw_csv"] = player_report
        checks.append(
            {
                "name": "player_gw_csv_contract",
                "passed": bool(player_report.get("passed")),
                "details": player_report,
            }
        )

    if fixtures_csv:
        fixture_report = _validate_csv(
            csv_path=fixtures_csv,
            season=season,
            required_aliases=FIXTURE_COLUMN_ALIASES,
            optional_aliases=FIXTURE_RECOMMENDED_ALIASES,
            duplicate_key_names=["gw", "home_team", "away_team"],
            sample_rows=sample_rows,
        )
        csv_reports["fixtures_csv"] = fixture_report
        checks.append(
            {
                "name": "fixtures_csv_contract",
                "passed": bool(fixture_report.get("passed")),
                "details": fixture_report,
            }
        )

        missing_recommended = []
        for logical_name, resolved in fixture_report.get("resolved_optional_columns", {}).items():
            if resolved is None:
                missing_recommended.append(logical_name)
        if missing_recommended:
            warnings.append(
                "Fixtures CSV is missing recommended columns for completed historical match features: %s"
                % missing_recommended
            )

    overall_ready = all(bool(check["passed"]) for check in checks)

    return {
        "created_at": _now_utc(),
        "season": season,
        "overall_ready": overall_ready,
        "warnings": warnings,
        "database": {
            "season_counts": season_counts,
            "target_season_existing_rows": target_season_existing_rows,
            "duplicate_results": duplicate_results,
        },
        "csv_reports": csv_reports,
        "checks": checks,
        "notes": [
            "This is a dry-run readiness report and does not write to the database.",
            "A future importer should support --season, --dry-run, and explicit upsert/overwrite behavior.",
            "Historical player identity mapping must be reviewed before writing player_gw_stats.",
        ],
    }


def print_summary(report: Dict[str, Any]) -> None:
    print("=== Historical Season Import Readiness ===")
    print("season:", report["season"])
    print("overall_ready:", report["overall_ready"])
    print()

    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print("-", warning)
        print()

    print("Target season existing rows:")
    for table_name, count in sorted(report["database"]["target_season_existing_rows"].items()):
        print("- %s: %s" % (table_name, count))
    print()

    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print("[%s] %s" % (status, check["name"]))
    print()


def maybe_save_report(report: Dict[str, Any], out_json: Optional[str]) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("saved_report:", path)


def main() -> None:
    args = parse_args()
    report = build_report(
        season=args.season,
        player_gw_csv=args.player_gw_csv,
        fixtures_csv=args.fixtures_csv,
        sample_rows=args.sample_rows,
    )
    print_summary(report)
    maybe_save_report(report, args.out_json)

    if not report["overall_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
