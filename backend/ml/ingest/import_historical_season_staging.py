from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal


STAGING_TABLES = [
    "historical_player_gw_stats",
    "historical_fixtures",
    "historical_players",
    "historical_teams",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import prepared historical FPL CSVs into staging tables. "
            "Defaults to dry-run; use --write to write to the database."
        )
    )
    parser.add_argument("--season", required=True, help="Historical season key, for example 2024_25.")
    parser.add_argument("--player-gw-csv", required=True, help="Prepared player_gw_stats.csv.")
    parser.add_argument("--fixtures-csv", required=True, help="Prepared fixtures.csv.")
    parser.add_argument("--team-mapping-csv", required=True, help="team_mapping_template.csv or audited mapping CSV.")
    parser.add_argument("--player-mapping-csv", required=True, help="player_mapping_template.csv or audited mapping CSV.")
    parser.add_argument("--write", action="store_true", help="Actually write staging rows.")
    parser.add_argument(
        "--replace-existing-staging",
        action="store_true",
        help="Delete existing staging rows for this season before writing.",
    )
    parser.add_argument("--out-json", default=None, help="Optional report JSON path.")
    parser.add_argument("--sample-rows", type=int, default=3)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_raw_id(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    if text_value.endswith(".0"):
        text_value = text_value[:-2]
    return text_value


def nullable_str(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    return text_value


def nullable_int(value: Any, field_name: str) -> Optional[int]:
    if pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        return int(float(text_value))
    except Exception as exc:
        raise ValueError("Cannot parse %s as int: %r" % (field_name, value)) from exc


def required_int(value: Any, field_name: str) -> int:
    parsed = nullable_int(value, field_name)
    if parsed is None:
        raise ValueError("Required integer field %s is blank." % field_name)
    return parsed


def nullable_float(value: Any, field_name: str) -> Optional[float]:
    if pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        return float(text_value)
    except Exception as exc:
        raise ValueError("Cannot parse %s as float: %r" % (field_name, value)) from exc


def nullable_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text_value = str(value).strip().lower()
    if not text_value:
        return None
    if text_value in {"1", "true", "t", "yes", "y"}:
        return True
    if text_value in {"0", "false", "f", "no", "n"}:
        return False
    return None


def nullable_datetime(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    return text_value


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


def count_staging_rows(table_name: str, season: str) -> int:
    if not table_exists(table_name):
        return 0

    columns = get_table_columns(table_name)
    if "season" not in columns:
        return 0

    db = SessionLocal()
    try:
        value = db.execute(
            text("SELECT COUNT(*) FROM %s WHERE season = :season" % table_name),
            {"season": season},
        ).scalar()
        return int(value or 0)
    finally:
        db.close()


def require_csv(path_str: str, label: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError("%s not found: %s" % (label, path_str))
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> List[str]:
    missing = [col for col in required if col not in df.columns]
    if missing:
        return ["%s missing columns %s. Available columns: %s" % (label, missing, list(df.columns))]
    return []


def validate_season_column(df: pd.DataFrame, season: str, label: str) -> List[str]:
    if "season" not in df.columns:
        return ["%s is missing season column." % label]
    mismatch_count = int((df["season"].astype(str) != season).sum())
    if mismatch_count:
        return ["%s has %s rows where season != %s." % (label, mismatch_count, season)]
    return []


def build_historical_team_rows(df: pd.DataFrame, season: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_team_id = normalize_raw_id(row.get("raw_team_id"))
        if raw_team_id is None:
            continue

        canonical_team_id = nullable_int(row.get("canonical_team_id"), "canonical_team_id")
        mapping_status = nullable_str(row.get("mapping_status"))
        if mapping_status is None:
            mapping_status = "mapped" if canonical_team_id is not None else "unmapped"

        rows.append(
            {
                "season": season,
                "raw_team_id": raw_team_id,
                "raw_team_name": nullable_str(row.get("raw_team_name")),
                "raw_team_short_name": nullable_str(row.get("raw_team_short_name")),
                "canonical_team_id": canonical_team_id,
                "canonical_team_name": nullable_str(row.get("canonical_team_name")),
                "mapping_status": mapping_status,
                "mapping_confidence": nullable_str(row.get("mapping_confidence")),
                "notes": nullable_str(row.get("notes")),
            }
        )
    return rows


def build_historical_player_rows(df: pd.DataFrame, season: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_player_id = normalize_raw_id(row.get("raw_player_id"))
        if raw_player_id is None:
            continue

        canonical_player_id = nullable_int(row.get("canonical_player_id"), "canonical_player_id")
        mapping_status = nullable_str(row.get("mapping_status"))
        if mapping_status is None:
            mapping_status = "mapped" if canonical_player_id is not None else "unmapped"

        rows.append(
            {
                "season": season,
                "raw_player_id": raw_player_id,
                "raw_player_name": nullable_str(row.get("raw_player_name")),
                "raw_team_id": normalize_raw_id(row.get("raw_team_id")),
                "raw_position": nullable_str(row.get("raw_position")),
                "canonical_player_id": canonical_player_id,
                "canonical_player_name": nullable_str(row.get("canonical_player_name")),
                "mapping_status": mapping_status,
                "mapping_confidence": nullable_str(row.get("mapping_confidence")),
                "notes": nullable_str(row.get("notes")),
            }
        )
    return rows


def team_canonical_lookup(team_rows: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    return {row["raw_team_id"]: row.get("canonical_team_id") for row in team_rows}


def build_historical_fixture_rows(
    df: pd.DataFrame,
    season: str,
    team_lookup: Dict[str, Optional[int]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_home = normalize_raw_id(row.get("team_h"))
        raw_away = normalize_raw_id(row.get("team_a"))
        if raw_home is None or raw_away is None:
            continue

        rows.append(
            {
                "season": season,
                "raw_fixture_id": normalize_raw_id(row.get("fpl_fixture_id")),
                "gw": required_int(row.get("gw"), "gw"),
                "raw_home_team_id": raw_home,
                "raw_away_team_id": raw_away,
                "canonical_home_team_id": team_lookup.get(raw_home),
                "canonical_away_team_id": team_lookup.get(raw_away),
                "kickoff_time": nullable_datetime(row.get("kickoff_time")),
                "finished": nullable_bool(row.get("finished")),
                "home_score": nullable_int(row.get("team_h_score"), "team_h_score"),
                "away_score": nullable_int(row.get("team_a_score"), "team_a_score"),
            }
        )
    return rows


def build_historical_player_gw_rows(df: pd.DataFrame, season: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_player_id = normalize_raw_id(row.get("element"))
        if raw_player_id is None:
            continue

        rows.append(
            {
                "season": season,
                "raw_player_id": raw_player_id,
                "gw": required_int(row.get("gw"), "gw"),
                "minutes": required_int(row.get("minutes"), "minutes"),
                "goals_scored": nullable_int(row.get("goals_scored"), "goals_scored"),
                "assists": nullable_int(row.get("assists"), "assists"),
                "clean_sheets": nullable_int(row.get("clean_sheets"), "clean_sheets"),
                "total_points": required_int(row.get("total_points"), "total_points"),
                "bonus": nullable_int(row.get("bonus"), "bonus"),
                "value": nullable_float(row.get("value"), "value"),
                "was_home": nullable_bool(row.get("was_home")),
                "raw_opponent_team_id": normalize_raw_id(row.get("opponent_team")),
                "raw_fixture_ids": nullable_str(row.get("raw_fixture_id")),
            }
        )
    return rows


def duplicate_key_count(rows: List[Dict[str, Any]], keys: Sequence[str]) -> int:
    seen = set()
    duplicate_count = 0
    for row in rows:
        key = tuple(row.get(col) for col in keys)
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
    return duplicate_count


def insert_rows(table_name: str, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    table_columns = set(get_table_columns(table_name))
    filtered_rows = []
    for row in rows:
        filtered_rows.append({key: value for key, value in row.items() if key in table_columns})

    all_columns = sorted(set().union(*(row.keys() for row in filtered_rows)))
    column_sql = ", ".join(all_columns)
    value_sql = ", ".join(":%s" % col for col in all_columns)

    sql = text("INSERT INTO %s (%s) VALUES (%s)" % (table_name, column_sql, value_sql))
    normalized_rows = [{col: row.get(col) for col in all_columns} for row in filtered_rows]

    db = SessionLocal()
    try:
        db.execute(sql, normalized_rows)
        db.commit()
        return len(normalized_rows)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_existing_staging(season: str) -> None:
    db = SessionLocal()
    try:
        for table_name in STAGING_TABLES:
            if table_exists(table_name):
                db.execute(
                    text("DELETE FROM %s WHERE season = :season" % table_name),
                    {"season": season},
                )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def save_report(report: Dict[str, Any], out_json: Optional[str]) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("saved_report:", path)


def print_report(report: Dict[str, Any], write: bool) -> None:
    print("=== Historical Staging Import Plan ===")
    print("season:", report["season"])
    print("mode:", "WRITE" if write else "DRY_RUN")
    print("passed:", report["passed"])
    print()

    if report["errors"]:
        print("Errors:")
        for error in report["errors"]:
            print("-", error)
        print()

    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print("-", warning)
        print()

    print("Existing staging rows:")
    for table_name, count in sorted(report["existing_staging_rows"].items()):
        print("- %s: %s" % (table_name, count))
    print()

    print("Planned rows:")
    for table_name, count in sorted(report["planned_rows"].items()):
        print("- %s: %s" % (table_name, count))
    print()

    print("Mapping status:")
    for key, value in sorted(report["mapping_status"].items()):
        print("- %s: %s" % (key, value))
    print()


def main() -> None:
    args = parse_args()

    errors: List[str] = []
    warnings: List[str] = []

    required_tables = [
        "historical_teams",
        "historical_players",
        "historical_fixtures",
        "historical_player_gw_stats",
    ]
    missing_tables = [table_name for table_name in required_tables if not table_exists(table_name)]
    if missing_tables:
        errors.append(
            "Missing historical staging tables: %s. Run Alembic upgrade first."
            % missing_tables
        )

    team_df = require_csv(args.team_mapping_csv, "team_mapping_csv")
    player_df = require_csv(args.player_mapping_csv, "player_mapping_csv")
    fixtures_df = require_csv(args.fixtures_csv, "fixtures_csv")
    player_gw_df = require_csv(args.player_gw_csv, "player_gw_csv")

    errors.extend(require_columns(team_df, ["season", "raw_team_id"], "team_mapping_csv"))
    errors.extend(require_columns(player_df, ["season", "raw_player_id"], "player_mapping_csv"))
    errors.extend(require_columns(fixtures_df, ["season", "gw", "team_h", "team_a"], "fixtures_csv"))
    errors.extend(require_columns(player_gw_df, ["season", "element", "gw", "minutes", "total_points"], "player_gw_csv"))

    errors.extend(validate_season_column(team_df, args.season, "team_mapping_csv"))
    errors.extend(validate_season_column(player_df, args.season, "player_mapping_csv"))
    errors.extend(validate_season_column(fixtures_df, args.season, "fixtures_csv"))
    errors.extend(validate_season_column(player_gw_df, args.season, "player_gw_csv"))

    existing_rows = {
        table_name: count_staging_rows(table_name, args.season)
        for table_name in required_tables
    }

    historical_team_rows: List[Dict[str, Any]] = []
    historical_player_rows: List[Dict[str, Any]] = []
    historical_fixture_rows: List[Dict[str, Any]] = []
    historical_player_gw_rows: List[Dict[str, Any]] = []

    if not errors:
        historical_team_rows = build_historical_team_rows(team_df, args.season)
        historical_player_rows = build_historical_player_rows(player_df, args.season)
        team_lookup = team_canonical_lookup(historical_team_rows)
        historical_fixture_rows = build_historical_fixture_rows(fixtures_df, args.season, team_lookup)
        historical_player_gw_rows = build_historical_player_gw_rows(player_gw_df, args.season)

        duplicate_specs = [
            ("historical_teams", historical_team_rows, ["season", "raw_team_id"]),
            ("historical_players", historical_player_rows, ["season", "raw_player_id"]),
            ("historical_fixtures", historical_fixture_rows, ["season", "gw", "raw_home_team_id", "raw_away_team_id"]),
            ("historical_player_gw_stats", historical_player_gw_rows, ["season", "raw_player_id", "gw"]),
        ]
        for label, rows, keys in duplicate_specs:
            duplicate_count = duplicate_key_count(rows, keys)
            if duplicate_count > 0:
                errors.append("%s has %s duplicate rows for keys %s." % (label, duplicate_count, keys))

    unmapped_team_count = sum(1 for row in historical_team_rows if row.get("canonical_team_id") is None)
    mapped_team_count = len(historical_team_rows) - unmapped_team_count
    unmapped_player_count = sum(1 for row in historical_player_rows if row.get("canonical_player_id") is None)
    mapped_player_count = len(historical_player_rows) - unmapped_player_count

    if unmapped_team_count:
        warnings.append(
            "%s historical teams are unmapped to canonical teams. "
            "This is allowed in staging and expected for promoted/relegated clubs."
            % unmapped_team_count
        )
    if unmapped_player_count:
        warnings.append(
            "%s historical players are unmapped to canonical players. "
            "This is allowed in staging; do not use canonical player features until mapping is resolved."
            % unmapped_player_count
        )

    planned_rows = {
        "historical_teams": len(historical_team_rows),
        "historical_players": len(historical_player_rows),
        "historical_fixtures": len(historical_fixture_rows),
        "historical_player_gw_stats": len(historical_player_gw_rows),
    }

    report = {
        "created_at": utc_now(),
        "season": args.season,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "existing_staging_rows": existing_rows,
        "planned_rows": planned_rows,
        "mapping_status": {
            "mapped_team_count": mapped_team_count,
            "unmapped_team_count": unmapped_team_count,
            "mapped_player_count": mapped_player_count,
            "unmapped_player_count": unmapped_player_count,
        },
        "samples": {
            "historical_teams": historical_team_rows[: args.sample_rows],
            "historical_players": historical_player_rows[: args.sample_rows],
            "historical_fixtures": historical_fixture_rows[: args.sample_rows],
            "historical_player_gw_stats": historical_player_gw_rows[: args.sample_rows],
        },
        "notes": [
            "This imports into historical staging tables only, not canonical fixtures/player_gw_stats.",
            "Unmapped historical teams and players are allowed in staging.",
            "Use downstream mapping/audit steps before writing canonical current-season model tables.",
        ],
    }

    print_report(report, write=args.write)
    save_report(report, args.out_json)

    if errors:
        raise SystemExit(1)

    if args.write:
        if any(count > 0 for count in existing_rows.values()) and not args.replace_existing_staging:
            raise SystemExit(
                "Refusing to write because staging rows already exist for season=%s. "
                "Use --replace-existing-staging only if intentional." % args.season
            )

        if args.replace_existing_staging:
            delete_existing_staging(args.season)

        inserted = {
            "historical_teams": insert_rows("historical_teams", historical_team_rows),
            "historical_players": insert_rows("historical_players", historical_player_rows),
            "historical_fixtures": insert_rows("historical_fixtures", historical_fixture_rows),
            "historical_player_gw_stats": insert_rows("historical_player_gw_stats", historical_player_gw_rows),
        }

        print("Inserted staging rows:")
        for table_name, count in sorted(inserted.items()):
            print("- %s: %s" % (table_name, count))
    else:
        print("Dry-run only. No database writes were performed.")


if __name__ == "__main__":
    main()
