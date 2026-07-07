from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal


PLAYER_GW_REQUIRED_ALIASES = {
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

FIXTURE_REQUIRED_ALIASES = {
    "gw": ["gw", "round", "event"],
    "home_team": ["home_team_id", "team_h", "home_team"],
    "away_team": ["away_team_id", "team_a", "away_team"],
}

FIXTURE_OPTIONAL_ALIASES = {
    "fpl_fixture_id": ["fpl_fixture_id", "fixture_id", "id"],
    "kickoff_time": ["kickoff_time", "kickoff", "kickoff_date"],
    "home_score": ["home_score", "team_h_score", "home_goals"],
    "away_score": ["away_score", "team_a_score", "away_goals"],
    "finished": ["finished", "complete", "is_finished"],
}

PLAYER_MAPPING_ALIASES = {
    "raw_player_id": ["raw_player_id", "source_player_id", "player_id", "element", "fpl_player_id"],
    "canonical_player_id": ["canonical_player_id", "player_id_canonical", "db_player_id"],
}

TEAM_MAPPING_ALIASES = {
    "raw_team_id": ["raw_team_id", "source_team_id", "team_id", "team_h", "team_a"],
    "canonical_team_id": ["canonical_team_id", "team_id_canonical", "db_team_id"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import historical FPL season CSVs safely. Defaults to dry-run and writes "
            "only when --write is supplied."
        )
    )
    parser.add_argument("--season", required=True, help="Historical season key, for example 2024_25.")
    parser.add_argument("--player-gw-csv", default=None, help="Historical player GW stats CSV.")
    parser.add_argument("--fixtures-csv", default=None, help="Historical fixtures/results CSV.")
    parser.add_argument("--player-mapping-csv", default=None, help="Optional raw player -> canonical player mapping CSV.")
    parser.add_argument("--team-mapping-csv", default=None, help="Optional raw team -> canonical team mapping CSV.")
    parser.add_argument(
        "--allow-direct-player-id-mapping",
        action="store_true",
        help=(
            "Allow direct mapping from the player key column to current database player IDs or "
            "fpl_player_id. Use only when you have confirmed the historical IDs are safe."
        ),
    )
    parser.add_argument(
        "--allow-direct-team-id-mapping",
        action="store_true",
        help=(
            "Allow direct mapping from fixture team IDs to current database team IDs. "
            "Use only when you have confirmed the historical IDs are safe."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write rows to the database. Without this flag, the command is dry-run only.",
    )
    parser.add_argument(
        "--allow-existing-target-season",
        action="store_true",
        help="Allow writing even if target season already has rows. Default is to refuse.",
    )
    parser.add_argument("--out-json", default=None, help="Optional import plan/report JSON path.")
    parser.add_argument("--sample-rows", type=int, default=3, help="Sample rows in report.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    text_value = str(value).strip()
    if text_value.endswith(".0"):
        text_value = text_value[:-2]
    return text_value


def to_int(value: Any, field_name: str) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError("Cannot parse %s as int: %r" % (field_name, value)) from exc


def to_float(value: Any, field_name: str) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception as exc:
        raise ValueError("Cannot parse %s as float: %r" % (field_name, value)) from exc


def to_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text_value = str(value).strip().lower()
    if text_value in {"1", "true", "t", "yes", "y"}:
        return True
    if text_value in {"0", "false", "f", "no", "n"}:
        return False
    return None


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


def count_rows(table_name: str, season: Optional[str] = None) -> int:
    if not table_exists(table_name):
        return 0

    columns = get_table_columns(table_name)
    db = SessionLocal()
    try:
        if season is not None and "season" in columns:
            value = db.execute(
                text("SELECT COUNT(*) FROM %s WHERE season = :season" % table_name),
                {"season": season},
            ).scalar()
        else:
            value = db.execute(text("SELECT COUNT(*) FROM %s" % table_name)).scalar()
        return int(value or 0)
    finally:
        db.close()


def resolve_alias(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lower_to_original = {col.lower(): col for col in columns}
    for alias in aliases:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]
    return None


def resolve_aliases(
    columns: Sequence[str],
    required_aliases: Dict[str, List[str]],
    optional_aliases: Dict[str, List[str]],
) -> Tuple[Dict[str, str], Dict[str, Optional[str]], List[str]]:
    errors: List[str] = []
    required: Dict[str, str] = {}
    optional: Dict[str, Optional[str]] = {}

    for logical_name, aliases in required_aliases.items():
        resolved = resolve_alias(columns, aliases)
        if resolved is None:
            errors.append("Missing required logical column %s. Accepted aliases: %s" % (logical_name, aliases))
        else:
            required[logical_name] = resolved

    for logical_name, aliases in optional_aliases.items():
        optional[logical_name] = resolve_alias(columns, aliases)

    return required, optional, errors


def read_csv(path_str: Optional[str], label: str) -> Optional[pd.DataFrame]:
    if not path_str:
        return None

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError("%s not found: %s" % (label, path_str))

    return pd.read_csv(path)


def validate_player_gw_csv(df: pd.DataFrame, season: str) -> Tuple[Dict[str, str], Dict[str, Optional[str]], List[str], List[str]]:
    required, optional, errors = resolve_aliases(
        list(df.columns),
        PLAYER_GW_REQUIRED_ALIASES,
        PLAYER_GW_OPTIONAL_ALIASES,
    )
    warnings: List[str] = []

    if "season" in df.columns:
        mismatched = df[df["season"].astype(str) != season]
        if not mismatched.empty:
            errors.append("player_gw_csv has %s rows where season != %s." % (len(mismatched), season))
    else:
        warnings.append("player_gw_csv has no season column; importer will attach season=%s." % season)

    for logical_name, col in required.items():
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            errors.append("player_gw_csv required column %s (%s) has %s null values." % (logical_name, col, null_count))

    if "player_key" in required and "gw" in required:
        duplicate_count = int(df.duplicated(subset=[required["player_key"], required["gw"]]).sum())
        if duplicate_count > 0:
            errors.append("player_gw_csv has %s duplicate player_key + gw rows." % duplicate_count)

    if "gw" in required:
        gw_values = pd.to_numeric(df[required["gw"]], errors="coerce")
        if gw_values.isna().any():
            errors.append("player_gw_csv has non-numeric gw values.")
        elif (gw_values <= 0).any():
            errors.append("player_gw_csv has gw values <= 0.")

    if "minutes" in required:
        minutes = pd.to_numeric(df[required["minutes"]], errors="coerce")
        if minutes.isna().any():
            errors.append("player_gw_csv has non-numeric minutes values.")
        elif (minutes < 0).any():
            errors.append("player_gw_csv has negative minutes values.")

    return required, optional, errors, warnings


def validate_fixtures_csv(df: pd.DataFrame, season: str) -> Tuple[Dict[str, str], Dict[str, Optional[str]], List[str], List[str]]:
    required, optional, errors = resolve_aliases(
        list(df.columns),
        FIXTURE_REQUIRED_ALIASES,
        FIXTURE_OPTIONAL_ALIASES,
    )
    warnings: List[str] = []

    if "season" in df.columns:
        mismatched = df[df["season"].astype(str) != season]
        if not mismatched.empty:
            errors.append("fixtures_csv has %s rows where season != %s." % (len(mismatched), season))
    else:
        warnings.append("fixtures_csv has no season column; importer will attach season=%s." % season)

    for logical_name, col in required.items():
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            errors.append("fixtures_csv required column %s (%s) has %s null values." % (logical_name, col, null_count))

    if {"gw", "home_team", "away_team"}.issubset(set(required.keys())):
        duplicate_count = int(df.duplicated(subset=[required["gw"], required["home_team"], required["away_team"]]).sum())
        if duplicate_count > 0:
            errors.append("fixtures_csv has %s duplicate gw + home_team + away_team rows." % duplicate_count)

        same_team = df[required["home_team"]].astype(str) == df[required["away_team"]].astype(str)
        if bool(same_team.any()):
            errors.append("fixtures_csv has %s rows where home_team == away_team." % int(same_team.sum()))

    if "gw" in required:
        gw_values = pd.to_numeric(df[required["gw"]], errors="coerce")
        if gw_values.isna().any():
            errors.append("fixtures_csv has non-numeric gw values.")
        elif (gw_values <= 0).any():
            errors.append("fixtures_csv has gw values <= 0.")

    for logical_name in ["kickoff_time", "home_score", "away_score", "finished"]:
        if optional.get(logical_name) is None:
            warnings.append("fixtures_csv missing recommended column: %s." % logical_name)

    return required, optional, errors, warnings


def load_mapping_csv(
    path_str: Optional[str],
    aliases: Dict[str, List[str]],
    raw_key_name: str,
    canonical_key_name: str,
    label: str,
) -> Dict[str, int]:
    if not path_str:
        return {}

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError("%s mapping CSV not found: %s" % (label, path_str))

    df = pd.read_csv(path)
    columns = list(df.columns)

    raw_col = resolve_alias(columns, aliases[raw_key_name])
    canonical_col = resolve_alias(columns, aliases[canonical_key_name])
    if raw_col is None or canonical_col is None:
        raise RuntimeError(
            "%s mapping CSV must include %s and %s. Columns: %s"
            % (label, raw_key_name, canonical_key_name, columns)
        )

    mapping: Dict[str, int] = {}
    duplicate_raw: List[str] = []
    for _, row in df.iterrows():
        raw = normalize_key(row[raw_col])
        canonical = to_int(row[canonical_col], "%s canonical id" % label)
        if not raw or canonical is None:
            continue
        if raw in mapping and mapping[raw] != canonical:
            duplicate_raw.append(raw)
        mapping[raw] = canonical

    if duplicate_raw:
        raise RuntimeError("%s mapping CSV has conflicting raw IDs: %s" % (label, sorted(set(duplicate_raw))[:10]))

    return mapping


def get_existing_ids(table_name: str, id_column: str) -> set:
    if not table_exists(table_name):
        return set()

    columns = get_table_columns(table_name)
    if id_column not in columns:
        return set()

    db = SessionLocal()
    try:
        rows = db.execute(text("SELECT %s FROM %s" % (id_column, table_name))).fetchall()
        return {normalize_key(row[0]) for row in rows}
    finally:
        db.close()


def get_player_fpl_id_to_id() -> Dict[str, int]:
    if not table_exists("players"):
        return {}

    columns = get_table_columns("players")
    if "id" not in columns or "fpl_player_id" not in columns:
        return {}

    db = SessionLocal()
    try:
        rows = db.execute(text("SELECT id, fpl_player_id FROM players WHERE fpl_player_id IS NOT NULL")).fetchall()
        return {normalize_key(row[1]): int(row[0]) for row in rows}
    finally:
        db.close()


def build_team_mapping(
    *,
    fixture_df: Optional[pd.DataFrame],
    fixture_required: Optional[Dict[str, str]],
    mapping_csv: Optional[str],
    allow_direct: bool,
) -> Tuple[Dict[str, int], List[str]]:
    warnings: List[str] = []
    explicit_mapping = load_mapping_csv(
        mapping_csv,
        TEAM_MAPPING_ALIASES,
        "raw_team_id",
        "canonical_team_id",
        "team",
    )
    if explicit_mapping:
        return explicit_mapping, warnings

    if fixture_df is None or fixture_required is None:
        return {}, warnings

    raw_values = set()
    for logical_name in ["home_team", "away_team"]:
        col = fixture_required[logical_name]
        raw_values.update(normalize_key(value) for value in fixture_df[col].dropna().tolist())

    if not allow_direct:
        if raw_values:
            warnings.append(
                "No team_mapping_csv supplied. Direct team ID mapping is disabled. "
                "Use --allow-direct-team-id-mapping only if raw team IDs are canonical."
            )
        return {}, warnings

    existing_team_ids = get_existing_ids("teams", "id")
    missing = sorted(raw for raw in raw_values if raw not in existing_team_ids)
    if missing:
        raise RuntimeError(
            "Direct team ID mapping was requested, but %s raw team IDs are not in teams.id. Examples: %s"
            % (len(missing), missing[:10])
        )

    return {raw: int(raw) for raw in raw_values}, warnings


def build_player_mapping(
    *,
    player_df: Optional[pd.DataFrame],
    player_required: Optional[Dict[str, str]],
    mapping_csv: Optional[str],
    allow_direct: bool,
) -> Tuple[Dict[str, int], List[str]]:
    warnings: List[str] = []
    explicit_mapping = load_mapping_csv(
        mapping_csv,
        PLAYER_MAPPING_ALIASES,
        "raw_player_id",
        "canonical_player_id",
        "player",
    )
    if explicit_mapping:
        return explicit_mapping, warnings

    if player_df is None or player_required is None:
        return {}, warnings

    raw_col = player_required["player_key"]
    raw_values = set(normalize_key(value) for value in player_df[raw_col].dropna().tolist())

    if not allow_direct:
        if raw_values:
            warnings.append(
                "No player_mapping_csv supplied. Direct player ID mapping is disabled. "
                "Use --allow-direct-player-id-mapping only if raw player IDs are canonical."
            )
        return {}, warnings

    existing_player_ids = get_existing_ids("players", "id")
    if raw_col.lower() == "player_id":
        missing = sorted(raw for raw in raw_values if raw not in existing_player_ids)
        if missing:
            raise RuntimeError(
                "Direct player_id mapping was requested, but %s raw player IDs are not in players.id. Examples: %s"
                % (len(missing), missing[:10])
            )
        return {raw: int(raw) for raw in raw_values}, warnings

    fpl_to_id = get_player_fpl_id_to_id()
    missing_fpl = sorted(raw for raw in raw_values if raw not in fpl_to_id)
    if missing_fpl:
        raise RuntimeError(
            "Direct fpl_player_id/element mapping was requested, but %s raw IDs are not in players.fpl_player_id. Examples: %s"
            % (len(missing_fpl), missing_fpl[:10])
        )

    return {raw: fpl_to_id[raw] for raw in raw_values}, warnings


def require_complete_mapping(
    *,
    label: str,
    raw_values: Iterable[str],
    mapping: Dict[str, int],
) -> List[str]:
    missing = sorted(raw for raw in set(raw_values) if raw and raw not in mapping)
    if missing:
        return [
            "%s mapping is incomplete. Missing %s raw IDs. Examples: %s"
            % (label, len(missing), missing[:10])
        ]
    return []


def maybe_get(row: pd.Series, col: Optional[str]) -> Any:
    if col is None:
        return None
    return row[col]


def prepare_gameweek_rows(season: str, player_df: Optional[pd.DataFrame], player_required: Optional[Dict[str, str]], fixture_df: Optional[pd.DataFrame], fixture_required: Optional[Dict[str, str]]) -> List[Dict[str, Any]]:
    gw_values = set()

    if player_df is not None and player_required is not None:
        gw_col = player_required["gw"]
        for value in player_df[gw_col].dropna().tolist():
            gw = to_int(value, "player gw")
            if gw is not None:
                gw_values.add(gw)

    if fixture_df is not None and fixture_required is not None:
        gw_col = fixture_required["gw"]
        for value in fixture_df[gw_col].dropna().tolist():
            gw = to_int(value, "fixture gw")
            if gw is not None:
                gw_values.add(gw)

    rows = []
    for gw in sorted(gw_values):
        rows.append(
            {
                "season": season,
                "gw": gw,
                "name": "GW %s" % gw,
                "is_finished": True,
            }
        )
    return rows


def prepare_fixture_rows(
    *,
    season: str,
    fixture_df: pd.DataFrame,
    required: Dict[str, str],
    optional: Dict[str, Optional[str]],
    team_mapping: Dict[str, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in fixture_df.iterrows():
        home_raw = normalize_key(row[required["home_team"]])
        away_raw = normalize_key(row[required["away_team"]])

        rows.append(
            {
                "season": season,
                "gw": to_int(row[required["gw"]], "fixture gw"),
                "fpl_fixture_id": to_int(maybe_get(row, optional.get("fpl_fixture_id")), "fpl_fixture_id"),
                "home_team_id": team_mapping[home_raw],
                "away_team_id": team_mapping[away_raw],
                "kickoff_time": maybe_get(row, optional.get("kickoff_time")),
                "finished": to_bool(maybe_get(row, optional.get("finished"))),
                "home_score": to_int(maybe_get(row, optional.get("home_score")), "home_score"),
                "away_score": to_int(maybe_get(row, optional.get("away_score")), "away_score"),
            }
        )
    return rows


def prepare_player_gw_rows(
    *,
    season: str,
    player_df: pd.DataFrame,
    required: Dict[str, str],
    optional: Dict[str, Optional[str]],
    player_mapping: Dict[str, int],
    team_mapping: Dict[str, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in player_df.iterrows():
        player_raw = normalize_key(row[required["player_key"]])
        opponent_raw = normalize_key(maybe_get(row, optional.get("opponent_team")))

        opponent_team_value = None
        if opponent_raw:
            opponent_team_value = team_mapping.get(opponent_raw)
            if opponent_team_value is None:
                opponent_team_value = to_int(opponent_raw, "opponent_team")

        rows.append(
            {
                "season": season,
                "player_id": player_mapping[player_raw],
                "gw": to_int(row[required["gw"]], "player gw"),
                "minutes": to_int(row[required["minutes"]], "minutes"),
                "total_points": to_float(row[required["total_points"]], "total_points"),
                "goals_scored": to_int(maybe_get(row, optional.get("goals_scored")), "goals_scored"),
                "assists": to_int(maybe_get(row, optional.get("assists")), "assists"),
                "clean_sheets": to_int(maybe_get(row, optional.get("clean_sheets")), "clean_sheets"),
                "bonus": to_int(maybe_get(row, optional.get("bonus")), "bonus"),
                "value": to_float(maybe_get(row, optional.get("value")), "value"),
                "was_home": to_bool(maybe_get(row, optional.get("was_home"))),
                "opponent_team": opponent_team_value,
            }
        )
    return rows


def insert_rows(table_name: str, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    table_columns = set(get_table_columns(table_name))
    insert_rows_filtered = []
    for row in rows:
        filtered = {key: value for key, value in row.items() if key in table_columns and value is not None}
        insert_rows_filtered.append(filtered)

    all_columns = sorted(set().union(*(row.keys() for row in insert_rows_filtered)))
    if not all_columns:
        return 0

    column_sql = ", ".join(all_columns)
    value_sql = ", ".join(":%s" % col for col in all_columns)
    sql = text("INSERT INTO %s (%s) VALUES (%s)" % (table_name, column_sql, value_sql))

    normalized_rows = []
    for row in insert_rows_filtered:
        normalized_rows.append({col: row.get(col) for col in all_columns})

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


def existing_target_rows(season: str) -> Dict[str, int]:
    return {
        "gameweeks": count_rows("gameweeks", season),
        "fixtures": count_rows("fixtures", season),
        "player_gw_stats": count_rows("player_gw_stats", season),
    }


def build_report(
    *,
    season: str,
    player_gw_csv: Optional[str],
    fixtures_csv: Optional[str],
    player_mapping_csv: Optional[str],
    team_mapping_csv: Optional[str],
    allow_direct_player_id_mapping: bool,
    allow_direct_team_id_mapping: bool,
    sample_rows: int,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not player_gw_csv and not fixtures_csv:
        errors.append("At least one of --player-gw-csv or --fixtures-csv is required.")

    existing_rows = existing_target_rows(season)

    player_df = read_csv(player_gw_csv, "player_gw_csv") if player_gw_csv else None
    fixture_df = read_csv(fixtures_csv, "fixtures_csv") if fixtures_csv else None

    player_required: Optional[Dict[str, str]] = None
    player_optional: Optional[Dict[str, Optional[str]]] = None
    fixture_required: Optional[Dict[str, str]] = None
    fixture_optional: Optional[Dict[str, Optional[str]]] = None

    if player_df is not None:
        player_required, player_optional, player_errors, player_warnings = validate_player_gw_csv(player_df, season)
        errors.extend(player_errors)
        warnings.extend(player_warnings)

    if fixture_df is not None:
        fixture_required, fixture_optional, fixture_errors, fixture_warnings = validate_fixtures_csv(fixture_df, season)
        errors.extend(fixture_errors)
        warnings.extend(fixture_warnings)

    team_mapping: Dict[str, int] = {}
    player_mapping: Dict[str, int] = {}

    if not errors:
        team_mapping, team_warnings = build_team_mapping(
            fixture_df=fixture_df,
            fixture_required=fixture_required,
            mapping_csv=team_mapping_csv,
            allow_direct=allow_direct_team_id_mapping,
        )
        warnings.extend(team_warnings)

        player_mapping, player_warnings = build_player_mapping(
            player_df=player_df,
            player_required=player_required,
            mapping_csv=player_mapping_csv,
            allow_direct=allow_direct_player_id_mapping,
        )
        warnings.extend(player_warnings)

        if fixture_df is not None and fixture_required is not None:
            team_raw_values = []
            for logical_name in ["home_team", "away_team"]:
                team_raw_values.extend(normalize_key(value) for value in fixture_df[fixture_required[logical_name]].dropna().tolist())
            errors.extend(require_complete_mapping(label="team", raw_values=team_raw_values, mapping=team_mapping))

        if player_df is not None and player_required is not None:
            player_raw_values = [normalize_key(value) for value in player_df[player_required["player_key"]].dropna().tolist()]
            errors.extend(require_complete_mapping(label="player", raw_values=player_raw_values, mapping=player_mapping))

    prepared: Dict[str, List[Dict[str, Any]]] = {
        "gameweeks": [],
        "fixtures": [],
        "player_gw_stats": [],
    }

    if not errors:
        prepared["gameweeks"] = prepare_gameweek_rows(
            season=season,
            player_df=player_df,
            player_required=player_required,
            fixture_df=fixture_df,
            fixture_required=fixture_required,
        )

        if fixture_df is not None and fixture_required is not None and fixture_optional is not None:
            prepared["fixtures"] = prepare_fixture_rows(
                season=season,
                fixture_df=fixture_df,
                required=fixture_required,
                optional=fixture_optional,
                team_mapping=team_mapping,
            )

        if player_df is not None and player_required is not None and player_optional is not None:
            prepared["player_gw_stats"] = prepare_player_gw_rows(
                season=season,
                player_df=player_df,
                required=player_required,
                optional=player_optional,
                player_mapping=player_mapping,
                team_mapping=team_mapping,
            )

    report = {
        "created_at": utc_now(),
        "season": season,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "existing_target_rows": existing_rows,
        "inputs": {
            "player_gw_csv": player_gw_csv,
            "fixtures_csv": fixtures_csv,
            "player_mapping_csv": player_mapping_csv,
            "team_mapping_csv": team_mapping_csv,
            "allow_direct_player_id_mapping": allow_direct_player_id_mapping,
            "allow_direct_team_id_mapping": allow_direct_team_id_mapping,
        },
        "resolved_columns": {
            "player_gw_required": player_required,
            "player_gw_optional": player_optional,
            "fixtures_required": fixture_required,
            "fixtures_optional": fixture_optional,
        },
        "mapping_summary": {
            "team_mapping_count": len(team_mapping),
            "player_mapping_count": len(player_mapping),
        },
        "planned_rows": {
            "gameweeks": len(prepared["gameweeks"]),
            "fixtures": len(prepared["fixtures"]),
            "player_gw_stats": len(prepared["player_gw_stats"]),
        },
        "samples": {
            "gameweeks": prepared["gameweeks"][:sample_rows],
            "fixtures": prepared["fixtures"][:sample_rows],
            "player_gw_stats": prepared["player_gw_stats"][:sample_rows],
        },
        "notes": [
            "Default mode is dry-run. Use --write to write rows.",
            "Direct player/team ID mapping is disabled unless explicitly allowed.",
            "Existing target season rows are refused at write time unless --allow-existing-target-season is supplied.",
        ],
    }

    return report, prepared


def print_report(report: Dict[str, Any], write: bool) -> None:
    print("=== Historical Season Import Plan ===")
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

    print("Existing target rows:")
    for table_name, count in sorted(report["existing_target_rows"].items()):
        print("- %s: %s" % (table_name, count))
    print()

    print("Planned rows:")
    for table_name, count in sorted(report["planned_rows"].items()):
        print("- %s: %s" % (table_name, count))
    print()

    print("Mapping summary:")
    for key, value in sorted(report["mapping_summary"].items()):
        print("- %s: %s" % (key, value))
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

    report, prepared = build_report(
        season=args.season,
        player_gw_csv=args.player_gw_csv,
        fixtures_csv=args.fixtures_csv,
        player_mapping_csv=args.player_mapping_csv,
        team_mapping_csv=args.team_mapping_csv,
        allow_direct_player_id_mapping=args.allow_direct_player_id_mapping,
        allow_direct_team_id_mapping=args.allow_direct_team_id_mapping,
        sample_rows=args.sample_rows,
    )

    print_report(report, write=args.write)
    save_report(report, args.out_json)

    if not report["passed"]:
        raise SystemExit(1)

    if args.write:
        existing = report["existing_target_rows"]
        if not args.allow_existing_target_season and any(count > 0 for count in existing.values()):
            raise SystemExit(
                "Refusing to write because target season already has rows: %s. "
                "Use --allow-existing-target-season only if this is intentional." % existing
            )

        inserted = {
            "gameweeks": insert_rows("gameweeks", prepared["gameweeks"]),
            "fixtures": insert_rows("fixtures", prepared["fixtures"]),
            "player_gw_stats": insert_rows("player_gw_stats", prepared["player_gw_stats"]),
        }

        print("Inserted rows:")
        for table_name, count in sorted(inserted.items()):
            print("- %s: %s" % (table_name, count))
    else:
        print("Dry-run only. No database writes were performed.")


if __name__ == "__main__":
    main()
