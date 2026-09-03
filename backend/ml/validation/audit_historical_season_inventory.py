from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from sqlalchemy import text

from app.core.db import SessionLocal


AUDIT_VERSION = "fpl_historical_season_inventory_v2"

EXPECTED_EPL = {
    "teams": 20,
    "fixtures": 380,
    "gameweeks": 38,
}

CANONICAL_TABLES = {
    "teams": {
        "duplicate_key": ["season", "fpl_team_id"],
        "critical_fields": ["season", "fpl_team_id", "name", "short_name"],
    },
    "players": {
        "duplicate_key": ["season", "fpl_player_id"],
        "critical_fields": [
            "season",
            "fpl_player_id",
            "team_id",
            "position",
            "now_cost",
            "status",
        ],
    },
    "fixtures": {
        "duplicate_key": ["season", "fpl_fixture_id"],
        "critical_fields": [
            "season",
            "fpl_fixture_id",
            "gw",
            "home_team_id",
            "away_team_id",
        ],
    },
    "player_gw_stats": {
        "duplicate_key": ["season", "player_id", "gw"],
        "critical_fields": [
            "season",
            "player_id",
            "gw",
            "minutes",
            "total_points",
        ],
    },
    "gameweeks": {
        "duplicate_key": ["season", "gw"],
        "critical_fields": ["season", "gw"],
    },
}

STAGING_TABLES = {
    "historical_teams": {
        "duplicate_key": ["season", "raw_team_id"],
        "critical_fields": ["season", "raw_team_id"],
    },
    "historical_players": {
        "duplicate_key": ["season", "raw_player_id"],
        "critical_fields": ["season", "raw_player_id"],
    },
    "historical_fixtures": {
        "duplicate_key": [
            "season",
            "gw",
            "raw_home_team_id",
            "raw_away_team_id",
        ],
        "critical_fields": [
            "season",
            "gw",
            "raw_home_team_id",
            "raw_away_team_id",
        ],
    },
    "historical_player_gw_stats": {
        "duplicate_key": ["season", "raw_player_id", "gw"],
        "critical_fields": [
            "season",
            "raw_player_id",
            "gw",
            "minutes",
            "total_points",
        ],
    },
}

SCORING_FIELDS = [
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]

PRICE_FIELDS = ["value", "now_cost", "cost", "price"]
STATUS_FIELDS = [
    "status",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "news",
    "news_added",
]

RAW_PLAYER_GW_CANDIDATES = [
    "gws/merged_gw.csv",
    "gws/merged_gws.csv",
]
RAW_FIXTURE_CANDIDATES = ["fixtures.csv"]
RAW_TEAM_CANDIDATES = ["teams.csv"]
RAW_PLAYER_CANDIDATES = ["players_raw.csv", "cleaned_players.csv"]

SEASON_RE = re.compile(r"^(20\d{2})[-_](\d{2})$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_season_key(value: str) -> Optional[str]:
    match = SEASON_RE.match(str(value).strip())
    if not match:
        return None
    return "%s_%s" % (match.group(1), match.group(2))


def season_to_folder(season: str) -> str:
    return season.replace("_", "-")


def safe_table_name(table_name: str) -> str:
    allowed = set(CANONICAL_TABLES) | set(STAGING_TABLES)
    if table_name not in allowed:
        raise ValueError("Unsupported table name: %s" % table_name)
    return table_name


def table_columns(db: Any, table_name: str) -> List[str]:
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
    ).scalars().all()
    return [str(value) for value in rows]


def table_exists(db: Any, table_name: str) -> bool:
    value = db.execute(
        text(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = :table_name
              AND table_type = 'BASE TABLE'
            """
        ),
        {"table_name": table_name},
    ).scalar()
    return int(value or 0) > 0


def discover_database_seasons(db: Any) -> Set[str]:
    seasons: Set[str] = set()
    for table_name in list(CANONICAL_TABLES) + list(STAGING_TABLES):
        if not table_exists(db, table_name):
            continue
        columns = table_columns(db, table_name)
        if "season" not in columns:
            continue
        rows = db.execute(
            text('SELECT DISTINCT season FROM "%s" WHERE season IS NOT NULL' % table_name)
        ).scalars().all()
        for value in rows:
            normalized = normalize_season_key(str(value))
            if normalized:
                seasons.add(normalized)
    return seasons


def count_rows(db: Any, table_name: str, season: str) -> int:
    safe_table_name(table_name)
    value = db.execute(
        text('SELECT COUNT(*) FROM "%s" WHERE season = :season' % table_name),
        {"season": season},
    ).scalar()
    return int(value or 0)


def duplicate_group_count(
    db: Any,
    table_name: str,
    season: str,
    key_columns: Sequence[str],
) -> int:
    safe_table_name(table_name)
    columns = table_columns(db, table_name)
    if any(column not in columns for column in key_columns):
        return -1
    key_sql = ", ".join('"%s"' % column for column in key_columns)
    value = db.execute(
        text(
            """
            SELECT COUNT(*)
            FROM (
                SELECT %s, COUNT(*) AS n
                FROM "%s"
                WHERE season = :season
                GROUP BY %s
                HAVING COUNT(*) > 1
            ) duplicate_groups
            """
            % (key_sql, table_name, key_sql)
        ),
        {"season": season},
    ).scalar()
    return int(value or 0)


def null_counts(
    db: Any,
    table_name: str,
    season: str,
    fields: Sequence[str],
) -> Dict[str, Optional[int]]:
    safe_table_name(table_name)
    columns = set(table_columns(db, table_name))
    result: Dict[str, Optional[int]] = {}
    for field in fields:
        if field not in columns:
            result[field] = None
            continue
        value = db.execute(
            text(
                'SELECT COUNT(*) FROM "%s" '
                'WHERE season = :season AND "%s" IS NULL'
                % (table_name, field)
            ),
            {"season": season},
        ).scalar()
        result[field] = int(value or 0)
    return result



def all_column_null_counts(
    db: Any,
    table_name: str,
    season: str,
) -> Dict[str, int]:
    safe_table_name(table_name)
    columns = table_columns(db, table_name)
    if not columns:
        return {}
    expressions = [
        'SUM(CASE WHEN "%s" IS NULL THEN 1 ELSE 0 END) AS "%s"'
        % (column, column)
        for column in columns
    ]
    row = db.execute(
        text(
            'SELECT %s FROM "%s" WHERE season = :season'
            % (", ".join(expressions), table_name)
        ),
        {"season": season},
    ).mappings().first()
    return {
        column: int((row or {}).get(column) or 0)
        for column in columns
    }


def numeric_field_summary(
    db: Any,
    table_name: str,
    season: str,
    field: str,
) -> Dict[str, Any]:
    safe_table_name(table_name)
    columns = set(table_columns(db, table_name))
    if field not in columns:
        return {
            "available": False,
            "non_null_count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    row = db.execute(
        text(
            'SELECT COUNT("%s") AS non_null_count, '
            'MIN("%s") AS min_value, '
            'MAX("%s") AS max_value, '
            'AVG("%s") AS mean_value '
            'FROM "%s" WHERE season = :season'
            % (field, field, field, field, table_name)
        ),
        {"season": season},
    ).mappings().first()
    mean_value = (row or {}).get("mean_value")
    return {
        "available": True,
        "non_null_count": int((row or {}).get("non_null_count") or 0),
        "min": (row or {}).get("min_value"),
        "max": (row or {}).get("max_value"),
        "mean": float(mean_value) if mean_value is not None else None,
    }


def categorical_field_counts(
    db: Any,
    table_name: str,
    season: str,
    field: str,
) -> Dict[str, Any]:
    safe_table_name(table_name)
    columns = set(table_columns(db, table_name))
    if field not in columns:
        return {"available": False, "counts": {}}
    rows = db.execute(
        text(
            'SELECT "%s" AS value, COUNT(*) AS row_count '
            'FROM "%s" WHERE season = :season '
            'GROUP BY "%s" ORDER BY "%s" NULLS LAST'
            % (field, table_name, field, field)
        ),
        {"season": season},
    ).mappings().all()
    counts: Dict[str, int] = {}
    for row in rows:
        key = "<NULL>" if row["value"] is None else str(row["value"])
        counts[key] = int(row["row_count"] or 0)
    return {"available": True, "counts": counts}


def gw_coverage(db: Any, table_name: str, season: str) -> Dict[str, Any]:
    safe_table_name(table_name)
    columns = set(table_columns(db, table_name))
    if "gw" not in columns:
        return {
            "available": False,
            "min_gw": None,
            "max_gw": None,
            "distinct_gw_count": 0,
            "missing_gws_1_to_38": list(range(1, 39)),
        }
    row = db.execute(
        text(
            """
            SELECT
                MIN(gw) AS min_gw,
                MAX(gw) AS max_gw,
                COUNT(DISTINCT gw) AS distinct_gw_count
            FROM "%s"
            WHERE season = :season
            """
            % table_name
        ),
        {"season": season},
    ).mappings().first()
    gw_rows = db.execute(
        text(
            'SELECT DISTINCT gw FROM "%s" '
            'WHERE season = :season AND gw IS NOT NULL ORDER BY gw'
            % table_name
        ),
        {"season": season},
    ).scalars().all()
    observed = set(int(value) for value in gw_rows)
    return {
        "available": True,
        "min_gw": row["min_gw"] if row else None,
        "max_gw": row["max_gw"] if row else None,
        "distinct_gw_count": int((row or {}).get("distinct_gw_count") or 0),
        "missing_gws_1_to_38": [
            gw for gw in range(1, 39) if gw not in observed
        ],
    }


def fixture_completion(
    db: Any,
    table_name: str,
    season: str,
) -> Dict[str, Any]:
    safe_table_name(table_name)
    columns = set(table_columns(db, table_name))
    finished_col = "finished" if "finished" in columns else None
    if table_name == "fixtures":
        score_cols = ("home_score", "away_score")
    else:
        score_cols = ("home_score", "away_score")

    result = {
        "finished_field_available": bool(finished_col),
        "finished_true_count": None,
        "score_fields_available": all(column in columns for column in score_cols),
        "rows_with_complete_scores": None,
    }
    if finished_col:
        value = db.execute(
            text(
                'SELECT COUNT(*) FROM "%s" '
                'WHERE season = :season AND finished IS TRUE'
                % table_name
            ),
            {"season": season},
        ).scalar()
        result["finished_true_count"] = int(value or 0)
    if result["score_fields_available"]:
        value = db.execute(
            text(
                'SELECT COUNT(*) FROM "%s" '
                'WHERE season = :season '
                'AND home_score IS NOT NULL AND away_score IS NOT NULL'
                % table_name
            ),
            {"season": season},
        ).scalar()
        result["rows_with_complete_scores"] = int(value or 0)
    return result


def mapping_summary(
    db: Any,
    table_name: str,
    season: str,
    canonical_field: str,
) -> Dict[str, Any]:
    safe_table_name(table_name)
    columns = set(table_columns(db, table_name))
    if canonical_field not in columns:
        return {"available": False, "mapped": 0, "unmapped": 0}
    row = db.execute(
        text(
            """
            SELECT
                SUM(CASE WHEN "%s" IS NULL THEN 0 ELSE 1 END) AS mapped,
                SUM(CASE WHEN "%s" IS NULL THEN 1 ELSE 0 END) AS unmapped
            FROM "%s"
            WHERE season = :season
            """
            % (canonical_field, canonical_field, table_name)
        ),
        {"season": season},
    ).mappings().first()
    return {
        "available": True,
        "mapped": int((row or {}).get("mapped") or 0),
        "unmapped": int((row or {}).get("unmapped") or 0),
    }


def field_availability(
    db: Any,
    table_name: str,
    season: str,
    fields: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    safe_table_name(table_name)
    columns = set(table_columns(db, table_name))
    result: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        if field not in columns:
            result[field] = {
                "available": False,
                "null_count": None,
                "non_null_count": 0,
            }
            continue
        row = db.execute(
            text(
                """
                SELECT
                    SUM(CASE WHEN "%s" IS NULL THEN 1 ELSE 0 END) AS null_count,
                    SUM(CASE WHEN "%s" IS NULL THEN 0 ELSE 1 END) AS non_null_count
                FROM "%s"
                WHERE season = :season
                """
                % (field, field, table_name)
            ),
            {"season": season},
        ).mappings().first()
        result[field] = {
            "available": True,
            "null_count": int((row or {}).get("null_count") or 0),
            "non_null_count": int((row or {}).get("non_null_count") or 0),
        }
    return result


def profile_database_table(
    db: Any,
    table_name: str,
    season: str,
    contract: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    if not table_exists(db, table_name):
        return {
            "exists": False,
            "row_count": 0,
            "columns": [],
            "duplicate_groups": None,
            "critical_nulls": {},
            "gw_coverage": None,
        }

    columns = table_columns(db, table_name)
    row_count = count_rows(db, table_name, season)
    result = {
        "exists": True,
        "row_count": row_count,
        "columns": columns,
        "duplicate_key": list(contract["duplicate_key"]),
        "duplicate_groups": duplicate_group_count(
            db,
            table_name,
            season,
            contract["duplicate_key"],
        ),
        "critical_nulls": null_counts(
            db,
            table_name,
            season,
            contract["critical_fields"],
        ),
        "column_null_counts": all_column_null_counts(
            db,
            table_name,
            season,
        ),
        "gw_coverage": (
            gw_coverage(db, table_name, season)
            if "gw" in columns
            else None
        ),
    }
    if table_name in {"fixtures", "historical_fixtures"}:
        result["fixture_completion"] = fixture_completion(
            db,
            table_name,
            season,
        )
    return result


def build_database_season_inventory(db: Any, season: str) -> Dict[str, Any]:
    canonical = {
        table_name: profile_database_table(
            db,
            table_name,
            season,
            contract,
        )
        for table_name, contract in CANONICAL_TABLES.items()
    }
    staging = {
        table_name: profile_database_table(
            db,
            table_name,
            season,
            contract,
        )
        for table_name, contract in STAGING_TABLES.items()
    }

    canonical_player_fields = {}
    if canonical["players"]["exists"]:
        canonical_player_fields = {
            "prices": field_availability(
                db,
                "players",
                season,
                PRICE_FIELDS,
            ),
            "price_summaries": {
                field: numeric_field_summary(
                    db,
                    "players",
                    season,
                    field,
                )
                for field in PRICE_FIELDS
            },
            "statuses": field_availability(
                db,
                "players",
                season,
                STATUS_FIELDS,
            ),
            "status_counts": {
                field: categorical_field_counts(
                    db,
                    "players",
                    season,
                    field,
                )
                for field in STATUS_FIELDS
            },
        }

    canonical_scoring = {}
    if canonical["player_gw_stats"]["exists"]:
        canonical_scoring = field_availability(
            db,
            "player_gw_stats",
            season,
            SCORING_FIELDS,
        )

    staging_player_fields = {}
    if staging["historical_players"]["exists"]:
        staging_player_fields = {
            "statuses": field_availability(
                db,
                "historical_players",
                season,
                STATUS_FIELDS,
            ),
        }
    if staging["historical_player_gw_stats"]["exists"]:
        staging_player_fields["prices"] = field_availability(
            db,
            "historical_player_gw_stats",
            season,
            PRICE_FIELDS,
        )
        staging_player_fields["price_summaries"] = {
            field: numeric_field_summary(
                db,
                "historical_player_gw_stats",
                season,
                field,
            )
            for field in PRICE_FIELDS
        }

    staging_scoring = {}
    if staging["historical_player_gw_stats"]["exists"]:
        staging_scoring = field_availability(
            db,
            "historical_player_gw_stats",
            season,
            SCORING_FIELDS,
        )

    mapping = {
        "teams": mapping_summary(
            db,
            "historical_teams",
            season,
            "canonical_team_id",
        )
        if staging["historical_teams"]["exists"]
        else {"available": False, "mapped": 0, "unmapped": 0},
        "players": mapping_summary(
            db,
            "historical_players",
            season,
            "canonical_player_id",
        )
        if staging["historical_players"]["exists"]
        else {"available": False, "mapped": 0, "unmapped": 0},
    }

    return {
        "canonical": canonical,
        "staging": staging,
        "field_availability": {
            "canonical_players": canonical_player_fields,
            "canonical_player_gw_scoring": canonical_scoring,
            "staging_players": staging_player_fields,
            "staging_player_gw_scoring": staging_scoring,
        },
        "mapping": mapping,
    }


def csv_header(path: Path) -> List[str]:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as handle:
            reader = csv.reader(handle)
            return [str(value) for value in next(reader, [])]
    except Exception:
        return []


def find_first(base: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        path = base / candidate
        if path.exists() and path.is_file():
            return path
    return None


def count_csv_rows(path: Path) -> Optional[int]:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return None


def profile_csv(
    path: Path,
    *,
    duplicate_key_candidates: Optional[Sequence[Sequence[str]]] = None,
    required_fields: Optional[Sequence[str]] = None,
    gw_candidates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    header = csv_header(path)
    result: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "columns": header,
        "row_count": None,
        "duplicate_key": None,
        "duplicate_rows": None,
        "required_nulls": {},
        "gw_coverage": None,
    }
    if not path.exists():
        return result

    duplicate_key: Optional[List[str]] = None
    for candidate in duplicate_key_candidates or []:
        if all(column in header for column in candidate):
            duplicate_key = list(candidate)
            break

    gw_field = None
    for candidate in gw_candidates or []:
        if candidate in header:
            gw_field = candidate
            break

    required_present = [
        field for field in (required_fields or []) if field in header
    ]
    required_missing = [
        field for field in (required_fields or []) if field not in header
    ]

    seen: Set[Tuple[str, ...]] = set()
    duplicate_rows = 0
    required_nulls = {field: 0 for field in required_present}
    observed_gws: Set[int] = set()
    row_count = 0

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_count += 1
                if duplicate_key:
                    key = tuple(str(row.get(field, "")).strip() for field in duplicate_key)
                    if key in seen:
                        duplicate_rows += 1
                    else:
                        seen.add(key)
                for field in required_present:
                    value = row.get(field)
                    if value is None or str(value).strip() == "":
                        required_nulls[field] += 1
                if gw_field:
                    value = row.get(gw_field)
                    try:
                        observed_gws.add(int(float(str(value))))
                    except Exception:
                        pass
    except Exception as exc:
        result["read_error"] = "%s: %s" % (type(exc).__name__, exc)
        return result

    result["row_count"] = row_count
    result["duplicate_key"] = duplicate_key
    result["duplicate_rows"] = duplicate_rows if duplicate_key else None
    result["required_nulls"] = required_nulls
    result["missing_required_fields"] = required_missing
    if gw_field:
        result["gw_coverage"] = {
            "field": gw_field,
            "min_gw": min(observed_gws) if observed_gws else None,
            "max_gw": max(observed_gws) if observed_gws else None,
            "distinct_gw_count": len(observed_gws),
            "missing_gws_1_to_38": [
                gw for gw in range(1, 39) if gw not in observed_gws
            ],
        }
    return result


def discover_season_dirs(root: Optional[Path]) -> Dict[str, Path]:
    if root is None or not root.exists():
        return {}
    result: Dict[str, Path] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        season = normalize_season_key(child.name)
        if season:
            result[season] = child
    return result


def raw_source_inventory(
    raw_root: Optional[Path],
    season: str,
) -> Dict[str, Any]:
    if raw_root is None:
        return {"root": None, "season_dir_exists": False}
    season_dir = raw_root / season_to_folder(season)
    if not season_dir.exists():
        alternative = raw_root / season
        season_dir = alternative if alternative.exists() else season_dir

    result: Dict[str, Any] = {
        "root": str(raw_root),
        "season_dir": str(season_dir),
        "season_dir_exists": season_dir.exists(),
        "adapter_compatible": False,
    }
    if not season_dir.exists():
        return result

    player_gw = find_first(season_dir, RAW_PLAYER_GW_CANDIDATES)
    fixtures = find_first(season_dir, RAW_FIXTURE_CANDIDATES)
    teams = find_first(season_dir, RAW_TEAM_CANDIDATES)
    players = find_first(season_dir, RAW_PLAYER_CANDIDATES)

    gws_dir = season_dir / "gws"
    individual_gws = []
    if gws_dir.exists():
        for path in gws_dir.glob("gw*.csv"):
            match = re.match(r"gw(\d+)\.csv$", path.name, flags=re.IGNORECASE)
            if match:
                individual_gws.append(int(match.group(1)))

    result.update(
        {
            "source_files": {
                "player_gw": str(player_gw) if player_gw else None,
                "fixtures": str(fixtures) if fixtures else None,
                "teams": str(teams) if teams else None,
                "players": str(players) if players else None,
            },
            "individual_gw_files": {
                "count": len(set(individual_gws)),
                "min_gw": min(individual_gws) if individual_gws else None,
                "max_gw": max(individual_gws) if individual_gws else None,
                "missing_gws_1_to_38": [
                    gw for gw in range(1, 39) if gw not in set(individual_gws)
                ],
            },
            "adapter_compatible": bool(player_gw and fixtures),
        }
    )

    if player_gw:
        result["player_gw_profile"] = profile_csv(
            player_gw,
            duplicate_key_candidates=[
                ["element", "round"],
                ["element", "gw"],
                ["element", "event"],
            ],
            required_fields=["minutes", "total_points"],
            gw_candidates=["round", "gw", "event"],
        )
        columns = result["player_gw_profile"].get("columns", [])
        result["player_gw_profile"]["price_fields"] = [
            field for field in PRICE_FIELDS if field in columns
        ]
        result["player_gw_profile"]["status_fields"] = [
            field for field in STATUS_FIELDS if field in columns
        ]
        result["player_gw_profile"]["scoring_fields"] = [
            field for field in SCORING_FIELDS if field in columns
        ]

    if fixtures:
        result["fixture_profile"] = profile_csv(
            fixtures,
            duplicate_key_candidates=[
                ["id"],
                ["event", "team_h", "team_a"],
            ],
            required_fields=["team_h", "team_a"],
            gw_candidates=["event", "gw", "round"],
        )

    if players:
        player_columns = csv_header(players)
        result["player_columns"] = player_columns
        result["player_price_fields"] = [
            field for field in PRICE_FIELDS if field in player_columns
        ]
        result["player_status_fields"] = [
            field for field in STATUS_FIELDS if field in player_columns
        ]

    if teams:
        result["team_columns"] = csv_header(teams)

    return result


def prepared_source_inventory(
    prepared_root: Optional[Path],
    season: str,
) -> Dict[str, Any]:
    if prepared_root is None:
        return {"root": None, "season_dir_exists": False}
    season_dir = prepared_root / season
    if not season_dir.exists():
        alternative = prepared_root / season_to_folder(season)
        season_dir = alternative if alternative.exists() else season_dir

    result: Dict[str, Any] = {
        "root": str(prepared_root),
        "season_dir": str(season_dir),
        "season_dir_exists": season_dir.exists(),
    }
    if not season_dir.exists():
        return result

    known_files = {
        "player_gw_stats": season_dir / "player_gw_stats.csv",
        "fixtures": season_dir / "fixtures.csv",
        "team_mapping": season_dir / "team_mapping_template.csv",
        "player_mapping": season_dir / "player_mapping_template.csv",
        "player_season_summary": season_dir / "player_season_summary.csv",
        "team_season_summary": season_dir / "team_season_summary.csv",
    }

    files: Dict[str, Any] = {}
    for label, path in known_files.items():
        if not path.exists():
            files[label] = {"path": str(path), "exists": False}
            continue
        if label == "player_gw_stats":
            files[label] = profile_csv(
                path,
                duplicate_key_candidates=[
                    ["element", "gw"],
                    ["raw_player_id", "gw"],
                ],
                required_fields=[
                    "season",
                    "gw",
                    "minutes",
                    "total_points",
                ],
                gw_candidates=["gw"],
            )
        elif label == "fixtures":
            files[label] = profile_csv(
                path,
                duplicate_key_candidates=[
                    ["fpl_fixture_id"],
                    ["gw", "team_h", "team_a"],
                ],
                required_fields=["season", "gw", "team_h", "team_a"],
                gw_candidates=["gw"],
            )
        else:
            files[label] = {
                "path": str(path),
                "exists": True,
                "row_count": count_csv_rows(path),
                "columns": csv_header(path),
            }
    result["files"] = files
    result["prepared_core_available"] = bool(
        files.get("player_gw_stats", {}).get("exists")
        and files.get("fixtures", {}).get("exists")
    )
    return result



def gw_coverage_is_standard_full(
    coverage: Optional[Mapping[str, Any]],
) -> bool:
    if not coverage:
        return False
    return (
        int(coverage.get("distinct_gw_count") or 0)
        == EXPECTED_EPL["gameweeks"]
        and coverage.get("min_gw") == 1
        and coverage.get("max_gw") == EXPECTED_EPL["gameweeks"]
        and not list(coverage.get("missing_gws_1_to_38") or [])
    )


def source_profile_structurally_complete(
    profile: Mapping[str, Any],
    expected_rows: Optional[int] = None,
) -> bool:
    if not profile or profile.get("read_error"):
        return False
    row_count = int(profile.get("row_count") or 0)
    if row_count <= 0:
        return False
    if expected_rows is not None and row_count != expected_rows:
        return False
    if profile.get("missing_required_fields"):
        return False
    if any(
        int(value or 0) > 0
        for value in profile.get("required_nulls", {}).values()
    ):
        return False
    return True


def source_profile_full_season(
    profile: Mapping[str, Any],
    expected_rows: Optional[int] = None,
) -> bool:
    if not source_profile_structurally_complete(
        profile,
        expected_rows=expected_rows,
    ):
        return False
    coverage = profile.get("gw_coverage")
    if coverage and not gw_coverage_is_standard_full(coverage):
        return False
    return True


def coverage_timeline_description(
    label: str,
    coverage: Optional[Mapping[str, Any]],
) -> Optional[str]:
    if not coverage or gw_coverage_is_standard_full(coverage):
        return None
    return (
        "%s GW timeline requires normalization: distinct=%s, range=%s-%s, "
        "missing standard labels=%s"
        % (
            label,
            coverage.get("distinct_gw_count"),
            coverage.get("min_gw"),
            coverage.get("max_gw"),
            list(coverage.get("missing_gws_1_to_38") or []),
        )
    )


def raw_source_structurally_complete(
    raw_source: Mapping[str, Any],
) -> bool:
    if not raw_source.get("adapter_compatible"):
        return False
    return (
        source_profile_structurally_complete(
            raw_source.get("fixture_profile", {}),
            expected_rows=EXPECTED_EPL["fixtures"],
        )
        and source_profile_structurally_complete(
            raw_source.get("player_gw_profile", {}),
        )
    )


def raw_source_timeline_anomalies(
    raw_source: Mapping[str, Any],
) -> List[str]:
    anomalies: List[str] = []
    for label, profile_key in [
        ("raw fixtures", "fixture_profile"),
        ("raw player-GW", "player_gw_profile"),
    ]:
        profile = raw_source.get(profile_key, {})
        description = coverage_timeline_description(
            label,
            profile.get("gw_coverage") if profile else None,
        )
        if description:
            anomalies.append(description)

    individual = raw_source.get("individual_gw_files") or {}
    if individual:
        file_count = int(individual.get("count") or 0)
        max_gw = individual.get("max_gw")
        missing = list(individual.get("missing_gws_1_to_38") or [])
        if (
            file_count != EXPECTED_EPL["gameweeks"]
            or max_gw != EXPECTED_EPL["gameweeks"]
            or missing
        ):
            anomalies.append(
                "raw individual GW files require timeline normalization: "
                "count=%s, range=%s-%s, missing standard labels=%s"
                % (
                    file_count,
                    individual.get("min_gw"),
                    max_gw,
                    missing,
                )
            )
    return anomalies


def raw_source_full_enough_for_staging(raw_source: Mapping[str, Any]) -> bool:
    if not raw_source.get("adapter_compatible"):
        return False
    fixture_profile = raw_source.get("fixture_profile", {})
    player_profile = raw_source.get("player_gw_profile", {})
    return (
        source_profile_full_season(
            fixture_profile,
            expected_rows=EXPECTED_EPL["fixtures"],
        )
        and source_profile_full_season(player_profile)
        and int(
            (player_profile.get("gw_coverage") or {}).get("distinct_gw_count")
            or 0
        )
        == EXPECTED_EPL["gameweeks"]
    )


def prepared_source_full_enough_for_staging(
    prepared_source: Mapping[str, Any],
) -> bool:
    if not prepared_source.get("prepared_core_available"):
        return False
    files = prepared_source.get("files", {})
    fixture_profile = files.get("fixtures", {})
    player_profile = files.get("player_gw_stats", {})
    if not source_profile_full_season(
        fixture_profile,
        expected_rows=EXPECTED_EPL["fixtures"],
    ):
        return False
    if not source_profile_full_season(player_profile):
        return False
    if int(player_profile.get("duplicate_rows") or 0) > 0:
        return False
    return (
        int(
            (player_profile.get("gw_coverage") or {}).get("distinct_gw_count")
            or 0
        )
        == EXPECTED_EPL["gameweeks"]
    )

def prepared_source_structurally_complete(
    prepared_source: Mapping[str, Any],
) -> bool:
    if not prepared_source.get("prepared_core_available"):
        return False
    files = prepared_source.get("files", {})
    return (
        source_profile_structurally_complete(
            files.get("fixtures", {}),
            expected_rows=EXPECTED_EPL["fixtures"],
        )
        and source_profile_structurally_complete(
            files.get("player_gw_stats", {}),
        )
        and int(
            files.get("player_gw_stats", {}).get("duplicate_rows") or 0
        )
        == 0
    )


def prepared_source_timeline_anomalies(
    prepared_source: Mapping[str, Any],
) -> List[str]:
    anomalies: List[str] = []
    files = prepared_source.get("files", {})
    for label, file_key in [
        ("prepared fixtures", "fixtures"),
        ("prepared player-GW", "player_gw_stats"),
    ]:
        profile = files.get(file_key, {})
        description = coverage_timeline_description(
            label,
            profile.get("gw_coverage") if profile else None,
        )
        if description:
            anomalies.append(description)
    return anomalies


def positive_rows(table: Mapping[str, Any]) -> bool:
    return bool(table.get("exists") and int(table.get("row_count") or 0) > 0)


def no_duplicate_or_null_errors(
    tables: Mapping[str, Mapping[str, Any]],
    required_tables: Sequence[str],
) -> Tuple[bool, List[str]]:
    blockers: List[str] = []
    for table_name in required_tables:
        item = tables.get(table_name, {})
        if not positive_rows(item):
            blockers.append("%s has no rows" % table_name)
            continue
        duplicate_groups = item.get("duplicate_groups")
        if duplicate_groups not in (0, None):
            blockers.append(
                "%s has %s duplicate groups"
                % (table_name, duplicate_groups)
            )
        for field, count in item.get("critical_nulls", {}).items():
            if count is None:
                blockers.append(
                    "%s missing required field %s" % (table_name, field)
                )
            elif int(count) > 0:
                blockers.append(
                    "%s.%s has %s null rows"
                    % (table_name, field, count)
                )
    return len(blockers) == 0, blockers


def classify_season(
    season: str,
    database: Mapping[str, Any],
    raw_source: Mapping[str, Any],
    prepared_source: Mapping[str, Any],
    latest_database_season: Optional[str],
) -> Dict[str, Any]:
    canonical = database.get("canonical", {})
    staging = database.get("staging", {})
    blockers: List[str] = []
    evidence: List[str] = []

    canonical_quality_ok, canonical_quality_blockers = no_duplicate_or_null_errors(
        canonical,
        ["teams", "players", "fixtures", "player_gw_stats"],
    )
    staging_quality_ok, staging_quality_blockers = no_duplicate_or_null_errors(
        staging,
        [
            "historical_teams",
            "historical_players",
            "historical_fixtures",
            "historical_player_gw_stats",
        ],
    )

    canonical_teams = int(canonical.get("teams", {}).get("row_count") or 0)
    canonical_fixtures = int(
        canonical.get("fixtures", {}).get("row_count") or 0
    )
    canonical_player_gw = int(
        canonical.get("player_gw_stats", {}).get("row_count") or 0
    )
    canonical_fixture_gws = int(
        (
            canonical.get("fixtures", {}).get("gw_coverage")
            or {}
        ).get("distinct_gw_count")
        or 0
    )
    canonical_player_gws = int(
        (
            canonical.get("player_gw_stats", {}).get("gw_coverage")
            or {}
        ).get("distinct_gw_count")
        or 0
    )
    fixture_completion = canonical.get("fixtures", {}).get(
        "fixture_completion", {}
    )
    canonical_finished = int(
        fixture_completion.get("finished_true_count") or 0
    )
    canonical_scored = int(
        fixture_completion.get("rows_with_complete_scores") or 0
    )

    canonical_full = (
        canonical_quality_ok
        and canonical_teams == EXPECTED_EPL["teams"]
        and canonical_fixtures == EXPECTED_EPL["fixtures"]
        and canonical_fixture_gws == EXPECTED_EPL["gameweeks"]
        and canonical_player_gw > 0
        and canonical_player_gws == EXPECTED_EPL["gameweeks"]
        and canonical_finished == EXPECTED_EPL["fixtures"]
        and canonical_scored == EXPECTED_EPL["fixtures"]
    )

    staging_teams = int(
        staging.get("historical_teams", {}).get("row_count") or 0
    )
    staging_fixtures = int(
        staging.get("historical_fixtures", {}).get("row_count") or 0
    )
    staging_player_gw = int(
        staging.get("historical_player_gw_stats", {}).get("row_count") or 0
    )
    staging_fixture_gws = int(
        (
            staging.get("historical_fixtures", {}).get("gw_coverage")
            or {}
        ).get("distinct_gw_count")
        or 0
    )
    staging_player_gws = int(
        (
            staging.get("historical_player_gw_stats", {}).get("gw_coverage")
            or {}
        ).get("distinct_gw_count")
        or 0
    )
    staging_completion = staging.get("historical_fixtures", {}).get(
        "fixture_completion", {}
    )
    staging_scored = int(
        staging_completion.get("rows_with_complete_scores") or 0
    )

    staging_full = (
        staging_quality_ok
        and staging_teams == EXPECTED_EPL["teams"]
        and staging_fixtures == EXPECTED_EPL["fixtures"]
        and staging_fixture_gws == EXPECTED_EPL["gameweeks"]
        and staging_player_gw > 0
        and staging_player_gws == EXPECTED_EPL["gameweeks"]
        and staging_scored == EXPECTED_EPL["fixtures"]
    )

    mapping = database.get("mapping", {})
    unmapped_teams = int(
        mapping.get("teams", {}).get("unmapped") or 0
    )
    unmapped_players = int(
        mapping.get("players", {}).get("unmapped") or 0
    )

    any_canonical_actuals = (
        canonical_fixtures > 0 or canonical_player_gw > 0
    )
    any_staging_actuals = (
        staging_fixtures > 0 or staging_player_gw > 0
    )
    raw_available = bool(raw_source.get("season_dir_exists"))
    raw_adapter_compatible = bool(raw_source.get("adapter_compatible"))
    raw_structurally_complete = raw_source_structurally_complete(raw_source)
    raw_full_enough = raw_source_full_enough_for_staging(raw_source)
    raw_timeline_anomalies = raw_source_timeline_anomalies(raw_source)

    prepared_available = bool(prepared_source.get("prepared_core_available"))
    prepared_structurally_complete = (
        prepared_source_structurally_complete(prepared_source)
    )
    prepared_full_enough = prepared_source_full_enough_for_staging(
        prepared_source
    )
    prepared_timeline_anomalies = prepared_source_timeline_anomalies(
        prepared_source
    )

    if canonical_full:
        evidence.append(
            "canonical season has 20 teams, 380 completed/scored fixtures, "
            "38 fixture GWs, and 38 player-stat GWs with clean critical keys"
        )
        return {
            "classification": "training-ready",
            "scope_role": (
                "current_or_latest_complete"
                if season == latest_database_season
                else "historical"
            ),
            "reasons": evidence,
            "blockers": [],
        }

    if staging_full and (unmapped_teams > 0 or unmapped_players > 0):
        evidence.append(
            "staging season is structurally complete and quality-clean"
        )
        blockers.append(
            "canonical identity mapping incomplete: %s teams, %s players unmapped"
            % (unmapped_teams, unmapped_players)
        )
        return {
            "classification": "mapping-required",
            "scope_role": "historical",
            "reasons": evidence,
            "blockers": blockers,
        }

    if any_canonical_actuals and canonical_quality_ok:
        evidence.append(
            "canonical actuals exist and critical keys/null checks pass"
        )
        if canonical_fixtures != EXPECTED_EPL["fixtures"]:
            blockers.append(
                "fixture coverage incomplete: %s/%s"
                % (canonical_fixtures, EXPECTED_EPL["fixtures"])
            )
        if canonical_player_gws != EXPECTED_EPL["gameweeks"]:
            blockers.append(
                "player-GW coverage incomplete: %s/%s GWs"
                % (
                    canonical_player_gws,
                    EXPECTED_EPL["gameweeks"],
                )
            )
        if canonical_finished != EXPECTED_EPL["fixtures"]:
            blockers.append(
                "completed fixtures: %s/%s"
                % (canonical_finished, EXPECTED_EPL["fixtures"])
            )
        return {
            "classification": "evaluation-only",
            "scope_role": (
                "current_in_progress"
                if season == latest_database_season
                else "partial_canonical"
            ),
            "reasons": evidence,
            "blockers": blockers,
        }

    if staging_full:
        evidence.append(
            "staging season is structurally complete"
        )
        blockers.append(
            "staging data is not established as a canonical/model-training input"
        )
        return {
            "classification": "evaluation-only",
            "scope_role": "historical_staging",
            "reasons": evidence,
            "blockers": blockers,
        }

    if (
        prepared_full_enough
        or prepared_structurally_complete
        or raw_full_enough
        or raw_structurally_complete
        or any_staging_actuals
    ):
        if prepared_full_enough:
            evidence.append(
                "prepared historical core CSVs have full standard "
                "1-38 GW / 380-fixture coverage"
            )
        elif prepared_structurally_complete:
            evidence.append(
                "prepared historical core CSVs are structurally complete "
                "but require GW timeline normalization"
            )
            blockers.extend(prepared_timeline_anomalies)

        if raw_full_enough:
            evidence.append(
                "raw source is adapter-compatible and has full standard "
                "1-38 GW / 380-fixture coverage"
            )
        elif raw_structurally_complete:
            evidence.append(
                "raw source has complete core fixture/player-GW data "
                "but requires GW timeline normalization"
            )
            blockers.extend(raw_timeline_anomalies)

        if any_staging_actuals:
            evidence.append("some historical staging rows exist")
        blockers.extend(staging_quality_blockers)
        if not staging_full:
            blockers.append(
                "full clean historical staging inventory is not established"
            )
        blockers.append(
            "canonical player/team identity mapping or canonical import remains unresolved"
        )
        return {
            "classification": "mapping-required",
            "scope_role": "historical_source_only",
            "reasons": evidence,
            "blockers": sorted(set(blockers)),
        }

    if prepared_available or raw_available:
        if prepared_available:
            evidence.append("prepared historical core CSVs are present")
        if raw_available:
            evidence.append("a raw season folder exists")
        if raw_adapter_compatible and not raw_structurally_complete:
            blockers.append(
                "raw source is adapter-compatible but minimum complete "
                "380-fixture + player-GW core quality is not established"
            )
        elif raw_available and not raw_adapter_compatible:
            blockers.append(
                "raw source is not adapter-compatible with both fixtures and merged player-GW data"
            )
        if prepared_available and not prepared_full_enough:
            blockers.append(
                "prepared source does not satisfy full 38-GW / 380-fixture quality checks"
            )
        return {
            "classification": "unusable",
            "scope_role": "raw_source_only",
            "reasons": evidence,
            "blockers": blockers,
        }

    blockers.extend(canonical_quality_blockers)
    blockers.extend(staging_quality_blockers)
    if not blockers:
        blockers.append("no canonical, staging, prepared, or raw source data found")
    return {
        "classification": "unusable",
        "scope_role": "no_usable_inventory",
        "reasons": [],
        "blockers": sorted(set(blockers)),
    }


def default_raw_root() -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    candidate = (
        repo_root.parent
        / "datasets"
        / "Fantasy-Premier-League"
        / "data"
    )
    return candidate if candidate.exists() else None


def default_prepared_root() -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root.parent / "datasets" / "prepared-fpl-historical"
    return candidate if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit canonical, historical staging, prepared, and raw FPL season "
            "availability. Read-only; does not modify database or source data."
        )
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument(
        "--raw-data-root",
        default=None,
        help=(
            "Optional Vaastav Fantasy-Premier-League/data root. "
            "Defaults to sibling datasets/Fantasy-Premier-League/data when present."
        ),
    )
    parser.add_argument(
        "--prepared-root",
        default=None,
        help=(
            "Optional prepared historical root. Defaults to sibling "
            "datasets/prepared-fpl-historical when present."
        ),
    )
    parser.add_argument(
        "--season",
        action="append",
        dest="seasons",
        default=None,
        help=(
            "Optional season filter. Repeat for multiple seasons. "
            "Without this flag the audit inventories every discovered season."
        ),
    )
    return parser.parse_args()


def summarize_classifications(
    seasons: Mapping[str, Mapping[str, Any]],
) -> Dict[str, List[str]]:
    result = {
        "training-ready": [],
        "evaluation-only": [],
        "mapping-required": [],
        "unusable": [],
    }
    for season, item in seasons.items():
        classification = item["classification"]["classification"]
        result.setdefault(classification, []).append(season)
    for values in result.values():
        values.sort()
    return result


def build_inventory(
    *,
    raw_root: Optional[Path],
    prepared_root: Optional[Path],
    requested_seasons: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        db_seasons = discover_database_seasons(db)
        raw_seasons = set(discover_season_dirs(raw_root).keys())
        prepared_seasons = set(discover_season_dirs(prepared_root).keys())

        if requested_seasons:
            seasons = set()
            for value in requested_seasons:
                normalized = normalize_season_key(value)
                if normalized is None:
                    raise ValueError(
                        "Invalid season key %r; expected YYYY_YY or YYYY-YY."
                        % value
                    )
                seasons.add(normalized)
        else:
            seasons = db_seasons | raw_seasons | prepared_seasons

        sorted_seasons = sorted(seasons)
        latest_database_season = (
            max(db_seasons) if db_seasons else None
        )

        season_results: Dict[str, Any] = {}
        for season in sorted_seasons:
            database = build_database_season_inventory(db, season)
            raw_source = raw_source_inventory(raw_root, season)
            prepared_source = prepared_source_inventory(
                prepared_root,
                season,
            )
            classification = classify_season(
                season,
                database,
                raw_source,
                prepared_source,
                latest_database_season,
            )
            season_results[season] = {
                "database": database,
                "raw_source": raw_source,
                "prepared_source": prepared_source,
                "classification": classification,
            }

        report = {
            "audit_version": AUDIT_VERSION,
            "created_at": utc_now(),
            "read_only": True,
            "expected_full_epl": EXPECTED_EPL,
            "source_roots": {
                "raw_data_root": str(raw_root) if raw_root else None,
                "prepared_root": (
                    str(prepared_root) if prepared_root else None
                ),
            },
            "discovery": {
                "database_seasons": sorted(db_seasons),
                "raw_source_seasons": sorted(raw_seasons),
                "prepared_seasons": sorted(prepared_seasons),
                "audited_seasons": sorted_seasons,
                "latest_database_season": latest_database_season,
            },
            "seasons": season_results,
        }
        report["classification_summary"] = summarize_classifications(
            season_results
        )
        return report
    finally:
        db.close()


def write_json(report: Mapping[str, Any], path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def source_label(item: Mapping[str, Any]) -> str:
    database = item["database"]
    canonical_rows = sum(
        int(table.get("row_count") or 0)
        for table in database.get("canonical", {}).values()
    )
    staging_rows = sum(
        int(table.get("row_count") or 0)
        for table in database.get("staging", {}).values()
    )
    labels = []
    if canonical_rows:
        labels.append("canonical")
    if staging_rows:
        labels.append("staging")
    if item.get("prepared_source", {}).get("season_dir_exists"):
        labels.append("prepared")
    if item.get("raw_source", {}).get("season_dir_exists"):
        labels.append("raw")
    return ", ".join(labels) if labels else "none"


def fields_with_season_data(
    metadata: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    return sorted(
        field
        for field, meta in metadata.items()
        if meta.get("available")
        and int(meta.get("non_null_count") or 0) > 0
    )


def coverage_text(
    coverage: Optional[Mapping[str, Any]],
) -> str:
    if not coverage:
        return "n/a"
    missing = list(coverage.get("missing_gws_1_to_38") or [])
    base = "%s GWs (%s-%s)" % (
        coverage.get("distinct_gw_count"),
        coverage.get("min_gw"),
        coverage.get("max_gw"),
    )
    if missing:
        base += "; missing standard labels=%s" % missing
    return base


def write_markdown(report: Mapping[str, Any], path_str: str) -> None:
    lines: List[str] = []
    lines.append("# Historical Season Inventory Audit")
    lines.append("")
    lines.append("- audit_version: `%s`" % report["audit_version"])
    lines.append("- created_at: `%s`" % report["created_at"])
    lines.append("- read_only: `%s`" % report["read_only"])
    lines.append("")
    lines.append("## Classification Summary")
    lines.append("")
    for classification in [
        "training-ready",
        "evaluation-only",
        "mapping-required",
        "unusable",
    ]:
        seasons = report["classification_summary"].get(
            classification, []
        )
        lines.append(
            "- **%s**: %s"
            % (
                classification,
                ", ".join("`%s`" % season for season in seasons)
                if seasons
                else "None",
            )
        )
    lines.append("")
    lines.append("## Season Inventory")
    lines.append("")
    lines.append(
        "| Season | Classification | Role | Sources | Canonical fixtures | "
        "Canonical player-GW | Staging fixtures | Staging player-GW |"
    )
    lines.append(
        "|---|---|---|---|---:|---:|---:|---:|"
    )
    for season, item in sorted(report["seasons"].items()):
        classification = item["classification"]
        canonical = item["database"]["canonical"]
        staging = item["database"]["staging"]
        lines.append(
            "| `%s` | **%s** | %s | %s | %s | %s | %s | %s |"
            % (
                season,
                classification["classification"],
                classification["scope_role"],
                source_label(item),
                canonical.get("fixtures", {}).get("row_count", 0),
                canonical.get("player_gw_stats", {}).get(
                    "row_count", 0
                ),
                staging.get("historical_fixtures", {}).get(
                    "row_count", 0
                ),
                staging.get(
                    "historical_player_gw_stats", {}
                ).get("row_count", 0),
            )
        )
    lines.append("")

    for season, item in sorted(report["seasons"].items()):
        classification = item["classification"]
        database = item["database"]
        canonical = database["canonical"]
        staging = database["staging"]
        lines.append("## %s" % season)
        lines.append("")
        lines.append(
            "- classification: **%s**"
            % classification["classification"]
        )
        lines.append(
            "- scope_role: `%s`" % classification["scope_role"]
        )
        for reason in classification.get("reasons", []):
            lines.append("- evidence: %s" % reason)
        for blocker in classification.get("blockers", []):
            lines.append("- blocker: %s" % blocker)
        lines.append("")
        lines.append("### Canonical")
        lines.append("")
        for table_name in [
            "teams",
            "players",
            "fixtures",
            "player_gw_stats",
            "gameweeks",
        ]:
            table = canonical.get(table_name, {})
            coverage = table.get("gw_coverage") or {}
            coverage_label = (
                "%s GWs (%s-%s)"
                % (
                    coverage.get("distinct_gw_count"),
                    coverage.get("min_gw"),
                    coverage.get("max_gw"),
                )
                if coverage
                else "n/a"
            )
            lines.append(
                "- `%s`: rows=%s, duplicate_groups=%s, gw=%s"
                % (
                    table_name,
                    table.get("row_count", 0),
                    table.get("duplicate_groups"),
                    coverage_label,
                )
            )
        lines.append("")
        lines.append("### Historical staging")
        lines.append("")
        for table_name in [
            "historical_teams",
            "historical_players",
            "historical_fixtures",
            "historical_player_gw_stats",
        ]:
            table = staging.get(table_name, {})
            coverage = table.get("gw_coverage") or {}
            coverage_label = (
                "%s GWs (%s-%s)"
                % (
                    coverage.get("distinct_gw_count"),
                    coverage.get("min_gw"),
                    coverage.get("max_gw"),
                )
                if coverage
                else "n/a"
            )
            lines.append(
                "- `%s`: rows=%s, duplicate_groups=%s, gw=%s"
                % (
                    table_name,
                    table.get("row_count", 0),
                    table.get("duplicate_groups"),
                    coverage_label,
                )
            )
        lines.append("")
        lines.append("### Price / status / scoring field availability")
        lines.append("")
        fields = database["field_availability"]
        canonical_player = fields.get("canonical_players", {})
        staging_player = fields.get("staging_players", {})
        canonical_scoring = fields.get(
            "canonical_player_gw_scoring", {}
        )
        staging_scoring = fields.get(
            "staging_player_gw_scoring", {}
        )
        raw_source = item["raw_source"]
        prepared_source = item["prepared_source"]
        raw_player_gw = raw_source.get("player_gw_profile", {})

        lines.append(
            "- canonical player rows: `%s`; price fields with season data: `%s`"
            % (
                canonical.get("players", {}).get("row_count", 0),
                ", ".join(
                    fields_with_season_data(
                        canonical_player.get("prices", {})
                    )
                ),
            )
        )
        lines.append(
            "- canonical player rows: `%s`; status fields with season data: `%s`"
            % (
                canonical.get("players", {}).get("row_count", 0),
                ", ".join(
                    fields_with_season_data(
                        canonical_player.get("statuses", {})
                    )
                ),
            )
        )
        lines.append(
            "- canonical player-GW rows: `%s`; scoring fields with season data: `%s`"
            % (
                canonical.get("player_gw_stats", {}).get("row_count", 0),
                ", ".join(fields_with_season_data(canonical_scoring)),
            )
        )
        lines.append(
            "- staging player rows: `%s`; price fields with season data: `%s`"
            % (
                staging.get("historical_players", {}).get("row_count", 0),
                ", ".join(
                    fields_with_season_data(
                        staging_player.get("prices", {})
                    )
                ),
            )
        )
        lines.append(
            "- staging player rows: `%s`; status fields with season data: `%s`"
            % (
                staging.get("historical_players", {}).get("row_count", 0),
                ", ".join(
                    fields_with_season_data(
                        staging_player.get("statuses", {})
                    )
                ),
            )
        )
        lines.append(
            "- staging player-GW rows: `%s`; scoring fields with season data: `%s`"
            % (
                staging.get(
                    "historical_player_gw_stats", {}
                ).get("row_count", 0),
                ", ".join(fields_with_season_data(staging_scoring)),
            )
        )
        lines.append(
            "- raw player price fields: `%s`"
            % ", ".join(raw_source.get("player_price_fields") or [])
        )
        lines.append(
            "- raw player status fields: `%s`"
            % ", ".join(raw_source.get("player_status_fields") or [])
        )
        lines.append(
            "- raw player-GW price fields: `%s`"
            % ", ".join(raw_player_gw.get("price_fields") or [])
        )
        lines.append(
            "- raw player-GW scoring fields: `%s`"
            % ", ".join(raw_player_gw.get("scoring_fields") or [])
        )
        lines.append("")
        lines.append("### Filesystem sources")
        lines.append("")
        lines.append(
            "- raw season dir: `%s`; adapter_compatible=`%s`"
            % (
                raw_source.get("season_dir_exists"),
                raw_source.get("adapter_compatible"),
            )
        )
        raw_fixture = raw_source.get("fixture_profile") or {}
        raw_player_gw = raw_source.get("player_gw_profile") or {}
        individual_gws = raw_source.get("individual_gw_files") or {}
        if raw_fixture:
            lines.append(
                "- raw fixtures: rows=%s, duplicate_rows=%s, gw=%s"
                % (
                    raw_fixture.get("row_count"),
                    raw_fixture.get("duplicate_rows"),
                    coverage_text(raw_fixture.get("gw_coverage")),
                )
            )
        if raw_player_gw:
            lines.append(
                "- raw player-GW: rows=%s, duplicate_rows=%s, gw=%s"
                % (
                    raw_player_gw.get("row_count"),
                    raw_player_gw.get("duplicate_rows"),
                    coverage_text(raw_player_gw.get("gw_coverage")),
                )
            )
        if individual_gws:
            lines.append(
                "- raw individual GW files: count=%s, range=%s-%s, "
                "missing standard labels=%s"
                % (
                    individual_gws.get("count"),
                    individual_gws.get("min_gw"),
                    individual_gws.get("max_gw"),
                    list(
                        individual_gws.get(
                            "missing_gws_1_to_38", []
                        )
                    ),
                )
            )
        lines.append(
            "- prepared season dir: `%s`; prepared_core_available=`%s`"
            % (
                prepared_source.get("season_dir_exists"),
                prepared_source.get("prepared_core_available", False),
            )
        )
        lines.append("")

    lines.append("## Interpretation Contract")
    lines.append("")
    lines.append(
        "- `training-ready`: complete, clean canonical season with 20 teams, "
        "380 completed/scored fixtures, and full 38-GW player actual coverage."
    )
    lines.append(
        "- `evaluation-only`: usable canonical actuals exist, but the season is "
        "partial/in-progress or otherwise not a complete training season."
    )
    lines.append(
        "- `mapping-required`: historical staging/prepared/raw data exists, "
        "but canonical identity/model integration is not yet safely resolved."
    )
    lines.append(
        "- `unusable`: the currently available source is incomplete or lacks "
        "the minimum adapter/database contracts required for safe use."
    )
    lines.append("")
    lines.append(
        "Classification is conservative and describes the project's current "
        "safe capability; it does not claim every folder on disk is model-ready."
    )
    lines.append("")

    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(report: Mapping[str, Any]) -> None:
    print("=== Historical Season Inventory Audit ===")
    print("audit_version:", report["audit_version"])
    print("seasons:", len(report["seasons"]))
    for classification in [
        "training-ready",
        "evaluation-only",
        "mapping-required",
        "unusable",
    ]:
        values = report["classification_summary"].get(
            classification, []
        )
        print(
            "%s: %s"
            % (
                classification,
                ", ".join(values) if values else "-",
            )
        )


def main() -> None:
    args = parse_args()

    raw_root = (
        Path(args.raw_data_root)
        if args.raw_data_root
        else default_raw_root()
    )
    prepared_root = (
        Path(args.prepared_root)
        if args.prepared_root
        else default_prepared_root()
    )

    report = build_inventory(
        raw_root=raw_root,
        prepared_root=prepared_root,
        requested_seasons=args.seasons,
    )
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_summary(report)
    print("saved_json:", args.out_json)
    print("saved_md:", args.out_md)


if __name__ == "__main__":
    main()
