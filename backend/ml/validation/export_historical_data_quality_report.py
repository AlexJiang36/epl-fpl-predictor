from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal


STAGING_TABLES = [
    "historical_teams",
    "historical_players",
    "historical_fixtures",
    "historical_player_gw_stats",
]

CANONICAL_SEASON_TABLES = [
    "fixtures",
    "player_gw_stats",
    "gameweeks",
    "predictions",
    "match_predictions",
]

DUPLICATE_CHECKS = [
    ("historical_teams", ["season", "raw_team_id"]),
    ("historical_players", ["season", "raw_player_id"]),
    ("historical_fixtures", ["season", "gw", "raw_home_team_id", "raw_away_team_id"]),
    ("historical_player_gw_stats", ["season", "raw_player_id", "gw"]),
]

CRITICAL_NULL_CHECKS = {
    "historical_teams": ["season", "raw_team_id"],
    "historical_players": ["season", "raw_player_id"],
    "historical_fixtures": ["season", "gw", "raw_home_team_id", "raw_away_team_id"],
    "historical_player_gw_stats": ["season", "raw_player_id", "gw", "minutes", "total_points"],
}

EXPECTED_FULL_EPL = {
    "teams": 20,
    "fixtures": 380,
    "team_matches_counted_twice": 760,
    "gameweeks": 38,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a historical data quality report from staging tables. "
            "This is read-only and does not modify the database."
        )
    )
    parser.add_argument("--season", required=True, help="Historical season key, for example 2024_25.")
    parser.add_argument("--out-json", required=True, help="Output JSON report path.")
    parser.add_argument("--out-md", default=None, help="Optional Markdown summary path.")
    parser.add_argument(
        "--prepared-dir",
        default=None,
        help=(
            "Optional prepared historical data directory. If supplied, the report validates "
            "player_season_summary.csv and team_season_summary.csv when present."
        ),
    )
    parser.add_argument(
        "--allow-canonical-season-rows",
        action="store_true",
        help=(
            "Allow canonical fixtures/player_gw_stats rows for this historical season. "
            "Do not use during Day64C unless canonical import is intentional."
        ),
    )
    parser.add_argument("--sample-limit", type=int, default=20)
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


def gw_coverage(table_name: str, season: str) -> Dict[str, Any]:
    if not table_exists(table_name):
        return {"exists": False, "min_gw": None, "max_gw": None, "distinct_gw_count": 0}

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
        return {
            "exists": True,
            "min_gw": row["min_gw"] if row else None,
            "max_gw": row["max_gw"] if row else None,
            "distinct_gw_count": int((row or {}).get("distinct_gw_count") or 0),
        }
    finally:
        db.close()


def normalize_raw_id(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    if text_value.endswith(".0"):
        text_value = text_value[:-2]
    return text_value


def normalized_id_set(series: pd.Series) -> Set[str]:
    result: Set[str] = set()
    for value in series.tolist():
        normalized = normalize_raw_id(value)
        if normalized is not None:
            result.add(normalized)
    return result


def sorted_sample(values: Set[str], sample_limit: int) -> List[str]:
    return sorted(values)[:sample_limit]


def null_counts(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for column in columns:
        if column not in df.columns:
            result[column] = -1
        else:
            result[column] = int(df[column].isna().sum())
    return result


def mapping_counts(df: pd.DataFrame, canonical_column: str) -> Dict[str, int]:
    if df.empty or canonical_column not in df.columns:
        return {"mapped": 0, "unmapped": 0}
    mapped = int(df[canonical_column].notna().sum())
    unmapped = int(df[canonical_column].isna().sum())
    return {"mapped": mapped, "unmapped": unmapped}


def fixture_team_records(fixtures: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for _, row in fixtures.iterrows():
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        records.append(
            {
                "season": row["season"],
                "raw_team_id": normalize_raw_id(row["raw_home_team_id"]),
                "gw": row["gw"],
                "goals_for": int(home_score),
                "goals_against": int(away_score),
            }
        )
        records.append(
            {
                "season": row["season"],
                "raw_team_id": normalize_raw_id(row["raw_away_team_id"]),
                "gw": row["gw"],
                "goals_for": int(away_score),
                "goals_against": int(home_score),
            }
        )

    return pd.DataFrame(records)


def build_table_snapshots(season: str) -> Dict[str, pd.DataFrame]:
    snapshots: Dict[str, pd.DataFrame] = {}

    for table_name in STAGING_TABLES:
        if not table_exists(table_name):
            snapshots[table_name] = pd.DataFrame()
            continue

        snapshots[table_name] = read_sql_dataframe(
            "SELECT * FROM %s WHERE season = :season" % table_name,
            {"season": season},
        )

    if table_exists("teams"):
        snapshots["current_teams"] = read_sql_dataframe(
            """
            SELECT id, fpl_team_id, name, short_name
            FROM teams
            ORDER BY id
            """,
            {},
        )
    else:
        snapshots["current_teams"] = pd.DataFrame()

    return snapshots


def validate_artifacts(
    prepared_dir: Optional[str],
    staging_counts: Dict[str, Optional[int]],
    player_stats_summary: Dict[str, Any],
    team_fixture_summary: Dict[str, Any],
) -> Dict[str, Any]:
    if not prepared_dir:
        return {
            "enabled": False,
            "notes": ["No --prepared-dir supplied; artifact validation skipped."],
        }

    base = Path(prepared_dir)
    player_path = base / "player_season_summary.csv"
    team_path = base / "team_season_summary.csv"

    result: Dict[str, Any] = {
        "enabled": True,
        "prepared_dir": str(base),
        "player_summary": {"path": str(player_path), "exists": player_path.exists()},
        "team_summary": {"path": str(team_path), "exists": team_path.exists()},
    }

    if player_path.exists():
        players = pd.read_csv(player_path)
        result["player_summary"].update(
            {
                "row_count": int(len(players)),
                "active_player_count": int((players["minutes"] > 0).sum()) if "minutes" in players.columns else None,
                "total_minutes": int(players["minutes"].sum()) if "minutes" in players.columns else None,
                "total_points": int(players["total_points"].sum()) if "total_points" in players.columns else None,
                "row_count_matches_historical_players": (
                    int(len(players)) == int(staging_counts.get("historical_players") or 0)
                ),
                "total_minutes_matches_staging": (
                    int(players["minutes"].sum()) == int(player_stats_summary.get("total_minutes") or 0)
                    if "minutes" in players.columns
                    else None
                ),
                "total_points_matches_staging": (
                    int(players["total_points"].sum()) == int(player_stats_summary.get("total_points") or 0)
                    if "total_points" in players.columns
                    else None
                ),
            }
        )

    if team_path.exists():
        teams = pd.read_csv(team_path)
        result["team_summary"].update(
            {
                "row_count": int(len(teams)),
                "scheduled_fixtures_sum": int(teams["scheduled_fixtures"].sum()) if "scheduled_fixtures" in teams.columns else None,
                "matches_sum": int(teams["matches"].sum()) if "matches" in teams.columns else None,
                "total_points": int(teams["points"].sum()) if "points" in teams.columns else None,
                "row_count_matches_historical_teams": (
                    int(len(teams)) == int(staging_counts.get("historical_teams") or 0)
                ),
                "matches_sum_matches_staging": (
                    int(teams["matches"].sum()) == int(team_fixture_summary.get("team_matches_counted_twice") or 0)
                    if "matches" in teams.columns
                    else None
                ),
            }
        )

    return result


def build_report(
    season: str,
    prepared_dir: Optional[str],
    allow_canonical_season_rows: bool,
    sample_limit: int,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    missing_staging_tables = [table_name for table_name in STAGING_TABLES if not table_exists(table_name)]
    if missing_staging_tables:
        errors.append("Missing historical staging tables: %s" % missing_staging_tables)

    data = build_table_snapshots(season)

    teams = data["historical_teams"]
    players = data["historical_players"]
    fixtures = data["historical_fixtures"]
    player_gw = data["historical_player_gw_stats"]
    current_teams = data["current_teams"]

    staging_counts = {
        table_name: count_rows(table_name, season)
        for table_name in STAGING_TABLES
    }
    canonical_counts = {
        table_name: count_rows(table_name, season)
        for table_name in CANONICAL_SEASON_TABLES
        if table_exists(table_name)
    }

    if (staging_counts.get("historical_teams") or 0) == 0:
        errors.append("No historical_teams rows found for season=%s." % season)
    if (staging_counts.get("historical_fixtures") or 0) == 0:
        errors.append("No historical_fixtures rows found for season=%s." % season)
    if (staging_counts.get("historical_player_gw_stats") or 0) == 0:
        errors.append("No historical_player_gw_stats rows found for season=%s." % season)

    if not allow_canonical_season_rows:
        for table_name in ["fixtures", "player_gw_stats"]:
            count = canonical_counts.get(table_name)
            if count is not None and count > 0:
                errors.append(
                    "Canonical table %s has %s rows for season=%s. "
                    "Day64C expects historical data to remain in staging only."
                    % (table_name, count, season)
                )

    duplicate_results = []
    for table_name, key_columns in DUPLICATE_CHECKS:
        duplicate_count = duplicate_group_count(table_name, season, key_columns)
        duplicate_results.append(
            {
                "table": table_name,
                "key_columns": list(key_columns),
                "duplicate_groups": duplicate_count,
            }
        )
        if duplicate_count and duplicate_count > 0:
            errors.append(
                "%s has %s duplicate groups for key %s."
                % (table_name, duplicate_count, list(key_columns))
            )

    null_check_results: Dict[str, Dict[str, int]] = {}
    for table_name, columns in CRITICAL_NULL_CHECKS.items():
        df = data.get(table_name, pd.DataFrame())
        null_check_results[table_name] = null_counts(df, columns)
        for column, count in null_check_results[table_name].items():
            if count == -1:
                errors.append("%s is missing expected column %s." % (table_name, column))
            elif count > 0:
                errors.append("%s.%s has %s null values." % (table_name, column, count))

    gw_results = {
        "historical_fixtures": gw_coverage("historical_fixtures", season),
        "historical_player_gw_stats": gw_coverage("historical_player_gw_stats", season),
    }
    for table_name, coverage in gw_results.items():
        if coverage["distinct_gw_count"] != EXPECTED_FULL_EPL["gameweeks"]:
            warnings.append(
                "%s has %s distinct GWs, expected %s for a full EPL season."
                % (table_name, coverage["distinct_gw_count"], EXPECTED_FULL_EPL["gameweeks"])
            )
        if coverage["min_gw"] != 1 or coverage["max_gw"] != 38:
            warnings.append(
                "%s GW range is %s-%s, expected 1-38."
                % (table_name, coverage["min_gw"], coverage["max_gw"])
            )

    team_ids = normalized_id_set(teams["raw_team_id"]) if "raw_team_id" in teams.columns else set()
    player_ids = normalized_id_set(players["raw_player_id"]) if "raw_player_id" in players.columns else set()

    fixture_team_ids: Set[str] = set()
    if not fixtures.empty:
        fixture_team_ids.update(normalized_id_set(fixtures["raw_home_team_id"]))
        fixture_team_ids.update(normalized_id_set(fixtures["raw_away_team_id"]))

    player_team_ids = normalized_id_set(players["raw_team_id"]) if "raw_team_id" in players.columns else set()

    stats_player_ids = normalized_id_set(player_gw["raw_player_id"]) if "raw_player_id" in player_gw.columns else set()
    opponent_team_ids = normalized_id_set(player_gw["raw_opponent_team_id"]) if "raw_opponent_team_id" in player_gw.columns else set()

    reference_integrity = {
        "fixture_team_ids_missing_from_historical_teams": {
            "count": len(fixture_team_ids - team_ids),
            "sample": sorted_sample(fixture_team_ids - team_ids, sample_limit),
        },
        "player_team_ids_missing_from_historical_teams": {
            "count": len(player_team_ids - team_ids),
            "sample": sorted_sample(player_team_ids - team_ids, sample_limit),
        },
        "opponent_team_ids_missing_from_historical_teams": {
            "count": len(opponent_team_ids - team_ids),
            "sample": sorted_sample(opponent_team_ids - team_ids, sample_limit),
        },
        "stats_player_ids_missing_from_historical_players": {
            "count": len(stats_player_ids - player_ids),
            "sample": sorted_sample(stats_player_ids - player_ids, sample_limit),
        },
        "historical_players_without_gw_stats": {
            "count": len(player_ids - stats_player_ids),
            "sample": sorted_sample(player_ids - stats_player_ids, sample_limit),
        },
    }

    for label, item in reference_integrity.items():
        if label == "historical_players_without_gw_stats":
            if item["count"] > 0:
                warnings.append(
                    "%s historical players have no player-GW stats. This can be normal for inactive/squad-only players."
                    % item["count"]
                )
        else:
            if item["count"] > 0:
                errors.append("%s has count=%s." % (label, item["count"]))

    mapping = {
        "teams": mapping_counts(teams, "canonical_team_id"),
        "players": mapping_counts(players, "canonical_player_id"),
    }
    if mapping["teams"]["unmapped"] > 0:
        warnings.append(
            "%s historical teams are unmapped to canonical teams. This is allowed in staging."
            % mapping["teams"]["unmapped"]
        )
    if mapping["players"]["unmapped"] > 0:
        warnings.append(
            "%s historical players are unmapped to canonical players. This is allowed in staging."
            % mapping["players"]["unmapped"]
        )

    fixture_scores_null_count = 0
    if not fixtures.empty:
        fixture_scores_null_count = int(
            fixtures[["home_score", "away_score"]].isna().any(axis=1).sum()
            if "home_score" in fixtures.columns and "away_score" in fixtures.columns
            else 0
        )
        if fixture_scores_null_count > 0:
            warnings.append(
                "%s historical fixtures have at least one missing score."
                % fixture_scores_null_count
            )

    team_match_records = fixture_team_records(fixtures) if not fixtures.empty else pd.DataFrame()
    team_fixture_summary = {
        "team_matches_counted_twice": int(len(team_match_records)),
        "teams_with_38_matches": 0,
        "teams_not_38_matches": [],
        "team_total_points": None,
    }
    if not team_match_records.empty:
        matches_by_team = (
            team_match_records.groupby("raw_team_id", dropna=False)
            .size()
            .reset_index(name="matches")
        )
        teams_not_38 = matches_by_team[matches_by_team["matches"] != 38]
        team_fixture_summary["teams_with_38_matches"] = int((matches_by_team["matches"] == 38).sum())
        team_fixture_summary["teams_not_38_matches"] = teams_not_38.to_dict(orient="records")

        def points_for_row(row: pd.Series) -> int:
            goals_for = int(row["goals_for"])
            goals_against = int(row["goals_against"])
            if goals_for > goals_against:
                return 3
            if goals_for == goals_against:
                return 1
            return 0

        team_match_records["points"] = team_match_records.apply(points_for_row, axis=1)
        team_fixture_summary["team_total_points"] = int(team_match_records["points"].sum())

        if int(len(team_match_records)) != EXPECTED_FULL_EPL["team_matches_counted_twice"]:
            warnings.append(
                "Team-match record count is %s, expected %s."
                % (len(team_match_records), EXPECTED_FULL_EPL["team_matches_counted_twice"])
            )
        if len(teams_not_38) > 0:
            warnings.append("%s historical teams do not have exactly 38 completed matches." % len(teams_not_38))

    player_stats_summary = {
        "row_count": int(len(player_gw)),
        "distinct_player_count": int(len(stats_player_ids)),
        "total_minutes": int(pd.to_numeric(player_gw["minutes"], errors="coerce").fillna(0).sum()) if "minutes" in player_gw.columns else 0,
        "total_points": int(pd.to_numeric(player_gw["total_points"], errors="coerce").fillna(0).sum()) if "total_points" in player_gw.columns else 0,
        "negative_minutes_rows": int((pd.to_numeric(player_gw["minutes"], errors="coerce").fillna(0) < 0).sum()) if "minutes" in player_gw.columns else 0,
        "negative_total_points_rows": int((pd.to_numeric(player_gw["total_points"], errors="coerce").fillna(0) < 0).sum()) if "total_points" in player_gw.columns else 0,
    }
    if player_stats_summary["negative_minutes_rows"] > 0:
        errors.append("historical_player_gw_stats has negative minutes rows.")
    if player_stats_summary["negative_total_points_rows"] > 0:
        warnings.append("historical_player_gw_stats has negative total_points rows; this can happen in FPL but should be reviewed.")

    promotion_relegation = {
        "historical_teams_matched_current_by_short_name": [],
        "historical_teams_missing_in_current_by_short_name": [],
        "current_teams_missing_in_historical_by_short_name": [],
    }
    if not teams.empty and not current_teams.empty:
        hist_short_to_team = {}
        for _, row in teams.iterrows():
            short_name = row.get("raw_team_short_name")
            if not pd.isna(short_name):
                hist_short_to_team[str(short_name).strip().upper()] = {
                    "raw_team_id": normalize_raw_id(row.get("raw_team_id")),
                    "raw_team_name": row.get("raw_team_name"),
                    "raw_team_short_name": str(short_name).strip().upper(),
                }

        current_short_to_team = {}
        for _, row in current_teams.iterrows():
            short_name = row.get("short_name")
            if not pd.isna(short_name):
                current_short_to_team[str(short_name).strip().upper()] = {
                    "canonical_team_id": int(row.get("id")),
                    "canonical_team_name": row.get("name"),
                    "canonical_team_short_name": str(short_name).strip().upper(),
                }

        hist_shorts = set(hist_short_to_team.keys())
        current_shorts = set(current_short_to_team.keys())

        for short_name in sorted(hist_shorts & current_shorts):
            combined = {}
            combined.update(hist_short_to_team[short_name])
            combined.update(current_short_to_team[short_name])
            promotion_relegation["historical_teams_matched_current_by_short_name"].append(combined)

        for short_name in sorted(hist_shorts - current_shorts):
            promotion_relegation["historical_teams_missing_in_current_by_short_name"].append(hist_short_to_team[short_name])

        for short_name in sorted(current_shorts - hist_shorts):
            promotion_relegation["current_teams_missing_in_historical_by_short_name"].append(current_short_to_team[short_name])

        if promotion_relegation["historical_teams_missing_in_current_by_short_name"]:
            warnings.append(
                "%s historical teams are not in current teams table by short_name. This is expected with promotion/relegation."
                % len(promotion_relegation["historical_teams_missing_in_current_by_short_name"])
            )
        if promotion_relegation["current_teams_missing_in_historical_by_short_name"]:
            warnings.append(
                "%s current teams are not in historical season by short_name. This is expected with promotion/relegation."
                % len(promotion_relegation["current_teams_missing_in_historical_by_short_name"])
            )

    artifact_validation = validate_artifacts(
        prepared_dir=prepared_dir,
        staging_counts=staging_counts,
        player_stats_summary=player_stats_summary,
        team_fixture_summary=team_fixture_summary,
    )

    if artifact_validation.get("enabled"):
        player_artifact = artifact_validation.get("player_summary", {})
        team_artifact = artifact_validation.get("team_summary", {})
        if not player_artifact.get("exists"):
            warnings.append("player_season_summary.csv was not found in prepared dir.")
        if not team_artifact.get("exists"):
            warnings.append("team_season_summary.csv was not found in prepared dir.")

        for label, artifact in [("player_summary", player_artifact), ("team_summary", team_artifact)]:
            for key, value in artifact.items():
                if key.endswith("_matches_staging") or key.startswith("row_count_matches") or key.startswith("total_"):
                    if value is False:
                        errors.append("%s artifact validation failed for %s." % (label, key))

    canonical_import_blockers: List[str] = []
    if mapping["teams"]["unmapped"] > 0:
        canonical_import_blockers.append("unmapped historical teams")
    if mapping["players"]["unmapped"] > 0:
        canonical_import_blockers.append("unmapped historical players")
    for label, item in reference_integrity.items():
        if label != "historical_players_without_gw_stats" and item["count"] > 0:
            canonical_import_blockers.append(label)
    if errors:
        canonical_import_blockers.append("data quality errors")

    ready_for_staging_priors = (
        len(errors) == 0
        and int(staging_counts.get("historical_player_gw_stats") or 0) > 0
    )
    ready_for_canonical_import = (
        len(errors) == 0
        and len(canonical_import_blockers) == 0
    )

    report = {
        "created_at": utc_now(),
        "season": season,
        "passed": len(errors) == 0,
        "ready_for_staging_priors": ready_for_staging_priors,
        "ready_for_canonical_import": ready_for_canonical_import,
        "canonical_import_blockers": sorted(set(canonical_import_blockers)),
        "errors": errors,
        "warnings": warnings,
        "row_counts": {
            "staging": staging_counts,
            "canonical_for_same_season": canonical_counts,
        },
        "duplicate_checks": duplicate_results,
        "critical_null_checks": null_check_results,
        "gw_coverage": gw_results,
        "mapping": mapping,
        "reference_integrity": reference_integrity,
        "fixture_score_null_count": fixture_scores_null_count,
        "team_fixture_summary": team_fixture_summary,
        "player_stats_summary": player_stats_summary,
        "promotion_relegation_analysis": promotion_relegation,
        "artifact_validation": artifact_validation,
        "notes": [
            "This report is read-only.",
            "Unmapped historical teams and players are allowed in staging but block canonical import.",
            "Promotion/relegation differences are expected across EPL seasons.",
            "Day64C should keep canonical 2024_25 fixtures/player_gw_stats empty unless canonical import is explicitly planned.",
        ],
    }
    return report


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("saved_json:", path)


def write_markdown(report: Dict[str, Any], out_md: Optional[str]) -> None:
    if not out_md:
        return

    lines: List[str] = []
    lines.append("# Historical Data Quality Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- season: `%s`" % report["season"])
    lines.append("- passed: `%s`" % report["passed"])
    lines.append("- ready_for_staging_priors: `%s`" % report["ready_for_staging_priors"])
    lines.append("- ready_for_canonical_import: `%s`" % report["ready_for_canonical_import"])
    lines.append("")

    lines.append("## Row Counts")
    lines.append("")
    lines.append("### Staging")
    for table_name, count in sorted(report["row_counts"]["staging"].items()):
        lines.append("- `%s`: %s" % (table_name, count))
    lines.append("")
    lines.append("### Canonical For Same Season")
    for table_name, count in sorted(report["row_counts"]["canonical_for_same_season"].items()):
        lines.append("- `%s`: %s" % (table_name, count))
    lines.append("")

    lines.append("## Mapping")
    lines.append("")
    lines.append("- teams mapped: %s" % report["mapping"]["teams"]["mapped"])
    lines.append("- teams unmapped: %s" % report["mapping"]["teams"]["unmapped"])
    lines.append("- players mapped: %s" % report["mapping"]["players"]["mapped"])
    lines.append("- players unmapped: %s" % report["mapping"]["players"]["unmapped"])
    lines.append("")

    lines.append("## Promotion / Relegation Analysis")
    lines.append("")
    matched = report["promotion_relegation_analysis"]["historical_teams_matched_current_by_short_name"]
    missing_hist = report["promotion_relegation_analysis"]["historical_teams_missing_in_current_by_short_name"]
    missing_current = report["promotion_relegation_analysis"]["current_teams_missing_in_historical_by_short_name"]
    lines.append("- historical teams matched current by short_name: %s" % len(matched))
    lines.append("- historical teams missing in current by short_name: %s" % len(missing_hist))
    lines.append("- current teams missing in historical by short_name: %s" % len(missing_current))
    if missing_hist:
        lines.append("- historical-only teams: `%s`" % ", ".join([str(item["raw_team_name"]) for item in missing_hist]))
    if missing_current:
        lines.append("- current-only teams: `%s`" % ", ".join([str(item["canonical_team_name"]) for item in missing_current]))
    lines.append("")

    lines.append("## Data Quality")
    lines.append("")
    lines.append("### Errors")
    if report["errors"]:
        for error in report["errors"]:
            lines.append("- %s" % error)
    else:
        lines.append("- None")
    lines.append("")
    lines.append("### Warnings")
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append("- %s" % warning)
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Canonical Import Blockers")
    lines.append("")
    if report["canonical_import_blockers"]:
        for blocker in report["canonical_import_blockers"]:
            lines.append("- %s" % blocker)
    else:
        lines.append("- None")
    lines.append("")

    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved_md:", path)


def print_report(report: Dict[str, Any]) -> None:
    print("=== Historical Data Quality Report ===")
    print("season:", report["season"])
    print("passed:", report["passed"])
    print("ready_for_staging_priors:", report["ready_for_staging_priors"])
    print("ready_for_canonical_import:", report["ready_for_canonical_import"])
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

    print("Row counts - staging:")
    for table_name, count in sorted(report["row_counts"]["staging"].items()):
        print("- %s: %s" % (table_name, count))
    print()

    print("Row counts - canonical same season:")
    for table_name, count in sorted(report["row_counts"]["canonical_for_same_season"].items()):
        print("- %s: %s" % (table_name, count))
    print()

    print("Mapping:")
    print("- teams:", report["mapping"]["teams"])
    print("- players:", report["mapping"]["players"])
    print()

    print("GW coverage:")
    for table_name, coverage in sorted(report["gw_coverage"].items()):
        print("- %s: %s" % (table_name, coverage))
    print()

    print("Reference integrity:")
    for label, item in sorted(report["reference_integrity"].items()):
        print("- %s: count=%s sample=%s" % (label, item["count"], item["sample"]))
    print()

    print("Promotion/relegation:")
    promo = report["promotion_relegation_analysis"]
    print("- matched historical/current teams:", len(promo["historical_teams_matched_current_by_short_name"]))
    print("- historical teams missing in current:", promo["historical_teams_missing_in_current_by_short_name"])
    print("- current teams missing in historical:", promo["current_teams_missing_in_historical_by_short_name"])
    print()

    print("Canonical import blockers:")
    if report["canonical_import_blockers"]:
        for blocker in report["canonical_import_blockers"]:
            print("-", blocker)
    else:
        print("- None")
    print()


def main() -> None:
    args = parse_args()
    report = build_report(
        season=args.season,
        prepared_dir=args.prepared_dir,
        allow_canonical_season_rows=args.allow_canonical_season_rows,
        sample_limit=args.sample_limit,
    )
    print_report(report)
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)

    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
