from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import text


REFRESH_VERSION = "day76d_v1"
ARTIFACT_TYPE = "pre_gw1_player_prediction_refresh"
PREDICTION_SOURCE = "pre_gw1_heuristic_preview"
PREDICTION_MODE = "pre_gw1_prior"
DEFAULT_ARTIFACT_ROOT = "/private/tmp/fpl-artifacts"

REQUIRED_LIMITATION_LABELS: Dict[str, Any] = {
    "prediction_source": PREDICTION_SOURCE,
    "production_approved": False,
    "historical_multi_season_backtest_complete": False,
    "component_model_stack_complete": False,
}

ROLLOVER_FILENAMES = {
    "current_player_pool": "current_player_pool.csv",
    "player_identity_mapping": "player_identity_mapping.csv",
    "team_identity_mapping": "team_identity_mapping.csv",
    "fixtures": "gw1_gw5_fixtures.csv",
}


class Day76DInputError(RuntimeError):
    """Raised when a refresh input cannot satisfy the read-only contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: Any, label: str) -> datetime:
    if value is None:
        raise Day76DInputError("%s is required." % label)
    normalized = str(value).strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise Day76DInputError("%s must be valid ISO-8601." % label) from exc
    if parsed.tzinfo is None:
        raise Day76DInputError("%s must include a timezone offset." % label)
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise Day76DInputError("JSON file does not exist: %s" % path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Day76DInputError("JSON root must be an object: %s" % path)
    return value


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise Day76DInputError("%s CSV does not exist: %s" % (label, path))
    return pd.read_csv(path, low_memory=False)


def required_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise Day76DInputError(
            "%s is missing required columns: %s" % (label, ", ".join(missing))
        )


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def nullable_int(value: Any) -> Optional[int]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def validate_rollover_report(
    report: Mapping[str, Any],
    source_season: Optional[str] = None,
    target_season: Optional[str] = None,
    target_gw: Optional[int] = None,
) -> Dict[str, Any]:
    blockers: List[str] = []
    actual_source = str(report.get("source_season") or "")
    actual_target = str(report.get("target_season") or "")
    actual_gw = nullable_int(report.get("target_gw"))

    if report.get("passed") is not True:
        blockers.append("Day76C rollover report must have passed=true.")
    if report.get("audit_only") is not True:
        blockers.append("Day76C rollover report must have audit_only=true.")
    if report.get("writes_database") is not False:
        blockers.append("Day76C rollover report must have writes_database=false.")
    if report.get("blockers"):
        blockers.append("Day76C rollover report contains blockers.")

    readiness = report.get("readiness") or {}
    if not isinstance(readiness, Mapping):
        blockers.append("Day76C readiness must be a mapping.")
        readiness = {}
    for key in (
        "current_player_pool_validated",
        "focused_player_identity_coverage_validated",
        "gw1_gw5_fixture_scope_validated",
        "target_season_rule_versions_validated",
        "target_team_transition_validated",
    ):
        if readiness.get(key) is not True:
            blockers.append("Day76C readiness.%s must be true." % key)
    if readiness.get("ready_for_prediction_write") is not False:
        blockers.append("Day76C ready_for_prediction_write must remain false.")

    if source_season is not None and actual_source != source_season:
        blockers.append("source_season does not match Day76C rollover report.")
    if target_season is not None and actual_target != target_season:
        blockers.append("target_season does not match Day76C rollover report.")
    if target_gw is not None and actual_gw != int(target_gw):
        blockers.append("target_gw does not match Day76C rollover report.")
    if actual_gw != 1:
        blockers.append("Day76D Fast Lane requires target_gw=1.")

    try:
        as_of = format_utc(parse_utc(report.get("as_of_time_utc"), "as_of_time_utc"))
    except Day76DInputError as exc:
        blockers.append(str(exc))
        as_of = ""

    if blockers:
        raise Day76DInputError(" ".join(blockers))

    return {
        "source_season": actual_source,
        "target_season": actual_target,
        "target_gw": actual_gw,
        "as_of_time_utc": as_of,
    }


def resolve_rollover_inputs(report_path: Path) -> Dict[str, Path]:
    run_dir = report_path.resolve().parent
    result = {"rollover_report": report_path.resolve()}
    for key, filename in ROLLOVER_FILENAMES.items():
        path = run_dir / filename
        if not path.is_file():
            raise Day76DInputError("Missing Day76C artifact: %s" % path)
        result[key] = path
    return result


def adapt_current_player_pool(current_players: pd.DataFrame) -> pd.DataFrame:
    required_columns(
        current_players,
        (
            "target_player_id",
            "target_player_code",
            "target_player_name",
            "target_web_name",
            "target_team_id",
            "target_position",
            "target_price",
            "target_status",
            "current_selection_eligible",
        ),
        "Day76C current player pool",
    )
    result = pd.DataFrame(
        {
            "player_id": current_players["target_player_id"].apply(nullable_int),
            "fpl_player_id": current_players["target_player_id"].apply(nullable_int),
            "player_name": current_players["target_player_name"],
            "web_name": current_players["target_web_name"],
            "team_id": current_players["target_team_id"].apply(nullable_int),
            "position": current_players["target_position"].astype(str).str.upper(),
            "price": pd.to_numeric(current_players["target_price"], errors="coerce"),
            "status": current_players["target_status"].fillna("").astype(str),
            "current_selection_eligible": current_players[
                "current_selection_eligible"
            ].apply(bool_value),
        }
    )
    if result["player_id"].isna().any() or result["team_id"].isna().any():
        raise Day76DInputError("Current player pool contains missing player/team IDs.")
    if result["player_id"].duplicated().any():
        raise Day76DInputError("Current player pool contains duplicate player IDs.")
    return result


def adapt_target_teams(current_players: pd.DataFrame) -> pd.DataFrame:
    required_columns(
        current_players,
        (
            "target_team_id",
            "target_team_name",
            "target_team_short_name",
        ),
        "Day76C current player pool",
    )
    result = current_players[
        ["target_team_id", "target_team_name", "target_team_short_name"]
    ].drop_duplicates().copy()
    result = result.rename(
        columns={
            "target_team_id": "team_id",
            "target_team_name": "team_name",
            "target_team_short_name": "team_short_name",
        }
    )
    result["team_id"] = result["team_id"].apply(nullable_int)
    result["fpl_team_id"] = result["team_id"]
    result = result[
        ["team_id", "fpl_team_id", "team_name", "team_short_name"]
    ].sort_values("team_id").reset_index(drop=True)
    if len(result) != 20:
        raise Day76DInputError("Target team pool must contain exactly 20 teams.")
    if result["team_id"].isna().any() or result["team_id"].duplicated().any():
        raise Day76DInputError("Target team pool contains invalid team IDs.")
    return result


def adapt_target_fixtures(fixtures: pd.DataFrame, target_gw: int) -> pd.DataFrame:
    required_columns(
        fixtures,
        (
            "target_season",
            "gameweek",
            "fixture_id",
            "kickoff_time_utc",
            "home_team_id",
            "away_team_id",
            "started",
            "finished",
        ),
        "Day76C fixtures",
    )
    working = fixtures[
        pd.to_numeric(fixtures["gameweek"], errors="coerce") == int(target_gw)
    ].copy()
    if len(working) != 10:
        raise Day76DInputError("GW%s must contain exactly 10 fixtures." % target_gw)
    result = pd.DataFrame(
        {
            "id": working["fixture_id"].apply(nullable_int),
            "fixture_id": working["fixture_id"].apply(nullable_int),
            "fpl_fixture_id": working["fixture_id"].apply(nullable_int),
            "home_team_id": working["home_team_id"].apply(nullable_int),
            "away_team_id": working["away_team_id"].apply(nullable_int),
            "kickoff_time": working["kickoff_time_utc"],
            "finished": working["finished"].apply(bool_value),
            "home_score": None,
            "away_score": None,
            "gw": int(target_gw),
            "season": working["target_season"].astype(str),
        }
    )
    if result["id"].isna().any() or result["id"].duplicated().any():
        raise Day76DInputError("GW fixture IDs must be present and unique.")
    return result


def adapt_player_identity_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    required_columns(
        mapping,
        (
            "source_season",
            "target_season",
            "source_player_id",
            "source_player_name",
            "target_player_id",
            "target_player_name",
            "mapping_status",
            "mapping_method",
            "mapping_reason",
            "historical_prior_eligible",
        ),
        "Day76C player identity mapping",
    )
    rows: List[Dict[str, Any]] = []
    for row in mapping.to_dict(orient="records"):
        target_id = nullable_int(row.get("target_player_id"))
        source_id = nullable_int(row.get("source_player_id"))
        eligible = bool_value(row.get("historical_prior_eligible"))
        safe = bool(eligible and source_id is not None and target_id is not None)
        raw_id: Any = source_id
        raw_name = row.get("source_player_name")
        if raw_id is None:
            raw_id = "unmapped_target_%s" % target_id
            raw_name = row.get("target_player_name")
        rows.append(
            {
                "source_season": row.get("source_season"),
                "target_season": row.get("target_season"),
                "raw_player_id": raw_id,
                "raw_player_name": raw_name,
                "candidate_player_id": target_id,
                "candidate_rank": 1,
                "candidate_score": 1.0 if safe else 0.0,
                "match_status": (
                    "auto_approved_player_candidate"
                    if safe
                    else "target_only_unmatched"
                ),
                "is_auto_approved": safe,
                "needs_manual_review": not safe,
                "is_ambiguous": False,
                "mapping_confidence": 1.0 if safe else 0.0,
                "mapping_reason": row.get("mapping_reason"),
                "day76c_mapping_status": row.get("mapping_status"),
                "day76c_mapping_method": row.get("mapping_method"),
                "historical_prior_eligible": eligible,
            }
        )
    result = pd.DataFrame(rows)
    if result["candidate_player_id"].isna().any():
        raise Day76DInputError("Player mapping contains missing target player IDs.")
    if result["candidate_player_id"].duplicated().any():
        raise Day76DInputError("Player mapping contains duplicate target player IDs.")
    return result


def adapt_player_priors(priors: pd.DataFrame) -> pd.DataFrame:
    required_columns(
        priors,
        (
            "source_season",
            "target_season",
            "player_id",
            "web_name",
            "prev_season_minutes",
            "prev_season_appearances",
            "prev_season_starts_proxy",
            "prev_season_total_points",
            "prev_season_points_per90",
            "prev_season_goals",
            "prev_season_assists",
            "prev_season_clean_sheets",
            "prev_season_bonus",
        ),
        "Canonical player priors",
    )
    result = priors.copy()
    result["raw_player_id"] = result["player_id"].apply(nullable_int)
    result["raw_player_name"] = result["web_name"]
    result["prior_identity_scope"] = "canonical_player_id"
    result["prior_source"] = "canonical_source_season_player_gw_stats"
    return result


def adapt_team_identity_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    required_columns(
        mapping,
        (
            "source_season",
            "target_season",
            "source_team_id",
            "source_team_name",
            "source_team_short_name",
            "target_team_id",
            "target_team_name",
            "target_team_short_name",
            "mapping_status",
            "historical_prior_eligible",
        ),
        "Day76C team identity mapping",
    )
    rows: List[Dict[str, Any]] = []
    for row in mapping.to_dict(orient="records"):
        source_id = nullable_int(row.get("source_team_id"))
        target_id = nullable_int(row.get("target_team_id"))
        safe = bool(
            bool_value(row.get("historical_prior_eligible"))
            and source_id is not None
            and target_id is not None
        )
        if source_id is None:
            raw_id: Any = "target_only_%s" % target_id
            raw_name = row.get("target_team_name")
            raw_short = row.get("target_team_short_name")
            status = "target_only_unmatched"
        else:
            raw_id = source_id
            raw_name = row.get("source_team_name")
            raw_short = row.get("source_team_short_name")
            status = (
                "auto_approved_team_candidate"
                if safe
                else "historical_only_unmatched"
            )
        rows.append(
            {
                "source_season": row.get("source_season"),
                "target_season": row.get("target_season"),
                "raw_team_id": raw_id,
                "raw_team_name": raw_name,
                "raw_team_short_name": raw_short,
                "candidate_rank": 1,
                "candidate_team_id": target_id,
                "candidate_team_name": row.get("target_team_name"),
                "candidate_team_short_name": row.get("target_team_short_name"),
                "candidate_score": 1.0 if safe else 0.0,
                "match_status": status,
                "is_auto_approved": safe,
                "needs_manual_review": not safe,
                "safe_team_match_for_auto_approval": safe,
                "match_reason": row.get("mapping_status"),
                "historical_prior_eligible": bool_value(
                    row.get("historical_prior_eligible")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_canonical_team_priors(
    source_season: str,
    target_season: str,
) -> pd.DataFrame:
    from app.core.db import SessionLocal

    sql = """
        WITH team_match AS (
            SELECT
                f.home_team_id AS team_id,
                TRUE AS is_home,
                f.home_score AS goals_for,
                f.away_score AS goals_against
            FROM fixtures AS f
            WHERE f.season = :source_season
              AND f.finished = TRUE
              AND f.home_score IS NOT NULL
              AND f.away_score IS NOT NULL

            UNION ALL

            SELECT
                f.away_team_id AS team_id,
                FALSE AS is_home,
                f.away_score AS goals_for,
                f.home_score AS goals_against
            FROM fixtures AS f
            WHERE f.season = :source_season
              AND f.finished = TRUE
              AND f.home_score IS NOT NULL
              AND f.away_score IS NOT NULL
        )
        SELECT
            t.id AS raw_team_id,
            t.name AS raw_team_name,
            t.short_name AS raw_team_short_name,
            COUNT(*) AS prev_season_matches,
            SUM(CASE WHEN tm.goals_for > tm.goals_against THEN 1 ELSE 0 END)
                AS prev_season_wins,
            SUM(CASE WHEN tm.goals_for = tm.goals_against THEN 1 ELSE 0 END)
                AS prev_season_draws,
            SUM(CASE WHEN tm.goals_for < tm.goals_against THEN 1 ELSE 0 END)
                AS prev_season_losses,
            SUM(tm.goals_for) AS prev_season_goals_for,
            SUM(tm.goals_against) AS prev_season_goals_against,
            SUM(tm.goals_for - tm.goals_against) AS prev_season_goal_difference,
            SUM(CASE WHEN tm.goals_against = 0 THEN 1 ELSE 0 END)
                AS prev_season_clean_sheets,
            SUM(
                CASE
                    WHEN tm.goals_for > tm.goals_against THEN 3
                    WHEN tm.goals_for = tm.goals_against THEN 1
                    ELSE 0
                END
            ) AS prev_season_total_points,
            SUM(
                CASE
                    WHEN tm.is_home AND tm.goals_for > tm.goals_against THEN 3
                    WHEN tm.is_home AND tm.goals_for = tm.goals_against THEN 1
                    ELSE 0
                END
            ) AS prev_season_home_points,
            SUM(
                CASE
                    WHEN NOT tm.is_home AND tm.goals_for > tm.goals_against THEN 3
                    WHEN NOT tm.is_home AND tm.goals_for = tm.goals_against THEN 1
                    ELSE 0
                END
            ) AS prev_season_away_points,
            SUM(CASE WHEN tm.is_home THEN tm.goals_for ELSE 0 END)
                AS prev_season_home_goals_for,
            SUM(CASE WHEN NOT tm.is_home THEN tm.goals_for ELSE 0 END)
                AS prev_season_away_goals_for,
            SUM(CASE WHEN tm.is_home THEN tm.goals_against ELSE 0 END)
                AS prev_season_home_goals_against,
            SUM(CASE WHEN NOT tm.is_home THEN tm.goals_against ELSE 0 END)
                AS prev_season_away_goals_against,
            SUM(CASE WHEN tm.is_home THEN 1 ELSE 0 END) AS prev_season_home_matches,
            SUM(CASE WHEN NOT tm.is_home THEN 1 ELSE 0 END) AS prev_season_away_matches
        FROM team_match AS tm
        JOIN teams AS t ON t.id = tm.team_id
        GROUP BY t.id, t.name, t.short_name
        ORDER BY t.id
    """
    session = SessionLocal()
    try:
        rows = session.execute(
            text(sql),
            {"source_season": source_season},
        ).mappings().all()
    finally:
        session.close()
    if len(rows) != 20:
        raise Day76DInputError(
            "Canonical source season must produce 20 team priors; got %s." % len(rows)
        )
    result = pd.DataFrame([dict(row) for row in rows])
    result.insert(0, "target_season", target_season)
    result.insert(0, "source_season", source_season)
    result["prior_identity_scope"] = "canonical_team_id"
    result["prior_source"] = "canonical_source_season_finished_fixtures"
    result["has_prev_season_data"] = True
    result["is_prev_season_active"] = True
    matches = pd.to_numeric(result["prev_season_matches"], errors="coerce")
    result["prev_season_scheduled_fixtures"] = matches
    result["prev_season_points_per_match"] = (
        pd.to_numeric(result["prev_season_total_points"], errors="coerce") / matches
    )
    result["prev_season_goals_for_per_match"] = (
        pd.to_numeric(result["prev_season_goals_for"], errors="coerce") / matches
    )
    result["prev_season_goals_against_per_match"] = (
        pd.to_numeric(result["prev_season_goals_against"], errors="coerce") / matches
    )
    result["prev_season_clean_sheet_rate"] = (
        pd.to_numeric(result["prev_season_clean_sheets"], errors="coerce") / matches
    )
    return result


def build_canonical_player_priors(
    source_season: str,
    target_season: str,
) -> pd.DataFrame:
    from ml.features.build_previous_season_player_priors import _build_prior_rows

    rows = _build_prior_rows(source_season, target_season)
    if not rows:
        raise Day76DInputError(
            "No canonical player priors were produced for source_season=%s."
            % source_season
        )
    return pd.DataFrame(rows)



def build_artifact_first_player_feature_report(
    args: argparse.Namespace,
    mode_result: Mapping[str, Any],
    features: pd.DataFrame,
    build_summary: Mapping[str, Any],
    player_priors: pd.DataFrame,
    player_mapping: pd.DataFrame,
    match_features: pd.DataFrame,
    target_player_count: int,
    target_team_count: int,
    target_fixture_count: int,
    feature_version: str,
    feature_scope: str,
) -> Dict[str, Any]:
    """Build the Day71A-compatible report without consulting target-season DB rows."""
    blockers: List[str] = list(mode_result.get("errors") or [])
    warnings: List[str] = list(mode_result.get("warnings") or [])
    resolved_mode = str(mode_result.get("resolved_prediction_mode") or "")
    if resolved_mode != PREDICTION_MODE:
        blockers.append(
            "Artifact-first Day71A requires resolved_prediction_mode=%s."
            % PREDICTION_MODE
        )

    feature_rows = int(len(features))
    duplicate_player_id_count = int(
        features["player_id"].duplicated(keep=False).sum()
    )
    players_with_prior = int(
        features["has_prev_season_player_prior"].apply(bool_value).sum()
    )
    players_without_prior = int(features["no_prior_flag"].apply(bool_value).sum())
    players_with_fixture = int(features["has_fixture"].apply(bool_value).sum())
    players_without_fixture = int(features["blank_gw_flag"].apply(bool_value).sum())
    players_on_promoted_teams = int(
        features["promoted_team_player_flag"].apply(bool_value).sum()
    )
    missing_team_context_count = int(
        features["missing_team_context_flag"].apply(bool_value).sum()
    )
    missing_fixture_context_count = int(
        features["missing_fixture_context_flag"].apply(bool_value).sum()
    )
    prediction_write_allowed_true_count = int(
        features["prediction_write_allowed"].apply(bool_value).sum()
    )
    production_ready_true_count = int(
        features["production_ready"].apply(bool_value).sum()
    )
    requires_manifest_false_count = int(
        (~features["requires_player_feature_manifest_before_prediction"].apply(bool_value)).sum()
    )
    prior_without_safe_mapping_count = int(
        (
            features["has_prev_season_player_prior"].apply(bool_value)
            & (features["player_mapping_status"] == "no_safe_accepted_mapping")
        ).sum()
    )

    mapping_summary = dict(build_summary["mapping_summary"])
    fixture_build_summary = dict(build_summary["fixture_build_summary"])
    expected_fallback_count = int(
        (~player_mapping["is_auto_approved"].apply(bool_value)).sum()
    )

    if target_player_count <= 0:
        blockers.append("Day76C target player pool is empty.")
    if target_team_count != 20:
        blockers.append("Day76C target team pool must contain exactly 20 teams.")
    if target_fixture_count != 10:
        blockers.append("Day76C GW1 fixture scope must contain exactly 10 fixtures.")
    if feature_rows != target_player_count:
        blockers.append(
            "Feature rows (%s) do not equal Day76C target player rows (%s)."
            % (feature_rows, target_player_count)
        )
    if duplicate_player_id_count:
        blockers.append("Player feature artifact contains duplicate player IDs.")
    if mapping_summary.get("accepted_mapping_rows", 0) <= 0:
        blockers.append("No safe accepted Day76C player mappings were loaded.")
    if mapping_summary.get("duplicate_accepted_candidate_player_id_count") != 0:
        blockers.append("Accepted mappings contain duplicate target player IDs.")
    if mapping_summary.get("duplicate_accepted_raw_player_id_count") != 0:
        blockers.append("Accepted mappings contain duplicate source player IDs.")
    if players_with_prior <= 0:
        blockers.append("No target player received a safe previous-season prior.")
    if players_without_prior != expected_fallback_count:
        blockers.append(
            "Visible no-prior rows (%s) do not match unsafe/unmapped Day76C rows (%s)."
            % (players_without_prior, expected_fallback_count)
        )
    if prior_without_safe_mapping_count:
        blockers.append("A player received a prior without a safe Day76C mapping.")
    if fixture_build_summary.get("duplicate_team_fixture_context_rows") != 0:
        blockers.append("A target team has duplicate GW1 fixture context.")
    if prediction_write_allowed_true_count:
        blockers.append("prediction_write_allowed must be false for every row.")
    if production_ready_true_count:
        blockers.append("production_ready must be false for every row.")
    if requires_manifest_false_count:
        blockers.append("Every player feature row must require a prediction manifest.")

    if "prev_season_goals_conceded" not in player_priors.columns:
        warnings.append(
            "Canonical player priors do not expose prev_season_goals_conceded; "
            "the Day71A field remains null."
        )
    if players_without_fixture:
        warnings.append(
            "%s target players have no GW%s fixture context."
            % (players_without_fixture, args.target_gw)
        )
    if missing_team_context_count:
        warnings.append(
            "%s player rows have incomplete effective team/opponent context."
            % missing_team_context_count
        )
    if missing_fixture_context_count:
        warnings.append(
            "%s player rows have incomplete fixture context."
            % missing_fixture_context_count
        )

    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    passed = not blockers
    return {
        "created_at": utc_now(),
        "source_season": args.source_season,
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": resolved_mode,
        "feature_version": feature_version,
        "feature_scope": feature_scope,
        "input_mode": "day76c_artifact_first",
        "audit_only": True,
        "writes_database": False,
        "uses_current_season_actual_player_gw_stats": False,
        "passed": passed,
        "ready_for_pre_gw1_player_features": passed,
        "ready_for_pre_gw1_player_prediction_preview": False,
        "ready_for_prediction_write": False,
        "inputs": {
            "player_prior_csv": str(Path(args.player_prior_csv)),
            "player_mapping_csv": str(Path(args.player_mapping_csv)),
            "match_features_csv": str(Path(args.match_features_csv)),
            "player_prior_rows": int(len(player_priors)),
            "player_mapping_rows": int(len(player_mapping)),
            "match_feature_rows": int(len(match_features)),
        },
        "outputs": {
            "out_csv": str(Path(args.out_csv)),
            "out_json": str(Path(args.out_json)),
            "out_md": str(Path(args.out_md)),
        },
        "row_counts": {
            "target_player_rows": int(target_player_count),
            "target_team_rows": int(target_team_count),
            "target_fixture_rows": int(target_fixture_count),
            "feature_rows": feature_rows,
            "duplicate_player_id_count": duplicate_player_id_count,
            "players_with_prior": players_with_prior,
            "players_without_prior": players_without_prior,
            "players_on_promoted_teams": players_on_promoted_teams,
        },
        "mapping_summary": mapping_summary,
        "fixture_summary": {
            **fixture_build_summary,
            "players_with_fixture": players_with_fixture,
            "players_without_fixture": players_without_fixture,
        },
        "team_context_summary": {
            "missing_team_context_count": missing_team_context_count,
            "team_fallback_player_rows": int(
                features["team_fallback_applied"].apply(bool_value).sum()
            ),
            "opponent_fallback_player_rows": int(
                features["opponent_fallback_applied"].apply(bool_value).sum()
            ),
        },
        "safety_summary": {
            "uses_current_season_actual_player_gw_stats": False,
            "prior_without_safe_mapping_count": prior_without_safe_mapping_count,
            "prediction_write_allowed_true_count": prediction_write_allowed_true_count,
            "production_ready_true_count": production_ready_true_count,
            "requires_manifest_false_count": requires_manifest_false_count,
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "Target players, teams, and fixtures come from Day76C immutable artifacts.",
            "Canonical source-season facts are read only to build previous-season priors.",
            "Only historical_prior_eligible Day76C mappings can inherit a prior.",
            "Unmapped players remain visible with no_prior_flag=True.",
            "No prediction or other database table is written.",
        ],
    }

def run_module(
    backend_root: Path,
    module: str,
    arguments: Sequence[str],
) -> None:
    command = [sys.executable, "-m", module] + list(arguments)
    completed = subprocess.run(
        command,
        cwd=str(backend_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Module failed: %s\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (" ".join(command), completed.stdout, completed.stderr)
        )


def write_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def build_limitation_contract(
    scoring_rules_version: str,
    bps_rules_version: Optional[str],
    bonus_prior_missing: bool,
) -> Dict[str, Any]:
    unresolved: List[Dict[str, Any]] = [
        {
            "component": "bonus_points_system",
            "status": "unresolved_event_level_reconstruction",
            "rules_version": bps_rules_version,
            "reason": (
                "The heuristic consumes awarded/estimated bonus and does not "
                "reconstruct every 2026/27 BPS event."
            ),
        },
        {
            "component": "expected_other_points",
            "status": "residual_component",
            "reason": (
                "Cards, saves, penalties, own goals, goals conceded, defensive "
                "contributions, and other sparse events are not all separate models."
            ),
        },
    ]
    if bonus_prior_missing:
        unresolved.append(
            {
                "component": "prev_season_bonus",
                "status": "source_column_unavailable",
                "reason": (
                    "Canonical 2025/26 player_gw_stats did not expose usable bonus "
                    "values for the previous-season prior artifact."
                ),
            }
        )
    return {
        **REQUIRED_LIMITATION_LABELS,
        "scoring_rules_version": scoring_rules_version,
        "unresolved_point_components": unresolved,
        "recommendation_status": "preview_only",
        "prediction_write_allowed": False,
        "writes_database": False,
    }


def normalize_contract_input_row(
    row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Convert pandas missing scalars into contract-safe Python values."""
    normalized: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (str, bytes, list, tuple, dict, set)):
            normalized[key] = value
            continue
        try:
            if bool(pd.isna(value)):
                normalized[key] = None
                continue
        except (TypeError, ValueError):
            pass
        normalized[key] = value

    # Empty CSV cells are read by pandas as float NaN. The Day76B contract
    # intentionally accepts risk flags only as a sequence or comma-separated
    # string, so represent an empty cell as an empty sequence rather than None.
    if normalized.get("risk_flags") is None:
        normalized["risk_flags"] = ()
    return normalized


def adapt_standard_predictions(
    preview: pd.DataFrame,
    row_adapter: Callable[[Mapping[str, Any]], Any],
    selection_eligibility: Optional[Mapping[int, bool]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in preview.to_dict(orient="records"):
        normalized_raw = normalize_contract_input_row(raw)
        contract_row = row_adapter(normalized_raw)
        optimizer_row = contract_row.to_optimizer_row()
        player_id = nullable_int(optimizer_row.get("player_id"))
        if selection_eligibility is not None and player_id is not None:
            current_eligible = bool(selection_eligibility.get(player_id, False))
            if not current_eligible:
                optimizer_row["selection_eligible"] = False
                optimizer_row[
                    "eligibility_reason"
                ] = "day76c_current_selection_ineligible"
        optimizer_row.update(REQUIRED_LIMITATION_LABELS)
        optimizer_row["prediction_write_allowed"] = False
        rows.append(optimizer_row)
    if len(rows) != len(preview):
        raise RuntimeError("Standard contract adaptation changed the row count.")
    return rows


def dataframe_to_csv_text(dataframe: pd.DataFrame) -> str:
    return dataframe.to_csv(index=False, lineterminator="\n")


def rows_to_csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    dataframe = pd.DataFrame(list(rows))
    return dataframe_to_csv_text(dataframe)


def build_markdown_report(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    limitations = report["limitations"]
    lines = [
        "# Day76D — 2026/27 Pre-GW1 Prediction Refresh",
        "",
        "- Created at: `%s`" % report["created_at_utc"],
        "- Run ID: `%s`" % report["run_metadata"]["run_id"],
        "- Source season: `%s`" % report["source_season"],
        "- Target season: `%s`" % report["target_season"],
        "- Target GW: `%s`" % report["target_gw"],
        "- Prediction source: `%s`" % limitations["prediction_source"],
        "- Scoring rules: `%s`" % limitations["scoring_rules_version"],
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `True`",
        "- Writes database: `False`",
        "- Prediction writes enabled: `False`",
        "",
        "## Validation",
        "",
        "- Target players retained: `%s`" % validation["target_player_count"],
        "- Safe historical mappings: `%s`" % validation["safe_mapping_count"],
        "- Visible fallback players: `%s`" % validation["fallback_player_count"],
        "- Standard contract rows: `%s`" % validation["standard_contract_row_count"],
        "- Standard selection-eligible rows: `%s`"
        % validation["standard_selection_eligible_count"],
        "- Immutable artifacts: `%s`" % validation["immutable_artifact_count"],
        "",
        "## Required limitation labels",
        "",
        "```text",
        "prediction_source: %s" % limitations["prediction_source"],
        "production_approved: false",
        "historical_multi_season_backtest_complete: false",
        "component_model_stack_complete: false",
        "```",
        "",
        "## Unresolved point components",
        "",
    ]
    for item in limitations["unresolved_point_components"]:
        lines.append("- `%s`: %s" % (item["component"], item["reason"]))
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend("- %s" % value for value in report["blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend("- %s" % value for value in report["warnings"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Stop point",
            "",
            "> A current, target-season, immutable player prediction preview exists "
            "for the 2026/27 GW1 player pool.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the 2026/27 GW1 player prediction preview from Day76C "
            "immutable artifacts and canonical 2025/26 history. Read-only."
        )
    )
    parser.add_argument("--rollover-report-json", required=True)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--source-season", default="")
    parser.add_argument("--target-season", default="")
    parser.add_argument("--target-gw", type=int, default=1)
    parser.add_argument("--as-of-time", default="")
    parser.add_argument("--scoring-rules-version", default="")
    parser.add_argument("--squad-transfer-rules-version", default="")
    parser.add_argument("--chip-rules-version", default="")
    parser.add_argument("--bps-rules-version", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--keep-work-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_root = Path(__file__).resolve().parents[2]
    rollover_path = Path(args.rollover_report_json).expanduser().resolve()
    rollover_report = load_json(rollover_path)

    requested_source = args.source_season or None
    requested_target = args.target_season or None
    validated = validate_rollover_report(
        rollover_report,
        source_season=requested_source,
        target_season=requested_target,
        target_gw=args.target_gw,
    )
    source_season = requested_source or validated["source_season"]
    target_season = requested_target or validated["target_season"]
    target_gw = int(validated["target_gw"])
    as_of_time = format_utc(
        parse_utc(args.as_of_time or validated["as_of_time_utc"], "as_of_time")
    )

    rule_versions = (
        (rollover_report.get("target_season_rules") or {}).get("actual_versions")
        or {}
    )
    scoring_rules_version = (
        args.scoring_rules_version or rule_versions.get("scoring") or ""
    )
    squad_transfer_rules_version = (
        args.squad_transfer_rules_version
        or rule_versions.get("squad_transfer")
        or ""
    )
    chip_rules_version = args.chip_rules_version or rule_versions.get("chips") or ""
    bps_rules_version = (
        args.bps_rules_version or rule_versions.get("bonus_points_system") or ""
    )
    if not scoring_rules_version:
        raise Day76DInputError("A target-season scoring rules version is required.")

    input_paths = resolve_rollover_inputs(rollover_path)
    current_pool = load_csv(input_paths["current_player_pool"], "current player pool")
    player_mapping_source = load_csv(
        input_paths["player_identity_mapping"], "player identity mapping"
    )
    team_mapping_source = load_csv(
        input_paths["team_identity_mapping"], "team identity mapping"
    )
    fixture_source = load_csv(input_paths["fixtures"], "GW1-GW5 fixtures")

    target_players = adapt_current_player_pool(current_pool)
    target_teams = adapt_target_teams(current_pool)
    target_fixtures = adapt_target_fixtures(fixture_source, target_gw)
    player_mapping = adapt_player_identity_mapping(player_mapping_source)
    team_mapping = adapt_team_identity_mapping(team_mapping_source)
    raw_player_priors = build_canonical_player_priors(source_season, target_season)
    player_priors = adapt_player_priors(raw_player_priors)
    team_priors = build_canonical_team_priors(source_season, target_season)

    own_work_dir = not bool(args.work_dir)
    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="day76d_"))

    try:
        paths: Dict[str, Path] = {
            "player_priors_csv": work_dir / "player_priors.csv",
            "team_priors_csv": work_dir / "team_priors.csv",
            "player_mapping_csv": work_dir / "player_mapping_adapter.csv",
            "team_mapping_csv": work_dir / "team_mapping_adapter.csv",
            "match_features_csv": work_dir / "match_features.csv",
            "match_features_json": work_dir / "match_features.json",
            "match_features_md": work_dir / "match_features.md",
            "effective_match_features_csv": work_dir / "effective_match_features.csv",
            "effective_match_features_json": work_dir / "effective_match_features.json",
            "effective_match_features_md": work_dir / "effective_match_features.md",
            "match_preview_csv": work_dir / "match_prediction_preview.csv",
            "match_preview_json": work_dir / "match_prediction_preview.json",
            "match_preview_md": work_dir / "match_prediction_preview.md",
            "scoreline_csv": work_dir / "scoreline_preview.csv",
            "scoreline_json": work_dir / "scoreline_preview.json",
            "scoreline_md": work_dir / "scoreline_preview.md",
            "player_features_csv": work_dir / "player_features.csv",
            "player_features_json": work_dir / "player_features.json",
            "player_features_md": work_dir / "player_features.md",
            "role_contract_json": work_dir / "player_role_contract.json",
            "role_contract_md": work_dir / "player_role_contract.md",
            "prediction_preview_csv": work_dir / "player_prediction_preview.csv",
            "prediction_preview_json": work_dir / "player_prediction_preview.json",
            "prediction_preview_md": work_dir / "player_prediction_preview.md",
            "prediction_manifest_json": work_dir / "player_prediction_manifest.json",
            "prediction_manifest_md": work_dir / "player_prediction_manifest.md",
        }
        write_dataframe(paths["player_priors_csv"], player_priors)
        write_dataframe(paths["team_priors_csv"], team_priors)
        write_dataframe(paths["player_mapping_csv"], player_mapping)
        write_dataframe(paths["team_mapping_csv"], team_mapping)

        from ml.features.build_pre_gw1_match_features import (
            build_match_features,
            build_report as build_match_feature_report,
            build_target_team_prior_lookup,
            write_markdown as write_match_feature_markdown,
        )
        from ml.features.build_pre_gw1_player_features import (
            FEATURE_SCOPE as DAY71A_FEATURE_SCOPE,
            FEATURE_VERSION as DAY71A_FEATURE_VERSION,
            build_player_features,
            write_markdown as write_player_feature_markdown,
        )
        from ml.validation.resolve_prediction_mode import resolve_prediction_mode

        mode_result = resolve_prediction_mode(
            season=target_season,
            target_gw=target_gw,
            requested_prediction_mode="auto",
            prior_season=source_season,
            stabilization_gw=6,
            allow_experimental_mode=False,
        )
        if not mode_result.get("valid") or mode_result.get(
            "resolved_prediction_mode"
        ) != PREDICTION_MODE:
            raise RuntimeError("Prediction mode did not resolve to pre_gw1_prior.")

        team_prior_lookup, team_mapping_diagnostics = (
            build_target_team_prior_lookup(team_priors, team_mapping)
        )
        match_features = build_match_features(
            fixtures=target_fixtures,
            target_teams=target_teams,
            team_prior_lookup=team_prior_lookup,
            source_season=source_season,
            target_season=target_season,
            target_gw=target_gw,
            resolved_prediction_mode=PREDICTION_MODE,
        )
        write_dataframe(paths["match_features_csv"], match_features)
        match_args = SimpleNamespace(
            source_season=source_season,
            target_season=target_season,
            target_gw=target_gw,
            prediction_mode="auto",
            team_prior_csv=str(paths["team_priors_csv"]),
            team_mapping_csv=str(paths["team_mapping_csv"]),
            out_csv=str(paths["match_features_csv"]),
            out_json=str(paths["match_features_json"]),
            out_md=str(paths["match_features_md"]),
        )
        match_report = build_match_feature_report(
            features=match_features,
            fixtures=target_fixtures,
            target_teams=target_teams,
            team_prior_lookup=team_prior_lookup,
            mapping_diagnostics=team_mapping_diagnostics,
            mode_result=mode_result,
            args=match_args,
            blockers=[],
            warnings=[],
        )
        match_report["input_mode"] = "day76c_artifact_first"
        match_report["writes_database"] = False
        write_json(paths["match_features_json"], match_report)
        write_match_feature_markdown(match_report, str(paths["match_features_md"]))
        if not match_report.get("passed"):
            raise RuntimeError("Artifact-first match feature report contains blockers.")

        run_module(
            backend_root,
            "ml.features.apply_pre_gw1_team_fallback_policy",
            [
                "--source-season", source_season,
                "--target-season", target_season,
                "--target-gw", str(target_gw),
                "--prediction-mode", "auto",
                "--pre-gw1-match-features-csv", str(paths["match_features_csv"]),
                "--team-prior-csv", str(paths["team_priors_csv"]),
                "--team-mapping-csv", str(paths["team_mapping_csv"]),
                "--out-csv", str(paths["effective_match_features_csv"]),
                "--out-json", str(paths["effective_match_features_json"]),
                "--out-md", str(paths["effective_match_features_md"]),
            ],
        )
        run_module(
            backend_root,
            "ml.features.build_pre_gw1_match_prediction_preview",
            [
                "--source-season", source_season,
                "--target-season", target_season,
                "--target-gw", str(target_gw),
                "--prediction-mode", "auto",
                "--match-features-csv", str(paths["effective_match_features_csv"]),
                "--out-csv", str(paths["match_preview_csv"]),
                "--out-json", str(paths["match_preview_json"]),
                "--out-md", str(paths["match_preview_md"]),
            ],
        )
        run_module(
            backend_root,
            "ml.features.build_pre_gw1_scoreline_preview",
            [
                "--source-season", source_season,
                "--target-season", target_season,
                "--target-gw", str(target_gw),
                "--prediction-mode", "auto",
                "--match-features-csv", str(paths["effective_match_features_csv"]),
                "--match-prediction-preview-csv", str(paths["match_preview_csv"]),
                "--out-csv", str(paths["scoreline_csv"]),
                "--out-json", str(paths["scoreline_json"]),
                "--out-md", str(paths["scoreline_md"]),
            ],
        )

        effective_match_features = load_csv(
            paths["effective_match_features_csv"], "effective match features"
        )
        player_features, player_build_summary = build_player_features(
            source_season=source_season,
            target_season=target_season,
            target_gw=target_gw,
            resolved_prediction_mode=PREDICTION_MODE,
            target_players=target_players,
            target_teams=target_teams,
            target_fixtures=target_fixtures,
            player_priors=player_priors,
            player_mapping=player_mapping,
            match_features=effective_match_features,
        )
        write_dataframe(paths["player_features_csv"], player_features)
        player_args = SimpleNamespace(
            source_season=source_season,
            target_season=target_season,
            target_gw=target_gw,
            prediction_mode="auto",
            stabilization_gw=6,
            allow_experimental_mode=False,
            player_prior_csv=str(paths["player_priors_csv"]),
            player_mapping_csv=str(paths["player_mapping_csv"]),
            match_features_csv=str(paths["effective_match_features_csv"]),
            out_csv=str(paths["player_features_csv"]),
            out_json=str(paths["player_features_json"]),
            out_md=str(paths["player_features_md"]),
        )
        player_report = build_artifact_first_player_feature_report(
            args=player_args,
            mode_result=mode_result,
            features=player_features,
            build_summary=player_build_summary,
            player_priors=player_priors,
            player_mapping=player_mapping,
            match_features=effective_match_features,
            target_player_count=len(target_players),
            target_team_count=len(target_teams),
            target_fixture_count=len(target_fixtures),
            feature_version=DAY71A_FEATURE_VERSION,
            feature_scope=DAY71A_FEATURE_SCOPE,
        )
        write_json(paths["player_features_json"], player_report)
        write_player_feature_markdown(
            player_report, str(paths["player_features_md"])
        )
        if not player_report.get("passed"):
            raise RuntimeError("Artifact-first player feature report contains blockers.")

        run_module(
            backend_root,
            "ml.validation.export_player_role_feature_contract",
            [
                "--source-season", source_season,
                "--target-season", target_season,
                "--target-gw", str(target_gw),
                "--as-of-time", as_of_time,
                "--prediction-mode", "auto",
                "--player-features-csv", str(paths["player_features_csv"]),
                "--day71a-json", str(paths["player_features_json"]),
                "--scoring-rules-version", scoring_rules_version,
                "--transfer-rules-version", squad_transfer_rules_version,
                "--chip-rules-version", chip_rules_version,
                "--status-source", "Day76C current_player_pool.csv",
                "--status-as-of", as_of_time,
                "--status-valid-for-prediction-cutoff",
                "--out-json", str(paths["role_contract_json"]),
                "--out-md", str(paths["role_contract_md"]),
            ],
        )
        run_module(
            backend_root,
            "ml.predict.build_pre_gw1_player_prediction_preview",
            [
                "--source-season", source_season,
                "--target-season", target_season,
                "--target-gw", str(target_gw),
                "--as-of-time", as_of_time,
                "--prediction-mode", "auto",
                "--player-features-csv", str(paths["player_features_csv"]),
                "--day71a-json", str(paths["player_features_json"]),
                "--day71b-json", str(paths["role_contract_json"]),
                "--scoreline-preview-csv", str(paths["scoreline_csv"]),
                "--day70c-json", str(paths["scoreline_json"]),
                "--scoring-rules-version", scoring_rules_version,
                "--out-csv", str(paths["prediction_preview_csv"]),
                "--out-json", str(paths["prediction_preview_json"]),
                "--out-md", str(paths["prediction_preview_md"]),
            ],
        )
        day72a_report = load_json(paths["prediction_preview_json"])
        role_report = load_json(paths["role_contract_json"])
        scoreline_report = load_json(paths["scoreline_json"])
        run_module(
            backend_root,
            "ml.validation.export_pre_gw1_player_prediction_manifest",
            [
                "--source-season", source_season,
                "--target-season", target_season,
                "--target-gw", str(target_gw),
                "--as-of-time", as_of_time,
                "--prediction-mode", "auto",
                "--prediction-preview-csv", str(paths["prediction_preview_csv"]),
                "--day72a-json", str(paths["prediction_preview_json"]),
                "--day71a-json", str(paths["player_features_json"]),
                "--day71b-json", str(paths["role_contract_json"]),
                "--day70c-json", str(paths["scoreline_json"]),
                "--expected-model-name", str(day72a_report["model_name"]),
                "--expected-model-version", str(day72a_report["model_version"]),
                "--expected-prediction-scope", str(day72a_report["prediction_scope"]),
                "--player-feature-version", str(day72a_report["player_feature_version"]),
                "--role-contract-version", str(day72a_report["role_contract_version"]),
                "--threshold-policy-version", str(day72a_report["threshold_policy_version"]),
                "--scoreline-model-name", str(day72a_report["scoreline_model_name"]),
                "--scoreline-model-version", str(day72a_report["scoreline_model_version"]),
                "--scoring-rules-version", scoring_rules_version,
                "--out-json", str(paths["prediction_manifest_json"]),
                "--out-md", str(paths["prediction_manifest_md"]),
            ],
        )

        preview = load_csv(paths["prediction_preview_csv"], "Day72A preview")
        manifest = load_json(paths["prediction_manifest_json"])
        from ml.contracts.predictions import (
            adapt_day72a_player_points_preview,
            adapt_day72b_player_prediction_manifest,
        )

        manifest_contract = adapt_day72b_player_prediction_manifest(manifest)
        if not manifest_contract.preview_schema_consumption_allowed:
            raise RuntimeError(
                "Day76B manifest adapter did not allow preview schema consumption."
            )
        selection_eligibility = {
            int(row.player_id): bool(row.current_selection_eligible)
            for row in target_players.itertuples(index=False)
        }
        standard_rows = adapt_standard_predictions(
            preview,
            adapt_day72a_player_points_preview,
            selection_eligibility=selection_eligibility,
        )

        standard_player_ids = [nullable_int(row.get("player_id")) for row in standard_rows]
        if len(standard_player_ids) != len(set(standard_player_ids)):
            raise RuntimeError("Standard prediction rows contain duplicate player IDs.")
        if set(standard_player_ids) != set(target_players["player_id"].astype(int)):
            raise RuntimeError("Standard prediction rows do not retain the full Day76C player pool.")
        unsafe_write_rows = [
            row for row in standard_rows
            if bool_value(row.get("prediction_write_allowed"))
            or bool_value(row.get("production_ready"))
            or bool_value(row.get("production_approved"))
        ]
        if unsafe_write_rows:
            raise RuntimeError("Standard prediction rows contain unsafe write/production flags.")
        ineligible_source_ids = {
            player_id
            for player_id, eligible in selection_eligibility.items()
            if not eligible
        }
        incorrectly_eligible = [
            row for row in standard_rows
            if nullable_int(row.get("player_id")) in ineligible_source_ids
            and bool_value(row.get("selection_eligible"))
        ]
        if incorrectly_eligible:
            raise RuntimeError("Day76C-ineligible players remained selection eligible.")

        bonus_prior_missing = bool(
            "prev_season_bonus" not in player_priors.columns
            or player_priors["prev_season_bonus"].isna().all()
        )
        limitations = build_limitation_contract(
            scoring_rules_version,
            bps_rules_version or None,
            bonus_prior_missing,
        )

        from ml.contracts.run_metadata import (
            build_run_metadata,
            provenance_inputs_from_file_metadata,
        )
        from ml.artifacts.paths import build_immutable_artifact_key
        from ml.artifacts.storage import LocalArtifactStorage

        created_at = utc_now()
        input_metadata: Dict[str, Dict[str, Any]] = {}
        for key, path in input_paths.items():
            input_metadata[key] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        parent_run_id = (
            (rollover_report.get("run_metadata") or {}).get("run_id")
            or rollover_path.parent.name.replace("run_id=", "")
        )
        run_metadata = build_run_metadata(
            run_id=None,
            run_type="prediction",
            artifact_type=ARTIFACT_TYPE,
            source_seasons=[source_season],
            target_season=target_season,
            target_gw=target_gw,
            horizon=1,
            as_of_time=as_of_time,
            prediction_mode=PREDICTION_MODE,
            created_at=created_at,
            feature_version=str(day72a_report["player_feature_version"]),
            model_version=str(day72a_report["model_version"]),
            rules_versions={
                "scoring": scoring_rules_version,
                "squad_transfer": squad_transfer_rules_version,
                "chips": chip_rules_version,
                "bonus_points_system": bps_rules_version,
                "role_contract": str(day72a_report["role_contract_version"]),
                "threshold_policy": str(day72a_report["threshold_policy_version"]),
            },
            manifest_version=REFRESH_VERSION,
            artifact_version=REFRESH_VERSION,
            additional_versions={
                "day72b_manifest": str(manifest.get("manifest_version")),
                "scoreline_model": str(scoreline_report.get("model_version")),
            },
            provenance={
                "producer": "ml.validation.refresh_pre_gw1_player_predictions",
                "inputs": provenance_inputs_from_file_metadata(input_metadata),
                "parent_run_ids": [str(parent_run_id)],
                "notes": [
                    "Artifact-first Day76D refresh using Day76C immutable inputs.",
                    "Canonical source-season tables are read only to build priors.",
                    "No prediction database writes are performed.",
                ],
            },
        ).to_dict()

        storage = LocalArtifactStorage(args.artifact_root)
        run_id = str(run_metadata["run_id"])
        stored: Dict[str, Dict[str, Any]] = {}

        payloads: Dict[str, Tuple[str, str]] = {}
        for key, path in paths.items():
            if path.is_file():
                extension = path.suffix.lstrip(".") or "txt"
                payloads[key] = (extension, path.read_text(encoding="utf-8"))
        payloads["standard_player_predictions_csv"] = (
            "csv",
            rows_to_csv_text(standard_rows),
        )
        payloads["standard_player_predictions_json"] = (
            "json",
            json.dumps(standard_rows, indent=2, sort_keys=True, default=str) + "\n",
        )

        for name, (extension, content) in payloads.items():
            key = build_immutable_artifact_key(
                artifact_type=ARTIFACT_TYPE,
                season=target_season,
                target_gw=target_gw,
                as_of_time=as_of_time,
                run_id=run_id,
                version=REFRESH_VERSION,
                filename=name,
                extension=extension,
            )
            artifact = storage.write_immutable_text(key, content)
            stored[name] = artifact.to_dict()

        safe_mapping_count = int(
            player_mapping["is_auto_approved"].apply(bool_value).sum()
        )
        fallback_player_count = int(len(target_players) - safe_mapping_count)
        standard_eligible = int(
            sum(bool_value(row.get("selection_eligible")) for row in standard_rows)
        )
        warnings: List[str] = []
        if bonus_prior_missing:
            warnings.append(
                "Canonical previous-season bonus is unavailable; expected bonus remains an unresolved heuristic component."
            )
        if fallback_player_count > 0:
            warnings.append(
                "%s target players remain visible through explicit fallback treatment."
                % fallback_player_count
            )

        report: Dict[str, Any] = {
            "created_at_utc": created_at,
            "refresh_version": REFRESH_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "source_season": source_season,
            "target_season": target_season,
            "target_gw": target_gw,
            "as_of_time_utc": as_of_time,
            "passed": True,
            "audit_only": True,
            "writes_database": False,
            "writes_predictions_table": False,
            "prediction_write_allowed": False,
            **REQUIRED_LIMITATION_LABELS,
            "recommendation_status": "preview_only",
            "run_metadata": run_metadata,
            "limitations": limitations,
            "validation": {
                "target_player_count": int(len(target_players)),
                "target_team_count": int(len(target_teams)),
                "gw1_fixture_count": int(len(target_fixtures)),
                "source_player_prior_count": int(len(player_priors)),
                "source_team_prior_count": int(len(team_priors)),
                "safe_mapping_count": safe_mapping_count,
                "fallback_player_count": fallback_player_count,
                "player_feature_row_count": int(len(player_features)),
                "prediction_preview_row_count": int(len(preview)),
                "standard_contract_row_count": int(len(standard_rows)),
                "standard_selection_eligible_count": standard_eligible,
                "day76c_current_selection_ineligible_count": int(
                    len(ineligible_source_ids)
                ),
                "day72b_manifest_passed": bool(manifest.get("passed")),
                "day76b_preview_schema_consumption_allowed": True,
                "day72b_original_optimizer_input_ready": bool(
                    manifest_contract.original_optimizer_input_ready
                ),
                "immutable_artifact_count": len(stored) + 2,
                "all_target_players_retained": (
                    len(target_players) == len(player_features) == len(preview) == len(standard_rows)
                ),
            },
            "source_artifacts": input_metadata,
            "immutable_artifacts": stored,
            "blockers": [],
            "warnings": warnings,
            "stop_point_satisfied": True,
            "ready_for_day79b_horizon_work": True,
            "ready_for_day97a_preview_source": True,
            "ready_for_production_prediction": False,
        }
        report_json_text = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ) + "\n"
        report_md_text = build_markdown_report(report)
        for filename, extension, content in (
            ("refresh_report", "json", report_json_text),
            ("refresh_report", "md", report_md_text),
        ):
            key = build_immutable_artifact_key(
                artifact_type=ARTIFACT_TYPE,
                season=target_season,
                target_gw=target_gw,
                as_of_time=as_of_time,
                run_id=run_id,
                version=REFRESH_VERSION,
                filename=filename,
                extension=extension,
            )
            artifact = storage.write_immutable_text(key, content)
            stored_key = "refresh_report_%s" % extension
            stored[stored_key] = artifact.to_dict()

        print("Day76D refresh complete.")
        print("run_id:", run_id)
        print("target_players:", len(target_players))
        print("safe_mappings:", safe_mapping_count)
        print("fallback_players:", fallback_player_count)
        print("standard_contract_rows:", len(standard_rows))
        print("immutable_artifacts:", len(stored))
        print("prediction_source:", PREDICTION_SOURCE)
        print("production_approved: false")
        print("historical_multi_season_backtest_complete: false")
        print("component_model_stack_complete: false")
        print("writes_database: false")
        print("stop_point_satisfied: true")
    finally:
        if own_work_dir and not args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        elif args.keep_work_dir:
            print("work_dir:", work_dir)


if __name__ == "__main__":
    main()
