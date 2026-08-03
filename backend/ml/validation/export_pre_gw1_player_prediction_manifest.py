from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ml.contracts.run_metadata import (
    build_run_metadata,
    provenance_inputs_from_file_metadata,
)
from ml.validation.resolve_prediction_mode import resolve_prediction_mode


MANIFEST_VERSION = "day72b_v1"
ARTIFACT_TYPE = "pre_gw1_player_prediction_manifest"
EXPECTED_MODEL_NAME = "pre_gw1_player_prior_heuristic_v0"
EXPECTED_MODEL_VERSION = "day72a_v0_1"
EXPECTED_PREDICTION_SCOPE = "read_only_pre_gw1_player_prediction_preview"
EXPECTED_CALIBRATION_STATUS = "not_calibrated_preview_only"
EXPECTED_GUARDRAIL_STATUS = (
    "basic_bounds_fixture_multiplier_and_role_uncertainty_v0_1"
)
EXPECTED_COMPONENT_ACCOUNTING_STATUS = (
    "heuristic_components_reconciled_to_final_points"
)
GOALKEEPER_TEAM_APPEARANCE_BUDGET = 0.98
VALID_POSITIONS = {"GKP", "DEF", "MID", "FWD"}
VALID_FALLBACK_LEVELS = {3, 4, 5}
VALID_CONFIDENCE_VALUES = {
    "very_low",
    "low",
    "medium",
    "medium_high",
    "high",
}
VALID_SAMPLE_TIERS = {
    "no_safe_prior",
    "zero_sample",
    "very_low_sample",
    "low_sample",
    "moderate_sample",
    "reliable_sample",
    "high_sample",
}
VALID_ROLE_PROXIES = {
    "no_safe_prior",
    "insufficient_sample",
    "bench_role_proxy",
    "rotation_or_unclear_role",
    "probable_starter_proxy",
    "established_starter_proxy",
}

REQUIRED_PREVIEW_COLUMNS = [
    "source_seasons",
    "target_season",
    "target_gw",
    "as_of_time",
    "prediction_mode",
    "prediction_scope",
    "run_id",
    "model_name",
    "model_version",
    "player_feature_version",
    "role_contract_version",
    "threshold_policy_version",
    "scoreline_model_name",
    "scoreline_model_version",
    "scoring_rules_version",
    "player_id",
    "fpl_player_id",
    "web_name",
    "team_id",
    "team_short_name",
    "position",
    "price",
    "now_cost",
    "fixture_id",
    "opponent_team_id",
    "opponent_short_name",
    "is_home",
    "has_fixture",
    "sample_reliability_tier",
    "role_proxy",
    "role_class",
    "role_confidence",
    "appearance_probability",
    "start_probability",
    "conditional_minutes_if_appears",
    "expected_minutes",
    "minutes_lower_bound",
    "minutes_upper_bound",
    "raw_expected_points",
    "fixture_adjustment",
    "guardrail_adjustment",
    "calibration_adjustment",
    "final_predicted_points",
    "predicted_points",
    "expected_appearance_points",
    "expected_goals",
    "expected_goal_points",
    "expected_assists",
    "expected_assist_points",
    "clean_sheet_probability",
    "expected_clean_sheet_points",
    "expected_bonus",
    "expected_other_points",
    "has_safe_prior",
    "fallback_policy_used",
    "fallback_level",
    "fallback_reason",
    "fallback_reference_sample_count",
    "position_price_band",
    "position_price_band_percentile",
    "fallback_rate_price_signal",
    "fallback_minutes_price_signal",
    "fallback_role_uncertainty_multiplier",
    "goalkeeper_team_candidate_count",
    "goalkeeper_team_share_cap",
    "goalkeeper_competition_adjustment",
    "price_band_reliable_sample_count",
    "risk_flags",
    "status_cutoff_valid",
    "status_hard_guardrail_applied",
    "data_quality_status",
    "prediction_confidence",
    "prediction_write_allowed",
    "production_ready",
    "requires_player_prediction_manifest_before_write",
    "calibration_status",
    "guardrail_status",
    "component_accounting_status",
]

POINT_COMPONENT_COLUMNS = [
    "expected_appearance_points",
    "expected_goal_points",
    "expected_assist_points",
    "expected_clean_sheet_points",
    "expected_bonus",
    "expected_other_points",
]

CURRENT_PRODUCTION_BLOCKERS = [
    "player_preview_is_prior_based_heuristic_not_trained_model",
    "minutes_and_start_probabilities_not_backtested_or_calibrated",
    "goal_assist_clean_sheet_and_bonus_components_not_calibrated",
    "player_points_not_validated_by_rolling_historical_backtest",
    "fallback_policy_not_evaluated_across_multiple_seasons",
    "status_metadata_not_proven_valid_at_historical_prediction_cutoff",
    "scoring_rules_registry_not_resolved_and_approved",
    "no_active_player_model_registry_entry",
    "no_fail_safe_database_write_and_rollback_path",
    "no_public_display_contract_for_uncertainty_and_fallbacks",
    "not_approved_as_opening_squad_or_transfer_optimizer_input",
]

REQUIRED_BEFORE_PRODUCTION = [
    "approved_minutes_start_and_event_model_policy",
    "multi_season_leakage_safe_rolling_backtests",
    "minutes_start_probability_and_points_calibration_reports",
    "fallback_policy_evaluation_for_new_promoted_and_low_sample_players",
    "cutoff_valid_player_status_and_availability_contract",
    "resolved_target_season_scoring_rules_registry",
    "active_player_model_registry_entry_with_versioned_artifacts",
    "database_write_path_with_dry_run_idempotency_rollback_and_manifest_gate",
    "public_copy_that_distinguishes_heuristic_preview_from_production_prediction",
    "explicit_approval_for_rankings_squad_optimizer_transfers_and_captaincy",
]


class ManifestInputError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the Day72B Pre-GW1 player prediction manifest and safety "
            "contract from the Day72A read-only player prediction preview. "
            "The manifest validates metadata, row shape, probabilities, minutes, "
            "point-component accounting, fallback visibility, artifact fingerprints, "
            "and write blocking. It does not train a model or write to the database."
        )
    )
    parser.add_argument(
        "--source-season",
        dest="source_seasons",
        action="append",
        required=True,
        help=(
            "Historical source season. Repeat for multiple lookback seasons, "
            "ordered oldest to newest."
        ),
    )
    parser.add_argument("--target-season", required=True)
    parser.add_argument("--target-gw", type=int, default=1)
    parser.add_argument("--as-of-time", required=True)
    parser.add_argument(
        "--prediction-mode",
        default="auto",
        choices=[
            "auto",
            "pre_gw1_prior",
            "early_season_blend",
            "normal_weekly",
        ],
    )
    parser.add_argument("--stabilization-gw", type=int, default=6)
    parser.add_argument("--allow-experimental-mode", action="store_true")

    parser.add_argument(
        "--prediction-preview-csv",
        required=True,
        help="Day72A player prediction preview CSV.",
    )
    parser.add_argument(
        "--day72a-json",
        required=True,
        help="Day72A player prediction preview JSON report.",
    )
    parser.add_argument(
        "--day71a-json",
        default="",
        help="Optional Day71A player feature JSON report.",
    )
    parser.add_argument(
        "--day71b-json",
        default="",
        help="Optional Day71B role/minutes contract JSON report.",
    )
    parser.add_argument(
        "--day70c-json",
        default="",
        help="Optional Day70C scoreline preview JSON report.",
    )

    parser.add_argument("--expected-model-name", default=EXPECTED_MODEL_NAME)
    parser.add_argument("--expected-model-version", default=EXPECTED_MODEL_VERSION)
    parser.add_argument(
        "--expected-prediction-scope",
        default=EXPECTED_PREDICTION_SCOPE,
    )
    parser.add_argument("--player-feature-version", default="day71a_v0")
    parser.add_argument("--role-contract-version", default="day71b_v1")
    parser.add_argument(
        "--threshold-policy-version",
        default="player_role_thresholds_v0",
    )
    parser.add_argument(
        "--scoreline-model-name",
        default="pre_gw1_scoreline_prior_heuristic_v0",
    )
    parser.add_argument("--scoreline-model-version", default="day70c_v0")
    parser.add_argument(
        "--scoring-rules-version",
        default="target_season_rules_unresolved",
    )

    parser.add_argument("--prediction-points-min", type=float, default=0.0)
    parser.add_argument("--prediction-points-max", type=float, default=15.0)
    parser.add_argument("--component-tolerance", type=float, default=0.00001)
    parser.add_argument("--top-risk-window", type=int, default=20)
    parser.add_argument("--top-rows-per-position", type=int, default=5)

    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso8601(value: str, label: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ManifestInputError(
            "%s must be valid ISO-8601: %s" % (label, value)
        ) from exc
    if parsed.tzinfo is None:
        raise ManifestInputError("%s must include a timezone offset." % label)
    return parsed


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def nested_get(
    data: Optional[Dict[str, Any]],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def sha256_file(path_value: str) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path_value: str) -> Dict[str, Any]:
    if not path_value:
        return {
            "path": None,
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }
    path = Path(path_value)
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path_value),
    }


def load_json_result(path_value: str, required: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": path_value or None,
        "exists": False,
        "loaded": False,
        "data": None,
        "error": None,
    }
    if not path_value:
        if required:
            result["error"] = "Required JSON path was not provided."
        return result

    path = Path(path_value)
    result["exists"] = path.exists()
    if not path.exists():
        result["error"] = "JSON file does not exist: %s" % path
        return result

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = "%s" % exc
        return result

    if not isinstance(data, dict):
        result["error"] = "JSON root must be an object."
        return result

    result["loaded"] = True
    result["data"] = data
    return result


def load_preview(path_value: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise ManifestInputError("Preview CSV does not exist: %s" % path)
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ManifestInputError(
            "Could not read preview CSV %s: %s" % (path, exc)
        ) from exc


def count_dict(series: pd.Series) -> Dict[str, int]:
    values = series.fillna("<missing>").astype(str)
    counts = values.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.to_dict().items()}


def unique_text_values(series: pd.Series) -> List[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def numeric_series(preview: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(preview[column], errors="coerce")


def same_path(left: Any, right: str) -> bool:
    if left is None:
        return False
    try:
        return Path(str(left)).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return str(left) == str(right)


def parse_risk_flag_counts(series: pd.Series) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in series.fillna("").astype(str):
        for item in value.split(","):
            flag = item.strip()
            if flag:
                counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def validate_cli_policy(args: argparse.Namespace) -> List[str]:
    blockers: List[str] = []
    if not args.source_seasons:
        blockers.append("At least one source season is required.")
    if args.target_gw < 1:
        blockers.append("target_gw must be >= 1.")
    if args.prediction_points_min > args.prediction_points_max:
        blockers.append(
            "prediction-points-min must not exceed prediction-points-max."
        )
    if args.component_tolerance < 0:
        blockers.append("component-tolerance must be non-negative.")
    if args.top_risk_window < 1:
        blockers.append("top-risk-window must be >= 1.")
    if args.top_rows_per_position < 1:
        blockers.append("top-rows-per-position must be >= 1.")
    return blockers


def validate_source_report(
    label: str,
    result: Dict[str, Any],
    required: bool,
    blockers: List[str],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    if required and not result.get("exists"):
        blockers.append("%s JSON does not exist." % label)
        return None
    if required and not result.get("loaded"):
        blockers.append(
            "%s JSON could not be loaded: %s"
            % (label, result.get("error"))
        )
        return None
    if result.get("exists") and not result.get("loaded"):
        warnings.append(
            "%s JSON was provided but could not be loaded: %s"
            % (label, result.get("error"))
        )
        return None
    return result.get("data") if result.get("loaded") else None


def validate_day72a_report(
    args: argparse.Namespace,
    report: Dict[str, Any],
    resolved_mode: str,
    blockers: List[str],
) -> None:
    checks: List[Tuple[bool, str]] = [
        (report.get("passed") is True, "Day72A report did not pass."),
        (
            report.get("audit_only") is True,
            "Day72A report must record audit_only=true.",
        ),
        (
            report.get("writes_database") is False,
            "Day72A report must record writes_database=false.",
        ),
        (
            report.get("uses_current_season_actual_player_gw_stats") is False,
            "Day72A report must not use current-season actual player GW stats.",
        ),
        (
            report.get("ready_for_day72b_player_prediction_manifest") is True,
            "Day72A report is not ready for Day72B manifest validation.",
        ),
        (
            report.get("ready_for_production_write") is False,
            "Day72A report must remain blocked from production write.",
        ),
        (
            report.get("model_name") == args.expected_model_name,
            "Day72A report model_name does not match the expected model.",
        ),
        (
            report.get("model_version") == args.expected_model_version,
            "Day72A report model_version does not match the expected version.",
        ),
        (
            report.get("prediction_scope") == args.expected_prediction_scope,
            "Day72A report prediction_scope does not match the expected scope.",
        ),
        (
            report.get("target_season") == args.target_season,
            "Day72A report target_season does not match CLI.",
        ),
        (
            int(report.get("target_gw") or -1) == int(args.target_gw),
            "Day72A report target_gw does not match CLI.",
        ),
        (
            report.get("resolved_prediction_mode") == resolved_mode,
            "Day72A report resolved prediction mode does not match resolver.",
        ),
        (
            report.get("player_feature_version") == args.player_feature_version,
            "Day72A player feature version does not match CLI.",
        ),
        (
            report.get("role_contract_version") == args.role_contract_version,
            "Day72A role contract version does not match CLI.",
        ),
        (
            report.get("threshold_policy_version")
            == args.threshold_policy_version,
            "Day72A threshold policy version does not match CLI.",
        ),
        (
            report.get("scoreline_model_name") == args.scoreline_model_name,
            "Day72A scoreline model name does not match CLI.",
        ),
        (
            report.get("scoreline_model_version")
            == args.scoreline_model_version,
            "Day72A scoreline model version does not match CLI.",
        ),
        (
            report.get("scoring_rules_version") == args.scoring_rules_version,
            "Day72A scoring rules version does not match CLI.",
        ),
        (
            not (report.get("blockers") or []),
            "Day72A report contains blockers.",
        ),
    ]
    for valid, message in checks:
        if not valid:
            blockers.append(message)

    try:
        report_as_of = parse_iso8601(str(report.get("as_of_time")), "Day72A as_of_time")
        if report_as_of != parse_iso8601(args.as_of_time, "as_of_time"):
            blockers.append("Day72A report as_of_time does not match CLI.")
    except ManifestInputError as exc:
        blockers.append(str(exc))

    report_source_seasons = report.get("source_seasons")
    if report_source_seasons != list(args.source_seasons):
        blockers.append("Day72A report source_seasons do not match CLI.")

    output_csv = nested_get(report, ["outputs", "out_csv"], None)
    if not same_path(output_csv, args.prediction_preview_csv):
        blockers.append(
            "Day72A report output CSV path does not match prediction-preview-csv."
        )


def validate_optional_source_reports(
    day71a: Optional[Dict[str, Any]],
    day71b: Optional[Dict[str, Any]],
    day70c: Optional[Dict[str, Any]],
    blockers: List[str],
) -> None:
    if day71a is not None:
        if day71a.get("passed") is not True:
            blockers.append("Day71A report did not pass.")
        if day71a.get("writes_database") is not False:
            blockers.append("Day71A report must record writes_database=false.")
        if day71a.get("blockers") or []:
            blockers.append("Day71A report contains blockers.")

    if day71b is not None:
        if day71b.get("passed") is not True:
            blockers.append("Day71B report did not pass.")
        if day71b.get("writes_database") is not False:
            blockers.append("Day71B report must record writes_database=false.")
        if day71b.get("ready_for_role_feature_contract") is not True:
            blockers.append("Day71B report is not ready for role feature contract.")
        if day71b.get("blockers") or []:
            blockers.append("Day71B report contains blockers.")

    if day70c is not None:
        if day70c.get("passed") is not True:
            blockers.append("Day70C report did not pass.")
        if day70c.get("writes_database") is not False:
            blockers.append("Day70C report must record writes_database=false.")
        if day70c.get("blockers") or []:
            blockers.append("Day70C report contains blockers.")


def validate_metadata(
    args: argparse.Namespace,
    preview: pd.DataFrame,
    resolved_mode: str,
    blockers: List[str],
) -> Dict[str, Any]:
    expected_source_seasons = ",".join(args.source_seasons)
    expected_values: Dict[str, str] = {
        "source_seasons": expected_source_seasons,
        "target_season": args.target_season,
        "prediction_mode": resolved_mode,
        "prediction_scope": args.expected_prediction_scope,
        "model_name": args.expected_model_name,
        "model_version": args.expected_model_version,
        "player_feature_version": args.player_feature_version,
        "role_contract_version": args.role_contract_version,
        "threshold_policy_version": args.threshold_policy_version,
        "scoreline_model_name": args.scoreline_model_name,
        "scoreline_model_version": args.scoreline_model_version,
        "scoring_rules_version": args.scoring_rules_version,
    }
    observed: Dict[str, Any] = {}
    for column, expected in expected_values.items():
        values = unique_text_values(preview[column])
        observed[column] = values
        if values != [str(expected)]:
            blockers.append(
                "%s expected [%s], got %s." % (column, expected, values)
            )

    gw_values = sorted(
        pd.to_numeric(preview["target_gw"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    observed["target_gw"] = gw_values
    if gw_values != [int(args.target_gw)]:
        blockers.append(
            "target_gw expected [%s], got %s." % (args.target_gw, gw_values)
        )

    as_of_values = unique_text_values(preview["as_of_time"])
    observed["as_of_time"] = as_of_values
    normalized_expected_as_of = parse_iso8601(
        args.as_of_time,
        "as_of_time",
    ).isoformat()
    normalized_observed_as_of: List[str] = []
    for value in as_of_values:
        try:
            normalized_observed_as_of.append(
                parse_iso8601(value, "preview as_of_time").isoformat()
            )
        except ManifestInputError as exc:
            blockers.append(str(exc))
    if normalized_observed_as_of != [normalized_expected_as_of]:
        blockers.append("Preview as_of_time does not match CLI.")

    run_ids = unique_text_values(preview["run_id"])
    observed["run_id"] = run_ids
    if len(run_ids) != 1:
        blockers.append("Preview must contain exactly one run_id.")
    return observed


def validate_preview(
    args: argparse.Namespace,
    preview: pd.DataFrame,
    day72a_report: Dict[str, Any],
    resolved_mode: str,
    blockers: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    missing_columns = [
        column for column in REQUIRED_PREVIEW_COLUMNS if column not in preview.columns
    ]
    if missing_columns:
        blockers.append("Preview missing required columns: %s." % missing_columns)
        return {
            "row_count": int(len(preview)),
            "missing_required_columns": missing_columns,
        }

    metadata_values = validate_metadata(
        args,
        preview,
        resolved_mode,
        blockers,
    )

    row_count = int(len(preview))
    if row_count <= 0:
        blockers.append("Preview CSV is empty.")

    player_id = numeric_series(preview, "player_id")
    fpl_player_id = numeric_series(preview, "fpl_player_id")
    missing_player_id_count = int(player_id.isna().sum())
    duplicate_player_id_count = int(player_id.dropna().duplicated().sum())
    missing_fpl_player_id_count = int(fpl_player_id.isna().sum())
    duplicate_fpl_player_id_count = int(fpl_player_id.dropna().duplicated().sum())
    if missing_player_id_count:
        blockers.append("Preview contains missing player_id values.")
    if duplicate_player_id_count:
        blockers.append("Preview contains duplicate player_id rows.")
    if missing_fpl_player_id_count:
        blockers.append("Preview contains missing fpl_player_id values.")
    if duplicate_fpl_player_id_count:
        blockers.append("Preview contains duplicate fpl_player_id rows.")

    missing_team_id_count = int(numeric_series(preview, "team_id").isna().sum())
    missing_position_count = int(preview["position"].isna().sum())
    invalid_positions = sorted(
        set(preview["position"].dropna().astype(str).unique()) - VALID_POSITIONS
    )
    if missing_team_id_count:
        blockers.append("Preview contains missing team_id values.")
    if missing_position_count:
        blockers.append("Preview contains missing position values.")
    if invalid_positions:
        blockers.append("Preview contains invalid positions: %s." % invalid_positions)

    has_fixture = preview["has_fixture"].apply(bool_value)
    fixture_id = numeric_series(preview, "fixture_id")
    fixture_missing_when_required = int((has_fixture & fixture_id.isna()).sum())
    fixture_present_when_blank = int(((~has_fixture) & fixture_id.notna()).sum())
    if fixture_missing_when_required:
        blockers.append("Rows with has_fixture=true must have fixture_id.")
    if fixture_present_when_blank:
        warnings.append("Some has_fixture=false rows still contain fixture_id.")

    probability_columns = [
        "appearance_probability",
        "start_probability",
        "clean_sheet_probability",
    ]
    probability_missing: Dict[str, int] = {}
    probability_range_violations: Dict[str, int] = {}
    for column in probability_columns:
        values = numeric_series(preview, column)
        missing = int(values.isna().sum())
        invalid = int(((values < 0.0) | (values > 1.0)).sum())
        probability_missing[column] = missing
        probability_range_violations[column] = invalid
        if missing:
            blockers.append("%s contains missing/non-numeric values." % column)
        if invalid:
            blockers.append("%s contains values outside [0, 1]." % column)

    appearance = numeric_series(preview, "appearance_probability")
    start = numeric_series(preview, "start_probability")
    start_exceeds_appearance_count = int((start > appearance + 1e-9).sum())
    if start_exceeds_appearance_count:
        blockers.append("start_probability exceeds appearance_probability.")

    conditional_minutes = numeric_series(
        preview,
        "conditional_minutes_if_appears",
    )
    expected_minutes = numeric_series(preview, "expected_minutes")
    minutes_lower = numeric_series(preview, "minutes_lower_bound")
    minutes_upper = numeric_series(preview, "minutes_upper_bound")
    minute_missing_counts = {
        "conditional_minutes_if_appears": int(conditional_minutes.isna().sum()),
        "expected_minutes": int(expected_minutes.isna().sum()),
        "minutes_lower_bound": int(minutes_lower.isna().sum()),
        "minutes_upper_bound": int(minutes_upper.isna().sum()),
    }
    if any(minute_missing_counts.values()):
        blockers.append("Minutes outputs contain missing/non-numeric values.")
    conditional_minutes_range_violations = int(
        ((conditional_minutes < 0.0) | (conditional_minutes > 90.0)).sum()
    )
    expected_minutes_range_violations = int(
        ((expected_minutes < 0.0) | (expected_minutes > 90.0)).sum()
    )
    lower_bound_violations = int((minutes_lower > expected_minutes + 1e-9).sum())
    upper_bound_violations = int((minutes_upper + 1e-9 < expected_minutes).sum())
    if conditional_minutes_range_violations:
        blockers.append("conditional_minutes_if_appears violates [0, 90].")
    if expected_minutes_range_violations:
        blockers.append("expected_minutes violates [0, 90].")
    if lower_bound_violations:
        blockers.append("minutes_lower_bound exceeds expected_minutes.")
    if upper_bound_violations:
        blockers.append("minutes_upper_bound is below expected_minutes.")

    points = numeric_series(preview, "final_predicted_points")
    alias_points = numeric_series(preview, "predicted_points")
    raw_points = numeric_series(preview, "raw_expected_points")
    fixture_adjustment = numeric_series(preview, "fixture_adjustment")
    guardrail_adjustment = numeric_series(preview, "guardrail_adjustment")
    calibration_adjustment = numeric_series(preview, "calibration_adjustment")
    points_missing_count = int(points.isna().sum())
    points_range_violations = int(
        (
            (points < args.prediction_points_min - 1e-9)
            | (points > args.prediction_points_max + 1e-9)
        ).sum()
    )
    max_points_alias_error = float((points - alias_points).abs().max() or 0.0)
    if points_missing_count:
        blockers.append("final_predicted_points contains missing/non-numeric values.")
    if points_range_violations:
        blockers.append("final_predicted_points violates configured guardrails.")
    if max_points_alias_error > 1e-9:
        blockers.append("predicted_points must equal final_predicted_points.")

    component_frame = preview[POINT_COMPONENT_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    component_missing_count = int(component_frame.isna().sum().sum())
    component_sum = component_frame.sum(axis=1)
    max_component_reconciliation_error = float(
        (component_sum - points).abs().max() or 0.0
    )
    if component_missing_count:
        blockers.append("Point components contain missing/non-numeric values.")
    if max_component_reconciliation_error > args.component_tolerance:
        blockers.append("Point components do not reconcile to final points.")

    pipeline_points = (
        raw_points
        + fixture_adjustment
        + guardrail_adjustment
        + calibration_adjustment
    )
    max_prediction_pipeline_reconciliation_error = float(
        (pipeline_points - points).abs().max() or 0.0
    )
    if max_prediction_pipeline_reconciliation_error > args.component_tolerance:
        blockers.append(
            "Raw points plus adjustments do not reconcile to final points."
        )

    calibration_nonzero_count = int((calibration_adjustment.abs() > 1e-12).sum())
    if calibration_nonzero_count:
        blockers.append("Day72A calibration_adjustment must remain zero.")

    price = numeric_series(preview, "price")
    now_cost = numeric_series(preview, "now_cost")
    missing_price_count = int(price.isna().sum())
    negative_price_count = int((price < 0.0).sum())
    max_price_cost_error = float((now_cost - price * 10.0).abs().max() or 0.0)
    if missing_price_count:
        blockers.append("Preview contains missing/non-numeric price values.")
    if negative_price_count:
        blockers.append("Preview contains negative price values.")
    if max_price_cost_error > 0.500001:
        blockers.append("now_cost is inconsistent with price.")

    fallback_levels = numeric_series(preview, "fallback_level")
    fallback_level_values = sorted(
        fallback_levels.dropna().astype(int).unique().tolist()
    )
    invalid_fallback_levels = sorted(
        set(fallback_level_values) - VALID_FALLBACK_LEVELS
    )
    if fallback_levels.isna().any():
        blockers.append("fallback_level contains missing/non-numeric values.")
    if invalid_fallback_levels:
        blockers.append(
            "Preview contains invalid fallback levels: %s."
            % invalid_fallback_levels
        )

    has_safe_prior = preview["has_safe_prior"].apply(bool_value)
    safe_prior_wrong_fallback_count = int(
        (has_safe_prior & (fallback_levels != 3)).sum()
    )
    no_prior_wrong_fallback_count = int(
        ((~has_safe_prior) & (~fallback_levels.isin([4, 5]))).sum()
    )
    if safe_prior_wrong_fallback_count:
        blockers.append("Safe-prior rows must use fallback level 3.")
    if no_prior_wrong_fallback_count:
        blockers.append("No-prior rows must use fallback level 4 or 5.")

    fallback_sample_count = numeric_series(
        preview,
        "fallback_reference_sample_count",
    )
    if fallback_sample_count.isna().any() or (fallback_sample_count < 0).any():
        blockers.append(
            "fallback_reference_sample_count must be non-negative and numeric."
        )

    percentile = numeric_series(preview, "position_price_band_percentile")
    rate_signal = numeric_series(preview, "fallback_rate_price_signal")
    minutes_signal = numeric_series(preview, "fallback_minutes_price_signal")
    role_multiplier = numeric_series(
        preview,
        "fallback_role_uncertainty_multiplier",
    )
    if ((percentile < 0.0) | (percentile > 1.0)).any():
        blockers.append("position_price_band_percentile violates [0, 1].")
    if ((rate_signal < 0.85 - 1e-9) | (rate_signal > 1.15 + 1e-9)).any():
        blockers.append("fallback_rate_price_signal violates [0.85, 1.15].")
    if (
        (minutes_signal < 0.90 - 1e-9)
        | (minutes_signal > 1.10 + 1e-9)
    ).any():
        blockers.append("fallback_minutes_price_signal violates [0.90, 1.10].")
    if ((role_multiplier <= 0.0) | (role_multiplier > 1.0 + 1e-9)).any():
        blockers.append(
            "fallback_role_uncertainty_multiplier must be in (0, 1]."
        )

    invalid_confidence_values = sorted(
        set(preview["prediction_confidence"].dropna().astype(str).unique())
        - VALID_CONFIDENCE_VALUES
    )
    invalid_sample_tiers = sorted(
        set(preview["sample_reliability_tier"].dropna().astype(str).unique())
        - VALID_SAMPLE_TIERS
    )
    invalid_role_proxies = sorted(
        set(preview["role_proxy"].dropna().astype(str).unique())
        - VALID_ROLE_PROXIES
    )
    if invalid_confidence_values:
        blockers.append(
            "Invalid prediction_confidence values: %s."
            % invalid_confidence_values
        )
    if invalid_sample_tiers:
        blockers.append("Invalid sample tiers: %s." % invalid_sample_tiers)
    if invalid_role_proxies:
        blockers.append("Invalid role proxies: %s." % invalid_role_proxies)

    gkp_rows = preview[preview["position"].astype(str) == "GKP"].copy()
    gkp_rows["appearance_probability_numeric"] = pd.to_numeric(
        gkp_rows["appearance_probability"],
        errors="coerce",
    )
    gkp_team_sums = gkp_rows.groupby("team_id")[
        "appearance_probability_numeric"
    ].sum()
    max_gkp_team_appearance_probability_sum = (
        float(gkp_team_sums.max()) if len(gkp_team_sums) else 0.0
    )
    gkp_team_budget_violation_count = int(
        (gkp_team_sums > GOALKEEPER_TEAM_APPEARANCE_BUDGET + 0.00001).sum()
    )
    if gkp_team_budget_violation_count:
        blockers.append(
            "Goalkeeper team appearance probability exceeds configured budget."
        )

    gkp_adjustment = numeric_series(
        preview,
        "goalkeeper_competition_adjustment",
    )
    positive_gkp_adjustment_count = int((gkp_adjustment > 1e-12).sum())
    if positive_gkp_adjustment_count:
        blockers.append("goalkeeper_competition_adjustment must not be positive.")

    write_true_count = int(
        preview["prediction_write_allowed"].apply(bool_value).sum()
    )
    production_true_count = int(preview["production_ready"].apply(bool_value).sum())
    manifest_false_count = int(
        (
            ~preview["requires_player_prediction_manifest_before_write"].apply(
                bool_value
            )
        ).sum()
    )
    status_hard_guardrail_true_count = int(
        preview["status_hard_guardrail_applied"].apply(bool_value).sum()
    )
    if write_true_count:
        blockers.append("prediction_write_allowed must be false for all rows.")
    if production_true_count:
        blockers.append("production_ready must be false for all rows.")
    if manifest_false_count:
        blockers.append("Every row must require a player prediction manifest.")
    if status_hard_guardrail_true_count:
        blockers.append(
            "Historical preview must not apply a hard status guardrail."
        )

    calibration_status_values = unique_text_values(preview["calibration_status"])
    guardrail_status_values = unique_text_values(preview["guardrail_status"])
    component_status_values = unique_text_values(
        preview["component_accounting_status"]
    )
    if calibration_status_values != [EXPECTED_CALIBRATION_STATUS]:
        blockers.append("Unexpected calibration_status values.")
    if guardrail_status_values != [EXPECTED_GUARDRAIL_STATUS]:
        blockers.append("Unexpected guardrail_status values.")
    if component_status_values != [EXPECTED_COMPONENT_ACCOUNTING_STATUS]:
        blockers.append("Unexpected component_accounting_status values.")

    no_prior_count = int((~has_safe_prior).sum())
    fallback_level_counts = count_dict(preview["fallback_level"])
    confidence_counts = count_dict(preview["prediction_confidence"])
    sample_tier_counts = count_dict(preview["sample_reliability_tier"])
    role_proxy_counts = count_dict(preview["role_proxy"])
    data_quality_counts = count_dict(preview["data_quality_status"])
    risk_flag_counts = parse_risk_flag_counts(preview["risk_flags"])

    report_validation = day72a_report.get("validation_summary") or {}
    report_comparisons: Dict[str, Any] = {}
    comparable_values = {
        "preview_rows": row_count,
        "duplicate_player_id_count": duplicate_player_id_count,
        "missing_fixture_join_count": fixture_missing_when_required,
        "output_no_prior_count": no_prior_count,
        "prediction_write_allowed_true_count": write_true_count,
        "production_ready_true_count": production_true_count,
        "requires_manifest_false_count": manifest_false_count,
        "status_hard_guardrail_true_count": status_hard_guardrail_true_count,
        "fallback_level_counts": fallback_level_counts,
        "confidence_counts": confidence_counts,
        "sample_tier_counts": sample_tier_counts,
        "role_proxy_counts": role_proxy_counts,
    }
    for key, actual_value in comparable_values.items():
        reported_value = report_validation.get(key)
        matches = reported_value == actual_value
        report_comparisons[key] = {
            "reported": reported_value,
            "recomputed": actual_value,
            "matches": matches,
        }
        if reported_value is not None and not matches:
            blockers.append(
                "Day72A report validation value %s does not match CSV." % key
            )

    reported_component_error = report_validation.get(
        "max_component_reconciliation_error"
    )
    component_error_matches = True
    if reported_component_error is not None:
        component_error_matches = (
            abs(
                float(reported_component_error)
                - max_component_reconciliation_error
            )
            <= args.component_tolerance
        )
        if not component_error_matches:
            blockers.append(
                "Day72A reported component reconciliation error does not match CSV."
            )
    report_comparisons["max_component_reconciliation_error"] = {
        "reported": reported_component_error,
        "recomputed": max_component_reconciliation_error,
        "matches": component_error_matches,
    }

    sorted_preview = preview.assign(
        _manifest_points=points,
        _manifest_minutes=expected_minutes,
    ).sort_values(
        ["_manifest_points", "_manifest_minutes", "player_id"],
        ascending=[False, False, True],
    )
    top_window = sorted_preview.head(min(args.top_risk_window, row_count))
    top_no_prior_count = int((~top_window["has_safe_prior"].apply(bool_value)).sum())
    top_very_low_confidence_count = int(
        (top_window["prediction_confidence"].astype(str) == "very_low").sum()
    )
    top_fallback_level_4_or_5_count = int(
        (pd.to_numeric(top_window["fallback_level"], errors="coerce") >= 4).sum()
    )
    if top_no_prior_count:
        warnings.append(
            "Top %s preview rows include %s no-prior player(s); internal rankings "
            "must surface fallback and confidence labels."
            % (len(top_window), top_no_prior_count)
        )
    if top_very_low_confidence_count:
        warnings.append(
            "Top %s preview rows include %s very-low-confidence player(s)."
            % (len(top_window), top_very_low_confidence_count)
        )

    if no_prior_count:
        warnings.append(
            "%s player rows use no-safe-prior fallbacks and must remain visible."
            % no_prior_count
        )
    if "sparse_price_band_fallback_flag" in risk_flag_counts:
        warnings.append(
            "%s rows use sparse price-band fallback context."
            % risk_flag_counts["sparse_price_band_fallback_flag"]
        )
    if "status_unknown_for_as_of_flag" in risk_flag_counts:
        warnings.append(
            "Status metadata is not proven cutoff-valid for %s rows."
            % risk_flag_counts["status_unknown_for_as_of_flag"]
        )
    if (
        "pending" in args.scoring_rules_version.lower()
        or "unresolved" in args.scoring_rules_version.lower()
    ):
        warnings.append(
            "Scoring rules version is pending/unresolved and blocks production use."
        )

    return {
        "row_count": row_count,
        "missing_required_columns": [],
        "metadata_values": metadata_values,
        "missing_player_id_count": missing_player_id_count,
        "duplicate_player_id_count": duplicate_player_id_count,
        "missing_fpl_player_id_count": missing_fpl_player_id_count,
        "duplicate_fpl_player_id_count": duplicate_fpl_player_id_count,
        "missing_team_id_count": missing_team_id_count,
        "missing_position_count": missing_position_count,
        "invalid_positions": invalid_positions,
        "fixture_missing_when_required_count": fixture_missing_when_required,
        "fixture_present_when_blank_count": fixture_present_when_blank,
        "probability_missing_counts": probability_missing,
        "probability_range_violation_counts": probability_range_violations,
        "start_exceeds_appearance_count": start_exceeds_appearance_count,
        "minute_missing_counts": minute_missing_counts,
        "conditional_minutes_range_violation_count": (
            conditional_minutes_range_violations
        ),
        "expected_minutes_range_violation_count": (
            expected_minutes_range_violations
        ),
        "minutes_lower_bound_violation_count": lower_bound_violations,
        "minutes_upper_bound_violation_count": upper_bound_violations,
        "points_missing_count": points_missing_count,
        "points_range_violation_count": points_range_violations,
        "max_points_alias_error": max_points_alias_error,
        "component_missing_count": component_missing_count,
        "max_component_reconciliation_error": (
            max_component_reconciliation_error
        ),
        "max_prediction_pipeline_reconciliation_error": (
            max_prediction_pipeline_reconciliation_error
        ),
        "calibration_nonzero_count": calibration_nonzero_count,
        "missing_price_count": missing_price_count,
        "negative_price_count": negative_price_count,
        "max_price_cost_error": max_price_cost_error,
        "fallback_level_values": fallback_level_values,
        "safe_prior_wrong_fallback_count": safe_prior_wrong_fallback_count,
        "no_prior_wrong_fallback_count": no_prior_wrong_fallback_count,
        "invalid_confidence_values": invalid_confidence_values,
        "invalid_sample_tiers": invalid_sample_tiers,
        "invalid_role_proxies": invalid_role_proxies,
        "max_gkp_team_appearance_probability_sum": (
            max_gkp_team_appearance_probability_sum
        ),
        "gkp_team_budget_violation_count": gkp_team_budget_violation_count,
        "positive_gkp_adjustment_count": positive_gkp_adjustment_count,
        "prediction_write_allowed_true_count": write_true_count,
        "production_ready_true_count": production_true_count,
        "requires_manifest_false_count": manifest_false_count,
        "status_hard_guardrail_true_count": status_hard_guardrail_true_count,
        "calibration_status_values": calibration_status_values,
        "guardrail_status_values": guardrail_status_values,
        "component_accounting_status_values": component_status_values,
        "no_safe_prior_count": no_prior_count,
        "fallback_level_counts": fallback_level_counts,
        "confidence_counts": confidence_counts,
        "sample_tier_counts": sample_tier_counts,
        "role_proxy_counts": role_proxy_counts,
        "data_quality_counts": data_quality_counts,
        "risk_flag_counts": risk_flag_counts,
        "day72a_report_comparisons": report_comparisons,
        "top_risk_window": {
            "row_count": int(len(top_window)),
            "no_safe_prior_count": top_no_prior_count,
            "very_low_confidence_count": top_very_low_confidence_count,
            "fallback_level_4_or_5_count": top_fallback_level_4_or_5_count,
        },
    }


def distribution_summary(preview: pd.DataFrame) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for position in ["GKP", "DEF", "MID", "FWD"]:
        rows = preview[preview["position"].astype(str) == position]
        points = pd.to_numeric(rows["predicted_points"], errors="coerce")
        minutes = pd.to_numeric(rows["expected_minutes"], errors="coerce")
        output[position] = {
            "row_count": int(len(rows)),
            "predicted_points_min": float(points.min()) if len(rows) else None,
            "predicted_points_median": (
                float(points.median()) if len(rows) else None
            ),
            "predicted_points_mean": float(points.mean()) if len(rows) else None,
            "predicted_points_max": float(points.max()) if len(rows) else None,
            "expected_minutes_median": (
                float(minutes.median()) if len(rows) else None
            ),
            "expected_minutes_mean": (
                float(minutes.mean()) if len(rows) else None
            ),
        }
    return output


def top_rows_by_position(
    preview: pd.DataFrame,
    limit: int,
) -> Dict[str, List[Dict[str, Any]]]:
    columns = [
        "player_id",
        "web_name",
        "team_short_name",
        "position",
        "price",
        "expected_minutes",
        "predicted_points",
        "prediction_confidence",
        "fallback_level",
        "has_safe_prior",
        "risk_flags",
    ]
    result: Dict[str, List[Dict[str, Any]]] = {}
    for position in ["GKP", "DEF", "MID", "FWD"]:
        rows = preview[preview["position"].astype(str) == position].copy()
        rows["predicted_points_numeric"] = pd.to_numeric(
            rows["predicted_points"],
            errors="coerce",
        )
        rows["expected_minutes_numeric"] = pd.to_numeric(
            rows["expected_minutes"],
            errors="coerce",
        )
        rows = rows.sort_values(
            [
                "predicted_points_numeric",
                "expected_minutes_numeric",
                "price",
                "player_id",
            ],
            ascending=[False, False, True, True],
        ).head(limit)
        result[position] = rows[columns].to_dict(orient="records")
    return result


def source_report_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    data = result.get("data") if result.get("loaded") else None
    return {
        "path": result.get("path"),
        "exists": result.get("exists"),
        "loaded": result.get("loaded"),
        "error": result.get("error"),
        "passed": nested_get(data, ["passed"], None),
        "audit_only": nested_get(data, ["audit_only"], None),
        "writes_database": nested_get(data, ["writes_database"], None),
        "blockers": nested_get(data, ["blockers"], None),
    }


def build_safety_contract(
    passed: bool,
    preview_validation: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "row_level_write_gate": {
            "prediction_write_allowed_true_count": preview_validation.get(
                "prediction_write_allowed_true_count"
            ),
            "production_ready_true_count": preview_validation.get(
                "production_ready_true_count"
            ),
            "requires_manifest_false_count": preview_validation.get(
                "requires_manifest_false_count"
            ),
            "write_gate_passed": bool(
                preview_validation.get(
                    "prediction_write_allowed_true_count"
                )
                == 0
                and preview_validation.get("production_ready_true_count") == 0
                and preview_validation.get("requires_manifest_false_count") == 0
            ),
        },
        "allowed_uses": {
            "local_audit": passed,
            "ops_preview": passed,
            "internal_player_ranking_preview_with_fallback_labels": passed,
            "portfolio_internal_demo_with_disclaimer": passed,
            "master_plan_v2_checkpoint_input": passed,
            "public_production_prediction": False,
            "opening_squad_optimizer_input": False,
            "transfer_recommendation_input": False,
            "captain_recommendation_input": False,
            "database_write": False,
            "active_model_registry_candidate": False,
        },
        "current_production_blockers": list(CURRENT_PRODUCTION_BLOCKERS),
        "required_before_production": list(REQUIRED_BEFORE_PRODUCTION),
    }


def build_standard_run_metadata(
    args: argparse.Namespace,
    mode_result: Dict[str, Any],
    created_at: str,
    manifest_run_id: str,
    artifact_fingerprints: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    # Build the Day75B block without changing Day72B identity fields.
    return build_run_metadata(
        run_id=manifest_run_id,
        run_type="prediction",
        artifact_type=ARTIFACT_TYPE,
        source_seasons=list(args.source_seasons),
        target_season=args.target_season,
        target_gw=args.target_gw,
        horizon=1,
        as_of_time=args.as_of_time,
        prediction_mode=str(mode_result.get("resolved_prediction_mode")),
        created_at=created_at,
        feature_version=args.player_feature_version,
        model_version=args.expected_model_version,
        rules_versions={
            "scoring": args.scoring_rules_version,
            "role_contract": args.role_contract_version,
            "threshold_policy": args.threshold_policy_version,
        },
        manifest_version=MANIFEST_VERSION,
        artifact_version=MANIFEST_VERSION,
        additional_versions={
            "scoreline_model": args.scoreline_model_version,
        },
        provenance={
            "producer": (
                "ml.validation."
                "export_pre_gw1_player_prediction_manifest"
            ),
            "inputs": provenance_inputs_from_file_metadata(
                artifact_fingerprints
            ),
            "parent_run_ids": [],
            "notes": [
                (
                    "Day75B compatibility proof: the existing Day72B "
                    "top-level metadata and run_id remain unchanged."
                )
            ],
        },
    ).to_dict()


def build_manifest(
    args: argparse.Namespace,
    preview: pd.DataFrame,
    mode_result: Dict[str, Any],
    day72a_result: Dict[str, Any],
    day71a_result: Dict[str, Any],
    day71b_result: Dict[str, Any],
    day70c_result: Dict[str, Any],
    preview_validation: Dict[str, Any],
    blockers: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    passed = len(blockers) == 0
    safety_contract = build_safety_contract(passed, preview_validation)
    created_at = utc_now()
    manifest_run_id = "%s_%s_gw%s_%s" % (
        MANIFEST_VERSION,
        args.target_season,
        args.target_gw,
        created_at.replace(":", "").replace("+", "_").replace(".", ""),
    )
    artifact_fingerprints = {
        "prediction_preview_csv": file_metadata(
            args.prediction_preview_csv
        ),
        "day72a_json": file_metadata(args.day72a_json),
        "day71a_json": file_metadata(args.day71a_json),
        "day71b_json": file_metadata(args.day71b_json),
        "day70c_json": file_metadata(args.day70c_json),
    }
    standard_run_metadata = build_standard_run_metadata(
        args=args,
        mode_result=mode_result,
        created_at=created_at,
        manifest_run_id=manifest_run_id,
        artifact_fingerprints=artifact_fingerprints,
    )
    return {
        "created_at": created_at,
        "run_id": manifest_run_id,
        "artifact_type": ARTIFACT_TYPE,
        "manifest_version": MANIFEST_VERSION,
        "run_metadata": standard_run_metadata,
        "source_seasons": list(args.source_seasons),
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "as_of_time": parse_iso8601(args.as_of_time, "as_of_time").isoformat(),
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": mode_result.get("resolved_prediction_mode"),
        "model_name": args.expected_model_name,
        "model_version": args.expected_model_version,
        "prediction_scope": args.expected_prediction_scope,
        "player_feature_version": args.player_feature_version,
        "role_contract_version": args.role_contract_version,
        "threshold_policy_version": args.threshold_policy_version,
        "scoreline_model_name": args.scoreline_model_name,
        "scoreline_model_version": args.scoreline_model_version,
        "scoring_rules_version": args.scoring_rules_version,
        "audit_only": True,
        "writes_database": False,
        "uses_current_season_actual_player_gw_stats": False,
        "trains_minutes_model": False,
        "trains_event_models": False,
        "trains_player_points_model": False,
        "passed": passed,
        "ready_for_pre_gw1_player_prediction_manifest": passed,
        "ready_for_ops_preview": passed,
        "ready_for_internal_player_ranking_preview": passed,
        "ready_for_master_plan_v2_checkpoint": passed,
        "ready_for_public_prediction": False,
        "ready_for_opening_squad_optimizer_input": False,
        "ready_for_transfer_recommendation_input": False,
        "ready_for_captain_recommendation_input": False,
        "ready_for_prediction_write": False,
        "ready_for_production_write": False,
        "reason_production_write_not_ready": (
            "Day72B validates and fingerprints a read-only Day72A heuristic preview. "
            "It does not approve trained-model claims, public predictions, optimizer "
            "inputs, registry activation, or database writes."
        ),
        "artifact_fingerprints": artifact_fingerprints,
        "source_reports": {
            "day72a": source_report_summary(day72a_result),
            "day71a": source_report_summary(day71a_result),
            "day71b": source_report_summary(day71b_result),
            "day70c": source_report_summary(day70c_result),
        },
        "mode_resolution": mode_result,
        "preview_validation": preview_validation,
        "distribution_summary_by_position": distribution_summary(preview),
        "top_preview_by_position": top_rows_by_position(
            preview,
            args.top_rows_per_position,
        ),
        "safety_contract": safety_contract,
        "architecture_principles": {
            "season_agnostic_contract": True,
            "gw_and_as_of_aware": True,
            "leakage_safe": True,
            "versioned_contracts": True,
            "artifact_fingerprints_recorded": True,
            "fallback_visibility_required": True,
            "confidence_visibility_required": True,
            "point_component_accounting_required": True,
            "goalkeeper_competition_guard_checked": True,
            "fail_safe_publishing": True,
            "pure_validation_separated_from_prediction_generation": True,
            "rolling_horizon_compatible": True,
        },
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "notes": [
            (
                "Day72B validates the Day72A preview independently and compares "
                "recomputed metrics with the Day72A JSON report."
            ),
            (
                "Fallback levels, confidence, risk flags, and no-prior players remain "
                "visible for internal audit and ranking preview."
            ),
            (
                "A passing manifest authorizes only internal read-only inspection; "
                "it does not authorize public claims, optimizer decisions, or writes."
            ),
        ],
    }


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    validation = report["preview_validation"]
    safety = report["safety_contract"]
    lines: List[str] = [
        "# Day72B — Pre-GW1 Player Prediction Manifest and Safety Contract",
        "",
        "Created at: `%s`" % report["created_at"],
        "",
        "## Scope",
        "",
        "- Manifest version: `%s`" % report["manifest_version"],
        "- Model: `%s` / `%s`"
        % (report["model_name"], report["model_version"]),
        "- Source seasons: `%s`" % report["source_seasons"],
        "- Target: `%s` GW `%s`"
        % (report["target_season"], report["target_gw"]),
        "- As-of: `%s`" % report["as_of_time"],
        "- Resolved mode: `%s`" % report["resolved_prediction_mode"],
        "",
        "## Readiness",
        "",
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `%s`" % report["audit_only"],
        "- Writes database: `%s`" % report["writes_database"],
        "- Ready for player prediction manifest: `%s`"
        % report["ready_for_pre_gw1_player_prediction_manifest"],
        "- Ready for ops preview: `%s`" % report["ready_for_ops_preview"],
        "- Ready for internal ranking preview: `%s`"
        % report["ready_for_internal_player_ranking_preview"],
        "- Ready for Master Plan v2 checkpoint: `%s`"
        % report["ready_for_master_plan_v2_checkpoint"],
        "- Ready for public prediction: `%s`"
        % report["ready_for_public_prediction"],
        "- Ready for opening squad optimizer: `%s`"
        % report["ready_for_opening_squad_optimizer_input"],
        "- Ready for prediction write: `%s`"
        % report["ready_for_prediction_write"],
        "",
        "## Core Validation",
        "",
        "- Row count: `%s`" % validation.get("row_count"),
        "- Duplicate player IDs: `%s`"
        % validation.get("duplicate_player_id_count"),
        "- Missing fixture joins: `%s`"
        % validation.get("fixture_missing_when_required_count"),
        "- No-safe-prior rows: `%s`"
        % validation.get("no_safe_prior_count"),
        "- Max component reconciliation error: `%s`"
        % validation.get("max_component_reconciliation_error"),
        "- Max prediction-pipeline reconciliation error: `%s`"
        % validation.get("max_prediction_pipeline_reconciliation_error"),
        "- Max GKP team appearance-probability sum: `%s`"
        % validation.get("max_gkp_team_appearance_probability_sum"),
        "- prediction_write_allowed=true rows: `%s`"
        % validation.get("prediction_write_allowed_true_count"),
        "- production_ready=true rows: `%s`"
        % validation.get("production_ready_true_count"),
        "- manifest-required=false rows: `%s`"
        % validation.get("requires_manifest_false_count"),
        "",
        "## Classification and Fallback Summary",
        "",
        "- Fallback levels: `%s`" % validation.get("fallback_level_counts"),
        "- Confidence: `%s`" % validation.get("confidence_counts"),
        "- Sample tiers: `%s`" % validation.get("sample_tier_counts"),
        "- Role proxies: `%s`" % validation.get("role_proxy_counts"),
        "- Data quality: `%s`" % validation.get("data_quality_counts"),
        "- Top risk window: `%s`" % validation.get("top_risk_window"),
        "",
        "## Allowed Uses",
        "",
    ]
    for key, value in safety["allowed_uses"].items():
        lines.append("- %s: `%s`" % (key, value))

    lines.extend(["", "## Current Production Blockers", ""])
    for item in safety["current_production_blockers"]:
        lines.append("- %s" % item)

    lines.extend(["", "## Required Before Production", ""])
    for item in safety["required_before_production"]:
        lines.append("- %s" % item)

    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend("- %s" % item for item in report["blockers"])
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend("- %s" % item for item in report["warnings"])
    else:
        lines.append("- None")

    lines.extend(["", "## Production Boundary", ""])
    lines.append(report["reason_production_write_not_ready"])
    lines.append("")

    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: Dict[str, Any]) -> None:
    validation = report["preview_validation"]
    print("=== Day72B Pre-GW1 Player Prediction Manifest ===")
    print("manifest_version:", report["manifest_version"])
    print("model_name:", report["model_name"])
    print("model_version:", report["model_version"])
    print("source_seasons:", report["source_seasons"])
    print("target_season:", report["target_season"])
    print("target_gw:", report["target_gw"])
    print("as_of_time:", report["as_of_time"])
    print("resolved_prediction_mode:", report["resolved_prediction_mode"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print(
        "ready_for_pre_gw1_player_prediction_manifest:",
        report["ready_for_pre_gw1_player_prediction_manifest"],
    )
    print("ready_for_ops_preview:", report["ready_for_ops_preview"])
    print(
        "ready_for_internal_player_ranking_preview:",
        report["ready_for_internal_player_ranking_preview"],
    )
    print(
        "ready_for_master_plan_v2_checkpoint:",
        report["ready_for_master_plan_v2_checkpoint"],
    )
    print("ready_for_public_prediction:", report["ready_for_public_prediction"])
    print(
        "ready_for_opening_squad_optimizer_input:",
        report["ready_for_opening_squad_optimizer_input"],
    )
    print("ready_for_prediction_write:", report["ready_for_prediction_write"])
    print()
    print("Preview validation:")
    for key in [
        "row_count",
        "duplicate_player_id_count",
        "duplicate_fpl_player_id_count",
        "fixture_missing_when_required_count",
        "no_safe_prior_count",
        "prediction_write_allowed_true_count",
        "production_ready_true_count",
        "requires_manifest_false_count",
        "status_hard_guardrail_true_count",
        "max_component_reconciliation_error",
        "max_prediction_pipeline_reconciliation_error",
        "max_gkp_team_appearance_probability_sum",
        "gkp_team_budget_violation_count",
    ]:
        print("- %s: %s" % (key, validation.get(key)))
    print("- fallback_level_counts:", validation.get("fallback_level_counts"))
    print("- confidence_counts:", validation.get("confidence_counts"))
    print("- sample_tier_counts:", validation.get("sample_tier_counts"))
    print("- role_proxy_counts:", validation.get("role_proxy_counts"))
    print("- top_risk_window:", validation.get("top_risk_window"))
    print()
    print("Allowed uses:")
    for key, value in report["safety_contract"]["allowed_uses"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Blockers:", report["blockers"] or "none")
    print("Warnings:", report["warnings"] or "none")
    print("saved_json:", report["outputs"]["out_json"])
    print("saved_md:", report["outputs"]["out_md"])


def main() -> None:
    args = parse_args()
    args.as_of_time = parse_iso8601(
        args.as_of_time,
        "as_of_time",
    ).isoformat()

    blockers = validate_cli_policy(args)
    warnings: List[str] = []

    mode_result = resolve_prediction_mode(
        season=args.target_season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.source_seasons[-1],
        stabilization_gw=args.stabilization_gw,
        allow_experimental_mode=args.allow_experimental_mode,
    )
    if not mode_result.get("valid"):
        blockers.append(
            "Prediction mode resolver returned invalid: %s"
            % mode_result.get("errors")
        )
    resolved_mode = str(mode_result.get("resolved_prediction_mode"))
    if resolved_mode != "pre_gw1_prior":
        blockers.append(
            "Day72B v1 expects resolved_prediction_mode=pre_gw1_prior."
        )
    if args.target_gw != 1:
        blockers.append("Day72B v1 expects target_gw=1.")

    preview = load_preview(args.prediction_preview_csv)
    day72a_result = load_json_result(args.day72a_json, required=True)
    day71a_result = load_json_result(args.day71a_json, required=False)
    day71b_result = load_json_result(args.day71b_json, required=False)
    day70c_result = load_json_result(args.day70c_json, required=False)

    day72a_report = validate_source_report(
        "Day72A",
        day72a_result,
        required=True,
        blockers=blockers,
        warnings=warnings,
    )
    day71a_report = validate_source_report(
        "Day71A",
        day71a_result,
        required=False,
        blockers=blockers,
        warnings=warnings,
    )
    day71b_report = validate_source_report(
        "Day71B",
        day71b_result,
        required=False,
        blockers=blockers,
        warnings=warnings,
    )
    day70c_report = validate_source_report(
        "Day70C",
        day70c_result,
        required=False,
        blockers=blockers,
        warnings=warnings,
    )

    if day72a_report is not None:
        validate_day72a_report(
            args,
            day72a_report,
            resolved_mode,
            blockers,
        )
    validate_optional_source_reports(
        day71a_report,
        day71b_report,
        day70c_report,
        blockers,
    )

    if day72a_report is None:
        preview_validation: Dict[str, Any] = {
            "row_count": int(len(preview)),
            "missing_required_columns": [],
        }
    else:
        preview_validation = validate_preview(
            args,
            preview,
            day72a_report,
            resolved_mode,
            blockers,
            warnings,
        )

    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))

    report = build_manifest(
        args=args,
        preview=preview,
        mode_result=mode_result,
        day72a_result=day72a_result,
        day71a_result=day71a_result,
        day71b_result=day71b_result,
        day70c_result=day70c_result,
        preview_validation=preview_validation,
        blockers=blockers,
        warnings=warnings,
    )
    report["outputs"] = {
        "out_json": str(Path(args.out_json)),
        "out_md": str(Path(args.out_md)),
    }
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
