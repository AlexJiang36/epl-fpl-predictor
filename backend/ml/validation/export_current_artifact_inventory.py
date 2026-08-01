from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from ml.validation.resolve_prediction_mode import resolve_prediction_mode


INVENTORY_VERSION = "day73b_v1"
ARTIFACT_TYPE = "current_pre_gw1_artifact_dependency_inventory"
EXPECTED_PREDICTION_MODE = "pre_gw1_prior"

EXPECTED_MATCH_MODEL_NAME = "pre_gw1_match_prior_heuristic_v0"
EXPECTED_MATCH_MODEL_VERSION = "day70a_v0"
EXPECTED_MATCH_SCOPE = "read_only_preview"
EXPECTED_MATCH_PROBABILITY_TYPE = "heuristic_scaffolding"

EXPECTED_SCORELINE_MODEL_NAME = "pre_gw1_scoreline_prior_heuristic_v0"
EXPECTED_SCORELINE_MODEL_VERSION = "day70c_v0"
EXPECTED_SCORELINE_SCOPE = "read_only_scoreline_preview"
EXPECTED_SCORELINE_PROBABILITY_TYPE = "poisson_scoreline_scaffolding"

EXPECTED_PLAYER_FEATURE_VERSION = "day71a_v0"
EXPECTED_PLAYER_FEATURE_SCOPE = "pre_gw1_player_features"
EXPECTED_ROLE_CONTRACT_VERSION = "day71b_v1"
EXPECTED_THRESHOLD_POLICY_VERSION = "player_role_thresholds_v0"
EXPECTED_PLAYER_MODEL_NAME = "pre_gw1_player_prior_heuristic_v0"
EXPECTED_PLAYER_MODEL_VERSION = "day72a_v0_1"
EXPECTED_PLAYER_PREDICTION_SCOPE = (
    "read_only_pre_gw1_player_prediction_preview"
)
EXPECTED_PLAYER_CALIBRATION_STATUS = "not_calibrated_preview_only"

CURRENT_UNRESOLVED_DEPENDENCIES: Dict[str, List[Dict[str, Any]]] = {
    "rules": [
        {
            "name": "target_season_scoring_rules_registry",
            "satisfied": False,
            "status": "not_satisfied_at_day73b_checkpoint",
            "blocks": [
                "production player-points interpretation",
                "production prediction write",
            ],
        },
        {
            "name": "squad_and_transfer_rules_registry",
            "satisfied": False,
            "status": "not_satisfied_at_day73b_checkpoint",
            "blocks": [
                "opening squad optimizer",
                "transfer optimizer",
            ],
        },
        {
            "name": "chip_rules_registry",
            "satisfied": False,
            "status": "not_satisfied_at_day73b_checkpoint",
            "blocks": ["chip planner"],
        },
    ],
    "historical_evaluation": [
        {
            "name": "multi_season_eligibility_and_identity_audit",
            "satisfied": False,
            "status": "not_satisfied_at_day73b_checkpoint",
            "blocks": ["multi-season model training", "rolling backtests"],
        },
        {
            "name": "as_of_snapshot_and_leakage_validation",
            "satisfied": False,
            "status": "not_satisfied_at_day73b_checkpoint",
            "blocks": ["production model evidence", "model selection"],
        },
    ],
    "models": [
        {
            "name": "approved_match_model",
            "satisfied": False,
            "status": "current artifact is heuristic preview only",
            "blocks": ["public match prediction", "season simulation"],
        },
        {
            "name": "approved_minutes_model",
            "satisfied": False,
            "status": "current outputs are role/minutes proxies only",
            "blocks": ["trained player prediction"],
        },
        {
            "name": "approved_player_event_models",
            "satisfied": False,
            "status": "current outputs are heuristic components only",
            "blocks": ["trained player prediction"],
        },
        {
            "name": "approved_player_points_combiner",
            "satisfied": False,
            "status": "current combiner is heuristic scaffolding only",
            "blocks": ["public player prediction"],
        },
    ],
    "calibration": [
        {
            "name": "match_probability_calibration",
            "satisfied": False,
            "status": "not calibrated",
            "blocks": ["public match probability claims"],
        },
        {
            "name": "scoreline_calibration",
            "satisfied": False,
            "status": "not calibrated",
            "blocks": ["public scoreline probability claims"],
        },
        {
            "name": "minutes_and_start_calibration",
            "satisfied": False,
            "status": "not calibrated",
            "blocks": ["approved minutes stack"],
        },
        {
            "name": "player_points_calibration",
            "satisfied": False,
            "status": "not calibrated",
            "blocks": ["public player prediction"],
        },
    ],
    "publishing": [
        {
            "name": "active_model_registry_policy",
            "satisfied": False,
            "status": "not satisfied for the Pre-GW1 production path",
            "blocks": ["model activation"],
        },
        {
            "name": "prediction_publisher_dry_run",
            "satisfied": False,
            "status": "not implemented for this artifact stack",
            "blocks": ["database publication"],
        },
        {
            "name": "idempotent_prediction_writer",
            "satisfied": False,
            "status": "not implemented for this artifact stack",
            "blocks": ["database publication"],
        },
        {
            "name": "last_successful_activation_and_rollback",
            "satisfied": False,
            "status": "not implemented",
            "blocks": ["safe production activation"],
        },
    ],
    "decision_support": [
        {
            "name": "approved_positional_rankings",
            "satisfied": False,
            "status": "internal ranking preview only",
            "blocks": ["public ranking recommendation"],
        },
        {
            "name": "opening_squad_optimizer_approval",
            "satisfied": False,
            "status": "Day72B explicitly blocks optimizer input",
            "blocks": ["opening model squad"],
        },
        {
            "name": "transfer_and_captain_approval",
            "satisfied": False,
            "status": "Day72B explicitly blocks recommendation input",
            "blocks": ["transfer recommendation", "captain recommendation"],
        },
        {
            "name": "chip_planner_approval",
            "satisfied": False,
            "status": "not implemented",
            "blocks": ["chip recommendation"],
        },
    ],
    "operations": [
        {
            "name": "unified_refresh_dag",
            "satisfied": False,
            "status": "not implemented for the complete v2 architecture",
            "blocks": ["one-command full refresh"],
        },
        {
            "name": "persistent_artifact_storage_interface",
            "satisfied": False,
            "status": "current artifacts are local external CSVs and /tmp reports",
            "blocks": ["cloud artifact lifecycle"],
        },
        {
            "name": "cloud_scheduler_monitoring_and_alerts",
            "satisfied": False,
            "status": "not implemented",
            "blocks": ["production cloud operations"],
        },
    ],
}


@dataclass(frozen=True)
class CsvArtifactSpec:
    key: str
    filename_template: str
    day: str
    description: str
    required_columns: Tuple[str, ...]
    unique_columns: Tuple[str, ...] = ()
    expected_values: Tuple[Tuple[str, Any], ...] = ()
    false_columns: Tuple[str, ...] = ()
    true_columns: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TemporaryJsonSpec:
    key: str
    filename_template: str
    day: str
    description: str
    required: bool
    expected_values: Tuple[Tuple[Tuple[str, ...], Any], ...] = ()


@dataclass(frozen=True)
class TemporaryDocumentSpec:
    key: str
    filename_template: str
    day: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the Day73B current Pre-GW1 artifact and dependency "
            "inventory. The inventory validates persistent Day65-Day72 "
            "artifacts, validates the required current Day71-Day72 JSON "
            "reports, records optional historical /tmp evidence, checks "
            "cross-artifact consistency and SHA256 provenance, and lists "
            "the dependencies that remain blocked. It is read-only and "
            "does not write to the database."
        )
    )
    parser.add_argument("--source-season", required=True)
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
        "--artifact-root",
        required=True,
        help=(
            "Persistent external dataset directory containing the "
            "Day65-Day72 CSV artifacts."
        ),
    )
    parser.add_argument(
        "--temporary-root",
        default="/tmp",
        help=(
            "Temporary report directory. Day71A-Day72B JSON reports are "
            "required for this checkpoint; earlier reports are recorded "
            "when present but are not required."
        ),
    )
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
        raise ValueError("%s must be ISO-8601: %s" % (label, exc))
    if parsed.tzinfo is None:
        raise ValueError("%s must include a timezone offset." % label)
    return parsed


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def sha256_file(path_value: Path) -> Optional[str]:
    if not path_value.exists() or not path_value.is_file():
        return None
    digest = hashlib.sha256()
    with path_value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_values(
    frame: pd.DataFrame,
    column: str,
    limit: int = 30,
) -> List[Any]:
    if column not in frame.columns:
        return []
    values = [
        normalize_scalar(value)
        for value in frame[column].drop_duplicates().tolist()
    ]
    values = [value for value in values if value is not None]
    values = sorted(values, key=lambda item: str(item))
    if len(values) > limit:
        return values[:limit] + ["<truncated:%s>" % (len(values) - limit)]
    return values


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


def resolved_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    return str(Path(text_value).expanduser().resolve())


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool_value(actual) is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(float(actual)) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-9
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def format_template(
    template: str,
    source_season: str,
    target_season: str,
) -> str:
    return template.format(
        source_season=source_season,
        target_season=target_season,
    )


def persistent_specs(args: argparse.Namespace) -> List[CsvArtifactSpec]:
    common_values = (
        ("source_season", args.source_season),
        ("target_season", args.target_season),
    )
    gw_values = common_values + (
        ("target_gw", args.target_gw),
        ("prediction_mode", EXPECTED_PREDICTION_MODE),
    )

    return [
        CsvArtifactSpec(
            key="team_priors_csv",
            filename_template=(
                "team_priors_{source_season}_to_"
                "{target_season}_from_staging.csv"
            ),
            day="Day68A",
            description="Historical staging team priors",
            required_columns=(
                "source_season",
                "target_season",
                "raw_team_id",
                "raw_team_name",
                "raw_team_short_name",
                "mapping_status",
                "prior_identity_scope",
                "prior_source",
                "has_prev_season_data",
            ),
            unique_columns=("raw_team_id",),
            expected_values=common_values + (
                ("prior_identity_scope", "historical_raw_team_id"),
                ("prior_source", "historical_staging"),
            ),
        ),
        CsvArtifactSpec(
            key="team_mapping_csv",
            filename_template=(
                "team_identity_mapping_candidates_"
                "{source_season}_to_{target_season}.csv"
            ),
            day="Day68B",
            description="Historical-to-target team mapping candidates",
            required_columns=(
                "source_season",
                "target_season",
                "raw_team_id",
                "raw_team_name",
                "raw_team_short_name",
                "candidate_rank",
                "candidate_team_id",
                "candidate_team_name",
                "candidate_team_short_name",
                "match_status",
                "is_auto_approved",
                "needs_manual_review",
                "safe_team_match_for_auto_approval",
            ),
            expected_values=common_values,
        ),
        CsvArtifactSpec(
            key="match_features_csv",
            filename_template=(
                "pre_gw1_match_features_{source_season}_to_"
                "{target_season}.csv"
            ),
            day="Day69A",
            description="Pre-GW1 match feature artifact",
            required_columns=(
                "source_season",
                "target_season",
                "target_gw",
                "prediction_mode",
                "fixture_id",
                "home_team_id",
                "away_team_id",
                "home_team_short_name",
                "away_team_short_name",
            ),
            unique_columns=("fixture_id",),
            expected_values=gw_values,
        ),
        CsvArtifactSpec(
            key="match_features_with_fallback_csv",
            filename_template=(
                "pre_gw1_match_features_with_fallback_"
                "{source_season}_to_{target_season}.csv"
            ),
            day="Day69B",
            description="Pre-GW1 match features with promoted-team fallback",
            required_columns=(
                "source_season",
                "target_season",
                "target_gw",
                "prediction_mode",
                "fixture_id",
                "home_team_id",
                "away_team_id",
                "home_team_fallback_applied",
                "away_team_fallback_applied",
                "any_team_fallback_applied",
            ),
            unique_columns=("fixture_id",),
            expected_values=gw_values,
        ),
        CsvArtifactSpec(
            key="match_prediction_preview_csv",
            filename_template=(
                "pre_gw1_match_prediction_preview_"
                "{source_season}_to_{target_season}.csv"
            ),
            day="Day70A",
            description="Read-only Pre-GW1 1X2 prediction preview",
            required_columns=(
                "source_season",
                "target_season",
                "target_gw",
                "prediction_mode",
                "prediction_scope",
                "probability_type",
                "model_name",
                "model_version",
                "fixture_id",
                "home_win_probability",
                "draw_probability",
                "away_win_probability",
                "predicted_result_label",
                "prediction_write_allowed",
                "production_ready",
                "requires_manifest_before_write",
                "calibration_status",
                "guardrail_status",
            ),
            unique_columns=("fixture_id",),
            expected_values=gw_values + (
                ("model_name", EXPECTED_MATCH_MODEL_NAME),
                ("model_version", EXPECTED_MATCH_MODEL_VERSION),
                ("prediction_scope", EXPECTED_MATCH_SCOPE),
                ("probability_type", EXPECTED_MATCH_PROBABILITY_TYPE),
            ),
            false_columns=(
                "prediction_write_allowed",
                "production_ready",
            ),
            true_columns=("requires_manifest_before_write",),
        ),
        CsvArtifactSpec(
            key="scoreline_preview_csv",
            filename_template=(
                "pre_gw1_scoreline_preview_"
                "{source_season}_to_{target_season}.csv"
            ),
            day="Day70C",
            description="Read-only Pre-GW1 scoreline preview",
            required_columns=(
                "source_season",
                "target_season",
                "target_gw",
                "prediction_mode",
                "prediction_scope",
                "probability_type",
                "model_name",
                "model_version",
                "fixture_id",
                "expected_home_goals",
                "expected_away_goals",
                "top_1_scoreline",
                "scoreline_home_win_probability",
                "scoreline_draw_probability",
                "scoreline_away_win_probability",
                "prediction_write_allowed",
                "production_ready",
                "requires_scoreline_manifest_before_write",
                "calibration_status",
                "guardrail_status",
            ),
            unique_columns=("fixture_id",),
            expected_values=gw_values + (
                ("model_name", EXPECTED_SCORELINE_MODEL_NAME),
                ("model_version", EXPECTED_SCORELINE_MODEL_VERSION),
                ("prediction_scope", EXPECTED_SCORELINE_SCOPE),
                ("probability_type", EXPECTED_SCORELINE_PROBABILITY_TYPE),
            ),
            false_columns=(
                "prediction_write_allowed",
                "production_ready",
            ),
            true_columns=("requires_scoreline_manifest_before_write",),
        ),
        CsvArtifactSpec(
            key="player_priors_csv",
            filename_template=(
                "player_priors_{source_season}_to_"
                "{target_season}_from_staging.csv"
            ),
            day="Day65",
            description="Historical staging player priors",
            required_columns=(
                "source_season",
                "target_season",
                "raw_player_id",
                "raw_player_name",
                "raw_team_id",
                "raw_position",
                "prev_season_minutes",
                "prev_season_appearances",
                "prev_season_starts_proxy",
                "prev_season_total_points",
                "prev_season_points_per90",
                "prev_season_goals",
                "prev_season_assists",
                "prev_season_bonus",
                "prior_identity_scope",
                "prior_source",
            ),
            unique_columns=("raw_player_id",),
            expected_values=common_values + (
                ("prior_identity_scope", "historical_raw_player_id"),
                ("prior_source", "historical_staging"),
            ),
        ),
        CsvArtifactSpec(
            key="player_mapping_csv",
            filename_template=(
                "player_identity_mapping_candidates_"
                "{source_season}_to_{target_season}.csv"
            ),
            day="Day66B",
            description="Historical-to-target player mapping candidates",
            required_columns=(
                "source_season",
                "target_season",
                "raw_player_id",
                "raw_player_name",
                "candidate_rank",
                "candidate_player_id",
                "match_status",
                "is_auto_approved",
            ),
            expected_values=common_values,
        ),
        CsvArtifactSpec(
            key="player_features_csv",
            filename_template=(
                "pre_gw1_player_features_"
                "{source_season}_to_{target_season}.csv"
            ),
            day="Day71A",
            description="Read-only Pre-GW1 player features",
            required_columns=(
                "source_season",
                "target_season",
                "target_gw",
                "prediction_mode",
                "feature_scope",
                "feature_version",
                "player_id",
                "fpl_player_id",
                "team_id",
                "position",
                "has_fixture",
                "has_prev_season_player_prior",
                "no_prior_flag",
                "prediction_write_allowed",
                "production_ready",
                "requires_player_feature_manifest_before_prediction",
            ),
            unique_columns=("player_id", "fpl_player_id"),
            expected_values=gw_values + (
                ("feature_scope", EXPECTED_PLAYER_FEATURE_SCOPE),
                ("feature_version", EXPECTED_PLAYER_FEATURE_VERSION),
            ),
            false_columns=(
                "prediction_write_allowed",
                "production_ready",
            ),
            true_columns=(
                "requires_player_feature_manifest_before_prediction",
            ),
        ),
        CsvArtifactSpec(
            key="player_prediction_preview_csv",
            filename_template=(
                "pre_gw1_player_prediction_preview_"
                "{source_season}_to_{target_season}.csv"
            ),
            day="Day72A",
            description="Read-only Pre-GW1 player-points preview",
            required_columns=(
                "source_seasons",
                "target_season",
                "target_gw",
                "as_of_time",
                "prediction_mode",
                "prediction_scope",
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
                "team_id",
                "position",
                "fixture_id",
                "has_fixture",
                "appearance_probability",
                "start_probability",
                "expected_minutes",
                "raw_expected_points",
                "final_predicted_points",
                "predicted_points",
                "fallback_level",
                "prediction_confidence",
                "prediction_write_allowed",
                "production_ready",
                "requires_player_prediction_manifest_before_write",
                "calibration_status",
                "guardrail_status",
                "component_accounting_status",
            ),
            unique_columns=("player_id", "fpl_player_id"),
            expected_values=(
                ("target_season", args.target_season),
                ("target_gw", args.target_gw),
                ("prediction_mode", EXPECTED_PREDICTION_MODE),
                ("prediction_scope", EXPECTED_PLAYER_PREDICTION_SCOPE),
                ("model_name", EXPECTED_PLAYER_MODEL_NAME),
                ("model_version", EXPECTED_PLAYER_MODEL_VERSION),
                ("player_feature_version", EXPECTED_PLAYER_FEATURE_VERSION),
                ("role_contract_version", EXPECTED_ROLE_CONTRACT_VERSION),
                (
                    "threshold_policy_version",
                    EXPECTED_THRESHOLD_POLICY_VERSION,
                ),
                ("scoreline_model_name", EXPECTED_SCORELINE_MODEL_NAME),
                ("scoreline_model_version", EXPECTED_SCORELINE_MODEL_VERSION),
                (
                    "calibration_status",
                    EXPECTED_PLAYER_CALIBRATION_STATUS,
                ),
            ),
            false_columns=(
                "prediction_write_allowed",
                "production_ready",
            ),
            true_columns=(
                "requires_player_prediction_manifest_before_write",
            ),
        ),
    ]


def temporary_json_specs() -> List[TemporaryJsonSpec]:
    return [
        TemporaryJsonSpec(
            key="day68a_team_priors_json",
            filename_template=(
                "team_priors_{source_season}_to_"
                "{target_season}_from_staging.json"
            ),
            day="Day68A",
            description="Team-prior audit report",
            required=False,
        ),
        TemporaryJsonSpec(
            key="day68b_team_mapping_json",
            filename_template=(
                "team_identity_mapping_candidates_"
                "{source_season}_to_{target_season}.json"
            ),
            day="Day68B",
            description="Team-mapping audit report",
            required=False,
        ),
        TemporaryJsonSpec(
            key="day69a_match_features_json",
            filename_template="day69a_pre_gw1_match_features.json",
            day="Day69A",
            description="Match-feature audit report",
            required=False,
        ),
        TemporaryJsonSpec(
            key="day69b_fallback_json",
            filename_template="day69b_promoted_team_fallback_policy.json",
            day="Day69B",
            description="Promoted-team fallback report",
            required=False,
        ),
        TemporaryJsonSpec(
            key="day70a_match_preview_json",
            filename_template="day70a_pre_gw1_match_prediction_preview.json",
            day="Day70A",
            description="1X2 preview report",
            required=False,
        ),
        TemporaryJsonSpec(
            key="day70b_match_manifest_json",
            filename_template="day70b_pre_gw1_match_prediction_manifest.json",
            day="Day70B",
            description="1X2 manifest",
            required=False,
        ),
        TemporaryJsonSpec(
            key="day70c_scoreline_json",
            filename_template="day70c_pre_gw1_scoreline_preview.json",
            day="Day70C",
            description="Scoreline preview report",
            required=False,
        ),
        TemporaryJsonSpec(
            key="day70d_scoreline_manifest_json",
            filename_template="day70d_pre_gw1_scoreline_manifest.json",
            day="Day70D",
            description="Scoreline manifest",
            required=False,
        ),
        TemporaryJsonSpec(
            key="day71a_player_features_json",
            filename_template="day71a_pre_gw1_player_features.json",
            day="Day71A",
            description="Player-feature report",
            required=True,
            expected_values=(
                (("passed",), True),
                (("audit_only",), True),
                (("writes_database",), False),
                (("uses_current_season_actual_player_gw_stats",), False),
                (("feature_version",), EXPECTED_PLAYER_FEATURE_VERSION),
                (("feature_scope",), EXPECTED_PLAYER_FEATURE_SCOPE),
                (("ready_for_pre_gw1_player_features",), True),
                (("ready_for_prediction_write",), False),
            ),
        ),
        TemporaryJsonSpec(
            key="day71b_role_contract_json",
            filename_template="day71b_player_role_feature_contract.json",
            day="Day71B",
            description="Player role/minutes contract",
            required=True,
            expected_values=(
                (("passed",), True),
                (("audit_only",), True),
                (("writes_database",), False),
                (("contract_version",), EXPECTED_ROLE_CONTRACT_VERSION),
                (
                    ("threshold_policy_version",),
                    EXPECTED_THRESHOLD_POLICY_VERSION,
                ),
                (("ready_for_role_feature_contract",), True),
                (
                    ("ready_for_day72a_prediction_preview_design",),
                    True,
                ),
                (("ready_for_trained_minutes_prediction",), False),
                (("ready_for_player_points_prediction",), False),
                (("ready_for_production_write",), False),
            ),
        ),
        TemporaryJsonSpec(
            key="day72a_player_preview_json",
            filename_template=(
                "day72a_pre_gw1_player_prediction_preview.json"
            ),
            day="Day72A",
            description="Player prediction preview report",
            required=True,
            expected_values=(
                (("passed",), True),
                (("audit_only",), True),
                (("writes_database",), False),
                (("uses_current_season_actual_player_gw_stats",), False),
                (("model_name",), EXPECTED_PLAYER_MODEL_NAME),
                (("model_version",), EXPECTED_PLAYER_MODEL_VERSION),
                (("prediction_scope",), EXPECTED_PLAYER_PREDICTION_SCOPE),
                (("player_feature_version",), EXPECTED_PLAYER_FEATURE_VERSION),
                (("role_contract_version",), EXPECTED_ROLE_CONTRACT_VERSION),
                (
                    ("threshold_policy_version",),
                    EXPECTED_THRESHOLD_POLICY_VERSION,
                ),
                (("ready_for_pre_gw1_player_prediction_preview",), True),
                (("ready_for_day72b_player_prediction_manifest",), True),
                (("ready_for_trained_minutes_prediction",), False),
                (("ready_for_calibrated_player_points_prediction",), False),
                (("ready_for_production_write",), False),
            ),
        ),
        TemporaryJsonSpec(
            key="day72b_player_manifest_json",
            filename_template=(
                "day72b_pre_gw1_player_prediction_manifest.json"
            ),
            day="Day72B",
            description="Player prediction manifest and safety contract",
            required=True,
            expected_values=(
                (("passed",), True),
                (("audit_only",), True),
                (("writes_database",), False),
                (("manifest_version",), "day72b_v1"),
                (("model_name",), EXPECTED_PLAYER_MODEL_NAME),
                (("model_version",), EXPECTED_PLAYER_MODEL_VERSION),
                (("ready_for_pre_gw1_player_prediction_manifest",), True),
                (("ready_for_ops_preview",), True),
                (("ready_for_internal_player_ranking_preview",), True),
                (("ready_for_master_plan_v2_checkpoint",), True),
                (("ready_for_public_prediction",), False),
                (("ready_for_opening_squad_optimizer_input",), False),
                (("ready_for_prediction_write",), False),
            ),
        ),
        TemporaryJsonSpec(
            key="day66b_player_mapping_json",
            filename_template=(
                "player_identity_mapping_candidates_"
                "{source_season}_to_{target_season}.json"
            ),
            day="Day66B",
            description="Player mapping audit report",
            required=False,
        ),
    ]


def temporary_document_specs() -> List[TemporaryDocumentSpec]:
    filenames = [
        (
            "day71a_player_features_md",
            "day71a_pre_gw1_player_features.md",
            "Day71A",
            "Player-feature Markdown report",
        ),
        (
            "day71b_role_contract_md",
            "day71b_player_role_feature_contract.md",
            "Day71B",
            "Role/minutes Markdown contract",
        ),
        (
            "day72a_player_preview_md",
            "day72a_pre_gw1_player_prediction_preview.md",
            "Day72A",
            "Player preview Markdown report",
        ),
        (
            "day72b_player_manifest_md",
            "day72b_pre_gw1_player_prediction_manifest.md",
            "Day72B",
            "Player manifest Markdown report",
        ),
        (
            "day68a_team_priors_md",
            (
                "team_priors_{source_season}_to_"
                "{target_season}_from_staging.md"
            ),
            "Day68A",
            "Team-prior Markdown report",
        ),
        (
            "day68b_team_mapping_md",
            (
                "team_identity_mapping_candidates_"
                "{source_season}_to_{target_season}.md"
            ),
            "Day68B",
            "Team-mapping Markdown report",
        ),
        (
            "day70b_match_manifest_md",
            "day70b_pre_gw1_match_prediction_manifest.md",
            "Day70B",
            "1X2 manifest Markdown report",
        ),
        (
            "day70d_scoreline_manifest_md",
            "day70d_pre_gw1_scoreline_manifest.md",
            "Day70D",
            "Scoreline manifest Markdown report",
        ),
    ]
    return [
        TemporaryDocumentSpec(
            key=key,
            filename_template=filename,
            day=day,
            description=description,
        )
        for key, filename, day, description in filenames
    ]


def validate_args(args: argparse.Namespace) -> List[str]:
    blockers: List[str] = []
    try:
        parse_iso8601(args.as_of_time, "as_of_time")
    except ValueError as exc:
        blockers.append(str(exc))
    if args.target_gw < 1:
        blockers.append("target_gw must be >= 1.")
    if not args.source_season.strip():
        blockers.append("source_season must not be empty.")
    if not args.target_season.strip():
        blockers.append("target_season must not be empty.")
    if not Path(args.artifact_root).expanduser().exists():
        blockers.append(
            "artifact_root does not exist: %s" % args.artifact_root
        )
    return blockers


def inspect_csv(
    spec: CsvArtifactSpec,
    root: Path,
    args: argparse.Namespace,
    blockers: List[str],
    warnings: List[str],
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    filename = format_template(
        spec.filename_template,
        args.source_season,
        args.target_season,
    )
    path = root / filename
    result: Dict[str, Any] = {
        "key": spec.key,
        "day": spec.day,
        "description": spec.description,
        "storage_class": "persistent_external_dataset",
        "required": True,
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path),
        "row_count": None,
        "column_count": None,
        "columns": [],
        "missing_required_columns": [],
        "metadata_values": {},
        "unique_key_validation": {},
        "safety_flag_validation": {},
        "passed": False,
        "error": None,
    }
    if not path.exists():
        blockers.append(
            "Required persistent artifact is missing: %s (%s)."
            % (spec.key, path)
        )
        result["error"] = "file_not_found"
        return result, None

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        blockers.append(
            "Required persistent artifact could not be read: %s: %s"
            % (spec.key, exc)
        )
        result["error"] = str(exc)
        return result, None

    result["readable"] = True
    result["row_count"] = int(len(frame))
    result["column_count"] = int(len(frame.columns))
    result["columns"] = [str(column) for column in frame.columns]

    if frame.empty:
        blockers.append("Required persistent artifact is empty: %s." % spec.key)

    missing_columns = [
        column
        for column in spec.required_columns
        if column not in frame.columns
    ]
    result["missing_required_columns"] = missing_columns
    if missing_columns:
        blockers.append(
            "%s is missing required columns: %s."
            % (spec.key, missing_columns)
        )

    expected_failures: List[Dict[str, Any]] = []
    for column, expected in spec.expected_values:
        if column not in frame.columns:
            continue
        observed = unique_values(frame, column)
        result["metadata_values"][column] = observed
        if len(observed) != 1 or not values_match(observed[0], expected):
            expected_failures.append(
                {
                    "column": column,
                    "expected": expected,
                    "observed": observed,
                }
            )
            blockers.append(
                "%s metadata mismatch for %s: expected %s, got %s."
                % (spec.key, column, expected, observed)
            )
    result["metadata_value_failures"] = expected_failures

    for column in spec.unique_columns:
        if column not in frame.columns:
            continue
        missing_count = int(frame[column].isna().sum())
        duplicate_count = int(
            frame[column].dropna().duplicated(keep=False).sum()
        )
        result["unique_key_validation"][column] = {
            "missing_count": missing_count,
            "duplicate_count": duplicate_count,
            "unique_count": int(frame[column].nunique(dropna=True)),
        }
        if missing_count:
            blockers.append(
                "%s contains %s missing %s values."
                % (spec.key, missing_count, column)
            )
        if duplicate_count:
            blockers.append(
                "%s contains %s duplicated %s rows."
                % (spec.key, duplicate_count, column)
            )

    for column in spec.false_columns:
        if column not in frame.columns:
            continue
        true_count = int(frame[column].apply(bool_value).sum())
        result["safety_flag_validation"][column] = {
            "expected": False,
            "true_count": true_count,
        }
        if true_count:
            blockers.append(
                "%s requires %s=False for all rows; found %s true rows."
                % (spec.key, column, true_count)
            )

    for column in spec.true_columns:
        if column not in frame.columns:
            continue
        false_count = int((~frame[column].apply(bool_value)).sum())
        result["safety_flag_validation"][column] = {
            "expected": True,
            "false_count": false_count,
        }
        if false_count:
            blockers.append(
                "%s requires %s=True for all rows; found %s false rows."
                % (spec.key, column, false_count)
            )

    if spec.key == "player_prediction_preview_csv":
        if "source_seasons" in frame.columns:
            observed_sources = set()
            for value in frame["source_seasons"].dropna().astype(str):
                for token in value.split(","):
                    token = token.strip()
                    if token:
                        observed_sources.add(token)
            result["metadata_values"]["source_seasons_normalized"] = sorted(
                observed_sources
            )
            if observed_sources != {args.source_season}:
                blockers.append(
                    "player_prediction_preview_csv source_seasons expected "
                    "[%s], got %s."
                    % (args.source_season, sorted(observed_sources))
                )

        if "as_of_time" in frame.columns:
            observed_as_of = unique_values(frame, "as_of_time")
            result["metadata_values"]["as_of_time"] = observed_as_of
            expected_as_of = parse_iso8601(
                args.as_of_time,
                "as_of_time",
            ).isoformat()
            if len(observed_as_of) != 1:
                blockers.append(
                    "player_prediction_preview_csv must contain one "
                    "as_of_time value."
                )
            else:
                try:
                    actual_as_of = parse_iso8601(
                        str(observed_as_of[0]),
                        "preview as_of_time",
                    ).isoformat()
                except ValueError:
                    actual_as_of = None
                if actual_as_of != expected_as_of:
                    blockers.append(
                        "player_prediction_preview_csv as_of_time expected "
                        "%s, got %s."
                        % (expected_as_of, observed_as_of)
                    )

        if {
            "appearance_probability",
            "start_probability",
        }.issubset(frame.columns):
            appearance = pd.to_numeric(
                frame["appearance_probability"],
                errors="coerce",
            )
            start = pd.to_numeric(
                frame["start_probability"],
                errors="coerce",
            )
            start_exceeds = int((start > appearance + 1e-9).sum())
            result["start_exceeds_appearance_count"] = start_exceeds
            if start_exceeds:
                blockers.append(
                    "player_prediction_preview_csv has %s rows where "
                    "start_probability exceeds appearance_probability."
                    % start_exceeds
                )

    result["passed"] = (
        result["readable"]
        and bool(result["row_count"])
        and not missing_columns
        and not expected_failures
        and not result["error"]
        and not any(
            validation.get("missing_count", 0)
            or validation.get("duplicate_count", 0)
            for validation in result["unique_key_validation"].values()
        )
        and not any(
            validation.get("true_count", 0)
            or validation.get("false_count", 0)
            for validation in result["safety_flag_validation"].values()
        )
    )
    return result, frame


def inspect_temporary_json(
    spec: TemporaryJsonSpec,
    root: Path,
    args: argparse.Namespace,
    blockers: List[str],
    warnings: List[str],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    filename = format_template(
        spec.filename_template,
        args.source_season,
        args.target_season,
    )
    path = root / filename
    result: Dict[str, Any] = {
        "key": spec.key,
        "day": spec.day,
        "description": spec.description,
        "storage_class": "temporary_report_json",
        "required": spec.required,
        "path": str(path),
        "exists": path.exists(),
        "loaded": False,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path),
        "expected_value_failures": [],
        "summary": {},
        "passed": False,
        "error": None,
    }

    if not path.exists():
        result["error"] = "file_not_found"
        if spec.required:
            blockers.append(
                "Required current temporary JSON is missing: %s (%s)."
                % (spec.key, path)
            )
        else:
            warnings.append(
                "Optional historical temporary JSON is unavailable: %s."
                % spec.key
            )
        return result, None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["error"] = str(exc)
        if spec.required:
            blockers.append(
                "Required current temporary JSON could not be loaded: "
                "%s: %s" % (spec.key, exc)
            )
        else:
            warnings.append(
                "Optional historical temporary JSON could not be loaded: "
                "%s: %s" % (spec.key, exc)
            )
        return result, None

    result["loaded"] = True
    summary_keys = [
        "artifact_type",
        "created_at",
        "source_season",
        "source_seasons",
        "target_season",
        "target_gw",
        "resolved_prediction_mode",
        "feature_version",
        "feature_scope",
        "contract_version",
        "threshold_policy_version",
        "model_name",
        "model_version",
        "manifest_version",
        "prediction_scope",
        "passed",
        "audit_only",
        "writes_database",
    ]
    result["summary"] = {
        key: data.get(key)
        for key in summary_keys
        if key in data
    }

    failures: List[Dict[str, Any]] = []
    for path_keys, expected in spec.expected_values:
        actual = nested_get(data, path_keys, None)
        if not values_match(actual, expected):
            failure = {
                "path": ".".join(path_keys),
                "expected": expected,
                "actual": actual,
            }
            failures.append(failure)
            blockers.append(
                "%s expected %s=%s, got %s."
                % (
                    spec.key,
                    ".".join(path_keys),
                    expected,
                    actual,
                )
            )
    result["expected_value_failures"] = failures

    if spec.key == "day72b_player_manifest_json":
        manifest_blockers = data.get("blockers") or []
        result["summary"]["blocker_count"] = len(manifest_blockers)
        if manifest_blockers:
            blockers.append(
                "Day72B manifest still contains blockers: %s."
                % manifest_blockers
            )

    result["passed"] = result["loaded"] and not failures
    return result, data


def inspect_temporary_document(
    spec: TemporaryDocumentSpec,
    root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    filename = format_template(
        spec.filename_template,
        args.source_season,
        args.target_season,
    )
    path = root / filename
    return {
        "key": spec.key,
        "day": spec.day,
        "description": spec.description,
        "storage_class": "temporary_human_readable_report",
        "required_for_machine_validation": False,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path),
    }


def top_candidate_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if "candidate_rank" not in frame.columns:
        return frame.iloc[0:0].copy()
    rank = pd.to_numeric(frame["candidate_rank"], errors="coerce")
    return frame[rank.isna() | (rank == 1)].copy()


def set_from_column(
    frame: Optional[pd.DataFrame],
    column: str,
) -> Set[str]:
    if frame is None or column not in frame.columns:
        return set()
    return set(
        frame[column]
        .dropna()
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .tolist()
    )


def add_cross_check(
    checks: List[Dict[str, Any]],
    name: str,
    passed: bool,
    expected: Any,
    actual: Any,
    blockers: List[str],
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "expected": expected,
            "actual": actual,
        }
    )
    if not passed:
        blockers.append(
            "Cross-artifact check failed: %s; expected %s, got %s."
            % (name, expected, actual)
        )


def build_cross_artifact_validation(
    frames: Dict[str, Optional[pd.DataFrame]],
    blockers: List[str],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    team_priors = frames.get("team_priors_csv")
    team_mapping = frames.get("team_mapping_csv")
    if team_priors is not None and team_mapping is not None:
        top_team_mapping = top_candidate_rows(team_mapping)
        add_cross_check(
            checks,
            "team_prior_rows_equal_top_mapping_rows",
            len(team_priors) == len(top_team_mapping),
            int(len(team_priors)),
            int(len(top_team_mapping)),
            blockers,
        )
        top_raw_team_duplicates = int(
            top_team_mapping["raw_team_id"]
            .dropna()
            .astype(str)
            .duplicated(keep=False)
            .sum()
        )
        add_cross_check(
            checks,
            "top_team_mapping_raw_team_id_unique",
            top_raw_team_duplicates == 0,
            0,
            top_raw_team_duplicates,
            blockers,
        )

    player_priors = frames.get("player_priors_csv")
    player_mapping = frames.get("player_mapping_csv")
    if player_priors is not None and player_mapping is not None:
        top_player_mapping = top_candidate_rows(player_mapping)
        add_cross_check(
            checks,
            "player_prior_rows_equal_top_mapping_rows",
            len(player_priors) == len(top_player_mapping),
            int(len(player_priors)),
            int(len(top_player_mapping)),
            blockers,
        )
        top_raw_player_duplicates = int(
            top_player_mapping["raw_player_id"]
            .dropna()
            .astype(str)
            .duplicated(keep=False)
            .sum()
        )
        add_cross_check(
            checks,
            "top_player_mapping_raw_player_id_unique",
            top_raw_player_duplicates == 0,
            0,
            top_raw_player_duplicates,
            blockers,
        )

    match_keys = [
        "match_features_csv",
        "match_features_with_fallback_csv",
        "match_prediction_preview_csv",
        "scoreline_preview_csv",
    ]
    match_frames = {
        key: frames.get(key)
        for key in match_keys
        if frames.get(key) is not None
    }
    if len(match_frames) == len(match_keys):
        row_counts = {
            key: int(len(frame))
            for key, frame in match_frames.items()
            if frame is not None
        }
        count_values = set(row_counts.values())
        add_cross_check(
            checks,
            "match_artifact_row_counts_equal",
            len(count_values) == 1,
            "one shared row count",
            row_counts,
            blockers,
        )

        fixture_sets = {
            key: set_from_column(frame, "fixture_id")
            for key, frame in match_frames.items()
        }
        fixture_set_values = list(fixture_sets.values())
        fixture_sets_equal = all(
            value == fixture_set_values[0]
            for value in fixture_set_values[1:]
        )
        add_cross_check(
            checks,
            "match_artifact_fixture_sets_equal",
            fixture_sets_equal,
            sorted(fixture_set_values[0]),
            {
                key: sorted(value)
                for key, value in fixture_sets.items()
            },
            blockers,
        )

    player_features = frames.get("player_features_csv")
    player_preview = frames.get("player_prediction_preview_csv")
    if player_features is not None and player_preview is not None:
        add_cross_check(
            checks,
            "player_feature_and_preview_row_counts_equal",
            len(player_features) == len(player_preview),
            int(len(player_features)),
            int(len(player_preview)),
            blockers,
        )
        feature_player_ids = set_from_column(
            player_features,
            "player_id",
        )
        preview_player_ids = set_from_column(
            player_preview,
            "player_id",
        )
        add_cross_check(
            checks,
            "player_feature_and_preview_player_sets_equal",
            feature_player_ids == preview_player_ids,
            "identical player_id sets",
            {
                "feature_only_count": len(
                    feature_player_ids - preview_player_ids
                ),
                "preview_only_count": len(
                    preview_player_ids - feature_player_ids
                ),
            },
            blockers,
        )

    match_preview = frames.get("match_prediction_preview_csv")
    if player_preview is not None and match_preview is not None:
        match_fixture_ids = set_from_column(match_preview, "fixture_id")
        player_fixture_ids = set_from_column(
            player_preview[
                player_preview["has_fixture"].apply(bool_value)
            ],
            "fixture_id",
        )
        add_cross_check(
            checks,
            "player_preview_fixture_ids_within_match_preview",
            player_fixture_ids.issubset(match_fixture_ids),
            "player fixture IDs subset of match preview fixture IDs",
            sorted(player_fixture_ids - match_fixture_ids),
            blockers,
        )

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def check_fingerprint(
    checks: List[Dict[str, Any]],
    name: str,
    recorded_sha256: Any,
    actual_path: Path,
    required: bool,
    blockers: List[str],
    warnings: List[str],
) -> None:
    actual_sha256 = sha256_file(actual_path)
    if recorded_sha256 is None:
        checks.append(
            {
                "name": name,
                "passed": not required,
                "required": required,
                "recorded_sha256": None,
                "actual_sha256": actual_sha256,
                "path": str(actual_path),
                "status": "recorded_fingerprint_missing",
            }
        )
        message = (
            "Recorded fingerprint is missing for %s." % name
        )
        if required:
            blockers.append(message)
        else:
            warnings.append(message)
        return

    passed = (
        actual_sha256 is not None
        and str(recorded_sha256) == str(actual_sha256)
    )
    checks.append(
        {
            "name": name,
            "passed": passed,
            "required": required,
            "recorded_sha256": recorded_sha256,
            "actual_sha256": actual_sha256,
            "path": str(actual_path),
            "status": "matched" if passed else "mismatch",
        }
    )
    if not passed:
        message = (
            "Fingerprint mismatch for %s: recorded=%s actual=%s."
            % (name, recorded_sha256, actual_sha256)
        )
        if required:
            blockers.append(message)
        else:
            warnings.append(message)


def build_report_consistency_validation(
    reports: Dict[str, Optional[Dict[str, Any]]],
    persistent_paths: Dict[str, Path],
    temporary_paths: Dict[str, Path],
    blockers: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    day71a = reports.get("day71a_player_features_json")
    if day71a is not None:
        expected_paths = {
            "day71a_input_player_prior_path": (
                nested_get(day71a, ["inputs", "player_prior_csv"]),
                persistent_paths["player_priors_csv"],
            ),
            "day71a_input_player_mapping_path": (
                nested_get(day71a, ["inputs", "player_mapping_csv"]),
                persistent_paths["player_mapping_csv"],
            ),
            "day71a_input_match_features_path": (
                nested_get(day71a, ["inputs", "match_features_csv"]),
                persistent_paths["match_features_with_fallback_csv"],
            ),
        }
        for name, pair in expected_paths.items():
            recorded, actual_path = pair
            passed = (
                resolved_path(recorded)
                == str(actual_path.expanduser().resolve())
            )
            checks.append(
                {
                    "name": name,
                    "passed": passed,
                    "recorded_path": recorded,
                    "actual_path": str(actual_path),
                }
            )
            if not passed:
                blockers.append(
                    "%s does not match the current artifact path." % name
                )

    day71b = reports.get("day71b_role_contract_json")
    if day71b is not None:
        check_fingerprint(
            checks,
            "day71b_player_features_csv",
            nested_get(
                day71b,
                ["inputs", "player_features_csv", "sha256"],
            ),
            persistent_paths["player_features_csv"],
            True,
            blockers,
            warnings,
        )
        check_fingerprint(
            checks,
            "day71b_day71a_json",
            nested_get(day71b, ["inputs", "day71a_json", "sha256"]),
            temporary_paths["day71a_player_features_json"],
            True,
            blockers,
            warnings,
        )

    day72a = reports.get("day72a_player_preview_json")
    if day72a is not None:
        check_fingerprint(
            checks,
            "day72a_player_features_csv",
            nested_get(
                day72a,
                ["inputs", "player_features_csv", "sha256"],
            ),
            persistent_paths["player_features_csv"],
            True,
            blockers,
            warnings,
        )
        check_fingerprint(
            checks,
            "day72a_day71a_json",
            nested_get(day72a, ["inputs", "day71a_json", "sha256"]),
            temporary_paths["day71a_player_features_json"],
            True,
            blockers,
            warnings,
        )
        check_fingerprint(
            checks,
            "day72a_day71b_json",
            nested_get(day72a, ["inputs", "day71b_json", "sha256"]),
            temporary_paths["day71b_role_contract_json"],
            True,
            blockers,
            warnings,
        )
        check_fingerprint(
            checks,
            "day72a_scoreline_preview_csv",
            nested_get(
                day72a,
                ["inputs", "scoreline_preview_csv", "sha256"],
            ),
            persistent_paths["scoreline_preview_csv"],
            True,
            blockers,
            warnings,
        )

        out_csv = nested_get(day72a, ["outputs", "out_csv"])
        if out_csv is not None:
            expected_path = persistent_paths[
                "player_prediction_preview_csv"
            ]
            passed = (
                resolved_path(out_csv)
                == str(expected_path.expanduser().resolve())
            )
            checks.append(
                {
                    "name": "day72a_output_preview_path",
                    "passed": passed,
                    "recorded_path": out_csv,
                    "actual_path": str(expected_path),
                }
            )
            if not passed:
                blockers.append(
                    "Day72A output path does not match the inventoried "
                    "player prediction preview."
                )

    day72b = reports.get("day72b_player_manifest_json")
    if day72b is not None:
        fingerprint_targets = [
            (
                "day72b_prediction_preview_csv",
                ["artifact_fingerprints", "prediction_preview_csv", "sha256"],
                persistent_paths["player_prediction_preview_csv"],
                True,
            ),
            (
                "day72b_day72a_json",
                ["artifact_fingerprints", "day72a_json", "sha256"],
                temporary_paths["day72a_player_preview_json"],
                True,
            ),
            (
                "day72b_day71a_json",
                ["artifact_fingerprints", "day71a_json", "sha256"],
                temporary_paths["day71a_player_features_json"],
                True,
            ),
            (
                "day72b_day71b_json",
                ["artifact_fingerprints", "day71b_json", "sha256"],
                temporary_paths["day71b_role_contract_json"],
                True,
            ),
            (
                "day72b_day70c_json",
                ["artifact_fingerprints", "day70c_json", "sha256"],
                temporary_paths["day70c_scoreline_json"],
                False,
            ),
        ]
        for name, key_path, actual_path, required in fingerprint_targets:
            check_fingerprint(
                checks,
                name,
                nested_get(day72b, key_path),
                actual_path,
                required,
                blockers,
                warnings,
            )

    return {
        "passed": all(
            check.get("passed", False)
            for check in checks
            if check.get("required", True)
        ),
        "checks": checks,
    }


def summarize_persistent_artifacts(
    artifacts: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "required_count": len(artifacts),
        "existing_count": sum(
            1 for artifact in artifacts.values() if artifact["exists"]
        ),
        "readable_count": sum(
            1 for artifact in artifacts.values() if artifact["readable"]
        ),
        "passed_count": sum(
            1 for artifact in artifacts.values() if artifact["passed"]
        ),
        "total_rows": sum(
            int(artifact["row_count"] or 0)
            for artifact in artifacts.values()
        ),
        "total_size_bytes": sum(
            int(artifact["size_bytes"] or 0)
            for artifact in artifacts.values()
        ),
    }


def summarize_temporary_artifacts(
    json_artifacts: Dict[str, Dict[str, Any]],
    document_artifacts: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    required_json = [
        artifact
        for artifact in json_artifacts.values()
        if artifact["required"]
    ]
    optional_json = [
        artifact
        for artifact in json_artifacts.values()
        if not artifact["required"]
    ]
    return {
        "required_json_count": len(required_json),
        "required_json_existing_count": sum(
            1 for artifact in required_json if artifact["exists"]
        ),
        "required_json_passed_count": sum(
            1 for artifact in required_json if artifact["passed"]
        ),
        "optional_json_count": len(optional_json),
        "optional_json_existing_count": sum(
            1 for artifact in optional_json if artifact["exists"]
        ),
        "markdown_count": len(document_artifacts),
        "markdown_existing_count": sum(
            1
            for artifact in document_artifacts.values()
            if artifact["exists"]
        ),
        "temporary_files_are_regenerable": True,
        "temporary_files_are_not_long_term_storage": True,
    }


def build_report(
    args: argparse.Namespace,
    mode_result: Dict[str, Any],
    persistent_artifacts: Dict[str, Dict[str, Any]],
    temporary_json_artifacts: Dict[str, Dict[str, Any]],
    temporary_document_artifacts: Dict[str, Dict[str, Any]],
    cross_validation: Dict[str, Any],
    report_consistency: Dict[str, Any],
    blockers: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    passed = len(blockers) == 0

    dependency_count = sum(
        len(items)
        for items in CURRENT_UNRESOLVED_DEPENDENCIES.values()
    )
    satisfied_dependency_count = sum(
        1
        for items in CURRENT_UNRESOLVED_DEPENDENCIES.values()
        for item in items
        if item["satisfied"]
    )

    return {
        "created_at": utc_now(),
        "inventory_version": INVENTORY_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "source_season": args.source_season,
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "as_of_time": parse_iso8601(
            args.as_of_time,
            "as_of_time",
        ).isoformat(),
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": mode_result.get(
            "resolved_prediction_mode"
        ),
        "audit_only": True,
        "writes_database": False,
        "passed": passed,
        "ready_for_current_artifact_inventory": passed,
        "ready_for_day74a_scoring_rules_registry": passed,
        "ready_for_public_prediction": False,
        "ready_for_opening_squad_optimizer_input": False,
        "ready_for_transfer_recommendation_input": False,
        "ready_for_captain_recommendation_input": False,
        "ready_for_database_write": False,
        "reason_production_not_ready": (
            "Day73B inventories and validates the existing read-only "
            "Pre-GW1 artifact stack. It does not resolve scoring rules, "
            "multi-season evidence, trained component models, calibration, "
            "active-model policy, safe publication, optimizer approval, "
            "or cloud operations."
        ),
        "roots": {
            "persistent_artifact_root": str(
                Path(args.artifact_root).expanduser()
            ),
            "temporary_report_root": str(
                Path(args.temporary_root).expanduser()
            ),
        },
        "mode_resolution": mode_result,
        "persistent_artifact_summary": summarize_persistent_artifacts(
            persistent_artifacts
        ),
        "persistent_artifacts": persistent_artifacts,
        "temporary_artifact_summary": summarize_temporary_artifacts(
            temporary_json_artifacts,
            temporary_document_artifacts,
        ),
        "temporary_json_artifacts": temporary_json_artifacts,
        "temporary_human_reports": temporary_document_artifacts,
        "cross_artifact_validation": cross_validation,
        "report_fingerprint_and_path_validation": report_consistency,
        "dependency_summary": {
            "dependency_count": dependency_count,
            "satisfied_dependency_count": satisfied_dependency_count,
            "unresolved_dependency_count": (
                dependency_count - satisfied_dependency_count
            ),
            "all_current_dependencies_are_explicit": True,
        },
        "unresolved_dependencies": CURRENT_UNRESOLVED_DEPENDENCIES,
        "current_allowed_uses": {
            "artifact_audit": passed,
            "ops_preview_reference": passed,
            "master_plan_v2_execution_input": passed,
            "day74a_rules_registry_input": passed,
            "public_match_prediction": False,
            "public_player_prediction": False,
            "opening_squad_optimizer_input": False,
            "transfer_recommendation_input": False,
            "captain_recommendation_input": False,
            "chip_recommendation_input": False,
            "database_write": False,
            "active_model_activation": False,
        },
        "completed_checkpoint_status": {
            "match_side_day68_to_day70_persistent_artifacts_present": (
                all(
                    persistent_artifacts[key]["passed"]
                    for key in [
                        "team_priors_csv",
                        "team_mapping_csv",
                        "match_features_csv",
                        "match_features_with_fallback_csv",
                        "match_prediction_preview_csv",
                        "scoreline_preview_csv",
                    ]
                )
            ),
            "player_side_day71_to_day72_persistent_artifacts_present": (
                all(
                    persistent_artifacts[key]["passed"]
                    for key in [
                        "player_priors_csv",
                        "player_mapping_csv",
                        "player_features_csv",
                        "player_prediction_preview_csv",
                    ]
                )
            ),
            "day71a_report_passed": temporary_json_artifacts[
                "day71a_player_features_json"
            ]["passed"],
            "day71b_report_passed": temporary_json_artifacts[
                "day71b_role_contract_json"
            ]["passed"],
            "day72a_report_passed": temporary_json_artifacts[
                "day72a_player_preview_json"
            ]["passed"],
            "day72b_report_passed": temporary_json_artifacts[
                "day72b_player_manifest_json"
            ]["passed"],
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            (
                "Persistent CSV artifacts live outside /tmp and are treated "
                "as the durable current evidence for Day65-Day72."
            ),
            (
                "Day71A-Day72B JSON reports are required for this checkpoint "
                "because they are still available and contain the current "
                "player-side readiness and fingerprint chain."
            ),
            (
                "Earlier Day68-Day70 /tmp reports are recorded when present "
                "but their absence does not invalidate the durable CSV stack."
            ),
            (
                "Markdown reports are human-readable evidence only and are "
                "not required for machine validation."
            ),
            (
                "Generated inventory JSON/Markdown may contain local paths "
                "and should remain uncommitted runtime artifacts."
            ),
        ],
    }


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)

    persistent = report["persistent_artifacts"]
    temp_json = report["temporary_json_artifacts"]
    temp_docs = report["temporary_human_reports"]
    dependency_summary = report["dependency_summary"]

    lines: List[str] = [
        "# Day73B — Current Artifact and Dependency Inventory",
        "",
        "- Created at: `%s`" % report["created_at"],
        "- Inventory version: `%s`" % report["inventory_version"],
        "- Source season: `%s`" % report["source_season"],
        "- Target season: `%s`" % report["target_season"],
        "- Target GW: `%s`" % report["target_gw"],
        "- As-of time: `%s`" % report["as_of_time"],
        "- Resolved prediction mode: `%s`"
        % report["resolved_prediction_mode"],
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `%s`" % report["audit_only"],
        "- Writes database: `%s`" % report["writes_database"],
        "",
        "## Readiness",
        "",
        "- Ready for current artifact inventory: `%s`"
        % report["ready_for_current_artifact_inventory"],
        "- Ready for Day74A scoring-rules registry: `%s`"
        % report["ready_for_day74a_scoring_rules_registry"],
        "- Ready for public prediction: `%s`"
        % report["ready_for_public_prediction"],
        "- Ready for opening-squad optimizer input: `%s`"
        % report["ready_for_opening_squad_optimizer_input"],
        "- Ready for database write: `%s`"
        % report["ready_for_database_write"],
        "",
        "## Persistent Artifact Summary",
        "",
    ]

    for key, value in report["persistent_artifact_summary"].items():
        lines.append("- %s: `%s`" % (key, value))

    lines.extend(
        [
            "",
            "## Persistent Artifacts",
            "",
            "| Key | Day | Exists | Rows | Passed | SHA256 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for key, artifact in persistent.items():
        sha = artifact.get("sha256") or ""
        lines.append(
            "| `%s` | %s | %s | %s | %s | `%s` |"
            % (
                key,
                artifact["day"],
                artifact["exists"],
                artifact.get("row_count"),
                artifact["passed"],
                sha[:16] + "..." if sha else "",
            )
        )

    lines.extend(
        [
            "",
            "## Temporary JSON Reports",
            "",
            "| Key | Day | Required | Exists | Loaded | Passed |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for key, artifact in temp_json.items():
        lines.append(
            "| `%s` | %s | %s | %s | %s | %s |"
            % (
                key,
                artifact["day"],
                artifact["required"],
                artifact["exists"],
                artifact["loaded"],
                artifact["passed"],
            )
        )

    lines.extend(
        [
            "",
            "## Temporary Markdown Reports",
            "",
            "| Key | Day | Exists |",
            "|---|---|---:|",
        ]
    )
    for key, artifact in temp_docs.items():
        lines.append(
            "| `%s` | %s | %s |"
            % (
                key,
                artifact["day"],
                artifact["exists"],
            )
        )

    lines.extend(
        [
            "",
            "## Cross-Artifact Validation",
            "",
        ]
    )
    for check in report["cross_artifact_validation"]["checks"]:
        lines.append(
            "- %s: `%s`"
            % (check["name"], check["passed"])
        )

    lines.extend(
        [
            "",
            "## Report Fingerprint and Path Validation",
            "",
        ]
    )
    for check in report[
        "report_fingerprint_and_path_validation"
    ]["checks"]:
        lines.append(
            "- %s: `%s`"
            % (check["name"], check["passed"])
        )

    lines.extend(
        [
            "",
            "## Unresolved Dependency Summary",
            "",
            "- Total dependencies: `%s`"
            % dependency_summary["dependency_count"],
            "- Satisfied at this checkpoint: `%s`"
            % dependency_summary["satisfied_dependency_count"],
            "- Unresolved: `%s`"
            % dependency_summary["unresolved_dependency_count"],
            "",
        ]
    )
    for category, items in report["unresolved_dependencies"].items():
        lines.append("### %s" % category.replace("_", " ").title())
        lines.append("")
        for item in items:
            lines.append(
                "- `%s`: %s"
                % (item["name"], item["status"])
            )
        lines.append("")

    lines.extend(["## Current Allowed Uses", ""])
    for key, value in report["current_allowed_uses"].items():
        lines.append("- %s: `%s`" % (key, value))

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

    lines.extend(["", "## Notes", ""])
    lines.extend("- %s" % item for item in report["notes"])
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(report: Dict[str, Any]) -> None:
    persistent = report["persistent_artifact_summary"]
    temporary = report["temporary_artifact_summary"]
    dependencies = report["dependency_summary"]

    print("=== Day73B Current Artifact and Dependency Inventory ===")
    print("inventory_version:", report["inventory_version"])
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("target_gw:", report["target_gw"])
    print("as_of_time:", report["as_of_time"])
    print(
        "resolved_prediction_mode:",
        report["resolved_prediction_mode"],
    )
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print(
        "ready_for_current_artifact_inventory:",
        report["ready_for_current_artifact_inventory"],
    )
    print(
        "ready_for_day74a_scoring_rules_registry:",
        report["ready_for_day74a_scoring_rules_registry"],
    )
    print(
        "ready_for_opening_squad_optimizer_input:",
        report["ready_for_opening_squad_optimizer_input"],
    )
    print("ready_for_database_write:", report["ready_for_database_write"])
    print()
    print("Persistent artifacts:")
    print("- required_count:", persistent["required_count"])
    print("- existing_count:", persistent["existing_count"])
    print("- readable_count:", persistent["readable_count"])
    print("- passed_count:", persistent["passed_count"])
    print("- total_rows:", persistent["total_rows"])
    print()
    print("Temporary evidence:")
    print(
        "- required_json_count:",
        temporary["required_json_count"],
    )
    print(
        "- required_json_existing_count:",
        temporary["required_json_existing_count"],
    )
    print(
        "- required_json_passed_count:",
        temporary["required_json_passed_count"],
    )
    print(
        "- optional_json_existing_count:",
        temporary["optional_json_existing_count"],
    )
    print(
        "- markdown_existing_count:",
        temporary["markdown_existing_count"],
    )
    print()
    print("Cross-artifact validation:")
    for check in report["cross_artifact_validation"]["checks"]:
        print("- %s: %s" % (check["name"], check["passed"]))
    print()
    print("Fingerprint/path validation:")
    for check in report[
        "report_fingerprint_and_path_validation"
    ]["checks"]:
        print("- %s: %s" % (check["name"], check["passed"]))
    print()
    print("Dependencies:")
    print("- dependency_count:", dependencies["dependency_count"])
    print(
        "- unresolved_dependency_count:",
        dependencies["unresolved_dependency_count"],
    )
    print()
    print("Blockers:", report["blockers"] or "none")
    print("Warnings:", report["warnings"] or "none")
    print("saved_json:", report["outputs"]["out_json"])
    print("saved_md:", report["outputs"]["out_md"])


def main() -> None:
    args = parse_args()
    blockers = validate_args(args)
    warnings: List[str] = []

    mode_result = resolve_prediction_mode(
        season=args.target_season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.source_season,
        stabilization_gw=args.stabilization_gw,
        allow_experimental_mode=args.allow_experimental_mode,
    )
    blockers.extend(list(mode_result.get("errors") or []))
    warnings.extend(list(mode_result.get("warnings") or []))
    if mode_result.get("resolved_prediction_mode") != EXPECTED_PREDICTION_MODE:
        blockers.append(
            "Day73B expects resolved_prediction_mode=%s."
            % EXPECTED_PREDICTION_MODE
        )

    artifact_root = Path(args.artifact_root).expanduser()
    temporary_root = Path(args.temporary_root).expanduser()

    persistent_artifacts: Dict[str, Dict[str, Any]] = {}
    frames: Dict[str, Optional[pd.DataFrame]] = {}
    persistent_paths: Dict[str, Path] = {}
    for spec in persistent_specs(args):
        filename = format_template(
            spec.filename_template,
            args.source_season,
            args.target_season,
        )
        persistent_paths[spec.key] = artifact_root / filename
        result, frame = inspect_csv(
            spec,
            artifact_root,
            args,
            blockers,
            warnings,
        )
        persistent_artifacts[spec.key] = result
        frames[spec.key] = frame

    temporary_json_artifacts: Dict[str, Dict[str, Any]] = {}
    reports: Dict[str, Optional[Dict[str, Any]]] = {}
    temporary_paths: Dict[str, Path] = {}
    for spec in temporary_json_specs():
        filename = format_template(
            spec.filename_template,
            args.source_season,
            args.target_season,
        )
        temporary_paths[spec.key] = temporary_root / filename
        result, data = inspect_temporary_json(
            spec,
            temporary_root,
            args,
            blockers,
            warnings,
        )
        temporary_json_artifacts[spec.key] = result
        reports[spec.key] = data

    temporary_document_artifacts = {
        spec.key: inspect_temporary_document(
            spec,
            temporary_root,
            args,
        )
        for spec in temporary_document_specs()
    }

    cross_validation = build_cross_artifact_validation(
        frames,
        blockers,
    )
    report_consistency = build_report_consistency_validation(
        reports,
        persistent_paths,
        temporary_paths,
        blockers,
        warnings,
    )

    report = build_report(
        args=args,
        mode_result=mode_result,
        persistent_artifacts=persistent_artifacts,
        temporary_json_artifacts=temporary_json_artifacts,
        temporary_document_artifacts=temporary_document_artifacts,
        cross_validation=cross_validation,
        report_consistency=report_consistency,
        blockers=blockers,
        warnings=warnings,
    )
    report["outputs"] = {
        "out_json": str(Path(args.out_json)),
        "out_md": str(Path(args.out_md)),
    }

    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_summary(report)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
