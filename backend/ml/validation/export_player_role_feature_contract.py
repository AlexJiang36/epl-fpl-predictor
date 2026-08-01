from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from ml.validation.resolve_prediction_mode import resolve_prediction_mode


CONTRACT_NAME = "player_role_minutes_feature_contract"
CONTRACT_VERSION = "day71b_v1"
THRESHOLD_POLICY_VERSION = "player_role_thresholds_v0"
ARTIFACT_TYPE = "player_role_minutes_contract"

VALID_POSITIONS = {"GKP", "DEF", "MID", "FWD"}

REQUIRED_FEATURE_COLUMNS = [
    "source_season",
    "target_season",
    "target_gw",
    "prediction_mode",
    "feature_version",
    "player_id",
    "team_id",
    "position",
    "status",
    "has_fixture",
    "blank_gw_flag",
    "has_prev_season_player_prior",
    "prev_season_minutes",
    "prev_season_appearances",
    "prev_season_starts_proxy",
    "no_prior_flag",
    "promoted_team_player_flag",
    "uncertain_status_flag",
    "team_fallback_applied",
    "opponent_fallback_applied",
    "prediction_write_allowed",
    "production_ready",
    "requires_player_feature_manifest_before_prediction",
]

PRODUCTION_BLOCKERS = [
    "no_trained_minutes_model",
    "no_minutes_backtest",
    "no_start_probability_calibration",
    "no_expected_minutes_calibration",
    "no_multi_season_leakage_audit_for_minutes_training",
    "no_fallback_policy_evaluation",
    "no_active_minutes_model_registry_entry",
    "no_approved_role_minutes_prediction_manifest",
    "no_production_database_write_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the Day71B player role/minutes feature contract. "
            "This is read-only, validates the Day71A player feature artifact, "
            "previews configurable sample/role classifications, and writes "
            "JSON/Markdown contract artifacts. It does not predict minutes or points."
        )
    )
    parser.add_argument(
        "--source-season",
        dest="source_seasons",
        action="append",
        required=True,
        help=(
            "Historical source season. Repeat the argument for multiple lookback "
            "seasons, ordered oldest to newest."
        ),
    )
    parser.add_argument("--target-season", required=True)
    parser.add_argument("--target-gw", type=int, required=True)
    parser.add_argument(
        "--as-of-time",
        required=True,
        help=(
            "ISO-8601 prediction cutoff represented by this contract, for example "
            "2025-08-15T17:30:00+00:00."
        ),
    )
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

    parser.add_argument("--player-features-csv", required=True)
    parser.add_argument("--day71a-json", required=True)

    parser.add_argument(
        "--player-feature-version",
        default="day71a_v0",
    )
    parser.add_argument(
        "--minutes-feature-contract-version",
        default=CONTRACT_VERSION,
    )
    parser.add_argument(
        "--scoring-rules-version",
        default="target_season_rules_unresolved",
    )
    parser.add_argument(
        "--transfer-rules-version",
        default="target_season_transfer_rules_unresolved",
    )
    parser.add_argument(
        "--chip-rules-version",
        default="target_season_chip_rules_unresolved",
    )
    parser.add_argument(
        "--match-minutes-cap",
        type=float,
        default=90.0,
    )

    parser.add_argument("--status-source", default="")
    parser.add_argument("--status-as-of", default="")
    parser.add_argument(
        "--status-valid-for-prediction-cutoff",
        action="store_true",
    )
    parser.add_argument(
        "--available-status-codes",
        default="a",
        help="Comma-separated target-season status codes treated as available.",
    )
    parser.add_argument(
        "--soft-risk-status-codes",
        default="d,i",
        help="Comma-separated target-season status codes treated as soft risk.",
    )
    parser.add_argument(
        "--hard-unavailable-status-codes",
        default="s,u",
        help=(
            "Comma-separated target-season status codes eligible for a hard "
            "availability rule only when status metadata is cutoff-valid."
        ),
    )

    parser.add_argument(
        "--very-low-sample-minutes",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--very-low-sample-appearances",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--low-sample-minutes",
        type=float,
        default=450.0,
    )
    parser.add_argument(
        "--low-sample-appearances",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--reliable-sample-minutes",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--high-sample-minutes",
        type=float,
        default=1800.0,
    )

    parser.add_argument(
        "--bench-min-appearances",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--bench-max-start-rate",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--bench-max-minutes-per-appearance",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--probable-starter-min-minutes",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--probable-starter-min-appearances",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--probable-starter-min-start-rate",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--probable-starter-min-minutes-per-appearance",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--established-starter-min-minutes",
        type=float,
        default=1800.0,
    )
    parser.add_argument(
        "--established-starter-min-appearances",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--established-starter-min-start-rate",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--established-starter-min-minutes-per-appearance",
        type=float,
        default=60.0,
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
        result = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("%s must be valid ISO-8601: %s" % (label, value)) from exc

    if result.tzinfo is None:
        raise ValueError("%s must include a timezone offset: %s" % (label, value))

    return result


def parse_code_set(value: str) -> Set[str]:
    return {
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    }


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

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def sha256_file(path_value: str) -> Optional[str]:
    path = Path(path_value)
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("Required JSON file does not exist: %s" % path)

    return json.loads(path.read_text(encoding="utf-8"))


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


def validate_thresholds(args: argparse.Namespace) -> List[str]:
    blockers: List[str] = []

    if args.target_gw < 1:
        blockers.append("target_gw must be >= 1.")

    if args.match_minutes_cap <= 0:
        blockers.append("match_minutes_cap must be positive.")

    minute_thresholds = [
        args.very_low_sample_minutes,
        args.low_sample_minutes,
        args.reliable_sample_minutes,
        args.high_sample_minutes,
    ]
    if any(value < 0 for value in minute_thresholds):
        blockers.append("Sample minute thresholds must be non-negative.")

    if minute_thresholds != sorted(minute_thresholds):
        blockers.append(
            "Sample minute thresholds must be non-decreasing: "
            "very_low <= low <= reliable <= high."
        )

    appearance_thresholds = [
        args.very_low_sample_appearances,
        args.low_sample_appearances,
    ]
    if any(value < 0 for value in appearance_thresholds):
        blockers.append("Sample appearance thresholds must be non-negative.")

    if appearance_thresholds != sorted(appearance_thresholds):
        blockers.append(
            "Sample appearance thresholds must be non-decreasing."
        )

    rate_values = [
        args.bench_max_start_rate,
        args.probable_starter_min_start_rate,
        args.established_starter_min_start_rate,
    ]
    if any(value < 0 or value > 1 for value in rate_values):
        blockers.append("All start-rate thresholds must be in [0, 1].")

    if not (
        args.bench_max_start_rate
        <= args.probable_starter_min_start_rate
        <= args.established_starter_min_start_rate
    ):
        blockers.append(
            "Role start-rate thresholds must satisfy "
            "bench <= probable starter <= established starter."
        )

    return blockers


def load_player_features(
    path_value: str,
    source_seasons: List[str],
    target_season: str,
    target_gw: int,
) -> Tuple[pd.DataFrame, List[str]]:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("Player feature CSV does not exist: %s" % path)

    dataframe = pd.read_csv(path, low_memory=False)
    if dataframe.empty:
        raise RuntimeError("Player feature CSV is empty: %s" % path)

    missing_columns = [
        column
        for column in REQUIRED_FEATURE_COLUMNS
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise RuntimeError(
            "Player feature CSV is missing required columns: %s"
            % missing_columns
        )

    target_gw_numeric = pd.to_numeric(
        dataframe["target_gw"],
        errors="coerce",
    )

    filtered = dataframe[
        (dataframe["target_season"].astype(str) == str(target_season))
        & (target_gw_numeric == int(target_gw))
    ].copy()

    if filtered.empty:
        raise RuntimeError(
            "No player feature rows found for target_season=%s target_gw=%s."
            % (target_season, target_gw)
        )

    artifact_source_seasons = sorted(
        {
            str(value)
            for value in filtered["source_season"].dropna().unique().tolist()
        }
    )

    return filtered, artifact_source_seasons


def build_sample_and_role_preview(
    features: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    preview = features.copy()

    has_prior = preview[
        "has_prev_season_player_prior"
    ].apply(bool_value)

    minutes = pd.to_numeric(
        preview["prev_season_minutes"],
        errors="coerce",
    )
    appearances = pd.to_numeric(
        preview["prev_season_appearances"],
        errors="coerce",
    )
    starts = pd.to_numeric(
        preview["prev_season_starts_proxy"],
        errors="coerce",
    )

    start_rate = starts / appearances.replace(0, pd.NA)
    minutes_per_appearance = minutes / appearances.replace(0, pd.NA)

    sample_tier = pd.Series("no_safe_prior", index=preview.index)

    zero_sample = (
        has_prior
        & (
            appearances.fillna(0).le(0)
            | minutes.fillna(0).le(0)
        )
    )

    very_low_sample = (
        has_prior
        & ~zero_sample
        & (
            minutes.lt(args.very_low_sample_minutes)
            | appearances.lt(args.very_low_sample_appearances)
        )
    )

    low_sample = (
        has_prior
        & ~zero_sample
        & ~very_low_sample
        & (
            minutes.lt(args.low_sample_minutes)
            | appearances.lt(args.low_sample_appearances)
        )
    )

    moderate_sample = (
        has_prior
        & ~zero_sample
        & ~very_low_sample
        & ~low_sample
        & minutes.lt(args.reliable_sample_minutes)
    )

    reliable_sample = (
        has_prior
        & ~zero_sample
        & ~very_low_sample
        & ~low_sample
        & ~moderate_sample
        & minutes.lt(args.high_sample_minutes)
    )

    high_sample = (
        has_prior
        & minutes.ge(args.high_sample_minutes)
    )

    sample_tier.loc[zero_sample] = "zero_sample"
    sample_tier.loc[very_low_sample] = "very_low_sample"
    sample_tier.loc[low_sample] = "low_sample"
    sample_tier.loc[moderate_sample] = "moderate_sample"
    sample_tier.loc[reliable_sample] = "reliable_sample"
    sample_tier.loc[high_sample] = "high_sample"

    role_proxy = pd.Series("no_safe_prior", index=preview.index)

    insufficient_role_sample = (
        has_prior
        & sample_tier.isin(
            {
                "zero_sample",
                "very_low_sample",
                "low_sample",
            }
        )
    )

    established_starter = (
        has_prior
        & minutes.ge(args.established_starter_min_minutes)
        & appearances.ge(
            args.established_starter_min_appearances
        )
        & start_rate.ge(args.established_starter_min_start_rate)
        & minutes_per_appearance.ge(
            args.established_starter_min_minutes_per_appearance
        )
    )

    probable_starter = (
        has_prior
        & ~established_starter
        & minutes.ge(args.probable_starter_min_minutes)
        & appearances.ge(
            args.probable_starter_min_appearances
        )
        & start_rate.ge(args.probable_starter_min_start_rate)
        & minutes_per_appearance.ge(
            args.probable_starter_min_minutes_per_appearance
        )
    )

    bench_role = (
        has_prior
        & ~insufficient_role_sample
        & ~established_starter
        & ~probable_starter
        & appearances.ge(args.bench_min_appearances)
        & start_rate.lt(args.bench_max_start_rate)
        & minutes_per_appearance.lt(
            args.bench_max_minutes_per_appearance
        )
    )

    rotation_or_unclear = (
        has_prior
        & ~insufficient_role_sample
        & ~established_starter
        & ~probable_starter
        & ~bench_role
    )

    role_proxy.loc[
        insufficient_role_sample
    ] = "insufficient_sample"
    role_proxy.loc[
        established_starter
    ] = "established_starter_proxy"
    role_proxy.loc[
        probable_starter
    ] = "probable_starter_proxy"
    role_proxy.loc[bench_role] = "bench_role_proxy"
    role_proxy.loc[
        rotation_or_unclear
    ] = "rotation_or_unclear_role"

    preview["sample_reliability_tier_v0"] = sample_tier
    preview["role_proxy_v0"] = role_proxy
    preview["derived_start_rate"] = start_rate
    preview[
        "derived_minutes_per_appearance"
    ] = minutes_per_appearance

    return preview


def dataframe_count_dict(series: pd.Series) -> Dict[str, int]:
    counts = series.fillna("<missing>").astype(str).value_counts()
    return {
        str(key): int(value)
        for key, value in counts.to_dict().items()
    }


def crosstab_dict(
    row_series: pd.Series,
    column_series: pd.Series,
) -> Dict[str, Dict[str, int]]:
    table = pd.crosstab(row_series, column_series)
    return {
        str(row_name): {
            str(column_name): int(value)
            for column_name, value in row.to_dict().items()
        }
        for row_name, row in table.iterrows()
    }


def build_contract(
    args: argparse.Namespace,
    features: pd.DataFrame,
    preview: pd.DataFrame,
    artifact_source_seasons: List[str],
    day71a_report: Dict[str, Any],
    mode_result: Dict[str, Any],
) -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []

    blockers.extend(validate_thresholds(args))
    blockers.extend(list(mode_result.get("errors") or []))
    warnings.extend(list(mode_result.get("warnings") or []))

    requested_source_seasons = [
        str(value)
        for value in args.source_seasons
    ]

    unexpected_source_seasons = sorted(
        set(artifact_source_seasons)
        - set(requested_source_seasons)
    )
    if unexpected_source_seasons:
        blockers.append(
            "Feature artifact contains source seasons not declared by "
            "--source-season: %s." % unexpected_source_seasons
        )

    resolved_prediction_mode = mode_result.get(
        "resolved_prediction_mode"
    )

    feature_mode_values = sorted(
        {
            str(value)
            for value in features["prediction_mode"]
            .dropna()
            .unique()
            .tolist()
        }
    )
    if feature_mode_values != [str(resolved_prediction_mode)]:
        blockers.append(
            "Feature prediction_mode values %s do not match resolved mode %s."
            % (feature_mode_values, resolved_prediction_mode)
        )

    feature_version_values = sorted(
        {
            str(value)
            for value in features["feature_version"]
            .dropna()
            .unique()
            .tolist()
        }
    )
    if feature_version_values != [args.player_feature_version]:
        blockers.append(
            "Feature version expected %s, got %s."
            % (
                args.player_feature_version,
                feature_version_values,
            )
        )

    as_of_time = parse_iso8601(
        args.as_of_time,
        "as_of_time",
    )

    status_as_of_time: Optional[datetime] = None
    if args.status_as_of:
        status_as_of_time = parse_iso8601(
            args.status_as_of,
            "status_as_of",
        )

    status_metadata_complete = bool(
        args.status_source
        and args.status_as_of
    )

    if args.status_valid_for_prediction_cutoff:
        if not status_metadata_complete:
            blockers.append(
                "status_valid_for_prediction_cutoff requires both "
                "status_source and status_as_of."
            )
        elif status_as_of_time is not None and status_as_of_time > as_of_time:
            blockers.append(
                "status_as_of occurs after the prediction cutoff."
            )
    else:
        warnings.append(
            "Status values are not proven cutoff-valid. Day71B records status "
            "as leakage-sensitive metadata and does not authorize hard status "
            "guardrails for historical prediction."
        )

    available_codes = parse_code_set(
        args.available_status_codes
    )
    soft_risk_codes = parse_code_set(
        args.soft_risk_status_codes
    )
    hard_unavailable_codes = parse_code_set(
        args.hard_unavailable_status_codes
    )

    overlapping_status_codes = sorted(
        (available_codes & soft_risk_codes)
        | (available_codes & hard_unavailable_codes)
        | (soft_risk_codes & hard_unavailable_codes)
    )
    if overlapping_status_codes:
        blockers.append(
            "Status-code groups overlap: %s."
            % overlapping_status_codes
        )

    row_count = int(len(features))
    duplicate_player_id_count = int(
        features["player_id"].duplicated(keep=False).sum()
    )
    missing_player_id_count = int(
        features["player_id"].isna().sum()
    )
    missing_team_id_count = int(
        features["team_id"].isna().sum()
    )
    missing_position_count = int(
        features["position"].isna().sum()
        + (
            features["position"]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        ).sum()
    )
    invalid_position_values = sorted(
        set(
            features["position"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        - VALID_POSITIONS
    )

    no_fixture_count = int(
        (~features["has_fixture"].apply(bool_value)).sum()
    )
    blank_gw_count = int(
        features["blank_gw_flag"].apply(bool_value).sum()
    )

    prediction_write_allowed_true_count = int(
        features[
            "prediction_write_allowed"
        ].apply(bool_value).sum()
    )
    production_ready_true_count = int(
        features["production_ready"].apply(bool_value).sum()
    )
    requires_manifest_false_count = int(
        (
            ~features[
                "requires_player_feature_manifest_before_prediction"
            ].apply(bool_value)
        ).sum()
    )

    day71a_passed = nested_get(
        day71a_report,
        ["passed"],
        None,
    )
    day71a_writes_database = nested_get(
        day71a_report,
        ["writes_database"],
        None,
    )
    day71a_feature_rows = nested_get(
        day71a_report,
        ["row_counts", "feature_rows"],
        None,
    )
    day71a_blockers = nested_get(
        day71a_report,
        ["blockers"],
        [],
    ) or []

    if day71a_passed is not True:
        blockers.append("Day71A report did not pass.")
    if day71a_writes_database is not False:
        blockers.append(
            "Day71A report must record writes_database=false."
        )
    if day71a_feature_rows is not None and int(
        day71a_feature_rows
    ) != row_count:
        blockers.append(
            "Day71A report feature row count does not match CSV."
        )
    if day71a_blockers:
        blockers.append(
            "Day71A report contains blockers: %s."
            % day71a_blockers
        )

    if duplicate_player_id_count:
        blockers.append(
            "Player feature artifact contains duplicate player_id rows."
        )
    if missing_player_id_count:
        blockers.append(
            "Player feature artifact contains missing player_id values."
        )
    if missing_team_id_count:
        blockers.append(
            "Player feature artifact contains missing team_id values."
        )
    if missing_position_count:
        blockers.append(
            "Player feature artifact contains missing position values."
        )
    if invalid_position_values:
        blockers.append(
            "Invalid position values found: %s."
            % invalid_position_values
        )
    if prediction_write_allowed_true_count:
        blockers.append(
            "prediction_write_allowed must remain false for every Day71A row."
        )
    if production_ready_true_count:
        blockers.append(
            "production_ready must remain false for every Day71A row."
        )
    if requires_manifest_false_count:
        blockers.append(
            "requires_player_feature_manifest_before_prediction must remain "
            "true for every Day71A row."
        )

    sample_tier_counts = dataframe_count_dict(
        preview["sample_reliability_tier_v0"]
    )
    role_proxy_counts = dataframe_count_dict(
        preview["role_proxy_v0"]
    )

    if sum(sample_tier_counts.values()) != row_count:
        blockers.append(
            "Sample-tier preview counts do not sum to feature rows."
        )
    if sum(role_proxy_counts.values()) != row_count:
        blockers.append(
            "Role-proxy preview counts do not sum to feature rows."
        )

    promoted_mask = features[
        "promoted_team_player_flag"
    ].apply(bool_value)

    promoted_role_counts = dataframe_count_dict(
        preview.loc[promoted_mask, "role_proxy_v0"]
    )

    normalized_status = (
        features["status"]
        .fillna("<missing>")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    observed_status_values = set(
        normalized_status.unique().tolist()
    )
    configured_status_values = (
        available_codes
        | soft_risk_codes
        | hard_unavailable_codes
        | {"<missing>"}
    )
    unconfigured_status_values = sorted(
        observed_status_values
        - configured_status_values
    )
    if unconfigured_status_values:
        warnings.append(
            "Observed status values are not covered by the configured status "
            "policy: %s." % unconfigured_status_values
        )

    status_hard_guardrail_enabled = bool(
        args.status_valid_for_prediction_cutoff
        and status_metadata_complete
        and not blockers
    )

    hard_unavailable_observed_count = int(
        normalized_status.isin(
            hard_unavailable_codes
        ).sum()
    )
    soft_status_risk_observed_count = int(
        normalized_status.isin(
            soft_risk_codes
        ).sum()
    )

    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    passed = len(blockers) == 0

    thresholds = {
        "threshold_policy_version": THRESHOLD_POLICY_VERSION,
        "sample_reliability": {
            "zero_sample": {
                "rule": (
                    "has_safe_prior and "
                    "(appearances <= 0 or minutes <= 0)"
                ),
            },
            "very_low_sample": {
                "minutes_below": args.very_low_sample_minutes,
                "appearances_below": (
                    args.very_low_sample_appearances
                ),
                "join_operator": "OR",
            },
            "low_sample": {
                "minutes_below": args.low_sample_minutes,
                "appearances_below": (
                    args.low_sample_appearances
                ),
                "join_operator": "OR",
            },
            "moderate_sample": {
                "minimum_previous_tier_passed": True,
                "minutes_below": (
                    args.reliable_sample_minutes
                ),
            },
            "reliable_sample": {
                "minimum_previous_tier_passed": True,
                "minutes_at_least": (
                    args.reliable_sample_minutes
                ),
                "minutes_below": args.high_sample_minutes,
            },
            "high_sample": {
                "minutes_at_least": (
                    args.high_sample_minutes
                ),
            },
        },
        "role_proxy": {
            "bench_role_proxy": {
                "appearances_at_least": (
                    args.bench_min_appearances
                ),
                "start_rate_below": (
                    args.bench_max_start_rate
                ),
                "minutes_per_appearance_below": (
                    args.bench_max_minutes_per_appearance
                ),
                "requires_non_insufficient_sample": True,
            },
            "probable_starter_proxy": {
                "minutes_at_least": (
                    args.probable_starter_min_minutes
                ),
                "appearances_at_least": (
                    args.probable_starter_min_appearances
                ),
                "start_rate_at_least": (
                    args.probable_starter_min_start_rate
                ),
                "minutes_per_appearance_at_least": (
                    args.probable_starter_min_minutes_per_appearance
                ),
            },
            "established_starter_proxy": {
                "minutes_at_least": (
                    args.established_starter_min_minutes
                ),
                "appearances_at_least": (
                    args.established_starter_min_appearances
                ),
                "start_rate_at_least": (
                    args.established_starter_min_start_rate
                ),
                "minutes_per_appearance_at_least": (
                    args.established_starter_min_minutes_per_appearance
                ),
            },
        },
        "policy": {
            "thresholds_are_configurable": True,
            "thresholds_are_v0_proxies_not_permanent_football_rules": True,
            "thresholds_must_be_recorded_in_manifests": True,
            "thresholds_should_be_evaluated_by_historical_backtest": True,
            "sample_tier_does_not_directly_set_expected_minutes": True,
            "role_proxy_does_not_directly_set_expected_minutes": True,
        },
    }

    contract = {
        "created_at": utc_now(),
        "contract_name": CONTRACT_NAME,
        "contract_version": args.minutes_feature_contract_version,
        "artifact_type": ARTIFACT_TYPE,
        "source_seasons": requested_source_seasons,
        "artifact_source_seasons": artifact_source_seasons,
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "as_of_time": as_of_time.isoformat(),
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": resolved_prediction_mode,
        "player_feature_version": args.player_feature_version,
        "scoring_rules_version": args.scoring_rules_version,
        "transfer_rules_version": args.transfer_rules_version,
        "chip_rules_version": args.chip_rules_version,
        "threshold_policy_version": THRESHOLD_POLICY_VERSION,
        "audit_only": True,
        "writes_database": False,
        "generates_minutes_predictions": False,
        "generates_player_points_predictions": False,
        "passed": passed,
        "ready_for_role_feature_contract": passed,
        "ready_for_day72a_prediction_preview_design": passed,
        "ready_for_trained_minutes_prediction": False,
        "ready_for_player_points_prediction": False,
        "ready_for_production_write": False,
        "inputs": {
            "player_features_csv": {
                "path": str(Path(args.player_features_csv)),
                "sha256": sha256_file(
                    args.player_features_csv
                ),
                "rows": row_count,
            },
            "day71a_json": {
                "path": str(Path(args.day71a_json)),
                "sha256": sha256_file(args.day71a_json),
                "passed": day71a_passed,
                "writes_database": day71a_writes_database,
            },
        },
        "architecture_principles": {
            "season_agnostic": True,
            "gw_and_as_of_aware": True,
            "leakage_safe": True,
            "config_driven_rules": True,
            "versioned_contracts": True,
            "standard_model_outputs": True,
            "pure_logic_separated_from_io": True,
            "reusable_helpers": True,
            "rolling_horizon_compatible": True,
            "fail_safe_publishing": True,
        },
        "role_minutes_input_contract": {
            "available_now_from_day71a": [
                "player_id",
                "team_id",
                "position",
                "price",
                "status",
                "has_fixture",
                "blank_gw_flag",
                "has_prev_season_player_prior",
                "prev_season_minutes",
                "prev_season_appearances",
                "prev_season_starts_proxy",
                "no_prior_flag",
                "promoted_team_player_flag",
                "team_fallback_applied",
                "opponent_fallback_applied",
                "uncertain_status_flag",
            ],
            "registered_future_feature_names": [
                "recent_minutes",
                "recent_starts",
                "consecutive_starts",
                "days_since_last_minutes",
                "injury_return_flag",
                "long_absence_flag",
                "transferred_team_flag",
                "new_signing_flag",
                "rotation_risk_flag",
                "manager_role_change_flag",
                "status_source",
                "status_as_of",
                "status_valid_for_prediction_cutoff",
            ],
            "unregistered_features_must_not_be_used_silently": True,
        },
        "standard_minutes_model_output_contract": {
            "required_fields": [
                "season",
                "target_gw",
                "as_of_time",
                "player_id",
                "appearance_probability",
                "start_probability",
                "expected_minutes",
                "minutes_lower_bound",
                "minutes_upper_bound",
                "role_class",
                "role_confidence",
                "role_risk_flags",
                "fallback_policy_used",
                "fallback_level",
                "fallback_reason",
                "model_name",
                "model_version",
                "feature_version",
                "rules_version",
                "run_id",
            ],
            "range_contract": {
                "appearance_probability": "[0, 1]",
                "start_probability": "[0, 1]",
                "expected_minutes": (
                    "[0, configured_match_minutes_cap]"
                ),
                "configured_match_minutes_cap": (
                    args.match_minutes_cap
                ),
                "minutes_lower_bound": (
                    "0 <= lower <= expected_minutes"
                ),
                "minutes_upper_bound": (
                    "expected_minutes <= upper <= "
                    "configured_match_minutes_cap"
                ),
                "role_confidence": "[0, 1]",
            },
            "relationship_contract": [
                "start_probability <= appearance_probability",
                "no_fixture implies expected_minutes == 0",
                "blank_gameweek implies expected_minutes == 0",
                "hard unavailable status may imply expected_minutes == 0 only when status is cutoff-valid",
                "no_safe_prior does not imply expected_minutes == 0",
            ],
        },
        "role_class_contract": {
            "standard_role_classes": [
                "established_starter",
                "probable_starter",
                "rotation_player",
                "bench_role",
                "low_sample_prior",
                "no_safe_prior",
                "promoted_team_uncertain",
                "new_or_transferred_player",
                "unavailable",
                "no_fixture",
                "unknown_role",
            ],
            "day71b_v0_role_proxies_are_not_model_predictions": True,
        },
        "hard_constraint_contract": {
            "prediction_blockers": [
                "player_not_in_target_player_pool",
                "missing_player_id",
                "missing_team_id",
                "missing_position",
                "invalid_position",
                "invalid_or_outdated_as_of_time",
                "feature_timestamp_after_prediction_cutoff",
            ],
            "zero_minutes_rules": [
                "no_fixture",
                "blank_gameweek",
            ],
            "conditional_zero_minutes_rules": [
                (
                    "unavailable_or_suspended only when status source and "
                    "status_as_of prove cutoff validity"
                ),
            ],
            "not_hard_blockers": [
                "no_safe_prior",
                "promoted_team_player",
                "new_signing",
                "transferred_team",
                "low_prior_minutes",
                "bench_role_proxy",
                "uncertain_status_without_cutoff_validity",
            ],
        },
        "soft_risk_contract": {
            "standard_soft_risk_flags": [
                "no_prior_flag",
                "zero_prior_sample_flag",
                "very_low_prior_sample_flag",
                "low_prior_sample_flag",
                "low_appearance_sample_flag",
                "bench_role_proxy_flag",
                "low_start_rate_flag",
                "promoted_team_player_flag",
                "new_signing_flag",
                "transferred_team_flag",
                "uncertain_status_flag",
                "injury_return_flag",
                "long_absence_flag",
                "rotation_risk_flag",
                "team_context_fallback_flag",
                "opponent_context_fallback_flag",
                "status_unknown_for_as_of_flag",
            ],
            "soft_risks_should_be_features_first": True,
            "soft_risks_may_reduce_role_confidence": True,
            "soft_risks_may_widen_uncertainty": True,
            "soft_risks_must_not_silently_force_zero_minutes": True,
        },
        "fallback_hierarchy": [
            {
                "level": 0,
                "name": "hard_fixture_or_availability_rule",
            },
            {
                "level": 1,
                "name": "trained_and_approved_minutes_model",
            },
            {
                "level": 2,
                "name": "safe_multi_season_player_prior",
            },
            {
                "level": 3,
                "name": "safe_previous_season_player_prior",
            },
            {
                "level": 4,
                "name": "team_position_price_band_prior",
            },
            {
                "level": 5,
                "name": "position_baseline_prior",
            },
            {
                "level": 6,
                "name": "explicit_unknown_role_fallback",
            },
        ],
        "fallback_output_requirements": [
            "fallback_policy_used",
            "fallback_level",
            "fallback_reason",
            "role_confidence",
            "role_risk_flags",
        ],
        "status_policy": {
            "status_source": args.status_source or None,
            "status_as_of": (
                status_as_of_time.isoformat()
                if status_as_of_time is not None
                else None
            ),
            "status_valid_for_prediction_cutoff": (
                args.status_valid_for_prediction_cutoff
            ),
            "status_hard_guardrail_enabled": (
                status_hard_guardrail_enabled
            ),
            "available_status_codes": sorted(
                available_codes
            ),
            "soft_risk_status_codes": sorted(
                soft_risk_codes
            ),
            "hard_unavailable_status_codes": sorted(
                hard_unavailable_codes
            ),
            "hard_unavailable_observed_count": (
                hard_unavailable_observed_count
            ),
            "soft_status_risk_observed_count": (
                soft_status_risk_observed_count
            ),
            "unconfigured_status_values": (
                unconfigured_status_values
            ),
            "principle": (
                "Status may become a hard availability rule only when its source "
                "and timestamp are proven valid for the prediction cutoff."
            ),
        },
        "thresholds": thresholds,
        "classification_preview": {
            "row_count": row_count,
            "sample_tier_counts": sample_tier_counts,
            "role_proxy_counts": role_proxy_counts,
            "role_by_position": crosstab_dict(
                preview["position"],
                preview["role_proxy_v0"],
            ),
            "promoted_team_role_counts": (
                promoted_role_counts
            ),
            "promoted_team_player_count": int(
                promoted_mask.sum()
            ),
            "players_without_safe_prior": int(
                (
                    preview["role_proxy_v0"]
                    == "no_safe_prior"
                ).sum()
            ),
            "classification_is_contract_preview_not_prediction": True,
        },
        "validation_summary": {
            "feature_rows": row_count,
            "duplicate_player_id_count": (
                duplicate_player_id_count
            ),
            "missing_player_id_count": (
                missing_player_id_count
            ),
            "missing_team_id_count": missing_team_id_count,
            "missing_position_count": (
                missing_position_count
            ),
            "invalid_position_values": (
                invalid_position_values
            ),
            "players_without_fixture": no_fixture_count,
            "blank_gameweek_rows": blank_gw_count,
            "prediction_write_allowed_true_count": (
                prediction_write_allowed_true_count
            ),
            "production_ready_true_count": (
                production_ready_true_count
            ),
            "requires_manifest_false_count": (
                requires_manifest_false_count
            ),
        },
        "allowed_uses": {
            "local_audit": True,
            "contract_reference": True,
            "day72a_preview_design_input": True,
            "feature_registry_input": True,
            "trained_minutes_prediction": False,
            "public_production_prediction": False,
            "database_write": False,
            "active_model_registry_candidate": False,
        },
        "production_blockers": PRODUCTION_BLOCKERS,
        "required_before_production_minutes_use": [
            "versioned multi-season minutes training dataset",
            "as-of-safe feature snapshot",
            "trained minutes model",
            "minutes MAE evaluation",
            "start probability calibration",
            "expected minutes calibration",
            "position and role-segment evaluation",
            "no-prior and promoted-team fallback evaluation",
            "approved model registry entry",
            "approved prediction manifest",
            "dry-run database write path",
            "fail-safe publishing policy",
        ],
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            (
                "has_prev_season_player_prior means a safe identity/prior join "
                "succeeded; it does not mean the role sample is reliable."
            ),
            (
                "Day71B sample tiers and role labels are configurable V0 proxies, "
                "not trained minutes predictions."
            ),
            (
                "No-safe-prior and promoted-team players remain eligible for future "
                "predictions through explicit fallback policies."
            ),
            (
                "The contract is compatible with multiple source seasons, but the "
                "current Day71A artifact may contain only one source season."
            ),
            (
                "Core model and optimization logic must remain separate from file "
                "and database I/O."
            ),
        ],
    }

    return contract


def write_json(
    contract: Dict[str, Any],
    out_json: str,
) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def write_markdown(
    contract: Dict[str, Any],
    out_md: str,
) -> None:
    preview = contract["classification_preview"]
    validation = contract["validation_summary"]
    status_policy = contract["status_policy"]

    lines = [
        "# Day71B — Player Role / Minutes Feature Contract",
        "",
        "- Created at: `%s`" % contract["created_at"],
        "- Contract version: `%s`"
        % contract["contract_version"],
        "- Threshold policy: `%s`"
        % contract["threshold_policy_version"],
        "- Source seasons: `%s`"
        % ", ".join(contract["source_seasons"]),
        "- Target season: `%s`" % contract["target_season"],
        "- Target GW: `%s`" % contract["target_gw"],
        "- As-of time: `%s`" % contract["as_of_time"],
        "- Resolved prediction mode: `%s`"
        % contract["resolved_prediction_mode"],
        "- Passed: `%s`" % contract["passed"],
        "- Audit only: `%s`" % contract["audit_only"],
        "- Writes database: `%s`"
        % contract["writes_database"],
        "- Generates minutes predictions: `%s`"
        % contract["generates_minutes_predictions"],
        "- Generates player-points predictions: `%s`"
        % contract["generates_player_points_predictions"],
        "",
        "## Readiness",
        "",
        "- Ready for role feature contract: `%s`"
        % contract["ready_for_role_feature_contract"],
        "- Ready for Day72A preview design: `%s`"
        % contract[
            "ready_for_day72a_prediction_preview_design"
        ],
        "- Ready for trained minutes prediction: `%s`"
        % contract["ready_for_trained_minutes_prediction"],
        "- Ready for player-points prediction: `%s`"
        % contract["ready_for_player_points_prediction"],
        "- Ready for production write: `%s`"
        % contract["ready_for_production_write"],
        "",
        "## Classification preview",
        "",
        "- Feature rows: `%s`" % preview["row_count"],
        "- Promoted-team player rows: `%s`"
        % preview["promoted_team_player_count"],
        "- Players without safe prior: `%s`"
        % preview["players_without_safe_prior"],
        "",
        "### Sample tiers",
        "",
    ]

    for name, count in preview[
        "sample_tier_counts"
    ].items():
        lines.append("- `%s`: `%s`" % (name, count))

    lines.extend(
        [
            "",
            "### Role proxies",
            "",
        ]
    )
    for name, count in preview[
        "role_proxy_counts"
    ].items():
        lines.append("- `%s`: `%s`" % (name, count))

    lines.extend(
        [
            "",
            "## Status policy",
            "",
            "- Source: `%s`"
            % (status_policy["status_source"] or "not_provided"),
            "- Status as-of: `%s`"
            % (
                status_policy["status_as_of"]
                or "not_provided"
            ),
            "- Cutoff-valid: `%s`"
            % status_policy[
                "status_valid_for_prediction_cutoff"
            ],
            "- Hard status guardrail enabled: `%s`"
            % status_policy[
                "status_hard_guardrail_enabled"
            ],
            "",
            "## Validation",
            "",
            "- Duplicate player IDs: `%s`"
            % validation["duplicate_player_id_count"],
            "- Missing player IDs: `%s`"
            % validation["missing_player_id_count"],
            "- Missing team IDs: `%s`"
            % validation["missing_team_id_count"],
            "- Missing positions: `%s`"
            % validation["missing_position_count"],
            "- Players without fixture: `%s`"
            % validation["players_without_fixture"],
            "- prediction_write_allowed=True: `%s`"
            % validation[
                "prediction_write_allowed_true_count"
            ],
            "- production_ready=True: `%s`"
            % validation["production_ready_true_count"],
            "- requires manifest=False: `%s`"
            % validation["requires_manifest_false_count"],
            "",
            "## Fallback hierarchy",
            "",
        ]
    )

    for item in contract["fallback_hierarchy"]:
        lines.append(
            "- Level `%s`: `%s`"
            % (item["level"], item["name"])
        )

    lines.extend(
        [
            "",
            "## Production blockers",
            "",
        ]
    )
    for item in contract["production_blockers"]:
        lines.append("- `%s`" % item)

    lines.extend(["", "## Blockers", ""])
    if contract["blockers"]:
        lines.extend(
            "- %s" % item
            for item in contract["blockers"]
        )
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    if contract["warnings"]:
        lines.extend(
            "- %s" % item
            for item in contract["warnings"]
        )
    else:
        lines.append("- None")

    lines.extend(["", "## Notes", ""])
    lines.extend(
        "- %s" % item
        for item in contract["notes"]
    )

    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def print_contract_summary(
    contract: Dict[str, Any],
) -> None:
    preview = contract["classification_preview"]
    validation = contract["validation_summary"]
    status_policy = contract["status_policy"]

    print("=== Day71B Player Role / Minutes Feature Contract ===")
    print("contract_version:", contract["contract_version"])
    print(
        "threshold_policy_version:",
        contract["threshold_policy_version"],
    )
    print("source_seasons:", contract["source_seasons"])
    print("target_season:", contract["target_season"])
    print("target_gw:", contract["target_gw"])
    print("as_of_time:", contract["as_of_time"])
    print(
        "resolved_prediction_mode:",
        contract["resolved_prediction_mode"],
    )
    print("passed:", contract["passed"])
    print("audit_only:", contract["audit_only"])
    print("writes_database:", contract["writes_database"])
    print(
        "ready_for_role_feature_contract:",
        contract["ready_for_role_feature_contract"],
    )
    print(
        "ready_for_day72a_prediction_preview_design:",
        contract[
            "ready_for_day72a_prediction_preview_design"
        ],
    )
    print(
        "ready_for_trained_minutes_prediction:",
        contract["ready_for_trained_minutes_prediction"],
    )
    print(
        "ready_for_player_points_prediction:",
        contract["ready_for_player_points_prediction"],
    )
    print(
        "ready_for_production_write:",
        contract["ready_for_production_write"],
    )

    print("\nClassification preview:")
    print(
        "- sample_tier_counts:",
        preview["sample_tier_counts"],
    )
    print(
        "- role_proxy_counts:",
        preview["role_proxy_counts"],
    )
    print(
        "- promoted_team_role_counts:",
        preview["promoted_team_role_counts"],
    )

    print("\nStatus policy:")
    print(
        "- status_valid_for_prediction_cutoff:",
        status_policy[
            "status_valid_for_prediction_cutoff"
        ],
    )
    print(
        "- status_hard_guardrail_enabled:",
        status_policy[
            "status_hard_guardrail_enabled"
        ],
    )

    print("\nValidation:")
    print(
        "- feature_rows:",
        validation["feature_rows"],
    )
    print(
        "- duplicate_player_id_count:",
        validation["duplicate_player_id_count"],
    )
    print(
        "- prediction_write_allowed_true_count:",
        validation[
            "prediction_write_allowed_true_count"
        ],
    )
    print(
        "- production_ready_true_count:",
        validation["production_ready_true_count"],
    )
    print(
        "- requires_manifest_false_count:",
        validation["requires_manifest_false_count"],
    )

    print("\nBlockers:")
    if contract["blockers"]:
        for blocker in contract["blockers"]:
            print("-", blocker)
    else:
        print("- none")

    print("\nWarnings:")
    if contract["warnings"]:
        for warning in contract["warnings"]:
            print("-", warning)
    else:
        print("- none")


def main() -> None:
    args = parse_args()

    player_features, artifact_source_seasons = (
        load_player_features(
            path_value=args.player_features_csv,
            source_seasons=args.source_seasons,
            target_season=args.target_season,
            target_gw=args.target_gw,
        )
    )
    day71a_report = load_json(args.day71a_json)

    mode_result = resolve_prediction_mode(
        season=args.target_season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.source_seasons[-1],
        stabilization_gw=args.stabilization_gw,
        allow_experimental_mode=args.allow_experimental_mode,
    )

    preview = build_sample_and_role_preview(
        features=player_features,
        args=args,
    )

    contract = build_contract(
        args=args,
        features=player_features,
        preview=preview,
        artifact_source_seasons=artifact_source_seasons,
        day71a_report=day71a_report,
        mode_result=mode_result,
    )

    write_json(contract, args.out_json)
    write_markdown(contract, args.out_md)
    print_contract_summary(contract)

    if contract["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
