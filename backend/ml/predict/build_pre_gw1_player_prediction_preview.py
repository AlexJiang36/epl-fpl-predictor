from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml.validation.export_player_role_feature_contract import (
    CONTRACT_VERSION as DAY71B_CONTRACT_VERSION,
    THRESHOLD_POLICY_VERSION as DAY71B_THRESHOLD_POLICY_VERSION,
    bool_value,
    build_sample_and_role_preview,
)
from ml.validation.resolve_prediction_mode import resolve_prediction_mode


MODEL_NAME = "pre_gw1_player_prior_heuristic_v0"
MODEL_VERSION = "day72a_v0_2"
PREDICTION_SCOPE = "read_only_pre_gw1_player_prediction_preview"
ARTIFACT_TYPE = "pre_gw1_player_prediction_preview"
COMPONENT_ACCOUNTING_STATUS = "heuristic_components_reconciled_to_final_points"

VALID_POSITIONS = {"GKP", "DEF", "MID", "FWD"}
PRICE_BAND_LABELS = [
    "price_q1_low",
    "price_q2_mid_low",
    "price_q3_mid_high",
    "price_q4_high",
]

# These weights are deliberately conservative and versioned as part of the
# Day72A preview policy. They are not trained coefficients.
PLAYER_PRIOR_WEIGHT_BY_SAMPLE_TIER = {
    "high_sample": 0.85,
    "reliable_sample": 0.70,
    "moderate_sample": 0.45,
    "low_sample": 0.20,
    "very_low_sample": 0.10,
    "zero_sample": 0.00,
    "no_safe_prior": 0.00,
}

# Low-sample points-per-90 is particularly unstable, so the points-rate prior
# receives less weight than the role/minutes prior.
POINTS_PRIOR_WEIGHT_BY_SAMPLE_TIER = {
    "high_sample": 0.85,
    "reliable_sample": 0.70,
    "moderate_sample": 0.35,
    "low_sample": 0.00,
    "very_low_sample": 0.00,
    "zero_sample": 0.00,
    "no_safe_prior": 0.00,
}

UNCERTAINTY_HALF_WIDTH_BY_SAMPLE_TIER = {
    "high_sample": 14.0,
    "reliable_sample": 20.0,
    "moderate_sample": 26.0,
    "low_sample": 32.0,
    "very_low_sample": 36.0,
    "zero_sample": 40.0,
    "no_safe_prior": 38.0,
}

# A sparse position/price cell falls back to a position-only baseline. That is
# less informative about role, so expected minutes are conservatively reduced.
POSITION_ONLY_FALLBACK_MINUTES_MULTIPLIER = {
    "GKP": 0.35,
    "DEF": 0.65,
    "MID": 0.65,
    "FWD": 0.65,
}

# Unknown-role and very-low-sample fallbacks must not inherit a reliable-player
# workload unchanged. These multipliers reduce appearance/start probabilities
# and expected minutes while keeping every player in the preview.
FALLBACK_ROLE_UNCERTAINTY_MULTIPLIER_BY_SAMPLE_TIER = {
    "no_safe_prior": 0.75,
    "zero_sample": 0.35,
    "very_low_sample": 0.50,
    "low_sample": 0.65,
    "moderate_sample": 0.85,
    "reliable_sample": 1.00,
    "high_sample": 1.00,
}

# Price is used only as a weak, transparent target-season market signal inside
# an already selected position/price fallback cell. It differentiates players
# that would otherwise receive identical fallback rows; it never creates a
# production-ready role assertion.
FALLBACK_RATE_PRICE_SIGNAL_MIN = 0.85
FALLBACK_RATE_PRICE_SIGNAL_MAX = 1.15
FALLBACK_MINUTES_PRICE_SIGNAL_MIN = 0.90
FALLBACK_MINUTES_PRICE_SIGNAL_MAX = 1.10

# At most one goalkeeper can normally appear for a club in a fixture. A soft
# team-level cap prevents several goalkeepers from the same club each receiving
# starter-like probabilities when role evidence is missing.
GOALKEEPER_TEAM_APPEARANCE_BUDGET = 0.98
GOALKEEPER_PRICE_STRENGTH = 1.40

# Fixture adjustment weights sum to at most 1.0; any remaining weight is
# neutral. The scoreline preview already contains the home/away modifiers.
FIXTURE_COMPONENT_WEIGHTS = {
    "GKP": {"attack": 0.00, "defence": 0.35},
    "DEF": {"attack": 0.15, "defence": 0.35},
    "MID": {"attack": 0.40, "defence": 0.05},
    "FWD": {"attack": 0.50, "defence": 0.00},
}

GOAL_POINTS_BY_POSITION = {
    "GKP": 6.0,
    "DEF": 6.0,
    "MID": 5.0,
    "FWD": 4.0,
}

CLEAN_SHEET_POINTS_BY_POSITION = {
    "GKP": 4.0,
    "DEF": 4.0,
    "MID": 1.0,
    "FWD": 0.0,
}

REQUIRED_PLAYER_FEATURE_COLUMNS = [
    "source_season",
    "target_season",
    "target_gw",
    "prediction_mode",
    "feature_scope",
    "feature_version",
    "player_id",
    "fpl_player_id",
    "player_name",
    "web_name",
    "team_id",
    "team_name",
    "team_short_name",
    "position",
    "price",
    "status",
    "chance_of_playing_next_round",
    "news",
    "news_added",
    "has_fixture",
    "fixture_id",
    "fpl_fixture_id",
    "opponent_team_id",
    "opponent_team_name",
    "opponent_short_name",
    "is_home",
    "blank_gw_flag",
    "has_prev_season_player_prior",
    "player_mapping_status",
    "player_mapping_confidence",
    "player_mapping_reason",
    "prior_source",
    "prior_identity_scope",
    "prev_season_minutes",
    "prev_season_total_points",
    "prev_season_points_per90",
    "prev_season_goals",
    "prev_season_assists",
    "prev_season_bonus",
    "prev_season_clean_sheets",
    "prev_season_starts_proxy",
    "prev_season_appearances",
    "prev_season_points_per_match",
    "prior_reliability_score",
    "team_fallback_applied",
    "opponent_fallback_applied",
    "no_prior_flag",
    "promoted_team_player_flag",
    "opponent_promoted_team_flag",
    "uncertain_status_flag",
    "missing_price_flag",
    "missing_position_flag",
    "missing_team_context_flag",
    "missing_fixture_context_flag",
    "prediction_write_allowed",
    "production_ready",
    "requires_player_feature_manifest_before_prediction",
]

REQUIRED_SCORELINE_COLUMNS = [
    "source_season",
    "target_season",
    "target_gw",
    "prediction_mode",
    "prediction_scope",
    "model_name",
    "model_version",
    "fixture_id",
    "home_team_id",
    "away_team_id",
    "home_team_short_name",
    "away_team_short_name",
    "any_team_fallback_applied",
    "expected_home_goals",
    "expected_away_goals",
    "scoreline_home_win_probability",
    "scoreline_draw_probability",
    "scoreline_away_win_probability",
    "prediction_write_allowed",
    "production_ready",
    "requires_scoreline_manifest_before_write",
    "calibration_status",
    "guardrail_status",
]

REQUIRED_OUTPUT_COLUMNS = [
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
    "chance_of_playing_next_round",
    "news",
    "news_added",
    "official_availability_probability",
    "official_availability_workload_factor",
    "official_availability_adjustment_applied",
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


METRIC_COLUMNS = [
    "appearance_probability_proxy",
    "start_probability_proxy",
    "conditional_minutes_if_appears_proxy",
    "expected_minutes_per_fixture_proxy",
    "points_per90_proxy",
    "goals_per90_proxy",
    "assists_per90_proxy",
    "bonus_per90_proxy",
    "clean_sheets_per90_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only Pre-GW1 player prediction preview from Day71A "
            "player features, the Day71B role/minutes contract, and the Day70C "
            "scoreline preview. This is transparent heuristic scaffolding only; "
            "it does not train a model or write to the database."
        )
    )
    parser.add_argument(
        "--source-season",
        dest="source_seasons",
        action="append",
        required=True,
        help=(
            "Historical source season. Repeat for multiple lookback seasons, "
            "ordered oldest to newest. Day72A v0 expects the Day71A artifact to "
            "represent the newest declared source season."
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

    parser.add_argument("--player-features-csv", required=True)
    parser.add_argument("--day71a-json", required=True)
    parser.add_argument("--day71b-json", required=True)
    parser.add_argument("--scoreline-preview-csv", required=True)
    parser.add_argument("--day70c-json", default="")

    parser.add_argument("--player-feature-version", default="day71a_v0_1")
    parser.add_argument(
        "--role-contract-version",
        default=DAY71B_CONTRACT_VERSION,
    )
    parser.add_argument(
        "--threshold-policy-version",
        default=DAY71B_THRESHOLD_POLICY_VERSION,
    )
    parser.add_argument(
        "--scoreline-model-name",
        default="pre_gw1_scoreline_prior_heuristic_v0",
    )
    parser.add_argument(
        "--scoreline-model-version",
        default="day70c_v0",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument(
        "--scoring-rules-version",
        default="target_season_rules_unresolved",
    )

    parser.add_argument("--source-team-matches", type=int, default=38)
    parser.add_argument(
        "--min-price-band-reliable-samples",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--fixture-signal-min",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--fixture-signal-max",
        type=float,
        default=1.25,
    )
    parser.add_argument(
        "--fixture-multiplier-min",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--fixture-multiplier-max",
        type=float,
        default=1.15,
    )
    parser.add_argument(
        "--prediction-points-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--prediction-points-max",
        type=float,
        default=15.0,
    )

    parser.add_argument("--out-csv", required=True)
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
        raise RuntimeError("%s must be valid ISO-8601: %s" % (label, value)) from exc
    if parsed.tzinfo is None:
        raise RuntimeError("%s must include a timezone offset." % label)
    return parsed


def nullable_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nullable_int(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def apply_official_availability_adjustment(
    appearance_probability: float,
    start_probability: float,
    expected_minutes: float,
    chance_of_playing_next_round: Any,
) -> Dict[str, Any]:
    """Apply official FPL availability as a conservative workload ceiling.

    A non-null official percentage may reduce the model's unconditional
    appearance probability. It never raises a more conservative model estimate.
    When the official ceiling binds, start probability and expected minutes are
    scaled by the same factor so the existing conditional role assumption is
    preserved.
    """
    baseline_appearance = clamp(float(appearance_probability), 0.0, 1.0)
    baseline_start = clamp(
        float(start_probability),
        0.0,
        baseline_appearance,
    )
    baseline_minutes = clamp(float(expected_minutes), 0.0, 90.0)
    chance_percent = nullable_float(chance_of_playing_next_round)

    result: Dict[str, Any] = {
        "appearance_probability": baseline_appearance,
        "start_probability": baseline_start,
        "expected_minutes": baseline_minutes,
        "official_availability_probability": None,
        "official_availability_workload_factor": 1.0,
        "official_availability_adjustment_applied": False,
    }

    if chance_percent is None:
        return result
    if chance_percent < 0.0 or chance_percent > 100.0:
        raise RuntimeError(
            "chance_of_playing_next_round must be null or within 0..100."
        )

    official_probability = chance_percent / 100.0
    result["official_availability_probability"] = official_probability

    if (
        baseline_appearance <= 1e-12
        or official_probability >= baseline_appearance - 1e-12
    ):
        return result

    workload_factor = official_probability / baseline_appearance
    result.update(
        {
            "appearance_probability": official_probability,
            "start_probability": clamp(
                baseline_start * workload_factor,
                0.0,
                official_probability,
            ),
            "expected_minutes": clamp(
                baseline_minutes * workload_factor,
                0.0,
                90.0,
            ),
            "official_availability_workload_factor": workload_factor,
            "official_availability_adjustment_applied": True,
        }
    )
    return result


def sha256_file(path_value: str) -> Optional[str]:
    path = Path(path_value)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path_value: str, required: bool = True) -> Dict[str, Any]:
    if not path_value:
        if required:
            raise RuntimeError("Required JSON path is empty.")
        return {}
    path = Path(path_value)
    if not path.exists():
        if required:
            raise RuntimeError("JSON file does not exist: %s" % path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("JSON root must be an object: %s" % path)
    return data


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


def load_csv(
    path_value: str,
    required_columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("%s CSV does not exist: %s" % (label, path))
    df = pd.read_csv(path, low_memory=False)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise RuntimeError("%s is missing required columns: %s" % (label, missing))
    return df


def filter_source_target_gw(
    df: pd.DataFrame,
    source_season: str,
    target_season: str,
    target_gw: int,
    label: str,
) -> pd.DataFrame:
    out = df[
        (df["source_season"].astype(str) == str(source_season))
        & (df["target_season"].astype(str) == str(target_season))
        & (pd.to_numeric(df["target_gw"], errors="coerce") == int(target_gw))
    ].copy()
    if out.empty:
        raise RuntimeError(
            "%s has no rows for source_season=%s target_season=%s target_gw=%s."
            % (label, source_season, target_season, target_gw)
        )
    return out


def dataframe_count_dict(series: pd.Series) -> Dict[str, int]:
    counts = series.fillna("<missing>").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.to_dict().items()}


def validate_args(args: argparse.Namespace) -> List[str]:
    blockers: List[str] = []
    if args.target_gw != 1:
        blockers.append("Day72A v0 expects target_gw=1.")
    if args.source_team_matches <= 0:
        blockers.append("--source-team-matches must be positive.")
    if args.min_price_band_reliable_samples <= 0:
        blockers.append("--min-price-band-reliable-samples must be positive.")
    if args.fixture_signal_min <= 0:
        blockers.append("--fixture-signal-min must be positive.")
    if args.fixture_signal_max < args.fixture_signal_min:
        blockers.append("fixture signal max must be >= fixture signal min.")
    if args.fixture_multiplier_max < args.fixture_multiplier_min:
        blockers.append("fixture multiplier max must be >= fixture multiplier min.")
    if args.prediction_points_max <= args.prediction_points_min:
        blockers.append("prediction points max must exceed prediction points min.")
    if not args.source_seasons:
        blockers.append("At least one --source-season is required.")
    return blockers


def role_namespace_from_contract(day71b_report: Dict[str, Any]) -> SimpleNamespace:
    thresholds = day71b_report.get("thresholds") or {}
    sample = thresholds.get("sample_reliability") or {}
    role = thresholds.get("role_proxy") or {}

    def require(path: Sequence[str]) -> Any:
        value = nested_get(thresholds, path, None)
        if value is None:
            raise RuntimeError(
                "Day71B contract is missing threshold path: %s"
                % ".".join(path)
            )
        return value

    return SimpleNamespace(
        very_low_sample_minutes=float(
            require(["sample_reliability", "very_low_sample", "minutes_below"])
        ),
        very_low_sample_appearances=float(
            require(["sample_reliability", "very_low_sample", "appearances_below"])
        ),
        low_sample_minutes=float(
            require(["sample_reliability", "low_sample", "minutes_below"])
        ),
        low_sample_appearances=float(
            require(["sample_reliability", "low_sample", "appearances_below"])
        ),
        reliable_sample_minutes=float(
            require(["sample_reliability", "reliable_sample", "minutes_at_least"])
        ),
        high_sample_minutes=float(
            require(["sample_reliability", "high_sample", "minutes_at_least"])
        ),
        bench_min_appearances=float(
            require(["role_proxy", "bench_role_proxy", "appearances_at_least"])
        ),
        bench_max_start_rate=float(
            require(["role_proxy", "bench_role_proxy", "start_rate_below"])
        ),
        bench_max_minutes_per_appearance=float(
            require(
                [
                    "role_proxy",
                    "bench_role_proxy",
                    "minutes_per_appearance_below",
                ]
            )
        ),
        probable_starter_min_minutes=float(
            require(["role_proxy", "probable_starter_proxy", "minutes_at_least"])
        ),
        probable_starter_min_appearances=float(
            require(
                ["role_proxy", "probable_starter_proxy", "appearances_at_least"]
            )
        ),
        probable_starter_min_start_rate=float(
            require(["role_proxy", "probable_starter_proxy", "start_rate_at_least"])
        ),
        probable_starter_min_minutes_per_appearance=float(
            require(
                [
                    "role_proxy",
                    "probable_starter_proxy",
                    "minutes_per_appearance_at_least",
                ]
            )
        ),
        established_starter_min_minutes=float(
            require(
                ["role_proxy", "established_starter_proxy", "minutes_at_least"]
            )
        ),
        established_starter_min_appearances=float(
            require(
                [
                    "role_proxy",
                    "established_starter_proxy",
                    "appearances_at_least",
                ]
            )
        ),
        established_starter_min_start_rate=float(
            require(
                ["role_proxy", "established_starter_proxy", "start_rate_at_least"]
            )
        ),
        established_starter_min_minutes_per_appearance=float(
            require(
                [
                    "role_proxy",
                    "established_starter_proxy",
                    "minutes_per_appearance_at_least",
                ]
            )
        ),
    )


def build_price_bands(features: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    out = features.copy()
    out["price_numeric"] = pd.to_numeric(out["price"], errors="coerce")
    out["position_price_band"] = "unclassified"
    threshold_rows: List[Dict[str, Any]] = []

    for position, group in out.groupby("position"):
        prices = group["price_numeric"]
        q25 = float(prices.quantile(0.25))
        q50 = float(prices.quantile(0.50))
        q75 = float(prices.quantile(0.75))
        threshold_rows.append(
            {
                "position": str(position),
                "min": float(prices.min()),
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "max": float(prices.max()),
            }
        )
        conditions = [
            prices.le(q25),
            prices.gt(q25) & prices.le(q50),
            prices.gt(q50) & prices.le(q75),
            prices.gt(q75),
        ]
        bands = np.select(
            conditions,
            PRICE_BAND_LABELS,
            default="unclassified",
        )
        out.loc[group.index, "position_price_band"] = bands

    out["position_price_band_percentile"] = (
        out.groupby(
            ["position", "position_price_band"],
            dropna=False,
        )["price_numeric"]
        .rank(method="average", pct=True)
        .fillna(0.5)
        .clip(0.0, 1.0)
    )
    out["fallback_rate_price_signal"] = (
        FALLBACK_RATE_PRICE_SIGNAL_MIN
        + (
            FALLBACK_RATE_PRICE_SIGNAL_MAX
            - FALLBACK_RATE_PRICE_SIGNAL_MIN
        )
        * out["position_price_band_percentile"]
    )
    out["fallback_minutes_price_signal"] = (
        FALLBACK_MINUTES_PRICE_SIGNAL_MIN
        + (
            FALLBACK_MINUTES_PRICE_SIGNAL_MAX
            - FALLBACK_MINUTES_PRICE_SIGNAL_MIN
        )
        * out["position_price_band_percentile"]
    )

    return out, sorted(threshold_rows, key=lambda row: row["position"])


def add_prior_proxy_metrics(
    features: pd.DataFrame,
    source_team_matches: int,
) -> pd.DataFrame:
    out = features.copy()
    numeric_sources = {
        "prior_minutes": "prev_season_minutes",
        "prior_appearances": "prev_season_appearances",
        "prior_starts": "prev_season_starts_proxy",
        "points_per90_proxy": "prev_season_points_per90",
        "prior_goals": "prev_season_goals",
        "prior_assists": "prev_season_assists",
        "prior_bonus": "prev_season_bonus",
        "prior_clean_sheets": "prev_season_clean_sheets",
    }
    for target, source in numeric_sources.items():
        out[target] = pd.to_numeric(out[source], errors="coerce").astype(float)

    denominator = float(source_team_matches)
    positive_appearances = out["prior_appearances"].where(
        out["prior_appearances"] > 0
    )
    positive_minutes = out["prior_minutes"].where(out["prior_minutes"] > 0)

    out["appearance_probability_proxy"] = (
        out["prior_appearances"] / denominator
    ).clip(0.0, 1.0)
    out["start_probability_proxy"] = (
        out["prior_starts"] / denominator
    ).clip(0.0, 1.0)
    out["conditional_minutes_if_appears_proxy"] = (
        out["prior_minutes"] / positive_appearances
    ).clip(0.0, 90.0)
    out["expected_minutes_per_fixture_proxy"] = (
        out["prior_minutes"] / denominator
    ).clip(0.0, 90.0)
    out["goals_per90_proxy"] = out["prior_goals"] * 90.0 / positive_minutes
    out["assists_per90_proxy"] = (
        out["prior_assists"] * 90.0 / positive_minutes
    )
    out["bonus_per90_proxy"] = out["prior_bonus"] * 90.0 / positive_minutes
    out["clean_sheets_per90_proxy"] = (
        out["prior_clean_sheets"] * 90.0 / positive_minutes
    )
    return out


def reliable_fallback_pool_mask(
    features: pd.DataFrame,
    role_args: SimpleNamespace,
) -> pd.Series:
    has_prior = features["has_prev_season_player_prior"].apply(bool_value)
    minutes = pd.to_numeric(features["prev_season_minutes"], errors="coerce")
    appearances = pd.to_numeric(
        features["prev_season_appearances"],
        errors="coerce",
    )
    return (
        has_prior
        & minutes.ge(role_args.reliable_sample_minutes)
        & appearances.ge(role_args.probable_starter_min_appearances)
    )


def build_fallback_tables(
    features: pd.DataFrame,
    role_args: SimpleNamespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    reliable = features[reliable_fallback_pool_mask(features, role_args)].copy()
    if reliable.empty:
        raise RuntimeError("Reliable fallback pool is empty.")

    grouped = reliable.groupby(["position", "position_price_band"], dropna=False)
    band_medians = grouped[METRIC_COLUMNS].median().reset_index()
    band_counts = grouped.size().rename("reliable_sample_count").reset_index()
    band_table = band_medians.merge(
        band_counts,
        on=["position", "position_price_band"],
        how="left",
        validate="one_to_one",
    )

    position_grouped = reliable.groupby("position", dropna=False)
    position_table = position_grouped[METRIC_COLUMNS].median().reset_index()
    position_counts = (
        position_grouped.size().rename("reliable_sample_count").reset_index()
    )
    position_table = position_table.merge(
        position_counts,
        on="position",
        how="left",
        validate="one_to_one",
    )

    pool_counts = {
        str(key): int(value)
        for key, value in reliable["position"].value_counts().to_dict().items()
    }
    return band_table, position_table, pool_counts


def row_from_table(
    table: pd.DataFrame,
    position: str,
    price_band: Optional[str] = None,
) -> Optional[pd.Series]:
    mask = table["position"].astype(str) == str(position)
    if price_band is not None:
        mask = mask & (
            table["position_price_band"].astype(str) == str(price_band)
        )
    matches = table[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def fallback_for_player(
    row: pd.Series,
    band_table: pd.DataFrame,
    position_table: pd.DataFrame,
    min_band_samples: int,
) -> Dict[str, Any]:
    position = str(row["position"])
    price_band = str(row["position_price_band"])
    band_row = row_from_table(band_table, position, price_band)
    band_count = (
        int(band_row["reliable_sample_count"])
        if band_row is not None
        else 0
    )

    if band_row is not None and band_count >= min_band_samples:
        selected = band_row
        level = 4
        policy = "position_price_band_prior"
        reason = (
            "Reliable position/price-band median with sample_count=%s."
            % band_count
        )
        position_only_multiplier = 1.0
    else:
        selected = row_from_table(position_table, position)
        if selected is None:
            raise RuntimeError("No position fallback available for %s." % position)
        level = 5
        policy = "position_baseline_prior"
        reason = (
            "Position/price-band cell was sparse (sample_count=%s, minimum=%s); "
            "used conservative position baseline."
            % (band_count, min_band_samples)
        )
        position_only_multiplier = POSITION_ONLY_FALLBACK_MINUTES_MULTIPLIER[
            position
        ]

    metrics = {
        metric: nullable_float(selected.get(metric))
        for metric in METRIC_COLUMNS
    }

    # Sparse position-only role fallback receives a conservative minutes and
    # probability reduction. Points/event rates are not numerically inflated.
    for metric in [
        "appearance_probability_proxy",
        "start_probability_proxy",
        "expected_minutes_per_fixture_proxy",
    ]:
        value = metrics.get(metric)
        if value is not None:
            metrics[metric] = value * position_only_multiplier

    if position == "GKP":
        metrics["conditional_minutes_if_appears_proxy"] = 90.0
        app_value = metrics.get("appearance_probability_proxy") or 0.0
        metrics["expected_minutes_per_fixture_proxy"] = 90.0 * app_value
        start_value = metrics.get("start_probability_proxy") or 0.0
        metrics["start_probability_proxy"] = min(app_value, start_value)

    return {
        "metrics": metrics,
        "fallback_level": level,
        "fallback_policy_used": policy,
        "fallback_reason": reason,
        "fallback_reference_sample_count": int(
            selected["reliable_sample_count"]
        ),
        "price_band_reliable_sample_count": band_count,
        "sparse_price_band_fallback": level == 5,
    }


def blended_value(
    player_value: Optional[float],
    fallback_value: Optional[float],
    player_weight: float,
) -> float:
    if player_value is None or not math.isfinite(player_value):
        player_weight = 0.0
    if fallback_value is None or not math.isfinite(fallback_value):
        if player_value is None or not math.isfinite(player_value):
            return 0.0
        return float(player_value)
    if player_value is None or not math.isfinite(player_value):
        return float(fallback_value)
    return float(player_weight * player_value + (1.0 - player_weight) * fallback_value)


def confidence_rank(value: str) -> int:
    order = {
        "very_low": 0,
        "low": 1,
        "medium": 2,
        "medium_high": 3,
        "high": 4,
    }
    return order.get(value, 0)


def confidence_from_rank(rank: int) -> str:
    values = ["very_low", "low", "medium", "medium_high", "high"]
    return values[max(0, min(len(values) - 1, rank))]


def downgrade_confidence(value: str, levels: int = 1) -> str:
    return confidence_from_rank(confidence_rank(value) - levels)


def base_role_confidence(sample_tier: str) -> str:
    return {
        "high_sample": "high",
        "reliable_sample": "medium_high",
        "moderate_sample": "medium",
        "low_sample": "low",
        "very_low_sample": "low",
        "zero_sample": "very_low",
        "no_safe_prior": "low",
    }.get(sample_tier, "very_low")



def fallback_role_uncertainty_multiplier(
    sample_tier: str,
    role_proxy: str,
) -> float:
    if role_proxy not in {"no_safe_prior", "insufficient_sample"}:
        return 1.0
    return float(
        FALLBACK_ROLE_UNCERTAINTY_MULTIPLIER_BY_SAMPLE_TIER.get(
            sample_tier,
            0.75,
        )
    )


def build_goalkeeper_competition_caps(
    features: pd.DataFrame,
) -> pd.DataFrame:
    goalkeepers = features[
        features["position"].astype(str) == "GKP"
    ].copy()
    if goalkeepers.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "goalkeeper_team_candidate_count",
                "goalkeeper_team_share_cap",
            ]
        )

    goalkeepers["price_numeric"] = pd.to_numeric(
        goalkeepers["price_numeric"],
        errors="coerce",
    ).fillna(0.0)
    goalkeepers["prior_appearance_signal"] = pd.to_numeric(
        goalkeepers.get("appearance_probability_proxy"),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)

    rows: List[Dict[str, Any]] = []
    for team_id, group in goalkeepers.groupby("team_id", dropna=False):
        max_price = float(group["price_numeric"].max())
        weights = []
        for _, row in group.iterrows():
            price_weight = math.exp(
                GOALKEEPER_PRICE_STRENGTH
                * (float(row["price_numeric"]) - max_price)
            )
            prior_weight = 0.65 + 0.35 * float(
                row["prior_appearance_signal"]
            )
            weights.append(max(1e-9, price_weight * prior_weight))

        total_weight = float(sum(weights))
        candidate_count = int(len(group))
        for (_, row), weight in zip(group.iterrows(), weights):
            share = (
                GOALKEEPER_TEAM_APPEARANCE_BUDGET
                * float(weight)
                / total_weight
            )
            rows.append(
                {
                    "player_id": nullable_int(row.get("player_id")),
                    "goalkeeper_team_candidate_count": candidate_count,
                    "goalkeeper_team_share_cap": round(share, 8),
                }
            )

    return pd.DataFrame(rows)


def build_risk_flags(
    row: pd.Series,
    sample_tier: str,
    role_proxy: str,
    fallback: Dict[str, Any],
    scoreline_fallback: bool,
    status_cutoff_valid: bool,
) -> List[str]:
    flags: List[str] = []

    def add_if(condition: bool, name: str) -> None:
        if condition and name not in flags:
            flags.append(name)

    add_if(bool_value(row.get("no_prior_flag")), "no_prior_flag")
    add_if(sample_tier == "zero_sample", "zero_prior_sample_flag")
    add_if(sample_tier == "very_low_sample", "very_low_prior_sample_flag")
    add_if(sample_tier == "low_sample", "low_prior_sample_flag")
    add_if(role_proxy == "insufficient_sample", "insufficient_sample_flag")
    add_if(role_proxy == "bench_role_proxy", "bench_role_proxy_flag")
    add_if(
        role_proxy == "rotation_or_unclear_role",
        "rotation_or_unclear_role_flag",
    )
    add_if(
        bool_value(row.get("promoted_team_player_flag")),
        "promoted_team_player_flag",
    )
    add_if(
        bool_value(row.get("opponent_promoted_team_flag")),
        "opponent_promoted_team_flag",
    )
    add_if(bool_value(row.get("uncertain_status_flag")), "uncertain_status_flag")
    add_if(
        bool_value(row.get("team_fallback_applied")),
        "team_context_fallback_flag",
    )
    add_if(
        bool_value(row.get("opponent_fallback_applied")),
        "opponent_context_fallback_flag",
    )
    add_if(scoreline_fallback, "scoreline_team_fallback_flag")
    add_if(
        bool(fallback.get("sparse_price_band_fallback")),
        "sparse_price_band_fallback_flag",
    )
    add_if(
        str(row.get("position")) == "GKP"
        and int(nullable_int(row.get("goalkeeper_team_candidate_count")) or 0) > 1,
        "goalkeeper_competition_normalized_flag",
    )
    add_if(not status_cutoff_valid, "status_unknown_for_as_of_flag")
    return flags


def build_preview(
    features: pd.DataFrame,
    scorelines: pd.DataFrame,
    args: argparse.Namespace,
    role_args: SimpleNamespace,
    day71b_report: Dict[str, Any],
    run_id: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    classified = build_sample_and_role_preview(features, role_args)
    classified, price_thresholds = build_price_bands(classified)
    classified = add_prior_proxy_metrics(
        classified,
        source_team_matches=args.source_team_matches,
    )

    band_table, position_table, fallback_pool_counts = build_fallback_tables(
        classified,
        role_args,
    )

    scoreline_subset = scorelines[
        [
            "fixture_id",
            "model_name",
            "model_version",
            "home_team_id",
            "away_team_id",
            "home_team_short_name",
            "away_team_short_name",
            "any_team_fallback_applied",
            "expected_home_goals",
            "expected_away_goals",
            "scoreline_home_win_probability",
            "scoreline_draw_probability",
            "scoreline_away_win_probability",
            "data_quality_status",
        ]
    ].copy()
    scoreline_subset = scoreline_subset.rename(
        columns={
            "model_name": "scoreline_model_name_input",
            "model_version": "scoreline_model_version_input",
            "data_quality_status": "scoreline_data_quality_status",
        }
    )
    merged = classified.merge(
        scoreline_subset,
        on="fixture_id",
        how="left",
        validate="many_to_one",
    )
    goalkeeper_caps = build_goalkeeper_competition_caps(merged)
    merged = merged.merge(
        goalkeeper_caps,
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    merged["goalkeeper_team_candidate_count"] = (
        pd.to_numeric(
            merged["goalkeeper_team_candidate_count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    merged["goalkeeper_team_share_cap"] = pd.to_numeric(
        merged["goalkeeper_team_share_cap"],
        errors="coerce",
    )

    all_team_expected_goals = pd.concat(
        [
            pd.to_numeric(scorelines["expected_home_goals"], errors="coerce"),
            pd.to_numeric(scorelines["expected_away_goals"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    league_mean_expected_goals = float(all_team_expected_goals.mean())

    all_team_clean_sheet_probabilities = pd.concat(
        [
            pd.to_numeric(scorelines["expected_away_goals"], errors="coerce").apply(
                lambda value: math.exp(-float(value)) if pd.notna(value) else np.nan
            ),
            pd.to_numeric(scorelines["expected_home_goals"], errors="coerce").apply(
                lambda value: math.exp(-float(value)) if pd.notna(value) else np.nan
            ),
        ],
        ignore_index=True,
    ).dropna()
    league_mean_clean_sheet_probability = float(
        all_team_clean_sheet_probabilities.mean()
    )

    status_cutoff_valid = bool(
        nested_get(
            day71b_report,
            ["status_policy", "status_valid_for_prediction_cutoff"],
            False,
        )
    )
    status_hard_guardrail_enabled = bool(
        nested_get(
            day71b_report,
            ["status_policy", "status_hard_guardrail_enabled"],
            False,
        )
    )

    rows: List[Dict[str, Any]] = []
    fallback_level_counts: Dict[str, int] = {}
    sparse_band_rows = 0
    scoreline_fallback_player_rows = 0
    guardrail_clipped_rows = 0
    goalkeeper_competition_adjusted_rows = 0
    official_availability_adjusted_rows = 0

    source_seasons_value = ",".join(args.source_seasons)

    for _, row in merged.iterrows():
        position = str(row["position"])
        sample_tier = str(row["sample_reliability_tier_v0"])
        role_proxy = str(row["role_proxy_v0"])
        has_safe_prior = bool_value(row.get("has_prev_season_player_prior"))
        position_price_band_percentile = clamp(
            nullable_float(row.get("position_price_band_percentile")) or 0.5,
            0.0,
            1.0,
        )
        fallback_rate_price_signal = clamp(
            nullable_float(row.get("fallback_rate_price_signal")) or 1.0,
            FALLBACK_RATE_PRICE_SIGNAL_MIN,
            FALLBACK_RATE_PRICE_SIGNAL_MAX,
        )
        fallback_minutes_price_signal = clamp(
            nullable_float(row.get("fallback_minutes_price_signal")) or 1.0,
            FALLBACK_MINUTES_PRICE_SIGNAL_MIN,
            FALLBACK_MINUTES_PRICE_SIGNAL_MAX,
        )
        fallback_role_multiplier = fallback_role_uncertainty_multiplier(
            sample_tier=sample_tier,
            role_proxy=role_proxy,
        )

        fallback = fallback_for_player(
            row,
            band_table=band_table,
            position_table=position_table,
            min_band_samples=args.min_price_band_reliable_samples,
        )
        fallback_metrics = fallback["metrics"]

        role_weight = PLAYER_PRIOR_WEIGHT_BY_SAMPLE_TIER.get(sample_tier, 0.0)
        points_weight = POINTS_PRIOR_WEIGHT_BY_SAMPLE_TIER.get(sample_tier, 0.0)
        if not has_safe_prior:
            role_weight = 0.0
            points_weight = 0.0

        appearance_probability = blended_value(
            nullable_float(row.get("appearance_probability_proxy")),
            fallback_metrics.get("appearance_probability_proxy"),
            role_weight,
        )
        start_probability = blended_value(
            nullable_float(row.get("start_probability_proxy")),
            fallback_metrics.get("start_probability_proxy"),
            role_weight,
        )
        conditional_minutes_if_appears = blended_value(
            nullable_float(row.get("conditional_minutes_if_appears_proxy")),
            fallback_metrics.get("conditional_minutes_if_appears_proxy"),
            role_weight,
        )
        expected_minutes = blended_value(
            nullable_float(row.get("expected_minutes_per_fixture_proxy")),
            fallback_metrics.get("expected_minutes_per_fixture_proxy"),
            role_weight,
        )

        fallback_workload_factor = 1.0
        if role_proxy in {"no_safe_prior", "insufficient_sample"}:
            fallback_workload_factor = (
                fallback_role_multiplier
                * fallback_minutes_price_signal
            )
            appearance_probability *= fallback_workload_factor
            start_probability *= fallback_workload_factor
            expected_minutes *= fallback_workload_factor

        goalkeeper_competition_adjustment = 0.0
        if position == "GKP":
            conditional_minutes_if_appears = 90.0
            provisional_appearance_probability = appearance_probability
            share_cap = nullable_float(
                row.get("goalkeeper_team_share_cap")
            )
            if share_cap is not None:
                appearance_probability = min(
                    appearance_probability,
                    share_cap,
                )
            goalkeeper_competition_adjustment = (
                appearance_probability
                - provisional_appearance_probability
            )
            if goalkeeper_competition_adjustment < -1e-12:
                goalkeeper_competition_adjusted_rows += 1
            start_probability = min(start_probability, appearance_probability)
            expected_minutes = 90.0 * appearance_probability
        else:
            start_probability = min(start_probability, appearance_probability)
            conditional_minutes_if_appears = clamp(
                conditional_minutes_if_appears,
                0.0,
                90.0,
            )
            expected_minutes = clamp(expected_minutes, 0.0, 90.0)

        appearance_probability = clamp(appearance_probability, 0.0, 1.0)
        start_probability = clamp(start_probability, 0.0, appearance_probability)

        availability_adjustment = apply_official_availability_adjustment(
            appearance_probability=appearance_probability,
            start_probability=start_probability,
            expected_minutes=expected_minutes,
            chance_of_playing_next_round=(
                row.get("chance_of_playing_next_round")
                if status_cutoff_valid
                else None
            ),
        )
        appearance_probability = float(
            availability_adjustment["appearance_probability"]
        )
        start_probability = float(
            availability_adjustment["start_probability"]
        )
        expected_minutes = float(
            availability_adjustment["expected_minutes"]
        )
        if availability_adjustment[
            "official_availability_adjustment_applied"
        ]:
            official_availability_adjusted_rows += 1

        points_per90 = blended_value(
            nullable_float(row.get("points_per90_proxy")),
            fallback_metrics.get("points_per90_proxy"),
            points_weight,
        )
        goals_per90 = blended_value(
            nullable_float(row.get("goals_per90_proxy")),
            fallback_metrics.get("goals_per90_proxy"),
            points_weight,
        )
        assists_per90 = blended_value(
            nullable_float(row.get("assists_per90_proxy")),
            fallback_metrics.get("assists_per90_proxy"),
            points_weight,
        )
        bonus_per90 = blended_value(
            nullable_float(row.get("bonus_per90_proxy")),
            fallback_metrics.get("bonus_per90_proxy"),
            points_weight,
        )
        clean_sheets_per90 = blended_value(
            nullable_float(row.get("clean_sheets_per90_proxy")),
            fallback_metrics.get("clean_sheets_per90_proxy"),
            points_weight,
        )

        if role_proxy in {"no_safe_prior", "insufficient_sample"}:
            points_per90 *= fallback_rate_price_signal
            goals_per90 *= fallback_rate_price_signal
            assists_per90 *= fallback_rate_price_signal
            bonus_per90 *= fallback_rate_price_signal

        is_home = bool_value(row.get("is_home"))
        expected_home_goals = nullable_float(row.get("expected_home_goals"))
        expected_away_goals = nullable_float(row.get("expected_away_goals"))
        if expected_home_goals is None or expected_away_goals is None:
            expected_team_goals = league_mean_expected_goals
            expected_opponent_goals = league_mean_expected_goals
        elif is_home:
            expected_team_goals = expected_home_goals
            expected_opponent_goals = expected_away_goals
        else:
            expected_team_goals = expected_away_goals
            expected_opponent_goals = expected_home_goals

        clean_sheet_probability = math.exp(-expected_opponent_goals)
        attack_signal = clamp(
            expected_team_goals / league_mean_expected_goals,
            args.fixture_signal_min,
            args.fixture_signal_max,
        )
        defence_signal = clamp(
            clean_sheet_probability / league_mean_clean_sheet_probability,
            args.fixture_signal_min,
            args.fixture_signal_max,
        )
        weights = FIXTURE_COMPONENT_WEIGHTS[position]
        neutral_weight = 1.0 - weights["attack"] - weights["defence"]
        fixture_multiplier = (
            neutral_weight
            + weights["attack"] * attack_signal
            + weights["defence"] * defence_signal
        )
        fixture_multiplier = clamp(
            fixture_multiplier,
            args.fixture_multiplier_min,
            args.fixture_multiplier_max,
        )

        raw_expected_points = points_per90 * expected_minutes / 90.0
        pre_guardrail_points = raw_expected_points * fixture_multiplier
        guarded_points = clamp(
            pre_guardrail_points,
            args.prediction_points_min,
            args.prediction_points_max,
        )
        guardrail_adjustment = guarded_points - pre_guardrail_points
        if abs(guardrail_adjustment) > 1e-12:
            guardrail_clipped_rows += 1

        expected_appearance_points = appearance_probability + start_probability
        expected_goals = (
            goals_per90 * expected_minutes / 90.0 * attack_signal
        )
        expected_goal_points = expected_goals * GOAL_POINTS_BY_POSITION[position]
        expected_assists = (
            assists_per90 * expected_minutes / 90.0 * attack_signal
        )
        expected_assist_points = expected_assists * 3.0
        expected_clean_sheet_points = (
            clean_sheet_probability
            * start_probability
            * CLEAN_SHEET_POINTS_BY_POSITION[position]
        )
        expected_bonus = bonus_per90 * expected_minutes / 90.0
        component_known_sum = (
            expected_appearance_points
            + expected_goal_points
            + expected_assist_points
            + expected_clean_sheet_points
            + expected_bonus
        )
        expected_other_points = guarded_points - component_known_sum

        scoreline_fallback = bool_value(row.get("any_team_fallback_applied"))
        if scoreline_fallback:
            scoreline_fallback_player_rows += 1

        risks = build_risk_flags(
            row=row,
            sample_tier=sample_tier,
            role_proxy=role_proxy,
            fallback=fallback,
            scoreline_fallback=scoreline_fallback,
            status_cutoff_valid=status_cutoff_valid,
        )
        if (
            availability_adjustment[
                "official_availability_adjustment_applied"
            ]
            and "official_availability_adjusted_flag" not in risks
        ):
            risks.append("official_availability_adjusted_flag")

        role_confidence = base_role_confidence(sample_tier)
        if fallback["fallback_level"] == 5:
            role_confidence = downgrade_confidence(role_confidence, 1)
        if role_proxy == "no_safe_prior":
            role_confidence = "low"
        if role_proxy == "insufficient_sample":
            role_confidence = downgrade_confidence(role_confidence, 1)

        prediction_confidence = role_confidence
        if scoreline_fallback:
            prediction_confidence = downgrade_confidence(
                prediction_confidence,
                1,
            )
        if bool_value(row.get("uncertain_status_flag")):
            prediction_confidence = downgrade_confidence(
                prediction_confidence,
                1,
            )

        uncertainty_half_width = UNCERTAINTY_HALF_WIDTH_BY_SAMPLE_TIER.get(
            sample_tier,
            40.0,
        )
        if fallback["fallback_level"] == 5:
            uncertainty_half_width += 6.0
        if scoreline_fallback:
            uncertainty_half_width += 3.0
        minutes_lower_bound = clamp(
            expected_minutes - uncertainty_half_width,
            0.0,
            90.0,
        )
        minutes_upper_bound = clamp(
            expected_minutes + uncertainty_half_width,
            0.0,
            90.0,
        )

        if has_safe_prior:
            output_fallback_level = 3
            output_fallback_policy = (
                "safe_previous_season_player_prior_blended_with_%s"
                % fallback["fallback_policy_used"]
            )
            output_fallback_reason = (
                "Player prior blended using sample_tier=%s, role_weight=%.2f, "
                "points_weight=%.2f; %s"
                % (
                    sample_tier,
                    role_weight,
                    points_weight,
                    fallback["fallback_reason"],
                )
            )
        else:
            output_fallback_level = int(fallback["fallback_level"])
            output_fallback_policy = str(fallback["fallback_policy_used"])
            output_fallback_reason = (
                "%s Applied unknown-role workload multiplier=%.4f, "
                "minutes_price_signal=%.4f, rate_price_signal=%.4f."
                % (
                    fallback["fallback_reason"],
                    fallback_role_multiplier,
                    fallback_minutes_price_signal,
                    fallback_rate_price_signal,
                )
            )

        fallback_key = str(output_fallback_level)
        fallback_level_counts[fallback_key] = (
            fallback_level_counts.get(fallback_key, 0) + 1
        )
        if fallback["sparse_price_band_fallback"]:
            sparse_band_rows += 1

        role_class = (
            "unknown_role_fallback"
            if role_proxy == "no_safe_prior"
            else role_proxy
        )

        data_quality_status = "full_prior_and_scoreline_context"
        if not has_safe_prior:
            data_quality_status = "no_player_prior_fallback"
        elif sample_tier in {"zero_sample", "very_low_sample", "low_sample"}:
            data_quality_status = "low_sample_player_prior_fallback_blend"
        elif scoreline_fallback:
            data_quality_status = "scoreline_team_context_fallback"

        output_row: Dict[str, Any] = {
            "source_seasons": source_seasons_value,
            "target_season": args.target_season,
            "target_gw": int(args.target_gw),
            "as_of_time": args.as_of_time,
            "prediction_mode": row.get("prediction_mode"),
            "prediction_scope": PREDICTION_SCOPE,
            "run_id": run_id,
            "model_name": args.model_name,
            "model_version": args.model_version,
            "player_feature_version": args.player_feature_version,
            "role_contract_version": args.role_contract_version,
            "threshold_policy_version": args.threshold_policy_version,
            "scoreline_model_name": row.get("scoreline_model_name_input"),
            "scoreline_model_version": row.get("scoreline_model_version_input"),
            "scoring_rules_version": args.scoring_rules_version,
            "player_id": nullable_int(row.get("player_id")),
            "fpl_player_id": nullable_int(row.get("fpl_player_id")),
            "player_name": row.get("player_name"),
            "web_name": row.get("web_name"),
            "team_id": nullable_int(row.get("team_id")),
            "team_name": row.get("team_name"),
            "team_short_name": row.get("team_short_name"),
            "position": position,
            "price": round(float(row.get("price_numeric")), 1),
            "now_cost": int(round(float(row.get("price_numeric")) * 10.0)),
            "status": row.get("status"),
            "chance_of_playing_next_round": nullable_float(
                row.get("chance_of_playing_next_round")
            ),
            "news": row.get("news"),
            "news_added": row.get("news_added"),
            "official_availability_probability": (
                None
                if availability_adjustment[
                    "official_availability_probability"
                ] is None
                else round(
                    float(
                        availability_adjustment[
                            "official_availability_probability"
                        ]
                    ),
                    6,
                )
            ),
            "official_availability_workload_factor": round(
                float(
                    availability_adjustment[
                        "official_availability_workload_factor"
                    ]
                ),
                6,
            ),
            "official_availability_adjustment_applied": bool(
                availability_adjustment[
                    "official_availability_adjustment_applied"
                ]
            ),
            "fixture_id": nullable_int(row.get("fixture_id")),
            "fpl_fixture_id": nullable_int(row.get("fpl_fixture_id")),
            "opponent_team_id": nullable_int(row.get("opponent_team_id")),
            "opponent_team_name": row.get("opponent_team_name"),
            "opponent_short_name": row.get("opponent_short_name"),
            "is_home": is_home,
            "has_fixture": bool_value(row.get("has_fixture")),
            "expected_team_goals": round(expected_team_goals, 4),
            "expected_opponent_goals": round(expected_opponent_goals, 4),
            "fixture_attack_signal": round(attack_signal, 4),
            "fixture_defence_signal": round(defence_signal, 4),
            "fixture_multiplier": round(fixture_multiplier, 4),
            "sample_reliability_tier": sample_tier,
            "role_proxy": role_proxy,
            "role_class": role_class,
            "role_confidence": role_confidence,
            "appearance_probability": round(appearance_probability, 6),
            "start_probability": round(start_probability, 6),
            "conditional_minutes_if_appears": round(
                conditional_minutes_if_appears,
                4,
            ),
            "expected_minutes": round(expected_minutes, 4),
            "minutes_lower_bound": round(minutes_lower_bound, 4),
            "minutes_upper_bound": round(minutes_upper_bound, 4),
            "blended_points_per90": round(points_per90, 6),
            "blended_goals_per90": round(goals_per90, 6),
            "blended_assists_per90": round(assists_per90, 6),
            "blended_bonus_per90": round(bonus_per90, 6),
            "blended_clean_sheets_per90": round(clean_sheets_per90, 6),
            "raw_expected_points": round(raw_expected_points, 6),
            "fixture_adjustment": round(
                pre_guardrail_points - raw_expected_points,
                6,
            ),
            "guardrail_adjustment": round(guardrail_adjustment, 6),
            "calibration_adjustment": 0.0,
            "final_predicted_points": round(guarded_points, 6),
            "predicted_points": round(guarded_points, 6),
            "expected_appearance_points": round(
                expected_appearance_points,
                6,
            ),
            "expected_goals": round(expected_goals, 6),
            "expected_goal_points": round(expected_goal_points, 6),
            "expected_assists": round(expected_assists, 6),
            "expected_assist_points": round(expected_assist_points, 6),
            "clean_sheet_probability": round(clean_sheet_probability, 6),
            "expected_clean_sheet_points": round(
                expected_clean_sheet_points,
                6,
            ),
            "expected_bonus": round(expected_bonus, 6),
            "expected_other_points": round(expected_other_points, 6),
            "has_safe_prior": has_safe_prior,
            "player_prior_role_weight": round(role_weight, 4),
            "player_prior_points_weight": round(points_weight, 4),
            "fallback_policy_used": output_fallback_policy,
            "fallback_level": output_fallback_level,
            "fallback_reason": output_fallback_reason,
            "fallback_reference_sample_count": int(
                fallback["fallback_reference_sample_count"]
            ),
            "position_price_band": row.get("position_price_band"),
            "position_price_band_percentile": round(
                position_price_band_percentile,
                6,
            ),
            "fallback_rate_price_signal": round(
                fallback_rate_price_signal,
                6,
            ),
            "fallback_minutes_price_signal": round(
                fallback_minutes_price_signal,
                6,
            ),
            "fallback_role_uncertainty_multiplier": round(
                fallback_role_multiplier,
                6,
            ),
            "goalkeeper_team_candidate_count": int(
                nullable_int(row.get("goalkeeper_team_candidate_count")) or 0
            ),
            "goalkeeper_team_share_cap": (
                round(
                    float(row.get("goalkeeper_team_share_cap")),
                    6,
                )
                if pd.notna(row.get("goalkeeper_team_share_cap"))
                else None
            ),
            "goalkeeper_competition_adjustment": round(
                goalkeeper_competition_adjustment,
                6,
            ),
            "price_band_reliable_sample_count": int(
                fallback["price_band_reliable_sample_count"]
            ),
            "risk_flags": ",".join(risks),
            "risk_flag_count": len(risks),
            "status_cutoff_valid": status_cutoff_valid,
            "status_hard_guardrail_applied": False,
            "data_quality_status": data_quality_status,
            "prediction_confidence": prediction_confidence,
            "prediction_write_allowed": False,
            "production_ready": False,
            "requires_player_prediction_manifest_before_write": True,
            "calibration_status": "not_calibrated_preview_only",
            "guardrail_status": "basic_bounds_fixture_multiplier_and_role_uncertainty_v0_1",
            "component_accounting_status": COMPONENT_ACCOUNTING_STATUS,
        }
        rows.append(output_row)

    output = pd.DataFrame(rows)
    gkp_team_probability_sums = (
        output[output["position"] == "GKP"]
        .groupby("team_id")["appearance_probability"]
        .sum()
    )
    max_gkp_team_appearance_probability_sum = (
        float(gkp_team_probability_sums.max())
        if len(gkp_team_probability_sums)
        else 0.0
    )
    diagnostics = {
        "price_thresholds": price_thresholds,
        "fallback_pool_counts_by_position": fallback_pool_counts,
        "fallback_level_counts": fallback_level_counts,
        "sparse_price_band_player_rows": sparse_band_rows,
        "scoreline_fallback_player_rows": scoreline_fallback_player_rows,
        "guardrail_clipped_rows": guardrail_clipped_rows,
        "goalkeeper_competition_adjusted_rows": (
            goalkeeper_competition_adjusted_rows
        ),
        "official_availability_adjusted_rows": (
            official_availability_adjusted_rows
        ),
        "max_gkp_team_appearance_probability_sum": round(
            max_gkp_team_appearance_probability_sum,
            8,
        ),
        "league_mean_expected_goals": round(league_mean_expected_goals, 6),
        "league_mean_clean_sheet_probability": round(
            league_mean_clean_sheet_probability,
            6,
        ),
        "band_table": band_table.to_dict(orient="records"),
        "position_table": position_table.to_dict(orient="records"),
    }
    return output, diagnostics


def validate_inputs_and_output(
    args: argparse.Namespace,
    features: pd.DataFrame,
    scorelines: pd.DataFrame,
    preview: pd.DataFrame,
    day71a_report: Dict[str, Any],
    day71b_report: Dict[str, Any],
    day70c_report: Dict[str, Any],
    mode_result: Dict[str, Any],
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    blockers: List[str] = []
    warnings: List[str] = []
    blockers.extend(validate_args(args))
    blockers.extend(list(mode_result.get("errors") or []))
    warnings.extend(list(mode_result.get("warnings") or []))

    resolved_mode = mode_result.get("resolved_prediction_mode")
    if resolved_mode != "pre_gw1_prior":
        blockers.append("Day72A expects resolved_prediction_mode=pre_gw1_prior.")

    if nested_get(day71a_report, ["passed"], None) is not True:
        blockers.append("Day71A report did not pass.")
    if nested_get(day71a_report, ["writes_database"], None) is not False:
        blockers.append("Day71A report must record writes_database=false.")
    if nested_get(day71a_report, ["blockers"], []) or []:
        blockers.append("Day71A report contains blockers.")

    if nested_get(day71b_report, ["passed"], None) is not True:
        blockers.append("Day71B contract did not pass.")
    if nested_get(day71b_report, ["writes_database"], None) is not False:
        blockers.append("Day71B contract must record writes_database=false.")
    if (
        nested_get(
            day71b_report,
            ["ready_for_day72a_prediction_preview_design"],
            None,
        )
        is not True
    ):
        blockers.append("Day71B did not authorize Day72A preview design.")
    if str(day71b_report.get("contract_version")) != args.role_contract_version:
        blockers.append(
            "Day71B contract version mismatch: expected %s, got %s."
            % (args.role_contract_version, day71b_report.get("contract_version"))
        )
    if (
        str(day71b_report.get("threshold_policy_version"))
        != args.threshold_policy_version
    ):
        blockers.append(
            "Day71B threshold policy version mismatch: expected %s, got %s."
            % (
                args.threshold_policy_version,
                day71b_report.get("threshold_policy_version"),
            )
        )
    if nested_get(day71b_report, ["blockers"], []) or []:
        blockers.append("Day71B contract contains blockers.")

    status_hard_guardrail_enabled = bool(
        nested_get(
            day71b_report,
            ["status_policy", "status_hard_guardrail_enabled"],
            False,
        )
    )
    if status_hard_guardrail_enabled:
        warnings.append(
            "Day71B status hard guardrail is enabled. Day72A still records the "
            "status policy but does not apply a numerical status adjustment."
        )
    else:
        warnings.append(
            "Status values are not proven cutoff-valid; Day72A records them as "
            "risk metadata and does not force zero minutes."
        )

    if day70c_report:
        if nested_get(day70c_report, ["passed"], None) is not True:
            blockers.append("Day70C report did not pass.")
        if nested_get(day70c_report, ["writes_database"], None) is not False:
            blockers.append("Day70C report must record writes_database=false.")
        if nested_get(day70c_report, ["blockers"], []) or []:
            blockers.append("Day70C report contains blockers.")

    feature_mode_values = sorted(
        features["prediction_mode"].dropna().astype(str).unique().tolist()
    )
    if feature_mode_values != [str(resolved_mode)]:
        blockers.append(
            "Player feature prediction_mode values %s do not match resolved mode %s."
            % (feature_mode_values, resolved_mode)
        )
    feature_versions = sorted(
        features["feature_version"].dropna().astype(str).unique().tolist()
    )
    if feature_versions != [args.player_feature_version]:
        blockers.append(
            "Player feature version expected %s, got %s."
            % (args.player_feature_version, feature_versions)
        )

    scoreline_modes = sorted(
        scorelines["prediction_mode"].dropna().astype(str).unique().tolist()
    )
    if scoreline_modes != [str(resolved_mode)]:
        blockers.append(
            "Scoreline prediction_mode values %s do not match resolved mode %s."
            % (scoreline_modes, resolved_mode)
        )
    scoreline_names = sorted(
        scorelines["model_name"].dropna().astype(str).unique().tolist()
    )
    if scoreline_names != [args.scoreline_model_name]:
        blockers.append(
            "Scoreline model name expected %s, got %s."
            % (args.scoreline_model_name, scoreline_names)
        )
    scoreline_versions = sorted(
        scorelines["model_version"].dropna().astype(str).unique().tolist()
    )
    if scoreline_versions != [args.scoreline_model_version]:
        blockers.append(
            "Scoreline model version expected %s, got %s."
            % (args.scoreline_model_version, scoreline_versions)
        )

    if features["prediction_write_allowed"].apply(bool_value).any():
        blockers.append("Input player features contain prediction_write_allowed=true.")
    if features["production_ready"].apply(bool_value).any():
        blockers.append("Input player features contain production_ready=true.")
    if (
        ~features["requires_player_feature_manifest_before_prediction"].apply(
            bool_value
        )
    ).any():
        blockers.append(
            "Every input player feature row must require a feature manifest."
        )

    if scorelines["prediction_write_allowed"].apply(bool_value).any():
        blockers.append("Input scorelines contain prediction_write_allowed=true.")
    if scorelines["production_ready"].apply(bool_value).any():
        blockers.append("Input scorelines contain production_ready=true.")
    if (
        ~scorelines["requires_scoreline_manifest_before_write"].apply(bool_value)
    ).any():
        blockers.append("Every input scoreline row must require a manifest.")

    missing_output_columns = [
        column for column in REQUIRED_OUTPUT_COLUMNS if column not in preview.columns
    ]
    if missing_output_columns:
        blockers.append("Preview missing required columns: %s." % missing_output_columns)

    feature_rows = int(len(features))
    preview_rows = int(len(preview))
    if preview_rows != feature_rows:
        blockers.append(
            "Preview row count %s does not equal player feature row count %s."
            % (preview_rows, feature_rows)
        )

    duplicate_player_id_count = int(preview["player_id"].dropna().duplicated().sum())
    if duplicate_player_id_count:
        blockers.append("Preview contains duplicate player_id rows.")

    missing_player_id_count = int(preview["player_id"].isna().sum())
    if missing_player_id_count:
        blockers.append("Preview contains missing player_id values.")

    missing_fixture_join_count = int(preview["scoreline_model_name"].isna().sum())
    if missing_fixture_join_count:
        blockers.append("Some player rows did not join to a scoreline fixture.")

    invalid_positions = sorted(
        set(preview["position"].dropna().astype(str).unique()) - VALID_POSITIONS
    )
    if invalid_positions:
        blockers.append("Preview contains invalid positions: %s." % invalid_positions)

    probability_columns = [
        "appearance_probability",
        "start_probability",
        "clean_sheet_probability",
    ]
    for column in probability_columns:
        values = pd.to_numeric(preview[column], errors="coerce")
        if values.isna().any():
            blockers.append("%s contains missing/non-numeric values." % column)
        if ((values < 0.0) | (values > 1.0)).any():
            blockers.append("%s contains values outside [0, 1]." % column)

    if (
        pd.to_numeric(preview["start_probability"], errors="coerce")
        > pd.to_numeric(preview["appearance_probability"], errors="coerce")
        + 1e-9
    ).any():
        blockers.append("start_probability exceeds appearance_probability.")

    gkp_team_probability_sums = (
        preview[preview["position"].astype(str) == "GKP"]
        .groupby("team_id")["appearance_probability"]
        .sum()
    )
    max_gkp_team_appearance_probability_sum = (
        float(gkp_team_probability_sums.max())
        if len(gkp_team_probability_sums)
        else 0.0
    )
    if (
        max_gkp_team_appearance_probability_sum
        > GOALKEEPER_TEAM_APPEARANCE_BUDGET + 0.00001
    ):
        blockers.append(
            "Goalkeeper team appearance-probability sum exceeds configured budget."
        )

    expected_minutes = pd.to_numeric(preview["expected_minutes"], errors="coerce")
    minutes_lower = pd.to_numeric(preview["minutes_lower_bound"], errors="coerce")
    minutes_upper = pd.to_numeric(preview["minutes_upper_bound"], errors="coerce")
    if expected_minutes.isna().any():
        blockers.append("expected_minutes contains missing/non-numeric values.")
    if ((expected_minutes < 0.0) | (expected_minutes > 90.0)).any():
        blockers.append("expected_minutes contains values outside [0, 90].")
    if (minutes_lower > expected_minutes + 1e-9).any():
        blockers.append("minutes_lower_bound exceeds expected_minutes.")
    if (minutes_upper + 1e-9 < expected_minutes).any():
        blockers.append("minutes_upper_bound is below expected_minutes.")

    points = pd.to_numeric(preview["final_predicted_points"], errors="coerce")
    alias_points = pd.to_numeric(preview["predicted_points"], errors="coerce")
    if points.isna().any():
        blockers.append("final_predicted_points contains missing/non-numeric values.")
    if (
        (points < args.prediction_points_min - 1e-9)
        | (points > args.prediction_points_max + 1e-9)
    ).any():
        blockers.append("final_predicted_points violates configured guardrails.")
    if (points - alias_points).abs().max() > 1e-9:
        blockers.append("predicted_points must equal final_predicted_points.")

    component_columns = [
        "expected_appearance_points",
        "expected_goal_points",
        "expected_assist_points",
        "expected_clean_sheet_points",
        "expected_bonus",
        "expected_other_points",
    ]
    component_sum = preview[component_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).sum(axis=1)
    max_component_reconciliation_error = float((component_sum - points).abs().max())
    if max_component_reconciliation_error > 0.00001:
        blockers.append("Heuristic point components do not reconcile to final points.")

    write_true_count = int(preview["prediction_write_allowed"].apply(bool_value).sum())
    production_true_count = int(preview["production_ready"].apply(bool_value).sum())
    manifest_false_count = int(
        (
            ~preview["requires_player_prediction_manifest_before_write"].apply(
                bool_value
            )
        ).sum()
    )
    status_guard_true_count = int(
        preview["status_hard_guardrail_applied"].apply(bool_value).sum()
    )
    if write_true_count:
        blockers.append("prediction_write_allowed must remain false for all rows.")
    if production_true_count:
        blockers.append("production_ready must remain false for all rows.")
    if manifest_false_count:
        blockers.append(
            "Every preview row must require a player prediction manifest."
        )
    if status_guard_true_count:
        blockers.append(
            "Day72A must not apply a hard status guardrail numerically."
        )

    input_no_prior_count = int(features["no_prior_flag"].apply(bool_value).sum())
    output_no_prior_count = int((~preview["has_safe_prior"].apply(bool_value)).sum())
    if input_no_prior_count != output_no_prior_count:
        blockers.append("No-prior players were not preserved exactly.")

    classification_expected = nested_get(
        day71b_report,
        ["classification_preview", "role_proxy_counts"],
        {},
    ) or {}
    classification_actual = dataframe_count_dict(preview["role_proxy"])
    if classification_expected and classification_expected != classification_actual:
        blockers.append(
            "Day72A role classification counts differ from Day71B contract preview."
        )

    sample_expected = nested_get(
        day71b_report,
        ["classification_preview", "sample_tier_counts"],
        {},
    ) or {}
    sample_actual = dataframe_count_dict(preview["sample_reliability_tier"])
    if sample_expected and sample_expected != sample_actual:
        blockers.append(
            "Day72A sample-tier counts differ from Day71B contract preview."
        )

    warnings.append(
        "Day72A uses a transparent prior-based heuristic, not a trained minutes, "
        "event, or points model."
    )
    warnings.append(
        "Unknown-role and insufficient-sample rows receive conservative workload "
        "multipliers plus weak within-band price signals; these are explicit "
        "fallback heuristics, not role labels."
    )
    warnings.append(
        "Goalkeeper appearance probabilities are softly normalized within each "
        "team to avoid assigning starter-like probability to multiple keepers."
    )
    warnings.append(
        "Position/price-band cells with fewer than %s reliable players fall back "
        "to a conservative position baseline."
        % args.min_price_band_reliable_samples
    )
    if (preview["price_band_reliable_sample_count"] < args.min_price_band_reliable_samples).any():
        warnings.append(
            "Some player rows use sparse price-band fallbacks; inspect fallback "
            "levels and risk flags before any downstream ranking."
        )
    if preview["risk_flags"].astype(str).str.contains(
        "scoreline_team_fallback_flag",
        regex=False,
    ).any():
        warnings.append(
            "Some player rows inherit Day70C team fallback context from promoted "
            "or otherwise missing-prior teams."
        )

    summary = {
        "feature_rows": feature_rows,
        "scoreline_rows": int(len(scorelines)),
        "preview_rows": preview_rows,
        "duplicate_player_id_count": duplicate_player_id_count,
        "missing_player_id_count": missing_player_id_count,
        "missing_fixture_join_count": missing_fixture_join_count,
        "input_no_prior_count": input_no_prior_count,
        "output_no_prior_count": output_no_prior_count,
        "prediction_write_allowed_true_count": write_true_count,
        "production_ready_true_count": production_true_count,
        "requires_manifest_false_count": manifest_false_count,
        "status_hard_guardrail_true_count": status_guard_true_count,
        "max_component_reconciliation_error": max_component_reconciliation_error,
        "max_gkp_team_appearance_probability_sum": (
            max_gkp_team_appearance_probability_sum
        ),
        "goalkeeper_competition_adjusted_rows": int(
            (
                pd.to_numeric(
                    preview["goalkeeper_competition_adjustment"],
                    errors="coerce",
                )
                < -1e-12
            ).sum()
        ),
        "sample_tier_counts": sample_actual,
        "role_proxy_counts": classification_actual,
        "fallback_level_counts": dataframe_count_dict(preview["fallback_level"]),
        "confidence_counts": dataframe_count_dict(preview["prediction_confidence"]),
        "data_quality_counts": dataframe_count_dict(preview["data_quality_status"]),
    }
    return blockers, warnings, summary


def top_preview_rows(preview: pd.DataFrame, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
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
        "risk_flags",
    ]
    for position in ["GKP", "DEF", "MID", "FWD"]:
        rows = (
            preview[preview["position"] == position]
            .sort_values(
                ["predicted_points", "expected_minutes", "price", "player_id"],
                ascending=[False, False, True, True],
            )
            .head(limit)[columns]
        )
        out[position] = rows.to_dict(orient="records")
    return out


def build_report(
    args: argparse.Namespace,
    features: pd.DataFrame,
    scorelines: pd.DataFrame,
    preview: pd.DataFrame,
    diagnostics: Dict[str, Any],
    day71a_report: Dict[str, Any],
    day71b_report: Dict[str, Any],
    day70c_report: Dict[str, Any],
    mode_result: Dict[str, Any],
    blockers: List[str],
    warnings: List[str],
    validation_summary: Dict[str, Any],
    created_at: str,
    run_id: str,
) -> Dict[str, Any]:
    passed = len(blockers) == 0
    return {
        "created_at": created_at,
        "run_id": run_id,
        "artifact_type": ARTIFACT_TYPE,
        "model_name": args.model_name,
        "model_version": args.model_version,
        "prediction_scope": PREDICTION_SCOPE,
        "source_seasons": list(args.source_seasons),
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "as_of_time": parse_iso8601(args.as_of_time, "as_of_time").isoformat(),
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": mode_result.get("resolved_prediction_mode"),
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
        "ready_for_pre_gw1_player_prediction_preview": passed,
        "ready_for_day72b_player_prediction_manifest": passed,
        "ready_for_trained_minutes_prediction": False,
        "ready_for_calibrated_player_points_prediction": False,
        "ready_for_production_write": False,
        "reason_production_write_not_ready": (
            "Day72A is read-only heuristic scaffolding. Production player predictions "
            "require Day72B manifest validation, historical backtests, calibration, "
            "approved minutes/event models, scoring-rule registry resolution, and an "
            "explicit fail-safe publication path."
        ),
        "inputs": {
            "player_features_csv": {
                "path": str(Path(args.player_features_csv)),
                "sha256": sha256_file(args.player_features_csv),
                "rows": int(len(features)),
            },
            "day71a_json": {
                "path": str(Path(args.day71a_json)),
                "sha256": sha256_file(args.day71a_json),
                "passed": day71a_report.get("passed"),
            },
            "day71b_json": {
                "path": str(Path(args.day71b_json)),
                "sha256": sha256_file(args.day71b_json),
                "passed": day71b_report.get("passed"),
            },
            "scoreline_preview_csv": {
                "path": str(Path(args.scoreline_preview_csv)),
                "sha256": sha256_file(args.scoreline_preview_csv),
                "rows": int(len(scorelines)),
            },
            "day70c_json": {
                "path": str(Path(args.day70c_json)) if args.day70c_json else None,
                "sha256": sha256_file(args.day70c_json) if args.day70c_json else None,
                "passed": day70c_report.get("passed") if day70c_report else None,
            },
        },
        "outputs": {
            "out_csv": str(Path(args.out_csv)),
            "out_json": str(Path(args.out_json)),
            "out_md": str(Path(args.out_md)),
        },
        "architecture_principles": {
            "season_agnostic_contract": True,
            "gw_and_as_of_aware": True,
            "leakage_safe": True,
            "config_driven_thresholds": True,
            "versioned_contracts": True,
            "standard_minutes_outputs": True,
            "long_format_player_gw_rows": True,
            "pure_logic_separated_from_io": True,
            "reuses_day71b_role_classifier": True,
            "consumes_day70c_scoreline_artifact": True,
            "rolling_horizon_compatible": True,
            "fail_safe_publishing": True,
        },
        "heuristic_policy": {
            "source_team_matches": args.source_team_matches,
            "min_price_band_reliable_samples": (
                args.min_price_band_reliable_samples
            ),
            "player_prior_weight_by_sample_tier": (
                PLAYER_PRIOR_WEIGHT_BY_SAMPLE_TIER
            ),
            "points_prior_weight_by_sample_tier": (
                POINTS_PRIOR_WEIGHT_BY_SAMPLE_TIER
            ),
            "uncertainty_half_width_by_sample_tier": (
                UNCERTAINTY_HALF_WIDTH_BY_SAMPLE_TIER
            ),
            "position_only_fallback_minutes_multiplier": (
                POSITION_ONLY_FALLBACK_MINUTES_MULTIPLIER
            ),
            "fixture_component_weights": FIXTURE_COMPONENT_WEIGHTS,
            "fixture_signal_bounds": [
                args.fixture_signal_min,
                args.fixture_signal_max,
            ],
            "fixture_multiplier_bounds": [
                args.fixture_multiplier_min,
                args.fixture_multiplier_max,
            ],
            "prediction_points_bounds": [
                args.prediction_points_min,
                args.prediction_points_max,
            ],
            "goal_points_by_position": GOAL_POINTS_BY_POSITION,
            "clean_sheet_points_by_position": (
                CLEAN_SHEET_POINTS_BY_POSITION
            ),
            "appearance_points_proxy": (
                "appearance_probability + start_probability; start probability "
                "is used as a transparent proxy for the second appearance point."
            ),
            "points_formula": (
                "blended prior points_per90 * expected_minutes / 90, with explicit "
                "unknown-role workload and within-band price fallback signals, then a "
                "conservative Day70C fixture multiplier and simple bounds."
            ),
            "components_are": COMPONENT_ACCOUNTING_STATUS,
            "heuristics_are_not_trained_coefficients": True,
        },
        "fallback_contract": {
            "level_3": "safe previous-season player prior blended with fallback",
            "level_4": "reliable position/price-band median",
            "level_5": "conservative position-only median for sparse price band",
            "no_prior_players_are_preserved": True,
            "promoted_team_players_are_preserved": True,
            "status_without_cutoff_validity_does_not_force_zero": True,
        },
        "diagnostics": diagnostics,
        "validation_summary": validation_summary,
        "top_preview_by_position": top_preview_rows(preview),
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "predicted_points is a compatibility alias for final_predicted_points.",
            "The preview contains one row per player and target GW for future horizon aggregation.",
            "Expected event components are transparent heuristic estimates and reconcile to final points through expected_other_points.",
            "Current target-season status is not used as a hard numerical filter unless a future validated contract explicitly authorizes it.",
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
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)

    validation = report["validation_summary"]
    lines = [
        "# Day72A Pre-GW1 Player Prediction Preview",
        "",
        "## Status",
        "",
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `%s`" % report["audit_only"],
        "- Writes database: `%s`" % report["writes_database"],
        "- Ready for preview: `%s`"
        % report["ready_for_pre_gw1_player_prediction_preview"],
        "- Ready for Day72B manifest: `%s`"
        % report["ready_for_day72b_player_prediction_manifest"],
        "- Ready for production write: `%s`"
        % report["ready_for_production_write"],
        "",
        "## Scope",
        "",
        "- Model: `%s` / `%s`"
        % (report["model_name"], report["model_version"]),
        "- Source seasons: `%s`" % report["source_seasons"],
        "- Target: `%s` GW `%s`"
        % (report["target_season"], report["target_gw"]),
        "- As-of: `%s`" % report["as_of_time"],
        "- Resolved mode: `%s`" % report["resolved_prediction_mode"],
        "",
        "## Validation",
        "",
    ]
    for key, value in validation.items():
        lines.append("- %s: `%s`" % (key, value))

    lines.extend(["", "## Top Preview Rows by Position", ""])
    for position, rows in report["top_preview_by_position"].items():
        lines.append("### %s" % position)
        lines.append("")
        for row in rows:
            lines.append(
                "- %s (%s), price %.1f, minutes %.2f, points %.3f, "
                "confidence %s, fallback L%s"
                % (
                    row.get("web_name"),
                    row.get("team_short_name"),
                    float(row.get("price") or 0.0),
                    float(row.get("expected_minutes") or 0.0),
                    float(row.get("predicted_points") or 0.0),
                    row.get("prediction_confidence"),
                    row.get("fallback_level"),
                )
            )
        lines.append("")

    lines.extend(["## Blockers", ""])
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
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: Dict[str, Any]) -> None:
    validation = report["validation_summary"]
    diagnostics = report["diagnostics"]
    print("=== Day72A Pre-GW1 Player Prediction Preview ===")
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
        "ready_for_pre_gw1_player_prediction_preview:",
        report["ready_for_pre_gw1_player_prediction_preview"],
    )
    print(
        "ready_for_day72b_player_prediction_manifest:",
        report["ready_for_day72b_player_prediction_manifest"],
    )
    print(
        "ready_for_production_write:",
        report["ready_for_production_write"],
    )
    print()
    print("Validation:")
    for key in [
        "feature_rows",
        "scoreline_rows",
        "preview_rows",
        "duplicate_player_id_count",
        "missing_fixture_join_count",
        "input_no_prior_count",
        "output_no_prior_count",
        "prediction_write_allowed_true_count",
        "production_ready_true_count",
        "requires_manifest_false_count",
        "status_hard_guardrail_true_count",
        "max_component_reconciliation_error",
    ]:
        print("- %s: %s" % (key, validation.get(key)))
    print("- sample_tier_counts:", validation.get("sample_tier_counts"))
    print("- role_proxy_counts:", validation.get("role_proxy_counts"))
    print("- fallback_level_counts:", validation.get("fallback_level_counts"))
    print("- confidence_counts:", validation.get("confidence_counts"))
    print()
    print("Diagnostics:")
    print(
        "- sparse_price_band_player_rows:",
        diagnostics.get("sparse_price_band_player_rows"),
    )
    print(
        "- scoreline_fallback_player_rows:",
        diagnostics.get("scoreline_fallback_player_rows"),
    )
    print(
        "- guardrail_clipped_rows:",
        diagnostics.get("guardrail_clipped_rows"),
    )
    print("Blockers:", report["blockers"] or "none")
    print("Warnings:", report["warnings"] or "none")
    print("saved_csv:", report["outputs"]["out_csv"])
    print("saved_json:", report["outputs"]["out_json"])
    print("saved_md:", report["outputs"]["out_md"])


def main() -> None:
    args = parse_args()
    created_at = utc_now()
    as_of_time = parse_iso8601(args.as_of_time, "as_of_time")
    args.as_of_time = as_of_time.isoformat()
    run_id = "%s_%s_gw%s_%s" % (
        args.model_version,
        args.target_season,
        args.target_gw,
        created_at.replace(":", "").replace("+", "_").replace(".", ""),
    )

    mode_result = resolve_prediction_mode(
        season=args.target_season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.source_seasons[-1],
        stabilization_gw=args.stabilization_gw,
        allow_experimental_mode=args.allow_experimental_mode,
    )

    source_season_for_artifact = args.source_seasons[-1]
    features = load_csv(
        args.player_features_csv,
        REQUIRED_PLAYER_FEATURE_COLUMNS,
        "Day71A player features",
    )
    features = filter_source_target_gw(
        features,
        source_season_for_artifact,
        args.target_season,
        args.target_gw,
        "Day71A player features",
    )

    scorelines = load_csv(
        args.scoreline_preview_csv,
        REQUIRED_SCORELINE_COLUMNS,
        "Day70C scoreline preview",
    )
    scorelines = filter_source_target_gw(
        scorelines,
        source_season_for_artifact,
        args.target_season,
        args.target_gw,
        "Day70C scoreline preview",
    )

    day71a_report = load_json(args.day71a_json, required=True)
    day71b_report = load_json(args.day71b_json, required=True)
    day70c_report = load_json(args.day70c_json, required=False)
    role_args = role_namespace_from_contract(day71b_report)

    preview, diagnostics = build_preview(
        features=features,
        scorelines=scorelines,
        args=args,
        role_args=role_args,
        day71b_report=day71b_report,
        run_id=run_id,
    )

    blockers, warnings, validation_summary = validate_inputs_and_output(
        args=args,
        features=features,
        scorelines=scorelines,
        preview=preview,
        day71a_report=day71a_report,
        day71b_report=day71b_report,
        day70c_report=day70c_report,
        mode_result=mode_result,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    preview.to_csv(out_csv, index=False)

    report = build_report(
        args=args,
        features=features,
        scorelines=scorelines,
        preview=preview,
        diagnostics=diagnostics,
        day71a_report=day71a_report,
        day71b_report=day71b_report,
        day70c_report=day70c_report,
        mode_result=mode_result,
        blockers=blockers,
        warnings=warnings,
        validation_summary=validation_summary,
        created_at=created_at,
        run_id=run_id,
    )
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
