from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ml.validation.resolve_prediction_mode import resolve_prediction_mode


MODEL_NAME = "pre_gw1_scoreline_prior_heuristic_v0"
MODEL_VERSION = "day70c_v0"
PREDICTION_SCOPE = "read_only_scoreline_preview"
PROBABILITY_TYPE = "poisson_scoreline_scaffolding"
DEFAULT_MAX_GOALS = 10

REQUIRED_FEATURE_COLUMNS = [
    "source_season",
    "target_season",
    "target_gw",
    "prediction_mode",
    "fixture_id",
    "home_team_id",
    "away_team_id",
    "home_team_short_name",
    "away_team_short_name",
    "home_effective_prev_season_goals_for_per_match",
    "away_effective_prev_season_goals_for_per_match",
    "home_effective_prev_season_goals_against_per_match",
    "away_effective_prev_season_goals_against_per_match",
    "home_effective_prev_season_points_per_match",
    "away_effective_prev_season_points_per_match",
    "home_effective_prev_season_clean_sheet_rate",
    "away_effective_prev_season_clean_sheet_rate",
    "both_teams_have_effective_team_features",
    "any_team_fallback_applied",
    "home_team_fallback_applied",
    "away_team_fallback_applied",
]

REQUIRED_MATCH_PREVIEW_COLUMNS = [
    "source_season",
    "target_season",
    "target_gw",
    "fixture_id",
    "home_team_short_name",
    "away_team_short_name",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "predicted_result_label",
    "data_quality_status",
    "prediction_write_allowed",
    "production_ready",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build read-only Pre-GW1 scoreline scaffolding preview from Day69B effective match features "
            "and Day70A 1X2 preview. This outputs expected goals and likely scorelines only; it never writes "
            "to the database and is not production-ready."
        )
    )
    parser.add_argument("--source-season", required=True)
    parser.add_argument("--target-season", required=True)
    parser.add_argument("--target-gw", type=int, default=1)
    parser.add_argument(
        "--prediction-mode",
        default="auto",
        choices=["auto", "pre_gw1_prior", "early_season_blend", "normal_weekly"],
    )
    parser.add_argument("--match-features-csv", required=True, help="Day69B features with effective fallback values.")
    parser.add_argument("--match-prediction-preview-csv", required=True, help="Day70A match prediction preview CSV.")
    parser.add_argument("--day70b-manifest-json", default="", help="Optional Day70B manifest JSON.")
    parser.add_argument("--score-grid-max-goals", type=int, default=DEFAULT_MAX_GOALS)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        return int(float(value))
    except (TypeError, ValueError):
        return None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    text_value = str(value).strip().lower()
    return text_value in {"true", "1", "yes", "y"}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_json(path_value: str, required: bool = False) -> Dict[str, Any]:
    if not path_value:
        return {"exists": False, "loaded": False, "path": "", "data": None, "error": None}

    path = Path(path_value)
    if not path.exists():
        if required:
            raise RuntimeError("JSON file does not exist: %s" % path)
        return {"exists": False, "loaded": False, "path": str(path), "data": None, "error": "file_not_found"}

    try:
        return {
            "exists": True,
            "loaded": True,
            "path": str(path),
            "data": json.loads(path.read_text(encoding="utf-8")),
            "error": None,
        }
    except Exception as exc:
        if required:
            raise
        return {"exists": True, "loaded": False, "path": str(path), "data": None, "error": str(exc)}


def nested_get(data: Optional[Dict[str, Any]], keys: Sequence[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_csv(path_value: str, required_columns: Sequence[str], label: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("%s CSV does not exist: %s" % (label, path))
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("%s CSV is empty: %s" % label)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise RuntimeError("%s CSV missing required columns: %s" % (label, missing))
    return df


def filter_source_target_gw(df: pd.DataFrame, source_season: str, target_season: str, target_gw: int, label: str) -> pd.DataFrame:
    out = df[
        (df["source_season"].astype(str) == str(source_season))
        & (df["target_season"].astype(str) == str(target_season))
        & (df["target_gw"].astype(int) == int(target_gw))
    ].copy()
    if out.empty:
        raise RuntimeError(
            "%s has no rows for source_season=%s target_season=%s target_gw=%s."
            % (label, source_season, target_season, target_gw)
        )
    return out


def load_inputs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = load_csv(args.match_features_csv, REQUIRED_FEATURE_COLUMNS, "Day69B match features")
    features = filter_source_target_gw(features, args.source_season, args.target_season, args.target_gw, "Day69B match features")

    preview = load_csv(args.match_prediction_preview_csv, REQUIRED_MATCH_PREVIEW_COLUMNS, "Day70A match prediction preview")
    preview = filter_source_target_gw(preview, args.source_season, args.target_season, args.target_gw, "Day70A match prediction preview")

    return features, preview


def poisson_pmf(lambda_value: float, k: int) -> float:
    return math.exp(-lambda_value) * (lambda_value ** k) / math.factorial(k)


def estimate_expected_goals(feature_row: pd.Series) -> Dict[str, Any]:
    home_gfpm = nullable_float(feature_row.get("home_effective_prev_season_goals_for_per_match"))
    away_gfpm = nullable_float(feature_row.get("away_effective_prev_season_goals_for_per_match"))
    home_gapm = nullable_float(feature_row.get("home_effective_prev_season_goals_against_per_match"))
    away_gapm = nullable_float(feature_row.get("away_effective_prev_season_goals_against_per_match"))
    home_ppm = nullable_float(feature_row.get("home_effective_prev_season_points_per_match"))
    away_ppm = nullable_float(feature_row.get("away_effective_prev_season_points_per_match"))
    home_cs = nullable_float(feature_row.get("home_effective_prev_season_clean_sheet_rate"))
    away_cs = nullable_float(feature_row.get("away_effective_prev_season_clean_sheet_rate"))

    required_values = {
        "home_gfpm": home_gfpm,
        "away_gfpm": away_gfpm,
        "home_gapm": home_gapm,
        "away_gapm": away_gapm,
        "home_ppm": home_ppm,
        "away_ppm": away_ppm,
        "home_cs": home_cs,
        "away_cs": away_cs,
    }
    missing = [name for name, value in required_values.items() if value is None]
    if missing:
        return {
            "valid": False,
            "missing_inputs": missing,
            "expected_home_goals": None,
            "expected_away_goals": None,
            "home_goal_signal": None,
            "away_goal_signal": None,
        }

    # Expected goals scaffold:
    # home attack + away defensive weakness, with a small home advantage.
    # away attack + home defensive weakness, with a small away dampener.
    home_raw = ((home_gfpm or 0.0) + (away_gapm or 0.0)) / 2.0
    away_raw = ((away_gfpm or 0.0) + (home_gapm or 0.0)) / 2.0

    home_advantage_multiplier = 1.08
    away_dampener = 0.92

    home_ppm_edge = (home_ppm or 0.0) - (away_ppm or 0.0)
    home_cs_edge = (home_cs or 0.0) - (away_cs or 0.0)

    # Small modifiers: enough to preserve relative team strength, not enough to make extreme claims.
    home_modifier = 1.0 + clamp(0.08 * home_ppm_edge + 0.12 * home_cs_edge, -0.18, 0.18)
    away_modifier = 1.0 + clamp(-0.08 * home_ppm_edge - 0.12 * home_cs_edge, -0.18, 0.18)

    expected_home_goals = home_raw * home_advantage_multiplier * home_modifier
    expected_away_goals = away_raw * away_dampener * away_modifier

    # Conservative clamp: this is a scaffold, not a goal model.
    expected_home_goals = clamp(expected_home_goals, 0.25, 3.5)
    expected_away_goals = clamp(expected_away_goals, 0.20, 3.3)

    return {
        "valid": True,
        "missing_inputs": [],
        "expected_home_goals": round(expected_home_goals, 4),
        "expected_away_goals": round(expected_away_goals, 4),
        "home_goal_signal": round(home_raw, 4),
        "away_goal_signal": round(away_raw, 4),
        "home_advantage_multiplier": home_advantage_multiplier,
        "away_dampener": away_dampener,
        "home_modifier": round(home_modifier, 4),
        "away_modifier": round(away_modifier, 4),
    }


def scoreline_grid(expected_home_goals: float, expected_away_goals: float, max_goals: int) -> Dict[str, Any]:
    score_rows: List[Tuple[int, int, float]] = []
    home_win_probability = 0.0
    draw_probability = 0.0
    away_win_probability = 0.0
    grid_probability_sum = 0.0

    for home_goals in range(max_goals + 1):
        home_p = poisson_pmf(expected_home_goals, home_goals)
        for away_goals in range(max_goals + 1):
            away_p = poisson_pmf(expected_away_goals, away_goals)
            probability = home_p * away_p
            grid_probability_sum += probability
            score_rows.append((home_goals, away_goals, probability))

            if home_goals > away_goals:
                home_win_probability += probability
            elif home_goals == away_goals:
                draw_probability += probability
            else:
                away_win_probability += probability

    score_rows = sorted(score_rows, key=lambda item: item[2], reverse=True)

    top = []
    for rank, (home_goals, away_goals, probability) in enumerate(score_rows[:5], start=1):
        top.append({
            "rank": rank,
            "scoreline": "%s-%s" % (home_goals, away_goals),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "probability": round(probability, 6),
        })

    tail_probability = max(0.0, 1.0 - grid_probability_sum)

    # Normalize outcome probabilities over the grid for comparison while preserving grid sum separately.
    if grid_probability_sum > 0:
        norm_home = home_win_probability / grid_probability_sum
        norm_draw = draw_probability / grid_probability_sum
        norm_away = away_win_probability / grid_probability_sum
    else:
        norm_home = None
        norm_draw = None
        norm_away = None

    labels = {
        "home_win": norm_home if norm_home is not None else -1,
        "draw": norm_draw if norm_draw is not None else -1,
        "away_win": norm_away if norm_away is not None else -1,
    }
    scoreline_result_label = max(labels, key=labels.get)

    return {
        "score_grid_max_goals": max_goals,
        "score_grid_probability_sum": round(grid_probability_sum, 6),
        "score_grid_tail_probability": round(tail_probability, 6),
        "scoreline_home_win_probability": round(norm_home, 6) if norm_home is not None else None,
        "scoreline_draw_probability": round(norm_draw, 6) if norm_draw is not None else None,
        "scoreline_away_win_probability": round(norm_away, 6) if norm_away is not None else None,
        "scoreline_result_label": scoreline_result_label,
        "top_scorelines": top,
    }


def build_scoreline_preview(features: pd.DataFrame, match_preview: pd.DataFrame, max_goals: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    preview_subset_cols = [
        "fixture_id",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "predicted_result_label",
        "data_quality_status",
        "prediction_write_allowed",
        "production_ready",
    ]
    preview_subset = match_preview[preview_subset_cols].copy()
    preview_subset = preview_subset.rename(columns={
        "home_win_probability": "one_x_two_home_win_probability",
        "draw_probability": "one_x_two_draw_probability",
        "away_win_probability": "one_x_two_away_win_probability",
        "predicted_result_label": "one_x_two_predicted_result_label",
        "data_quality_status": "one_x_two_data_quality_status",
        "prediction_write_allowed": "one_x_two_prediction_write_allowed",
        "production_ready": "one_x_two_production_ready",
    })

    merged = features.merge(preview_subset, on="fixture_id", how="left", validate="one_to_one")

    rows: List[Dict[str, Any]] = []
    invalid_goal_signal_count = 0
    scoreline_label_disagreement_count = 0

    for _, row in merged.iterrows():
        goal_estimate = estimate_expected_goals(row)

        if not goal_estimate["valid"]:
            invalid_goal_signal_count += 1
            grid = {
                "score_grid_max_goals": max_goals,
                "score_grid_probability_sum": None,
                "score_grid_tail_probability": None,
                "scoreline_home_win_probability": None,
                "scoreline_draw_probability": None,
                "scoreline_away_win_probability": None,
                "scoreline_result_label": None,
                "top_scorelines": [],
            }
        else:
            grid = scoreline_grid(
                expected_home_goals=float(goal_estimate["expected_home_goals"]),
                expected_away_goals=float(goal_estimate["expected_away_goals"]),
                max_goals=max_goals,
            )

        top = grid["top_scorelines"]

        one_x_two_label = row.get("one_x_two_predicted_result_label")
        scoreline_label = grid.get("scoreline_result_label")
        label_matches = bool(one_x_two_label == scoreline_label)
        if scoreline_label is not None and not label_matches:
            scoreline_label_disagreement_count += 1

        out: Dict[str, Any] = {
            "source_season": row.get("source_season"),
            "target_season": row.get("target_season"),
            "target_gw": nullable_int(row.get("target_gw")),
            "prediction_mode": row.get("prediction_mode"),
            "prediction_scope": PREDICTION_SCOPE,
            "probability_type": PROBABILITY_TYPE,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "fixture_id": nullable_int(row.get("fixture_id")),
            "fpl_fixture_id": nullable_int(row.get("fpl_fixture_id")),
            "kickoff_time": row.get("kickoff_time"),
            "home_team_id": nullable_int(row.get("home_team_id")),
            "away_team_id": nullable_int(row.get("away_team_id")),
            "home_team_short_name": row.get("home_team_short_name"),
            "away_team_short_name": row.get("away_team_short_name"),
            "home_team_name": row.get("home_team_name"),
            "away_team_name": row.get("away_team_name"),
            "any_team_fallback_applied": bool_value(row.get("any_team_fallback_applied")),
            "home_team_fallback_applied": bool_value(row.get("home_team_fallback_applied")),
            "away_team_fallback_applied": bool_value(row.get("away_team_fallback_applied")),
            "data_quality_status": "fallback_effective_features" if bool_value(row.get("any_team_fallback_applied")) else "full_prior_features",
            "goal_model_status": "valid_expected_goals" if goal_estimate["valid"] else "invalid_missing_goal_inputs",
            "missing_goal_inputs": ",".join(goal_estimate["missing_inputs"]),
            "expected_home_goals": goal_estimate["expected_home_goals"],
            "expected_away_goals": goal_estimate["expected_away_goals"],
            "expected_total_goals": (
                round(float(goal_estimate["expected_home_goals"]) + float(goal_estimate["expected_away_goals"]), 4)
                if goal_estimate["valid"] else None
            ),
            "expected_goal_diff_home_minus_away": (
                round(float(goal_estimate["expected_home_goals"]) - float(goal_estimate["expected_away_goals"]), 4)
                if goal_estimate["valid"] else None
            ),
            "home_goal_signal": goal_estimate["home_goal_signal"],
            "away_goal_signal": goal_estimate["away_goal_signal"],
            "home_advantage_multiplier": goal_estimate.get("home_advantage_multiplier"),
            "away_dampener": goal_estimate.get("away_dampener"),
            "home_modifier": goal_estimate.get("home_modifier"),
            "away_modifier": goal_estimate.get("away_modifier"),
            "score_grid_max_goals": grid["score_grid_max_goals"],
            "score_grid_probability_sum": grid["score_grid_probability_sum"],
            "score_grid_tail_probability": grid["score_grid_tail_probability"],
            "scoreline_home_win_probability": grid["scoreline_home_win_probability"],
            "scoreline_draw_probability": grid["scoreline_draw_probability"],
            "scoreline_away_win_probability": grid["scoreline_away_win_probability"],
            "scoreline_result_label": grid["scoreline_result_label"],
            "one_x_two_home_win_probability": row.get("one_x_two_home_win_probability"),
            "one_x_two_draw_probability": row.get("one_x_two_draw_probability"),
            "one_x_two_away_win_probability": row.get("one_x_two_away_win_probability"),
            "one_x_two_predicted_result_label": one_x_two_label,
            "scoreline_label_matches_one_x_two_label": label_matches,
            "prediction_write_allowed": False,
            "production_ready": False,
            "requires_scoreline_manifest_before_write": True,
            "calibration_status": "not_calibrated_scoreline_preview_only",
            "guardrail_status": "basic_score_grid_sanity_only",
        }

        for idx in range(5):
            prefix = "top_%s_" % (idx + 1)
            if idx < len(top):
                out[prefix + "scoreline"] = top[idx]["scoreline"]
                out[prefix + "scoreline_probability"] = top[idx]["probability"]
                out[prefix + "home_goals"] = top[idx]["home_goals"]
                out[prefix + "away_goals"] = top[idx]["away_goals"]
            else:
                out[prefix + "scoreline"] = None
                out[prefix + "scoreline_probability"] = None
                out[prefix + "home_goals"] = None
                out[prefix + "away_goals"] = None

        rows.append(out)

    output = pd.DataFrame(rows)
    diagnostics = {
        "invalid_goal_signal_count": invalid_goal_signal_count,
        "scoreline_label_disagreement_count": scoreline_label_disagreement_count,
    }
    return output, diagnostics


def build_report(
    args: argparse.Namespace,
    features: pd.DataFrame,
    match_preview: pd.DataFrame,
    scorelines: pd.DataFrame,
    diagnostics: Dict[str, Any],
    day70b_manifest: Dict[str, Any],
    mode_result: Dict[str, Any],
    blockers: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    if not mode_result["valid"]:
        blockers.append("Prediction mode resolver returned invalid: %s" % mode_result["errors"])
    if mode_result["resolved_prediction_mode"] != "pre_gw1_prior":
        blockers.append("Day70C expects resolved_prediction_mode=pre_gw1_prior.")
    if args.target_gw != 1:
        blockers.append("Day70C expects target_gw=1.")

    if len(features) != len(scorelines):
        blockers.append("Scoreline row count does not equal feature row count.")
    if len(match_preview) != len(scorelines):
        blockers.append("Scoreline row count does not equal match preview row count.")

    duplicate_fixture_count = int(scorelines["fixture_id"].dropna().duplicated().sum())
    if duplicate_fixture_count > 0:
        blockers.append("Duplicate fixture_id values found in scoreline preview.")

    if diagnostics["invalid_goal_signal_count"] > 0:
        blockers.append("Some rows have missing goal inputs and could not produce scoreline preview.")

    required_output_cols = [
        "expected_home_goals",
        "expected_away_goals",
        "top_1_scoreline",
        "top_1_scoreline_probability",
        "scoreline_home_win_probability",
        "scoreline_draw_probability",
        "scoreline_away_win_probability",
        "prediction_write_allowed",
        "production_ready",
        "requires_scoreline_manifest_before_write",
    ]
    missing_output_cols = [col for col in required_output_cols if col not in scorelines.columns]
    if missing_output_cols:
        blockers.append("Scoreline output missing required columns: %s" % missing_output_cols)

    for col in ["expected_home_goals", "expected_away_goals", "top_1_scoreline_probability"]:
        if col in scorelines.columns and int(scorelines[col].isna().sum()) > 0:
            blockers.append("%s contains missing values." % col)

    probability_cols = ["scoreline_home_win_probability", "scoreline_draw_probability", "scoreline_away_win_probability"]
    for col in probability_cols:
        if col in scorelines.columns:
            if int(scorelines[col].isna().sum()) > 0:
                blockers.append("%s contains missing values." % col)
            if int(((scorelines[col] < 0) | (scorelines[col] > 1)).sum()) > 0:
                blockers.append("%s has values outside [0, 1]." % col)

    if all(col in scorelines.columns for col in probability_cols):
        max_result_probability_sum_error = float((scorelines[probability_cols].sum(axis=1) - 1.0).abs().max())
        if max_result_probability_sum_error > 0.00001:
            blockers.append("Scoreline result probabilities do not sum to 1 within tolerance.")
    else:
        max_result_probability_sum_error = None

    if "score_grid_probability_sum" in scorelines.columns:
        min_score_grid_probability_sum = float(scorelines["score_grid_probability_sum"].min())
        max_score_grid_tail_probability = float(scorelines["score_grid_tail_probability"].max())
        if min_score_grid_probability_sum < 0.995:
            warnings.append("Score grid probability sum below 0.995; consider increasing --score-grid-max-goals.")
    else:
        min_score_grid_probability_sum = None
        max_score_grid_tail_probability = None

    write_true_count = int(scorelines["prediction_write_allowed"].apply(bool_value).sum())
    production_true_count = int(scorelines["production_ready"].apply(bool_value).sum())
    manifest_false_count = int((~scorelines["requires_scoreline_manifest_before_write"].apply(bool_value)).sum())

    if write_true_count != 0:
        blockers.append("prediction_write_allowed must be false for every scoreline preview row.")
    if production_true_count != 0:
        blockers.append("production_ready must be false for every scoreline preview row.")
    if manifest_false_count != 0:
        blockers.append("requires_scoreline_manifest_before_write must be true for every scoreline preview row.")

    if day70b_manifest.get("exists") and not day70b_manifest.get("loaded"):
        warnings.append("Day70B manifest was provided but could not be loaded: %s" % day70b_manifest.get("error"))
    if day70b_manifest.get("loaded"):
        if nested_get(day70b_manifest.get("data"), ["passed"], None) is not True:
            blockers.append("Day70B manifest did not pass.")
        if nested_get(day70b_manifest.get("data"), ["ready_for_ops_preview"], None) is not True:
            blockers.append("Day70B manifest is not ready_for_ops_preview.")
        if nested_get(day70b_manifest.get("data"), ["ready_for_prediction_write"], None) is not False:
            blockers.append("Day70B manifest must remain ready_for_prediction_write=false.")

    label_disagreement_count = int(diagnostics["scoreline_label_disagreement_count"])
    if label_disagreement_count > 0:
        warnings.append(
            "%s scoreline result labels differ from Day70A 1X2 preview labels. "
            "This is acceptable for scaffolding but must be reviewed before production."
            % label_disagreement_count
        )

    label_counts = scorelines["scoreline_result_label"].value_counts(dropna=False).to_dict()
    data_quality_counts = scorelines["data_quality_status"].value_counts(dropna=False).to_dict()
    top_scoreline_counts = scorelines["top_1_scoreline"].value_counts(dropna=False).to_dict()

    return {
        "created_at": utc_now(),
        "source_season": args.source_season,
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": mode_result["resolved_prediction_mode"],
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "prediction_scope": PREDICTION_SCOPE,
        "probability_type": PROBABILITY_TYPE,
        "score_grid_max_goals": args.score_grid_max_goals,
        "audit_only": True,
        "writes_database": False,
        "passed": len(blockers) == 0,
        "ready_for_pre_gw1_scoreline_preview": len(blockers) == 0,
        "ready_for_full_pre_gw1_scoreline_prediction": False,
        "ready_for_prediction_write": False,
        "reason_prediction_write_not_ready": (
            "Day70C creates read-only scoreline scaffolding only. Production scoreline prediction requires "
            "a scoreline manifest, backtest/calibration, guardrails, and an explicit DB write path."
        ),
        "inputs": {
            "match_features_csv": args.match_features_csv,
            "match_prediction_preview_csv": args.match_prediction_preview_csv,
            "day70b_manifest_json": args.day70b_manifest_json or None,
            "out_csv": args.out_csv,
            "out_json": args.out_json,
            "out_md": args.out_md,
        },
        "mode_resolution": mode_result,
        "row_counts": {
            "feature_rows": int(len(features)),
            "match_prediction_preview_rows": int(len(match_preview)),
            "scoreline_preview_rows": int(len(scorelines)),
            "fallback_scoreline_rows": int(scorelines["any_team_fallback_applied"].apply(bool_value).sum()),
            "full_prior_scoreline_rows": int((~scorelines["any_team_fallback_applied"].apply(bool_value)).sum()),
        },
        "scoreline_quality": {
            "duplicate_fixture_count": duplicate_fixture_count,
            "invalid_goal_signal_count": diagnostics["invalid_goal_signal_count"],
            "scoreline_label_disagreement_count": label_disagreement_count,
            "max_result_probability_sum_error": max_result_probability_sum_error,
            "min_score_grid_probability_sum": min_score_grid_probability_sum,
            "max_score_grid_tail_probability": max_score_grid_tail_probability,
            "min_expected_home_goals": float(scorelines["expected_home_goals"].min()) if len(scorelines) else None,
            "max_expected_home_goals": float(scorelines["expected_home_goals"].max()) if len(scorelines) else None,
            "min_expected_away_goals": float(scorelines["expected_away_goals"].min()) if len(scorelines) else None,
            "max_expected_away_goals": float(scorelines["expected_away_goals"].max()) if len(scorelines) else None,
        },
        "scoreline_summary": {
            "scoreline_result_label_counts": label_counts,
            "top_1_scoreline_counts": top_scoreline_counts,
            "data_quality_status_counts": data_quality_counts,
            "prediction_write_allowed_true_count": write_true_count,
            "production_ready_true_count": production_true_count,
            "requires_scoreline_manifest_false_count": manifest_false_count,
        },
        "source_reports": {
            "day70b_manifest": {
                "exists": day70b_manifest.get("exists"),
                "loaded": day70b_manifest.get("loaded"),
                "passed": nested_get(day70b_manifest.get("data"), ["passed"], None),
                "ready_for_ops_preview": nested_get(day70b_manifest.get("data"), ["ready_for_ops_preview"], None),
                "ready_for_prediction_write": nested_get(day70b_manifest.get("data"), ["ready_for_prediction_write"], None),
            }
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "This is a scoreline scaffolding preview, not a trained or calibrated score model.",
            "Expected goals are heuristic estimates from effective team attacking/defensive prior features.",
            "Scoreline probabilities use an independent Poisson grid for auditability.",
            "Fallback rows remain marked and must be reviewed before production.",
            "All rows keep prediction_write_allowed=false and production_ready=false.",
        ],
    }


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    lines: List[str] = []
    lines.append("# Day70C — Pre-GW1 Scoreline Scaffolding Preview")
    lines.append("")
    lines.append("Created at: `%s`" % report["created_at"])
    lines.append("")
    lines.append("Source season: `%s`" % report["source_season"])
    lines.append("Target season: `%s`" % report["target_season"])
    lines.append("Target GW: `%s`" % report["target_gw"])
    lines.append("Resolved prediction mode: `%s`" % report["resolved_prediction_mode"])
    lines.append("Model: `%s` / `%s`" % (report["model_name"], report["model_version"]))
    lines.append("Probability type: `%s`" % report["probability_type"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Passed: `%s`" % report["passed"])
    lines.append("- Audit only: `%s`" % report["audit_only"])
    lines.append("- Writes database: `%s`" % report["writes_database"])
    lines.append("- Ready for Pre-GW1 scoreline preview: `%s`" % report["ready_for_pre_gw1_scoreline_preview"])
    lines.append("- Ready for full Pre-GW1 scoreline prediction: `%s`" % report["ready_for_full_pre_gw1_scoreline_prediction"])
    lines.append("- Ready for prediction write: `%s`" % report["ready_for_prediction_write"])
    lines.append("")
    lines.append("## Row Counts")
    lines.append("")
    for key, value in report["row_counts"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Scoreline Quality")
    lines.append("")
    for key, value in report["scoreline_quality"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Scoreline Summary")
    lines.append("")
    for key, value in report["scoreline_summary"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append("- %s" % blocker)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append("- %s" % warning)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append("- %s" % note)
    lines.append("")

    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: Dict[str, Any]) -> None:
    print("=== Day70C Pre-GW1 Scoreline Scaffolding Preview ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("target_gw:", report["target_gw"])
    print("resolved_prediction_mode:", report["resolved_prediction_mode"])
    print("model_name:", report["model_name"])
    print("model_version:", report["model_version"])
    print("prediction_scope:", report["prediction_scope"])
    print("probability_type:", report["probability_type"])
    print("score_grid_max_goals:", report["score_grid_max_goals"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("ready_for_pre_gw1_scoreline_preview:", report["ready_for_pre_gw1_scoreline_preview"])
    print("ready_for_full_pre_gw1_scoreline_prediction:", report["ready_for_full_pre_gw1_scoreline_prediction"])
    print("ready_for_prediction_write:", report["ready_for_prediction_write"])
    print("saved_csv:", report["inputs"]["out_csv"])
    print("saved_json:", report["inputs"]["out_json"])
    print("saved_md:", report["inputs"]["out_md"])
    print()
    print("Row counts:")
    for key, value in report["row_counts"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Scoreline quality:")
    for key, value in report["scoreline_quality"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Scoreline summary:")
    for key, value in report["scoreline_summary"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Blockers:", report["blockers"] or "none")
    print("Warnings:", report["warnings"] or "none")


def main() -> None:
    args = parse_args()
    blockers: List[str] = []
    warnings: List[str] = []

    mode_result = resolve_prediction_mode(
        season=args.target_season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.source_season,
        stabilization_gw=6,
        allow_experimental_mode=False,
    )

    if args.score_grid_max_goals < 6:
        blockers.append("--score-grid-max-goals must be at least 6.")
    if args.score_grid_max_goals > 15:
        warnings.append("--score-grid-max-goals above 15 may be unnecessarily large for this scaffold.")

    features, match_preview = load_inputs(args)

    if not features["both_teams_have_effective_team_features"].apply(bool_value).all():
        blockers.append("Not every Day69B feature row has complete effective team features.")

    if match_preview["prediction_write_allowed"].apply(bool_value).any():
        blockers.append("Input Day70A preview has prediction_write_allowed=true row(s).")
    if match_preview["production_ready"].apply(bool_value).any():
        blockers.append("Input Day70A preview has production_ready=true row(s).")

    day70b_manifest = load_json(args.day70b_manifest_json, required=False)

    scorelines, diagnostics = build_scoreline_preview(features, match_preview, args.score_grid_max_goals)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scorelines.to_csv(out_csv, index=False)

    report = build_report(
        args=args,
        features=features,
        match_preview=match_preview,
        scorelines=scorelines,
        diagnostics=diagnostics,
        day70b_manifest=day70b_manifest,
        mode_result=mode_result,
        blockers=blockers,
        warnings=warnings,
    )

    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
