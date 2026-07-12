from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ml.validation.resolve_prediction_mode import resolve_prediction_mode


MODEL_NAME = "pre_gw1_match_prior_heuristic_v0"
MODEL_VERSION = "day70a_v0"
PREDICTION_SCOPE = "read_only_preview"
PROBABILITY_TYPE = "heuristic_scaffolding"


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
    "home_effective_prev_season_points_per_match",
    "away_effective_prev_season_points_per_match",
    "home_effective_prev_season_goals_for_per_match",
    "away_effective_prev_season_goals_for_per_match",
    "home_effective_prev_season_goals_against_per_match",
    "away_effective_prev_season_goals_against_per_match",
    "home_effective_prev_season_clean_sheet_rate",
    "away_effective_prev_season_clean_sheet_rate",
    "home_effective_prev_season_goal_difference",
    "away_effective_prev_season_goal_difference",
    "both_teams_have_effective_team_features",
    "any_team_fallback_applied",
    "home_team_fallback_applied",
    "away_team_fallback_applied",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build read-only Pre-GW1 match prediction scaffolding from complete effective team features. "
            "This outputs heuristic preview probabilities only and never writes to the database."
        )
    )
    parser.add_argument("--source-season", required=True, help="Prior/source season, for example 2024_25.")
    parser.add_argument("--target-season", required=True, help="Target season, for example 2025_26.")
    parser.add_argument("--target-gw", type=int, default=1)
    parser.add_argument(
        "--prediction-mode",
        default="auto",
        choices=["auto", "pre_gw1_prior", "early_season_blend", "normal_weekly"],
    )
    parser.add_argument(
        "--match-features-csv",
        required=True,
        help="Day69B Pre-GW1 match features with effective fallback values.",
    )
    parser.add_argument("--out-csv", required=True, help="Output read-only match prediction preview CSV.")
    parser.add_argument("--out-json", required=True, help="Output JSON audit report.")
    parser.add_argument("--out-md", required=True, help="Output Markdown audit report.")
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


def load_match_features(path_value: str, source_season: str, target_season: str, target_gw: int) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("match features CSV does not exist: %s" % path)

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("match features CSV is empty: %s" % path)

    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError("match features CSV missing required columns: %s" % missing)

    filtered = df[
        (df["source_season"].astype(str) == str(source_season))
        & (df["target_season"].astype(str) == str(target_season))
        & (df["target_gw"].astype(int) == int(target_gw))
    ].copy()

    if filtered.empty:
        raise RuntimeError(
            "No match feature rows found for source_season=%s target_season=%s target_gw=%s."
            % (source_season, target_season, target_gw)
        )

    return filtered


def softmax3(home_score: float, draw_score: float, away_score: float) -> Tuple[float, float, float]:
    max_score = max(home_score, draw_score, away_score)
    h = math.exp(home_score - max_score)
    d = math.exp(draw_score - max_score)
    a = math.exp(away_score - max_score)
    total = h + d + a
    return h / total, d / total, a / total


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_signal(row: pd.Series) -> Dict[str, Any]:
    home_ppm = nullable_float(row.get("home_effective_prev_season_points_per_match"))
    away_ppm = nullable_float(row.get("away_effective_prev_season_points_per_match"))
    home_gfpm = nullable_float(row.get("home_effective_prev_season_goals_for_per_match"))
    away_gfpm = nullable_float(row.get("away_effective_prev_season_goals_for_per_match"))
    home_gapm = nullable_float(row.get("home_effective_prev_season_goals_against_per_match"))
    away_gapm = nullable_float(row.get("away_effective_prev_season_goals_against_per_match"))
    home_cs = nullable_float(row.get("home_effective_prev_season_clean_sheet_rate"))
    away_cs = nullable_float(row.get("away_effective_prev_season_clean_sheet_rate"))
    home_gd = nullable_float(row.get("home_effective_prev_season_goal_difference"))
    away_gd = nullable_float(row.get("away_effective_prev_season_goal_difference"))

    missing = []
    for name, value in [
        ("home_ppm", home_ppm),
        ("away_ppm", away_ppm),
        ("home_gfpm", home_gfpm),
        ("away_gfpm", away_gfpm),
        ("home_gapm", home_gapm),
        ("away_gapm", away_gapm),
        ("home_cs", home_cs),
        ("away_cs", away_cs),
        ("home_gd", home_gd),
        ("away_gd", away_gd),
    ]:
        if value is None:
            missing.append(name)

    if missing:
        return {
            "valid": False,
            "missing_inputs": missing,
            "home_advantage_signal": None,
            "points_signal": None,
            "attack_signal": None,
            "defense_signal": None,
            "clean_sheet_signal": None,
            "goal_difference_signal": None,
            "combined_signal": None,
        }

    home_advantage_signal = 0.18
    points_signal = 0.85 * ((home_ppm or 0.0) - (away_ppm or 0.0))
    attack_signal = 0.35 * ((home_gfpm or 0.0) - (away_gfpm or 0.0))

    # Lower goals-against is better. If away conceded more than home, signal favors home.
    defense_signal = 0.30 * ((away_gapm or 0.0) - (home_gapm or 0.0))
    clean_sheet_signal = 0.20 * ((home_cs or 0.0) - (away_cs or 0.0))

    # Goal difference is season total, so scale it down.
    goal_difference_signal = 0.01 * ((home_gd or 0.0) - (away_gd or 0.0))

    combined_signal = (
        home_advantage_signal
        + points_signal
        + attack_signal
        + defense_signal
        + clean_sheet_signal
        + goal_difference_signal
    )

    # Keep the preview heuristic conservative so it does not generate extreme claims.
    combined_signal = clamp(combined_signal, -2.2, 2.2)

    return {
        "valid": True,
        "missing_inputs": [],
        "home_advantage_signal": round(home_advantage_signal, 4),
        "points_signal": round(points_signal, 4),
        "attack_signal": round(attack_signal, 4),
        "defense_signal": round(defense_signal, 4),
        "clean_sheet_signal": round(clean_sheet_signal, 4),
        "goal_difference_signal": round(goal_difference_signal, 4),
        "combined_signal": round(combined_signal, 4),
    }


def probability_from_signal(combined_signal: float, any_fallback_applied: bool) -> Dict[str, Any]:
    # Temperature controls how sharp the preview probabilities are.
    temperature = 1.35

    # Draw score is highest when teams look close, lower when the signal is large.
    draw_base = 0.08
    draw_score = draw_base - 0.34 * abs(combined_signal)

    home_score = combined_signal / temperature
    away_score = -combined_signal / temperature

    home_prob, draw_prob, away_prob = softmax3(home_score, draw_score, away_score)

    # If fallback was used, make the preview slightly more conservative by pulling the highest outcome
    # a little toward draw. This is a transparent uncertainty penalty, not a learned calibration.
    fallback_uncertainty_penalty = 0.04 if any_fallback_applied else 0.0
    if fallback_uncertainty_penalty > 0:
        probs = {"home_win": home_prob, "draw": draw_prob, "away_win": away_prob}
        top_key = max(probs, key=probs.get)
        probs[top_key] = probs[top_key] - fallback_uncertainty_penalty
        probs["draw"] = probs["draw"] + fallback_uncertainty_penalty
        home_prob = probs["home_win"]
        draw_prob = probs["draw"]
        away_prob = probs["away_win"]

    # Numerical safety and final normalization.
    home_prob = max(0.001, home_prob)
    draw_prob = max(0.001, draw_prob)
    away_prob = max(0.001, away_prob)
    total = home_prob + draw_prob + away_prob
    home_prob = home_prob / total
    draw_prob = draw_prob / total
    away_prob = away_prob / total

    outcomes = {
        "home_win": home_prob,
        "draw": draw_prob,
        "away_win": away_prob,
    }
    predicted_result_label = max(outcomes, key=outcomes.get)
    confidence_score = max(outcomes.values()) - sorted(outcomes.values())[-2]

    return {
        "home_win_probability": round(home_prob, 6),
        "draw_probability": round(draw_prob, 6),
        "away_win_probability": round(away_prob, 6),
        "probability_sum": round(home_prob + draw_prob + away_prob, 6),
        "predicted_result_label": predicted_result_label,
        "confidence_score": round(confidence_score, 6),
        "fallback_uncertainty_penalty": fallback_uncertainty_penalty,
    }


def build_prediction_preview(features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    invalid_signal_count = 0

    for _, row in features.iterrows():
        signal = compute_signal(row)
        any_fallback = bool_value(row.get("any_team_fallback_applied"))

        if not signal["valid"]:
            invalid_signal_count += 1
            probs = {
                "home_win_probability": None,
                "draw_probability": None,
                "away_win_probability": None,
                "probability_sum": None,
                "predicted_result_label": None,
                "confidence_score": None,
                "fallback_uncertainty_penalty": 0.0,
            }
        else:
            probs = probability_from_signal(float(signal["combined_signal"]), any_fallback)

        data_quality_status = "full_prior_features"
        if any_fallback:
            data_quality_status = "fallback_effective_features"
        if not signal["valid"]:
            data_quality_status = "invalid_missing_effective_features"

        out = {
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
            "home_team_fallback_applied": any_fallback and bool_value(row.get("home_team_fallback_applied")),
            "away_team_fallback_applied": any_fallback and bool_value(row.get("away_team_fallback_applied")),
            "any_team_fallback_applied": any_fallback,
            "home_effective_team_feature_status": row.get("home_effective_team_feature_status"),
            "away_effective_team_feature_status": row.get("away_effective_team_feature_status"),
            "both_teams_have_effective_team_features": bool_value(row.get("both_teams_have_effective_team_features")),
            "data_quality_status": data_quality_status,
            "prediction_write_allowed": False,
            "production_ready": False,
            "requires_manifest_before_write": True,
            "calibration_status": "not_calibrated_preview_only",
            "guardrail_status": "basic_probability_sanity_only",
            "missing_signal_inputs": ",".join(signal["missing_inputs"]),
            "home_advantage_signal": signal["home_advantage_signal"],
            "points_signal": signal["points_signal"],
            "attack_signal": signal["attack_signal"],
            "defense_signal": signal["defense_signal"],
            "clean_sheet_signal": signal["clean_sheet_signal"],
            "goal_difference_signal": signal["goal_difference_signal"],
            "combined_signal_home_minus_away": signal["combined_signal"],
            "home_effective_prev_season_points_per_match": row.get("home_effective_prev_season_points_per_match"),
            "away_effective_prev_season_points_per_match": row.get("away_effective_prev_season_points_per_match"),
            "effective_prior_points_per_match_diff_home_minus_away": row.get("effective_prior_points_per_match_diff_home_minus_away"),
            "home_effective_prev_season_goals_for_per_match": row.get("home_effective_prev_season_goals_for_per_match"),
            "away_effective_prev_season_goals_for_per_match": row.get("away_effective_prev_season_goals_for_per_match"),
            "effective_prior_goals_for_per_match_diff_home_minus_away": row.get("effective_prior_goals_for_per_match_diff_home_minus_away"),
            "home_effective_prev_season_goals_against_per_match": row.get("home_effective_prev_season_goals_against_per_match"),
            "away_effective_prev_season_goals_against_per_match": row.get("away_effective_prev_season_goals_against_per_match"),
            "effective_prior_goals_against_per_match_diff_home_minus_away": row.get("effective_prior_goals_against_per_match_diff_home_minus_away"),
        }
        out.update(probs)
        rows.append(out)

    predictions = pd.DataFrame(rows)
    diagnostics = {
        "invalid_signal_count": invalid_signal_count,
    }
    return predictions, diagnostics


def build_report(
    args: argparse.Namespace,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    diagnostics: Dict[str, Any],
    mode_result: Dict[str, Any],
    blockers: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    if len(features) != len(predictions):
        blockers.append("Prediction preview row count does not equal input feature row count.")

    if int(predictions["fixture_id"].dropna().duplicated().sum()) > 0:
        blockers.append("Duplicate fixture_id values found in prediction preview.")

    if diagnostics["invalid_signal_count"] > 0:
        blockers.append("Some rows have missing effective feature inputs and could not be scored.")

    probability_cols = ["home_win_probability", "draw_probability", "away_win_probability"]
    for col in probability_cols:
        if col not in predictions.columns:
            blockers.append("Missing probability column: %s" % col)
        elif int(predictions[col].isna().sum()) > 0:
            blockers.append("%s contains missing values." % col)

    if "probability_sum" in predictions.columns:
        max_probability_sum_error = float((predictions["probability_sum"] - 1.0).abs().max())
        if max_probability_sum_error > 0.00001:
            blockers.append("Probability sums are not within tolerance.")
    else:
        max_probability_sum_error = None

    if int(predictions["prediction_write_allowed"].sum()) != 0:
        blockers.append("prediction_write_allowed must be false for every preview row.")

    if int(predictions["production_ready"].sum()) != 0:
        blockers.append("production_ready must be false for every preview row.")

    fallback_count = int(predictions["any_team_fallback_applied"].sum())
    full_prior_count = int((predictions["any_team_fallback_applied"] == False).sum())

    label_counts = predictions["predicted_result_label"].value_counts(dropna=False).to_dict()
    data_quality_counts = predictions["data_quality_status"].value_counts(dropna=False).to_dict()

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
        "audit_only": True,
        "writes_database": False,
        "passed": len(blockers) == 0,
        "ready_for_pre_gw1_match_prediction_preview": len(blockers) == 0,
        "ready_for_full_pre_gw1_match_prediction": False,
        "ready_for_prediction_write": False,
        "reason_prediction_write_not_ready": (
            "Day70A creates a read-only heuristic probability preview only. "
            "Production writes require explicit model policy, calibration/guardrail contract, manifest, and DB write path."
        ),
        "inputs": {
            "match_features_csv": args.match_features_csv,
            "out_csv": args.out_csv,
            "out_json": args.out_json,
            "out_md": args.out_md,
        },
        "mode_resolution": mode_result,
        "row_counts": {
            "feature_rows": int(len(features)),
            "prediction_preview_rows": int(len(predictions)),
            "fallback_feature_rows": fallback_count,
            "full_prior_feature_rows": full_prior_count,
        },
        "probability_quality": {
            "max_probability_sum_error": max_probability_sum_error,
            "min_home_win_probability": float(predictions["home_win_probability"].min()) if len(predictions) else None,
            "max_home_win_probability": float(predictions["home_win_probability"].max()) if len(predictions) else None,
            "min_draw_probability": float(predictions["draw_probability"].min()) if len(predictions) else None,
            "max_draw_probability": float(predictions["draw_probability"].max()) if len(predictions) else None,
            "min_away_win_probability": float(predictions["away_win_probability"].min()) if len(predictions) else None,
            "max_away_win_probability": float(predictions["away_win_probability"].max()) if len(predictions) else None,
            "invalid_signal_count": diagnostics["invalid_signal_count"],
        },
        "prediction_summary": {
            "predicted_result_label_counts": label_counts,
            "data_quality_status_counts": data_quality_counts,
            "fixture_duplicate_count": int(predictions["fixture_id"].dropna().duplicated().sum()),
            "prediction_write_allowed_true_count": int(predictions["prediction_write_allowed"].sum()),
            "production_ready_true_count": int(predictions["production_ready"].sum()),
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "This is not a trained or calibrated match model.",
            "The preview probabilities are deterministic scaffolding from effective team prior features.",
            "Fallback feature rows are intentionally marked in data_quality_status.",
            "All output rows keep prediction_write_allowed=false and production_ready=false.",
            "Do not surface these probabilities as production predictions without a later model policy and manifest.",
        ],
    }


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    lines: List[str] = []
    lines.append("# Day70A — Pre-GW1 Match Prediction Scaffolding")
    lines.append("")
    lines.append("Created at: `%s`" % report["created_at"])
    lines.append("")
    lines.append("Source season: `%s`" % report["source_season"])
    lines.append("Target season: `%s`" % report["target_season"])
    lines.append("Target GW: `%s`" % report["target_gw"])
    lines.append("Resolved prediction mode: `%s`" % report["resolved_prediction_mode"])
    lines.append("Model name: `%s`" % report["model_name"])
    lines.append("Model version: `%s`" % report["model_version"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Passed: `%s`" % report["passed"])
    lines.append("- Audit only: `%s`" % report["audit_only"])
    lines.append("- Writes database: `%s`" % report["writes_database"])
    lines.append("- Ready for Pre-GW1 match prediction preview: `%s`" % report["ready_for_pre_gw1_match_prediction_preview"])
    lines.append("- Ready for full Pre-GW1 match prediction: `%s`" % report["ready_for_full_pre_gw1_match_prediction"])
    lines.append("- Ready for prediction write: `%s`" % report["ready_for_prediction_write"])
    lines.append("")
    lines.append("## Row Counts")
    lines.append("")
    for key, value in report["row_counts"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Probability Quality")
    lines.append("")
    for key, value in report["probability_quality"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Prediction Summary")
    lines.append("")
    for key, value in report["prediction_summary"].items():
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
    print("=== Day70A Pre-GW1 Match Prediction Scaffolding ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("target_gw:", report["target_gw"])
    print("resolved_prediction_mode:", report["resolved_prediction_mode"])
    print("model_name:", report["model_name"])
    print("model_version:", report["model_version"])
    print("prediction_scope:", report["prediction_scope"])
    print("probability_type:", report["probability_type"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("ready_for_pre_gw1_match_prediction_preview:", report["ready_for_pre_gw1_match_prediction_preview"])
    print("ready_for_full_pre_gw1_match_prediction:", report["ready_for_full_pre_gw1_match_prediction"])
    print("ready_for_prediction_write:", report["ready_for_prediction_write"])
    print("saved_csv:", report["inputs"]["out_csv"])
    print("saved_json:", report["inputs"]["out_json"])
    print("saved_md:", report["inputs"]["out_md"])
    print()
    print("Row counts:")
    for key, value in report["row_counts"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Probability quality:")
    for key, value in report["probability_quality"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Prediction summary:")
    for key, value in report["prediction_summary"].items():
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

    if not mode_result["valid"]:
        blockers.append("Prediction mode resolver returned invalid: %s" % mode_result["errors"])
    if mode_result["resolved_prediction_mode"] != "pre_gw1_prior":
        blockers.append(
            "Day70A expects resolved_prediction_mode=pre_gw1_prior, got %s."
            % mode_result["resolved_prediction_mode"]
        )
    if args.target_gw != 1:
        blockers.append("Day70A expects target_gw=1, got %s." % args.target_gw)

    features = load_match_features(args.match_features_csv, args.source_season, args.target_season, args.target_gw)

    if not features["both_teams_have_effective_team_features"].apply(bool_value).all():
        blockers.append("Not every input fixture has complete effective team features.")

    predictions, diagnostics = build_prediction_preview(features)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_csv, index=False)

    report = build_report(
        args=args,
        features=features,
        predictions=predictions,
        diagnostics=diagnostics,
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
