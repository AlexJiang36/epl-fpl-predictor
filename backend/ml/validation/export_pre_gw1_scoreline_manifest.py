from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ml.validation.resolve_prediction_mode import resolve_prediction_mode


EXPECTED_MODEL_NAME = "pre_gw1_scoreline_prior_heuristic_v0"
EXPECTED_MODEL_VERSION = "day70c_v0"
EXPECTED_PREDICTION_SCOPE = "read_only_scoreline_preview"
EXPECTED_PROBABILITY_TYPE = "poisson_scoreline_scaffolding"

REQUIRED_SCORELINE_COLUMNS = [
    "source_season",
    "target_season",
    "target_gw",
    "prediction_mode",
    "prediction_scope",
    "probability_type",
    "model_name",
    "model_version",
    "fixture_id",
    "home_team_short_name",
    "away_team_short_name",
    "data_quality_status",
    "expected_home_goals",
    "expected_away_goals",
    "expected_total_goals",
    "expected_goal_diff_home_minus_away",
    "top_1_scoreline",
    "top_1_scoreline_probability",
    "top_2_scoreline",
    "top_2_scoreline_probability",
    "top_3_scoreline",
    "top_3_scoreline_probability",
    "scoreline_home_win_probability",
    "scoreline_draw_probability",
    "scoreline_away_win_probability",
    "scoreline_result_label",
    "one_x_two_predicted_result_label",
    "scoreline_label_matches_one_x_two_label",
    "any_team_fallback_applied",
    "prediction_write_allowed",
    "production_ready",
    "requires_scoreline_manifest_before_write",
    "calibration_status",
    "guardrail_status",
]

PRODUCTION_BLOCKERS = [
    "scoreline_model_is_heuristic_not_trained",
    "scoreline_model_is_not_calibrated",
    "scoreline_label_disagreement_requires_review",
    "prediction_write_allowed_false_for_all_rows",
    "production_ready_false_for_all_rows",
    "no_scoreline_backtest_or_calibration_report",
    "no_explicit_scoreline_db_write_path_enabled",
    "no_public_display_contract_for_scoreline_uncertainty",
]

REQUIRED_BEFORE_PRODUCTION_WRITE = [
    "trained_or_explicitly_approved_scoreline_model_policy",
    "scoreline_backtest_report",
    "expected_goals_calibration_report",
    "scoreline_probability_calibration_report",
    "guardrail_contract_for_extreme_scorelines_and_goal_totals",
    "fallback_policy_review_for_promoted_teams",
    "consistency_policy_between_1x2_and_scoreline_outputs",
    "active_model_registry_entry_for_scoreline_model",
    "database_write_path_with_dry_run",
    "public_display_copy_that_distinguishes preview from prediction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Pre-GW1 scoreline manifest and safety contract from the Day70C "
            "read-only scoreline preview. This validates scoreline preview safety and explicitly "
            "blocks production/DB writes."
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
    parser.add_argument("--scoreline-preview-csv", required=True, help="Day70C scoreline preview CSV.")
    parser.add_argument("--day70c-json", required=True, help="Day70C JSON audit report.")
    parser.add_argument("--day70b-manifest-json", default="", help="Optional Day70B 1X2 manifest JSON.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    text_value = str(value).strip().lower()
    return text_value in {"true", "1", "yes", "y"}


def sha256_file(path_value: str) -> Optional[str]:
    if not path_value:
        return None

    path = Path(path_value)
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path_value: str, required: bool = True) -> Dict[str, Any]:
    if not path_value:
        if required:
            raise RuntimeError("Required JSON path was empty.")
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
            "sha256": sha256_file(str(path)),
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


def load_scoreline_csv(path_value: str, source_season: str, target_season: str, target_gw: int) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("scoreline preview CSV does not exist: %s" % path)

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("scoreline preview CSV is empty: %s" % path)

    missing = [col for col in REQUIRED_SCORELINE_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError("scoreline preview CSV missing required columns: %s" % missing)

    filtered = df[
        (df["source_season"].astype(str) == str(source_season))
        & (df["target_season"].astype(str) == str(target_season))
        & (df["target_gw"].astype(int) == int(target_gw))
    ].copy()

    if filtered.empty:
        raise RuntimeError(
            "No scoreline rows found for source_season=%s target_season=%s target_gw=%s."
            % (source_season, target_season, target_gw)
        )

    return filtered


def validate_scoreline_rows(scorelines: pd.DataFrame, day70c_report: Dict[str, Any], blockers: List[str], warnings: List[str]) -> Dict[str, Any]:
    fixture_duplicate_count = int(scorelines["fixture_id"].dropna().duplicated().sum())
    if fixture_duplicate_count > 0:
        blockers.append("Scoreline preview contains duplicate fixture_id rows.")

    expected_goal_missing = {
        "expected_home_goals": int(scorelines["expected_home_goals"].isna().sum()),
        "expected_away_goals": int(scorelines["expected_away_goals"].isna().sum()),
        "expected_total_goals": int(scorelines["expected_total_goals"].isna().sum()),
    }
    for col, missing_count in expected_goal_missing.items():
        if missing_count > 0:
            blockers.append("%s contains missing values." % col)

    expected_goal_range_violations = {
        "expected_home_goals": int(((scorelines["expected_home_goals"] <= 0) | (scorelines["expected_home_goals"] > 5)).sum()),
        "expected_away_goals": int(((scorelines["expected_away_goals"] <= 0) | (scorelines["expected_away_goals"] > 5)).sum()),
    }
    for col, bad_count in expected_goal_range_violations.items():
        if bad_count > 0:
            blockers.append("%s has values outside expected preview range." % col)

    top_scoreline_missing = {
        "top_1_scoreline": int(scorelines["top_1_scoreline"].isna().sum()),
        "top_2_scoreline": int(scorelines["top_2_scoreline"].isna().sum()),
        "top_3_scoreline": int(scorelines["top_3_scoreline"].isna().sum()),
    }
    for col, missing_count in top_scoreline_missing.items():
        if missing_count > 0:
            blockers.append("%s contains missing values." % col)

    scoreline_probability_cols = [
        "top_1_scoreline_probability",
        "top_2_scoreline_probability",
        "top_3_scoreline_probability",
        "scoreline_home_win_probability",
        "scoreline_draw_probability",
        "scoreline_away_win_probability",
    ]

    probability_missing_values: Dict[str, int] = {}
    probability_range_violations: Dict[str, int] = {}

    for col in scoreline_probability_cols:
        missing_count = int(scorelines[col].isna().sum())
        probability_missing_values[col] = missing_count
        if missing_count > 0:
            blockers.append("%s contains missing values." % col)

        range_bad = int(((scorelines[col] < 0) | (scorelines[col] > 1)).sum())
        probability_range_violations[col] = range_bad
        if range_bad > 0:
            blockers.append("%s has values outside [0, 1]." % col)

    result_probability_cols = [
        "scoreline_home_win_probability",
        "scoreline_draw_probability",
        "scoreline_away_win_probability",
    ]
    max_result_probability_sum_error = float((scorelines[result_probability_cols].sum(axis=1) - 1.0).abs().max())
    if max_result_probability_sum_error > 0.00001:
        blockers.append("Scoreline result probabilities do not sum to 1 within tolerance.")

    if "score_grid_probability_sum" in scorelines.columns:
        min_score_grid_probability_sum = float(scorelines["score_grid_probability_sum"].min())
        max_score_grid_tail_probability = float(scorelines["score_grid_tail_probability"].max())
        if min_score_grid_probability_sum < 0.995:
            warnings.append("Score grid probability sum below 0.995; consider increasing score grid max goals.")
    else:
        min_score_grid_probability_sum = None
        max_score_grid_tail_probability = None

    valid_labels = {"home_win", "draw", "away_win"}
    scoreline_labels = set(scorelines["scoreline_result_label"].dropna().astype(str).unique().tolist())
    invalid_labels = sorted(scoreline_labels - valid_labels)
    if invalid_labels:
        blockers.append("Invalid scoreline_result_label values found: %s" % invalid_labels)

    label_match_false_count = int((~scorelines["scoreline_label_matches_one_x_two_label"].apply(bool_value)).sum())
    if label_match_false_count > 0:
        warnings.append(
            "%s scoreline result label(s) differ from Day70A 1X2 preview labels. "
            "This remains allowed for preview but blocks production use until reviewed."
            % label_match_false_count
        )

    write_true_count = int(scorelines["prediction_write_allowed"].apply(bool_value).sum())
    production_true_count = int(scorelines["production_ready"].apply(bool_value).sum())
    manifest_false_count = int((~scorelines["requires_scoreline_manifest_before_write"].apply(bool_value)).sum())

    if write_true_count != 0:
        blockers.append("prediction_write_allowed must remain false for every scoreline preview row.")
    if production_true_count != 0:
        blockers.append("production_ready must remain false for every scoreline preview row.")
    if manifest_false_count != 0:
        blockers.append("requires_scoreline_manifest_before_write must be true for every scoreline preview row.")

    expected_values = {
        "model_name": EXPECTED_MODEL_NAME,
        "model_version": EXPECTED_MODEL_VERSION,
        "prediction_scope": EXPECTED_PREDICTION_SCOPE,
        "probability_type": EXPECTED_PROBABILITY_TYPE,
    }
    for col, expected in expected_values.items():
        values = set(scorelines[col].dropna().astype(str).unique().tolist())
        if values != {expected}:
            blockers.append("%s expected %s, got %s." % (col, expected, sorted(values)))

    calibration_values = sorted(scorelines["calibration_status"].dropna().astype(str).unique().tolist())
    guardrail_values = sorted(scorelines["guardrail_status"].dropna().astype(str).unique().tolist())

    if calibration_values != ["not_calibrated_scoreline_preview_only"]:
        warnings.append("Unexpected calibration_status values: %s" % calibration_values)
    if guardrail_values != ["basic_score_grid_sanity_only"]:
        warnings.append("Unexpected guardrail_status values: %s" % guardrail_values)

    day70c_passed = nested_get(day70c_report, ["passed"], None)
    if day70c_passed is not True:
        blockers.append("Day70C JSON did not pass.")

    day70c_rows = nested_get(day70c_report, ["row_counts", "scoreline_preview_rows"], None)
    if day70c_rows is not None and int(day70c_rows) != int(len(scorelines)):
        blockers.append("Day70C JSON row count does not match scoreline CSV row count.")

    day70c_write_ready = nested_get(day70c_report, ["ready_for_prediction_write"], None)
    if day70c_write_ready is not False:
        blockers.append("Day70C report must keep ready_for_prediction_write=false.")

    label_counts = scorelines["scoreline_result_label"].value_counts(dropna=False).to_dict()
    top_scoreline_counts = scorelines["top_1_scoreline"].value_counts(dropna=False).to_dict()
    data_quality_counts = scorelines["data_quality_status"].value_counts(dropna=False).to_dict()

    return {
        "row_count": int(len(scorelines)),
        "fixture_duplicate_count": fixture_duplicate_count,
        "expected_goal_missing_values": expected_goal_missing,
        "expected_goal_range_violations": expected_goal_range_violations,
        "probability_missing_values": probability_missing_values,
        "probability_range_violations": probability_range_violations,
        "top_scoreline_missing_values": top_scoreline_missing,
        "max_result_probability_sum_error": max_result_probability_sum_error,
        "min_score_grid_probability_sum": min_score_grid_probability_sum,
        "max_score_grid_tail_probability": max_score_grid_tail_probability,
        "invalid_labels": invalid_labels,
        "scoreline_label_disagreement_count": label_match_false_count,
        "prediction_write_allowed_true_count": write_true_count,
        "production_ready_true_count": production_true_count,
        "requires_scoreline_manifest_false_count": manifest_false_count,
        "scoreline_result_label_counts": label_counts,
        "top_1_scoreline_counts": top_scoreline_counts,
        "data_quality_status_counts": data_quality_counts,
        "fallback_row_count": int(scorelines["any_team_fallback_applied"].apply(bool_value).sum()),
        "full_prior_row_count": int((~scorelines["any_team_fallback_applied"].apply(bool_value)).sum()),
        "calibration_status_values": calibration_values,
        "guardrail_status_values": guardrail_values,
    }


def build_safety_contract(scoreline_validation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "contract_version": "day70d_v1",
        "artifact_type": "pre_gw1_scoreline_preview_manifest",
        "allowed_uses": {
            "local_audit": True,
            "ops_preview": True,
            "portfolio_internal_demo_with_disclaimer": True,
            "public_scoreline_prediction": False,
            "db_write": False,
            "active_model_registry_candidate": False,
            "champion_simulation_input": False,
            "betting_or_decision_recommendation": False,
        },
        "disallowed_uses": [
            "Do not write this scoreline preview to production match_predictions.",
            "Do not present likely scorelines as calibrated predictions.",
            "Do not use scorelines as champion simulation inputs yet.",
            "Do not hide the scoreline/1X2 disagreement count.",
            "Do not hide fallback rows from downstream consumers.",
            "Do not set production_ready=true without a later scoreline model policy and manifest update.",
        ],
        "required_before_production_write": REQUIRED_BEFORE_PRODUCTION_WRITE,
        "current_production_blockers": PRODUCTION_BLOCKERS,
        "row_level_requirements_enforced": [
            "fixture_id unique",
            "expected goals present and positive",
            "top scorelines present",
            "scoreline probabilities present and within [0, 1]",
            "scoreline outcome probabilities sum to 1",
            "valid scoreline_result_label",
            "prediction_write_allowed=false",
            "production_ready=false",
            "requires_scoreline_manifest_before_write=true",
        ],
        "preview_validation_summary": scoreline_validation,
    }


def build_manifest(
    args: argparse.Namespace,
    scorelines: pd.DataFrame,
    day70c: Dict[str, Any],
    day70b: Dict[str, Any],
    mode_result: Dict[str, Any],
    scoreline_validation: Dict[str, Any],
    safety_contract: Dict[str, Any],
    blockers: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    if not mode_result["valid"]:
        blockers.append("Prediction mode resolver returned invalid: %s" % mode_result["errors"])
    if mode_result["resolved_prediction_mode"] != "pre_gw1_prior":
        blockers.append("Day70D expects resolved_prediction_mode=pre_gw1_prior.")
    if args.target_gw != 1:
        blockers.append("Day70D expects target_gw=1.")

    day70b_data = day70b.get("data") if day70b.get("loaded") else None
    if day70b.get("exists") and not day70b.get("loaded"):
        warnings.append("Day70B manifest JSON was provided but could not be loaded: %s" % day70b.get("error"))
    if day70b_data is not None:
        if nested_get(day70b_data, ["passed"], None) is not True:
            blockers.append("Day70B manifest did not pass.")
        if nested_get(day70b_data, ["ready_for_ops_preview"], None) is not True:
            blockers.append("Day70B manifest is not ready_for_ops_preview.")
        if nested_get(day70b_data, ["ready_for_prediction_write"], None) is not False:
            blockers.append("Day70B manifest must keep ready_for_prediction_write=false.")

    artifact_fingerprints = {
        "scoreline_preview_csv": {
            "path": args.scoreline_preview_csv,
            "sha256": sha256_file(args.scoreline_preview_csv),
        },
        "day70c_json": {
            "path": args.day70c_json,
            "sha256": sha256_file(args.day70c_json),
        },
        "day70b_manifest_json": {
            "path": args.day70b_manifest_json or None,
            "sha256": sha256_file(args.day70b_manifest_json) if args.day70b_manifest_json else None,
        },
    }

    return {
        "created_at": utc_now(),
        "source_season": args.source_season,
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": mode_result["resolved_prediction_mode"],
        "manifest_version": "day70d_v1",
        "model_name": EXPECTED_MODEL_NAME,
        "model_version": EXPECTED_MODEL_VERSION,
        "prediction_scope": EXPECTED_PREDICTION_SCOPE,
        "probability_type": EXPECTED_PROBABILITY_TYPE,
        "audit_only": True,
        "writes_database": False,
        "passed": len(blockers) == 0,
        "ready_for_pre_gw1_scoreline_manifest": len(blockers) == 0,
        "ready_for_ops_preview": len(blockers) == 0,
        "ready_for_public_scoreline_prediction": False,
        "ready_for_full_pre_gw1_scoreline_prediction": False,
        "ready_for_prediction_write": False,
        "reason_prediction_write_not_ready": (
            "Day70D verifies and documents a read-only scoreline preview. Production scoreline writes remain blocked "
            "until scoreline model policy, backtest/calibration, guardrails, consistency review, active registry policy, "
            "and DB write path exist."
        ),
        "artifact_fingerprints": artifact_fingerprints,
        "source_reports": {
            "day70c": {
                "exists": day70c.get("exists"),
                "loaded": day70c.get("loaded"),
                "passed": nested_get(day70c.get("data"), ["passed"], None),
                "ready_for_pre_gw1_scoreline_preview": nested_get(
                    day70c.get("data"), ["ready_for_pre_gw1_scoreline_preview"], None
                ),
                "ready_for_prediction_write": nested_get(day70c.get("data"), ["ready_for_prediction_write"], None),
            },
            "day70b": {
                "exists": day70b.get("exists"),
                "loaded": day70b.get("loaded"),
                "passed": nested_get(day70b.get("data"), ["passed"], None),
                "ready_for_ops_preview": nested_get(day70b.get("data"), ["ready_for_ops_preview"], None),
                "ready_for_prediction_write": nested_get(day70b.get("data"), ["ready_for_prediction_write"], None),
            },
        },
        "mode_resolution": mode_result,
        "scoreline_validation": scoreline_validation,
        "safety_contract": safety_contract,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "This manifest makes Day70C scoreline preview auditable and intentionally non-production.",
            "The preview may be used for internal ops/demo inspection with clear disclaimer.",
            "Scoreline/1X2 disagreements are visible and must be reviewed before production.",
            "Every row remains blocked from DB write by prediction_write_allowed=false and production_ready=false.",
            "Fallback rows remain visible through data_quality_status and any_team_fallback_applied.",
        ],
    }


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    lines: List[str] = []
    lines.append("# Day70D — Pre-GW1 Scoreline Manifest and Safety Contract")
    lines.append("")
    lines.append("Created at: `%s`" % report["created_at"])
    lines.append("")
    lines.append("Source season: `%s`" % report["source_season"])
    lines.append("Target season: `%s`" % report["target_season"])
    lines.append("Target GW: `%s`" % report["target_gw"])
    lines.append("Resolved prediction mode: `%s`" % report["resolved_prediction_mode"])
    lines.append("Manifest version: `%s`" % report["manifest_version"])
    lines.append("Model: `%s` / `%s`" % (report["model_name"], report["model_version"]))
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Passed: `%s`" % report["passed"])
    lines.append("- Audit only: `%s`" % report["audit_only"])
    lines.append("- Writes database: `%s`" % report["writes_database"])
    lines.append("- Ready for Pre-GW1 scoreline manifest: `%s`" % report["ready_for_pre_gw1_scoreline_manifest"])
    lines.append("- Ready for ops preview: `%s`" % report["ready_for_ops_preview"])
    lines.append("- Ready for public scoreline prediction: `%s`" % report["ready_for_public_scoreline_prediction"])
    lines.append("- Ready for prediction write: `%s`" % report["ready_for_prediction_write"])
    lines.append("")
    lines.append("## Scoreline Validation")
    lines.append("")
    for key, value in report["scoreline_validation"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Allowed Uses")
    lines.append("")
    for key, value in report["safety_contract"]["allowed_uses"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Production Blockers")
    lines.append("")
    for blocker in report["safety_contract"]["current_production_blockers"]:
        lines.append("- %s" % blocker)
    lines.append("")
    lines.append("## Required Before Production Write")
    lines.append("")
    for item in report["safety_contract"]["required_before_production_write"]:
        lines.append("- %s" % item)
    lines.append("")
    lines.append("## Artifact Fingerprints")
    lines.append("")
    for key, value in report["artifact_fingerprints"].items():
        lines.append("- %s: path=`%s`, sha256=`%s`" % (key, value.get("path"), value.get("sha256")))
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
    print("=== Day70D Pre-GW1 Scoreline Manifest ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("target_gw:", report["target_gw"])
    print("resolved_prediction_mode:", report["resolved_prediction_mode"])
    print("manifest_version:", report["manifest_version"])
    print("model_name:", report["model_name"])
    print("model_version:", report["model_version"])
    print("prediction_scope:", report["prediction_scope"])
    print("probability_type:", report["probability_type"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("ready_for_pre_gw1_scoreline_manifest:", report["ready_for_pre_gw1_scoreline_manifest"])
    print("ready_for_ops_preview:", report["ready_for_ops_preview"])
    print("ready_for_public_scoreline_prediction:", report["ready_for_public_scoreline_prediction"])
    print("ready_for_prediction_write:", report["ready_for_prediction_write"])
    print()
    print("Scoreline validation:")
    for key, value in report["scoreline_validation"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Allowed uses:")
    for key, value in report["safety_contract"]["allowed_uses"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Current production blockers:")
    for blocker in report["safety_contract"]["current_production_blockers"]:
        print("-", blocker)
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

    scorelines = load_scoreline_csv(args.scoreline_preview_csv, args.source_season, args.target_season, args.target_gw)
    day70c = load_json(args.day70c_json, required=True)
    day70b = load_json(args.day70b_manifest_json, required=False)

    scoreline_validation = validate_scoreline_rows(
        scorelines=scorelines,
        day70c_report=day70c["data"],
        blockers=blockers,
        warnings=warnings,
    )
    safety_contract = build_safety_contract(scoreline_validation)

    manifest = build_manifest(
        args=args,
        scorelines=scorelines,
        day70c=day70c,
        day70b=day70b,
        mode_result=mode_result,
        scoreline_validation=scoreline_validation,
        safety_contract=safety_contract,
        blockers=blockers,
        warnings=warnings,
    )

    write_json(manifest, args.out_json)
    write_markdown(manifest, args.out_md)
    print_report(manifest)

    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
