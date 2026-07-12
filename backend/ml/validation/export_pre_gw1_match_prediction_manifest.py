from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ml.validation.resolve_prediction_mode import resolve_prediction_mode


EXPECTED_MODEL_NAME = "pre_gw1_match_prior_heuristic_v0"
EXPECTED_MODEL_VERSION = "day70a_v0"
EXPECTED_PREDICTION_SCOPE = "read_only_preview"
EXPECTED_PROBABILITY_TYPE = "heuristic_scaffolding"

REQUIRED_PREVIEW_COLUMNS = [
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
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "probability_sum",
    "predicted_result_label",
    "any_team_fallback_applied",
    "prediction_write_allowed",
    "production_ready",
    "requires_manifest_before_write",
    "calibration_status",
    "guardrail_status",
]

PRODUCTION_BLOCKERS = [
    "preview_model_is_heuristic_not_trained",
    "preview_model_is_not_calibrated",
    "prediction_write_allowed_false_for_all_rows",
    "production_ready_false_for_all_rows",
    "no_explicit_db_write_path_enabled",
    "no_active_model_registry_entry_for_pre_gw1_production",
    "no_public_display_contract_for_probability_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Pre-GW1 match prediction manifest and safety contract from the Day70A "
            "read-only prediction preview. This validates preview safety and explicitly blocks "
            "production/DB writes."
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
    parser.add_argument("--prediction-preview-csv", required=True, help="Day70A preview CSV.")
    parser.add_argument("--day70a-json", required=True, help="Day70A JSON audit report.")
    parser.add_argument("--day69b-json", default="", help="Optional Day69B fallback policy JSON report.")
    parser.add_argument("--out-json", required=True, help="Output manifest JSON.")
    parser.add_argument("--out-md", required=True, help="Output manifest Markdown.")
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


def load_preview_csv(path_value: str, source_season: str, target_season: str, target_gw: int) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("prediction preview CSV does not exist: %s" % path)

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("prediction preview CSV is empty: %s" % path)

    missing = [col for col in REQUIRED_PREVIEW_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError("prediction preview CSV missing required columns: %s" % missing)

    filtered = df[
        (df["source_season"].astype(str) == str(source_season))
        & (df["target_season"].astype(str) == str(target_season))
        & (df["target_gw"].astype(int) == int(target_gw))
    ].copy()

    if filtered.empty:
        raise RuntimeError(
            "No preview rows found for source_season=%s target_season=%s target_gw=%s."
            % (source_season, target_season, target_gw)
        )

    return filtered


def validate_preview_rows(preview: pd.DataFrame, day70a_report: Dict[str, Any], blockers: List[str], warnings: List[str]) -> Dict[str, Any]:
    fixture_duplicate_count = int(preview["fixture_id"].dropna().duplicated().sum())
    if fixture_duplicate_count > 0:
        blockers.append("Prediction preview contains duplicate fixture_id rows.")

    probability_cols = ["home_win_probability", "draw_probability", "away_win_probability"]
    missing_probability_values: Dict[str, int] = {}
    probability_range_violations: Dict[str, int] = {}

    for col in probability_cols:
        missing_count = int(preview[col].isna().sum())
        missing_probability_values[col] = missing_count
        if missing_count > 0:
            blockers.append("%s contains missing values." % col)

        range_bad = int(((preview[col] < 0) | (preview[col] > 1)).sum())
        probability_range_violations[col] = range_bad
        if range_bad > 0:
            blockers.append("%s has probability values outside [0, 1]." % col)

    computed_probability_sum_error = float((preview[probability_cols].sum(axis=1) - 1.0).abs().max())
    reported_probability_sum_error = float((preview["probability_sum"] - 1.0).abs().max())
    max_probability_sum_error = max(computed_probability_sum_error, reported_probability_sum_error)

    if max_probability_sum_error > 0.00001:
        blockers.append("Probability sums are not within tolerance.")

    valid_labels = {"home_win", "draw", "away_win"}
    labels = set(preview["predicted_result_label"].dropna().astype(str).unique().tolist())
    invalid_labels = sorted(labels - valid_labels)
    if invalid_labels:
        blockers.append("Invalid predicted_result_label values found: %s" % invalid_labels)

    write_true_count = int(preview["prediction_write_allowed"].apply(bool_value).sum())
    production_true_count = int(preview["production_ready"].apply(bool_value).sum())
    manifest_required_false_count = int((~preview["requires_manifest_before_write"].apply(bool_value)).sum())

    if write_true_count != 0:
        blockers.append("prediction_write_allowed must remain false for every Day70A preview row.")
    if production_true_count != 0:
        blockers.append("production_ready must remain false for every Day70A preview row.")
    if manifest_required_false_count != 0:
        blockers.append("requires_manifest_before_write must be true for every Day70A preview row.")

    expected_values = {
        "model_name": EXPECTED_MODEL_NAME,
        "model_version": EXPECTED_MODEL_VERSION,
        "prediction_scope": EXPECTED_PREDICTION_SCOPE,
        "probability_type": EXPECTED_PROBABILITY_TYPE,
    }

    for col, expected in expected_values.items():
        values = set(preview[col].dropna().astype(str).unique().tolist())
        if values != {expected}:
            blockers.append("%s expected %s, got %s." % (col, expected, sorted(values)))

    calibration_values = set(preview["calibration_status"].dropna().astype(str).unique().tolist())
    guardrail_values = set(preview["guardrail_status"].dropna().astype(str).unique().tolist())

    if calibration_values != {"not_calibrated_preview_only"}:
        warnings.append("Unexpected calibration_status values: %s" % sorted(calibration_values))
    if guardrail_values != {"basic_probability_sanity_only"}:
        warnings.append("Unexpected guardrail_status values: %s" % sorted(guardrail_values))

    day70a_rows = nested_get(day70a_report, ["row_counts", "prediction_preview_rows"], None)
    if day70a_rows is not None and int(day70a_rows) != int(len(preview)):
        blockers.append("Day70A JSON row count does not match preview CSV row count.")

    day70a_passed = nested_get(day70a_report, ["passed"], None)
    if day70a_passed is not True:
        blockers.append("Day70A JSON did not pass.")

    day70a_write_ready = nested_get(day70a_report, ["ready_for_prediction_write"], None)
    if day70a_write_ready is not False:
        blockers.append("Day70A report must keep ready_for_prediction_write=false.")

    label_counts = preview["predicted_result_label"].value_counts(dropna=False).to_dict()
    data_quality_counts = preview["data_quality_status"].value_counts(dropna=False).to_dict()

    return {
        "row_count": int(len(preview)),
        "fixture_duplicate_count": fixture_duplicate_count,
        "missing_probability_values": missing_probability_values,
        "probability_range_violations": probability_range_violations,
        "max_probability_sum_error": max_probability_sum_error,
        "invalid_labels": invalid_labels,
        "prediction_write_allowed_true_count": write_true_count,
        "production_ready_true_count": production_true_count,
        "requires_manifest_false_count": manifest_required_false_count,
        "predicted_result_label_counts": label_counts,
        "data_quality_status_counts": data_quality_counts,
        "fallback_row_count": int(preview["any_team_fallback_applied"].apply(bool_value).sum()),
        "full_prior_row_count": int((~preview["any_team_fallback_applied"].apply(bool_value)).sum()),
        "calibration_status_values": sorted(calibration_values),
        "guardrail_status_values": sorted(guardrail_values),
    }


def build_safety_contract(preview_validation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "contract_version": "day70b_v1",
        "artifact_type": "pre_gw1_match_prediction_preview_manifest",
        "allowed_uses": {
            "local_audit": True,
            "ops_preview": True,
            "portfolio_internal_demo_with_disclaimer": True,
            "public_production_prediction": False,
            "db_write": False,
            "active_model_registry_candidate": False,
            "champion_simulation_input": False,
        },
        "disallowed_uses": [
            "Do not write this preview to production match_predictions.",
            "Do not present this as a calibrated model output.",
            "Do not use it as champion simulation input yet.",
            "Do not hide fallback rows from downstream consumers.",
            "Do not set production_ready=true without a later model policy and manifest update.",
        ],
        "required_before_production_write": [
            "trained_or_explicitly_approved_model_policy",
            "calibration_or_backtest_report",
            "guardrail_contract_for_extreme_probabilities",
            "fallback_policy_review_for_promoted_teams",
            "active_model_registry_entry",
            "database_write_path_with_dry_run",
            "public_display_copy_that_distinguishes preview from prediction",
        ],
        "current_production_blockers": PRODUCTION_BLOCKERS,
        "row_level_requirements_enforced": [
            "fixture_id unique",
            "probabilities present and within [0, 1]",
            "probabilities sum to 1",
            "valid predicted_result_label",
            "prediction_write_allowed=false",
            "production_ready=false",
            "requires_manifest_before_write=true",
        ],
        "preview_validation_summary": preview_validation,
    }


def build_manifest(
    args: argparse.Namespace,
    preview: pd.DataFrame,
    day70a: Dict[str, Any],
    day69b: Dict[str, Any],
    mode_result: Dict[str, Any],
    preview_validation: Dict[str, Any],
    safety_contract: Dict[str, Any],
    blockers: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    if not mode_result["valid"]:
        blockers.append("Prediction mode resolver returned invalid: %s" % mode_result["errors"])
    if mode_result["resolved_prediction_mode"] != "pre_gw1_prior":
        blockers.append("Day70B expects resolved_prediction_mode=pre_gw1_prior.")
    if args.target_gw != 1:
        blockers.append("Day70B expects target_gw=1.")

    day69b_data = day69b.get("data") if day69b.get("loaded") else None
    if day69b.get("exists") and not day69b.get("loaded"):
        warnings.append("Day69B JSON was provided but could not be loaded: %s" % day69b.get("error"))
    if day69b_data is not None:
        if nested_get(day69b_data, ["passed"], None) is not True:
            blockers.append("Day69B JSON did not pass.")
        if nested_get(day69b_data, ["ready_for_pre_gw1_match_features_with_fallback"], None) is not True:
            blockers.append("Day69B report is not ready_for_pre_gw1_match_features_with_fallback.")

    artifact_fingerprints = {
        "prediction_preview_csv": {
            "path": args.prediction_preview_csv,
            "sha256": sha256_file(args.prediction_preview_csv),
        },
        "day70a_json": {
            "path": args.day70a_json,
            "sha256": sha256_file(args.day70a_json),
        },
        "day69b_json": {
            "path": args.day69b_json or None,
            "sha256": sha256_file(args.day69b_json) if args.day69b_json else None,
        },
    }

    manifest = {
        "created_at": utc_now(),
        "source_season": args.source_season,
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": mode_result["resolved_prediction_mode"],
        "manifest_version": "day70b_v1",
        "model_name": EXPECTED_MODEL_NAME,
        "model_version": EXPECTED_MODEL_VERSION,
        "prediction_scope": EXPECTED_PREDICTION_SCOPE,
        "probability_type": EXPECTED_PROBABILITY_TYPE,
        "audit_only": True,
        "writes_database": False,
        "passed": len(blockers) == 0,
        "ready_for_pre_gw1_match_prediction_manifest": len(blockers) == 0,
        "ready_for_ops_preview": len(blockers) == 0,
        "ready_for_public_prediction": False,
        "ready_for_full_pre_gw1_match_prediction": False,
        "ready_for_prediction_write": False,
        "reason_prediction_write_not_ready": (
            "Day70B verifies and documents a read-only heuristic preview. Production writes remain blocked "
            "until model policy, calibration/backtest, guardrails, active registry policy, and DB write path exist."
        ),
        "artifact_fingerprints": artifact_fingerprints,
        "source_reports": {
            "day70a": {
                "exists": day70a.get("exists"),
                "loaded": day70a.get("loaded"),
                "passed": nested_get(day70a.get("data"), ["passed"], None),
                "ready_for_prediction_write": nested_get(day70a.get("data"), ["ready_for_prediction_write"], None),
                "ready_for_pre_gw1_match_prediction_preview": nested_get(
                    day70a.get("data"), ["ready_for_pre_gw1_match_prediction_preview"], None
                ),
            },
            "day69b": {
                "exists": day69b.get("exists"),
                "loaded": day69b.get("loaded"),
                "passed": nested_get(day69b.get("data"), ["passed"], None),
                "ready_for_pre_gw1_match_features_with_fallback": nested_get(
                    day69b.get("data"), ["ready_for_pre_gw1_match_features_with_fallback"], None
                ),
            },
        },
        "mode_resolution": mode_result,
        "preview_validation": preview_validation,
        "safety_contract": safety_contract,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "This manifest makes Day70A preview auditable and intentionally non-production.",
            "The preview may be used for internal ops/demo inspection with clear disclaimer.",
            "Every row remains blocked from DB write by prediction_write_allowed=false and production_ready=false.",
            "Fallback rows remain visible through data_quality_status and any_team_fallback_applied.",
        ],
    }
    return manifest


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    lines: List[str] = []
    lines.append("# Day70B — Pre-GW1 Match Prediction Manifest and Safety Contract")
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
    lines.append("- Ready for Pre-GW1 match prediction manifest: `%s`" % report["ready_for_pre_gw1_match_prediction_manifest"])
    lines.append("- Ready for ops preview: `%s`" % report["ready_for_ops_preview"])
    lines.append("- Ready for public prediction: `%s`" % report["ready_for_public_prediction"])
    lines.append("- Ready for prediction write: `%s`" % report["ready_for_prediction_write"])
    lines.append("")
    lines.append("## Preview Validation")
    lines.append("")
    for key, value in report["preview_validation"].items():
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
    print("=== Day70B Pre-GW1 Match Prediction Manifest ===")
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
    print("ready_for_pre_gw1_match_prediction_manifest:", report["ready_for_pre_gw1_match_prediction_manifest"])
    print("ready_for_ops_preview:", report["ready_for_ops_preview"])
    print("ready_for_public_prediction:", report["ready_for_public_prediction"])
    print("ready_for_prediction_write:", report["ready_for_prediction_write"])
    print()
    print("Preview validation:")
    for key, value in report["preview_validation"].items():
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

    preview = load_preview_csv(args.prediction_preview_csv, args.source_season, args.target_season, args.target_gw)
    day70a = load_json(args.day70a_json, required=True)
    day69b = load_json(args.day69b_json, required=False)

    preview_validation = validate_preview_rows(
        preview=preview,
        day70a_report=day70a["data"],
        blockers=blockers,
        warnings=warnings,
    )
    safety_contract = build_safety_contract(preview_validation)

    manifest = build_manifest(
        args=args,
        preview=preview,
        day70a=day70a,
        day69b=day69b,
        mode_result=mode_result,
        preview_validation=preview_validation,
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
