from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_PREDICTION_MODES = {
    "auto",
    "pre_gw1_prior",
    "early_season_blend",
    "normal_weekly",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the prediction mode for a target GW. "
            "This is read-only and does not inspect or write the database."
        )
    )
    parser.add_argument("--season", required=True, help="Target season, for example 2026_27.")
    parser.add_argument("--target-gw", "--target_gw", type=int, required=True, help="Target gameweek.")
    parser.add_argument(
        "--prediction-mode",
        "--prediction_mode",
        default="auto",
        choices=sorted(VALID_PREDICTION_MODES),
        help="Requested prediction mode. Default: auto.",
    )
    parser.add_argument(
        "--prior-season",
        "--prior_season",
        default=None,
        help="Prior/source season. Required for pre_gw1_prior and early_season_blend.",
    )
    parser.add_argument(
        "--stabilization-gw",
        "--stabilization_gw",
        type=int,
        default=6,
        help="First GW that should resolve to normal_weekly in auto mode. Default: 6.",
    )
    parser.add_argument(
        "--allow-experimental-mode",
        action="store_true",
        help="Allow forcing a prediction mode outside its normal GW window.",
    )
    parser.add_argument("--out-json", "--out_json", default="", help="Optional JSON output path.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_prior_weight_for_resolved_mode(target_gw: int, resolved_mode: str, stabilization_gw: int) -> float:
    if resolved_mode == "pre_gw1_prior":
        return 1.0

    if resolved_mode == "early_season_blend":
        denominator = max(1, stabilization_gw - 1)
        weight = float(stabilization_gw - target_gw) / float(denominator)
        return round(max(0.0, min(1.0, weight)), 4)

    return 0.0


def auto_resolve_mode(target_gw: int, stabilization_gw: int) -> str:
    if target_gw == 1:
        return "pre_gw1_prior"
    if 2 <= target_gw < stabilization_gw:
        return "early_season_blend"
    return "normal_weekly"


def expected_mode_window(resolved_mode: str, stabilization_gw: int) -> str:
    if resolved_mode == "pre_gw1_prior":
        return "target_gw == 1"
    if resolved_mode == "early_season_blend":
        return "2 <= target_gw < %s" % stabilization_gw
    return "target_gw >= %s" % stabilization_gw


def is_in_expected_window(target_gw: int, mode: str, stabilization_gw: int) -> bool:
    if mode == "pre_gw1_prior":
        return target_gw == 1
    if mode == "early_season_blend":
        return 2 <= target_gw < stabilization_gw
    if mode == "normal_weekly":
        return target_gw >= stabilization_gw
    return False


def build_mode_metadata(
    resolved_mode: str,
    target_gw: int,
    prior_season: Optional[str],
    stabilization_gw: int,
) -> Dict[str, Any]:
    prior_weight = default_prior_weight_for_resolved_mode(
        target_gw=target_gw,
        resolved_mode=resolved_mode,
        stabilization_gw=stabilization_gw,
    )

    requires_prior = resolved_mode in {"pre_gw1_prior", "early_season_blend"}

    if resolved_mode == "pre_gw1_prior":
        allowed_outputs = [
            "player_predictions",
            "match_predictions",
            "opening_squad",
            "captain_recommendations",
            "season_outlook",
            "champion_probabilities",
            "golden_boot_probabilities",
        ]
        model_name_suffix = "_pre_gw1"
    elif resolved_mode == "early_season_blend":
        allowed_outputs = [
            "player_predictions",
            "match_predictions",
            "squad_recommendations",
            "transfer_recommendations",
            "captain_recommendations",
            "season_outlook",
        ]
        model_name_suffix = "_early"
    else:
        allowed_outputs = [
            "player_predictions",
            "match_predictions",
            "squad_recommendations",
            "transfer_recommendations",
            "captain_recommendations",
        ]
        model_name_suffix = "_normal"

    return {
        "resolved_prediction_mode": resolved_mode,
        "mode_window": expected_mode_window(resolved_mode, stabilization_gw),
        "requires_prior_season": requires_prior,
        "requires_current_season_actuals": resolved_mode != "pre_gw1_prior",
        "prior_artifacts_required": requires_prior,
        "safe_to_run_without_prior_artifacts": resolved_mode == "normal_weekly",
        "default_prior_weight": prior_weight,
        "default_current_weight": round(1.0 - prior_weight, 4),
        "prior_season_used": prior_season if requires_prior else None,
        "model_name_suffix": model_name_suffix,
        "allowed_outputs": allowed_outputs,
        "should_include_season_outlook_by_default": resolved_mode in {"pre_gw1_prior", "early_season_blend"},
        "gw6_safe_exit_applies": resolved_mode == "normal_weekly" and target_gw >= stabilization_gw,
    }


def resolve_prediction_mode(
    season: str,
    target_gw: int,
    requested_prediction_mode: str = "auto",
    prior_season: Optional[str] = None,
    stabilization_gw: int = 6,
    allow_experimental_mode: bool = False,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if requested_prediction_mode not in VALID_PREDICTION_MODES:
        errors.append("Invalid prediction mode: %s" % requested_prediction_mode)

    if target_gw < 1:
        errors.append("target_gw must be >= 1.")

    if stabilization_gw < 3:
        errors.append("stabilization_gw must be >= 3 so GW2 can be early-season blend.")

    if errors:
        resolved_mode = "invalid"
        valid = False
        metadata: Dict[str, Any] = {
            "resolved_prediction_mode": resolved_mode,
            "requires_prior_season": None,
            "requires_current_season_actuals": None,
            "prior_artifacts_required": None,
            "safe_to_run_without_prior_artifacts": None,
            "default_prior_weight": None,
            "default_current_weight": None,
        }
    else:
        if requested_prediction_mode == "auto":
            resolved_mode = auto_resolve_mode(target_gw, stabilization_gw)
        else:
            resolved_mode = requested_prediction_mode
            if not is_in_expected_window(target_gw, resolved_mode, stabilization_gw):
                message = (
                    "Forced prediction_mode=%s is outside its normal GW window (%s) for target_gw=%s."
                    % (resolved_mode, expected_mode_window(resolved_mode, stabilization_gw), target_gw)
                )
                if allow_experimental_mode:
                    warnings.append(message + " allow_experimental_mode=True, so this is permitted.")
                else:
                    errors.append(message + " Use --allow-experimental-mode to override.")

        metadata = build_mode_metadata(
            resolved_mode=resolved_mode,
            target_gw=target_gw,
            prior_season=prior_season,
            stabilization_gw=stabilization_gw,
        )

        if metadata["requires_prior_season"] and not prior_season:
            errors.append(
                "prior_season is required for resolved_prediction_mode=%s." % resolved_mode
            )

        if resolved_mode == "normal_weekly" and prior_season:
            warnings.append(
                "prior_season was supplied but will not be used in normal_weekly mode."
            )

        valid = len(errors) == 0

    return {
        "created_at": utc_now(),
        "season": season,
        "target_gw": target_gw,
        "requested_prediction_mode": requested_prediction_mode,
        "resolved_prediction_mode": metadata.get("resolved_prediction_mode"),
        "stabilization_gw": stabilization_gw,
        "prior_season": prior_season,
        "allow_experimental_mode": allow_experimental_mode,
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "metadata": metadata,
        "contract": {
            "auto_rules": {
                "GW1": "pre_gw1_prior",
                "GW2_to_before_stabilization_gw": "early_season_blend",
                "stabilization_gw_and_later": "normal_weekly",
            },
            "default_stabilization_gw": 6,
            "gw6_safe_exit_rule": (
                "With stabilization_gw=6, target_gw >= 6 must resolve to normal_weekly in auto mode "
                "and must not require prior-season artifacts."
            ),
        },
    }


def write_json(result: Dict[str, Any], out_json: str) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")


def print_summary(result: Dict[str, Any]) -> None:
    metadata = result["metadata"]

    print("=== Day67A Prediction Mode Resolver ===")
    print("season:", result["season"])
    print("target_gw:", result["target_gw"])
    print("requested_prediction_mode:", result["requested_prediction_mode"])
    print("resolved_prediction_mode:", result["resolved_prediction_mode"])
    print("valid:", result["valid"])
    print("prior_season:", result["prior_season"])
    print("stabilization_gw:", result["stabilization_gw"])
    print()
    print("Mode metadata:")
    for key in [
        "mode_window",
        "requires_prior_season",
        "requires_current_season_actuals",
        "prior_artifacts_required",
        "safe_to_run_without_prior_artifacts",
        "default_prior_weight",
        "default_current_weight",
        "prior_season_used",
        "model_name_suffix",
        "should_include_season_outlook_by_default",
        "gw6_safe_exit_applies",
    ]:
        print("- %s: %s" % (key, metadata.get(key)))

    if result["warnings"]:
        print()
        print("Warnings:")
        for warning in result["warnings"]:
            print("-", warning)

    if result["errors"]:
        print()
        print("Errors:")
        for error in result["errors"]:
            print("-", error)


def main() -> None:
    args = parse_args()

    result = resolve_prediction_mode(
        season=args.season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.prior_season,
        stabilization_gw=args.stabilization_gw,
        allow_experimental_mode=args.allow_experimental_mode,
    )

    write_json(result, args.out_json)
    print_summary(result)

    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
