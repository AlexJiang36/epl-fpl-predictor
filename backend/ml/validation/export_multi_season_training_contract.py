from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the Day66D multi-season training and adjustment contract. "
            "This is read-only and writes JSON/Markdown contract artifacts."
        )
    )
    parser.add_argument("--source-season", required=True, help="Prior/historical source season, for example 2024_25.")
    parser.add_argument("--target-season", required=True, help="Target season, for example 2025_26.")
    parser.add_argument(
        "--day66a-json",
        default="/tmp/day66_multi_season_model_training_audit.json",
        help="Day66A multi-season model training audit JSON.",
    )
    parser.add_argument(
        "--day66b-json",
        default="/tmp/player_identity_mapping_candidates_2024_25_to_2025_26.json",
        help="Day66B identity mapping audit JSON.",
    )
    parser.add_argument(
        "--day66c-json",
        default="/tmp/day66c_previous_season_prior_join_audit.json",
        help="Day66C prior join dry-run audit JSON.",
    )
    parser.add_argument("--out-json", required=True, help="Output JSON contract path.")
    parser.add_argument("--out-md", required=True, help="Output Markdown contract path.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_optional_json(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "loaded": False,
            "data": None,
            "error": "file_not_found",
        }

    try:
        return {
            "exists": True,
            "path": str(path),
            "loaded": True,
            "data": json.loads(path.read_text(encoding="utf-8")),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - defensive reporting
        return {
            "exists": True,
            "path": str(path),
            "loaded": False,
            "data": None,
            "error": str(exc),
        }


def nested_get(data: Optional[Dict[str, Any]], path: List[str], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def build_contract(
    source_season: str,
    target_season: str,
    day66a: Dict[str, Any],
    day66b: Dict[str, Any],
    day66c: Dict[str, Any],
) -> Dict[str, Any]:
    day66a_data = day66a.get("data") if day66a.get("loaded") else None
    day66b_data = day66b.get("data") if day66b.get("loaded") else None
    day66c_data = day66c.get("data") if day66c.get("loaded") else None

    day66a_passed = nested_get(day66a_data, ["passed"], None)
    day66a_ready = nested_get(day66a_data, ["multi_season_model_training_ready"], None)

    day66b_passed = nested_get(day66b_data, ["passed"], None)
    day66b_audit_only = nested_get(day66b_data, ["audit_only"], None)
    day66b_writes_database = nested_get(day66b_data, ["writes_database"], None)
    day66b_mapping_summary = nested_get(day66b_data, ["mapping_summary"], {}) or {}

    day66c_passed = nested_get(day66c_data, ["passed"], None)
    day66c_ready = nested_get(day66c_data, ["ready_for_prior_feature_export"], None)
    day66c_writes_database = nested_get(day66c_data, ["writes_database"], None)
    day66c_join_summary = nested_get(day66c_data, ["join_summary"], {}) or {}
    day66c_mapping_summary = nested_get(day66c_data, ["mapping_summary"], {}) or {}
    day66c_blockers = nested_get(day66c_data, ["blockers"], []) or []
    day66c_warnings = nested_get(day66c_data, ["warnings"], []) or []

    source_reports = {
        "day66a": {
            "path": day66a.get("path"),
            "exists": day66a.get("exists"),
            "loaded": day66a.get("loaded"),
            "passed": day66a_passed,
            "multi_season_model_training_ready": day66a_ready,
            "expected_false_ready_reason": (
                "Day66A is an audit-only readiness check. It is expected to report "
                "multi_season_model_training_ready=False until identity mapping and safe prior joins are resolved."
            ),
        },
        "day66b": {
            "path": day66b.get("path"),
            "exists": day66b.get("exists"),
            "loaded": day66b.get("loaded"),
            "passed": day66b_passed,
            "audit_only": day66b_audit_only,
            "writes_database": day66b_writes_database,
            "mapping_summary": day66b_mapping_summary,
        },
        "day66c": {
            "path": day66c.get("path"),
            "exists": day66c.get("exists"),
            "loaded": day66c.get("loaded"),
            "passed": day66c_passed,
            "ready_for_prior_feature_export": day66c_ready,
            "writes_database": day66c_writes_database,
            "join_summary": day66c_join_summary,
            "mapping_summary": day66c_mapping_summary,
            "blockers": day66c_blockers,
            "warnings": day66c_warnings,
        },
    }

    prior_join_allowed = bool(
        day66c_passed is True
        and day66c_ready is True
        and day66c_writes_database is False
        and len(day66c_blockers) == 0
        and day66c_join_summary.get("row_count_preserved") is True
        and day66c_join_summary.get("joined_feature_key_duplicate_count") == 0
    )

    accepted_mapping_statuses = ["auto_approved_candidate"]
    excluded_mapping_statuses = [
        "ambiguous_top_candidate",
        "low_confidence_top_candidate",
        "high_score_manual_review_candidate",
        "duplicate_auto_approved_manual_review",
        "unmatched",
        "alternative_candidate",
    ]

    contract = {
        "created_at": utc_now(),
        "source_season": source_season,
        "target_season": target_season,
        "contract_name": "multi_season_training_and_adjustment_contract",
        "contract_version": "day66d_v1",
        "audit_only": True,
        "writes_database": False,
        "source_reports": source_reports,
        "overall": {
            "passed": True,
            "ready_for_prior_feature_export": prior_join_allowed,
            "ready_for_full_multi_season_training": False,
            "ready_for_pre_gw1_implementation": prior_join_allowed,
            "reason_full_multi_season_training_not_ready": (
                "Only safe auto-approved prior mappings are currently accepted. "
                "Manual-review mappings are excluded, and true multi-season feature generation still requires "
                "season-aware rolling feature grouping and model-specific training/evaluation updates."
            ),
        },
        "identity_mapping_contract": {
            "accepted_mapping_statuses_for_automatic_prior_join": accepted_mapping_statuses,
            "excluded_mapping_statuses": excluded_mapping_statuses,
            "required_properties_for_accepted_mapping": [
                "candidate_rank == 1",
                "match_status == auto_approved_candidate",
                "is_auto_approved == True",
                "needs_manual_review == False",
                "safe_name_match_for_auto_approval == True",
                "candidate_player_id is not null",
                "candidate_player_id is unique among accepted mappings",
                "raw_player_id is unique among accepted mappings",
            ],
            "manual_review_policy": {
                "manual_review_rows_must_not_join_by_default": True,
                "manual_review_may_be_accepted_later_only_with_explicit_review_artifact": True,
                "duplicate_auto_approved_rows_must_be_demoted": True,
            },
        },
        "prior_feature_join_contract": {
            "join_grain": ["target_season", "canonical_player_id", "target_gw"],
            "join_key": {
                "prior_side": "raw_player_id -> accepted mapping -> candidate_player_id",
                "feature_side": "player_id",
            },
            "row_count_must_be_preserved": True,
            "duplicate_key_count_after_join_must_be_zero": True,
            "accepted_partial_coverage": True,
            "coverage_from_day66c": {
                "feature_unique_players": day66c_join_summary.get("feature_unique_players"),
                "feature_players_with_prior": day66c_join_summary.get("feature_players_with_prior"),
                "feature_players_without_prior": day66c_join_summary.get("feature_players_without_prior"),
                "feature_rows_with_prior": day66c_join_summary.get("feature_rows_with_prior"),
                "feature_rows_without_prior": day66c_join_summary.get("feature_rows_without_prior"),
                "prior_coverage_rate_players": day66c_join_summary.get("prior_coverage_rate_players"),
                "prior_coverage_rate_rows": day66c_join_summary.get("prior_coverage_rate_rows"),
            },
            "allowed_prior_columns_v1": [
                "prev_season_minutes",
                "prev_season_appearances",
                "prev_season_starts_proxy",
                "prev_season_starts_proxy_rate",
                "prev_season_total_points",
                "prev_season_points_per_appearance",
                "prev_season_points_per90",
                "prev_season_minutes_per_appearance",
                "prev_season_goals",
                "prev_season_assists",
                "prev_season_clean_sheets",
                "prev_season_bonus",
                "prev_season_latest_value",
                "prev_season_max_value",
                "prev_season_negative_points_gws",
                "prev_season_zero_minute_rows",
            ],
            "required_metadata_columns": [
                "prior_source_season",
                "prior_target_season",
                "prior_raw_player_id",
                "prior_raw_player_name",
                "prior_mapping_status",
                "prior_mapping_confidence",
            ],
        },
        "prediction_modes": {
            "pre_gw1_prior": {
                "target_gw_rule": "target_gw == 1",
                "current_season_actuals_required": False,
                "prior_season_required": True,
                "allowed_outputs": [
                    "GW1 player predictions",
                    "GW1 match predictions",
                    "opening squad",
                    "captain suggestions",
                    "champion probabilities",
                    "golden boot probabilities",
                ],
            },
            "early_season_blend": {
                "target_gw_rule": "2 <= target_gw <= 5",
                "current_season_actuals_required": True,
                "prior_season_required": True,
                "v0_default_prior_weights": {
                    "2": 0.80,
                    "3": 0.60,
                    "4": 0.40,
                    "5": 0.20,
                },
                "v1_learned_weights_required_before_replacing_v0": True,
            },
            "normal_weekly": {
                "target_gw_rule": "target_gw >= 6",
                "current_season_actuals_required": True,
                "prior_season_required": False,
                "must_work_without_prior_artifacts": True,
            },
        },
        "learned_adjustment_contract": {
            "principle": (
                "Hardcoded weights and caps are allowed for V0 fallback, but model-relevant football effects "
                "should be represented as features and learned where historical data is sufficient."
            ),
            "v0_hardcoded_blend_allowed": True,
            "v1_learned_blend_weight_target": "learned_prior_weight",
            "learned_blend_formula": (
                "final_prediction = learned_prior_weight * prior_season_prediction "
                "+ (1 - learned_prior_weight) * current_season_prediction"
            ),
            "blend_weight_features": [
                "target_gw",
                "current_season_sample_size",
                "prior_season_minutes",
                "current_season_minutes",
                "prior_season_starts_proxy",
                "current_season_starts",
                "minutes_stability",
                "team_stability",
                "transferred_team_flag",
                "new_signing_flag",
                "promoted_team_flag",
                "injury_return_flag",
                "days_since_last_minutes",
                "position",
                "price",
                "status",
                "fixture_difficulty",
            ],
            "do_not_hardcode_as_permanent_rules": [
                "three straight goals means next game goal probability must decrease",
                "one high FPL score means next match should be boosted by a fixed amount",
                "one poor score means next match should be penalized by a fixed amount",
            ],
        },
        "player_prediction_architecture_contract": {
            "long_term_architecture": [
                "features -> minutes model -> appearance_probability, start_probability, expected_minutes",
                "features -> event models -> expected_goals, expected_assists, clean_sheet_probability, bonus_expectation",
                "combine -> raw_expected_points",
                "guardrails / calibration -> final_predicted_points",
            ],
            "layer_1_feature_builder": [
                "prior/current/recent/status/fixture/risk features",
                "sample-size features",
                "default blend policy features",
            ],
            "layer_2_predictive_models": [
                "learn minutes, starts, events, points, and blend weights",
            ],
            "layer_3_guardrails": [
                "only hard constraints and extreme safety caps",
            ],
            "layer_4_calibration_manifest": [
                "record raw prediction, final prediction, adjustments, calibration, and safe_for_frontend",
            ],
        },
        "guardrail_contract": {
            "hard_guardrails": [
                "no fixture -> predicted_points = 0",
                "blank GW -> predicted_points = 0",
                "unavailable or suspended -> appearance_probability near 0 and predicted_points near 0",
                "player not in target-season player table -> block prediction",
                "missing required identity mapping for prior-driven mode -> block prior join for that player",
            ],
            "soft_constraints_as_features_first": [
                "low recent minutes",
                "bench role",
                "minutes drop",
                "injury return",
                "long absence",
                "status changed from injured to available",
                "consecutive starts",
                "consecutive blanks",
                "consecutive attacking returns",
                "hot streak",
                "possible mean reversion",
            ],
            "soft_caps_allowed_in_v0_if_manifested": [
                "return_from_absence expected_minutes cap",
                "very_low_recent_minutes start_probability cap",
                "uncertain_status risk discount",
            ],
        },
        "calibration_contract": {
            "calibration_required_before_production_claims": True,
            "must_record": [
                "raw_expected_points",
                "final_predicted_points",
                "guardrail_adjustment_applied",
                "calibration_applied",
                "calibration_version",
                "safe_for_frontend",
            ],
            "recommended_checks": [
                "prediction bins calibration",
                "high-score shrinkage if overconfident",
                "minutes model calibration",
                "position-level MAE",
                "GW-level MAE",
            ],
        },
        "training_contract": {
            "single_season_export_rule": (
                "Current export_features_v2_1 can be used one season at a time. "
                "Do not pass a multi-season dataframe into rolling feature code unless groupby includes season."
            ),
            "multi_season_rolling_rule": [
                "player rolling features must group by season + player_id",
                "team rolling features must group by season + team_id",
                "match team context must group by season + team_id",
            ],
            "csv_trainer_rule": (
                "CSV-based v2_1 trainers may train on multi-season data only if the CSV artifact is already safely constructed."
            ),
            "evaluation_rule": [
                "train_seasons must be explicit",
                "target/evaluation season must be explicit",
                "metrics must be season-aware",
                "no leakage from future GWs or future seasons",
            ],
        },
        "model_naming_contract": {
            "pre_gw1_player": "player_prior_v0_pre_gw1",
            "early_season_player": "player_blend_v0_early",
            "learned_blend_player": "player_blend_weight_v1_early",
            "normal_player": "ridge_rollform_v1_normal",
            "pre_gw1_match": "match_team_prior_v0_pre_gw1",
            "early_season_match": "match_blend_v0_early",
            "normal_match": "match_logreg_v2_normal",
            "rule": "Never overwrite normal model artifacts with pre_gw1 or early-season artifacts.",
        },
        "frontend_contract": {
            "existing_pages_to_reuse": [
                "/predictions",
                "/match-predictions",
                "/transfers/targets",
                "/model-squad",
                "/model-evaluation",
                "/ops",
            ],
            "new_pages_for_new_outputs": [
                "/season-outlook",
                "/season-outlook/champion",
                "/season-outlook/golden-boot",
            ],
            "required_badges": [
                "prediction_mode",
                "prior_season",
                "uncertainty_warning",
                "guardrail_adjusted",
                "calibrated",
                "refresh_timestamp",
            ],
            "frontend_must_not_choose_production_model": True,
            "frontend_should_read_active_model_from_backend_or_manifest": True,
        },
        "refresh_contract": {
            "one_command_refresh_required": True,
            "prediction_mode_auto_rules": {
                "GW1": "pre_gw1_prior",
                "GW2-GW5": "early_season_blend",
                "GW6+": "normal_weekly",
            },
            "manifest_required": True,
            "manifest_fields": [
                "season",
                "prior_season",
                "target_gw",
                "requested_prediction_mode",
                "resolved_prediction_mode",
                "started_at",
                "finished_at",
                "status",
                "steps_run",
                "steps_skipped",
                "player_prediction_model",
                "match_prediction_model",
                "season_outlook_included",
                "row_counts",
                "artifact_paths",
                "warnings",
                "errors",
                "safe_for_frontend",
                "raw_prediction_available",
                "guardrail_adjustment_applied",
                "calibration_applied",
                "blend_weight_source",
                "minutes_model_used",
                "event_models_used",
            ],
        },
        "gw6_safe_exit_contract": {
            "hard_requirement": (
                "From GW6 onward, normal weekly refresh must not depend on historical/prior artifacts."
            ),
            "target_gw_auto_rule": "if target_gw >= 6 and prediction_mode=auto, resolved mode must be normal_weekly",
            "prior_artifacts_required_after_gw6": False,
            "must_not_require_after_gw6": [
                "historical staging tables",
                "Day65 player prior artifact",
                "player identity mapping candidates",
                "prior-feature join artifacts",
                "preseason team priors",
                "season outcome simulations",
                "learned early-season blend model",
            ],
            "frontend_after_gw6": [
                "default pages show Normal Weekly Mode",
                "pre_gw1 and early models are archived/advanced only",
                "preseason uncertainty banner hidden",
                "ops page shows prior artifacts not required",
            ],
        },
        "day67_readiness": {
            "next_recommended_step": "Day67A or Day67B depending on chosen split",
            "safe_to_start_pre_gw1_readiness_checker": prior_join_allowed,
            "safe_to_start_actual_pre_gw1_prediction": False,
            "reason_actual_prediction_not_next": (
                "A readiness checker, team priors, and mode resolver should be added before actual prediction writes."
            ),
        },
    }

    return contract


def write_json(contract: Dict[str, Any], out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, sort_keys=True, default=str), encoding="utf-8")


def md_bool(value: Any) -> str:
    if value is True:
        return "`True`"
    if value is False:
        return "`False`"
    if value is None:
        return "`None`"
    return "`%s`" % value


def bullet_list(items: List[Any]) -> str:
    return "\n".join("- %s" % item for item in items)


def write_markdown(contract: Dict[str, Any], out_path: str) -> None:
    overall = contract["overall"]
    day66b = contract["source_reports"]["day66b"]
    day66c = contract["source_reports"]["day66c"]
    join = contract["prior_feature_join_contract"]["coverage_from_day66c"]

    lines: List[str] = []
    lines.append("# Day66D — Multi-Season Training and Adjustment Contract")
    lines.append("")
    lines.append("Created at: `%s`" % contract["created_at"])
    lines.append("")
    lines.append("Source season: `%s`" % contract["source_season"])
    lines.append("Target season: `%s`" % contract["target_season"])
    lines.append("Contract version: `%s`" % contract["contract_version"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Audit only: %s" % md_bool(contract["audit_only"]))
    lines.append("- Writes database: %s" % md_bool(contract["writes_database"]))
    lines.append("- Ready for prior feature export: %s" % md_bool(overall["ready_for_prior_feature_export"]))
    lines.append("- Ready for full multi-season training: %s" % md_bool(overall["ready_for_full_multi_season_training"]))
    lines.append("- Ready for Pre-GW1 implementation scaffolding: %s" % md_bool(overall["ready_for_pre_gw1_implementation"]))
    lines.append("")
    lines.append("Full multi-season training is not considered ready because only safe auto-approved mappings are accepted and true multi-season rolling feature generation still requires season-aware grouping.")
    lines.append("")
    lines.append("## Day66 Evidence")
    lines.append("")
    lines.append("### Day66B Identity Mapping")
    lines.append("")
    lines.append("- Report path: `%s`" % day66b["path"])
    lines.append("- Passed: %s" % md_bool(day66b["passed"]))
    lines.append("- Writes database: %s" % md_bool(day66b["writes_database"]))
    lines.append("- Mapping summary: `%s`" % day66b["mapping_summary"])
    lines.append("")
    lines.append("### Day66C Prior Join Dry-Run")
    lines.append("")
    lines.append("- Report path: `%s`" % day66c["path"])
    lines.append("- Passed: %s" % md_bool(day66c["passed"]))
    lines.append("- Ready for prior feature export: %s" % md_bool(day66c["ready_for_prior_feature_export"]))
    lines.append("- Blockers: `%s`" % day66c["blockers"])
    lines.append("- Warnings: `%s`" % day66c["warnings"])
    lines.append("")
    lines.append("Coverage from Day66C:")
    lines.append("")
    lines.append("- Feature unique players: `%s`" % join.get("feature_unique_players"))
    lines.append("- Feature players with prior: `%s`" % join.get("feature_players_with_prior"))
    lines.append("- Feature players without prior: `%s`" % join.get("feature_players_without_prior"))
    lines.append("- Feature rows with prior: `%s`" % join.get("feature_rows_with_prior"))
    lines.append("- Feature rows without prior: `%s`" % join.get("feature_rows_without_prior"))
    lines.append("- Prior coverage rate by players: `%s`" % join.get("prior_coverage_rate_players"))
    lines.append("- Prior coverage rate by rows: `%s`" % join.get("prior_coverage_rate_rows"))
    lines.append("")
    lines.append("## Identity Mapping Contract")
    lines.append("")
    lines.append("Accepted automatic mapping statuses:")
    lines.append("")
    lines.append(bullet_list(contract["identity_mapping_contract"]["accepted_mapping_statuses_for_automatic_prior_join"]))
    lines.append("")
    lines.append("Excluded mapping statuses:")
    lines.append("")
    lines.append(bullet_list(contract["identity_mapping_contract"]["excluded_mapping_statuses"]))
    lines.append("")
    lines.append("Required properties for accepted mappings:")
    lines.append("")
    lines.append(bullet_list(contract["identity_mapping_contract"]["required_properties_for_accepted_mapping"]))
    lines.append("")
    lines.append("## Prior Feature Join Contract")
    lines.append("")
    lines.append("- Join grain: `%s`" % contract["prior_feature_join_contract"]["join_grain"])
    lines.append("- Row count must be preserved: %s" % md_bool(contract["prior_feature_join_contract"]["row_count_must_be_preserved"]))
    lines.append("- Duplicate key count after join must be zero: %s" % md_bool(contract["prior_feature_join_contract"]["duplicate_key_count_after_join_must_be_zero"]))
    lines.append("- Partial coverage accepted: %s" % md_bool(contract["prior_feature_join_contract"]["accepted_partial_coverage"]))
    lines.append("")
    lines.append("Allowed prior columns v1:")
    lines.append("")
    lines.append(bullet_list(contract["prior_feature_join_contract"]["allowed_prior_columns_v1"]))
    lines.append("")
    lines.append("## Prediction Modes")
    lines.append("")
    for mode_name, mode in contract["prediction_modes"].items():
        lines.append("### `%s`" % mode_name)
        lines.append("")
        for key, value in mode.items():
            lines.append("- %s: `%s`" % (key, value))
        lines.append("")
    lines.append("## Learned Adjustment Contract")
    lines.append("")
    learned = contract["learned_adjustment_contract"]
    lines.append(learned["principle"])
    lines.append("")
    lines.append("- V0 hardcoded blend allowed: %s" % md_bool(learned["v0_hardcoded_blend_allowed"]))
    lines.append("- V1 learned blend target: `%s`" % learned["v1_learned_blend_weight_target"])
    lines.append("- Formula: `%s`" % learned["learned_blend_formula"])
    lines.append("")
    lines.append("Blend weight features:")
    lines.append("")
    lines.append(bullet_list(learned["blend_weight_features"]))
    lines.append("")
    lines.append("Do not hardcode as permanent rules:")
    lines.append("")
    lines.append(bullet_list(learned["do_not_hardcode_as_permanent_rules"]))
    lines.append("")
    lines.append("## Player Prediction Architecture")
    lines.append("")
    lines.append("Long-term architecture:")
    lines.append("")
    lines.append("```text")
    lines.append("features")
    lines.append("  ↓")
    lines.append("minutes model")
    lines.append("  → appearance_probability")
    lines.append("  → start_probability")
    lines.append("  → expected_minutes")
    lines.append("")
    lines.append("features")
    lines.append("  ↓")
    lines.append("event models")
    lines.append("  → expected_goals")
    lines.append("  → expected_assists")
    lines.append("  → clean_sheet_probability")
    lines.append("  → bonus_expectation")
    lines.append("")
    lines.append("combine")
    lines.append("  ↓")
    lines.append("raw_expected_points")
    lines.append("")
    lines.append("guardrails / calibration")
    lines.append("  ↓")
    lines.append("final_predicted_points")
    lines.append("```")
    lines.append("")
    lines.append("Four-layer system:")
    lines.append("")
    lines.append("- Layer 1: Feature Builder — prior/current/recent/status/fixture/risk features.")
    lines.append("- Layer 2: Predictive Models — learn minutes, starts, events, points, and blend weights.")
    lines.append("- Layer 3: Guardrails — only hard constraints and extreme safety caps.")
    lines.append("- Layer 4: Calibration + Manifest — record raw/final predictions, adjustments, calibration, and safe_for_frontend.")
    lines.append("")
    lines.append("## Guardrails vs Learned Adjustments")
    lines.append("")
    lines.append("Hard guardrails:")
    lines.append("")
    lines.append(bullet_list(contract["guardrail_contract"]["hard_guardrails"]))
    lines.append("")
    lines.append("Soft constraints should become features first:")
    lines.append("")
    lines.append(bullet_list(contract["guardrail_contract"]["soft_constraints_as_features_first"]))
    lines.append("")
    lines.append("## Training Contract")
    lines.append("")
    lines.append("- Single-season export rule: %s" % contract["training_contract"]["single_season_export_rule"])
    lines.append("- CSV trainer rule: %s" % contract["training_contract"]["csv_trainer_rule"])
    lines.append("")
    lines.append("Multi-season rolling rules:")
    lines.append("")
    lines.append(bullet_list(contract["training_contract"]["multi_season_rolling_rule"]))
    lines.append("")
    lines.append("## Frontend Contract")
    lines.append("")
    lines.append("Existing pages to reuse:")
    lines.append("")
    lines.append(bullet_list(contract["frontend_contract"]["existing_pages_to_reuse"]))
    lines.append("")
    lines.append("New pages:")
    lines.append("")
    lines.append(bullet_list(contract["frontend_contract"]["new_pages_for_new_outputs"]))
    lines.append("")
    lines.append("Required badges:")
    lines.append("")
    lines.append(bullet_list(contract["frontend_contract"]["required_badges"]))
    lines.append("")
    lines.append("## GW6 Safe Exit Contract")
    lines.append("")
    safe_exit = contract["gw6_safe_exit_contract"]
    lines.append(safe_exit["hard_requirement"])
    lines.append("")
    lines.append("- Auto rule: `%s`" % safe_exit["target_gw_auto_rule"])
    lines.append("- Prior artifacts required after GW6: %s" % md_bool(safe_exit["prior_artifacts_required_after_gw6"]))
    lines.append("")
    lines.append("Must not require after GW6:")
    lines.append("")
    lines.append(bullet_list(safe_exit["must_not_require_after_gw6"]))
    lines.append("")
    lines.append("Frontend after GW6:")
    lines.append("")
    lines.append(bullet_list(safe_exit["frontend_after_gw6"]))
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append("- Next recommended step: `%s`" % contract["day67_readiness"]["next_recommended_step"])
    lines.append("- Safe to start Pre-GW1 readiness checker: %s" % md_bool(contract["day67_readiness"]["safe_to_start_pre_gw1_readiness_checker"]))
    lines.append("- Safe to start actual Pre-GW1 prediction: %s" % md_bool(contract["day67_readiness"]["safe_to_start_actual_pre_gw1_prediction"]))
    lines.append("")
    lines.append(contract["day67_readiness"]["reason_actual_prediction_not_next"])
    lines.append("")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(contract: Dict[str, Any], out_json: str, out_md: str) -> None:
    print("=== Day66D Multi-Season Training and Adjustment Contract ===")
    print("source_season:", contract["source_season"])
    print("target_season:", contract["target_season"])
    print("contract_version:", contract["contract_version"])
    print("audit_only:", contract["audit_only"])
    print("writes_database:", contract["writes_database"])
    print("ready_for_prior_feature_export:", contract["overall"]["ready_for_prior_feature_export"])
    print("ready_for_full_multi_season_training:", contract["overall"]["ready_for_full_multi_season_training"])
    print("ready_for_pre_gw1_implementation:", contract["overall"]["ready_for_pre_gw1_implementation"])
    print("safe_to_start_pre_gw1_readiness_checker:", contract["day67_readiness"]["safe_to_start_pre_gw1_readiness_checker"])
    print("safe_to_start_actual_pre_gw1_prediction:", contract["day67_readiness"]["safe_to_start_actual_pre_gw1_prediction"])
    print("saved_json:", out_json)
    print("saved_md:", out_md)
    print()
    print("Day66B mapping summary:", contract["source_reports"]["day66b"]["mapping_summary"])
    print("Day66C join summary:", contract["source_reports"]["day66c"]["join_summary"])
    print()
    print("Key rules:")
    print("- Only auto_approved_candidate mappings may join automatically.")
    print("- Manual-review, ambiguous, duplicate-demoted, and unmatched mappings are excluded.")
    print("- Prior join must preserve row count and season/player/GW grain.")
    print("- V0 hardcoded early-season weights are allowed, but V1 should learn weights.")
    print("- Guardrails handle hard constraints only; soft football effects should become features.")
    print("- GW6+ normal refresh must not require prior artifacts.")


def main() -> None:
    args = parse_args()

    day66a = load_optional_json(args.day66a_json)
    day66b = load_optional_json(args.day66b_json)
    day66c = load_optional_json(args.day66c_json)

    contract = build_contract(
        source_season=args.source_season,
        target_season=args.target_season,
        day66a=day66a,
        day66b=day66b,
        day66c=day66c,
    )

    write_json(contract, args.out_json)
    write_markdown(contract, args.out_md)
    print_summary(contract, args.out_json, args.out_md)


if __name__ == "__main__":
    main()
