from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.rules.squad import (
    SquadTransferRules,
    SquadTransferRulesError,
    load_squad_transfer_rules,
    normalize_player,
    normalize_players,
    require_bool,
    require_int,
    require_list,
    require_mapping,
    require_nonempty_string,
    require_nonnegative_int,
    validate_lineup_examples,
    validate_squad,
    validate_squad_examples,
)


VALID_PHASES = ("pre_season", "in_season")
VALID_TRANSFER_CHIPS = ("wildcard", "free_hit")
VALIDATION_VERSION = "day74b_v1"
APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_SOURCES = (
    APP_ROOT / "utils" / "wildcard_builder.py",
    APP_ROOT / "api" / "routes" / "chips.py",
    APP_ROOT / "api" / "routes" / "recommendations.py",
)


class TransferValidationError(SquadTransferRulesError):
    """Raised when a transfer-state input is structurally invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_phase(phase: str) -> str:
    normalized = require_nonempty_string(phase, "phase")
    if normalized not in VALID_PHASES:
        raise TransferValidationError(
            "phase=%s is invalid; expected one of %s."
            % (normalized, VALID_PHASES)
        )
    return normalized


def normalize_chip(chip: Optional[str]) -> Optional[str]:
    if chip is None or str(chip).strip() == "":
        return None
    normalized = str(chip).strip()
    if normalized not in VALID_TRANSFER_CHIPS:
        raise TransferValidationError(
            "chip=%s is invalid; expected one of %s or null."
            % (normalized, VALID_TRANSFER_CHIPS)
        )
    return normalized


def maximum_stored_free_transfers(rules: SquadTransferRules) -> int:
    return int(rules.transfers["weekly"]["maximum_stored_free_transfers"])


def validate_available_free_transfers(
    rules: SquadTransferRules,
    available_free_transfers: int,
) -> int:
    available = require_nonnegative_int(
        available_free_transfers, "available_free_transfers"
    )
    cap = maximum_stored_free_transfers(rules)
    if available > cap:
        raise TransferValidationError(
            "available_free_transfers=%s exceeds storage cap=%s."
            % (available, cap)
        )
    return available


def price_transfer_plan(
    rules: SquadTransferRules,
    transfer_count: int,
    available_free_transfers: int,
    phase: str = "in_season",
    chip: Optional[str] = None,
) -> Dict[str, Any]:
    """Price any number of transfers without assuming one transfer per GW."""

    count = require_nonnegative_int(transfer_count, "transfer_count")
    available = validate_available_free_transfers(
        rules, available_free_transfers
    )
    normalized_phase = normalize_phase(phase)
    normalized_chip = normalize_chip(chip)

    if normalized_phase == "pre_season":
        if normalized_chip is not None:
            raise TransferValidationError(
                "Transfer chips are not applicable in pre_season phase."
            )
        pre_season = rules.transfers["pre_season"]
        if not bool(pre_season["unlimited_free_transfers"]):
            raise TransferValidationError(
                "Pre-season policy does not authorize unlimited free transfers."
            )
        return {
            "phase": normalized_phase,
            "chip": None,
            "transfer_count": count,
            "available_free_transfers_before": available,
            "free_transfers_used": 0,
            "charged_transfers": 0,
            "points_cost": int(pre_season["hit_cost_points"]),
            "free_transfers_after_transfers": available,
            "unlimited_free_transfers": True,
        }

    if normalized_chip is not None:
        chip_policy = rules.transfers["chip_behavior"][normalized_chip]
        if not bool(chip_policy["unlimited_free_transfers"]):
            raise TransferValidationError(
                "%s policy does not authorize unlimited transfers."
                % normalized_chip
            )
        points_cost = 0 if bool(chip_policy["remove_transfer_hits"]) else None
        if points_cost is None:
            raise TransferValidationError(
                "%s policy must define transfer-hit removal." % normalized_chip
            )
        preserved = (
            available if bool(chip_policy["preserve_saved_free_transfers"]) else 0
        )
        return {
            "phase": normalized_phase,
            "chip": normalized_chip,
            "transfer_count": count,
            "available_free_transfers_before": available,
            "free_transfers_used": 0,
            "charged_transfers": 0,
            "points_cost": points_cost,
            "free_transfers_after_transfers": preserved,
            "unlimited_free_transfers": True,
        }

    weekly = rules.transfers["weekly"]
    free_used = min(count, available)
    charged = max(0, count - free_used)
    points_cost = charged * int(
        weekly["hit_cost_points_per_additional_transfer"]
    )
    return {
        "phase": normalized_phase,
        "chip": None,
        "transfer_count": count,
        "available_free_transfers_before": available,
        "free_transfers_used": free_used,
        "charged_transfers": charged,
        "points_cost": points_cost,
        "free_transfers_after_transfers": available - free_used,
        "unlimited_free_transfers": False,
    }


def special_event_after_gameweek(
    rules: SquadTransferRules,
    completed_gameweek: int,
) -> Optional[Mapping[str, Any]]:
    completed = require_nonnegative_int(completed_gameweek, "completed_gameweek")
    matches = [
        event
        for event in rules.transfers["special_events"]
        if int(event["after_gameweek"]) == completed
    ]
    if len(matches) > 1:
        raise TransferValidationError(
            "Multiple transfer special events apply after gameweek=%s." % completed
        )
    return matches[0] if matches else None


def advance_free_transfer_bank(
    rules: SquadTransferRules,
    free_transfers_after_transfers: int,
    completed_gameweek: int,
    phase: str = "in_season",
    chip: Optional[str] = None,
    free_transfers_before_transfers: Optional[int] = None,
) -> Dict[str, Any]:
    """Advance the free-transfer state to the next GW deadline window."""

    remaining = validate_available_free_transfers(
        rules, free_transfers_after_transfers
    )
    completed = require_nonnegative_int(completed_gameweek, "completed_gameweek")
    normalized_phase = normalize_phase(phase)
    normalized_chip = normalize_chip(chip)
    cap = maximum_stored_free_transfers(rules)

    if normalized_phase == "pre_season":
        if normalized_chip is not None:
            raise TransferValidationError(
                "Transfer chips are not applicable in pre_season phase."
            )
        base_next = min(
            cap,
            int(rules.transfers["weekly"]["free_transfers_accrued_per_gameweek"]),
        )
        accrual_applied = base_next
    elif normalized_chip is not None:
        chip_policy = rules.transfers["chip_behavior"][normalized_chip]
        if free_transfers_before_transfers is None:
            raise TransferValidationError(
                "free_transfers_before_transfers is required when a transfer chip is active."
            )
        before = validate_available_free_transfers(
            rules, free_transfers_before_transfers
        )
        if bool(chip_policy["preserve_saved_free_transfers"]):
            base_next = before
        else:
            base_next = remaining
        accrual_applied = int(chip_policy["weekly_accrual_after_chip"])
        base_next = min(cap, base_next + accrual_applied)
    else:
        accrual_applied = int(
            rules.transfers["weekly"]["free_transfers_accrued_per_gameweek"]
        )
        base_next = min(cap, remaining + accrual_applied)

    event = special_event_after_gameweek(rules, completed)
    event_applied: Optional[str] = None
    final_next = base_next
    if event is not None:
        if event["operation"] != "top_up_to":
            raise TransferValidationError(
                "Unsupported transfer special-event operation=%s."
                % event["operation"]
            )
        final_next = min(
            cap,
            max(base_next, int(event["target_free_transfers"])),
        )
        event_applied = str(event["event_id"])

    return {
        "completed_gameweek": completed,
        "next_gameweek": completed + 1,
        "phase": normalized_phase,
        "chip": normalized_chip,
        "free_transfers_after_transfers": remaining,
        "weekly_accrual_applied": accrual_applied,
        "special_event_applied": event_applied,
        "free_transfers_next_gameweek": final_next,
        "storage_cap": cap,
    }


def evaluate_transfer_step(
    rules: SquadTransferRules,
    transfer_count: int,
    available_free_transfers: int,
    completed_gameweek: int,
    phase: str = "in_season",
    chip: Optional[str] = None,
) -> Dict[str, Any]:
    priced = price_transfer_plan(
        rules=rules,
        transfer_count=transfer_count,
        available_free_transfers=available_free_transfers,
        phase=phase,
        chip=chip,
    )
    advanced = advance_free_transfer_bank(
        rules=rules,
        free_transfers_after_transfers=int(
            priced["free_transfers_after_transfers"]
        ),
        completed_gameweek=completed_gameweek,
        phase=phase,
        chip=chip,
        free_transfers_before_transfers=available_free_transfers,
    )
    result = dict(priced)
    result.update(
        {
            "completed_gameweek": int(completed_gameweek),
            "next_gameweek": advanced["next_gameweek"],
            "weekly_accrual_applied": advanced["weekly_accrual_applied"],
            "special_event_applied": advanced["special_event_applied"],
            "free_transfers_next_gameweek": advanced[
                "free_transfers_next_gameweek"
            ],
            "storage_cap": advanced["storage_cap"],
        }
    )
    return result


def validate_transfer_legality(
    rules: SquadTransferRules,
    current_squad: Sequence[Mapping[str, Any]],
    outgoing_player_ids: Sequence[Any],
    incoming_players: Sequence[Mapping[str, Any]],
    budget_limit_units: int,
    available_free_transfers: int,
    completed_gameweek: int,
    phase: str = "in_season",
    chip: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a complete multi-transfer operation and its resulting squad."""

    errors: List[str] = []
    current = normalize_players(current_squad)
    incoming = normalize_players(incoming_players)
    outgoing_ids = list(outgoing_player_ids)

    current_validation = validate_squad(
        rules, current, budget_limit_units=budget_limit_units
    )
    if not current_validation["valid"]:
        errors.append("current_squad_invalid")

    if len(set(outgoing_ids)) != len(outgoing_ids):
        errors.append("duplicate_outgoing_player_ids")
    current_by_id = {player["player_id"]: player for player in current}
    missing_outgoing = [player_id for player_id in outgoing_ids if player_id not in current_by_id]
    if missing_outgoing:
        errors.append("outgoing_players_not_in_current_squad: %s" % missing_outgoing)

    incoming_ids = [player["player_id"] for player in incoming]
    if len(set(incoming_ids)) != len(incoming_ids):
        errors.append("duplicate_incoming_player_ids")

    if bool(
        rules.transfers["weekly"]["incoming_count_must_equal_outgoing_count"]
    ) and len(incoming) != len(outgoing_ids):
        errors.append(
            "incoming_outgoing_count_mismatch: incoming=%s outgoing=%s"
            % (len(incoming), len(outgoing_ids))
        )

    remaining = [
        player for player in current if player["player_id"] not in set(outgoing_ids)
    ]
    remaining_ids = {player["player_id"] for player in remaining}
    already_owned = [player_id for player_id in incoming_ids if player_id in remaining_ids]
    if already_owned:
        errors.append("incoming_players_already_owned: %s" % already_owned)

    final_squad = remaining + incoming
    final_validation = validate_squad(
        rules, final_squad, budget_limit_units=budget_limit_units
    )
    if not final_validation["valid"]:
        errors.append("resulting_squad_invalid")

    transfer_count = max(len(outgoing_ids), len(incoming))
    economics: Optional[Dict[str, Any]] = None
    try:
        economics = evaluate_transfer_step(
            rules=rules,
            transfer_count=transfer_count,
            available_free_transfers=available_free_transfers,
            completed_gameweek=completed_gameweek,
            phase=phase,
            chip=chip,
        )
    except TransferValidationError as exc:
        errors.append("transfer_state_invalid: %s" % exc)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "transfer_count": transfer_count,
        "outgoing_player_ids": outgoing_ids,
        "incoming_player_ids": incoming_ids,
        "current_squad_validation": current_validation,
        "resulting_squad_validation": final_validation,
        "economics": economics,
    }


def validate_transfer_examples(rules: SquadTransferRules) -> Dict[str, Any]:
    raw_cases = rules.data["deterministic_examples"]["transfer_cases"]
    results: List[Dict[str, Any]] = []

    for raw_case in raw_cases:
        case = require_mapping(raw_case, "transfer deterministic case")
        name = require_nonempty_string(case.get("name"), "transfer case name")
        expected_valid = require_bool(
            case.get("expected_valid"), "%s.expected_valid" % name
        )
        current_bank = case.get("initial_free_transfers")
        steps = require_list(case.get("steps"), "%s.steps" % name)
        step_results: List[Dict[str, Any]] = []
        actual_valid = True
        error: Optional[str] = None

        try:
            current_bank = validate_available_free_transfers(
                rules, require_int(current_bank, "%s.initial_free_transfers" % name)
            )
            for index, raw_step in enumerate(steps):
                step = require_mapping(raw_step, "%s.steps[%s]" % (name, index))
                result = evaluate_transfer_step(
                    rules=rules,
                    transfer_count=require_int(
                        step.get("transfer_count"),
                        "%s.steps[%s].transfer_count" % (name, index),
                    ),
                    available_free_transfers=current_bank,
                    completed_gameweek=require_int(
                        step.get("completed_gameweek"),
                        "%s.steps[%s].completed_gameweek" % (name, index),
                    ),
                    phase=str(step.get("phase", "in_season")),
                    chip=step.get("chip"),
                )
                expected = require_mapping(
                    step.get("expected"),
                    "%s.steps[%s].expected" % (name, index),
                )
                mismatches: List[str] = []
                for key, expected_value in expected.items():
                    if result.get(key) != expected_value:
                        mismatches.append(
                            "%s expected=%r actual=%r"
                            % (key, expected_value, result.get(key))
                        )
                result["expectation_mismatches"] = mismatches
                result["step_passed"] = len(mismatches) == 0
                if mismatches:
                    actual_valid = False
                step_results.append(result)
                current_bank = int(result["free_transfers_next_gameweek"])
        except (SquadTransferRulesError, TransferValidationError) as exc:
            actual_valid = False
            error = str(exc)

        passed = actual_valid == expected_valid
        if expected_valid and any(
            not bool(step.get("step_passed")) for step in step_results
        ):
            passed = False
        results.append(
            {
                "name": name,
                "expected_valid": expected_valid,
                "actual_valid": actual_valid,
                "passed": passed,
                "error": error,
                "steps": step_results,
            }
        )

    return {
        "case_count": len(results),
        "passed_count": sum(1 for result in results if result["passed"]),
        "all_passed": all(result["passed"] for result in results),
        "cases": results,
    }


def validate_transfer_legality_examples(rules: SquadTransferRules) -> Dict[str, Any]:
    examples = rules.data["deterministic_examples"]
    base_squad = examples["base_squad"]
    raw_cases = examples["transfer_legality_cases"]
    results: List[Dict[str, Any]] = []

    for raw_case in raw_cases:
        case = require_mapping(raw_case, "transfer legality deterministic case")
        name = require_nonempty_string(case.get("name"), "transfer legality case name")
        expected_valid = require_bool(
            case.get("expected_valid"), "%s.expected_valid" % name
        )
        result = validate_transfer_legality(
            rules=rules,
            current_squad=base_squad,
            outgoing_player_ids=require_list(
                case.get("outgoing_player_ids"), "%s.outgoing_player_ids" % name
            ),
            incoming_players=require_list(
                case.get("incoming_players"), "%s.incoming_players" % name
            ),
            budget_limit_units=require_int(
                case.get("budget_limit_units"), "%s.budget_limit_units" % name
            ),
            available_free_transfers=require_int(
                case.get("available_free_transfers"),
                "%s.available_free_transfers" % name,
            ),
            completed_gameweek=require_int(
                case.get("completed_gameweek"), "%s.completed_gameweek" % name
            ),
            phase=str(case.get("phase", "in_season")),
            chip=case.get("chip"),
        )
        passed = bool(result["valid"]) == expected_valid
        if "expected_points_cost" in case and result.get("economics") is not None:
            passed = passed and (
                int(result["economics"]["points_cost"])
                == int(case["expected_points_cost"])
            )
        results.append(
            {
                "name": name,
                "expected_valid": expected_valid,
                "actual_valid": bool(result["valid"]),
                "passed": passed,
                "errors": result["errors"],
                "points_cost": (
                    result["economics"].get("points_cost")
                    if result.get("economics")
                    else None
                ),
            }
        )

    return {
        "case_count": len(results),
        "passed_count": sum(1 for result in results if result["passed"]),
        "all_passed": all(result["passed"] for result in results),
        "cases": results,
    }


LEGACY_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("squad_position_constant", r"\bSQUAD_RULES\s*="),
    ("fixed_starting_formation_constant", r"\bSTARTING_(?:RULES|FORMATION)\s*="),
    ("hardcoded_squad_size_15", r"len\([^\n]+\)\s*!=\s*15"),
    ("hardcoded_starting_size_11", r"len\([^\n]+\)\s*!=\s*11"),
    ("hardcoded_club_limit_3", r"(?:max 3 players per club|>\s*3)"),
    ("hardcoded_budget_100", r"(?:default\s*=\s*100\.0|budget[^\n]*100\.0)"),
    ("single_free_transfer_default", r"free_transfers[^\n]*default\s*=\s*1"),
    ("hardcoded_hit_cost_4", r"transfer_cost_points[^\n]*\b4\b"),
)


def scan_legacy_source(path: Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
            "uses_registry": False,
            "findings": [],
            "hardcoded_rule_count": 0,
        }
    text = resolved.read_text(encoding="utf-8")
    findings = [
        label
        for label, pattern in LEGACY_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    ]
    uses_registry = bool(
        re.search(
            r"(?:app\.rules\.squad|load_squad_transfer_rules|SquadTransferRules)",
            text,
        )
    )
    return {
        "path": str(resolved),
        "exists": True,
        "uses_registry": uses_registry,
        "findings": findings,
        "hardcoded_rule_count": len(findings),
    }


def scan_legacy_sources(paths: Sequence[Path]) -> Dict[str, Any]:
    sources = [scan_legacy_source(path) for path in paths]
    existing = [source for source in sources if source["exists"]]
    return {
        "source_count": len(sources),
        "existing_source_count": len(existing),
        "registry_integrated_source_count": sum(
            1 for source in existing if source["uses_registry"]
        ),
        "sources_with_hardcoded_rules": sum(
            1 for source in existing if source["hardcoded_rule_count"] > 0
        ),
        "hardcoded_rule_finding_count": sum(
            int(source["hardcoded_rule_count"]) for source in existing
        ),
        "all_existing_sources_use_registry": bool(existing)
        and all(source["uses_registry"] for source in existing),
        "sources": sources,
    }


def build_validation_report(
    rules: SquadTransferRules,
    legacy_sources: Sequence[Path],
) -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []

    squad_examples = validate_squad_examples(rules)
    lineup_examples = validate_lineup_examples(rules)
    transfer_examples = validate_transfer_examples(rules)
    transfer_legality_examples = validate_transfer_legality_examples(rules)
    legacy_scan = scan_legacy_sources(legacy_sources)

    if not squad_examples["all_passed"]:
        blockers.append("One or more deterministic squad examples failed.")
    if not lineup_examples["all_passed"]:
        blockers.append("One or more deterministic lineup examples failed.")
    if not transfer_examples["all_passed"]:
        blockers.append("One or more deterministic transfer examples failed.")
    if not transfer_legality_examples["all_passed"]:
        blockers.append("One or more deterministic transfer-legality examples failed.")

    if legacy_scan["sources_with_hardcoded_rules"] > 0:
        warnings.append(
            "Existing squad/transfer callers still contain hardcoded rules; Day74B "
            "creates the registry and pure validators but intentionally does not migrate "
            "legacy optimizers or API routes."
        )
    if legacy_scan["existing_source_count"] < legacy_scan["source_count"]:
        warnings.append(
            "One or more optional legacy comparison sources were unavailable."
        )

    passed = len(blockers) == 0
    return {
        "created_at": utc_now(),
        "validation_version": VALIDATION_VERSION,
        "effective_season": rules.effective_season,
        "rules_version": rules.rules_version,
        "schema_version": rules.schema_version,
        "rules_path": str(rules.path),
        "rules_sha256": rules.sha256,
        "passed": passed,
        "audit_only": True,
        "writes_database": False,
        "ready_for_squad_transfer_rules_loading": passed,
        "ready_for_pure_squad_validation": passed
        and bool(squad_examples["all_passed"]),
        "ready_for_pure_lineup_validation": passed
        and bool(lineup_examples["all_passed"]),
        "ready_for_pure_transfer_validation": passed
        and bool(transfer_examples["all_passed"])
        and bool(transfer_legality_examples["all_passed"]),
        "ready_for_optimizer_rule_migration": passed,
        "legacy_callers_use_registry": legacy_scan[
            "all_existing_sources_use_registry"
        ],
        "ready_for_constant_free_legacy_optimizers": passed
        and legacy_scan["all_existing_sources_use_registry"],
        "ready_for_production_decision_write": False,
        "policy_summary": {
            "initial_budget_units": rules.initial_budget_units,
            "squad_size": int(rules.squad["size"]),
            "position_quotas": rules.position_quotas,
            "starting_size": int(rules.lineup["starting_size"]),
            "formation_bounds": rules.lineup["position_bounds"],
            "max_players_per_club": int(
                rules.squad["max_players_per_club"]
            ),
            "weekly_free_transfer_accrual": int(
                rules.transfers["weekly"][
                    "free_transfers_accrued_per_gameweek"
                ]
            ),
            "maximum_stored_free_transfers": maximum_stored_free_transfers(
                rules
            ),
            "hit_cost_points_per_additional_transfer": int(
                rules.transfers["weekly"][
                    "hit_cost_points_per_additional_transfer"
                ]
            ),
            "special_event_count": len(rules.transfers["special_events"]),
        },
        "deterministic_examples": {
            "squad": squad_examples,
            "lineup": lineup_examples,
            "transfers": transfer_examples,
            "transfer_legality": transfer_legality_examples,
        },
        "legacy_comparison": legacy_scan,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "All legality functions are pure and receive a loaded versioned policy.",
            "Transfer pricing accepts any non-negative transfer_count and never assumes exactly one transfer per Gameweek.",
            "Ongoing-squad affordability consumes a caller-supplied budget_limit_units because live selling value and bank are external inputs.",
            "Day74B does not alter optimizers, API routes, database rows, or recommendation outputs.",
        ],
    }


def write_json(report: Mapping[str, Any], path_value: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def write_markdown(report: Mapping[str, Any], path_value: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = report["deterministic_examples"]
    legacy = report["legacy_comparison"]
    lines = [
        "# Day74B Squad and Transfer Rules Validation",
        "",
        "- Validation version: `%s`" % report["validation_version"],
        "- Effective season: `%s`" % report["effective_season"],
        "- Rules version: `%s`" % report["rules_version"],
        "- Rules SHA256: `%s`" % report["rules_sha256"],
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `%s`" % report["audit_only"],
        "- Writes database: `%s`" % report["writes_database"],
        "",
        "## Readiness",
        "",
        "- Squad rules loading: `%s`"
        % report["ready_for_squad_transfer_rules_loading"],
        "- Pure squad validation: `%s`"
        % report["ready_for_pure_squad_validation"],
        "- Pure lineup validation: `%s`"
        % report["ready_for_pure_lineup_validation"],
        "- Pure transfer validation: `%s`"
        % report["ready_for_pure_transfer_validation"],
        "- Optimizer migration ready: `%s`"
        % report["ready_for_optimizer_rule_migration"],
        "- Legacy callers use registry: `%s`"
        % report["legacy_callers_use_registry"],
        "- Production decision write: `%s`"
        % report["ready_for_production_decision_write"],
        "",
        "## Deterministic Examples",
        "",
        "- Squad: `%s/%s` passed"
        % (examples["squad"]["passed_count"], examples["squad"]["case_count"]),
        "- Lineup: `%s/%s` passed"
        % (examples["lineup"]["passed_count"], examples["lineup"]["case_count"]),
        "- Transfer sequences: `%s/%s` passed"
        % (
            examples["transfers"]["passed_count"],
            examples["transfers"]["case_count"],
        ),
        "- Transfer legality: `%s/%s` passed"
        % (
            examples["transfer_legality"]["passed_count"],
            examples["transfer_legality"]["case_count"],
        ),
        "",
        "## Legacy Comparison",
        "",
        "- Existing sources: `%s/%s`"
        % (legacy["existing_source_count"], legacy["source_count"]),
        "- Sources with hardcoded rules: `%s`"
        % legacy["sources_with_hardcoded_rules"],
        "- Hardcoded findings: `%s`"
        % legacy["hardcoded_rule_finding_count"],
        "- Registry-integrated sources: `%s`"
        % legacy["registry_integrated_source_count"],
        "",
        "## Blockers",
        "",
    ]
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(report: Mapping[str, Any], out_json: str, out_md: str) -> None:
    examples = report["deterministic_examples"]
    legacy = report["legacy_comparison"]
    print("=== Day74B Squad and Transfer Rules Registry ===")
    print("validation_version:", report["validation_version"])
    print("effective_season:", report["effective_season"])
    print("rules_version:", report["rules_version"])
    print("schema_version:", report["schema_version"])
    print("rules_sha256:", report["rules_sha256"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print(
        "ready_for_squad_transfer_rules_loading:",
        report["ready_for_squad_transfer_rules_loading"],
    )
    print(
        "ready_for_pure_squad_validation:",
        report["ready_for_pure_squad_validation"],
    )
    print(
        "ready_for_pure_lineup_validation:",
        report["ready_for_pure_lineup_validation"],
    )
    print(
        "ready_for_pure_transfer_validation:",
        report["ready_for_pure_transfer_validation"],
    )
    print(
        "ready_for_optimizer_rule_migration:",
        report["ready_for_optimizer_rule_migration"],
    )
    print("legacy_callers_use_registry:", report["legacy_callers_use_registry"])
    print(
        "ready_for_constant_free_legacy_optimizers:",
        report["ready_for_constant_free_legacy_optimizers"],
    )
    print(
        "ready_for_production_decision_write:",
        report["ready_for_production_decision_write"],
    )
    print()
    print("Deterministic examples:")
    print(
        "- squad: %s/%s passed"
        % (examples["squad"]["passed_count"], examples["squad"]["case_count"])
    )
    print(
        "- lineup: %s/%s passed"
        % (examples["lineup"]["passed_count"], examples["lineup"]["case_count"])
    )
    print(
        "- transfer_sequences: %s/%s passed"
        % (
            examples["transfers"]["passed_count"],
            examples["transfers"]["case_count"],
        )
    )
    print(
        "- transfer_legality: %s/%s passed"
        % (
            examples["transfer_legality"]["passed_count"],
            examples["transfer_legality"]["case_count"],
        )
    )
    print()
    print("Legacy comparison:")
    print("- source_count:", legacy["source_count"])
    print("- existing_source_count:", legacy["existing_source_count"])
    print(
        "- sources_with_hardcoded_rules:",
        legacy["sources_with_hardcoded_rules"],
    )
    print(
        "- hardcoded_rule_finding_count:",
        legacy["hardcoded_rule_finding_count"],
    )
    print(
        "- registry_integrated_source_count:",
        legacy["registry_integrated_source_count"],
    )
    print()
    print("Blockers:", report["blockers"] or "none")
    print("Warnings:", report["warnings"] or "none")
    if out_json:
        print("saved_json:", out_json)
    if out_md:
        print("saved_md:", out_md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load and validate versioned FPL squad/transfer rules, run pure "
            "deterministic legality examples, and audit legacy hardcoded callers."
        )
    )
    parser.add_argument("--season", required=True)
    parser.add_argument("--config-path", default="")
    parser.add_argument(
        "--legacy-source",
        action="append",
        default=[],
        help="Optional legacy source path. Repeat for multiple files.",
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path) if args.config_path else None
    rules = load_squad_transfer_rules(
        season=args.season,
        config_path=config_path,
    )
    legacy_sources = (
        [Path(value) for value in args.legacy_source]
        if args.legacy_source
        else list(DEFAULT_LEGACY_SOURCES)
    )
    report = build_validation_report(rules, legacy_sources)
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report, args.out_json, args.out_md)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
