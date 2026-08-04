from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.rules.squad import (
    SquadTransferRules,
    load_squad_transfer_rules,
    validate_lineup as validate_policy_lineup,
    validate_squad as validate_policy_squad,
)


ENGINE_VERSION = "day100a_v1"
VALID_POSITIONS = ("GKP", "DEF", "MID", "FWD")
_MISSING = object()


@dataclass(frozen=True)
class LegalityIssue:
    """One deterministic failed constraint.

    The issue shape is intentionally JSON-safe so optimization, audit, and
    presentation layers can preserve the exact reason a plan was rejected.
    """

    code: str
    scope: str
    message: str
    constraint: str
    expected: Any = None
    actual: Any = None
    player_ids: Tuple[Any, ...] = field(default_factory=tuple)
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "scope": self.scope,
            "severity": self.severity,
            "message": self.message,
            "constraint": self.constraint,
            "expected": self.expected,
            "actual": self.actual,
            "player_ids": list(self.player_ids),
            "details": dict(self.details),
        }


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Any, Optional[str]]:
    for key in keys:
        if key in mapping:
            return mapping[key], key
    return _MISSING, None


def _is_valid_identity(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and bool(value.strip())


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _coerce_nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if value.is_integer() and value >= 0:
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _price_to_units(value: Any) -> Optional[int]:
    """Convert a price expressed in millions to integer tenths.

    This fallback is used only when an explicit ``price_units`` field is absent.
    Values that are not exact tenths fail closed.
    """

    if isinstance(value, bool):
        return None
    try:
        decimal_value = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None

    units = decimal_value * Decimal("10")
    if units != units.to_integral_value() or units < 0:
        return None
    return int(units)


def _duplicate_values(values: Sequence[Any]) -> List[Any]:
    seen = set()
    duplicates: List[Any] = []
    duplicate_keys = set()
    for value in values:
        key = (type(value).__name__, str(value))
        if key in seen and key not in duplicate_keys:
            duplicates.append(value)
            duplicate_keys.add(key)
        seen.add(key)
    return duplicates


def _issue(
    *,
    code: str,
    scope: str,
    message: str,
    constraint: str,
    expected: Any = None,
    actual: Any = None,
    player_ids: Sequence[Any] = (),
    details: Optional[Mapping[str, Any]] = None,
) -> LegalityIssue:
    return LegalityIssue(
        code=code,
        scope=scope,
        message=message,
        constraint=constraint,
        expected=expected,
        actual=actual,
        player_ids=tuple(player_ids),
        details=dict(details or {}),
    )


def _normalize_optimizer_players(
    players: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[LegalityIssue]]:
    issues: List[LegalityIssue] = []
    normalized: List[Dict[str, Any]] = []

    if isinstance(players, (str, bytes)) or not isinstance(players, Sequence):
        return [], [
            _issue(
                code="players_not_sequence",
                scope="input",
                message="Squad players must be provided as a sequence of player rows.",
                constraint="players_sequence",
                expected="sequence_of_mappings",
                actual=type(players).__name__,
            )
        ]

    for index, raw_player in enumerate(players):
        label = "players[%s]" % index
        if not isinstance(raw_player, Mapping):
            issues.append(
                _issue(
                    code="player_row_not_mapping",
                    scope="input",
                    message="%s must be a mapping." % label,
                    constraint="player_row_shape",
                    expected="mapping",
                    actual=type(raw_player).__name__,
                    details={"row_index": index},
                )
            )
            continue

        row_issues: List[LegalityIssue] = []

        player_id, player_id_source = _first_present(
            raw_player,
            ("player_id", "target_player_id"),
        )
        if player_id is _MISSING or not _is_valid_identity(player_id):
            row_issues.append(
                _issue(
                    code="player_id_invalid",
                    scope="input",
                    message="%s has no valid player identity." % label,
                    constraint="player_identity",
                    expected="nonempty_int_or_string",
                    actual=None if player_id is _MISSING else player_id,
                    details={"row_index": index},
                )
            )

        position, position_source = _first_present(
            raw_player,
            ("position", "target_position"),
        )
        normalized_position = (
            str(position).strip().upper()
            if position is not _MISSING and position is not None
            else ""
        )
        if normalized_position not in VALID_POSITIONS:
            row_issues.append(
                _issue(
                    code="player_position_invalid",
                    scope="input",
                    message="%s has an invalid target-season position." % label,
                    constraint="player_position",
                    expected=list(VALID_POSITIONS),
                    actual=normalized_position or None,
                    player_ids=()
                    if player_id is _MISSING
                    else (player_id,),
                    details={"row_index": index},
                )
            )

        club_id, club_id_source = _first_present(
            raw_player,
            (
                "club_id",
                "team_id",
                "target_team_id",
                "team_short_name",
                "target_team_short_name",
            ),
        )
        if club_id is _MISSING or not _is_valid_identity(club_id):
            row_issues.append(
                _issue(
                    code="player_club_invalid",
                    scope="input",
                    message="%s has no valid target-season club identity." % label,
                    constraint="player_club_identity",
                    expected="nonempty_int_or_string",
                    actual=None if club_id is _MISSING else club_id,
                    player_ids=()
                    if player_id is _MISSING
                    else (player_id,),
                    details={"row_index": index},
                )
            )

        price_units_value, price_units_source = _first_present(
            raw_player,
            ("price_units", "target_price_units", "now_cost"),
        )
        price_units: Optional[int]
        if price_units_value is not _MISSING:
            price_units = _coerce_nonnegative_int(price_units_value)
        else:
            price_value, price_source = _first_present(
                raw_player,
                ("price", "target_price"),
            )
            if price_value is _MISSING:
                price_units = None
                price_units_source = None
            else:
                price_units = _price_to_units(price_value)
                price_units_source = price_source

        if price_units is None:
            row_issues.append(
                _issue(
                    code="player_price_units_invalid",
                    scope="input",
                    message="%s has no valid nonnegative price in policy units." % label,
                    constraint="player_price_units",
                    expected="nonnegative_integer_tenths",
                    actual=None
                    if price_units_value is _MISSING
                    else price_units_value,
                    player_ids=()
                    if player_id is _MISSING
                    else (player_id,),
                    details={"row_index": index},
                )
            )

        selection_value, selection_source = _first_present(
            raw_player,
            ("selection_eligible",),
        )
        selection_eligible = (
            None
            if selection_value is _MISSING
            else _coerce_bool(selection_value)
        )
        if selection_value is _MISSING:
            row_issues.append(
                _issue(
                    code="player_eligibility_missing",
                    scope="eligibility",
                    message="%s is missing explicit selection eligibility." % label,
                    constraint="selection_eligible_present",
                    expected=True,
                    actual=None,
                    player_ids=()
                    if player_id is _MISSING
                    else (player_id,),
                    details={"row_index": index},
                )
            )
        elif selection_eligible is None:
            row_issues.append(
                _issue(
                    code="player_eligibility_invalid",
                    scope="eligibility",
                    message="%s has a non-boolean selection eligibility value." % label,
                    constraint="selection_eligible_boolean",
                    expected="boolean",
                    actual=selection_value,
                    player_ids=()
                    if player_id is _MISSING
                    else (player_id,),
                    details={"row_index": index},
                )
            )

        eligibility_reason_value, _ = _first_present(
            raw_player,
            ("eligibility_reason",),
        )
        eligibility_reason = (
            ""
            if eligibility_reason_value is _MISSING
            or eligibility_reason_value is None
            else str(eligibility_reason_value).strip()
        )

        cutoff_value, _ = _first_present(raw_player, ("status_cutoff_valid",))
        cutoff_valid = (
            None if cutoff_value is _MISSING else _coerce_bool(cutoff_value)
        )
        if cutoff_value is not _MISSING and cutoff_valid is None:
            row_issues.append(
                _issue(
                    code="status_cutoff_valid_invalid",
                    scope="eligibility",
                    message="%s has a non-boolean status_cutoff_valid value." % label,
                    constraint="status_cutoff_valid_boolean",
                    expected="boolean",
                    actual=cutoff_value,
                    player_ids=()
                    if player_id is _MISSING
                    else (player_id,),
                    details={"row_index": index},
                )
            )

        guardrail_value, _ = _first_present(
            raw_player,
            ("status_hard_guardrail_applied",),
        )
        hard_guardrail = (
            None
            if guardrail_value is _MISSING
            else _coerce_bool(guardrail_value)
        )
        if guardrail_value is not _MISSING and hard_guardrail is None:
            row_issues.append(
                _issue(
                    code="status_hard_guardrail_invalid",
                    scope="eligibility",
                    message="%s has a non-boolean hard-guardrail value." % label,
                    constraint="status_hard_guardrail_boolean",
                    expected="boolean",
                    actual=guardrail_value,
                    player_ids=()
                    if player_id is _MISSING
                    else (player_id,),
                    details={"row_index": index},
                )
            )

        if selection_eligible is False:
            row_issues.append(
                _issue(
                    code="player_ineligible",
                    scope="eligibility",
                    message="%s is not eligible for squad selection." % label,
                    constraint="selected_players_eligible",
                    expected=True,
                    actual=False,
                    player_ids=()
                    if player_id is _MISSING
                    else (player_id,),
                    details={
                        "row_index": index,
                        "eligibility_reason": eligibility_reason
                        or "unspecified",
                    },
                )
            )

        if selection_eligible is True and cutoff_valid is False:
            row_issues.append(
                _issue(
                    code="eligible_player_status_cutoff_invalid",
                    scope="eligibility",
                    message=(
                        "%s is marked eligible despite an invalid status cutoff."
                        % label
                    ),
                    constraint="eligible_requires_valid_status_cutoff",
                    expected=True,
                    actual=False,
                    player_ids=()
                    if player_id is _MISSING
                    else (player_id,),
                    details={"row_index": index},
                )
            )

        if selection_eligible is True and hard_guardrail is True:
            row_issues.append(
                _issue(
                    code="eligible_player_hard_guardrail_conflict",
                    scope="eligibility",
                    message=(
                        "%s is marked eligible despite a hard status guardrail."
                        % label
                    ),
                    constraint="eligible_requires_no_hard_guardrail",
                    expected=False,
                    actual=True,
                    player_ids=()
                    if player_id is _MISSING
                    else (player_id,),
                    details={"row_index": index},
                )
            )

        issues.extend(row_issues)
        if row_issues:
            continue

        normalized.append(
            {
                "player_id": player_id,
                "position": normalized_position,
                "club_id": club_id,
                "price_units": int(price_units),
                "selection_eligible": bool(selection_eligible),
                "eligibility_reason": eligibility_reason,
                "status_cutoff_valid": cutoff_valid,
                "status_hard_guardrail_applied": hard_guardrail,
                "player_name": raw_player.get("player_name")
                or raw_player.get("target_player_name"),
                "source_fields": {
                    "player_id": player_id_source,
                    "position": position_source,
                    "club_id": club_id_source,
                    "price_units": price_units_source,
                    "selection_eligible": selection_source,
                },
            }
        )

    return normalized, issues


def _squad_issues_from_policy_result(
    rules: SquadTransferRules,
    normalized_players: Sequence[Mapping[str, Any]],
    policy_result: Mapping[str, Any],
) -> List[LegalityIssue]:
    issues: List[LegalityIssue] = []
    player_ids = [player["player_id"] for player in normalized_players]

    for raw_error in policy_result.get("errors", []):
        code = str(raw_error).split(":", 1)[0]
        if code == "squad_size_invalid":
            issues.append(
                _issue(
                    code=code,
                    scope="squad",
                    message="The squad does not contain the policy-defined number of players.",
                    constraint="squad_size",
                    expected=int(rules.squad["size"]),
                    actual=int(policy_result["player_count"]),
                )
            )
        elif code == "duplicate_player_ids":
            duplicates = _duplicate_values(player_ids)
            issues.append(
                _issue(
                    code=code,
                    scope="squad",
                    message="Every selected player must be unique.",
                    constraint="player_uniqueness",
                    expected="all_player_ids_unique",
                    actual=duplicates,
                    player_ids=duplicates,
                )
            )
        elif code == "position_quotas_invalid":
            issues.append(
                _issue(
                    code=code,
                    scope="squad",
                    message="The squad position quotas do not match the loaded policy.",
                    constraint="position_quotas",
                    expected=dict(rules.position_quotas),
                    actual=dict(policy_result["position_counts"]),
                )
            )
        elif code == "club_limit_exceeded":
            max_per_club = int(rules.squad["max_players_per_club"])
            violations = {
                club_id: count
                for club_id, count in policy_result["club_counts"].items()
                if int(count) > max_per_club
            }
            violating_players = [
                player["player_id"]
                for player in normalized_players
                if str(player["club_id"]) in violations
            ]
            issues.append(
                _issue(
                    code=code,
                    scope="squad",
                    message="One or more clubs exceed the policy-defined player limit.",
                    constraint="max_players_per_club",
                    expected=max_per_club,
                    actual=violations,
                    player_ids=violating_players,
                )
            )
        elif code == "budget_exceeded":
            issues.append(
                _issue(
                    code=code,
                    scope="budget",
                    message="The selected squad costs more than the allowed budget.",
                    constraint="total_budget",
                    expected=int(policy_result["budget_limit_units"]),
                    actual=int(policy_result["total_price_units"]),
                )
            )
        else:
            issues.append(
                _issue(
                    code=code or "squad_policy_error",
                    scope="squad",
                    message="The canonical squad validator rejected the squad.",
                    constraint="canonical_squad_policy",
                    actual=str(raw_error),
                    details={"raw_error": str(raw_error)},
                )
            )

    return issues


def _lineup_issues_from_policy_result(
    rules: SquadTransferRules,
    normalized_players: Sequence[Mapping[str, Any]],
    starting_player_ids: Sequence[Any],
    bench_order: Sequence[Any],
    captain_player_id: Optional[Any],
    vice_captain_player_id: Optional[Any],
    policy_result: Mapping[str, Any],
) -> List[LegalityIssue]:
    issues: List[LegalityIssue] = []
    squad_by_id = {
        player["player_id"]: player for player in normalized_players
    }
    starting_ids = list(starting_player_ids)
    bench_ids = list(bench_order)

    for raw_error in policy_result.get("errors", []):
        code = str(raw_error).split(":", 1)[0]

        if code == "starting_size_invalid":
            issues.append(
                _issue(
                    code=code,
                    scope="lineup",
                    message="The starting lineup size does not match the loaded policy.",
                    constraint="starting_lineup_size",
                    expected=int(rules.lineup["starting_size"]),
                    actual=len(starting_ids),
                    player_ids=starting_ids,
                )
            )
        elif code == "bench_size_invalid":
            issues.append(
                _issue(
                    code=code,
                    scope="bench",
                    message="The bench size does not match the loaded policy.",
                    constraint="bench_size",
                    expected=int(rules.lineup["bench_size"]),
                    actual=len(bench_ids),
                    player_ids=bench_ids,
                )
            )
        elif code == "duplicate_starting_player_ids":
            duplicates = _duplicate_values(starting_ids)
            issues.append(
                _issue(
                    code=code,
                    scope="lineup",
                    message="The starting lineup contains duplicate players.",
                    constraint="starting_player_uniqueness",
                    expected="all_starting_ids_unique",
                    actual=duplicates,
                    player_ids=duplicates,
                )
            )
        elif code == "duplicate_bench_player_ids":
            duplicates = _duplicate_values(bench_ids)
            issues.append(
                _issue(
                    code=code,
                    scope="bench",
                    message="The bench order contains duplicate players.",
                    constraint="bench_player_uniqueness",
                    expected="all_bench_ids_unique",
                    actual=duplicates,
                    player_ids=duplicates,
                )
            )
        elif code == "starting_and_bench_overlap":
            overlap = [
                player_id
                for player_id in starting_ids
                if player_id in set(bench_ids)
            ]
            issues.append(
                _issue(
                    code=code,
                    scope="lineup",
                    message="A player cannot be both a starter and a substitute.",
                    constraint="starting_bench_disjoint",
                    expected="no_overlap",
                    actual=overlap,
                    player_ids=overlap,
                )
            )
        elif code == "starting_players_not_in_squad":
            unknown = [
                player_id
                for player_id in starting_ids
                if player_id not in squad_by_id
            ]
            issues.append(
                _issue(
                    code=code,
                    scope="lineup",
                    message="Every starter must belong to the validated squad.",
                    constraint="starting_players_in_squad",
                    expected=True,
                    actual=False,
                    player_ids=unknown,
                )
            )
        elif code == "bench_players_not_in_squad":
            unknown = [
                player_id
                for player_id in bench_ids
                if player_id not in squad_by_id
            ]
            issues.append(
                _issue(
                    code=code,
                    scope="bench",
                    message="Every substitute must belong to the validated squad.",
                    constraint="bench_players_in_squad",
                    expected=True,
                    actual=False,
                    player_ids=unknown,
                )
            )
        elif code == "starting_and_bench_do_not_partition_squad":
            partition_ids = set(starting_ids) | set(bench_ids)
            squad_ids = set(squad_by_id.keys())
            issues.append(
                _issue(
                    code=code,
                    scope="lineup",
                    message="Starters and bench must partition the complete squad.",
                    constraint="lineup_partitions_squad",
                    expected=sorted(str(value) for value in squad_ids),
                    actual=sorted(str(value) for value in partition_ids),
                    details={
                        "missing_from_plan": sorted(
                            str(value) for value in squad_ids - partition_ids
                        ),
                        "extra_in_plan": sorted(
                            str(value) for value in partition_ids - squad_ids
                        ),
                    },
                )
            )
        elif code.startswith("formation_invalid_"):
            position = code.rsplit("_", 1)[-1]
            bounds = rules.lineup["position_bounds"][position]
            issues.append(
                _issue(
                    code=code,
                    scope="lineup",
                    message=(
                        "The starting lineup violates the %s formation bound."
                        % position
                    ),
                    constraint="formation_%s" % position,
                    expected={
                        "min": int(bounds["min"]),
                        "max": int(bounds["max"]),
                    },
                    actual=int(
                        policy_result["starting_position_counts"][position]
                    ),
                    player_ids=[
                        player_id
                        for player_id in starting_ids
                        if player_id in squad_by_id
                        and squad_by_id[player_id]["position"] == position
                    ],
                )
            )
        elif code == "bench_goalkeeper_count_invalid":
            issues.append(
                _issue(
                    code=code,
                    scope="bench",
                    message="The bench must contain the policy-defined goalkeeper count.",
                    constraint="bench_goalkeeper_count",
                    expected=int(rules.lineup["bench"]["goalkeepers"]),
                    actual=int(
                        policy_result["bench_position_counts"]["GKP"]
                    ),
                )
            )
        elif code == "bench_outfield_count_invalid":
            actual_goalkeepers = int(
                policy_result["bench_position_counts"]["GKP"]
            )
            issues.append(
                _issue(
                    code=code,
                    scope="bench",
                    message="The bench must contain the policy-defined outfield count.",
                    constraint="bench_outfield_count",
                    expected=int(rules.lineup["bench"]["outfield_players"]),
                    actual=max(
                        0,
                        int(policy_result["bench_player_count"])
                        - actual_goalkeepers,
                    ),
                )
            )
        elif code == "captain_missing":
            issues.append(
                _issue(
                    code=code,
                    scope="captaincy",
                    message="A captain is required by the loaded policy.",
                    constraint="captain_required",
                    expected=True,
                    actual=False,
                )
            )
        elif code == "captain_not_in_starting_lineup":
            issues.append(
                _issue(
                    code=code,
                    scope="captaincy",
                    message="The captain must be selected in the starting lineup.",
                    constraint="captain_is_starter",
                    expected=True,
                    actual=False,
                    player_ids=()
                    if captain_player_id is None
                    else (captain_player_id,),
                )
            )
        elif code == "vice_captain_missing":
            issues.append(
                _issue(
                    code=code,
                    scope="captaincy",
                    message="A vice-captain is required by the loaded policy.",
                    constraint="vice_captain_required",
                    expected=True,
                    actual=False,
                )
            )
        elif code == "vice_captain_not_in_starting_lineup":
            issues.append(
                _issue(
                    code=code,
                    scope="captaincy",
                    message="The vice-captain must be selected in the starting lineup.",
                    constraint="vice_captain_is_starter",
                    expected=True,
                    actual=False,
                    player_ids=()
                    if vice_captain_player_id is None
                    else (vice_captain_player_id,),
                )
            )
        elif code == "captain_and_vice_captain_must_differ":
            player_ids = (
                ()
                if captain_player_id is None
                else (captain_player_id,)
            )
            issues.append(
                _issue(
                    code=code,
                    scope="captaincy",
                    message="Captain and vice-captain must be different players.",
                    constraint="captain_vice_distinct",
                    expected="different_player_ids",
                    actual=captain_player_id,
                    player_ids=player_ids,
                )
            )
        else:
            issues.append(
                _issue(
                    code=code or "lineup_policy_error",
                    scope="lineup",
                    message="The canonical lineup validator rejected the plan.",
                    constraint="canonical_lineup_policy",
                    actual=str(raw_error),
                    details={"raw_error": str(raw_error)},
                )
            )

    return issues


def _bench_slot_issues(
    normalized_players: Sequence[Mapping[str, Any]],
    bench_order: Sequence[Any],
) -> List[LegalityIssue]:
    """Validate the ordered bench representation used by the optimizer.

    Day100A defines index 0 as the substitute-goalkeeper slot and indices 1-3
    as ordered outfield substitute slots. This is an interface contract, not a
    new season-specific quota.
    """

    squad_by_id = {
        player["player_id"]: player for player in normalized_players
    }
    bench_ids = list(bench_order)
    issues: List[LegalityIssue] = []

    if not bench_ids:
        return issues

    first_id = bench_ids[0]
    if first_id in squad_by_id:
        first_position = squad_by_id[first_id]["position"]
        if first_position != "GKP":
            issues.append(
                _issue(
                    code="bench_goalkeeper_slot_invalid",
                    scope="bench",
                    message="Bench slot 0 is reserved for the substitute goalkeeper.",
                    constraint="bench_goalkeeper_slot",
                    expected="GKP",
                    actual=first_position,
                    player_ids=(first_id,),
                    details={"bench_index": 0},
                )
            )

    misplaced_goalkeepers = [
        player_id
        for player_id in bench_ids[1:]
        if player_id in squad_by_id
        and squad_by_id[player_id]["position"] == "GKP"
    ]
    if misplaced_goalkeepers:
        issues.append(
            _issue(
                code="bench_outfield_slots_contain_goalkeeper",
                scope="bench",
                message="Ordered outfield bench slots cannot contain a goalkeeper.",
                constraint="bench_outfield_slots",
                expected="outfield_players_only",
                actual="goalkeeper_present",
                player_ids=misplaced_goalkeepers,
                details={"bench_indices": [bench_ids.index(value) for value in misplaced_goalkeepers]},
            )
        )

    return issues


def _formation_label(position_counts: Mapping[str, Any]) -> Optional[str]:
    if not position_counts:
        return None
    try:
        return "%s-%s-%s" % (
            int(position_counts["DEF"]),
            int(position_counts["MID"]),
            int(position_counts["FWD"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


class SquadLegalityEngine:
    """Optimizer-facing deterministic legality engine.

    The engine delegates canonical season rules to ``app.rules.squad`` and
    adds the Fast Lane contracts that are intentionally outside the base
    registry: explicit player eligibility, bank reconciliation, ordered bench
    slots, and structured failure reasons.
    """

    def __init__(self, rules: SquadTransferRules):
        self.rules = rules

    @classmethod
    def from_season(
        cls,
        target_season: str,
        config_path: Optional[Path] = None,
    ) -> "SquadLegalityEngine":
        rules = load_squad_transfer_rules(
            target_season,
            config_path=config_path,
        )
        return cls(rules)

    def rules_metadata(self) -> Dict[str, Any]:
        return {
            "effective_season": self.rules.effective_season,
            "rules_version": self.rules.rules_version,
            "schema_version": self.rules.schema_version,
            "rules_sha256": self.rules.sha256,
            "rules_path": str(self.rules.path),
        }

    def validate_squad(
        self,
        squad_players: Sequence[Mapping[str, Any]],
        *,
        declared_bank_units: Optional[int] = None,
        budget_limit_units: Optional[int] = None,
        require_declared_bank: bool = False,
    ) -> Dict[str, Any]:
        normalized, issues = _normalize_optimizer_players(squad_players)

        resolved_budget_limit = (
            self.rules.initial_budget_units
            if budget_limit_units is None
            else _coerce_nonnegative_int(budget_limit_units)
        )
        if resolved_budget_limit is None:
            issues.append(
                _issue(
                    code="budget_limit_units_invalid",
                    scope="budget",
                    message="Budget limit must be a nonnegative integer in policy units.",
                    constraint="budget_limit_units",
                    expected="nonnegative_integer",
                    actual=budget_limit_units,
                )
            )

        declared_bank: Optional[int]
        if declared_bank_units is None:
            declared_bank = None
            if require_declared_bank:
                issues.append(
                    _issue(
                        code="declared_bank_missing",
                        scope="bank",
                        message="A complete squad plan must declare its remaining bank.",
                        constraint="declared_bank_present",
                        expected=True,
                        actual=False,
                    )
                )
        else:
            declared_bank = _coerce_nonnegative_int(declared_bank_units)
            if declared_bank is None:
                issues.append(
                    _issue(
                        code="declared_bank_units_invalid",
                        scope="bank",
                        message="Declared bank must be a nonnegative integer in policy units.",
                        constraint="declared_bank_units",
                        expected="nonnegative_integer",
                        actual=declared_bank_units,
                    )
                )

        policy_result: Optional[Dict[str, Any]] = None
        if not any(issue.scope == "input" for issue in issues) and resolved_budget_limit is not None:
            # Eligibility issues do not prevent the canonical validator from
            # reporting every independent squad constraint.
            policy_result = validate_policy_squad(
                self.rules,
                normalized,
                budget_limit_units=resolved_budget_limit,
            )
            issues.extend(
                _squad_issues_from_policy_result(
                    self.rules,
                    normalized,
                    policy_result,
                )
            )

        total_price_units = (
            sum(int(player["price_units"]) for player in normalized)
            if normalized
            else None
        )
        computed_bank_units = (
            resolved_budget_limit - total_price_units
            if resolved_budget_limit is not None
            and total_price_units is not None
            else None
        )

        if (
            declared_bank is not None
            and computed_bank_units is not None
            and declared_bank != computed_bank_units
        ):
            issues.append(
                _issue(
                    code="declared_bank_mismatch",
                    scope="bank",
                    message="Declared bank does not reconcile with budget minus squad cost.",
                    constraint="bank_reconciliation",
                    expected=computed_bank_units,
                    actual=declared_bank,
                    details={
                        "budget_limit_units": resolved_budget_limit,
                        "total_price_units": total_price_units,
                    },
                )
            )

        issue_dicts = [issue.to_dict() for issue in issues]
        return {
            "valid": len(issues) == 0,
            "engine_version": ENGINE_VERSION,
            "rules": self.rules_metadata(),
            "issues": issue_dicts,
            "issue_codes": list(dict.fromkeys(issue["code"] for issue in issue_dicts)),
            "normalized_player_count": len(normalized),
            "eligible_player_count": sum(
                1
                for player in normalized
                if player["selection_eligible"] is True
            ),
            "ineligible_player_ids": [
                player["player_id"]
                for player in normalized
                if player["selection_eligible"] is not True
            ],
            "budget_limit_units": resolved_budget_limit,
            "total_price_units": total_price_units,
            "computed_bank_units": computed_bank_units,
            "declared_bank_units": declared_bank,
            "policy_validation": policy_result,
            "_normalized_players": normalized,
        }

    def validate_plan(
        self,
        squad_players: Sequence[Mapping[str, Any]],
        *,
        starting_player_ids: Sequence[Any],
        bench_order: Sequence[Any],
        captain_player_id: Optional[Any],
        vice_captain_player_id: Optional[Any],
        declared_bank_units: int,
        budget_limit_units: Optional[int] = None,
    ) -> Dict[str, Any]:
        squad_result = self.validate_squad(
            squad_players,
            declared_bank_units=declared_bank_units,
            budget_limit_units=budget_limit_units,
            require_declared_bank=True,
        )
        normalized = squad_result.pop("_normalized_players")
        issues = [
            LegalityIssue(
                code=issue["code"],
                scope=issue["scope"],
                severity=issue["severity"],
                message=issue["message"],
                constraint=issue["constraint"],
                expected=issue["expected"],
                actual=issue["actual"],
                player_ids=tuple(issue["player_ids"]),
                details=dict(issue["details"]),
            )
            for issue in squad_result["issues"]
        ]

        lineup_policy_result: Optional[Dict[str, Any]] = None
        has_input_errors = any(issue.scope == "input" for issue in issues)
        if not has_input_errors:
            lineup_policy_result = validate_policy_lineup(
                self.rules,
                normalized,
                starting_player_ids,
                bench_order,
                captain_player_id,
                vice_captain_player_id,
            )
            issues.extend(
                _lineup_issues_from_policy_result(
                    self.rules,
                    normalized,
                    starting_player_ids,
                    bench_order,
                    captain_player_id,
                    vice_captain_player_id,
                    lineup_policy_result,
                )
            )
            issues.extend(_bench_slot_issues(normalized, bench_order))

        issue_dicts = [issue.to_dict() for issue in issues]
        starting_counts = (
            dict(lineup_policy_result["starting_position_counts"])
            if lineup_policy_result is not None
            else {}
        )
        bench_counts = (
            dict(lineup_policy_result["bench_position_counts"])
            if lineup_policy_result is not None
            else {}
        )

        squad_scopes = {"input", "eligibility", "squad", "budget", "bank"}
        lineup_scopes = {"lineup", "bench", "captaincy"}

        return {
            "valid": len(issues) == 0,
            "engine_version": ENGINE_VERSION,
            "rules": self.rules_metadata(),
            "issues": issue_dicts,
            "issue_codes": list(dict.fromkeys(issue["code"] for issue in issue_dicts)),
            "squad": {
                key: value
                for key, value in squad_result.items()
                if key
                not in {
                    "valid",
                    "engine_version",
                    "rules",
                    "issues",
                    "issue_codes",
                }
            },
            "lineup": {
                "policy_validation": lineup_policy_result,
                "starting_player_ids": list(starting_player_ids),
                "bench_order": list(bench_order),
                "captain_player_id": captain_player_id,
                "vice_captain_player_id": vice_captain_player_id,
                "starting_position_counts": starting_counts,
                "bench_position_counts": bench_counts,
                "formation": _formation_label(starting_counts),
                "bench_slot_contract": {
                    "slot_0": "substitute_goalkeeper",
                    "slots_1_to_3": "ordered_outfield_substitutes",
                },
            },
            "component_validity": {
                "squad": not any(
                    issue.scope in squad_scopes for issue in issues
                ),
                "lineup": not any(
                    issue.scope in lineup_scopes for issue in issues
                ),
                "captaincy": not any(
                    issue.scope == "captaincy" for issue in issues
                ),
                "bench": not any(issue.scope == "bench" for issue in issues),
                "bank": not any(issue.scope == "bank" for issue in issues),
                "eligibility": not any(
                    issue.scope == "eligibility" for issue in issues
                ),
            },
        }


def validate_complete_squad_plan(
    *,
    target_season: str,
    squad_players: Sequence[Mapping[str, Any]],
    starting_player_ids: Sequence[Any],
    bench_order: Sequence[Any],
    captain_player_id: Optional[Any],
    vice_captain_player_id: Optional[Any],
    declared_bank_units: int,
    budget_limit_units: Optional[int] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the target-season policy and validate one complete plan."""

    engine = SquadLegalityEngine.from_season(
        target_season,
        config_path=config_path,
    )
    return engine.validate_plan(
        squad_players,
        starting_player_ids=starting_player_ids,
        bench_order=bench_order,
        captain_player_id=captain_player_id,
        vice_captain_player_id=vice_captain_player_id,
        declared_bank_units=declared_bank_units,
        budget_limit_units=budget_limit_units,
    )
