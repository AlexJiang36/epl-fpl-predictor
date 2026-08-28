from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from app.rules.squad import (
    SquadTransferRules,
    SquadTransferRulesError,
    load_squad_transfer_rules,
)
from app.rules.transfers import (
    TransferValidationError,
    evaluate_transfer_step,
    maximum_stored_free_transfers,
)


LEDGER_VERSION = "fpl_free_transfer_ledger_v1"
STATE_KINDS: Tuple[str, ...] = ("model_team", "team_alex")
TRANSFER_CHIPS: Tuple[str, ...] = ("wildcard", "free_hit")
TRANSFER_NEUTRAL_CHIPS: Tuple[str, ...] = ("bench_boost", "triple_captain")
VALID_CHIPS: Tuple[str, ...] = TRANSFER_CHIPS + TRANSFER_NEUTRAL_CHIPS


class FreeTransferLedgerError(ValueError):
    """Raised when a free-transfer ledger state or transition is invalid."""


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FreeTransferLedgerError("%s must be an integer." % label)
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    parsed = _require_int(value, label)
    if parsed < 0:
        raise FreeTransferLedgerError("%s must be non-negative." % label)
    return parsed


def _require_gameweek(value: Any, label: str = "gameweek") -> int:
    parsed = _require_int(value, label)
    if parsed < 1 or parsed > 38:
        raise FreeTransferLedgerError("%s must be between 1 and 38." % label)
    return parsed


def _normalize_season(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreeTransferLedgerError("season must be a non-empty string.")
    season = value.strip()
    pieces = season.split("_")
    if (
        len(pieces) != 2
        or len(pieces[0]) != 4
        or len(pieces[1]) != 2
        or not pieces[0].isdigit()
        or not pieces[1].isdigit()
    ):
        raise FreeTransferLedgerError(
            "season must use YYYY_YY format, for example 2026_27."
        )
    return season


def _normalize_state_kind(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreeTransferLedgerError("state_kind must be a non-empty string.")
    normalized = value.strip().lower()
    if normalized not in STATE_KINDS:
        raise FreeTransferLedgerError(
            "state_kind=%s is invalid; expected one of %s."
            % (normalized, STATE_KINDS)
        )
    return normalized


def _normalize_chip(value: Optional[Any]) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    if normalized not in VALID_CHIPS:
        raise FreeTransferLedgerError(
            "chip=%s is invalid; expected one of %s or null."
            % (normalized, VALID_CHIPS)
        )
    return normalized


@dataclass(frozen=True)
class FreeTransferLedgerState:
    """Pure manager-specific FT state at one Gameweek deadline window."""

    season: str
    state_kind: str
    gameweek: int
    available_free_transfers: int
    cumulative_transfer_count: int = 0
    cumulative_free_transfers_used: int = 0
    cumulative_charged_transfers: int = 0
    cumulative_hit_points: int = 0
    transition_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "season", _normalize_season(self.season))
        object.__setattr__(
            self, "state_kind", _normalize_state_kind(self.state_kind)
        )
        object.__setattr__(
            self, "gameweek", _require_gameweek(self.gameweek, "gameweek")
        )
        for field_name in (
            "available_free_transfers",
            "cumulative_transfer_count",
            "cumulative_free_transfers_used",
            "cumulative_charged_transfers",
            "cumulative_hit_points",
            "transition_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonnegative_int(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransferFinancialEffect:
    """Price/bank metadata recorded separately from FT-count accounting."""

    bank_before_units: Optional[int] = None
    bank_after_units: Optional[int] = None
    sales_units: Optional[int] = None
    purchases_units: Optional[int] = None
    persistent: bool = True
    note: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "bank_before_units",
            "bank_after_units",
            "sales_units",
            "purchases_units",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonnegative_int(value, field_name),
                )
        if not isinstance(self.persistent, bool):
            raise FreeTransferLedgerError("persistent must be boolean.")
        if self.note is not None and not isinstance(self.note, str):
            raise FreeTransferLedgerError("note must be a string or null.")

        if (
            self.bank_before_units is not None
            and self.bank_after_units is not None
            and self.sales_units is not None
            and self.purchases_units is not None
        ):
            expected_after = (
                self.bank_before_units
                + self.sales_units
                - self.purchases_units
            )
            if expected_after != self.bank_after_units:
                raise FreeTransferLedgerError(
                    "Financial effect does not reconcile: "
                    "bank_after_units=%s but bank_before_units + sales_units "
                    "- purchases_units=%s."
                    % (self.bank_after_units, expected_after)
                )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if (
            self.bank_before_units is not None
            and self.bank_after_units is not None
        ):
            result["bank_delta_units"] = (
                self.bank_after_units - self.bank_before_units
            )
        else:
            result["bank_delta_units"] = None
        return result


@dataclass(frozen=True)
class FreeTransferTransition:
    ledger_version: str
    season: str
    state_kind: str
    completed_gameweek: int
    next_gameweek: int
    action: str
    chip: Optional[str]
    transfer_count: int
    available_free_transfers_before: int
    free_transfers_used: int
    charged_transfers: int
    hit_points: int
    free_transfers_after_transfers: int
    weekly_accrual_applied: int
    special_event_applied: Optional[str]
    available_free_transfers_next_gameweek: int
    storage_cap: int
    unlimited_free_transfers: bool
    financial_effect: Optional[TransferFinancialEffect]
    next_state: FreeTransferLedgerState
    writes_manager_state: bool = False
    price_bank_effect_applied_to_ledger: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_version": self.ledger_version,
            "season": self.season,
            "state_kind": self.state_kind,
            "completed_gameweek": self.completed_gameweek,
            "next_gameweek": self.next_gameweek,
            "action": self.action,
            "chip": self.chip,
            "transfer_count": self.transfer_count,
            "available_free_transfers_before": (
                self.available_free_transfers_before
            ),
            "free_transfers_used": self.free_transfers_used,
            "charged_transfers": self.charged_transfers,
            "hit_points": self.hit_points,
            "free_transfers_after_transfers": (
                self.free_transfers_after_transfers
            ),
            "weekly_accrual_applied": self.weekly_accrual_applied,
            "special_event_applied": self.special_event_applied,
            "available_free_transfers_next_gameweek": (
                self.available_free_transfers_next_gameweek
            ),
            "storage_cap": self.storage_cap,
            "unlimited_free_transfers": self.unlimited_free_transfers,
            "financial_effect": (
                self.financial_effect.to_dict()
                if self.financial_effect is not None
                else None
            ),
            "next_state": self.next_state.to_dict(),
            "writes_manager_state": self.writes_manager_state,
            "price_bank_effect_applied_to_ledger": (
                self.price_bank_effect_applied_to_ledger
            ),
        }


def load_transfer_policy(season: str) -> SquadTransferRules:
    """Load the versioned target-season squad/transfer policy."""

    normalized = _normalize_season(season)
    try:
        return load_squad_transfer_rules(normalized)
    except SquadTransferRulesError as exc:
        raise FreeTransferLedgerError(str(exc)) from exc


def transfer_policy_summary(rules: SquadTransferRules) -> Dict[str, Any]:
    weekly = rules.transfers["weekly"]
    return {
        "effective_season": rules.effective_season,
        "rules_version": rules.rules_version,
        "schema_version": rules.schema_version,
        "rules_path": str(rules.path),
        "rules_sha256": rules.sha256,
        "free_transfers_accrued_per_gameweek": int(
            weekly["free_transfers_accrued_per_gameweek"]
        ),
        "maximum_stored_free_transfers": int(
            weekly["maximum_stored_free_transfers"]
        ),
        "hit_cost_points_per_additional_transfer": int(
            weekly["hit_cost_points_per_additional_transfer"]
        ),
        "transfer_chip_behavior": {
            chip: dict(rules.transfers["chip_behavior"][chip])
            for chip in TRANSFER_CHIPS
        },
        "special_events": [
            dict(event) for event in rules.transfers["special_events"]
        ],
    }


def build_ledger_state(
    *,
    season: str,
    state_kind: str,
    gameweek: int,
    available_free_transfers: int,
    cumulative_transfer_count: int = 0,
    cumulative_free_transfers_used: int = 0,
    cumulative_charged_transfers: int = 0,
    cumulative_hit_points: int = 0,
    transition_count: int = 0,
    rules: Optional[SquadTransferRules] = None,
) -> FreeTransferLedgerState:
    state = FreeTransferLedgerState(
        season=season,
        state_kind=state_kind,
        gameweek=gameweek,
        available_free_transfers=available_free_transfers,
        cumulative_transfer_count=cumulative_transfer_count,
        cumulative_free_transfers_used=cumulative_free_transfers_used,
        cumulative_charged_transfers=cumulative_charged_transfers,
        cumulative_hit_points=cumulative_hit_points,
        transition_count=transition_count,
    )
    resolved_rules = rules or load_transfer_policy(state.season)
    _validate_state_against_policy(state, resolved_rules)
    return state


def _validate_state_against_policy(
    state: FreeTransferLedgerState,
    rules: SquadTransferRules,
) -> None:
    if rules.effective_season != state.season:
        raise FreeTransferLedgerError(
            "Ledger season=%s does not match transfer policy season=%s."
            % (state.season, rules.effective_season)
        )
    cap = maximum_stored_free_transfers(rules)
    if state.available_free_transfers > cap:
        raise FreeTransferLedgerError(
            "available_free_transfers=%s exceeds target-season storage cap=%s."
            % (state.available_free_transfers, cap)
        )


def _coerce_financial_effect(
    value: Optional[Any],
) -> Optional[TransferFinancialEffect]:
    if value is None:
        return None
    if isinstance(value, TransferFinancialEffect):
        return value
    if not isinstance(value, Mapping):
        raise FreeTransferLedgerError(
            "financial_effect must be a TransferFinancialEffect, mapping, or null."
        )
    return TransferFinancialEffect(
        bank_before_units=value.get("bank_before_units"),
        bank_after_units=value.get("bank_after_units"),
        sales_units=value.get("sales_units"),
        purchases_units=value.get("purchases_units"),
        persistent=value.get("persistent", True),
        note=value.get("note"),
    )


def _action_for(transfer_count: int, chip: Optional[str]) -> str:
    if chip == "wildcard":
        return "WILDCARD"
    if chip == "free_hit":
        return "FREE_HIT"
    return "ROLL" if transfer_count == 0 else "TRANSFER"


def transition_free_transfer_ledger(
    state: FreeTransferLedgerState,
    *,
    transfer_count: int,
    completed_gameweek: Optional[int] = None,
    chip: Optional[str] = None,
    financial_effect: Optional[Any] = None,
    rules: Optional[SquadTransferRules] = None,
) -> FreeTransferTransition:
    """Produce one pure GW-to-GW free-transfer state transition.

    Bank and price metadata may be attached for audit, but it never changes FT
    accounting. No manager state or database row is written.
    """

    count = _require_nonnegative_int(transfer_count, "transfer_count")
    completed = (
        state.gameweek
        if completed_gameweek is None
        else _require_gameweek(completed_gameweek, "completed_gameweek")
    )
    if completed != state.gameweek:
        raise FreeTransferLedgerError(
            "completed_gameweek=%s does not match ledger gameweek=%s."
            % (completed, state.gameweek)
        )
    if completed >= 38:
        raise FreeTransferLedgerError(
            "No next-Gameweek ledger state exists after completed_gameweek=38."
        )

    normalized_chip = _normalize_chip(chip)
    resolved_rules = rules or load_transfer_policy(state.season)
    _validate_state_against_policy(state, resolved_rules)

    effect = _coerce_financial_effect(financial_effect)
    if normalized_chip == "free_hit" and effect is not None and effect.persistent:
        raise FreeTransferLedgerError(
            "Free Hit financial effects are temporary; "
            "financial_effect.persistent must be False."
        )

    # Day74B transfer economics only accepts transfer-affecting chips.
    # Bench Boost and Triple Captain are transfer-neutral and therefore use
    # normal FT/hit accounting while remaining visible in this ledger record.
    transfer_chip = normalized_chip if normalized_chip in TRANSFER_CHIPS else None

    try:
        priced = evaluate_transfer_step(
            rules=resolved_rules,
            transfer_count=count,
            available_free_transfers=state.available_free_transfers,
            completed_gameweek=completed,
            phase="in_season",
            chip=transfer_chip,
        )
    except (TransferValidationError, SquadTransferRulesError) as exc:
        raise FreeTransferLedgerError(str(exc)) from exc

    next_state = FreeTransferLedgerState(
        season=state.season,
        state_kind=state.state_kind,
        gameweek=int(priced["next_gameweek"]),
        available_free_transfers=int(priced["free_transfers_next_gameweek"]),
        cumulative_transfer_count=state.cumulative_transfer_count + count,
        cumulative_free_transfers_used=(
            state.cumulative_free_transfers_used
            + int(priced["free_transfers_used"])
        ),
        cumulative_charged_transfers=(
            state.cumulative_charged_transfers
            + int(priced["charged_transfers"])
        ),
        cumulative_hit_points=(
            state.cumulative_hit_points + int(priced["points_cost"])
        ),
        transition_count=state.transition_count + 1,
    )
    _validate_state_against_policy(next_state, resolved_rules)

    return FreeTransferTransition(
        ledger_version=LEDGER_VERSION,
        season=state.season,
        state_kind=state.state_kind,
        completed_gameweek=completed,
        next_gameweek=int(priced["next_gameweek"]),
        action=_action_for(count, normalized_chip),
        chip=normalized_chip,
        transfer_count=count,
        available_free_transfers_before=int(
            priced["available_free_transfers_before"]
        ),
        free_transfers_used=int(priced["free_transfers_used"]),
        charged_transfers=int(priced["charged_transfers"]),
        hit_points=int(priced["points_cost"]),
        free_transfers_after_transfers=int(
            priced["free_transfers_after_transfers"]
        ),
        weekly_accrual_applied=int(priced["weekly_accrual_applied"]),
        special_event_applied=priced["special_event_applied"],
        available_free_transfers_next_gameweek=int(
            priced["free_transfers_next_gameweek"]
        ),
        storage_cap=int(priced["storage_cap"]),
        unlimited_free_transfers=bool(priced["unlimited_free_transfers"]),
        financial_effect=effect,
        next_state=next_state,
        writes_manager_state=False,
        price_bank_effect_applied_to_ledger=False,
    )


def run_ledger_scenario(
    initial_state: FreeTransferLedgerState,
    steps: Sequence[Mapping[str, Any]],
    *,
    rules: Optional[SquadTransferRules] = None,
) -> Dict[str, Any]:
    """Run a pure multi-GW scenario without persisting manager state."""

    resolved_rules = rules or load_transfer_policy(initial_state.season)
    _validate_state_against_policy(initial_state, resolved_rules)

    current = initial_state
    transitions = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise FreeTransferLedgerError("steps[%s] must be a mapping." % index)
        if "transfer_count" not in step:
            raise FreeTransferLedgerError(
                "steps[%s] is missing transfer_count." % index
            )
        transition = transition_free_transfer_ledger(
            current,
            transfer_count=step["transfer_count"],
            completed_gameweek=step.get("completed_gameweek"),
            chip=step.get("chip"),
            financial_effect=step.get("financial_effect"),
            rules=resolved_rules,
        )
        transitions.append(transition)
        current = transition.next_state

    return {
        "ledger_version": LEDGER_VERSION,
        "season": initial_state.season,
        "state_kind": initial_state.state_kind,
        "policy": transfer_policy_summary(resolved_rules),
        "initial_state": initial_state.to_dict(),
        "transition_count": len(transitions),
        "transitions": [item.to_dict() for item in transitions],
        "final_state": current.to_dict(),
        "writes_manager_state": False,
    }
