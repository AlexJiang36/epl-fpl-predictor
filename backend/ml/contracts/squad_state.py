"""Versioned, artifact-first owned squad state for weekly FPL planning.

This module contains state representation, pure validation, lineage, fingerprinting,
and JSON artifact I/O only. It intentionally contains no transfer/chip decision logic
and no database write surface. Monetary values use integer 0.1m units.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.rules.squad import SquadTransferRules, validate_lineup, validate_squad
from app.rules.transfers import maximum_stored_free_transfers
from ml.contracts.gameweek_cycle import GameweekPhase, build_phase_id


CONTRACT_VERSION = "fpl_squad_state_v1"
VALID_STATE_KINDS = ("model_team", "team_alex")
VALID_STATE_STATUSES = ("planning", "frozen")
VALID_POSITIONS = ("GKP", "DEF", "MID", "FWD")
WRITES_DATABASE = False


class SquadStateError(ValueError):
    """Raised when a canonical squad state violates the versioned contract."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SquadStateError("%s must be a non-empty string." % label)
    return value.strip()


def _int(value: Any, label: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SquadStateError("%s must be an integer." % label)
    result = int(value)
    if minimum is not None and result < minimum:
        raise SquadStateError("%s must be at least %s." % (label, minimum))
    return result


def _optional_float(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SquadStateError("%s must be numeric or null." % label)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SquadStateError("%s must be numeric or null." % label)


def _tuple_ids(values: Sequence[Any], label: str, expected_size: Optional[int] = None) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SquadStateError("%s must be a sequence." % label)
    result = tuple(_int(value, "%s[%s]" % (label, index), 1) for index, value in enumerate(values))
    if expected_size is not None and len(result) != expected_size:
        raise SquadStateError("%s must contain exactly %s player IDs." % (label, expected_size))
    if len(set(result)) != len(result):
        raise SquadStateError("%s must not contain duplicate player IDs." % label)
    return result


def _validate_as_of_utc(value: str) -> str:
    raw = _text(value, "as_of_utc")
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise SquadStateError("as_of_utc must be ISO-8601 datetime text.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SquadStateError("as_of_utc must include an explicit timezone offset.")
    if parsed.utcoffset().total_seconds() != 0:
        raise SquadStateError("as_of_utc must be expressed in UTC.")
    return parsed.isoformat().replace("+00:00", "Z")


def normalize_state_kind(value: Any) -> str:
    raw = _text(value, "state_kind").lower()
    if raw not in VALID_STATE_KINDS:
        raise SquadStateError(
            "state_kind=%s is invalid; expected one of %s. shadow_optimal is diagnostic only."
            % (raw, VALID_STATE_KINDS)
        )
    return raw


def normalize_state_status(value: Any) -> str:
    raw = _text(value, "state_status").lower()
    if raw not in VALID_STATE_STATUSES:
        raise SquadStateError(
            "state_status=%s is invalid; expected one of %s."
            % (raw, VALID_STATE_STATUSES)
        )
    return raw


def calculate_selling_price_units(purchase_price_units: int, current_price_units: int) -> int:
    """Return FPL selling value in 0.1m units using half-profit rounding down."""

    purchase = _int(purchase_price_units, "purchase_price_units", 0)
    current = _int(current_price_units, "current_price_units", 0)
    if current <= purchase:
        return current
    return purchase + ((current - purchase) // 2)


@dataclass(frozen=True)
class SquadPlayerState:
    fpl_player_id: int
    position: str
    club_id: int
    purchase_price_units: int
    current_price_units: int
    selling_price_units: int
    player_name: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fpl_player_id", _int(self.fpl_player_id, "fpl_player_id", 1))
        position = _text(self.position, "position").upper()
        if position not in VALID_POSITIONS:
            raise SquadStateError(
                "position=%s is invalid; expected one of %s." % (position, VALID_POSITIONS)
            )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "club_id", _int(self.club_id, "club_id", 1))
        purchase = _int(self.purchase_price_units, "purchase_price_units", 0)
        current = _int(self.current_price_units, "current_price_units", 0)
        selling = _int(self.selling_price_units, "selling_price_units", 0)
        expected = calculate_selling_price_units(purchase, current)
        if selling != expected:
            raise SquadStateError(
                "selling_price_units=%s does not match FPL sale value=%s for player_id=%s."
                % (selling, expected, self.fpl_player_id)
            )
        object.__setattr__(self, "purchase_price_units", purchase)
        object.__setattr__(self, "current_price_units", current)
        object.__setattr__(self, "selling_price_units", selling)
        if self.player_name is not None:
            object.__setattr__(self, "player_name", _text(self.player_name, "player_name"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SquadSelectionState:
    starting_xi_player_ids: Tuple[int, ...]
    bench_order_player_ids: Tuple[int, ...]
    captain_player_id: int
    vice_captain_player_id: int

    def __post_init__(self) -> None:
        starting = _tuple_ids(self.starting_xi_player_ids, "starting_xi_player_ids", 11)
        bench = _tuple_ids(self.bench_order_player_ids, "bench_order_player_ids", 4)
        captain = _int(self.captain_player_id, "captain_player_id", 1)
        vice = _int(self.vice_captain_player_id, "vice_captain_player_id", 1)
        if set(starting) & set(bench):
            raise SquadStateError("starting XI and bench must not overlap.")
        if captain not in set(starting):
            raise SquadStateError("captain_player_id must be in the starting XI.")
        if vice not in set(starting):
            raise SquadStateError("vice_captain_player_id must be in the starting XI.")
        if captain == vice:
            raise SquadStateError("captain and vice-captain must differ.")
        object.__setattr__(self, "starting_xi_player_ids", starting)
        object.__setattr__(self, "bench_order_player_ids", bench)
        object.__setattr__(self, "captain_player_id", captain)
        object.__setattr__(self, "vice_captain_player_id", vice)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["starting_xi_player_ids"] = list(self.starting_xi_player_ids)
        payload["bench_order_player_ids"] = list(self.bench_order_player_ids)
        return payload


@dataclass(frozen=True)
class ChipInventoryEntry:
    chip_id: str
    remaining: int
    available_now: int
    window_id: Optional[str] = None

    def __post_init__(self) -> None:
        chip_id = _text(self.chip_id, "chip_id").lower()
        remaining = _int(self.remaining, "%s.remaining" % chip_id, 0)
        available = _int(self.available_now, "%s.available_now" % chip_id, 0)
        if available > remaining:
            raise SquadStateError(
                "chip %s available_now cannot exceed remaining inventory." % chip_id
            )
        object.__setattr__(self, "chip_id", chip_id)
        object.__setattr__(self, "remaining", remaining)
        object.__setattr__(self, "available_now", available)
        if self.window_id is not None:
            object.__setattr__(self, "window_id", _text(self.window_id, "window_id"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChipInventoryState:
    as_of_gameweek: int
    entries: Tuple[ChipInventoryEntry, ...]

    def __post_init__(self) -> None:
        gameweek = _int(self.as_of_gameweek, "chip_inventory.as_of_gameweek", 1)
        entries = tuple(self.entries)
        if any(not isinstance(entry, ChipInventoryEntry) for entry in entries):
            raise SquadStateError("chip_inventory.entries must contain ChipInventoryEntry values.")
        ids = [entry.chip_id for entry in entries]
        if len(set(ids)) != len(ids):
            raise SquadStateError("chip_inventory contains duplicate chip IDs.")
        object.__setattr__(self, "as_of_gameweek", gameweek)
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda entry: entry.chip_id)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_gameweek": self.as_of_gameweek,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class FreeTransferState:
    available_for_gameweek: int
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "available_for_gameweek",
            _int(self.available_for_gameweek, "free_transfers.available_for_gameweek", 1),
        )
        object.__setattr__(self, "count", _int(self.count, "free_transfers.count", 0))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SquadStatePredecessor:
    state_id: str
    owned_state_fingerprint: str
    season: str
    gameweek: int
    state_kind: str
    source_phase_id: str
    immutable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _text(self.state_id, "predecessor.state_id"))
        fingerprint = _text(
            self.owned_state_fingerprint,
            "predecessor.owned_state_fingerprint",
        ).lower()
        if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
            raise SquadStateError("predecessor.owned_state_fingerprint must be a SHA256 hex digest.")
        object.__setattr__(self, "owned_state_fingerprint", fingerprint)
        object.__setattr__(self, "season", _text(self.season, "predecessor.season"))
        object.__setattr__(self, "gameweek", _int(self.gameweek, "predecessor.gameweek", 1))
        object.__setattr__(self, "state_kind", normalize_state_kind(self.state_kind))
        object.__setattr__(self, "source_phase_id", _text(self.source_phase_id, "predecessor.source_phase_id"))
        if not isinstance(self.immutable, bool):
            raise SquadStateError("predecessor.immutable must be boolean.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowOptimalDiagnostic:
    target_gameweek: int
    player_ids: Tuple[int, ...]
    source_run_id: str
    objective_value: Optional[float] = None
    note: str = "Diagnostic only; never owned squad state."
    diagnostic_only: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_gameweek", _int(self.target_gameweek, "shadow_optimal.target_gameweek", 1))
        object.__setattr__(self, "player_ids", _tuple_ids(self.player_ids, "shadow_optimal.player_ids", 15))
        object.__setattr__(self, "source_run_id", _text(self.source_run_id, "shadow_optimal.source_run_id"))
        object.__setattr__(self, "objective_value", _optional_float(self.objective_value, "shadow_optimal.objective_value"))
        object.__setattr__(self, "note", _text(self.note, "shadow_optimal.note"))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["player_ids"] = list(self.player_ids)
        return payload


@dataclass(frozen=True)
class SquadState:
    season: str
    gameweek: int
    as_of_utc: str
    state_version: str
    state_kind: str
    state_status: str
    source_phase_id: str
    source_run_id: str
    players: Tuple[SquadPlayerState, ...]
    selection: SquadSelectionState
    bank_units: int
    chip_inventory: ChipInventoryState
    free_transfers: FreeTransferState
    predecessor: Optional[SquadStatePredecessor] = None
    shadow_optimal: Optional[ShadowOptimalDiagnostic] = None
    contract_version: str = CONTRACT_VERSION
    writes_database: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise SquadStateError(
                "contract_version=%s is unsupported; expected %s."
                % (self.contract_version, CONTRACT_VERSION)
            )
        season = _text(self.season, "season")
        gameweek = _int(self.gameweek, "gameweek", 1)
        as_of_utc = _validate_as_of_utc(self.as_of_utc)
        state_version = _text(self.state_version, "state_version")
        state_kind = normalize_state_kind(self.state_kind)
        state_status = normalize_state_status(self.state_status)
        source_phase_id = _text(self.source_phase_id, "source_phase_id")
        source_run_id = _text(self.source_run_id, "source_run_id")
        players = tuple(self.players)
        if any(not isinstance(player, SquadPlayerState) for player in players):
            raise SquadStateError("players must contain SquadPlayerState values.")
        if len(players) != 15:
            raise SquadStateError("players must contain exactly 15 owned players.")
        ids = [player.fpl_player_id for player in players]
        if len(set(ids)) != 15:
            raise SquadStateError("players must contain 15 unique fpl_player_id values.")
        if not isinstance(self.selection, SquadSelectionState):
            raise SquadStateError("selection must be a SquadSelectionState.")
        if set(self.selection.starting_xi_player_ids) | set(self.selection.bench_order_player_ids) != set(ids):
            raise SquadStateError("selection must partition all 15 owned players.")
        bank = _int(self.bank_units, "bank_units", 0)
        if not isinstance(self.chip_inventory, ChipInventoryState):
            raise SquadStateError("chip_inventory must be a ChipInventoryState.")
        if not isinstance(self.free_transfers, FreeTransferState):
            raise SquadStateError("free_transfers must be a FreeTransferState.")
        if self.predecessor is not None and not isinstance(self.predecessor, SquadStatePredecessor):
            raise SquadStateError("predecessor must be a SquadStatePredecessor or null.")
        if self.shadow_optimal is not None and not isinstance(self.shadow_optimal, ShadowOptimalDiagnostic):
            raise SquadStateError("shadow_optimal must be a ShadowOptimalDiagnostic or null.")

        expected_phase = build_phase_id(
            gameweek,
            GameweekPhase.FREEZE.value if state_status == "frozen" else GameweekPhase.PRE.value,
        )
        if source_phase_id != expected_phase:
            raise SquadStateError(
                "source_phase_id=%s does not match state_status=%s expected phase=%s."
                % (source_phase_id, state_status, expected_phase)
            )
        if state_status == "frozen" and self.predecessor is not None and not self.predecessor.immutable:
            raise SquadStateError("A frozen rolling state may not point to a mutable predecessor.")

        object.__setattr__(self, "season", season)
        object.__setattr__(self, "gameweek", gameweek)
        object.__setattr__(self, "as_of_utc", as_of_utc)
        object.__setattr__(self, "state_version", state_version)
        object.__setattr__(self, "state_kind", state_kind)
        object.__setattr__(self, "state_status", state_status)
        object.__setattr__(self, "source_phase_id", source_phase_id)
        object.__setattr__(self, "source_run_id", source_run_id)
        object.__setattr__(self, "players", tuple(sorted(players, key=lambda player: player.fpl_player_id)))
        object.__setattr__(self, "bank_units", bank)

    @property
    def state_id(self) -> str:
        kind = self.state_kind.replace("_", "-").upper()
        return "%s-GW%02d-%s-SQUAD-%s-%s" % (
            self.season,
            self.gameweek,
            kind,
            self.state_status.upper(),
            self.state_version,
        )

    @property
    def immutable(self) -> bool:
        return self.state_status == "frozen"

    @property
    def owned_state_fingerprint(self) -> str:
        payload = owned_state_payload(self)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self, include_shadow_optimal: bool = True) -> Dict[str, Any]:
        payload = owned_state_payload(self)
        payload["state_id"] = self.state_id
        payload["owned_state_fingerprint"] = self.owned_state_fingerprint
        payload["writes_database"] = False
        if include_shadow_optimal:
            payload["shadow_optimal"] = (
                None if self.shadow_optimal is None else self.shadow_optimal.to_dict()
            )
        return payload


@dataclass(frozen=True)
class SquadStateValidationReport:
    valid: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    state_id: str
    owned_state_fingerprint: str
    state_kind: str
    gameweek: int
    structural_squad_valid: bool
    lineup_valid: bool
    chip_inventory_valid: bool
    free_transfers_valid: bool
    predecessor_valid: bool
    ready_for_optimization: bool
    writes_database: bool = field(default=False, init=False)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["errors"] = list(self.errors)
        payload["warnings"] = list(self.warnings)
        return payload


def owned_state_payload(state: SquadState) -> Dict[str, Any]:
    """Canonical owned-state payload. shadow_optimal is deliberately excluded."""

    return {
        "contract_version": state.contract_version,
        "season": state.season,
        "gameweek": state.gameweek,
        "as_of_utc": state.as_of_utc,
        "state_version": state.state_version,
        "state_kind": state.state_kind,
        "state_status": state.state_status,
        "source_phase_id": state.source_phase_id,
        "source_run_id": state.source_run_id,
        "players": [player.to_dict() for player in state.players],
        "selection": state.selection.to_dict(),
        "bank_units": state.bank_units,
        "chip_inventory": state.chip_inventory.to_dict(),
        "free_transfers": state.free_transfers.to_dict(),
        "predecessor": None if state.predecessor is None else state.predecessor.to_dict(),
    }


def predecessor_reference(state: SquadState) -> SquadStatePredecessor:
    if not state.immutable:
        raise SquadStateError("Only a frozen squad state may become an immutable predecessor.")
    return SquadStatePredecessor(
        state_id=state.state_id,
        owned_state_fingerprint=state.owned_state_fingerprint,
        season=state.season,
        gameweek=state.gameweek,
        state_kind=state.state_kind,
        source_phase_id=state.source_phase_id,
        immutable=True,
    )


def validate_predecessor_lineage(state: SquadState) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    predecessor = state.predecessor
    if state.gameweek == 1:
        if predecessor is not None:
            errors.append("GW1 squad state must not have a predecessor.")
        return len(errors) == 0, errors

    if predecessor is None:
        errors.append("GW%s squad state requires the immutable prior-GW predecessor." % state.gameweek)
        return False, errors

    if predecessor.season != state.season:
        errors.append("predecessor season does not match current state season.")
    if predecessor.gameweek != state.gameweek - 1:
        errors.append("predecessor gameweek must equal current gameweek - 1.")
    if predecessor.state_kind != state.state_kind:
        errors.append("predecessor state_kind must match current state_kind.")
    expected_phase = build_phase_id(state.gameweek - 1, GameweekPhase.FREEZE.value)
    if predecessor.source_phase_id != expected_phase:
        errors.append("predecessor must originate from prior-GW FREEZE: expected %s." % expected_phase)
    if not predecessor.immutable:
        errors.append("predecessor must be immutable.")
    return len(errors) == 0, errors


def validate_state_transition(previous: SquadState, current: SquadState) -> Tuple[bool, List[str]]:
    """Validate exact rolling lineage and retained financial identity."""

    errors: List[str] = []
    if not previous.immutable:
        errors.append("previous squad state must be frozen before it can roll forward.")
    if current.gameweek != previous.gameweek + 1:
        errors.append("current gameweek must be previous gameweek + 1.")
    if current.season != previous.season:
        errors.append("season cannot change across a rolling squad-state transition.")
    if current.state_kind != previous.state_kind:
        errors.append("state_kind cannot change across a rolling squad-state transition.")
    if current.predecessor is None:
        errors.append("current state is missing predecessor lineage.")
    else:
        if current.predecessor.state_id != previous.state_id:
            errors.append("current predecessor state_id does not match previous state.")
        if current.predecessor.owned_state_fingerprint != previous.owned_state_fingerprint:
            errors.append("current predecessor fingerprint does not match previous frozen state.")

    previous_players = {player.fpl_player_id: player for player in previous.players}
    current_players = {player.fpl_player_id: player for player in current.players}
    retained_ids = sorted(set(previous_players) & set(current_players))
    for player_id in retained_ids:
        before = previous_players[player_id]
        after = current_players[player_id]
        if after.purchase_price_units != before.purchase_price_units:
            errors.append(
                "retained player_id=%s purchase_price_units changed from %s to %s."
                % (player_id, before.purchase_price_units, after.purchase_price_units)
            )

    # If ownership did not change at all, no transfer occurred, so cash in the bank
    # cannot change merely because market prices moved.
    if set(previous_players) == set(current_players) and current.bank_units != previous.bank_units:
        errors.append(
            "bank_units changed from %s to %s without any owned-player change."
            % (previous.bank_units, current.bank_units)
        )

    return len(errors) == 0, errors


def _rule_players(state: SquadState) -> List[Dict[str, Any]]:
    return [
        {
            "player_id": player.fpl_player_id,
            "position": player.position,
            "club_id": player.club_id,
            "price_units": player.current_price_units,
        }
        for player in state.players
    ]


def validate_squad_state(
    state: SquadState,
    squad_rules: SquadTransferRules,
    chip_rules: Any,
) -> SquadStateValidationReport:
    """Validate one complete manager state before transfer/chip optimization."""

    errors: List[str] = []
    warnings: List[str] = []
    rule_players = _rule_players(state)

    # Ongoing squads may appreciate above the original £100m; use current market
    # value as a neutral budget ceiling so Day74B validates structure/club quotas.
    current_market_value = sum(player.current_price_units for player in state.players)
    squad_validation = validate_squad(
        squad_rules,
        rule_players,
        budget_limit_units=max(current_market_value, squad_rules.initial_budget_units),
    )
    if not squad_validation["valid"]:
        errors.extend("squad:%s" % item for item in squad_validation["errors"])

    lineup_validation = validate_lineup(
        squad_rules,
        rule_players,
        state.selection.starting_xi_player_ids,
        state.selection.bench_order_player_ids,
        state.selection.captain_player_id,
        state.selection.vice_captain_player_id,
    )
    if not lineup_validation["valid"]:
        errors.extend("lineup:%s" % item for item in lineup_validation["errors"])

    free_transfer_valid = True
    expected_ft_gw = state.gameweek + 1 if state.state_status == "frozen" else state.gameweek
    if state.free_transfers.available_for_gameweek != expected_ft_gw:
        free_transfer_valid = False
        errors.append(
            "free_transfers.available_for_gameweek must be %s for state_status=%s."
            % (expected_ft_gw, state.state_status)
        )
    cap = maximum_stored_free_transfers(squad_rules)
    if state.free_transfers.count > cap:
        free_transfer_valid = False
        errors.append(
            "free_transfers.count=%s exceeds storage cap=%s."
            % (state.free_transfers.count, cap)
        )

    chip_valid = True
    if state.chip_inventory.as_of_gameweek != state.gameweek:
        chip_valid = False
        errors.append("chip_inventory.as_of_gameweek must match state.gameweek.")
    if chip_rules is None or not hasattr(chip_rules, "chips"):
        chip_valid = False
        errors.append("versioned chip_rules are required to validate canonical chip inventory.")
    else:
        expected_chip_ids = set(str(key) for key in chip_rules.chips.keys())
        actual_chip_ids = set(entry.chip_id for entry in state.chip_inventory.entries)
        if actual_chip_ids != expected_chip_ids:
            chip_valid = False
            errors.append(
                "chip_inventory chip IDs do not match versioned chip rules: expected=%s actual=%s."
                % (sorted(expected_chip_ids), sorted(actual_chip_ids))
            )
        for entry in state.chip_inventory.entries:
            if entry.chip_id not in chip_rules.chips:
                continue
            limit = int(chip_rules.chips[entry.chip_id].get("inventory_per_window", 1))
            if entry.remaining > limit or entry.available_now > limit:
                chip_valid = False
                errors.append(
                    "chip_inventory %s exceeds inventory_per_window=%s."
                    % (entry.chip_id, limit)
                )

    predecessor_valid, predecessor_errors = validate_predecessor_lineage(state)
    errors.extend("predecessor:%s" % item for item in predecessor_errors)

    if state.shadow_optimal is not None:
        if state.shadow_optimal.target_gameweek != state.gameweek:
            warnings.append(
                "shadow_optimal target_gameweek differs from owned state gameweek; diagnostic remains non-authoritative."
            )

    valid = len(errors) == 0
    return SquadStateValidationReport(
        valid=valid,
        errors=tuple(errors),
        warnings=tuple(warnings),
        state_id=state.state_id,
        owned_state_fingerprint=state.owned_state_fingerprint,
        state_kind=state.state_kind,
        gameweek=state.gameweek,
        structural_squad_valid=bool(squad_validation["valid"]),
        lineup_valid=bool(lineup_validation["valid"]),
        chip_inventory_valid=chip_valid,
        free_transfers_valid=free_transfer_valid,
        predecessor_valid=predecessor_valid,
        ready_for_optimization=valid,
    )


def require_valid_squad_state(
    state: SquadState,
    squad_rules: SquadTransferRules,
    chip_rules: Any,
) -> SquadState:
    report = validate_squad_state(state, squad_rules, chip_rules=chip_rules)
    if not report.valid:
        raise SquadStateError(
            "Squad state is not ready for optimization: %s" % list(report.errors)
        )
    return state


def squad_state_from_mapping(value: Mapping[str, Any]) -> SquadState:
    mapping = dict(value)
    players = tuple(SquadPlayerState(**dict(item)) for item in mapping["players"])
    selection = SquadSelectionState(
        starting_xi_player_ids=tuple(mapping["selection"]["starting_xi_player_ids"]),
        bench_order_player_ids=tuple(mapping["selection"]["bench_order_player_ids"]),
        captain_player_id=mapping["selection"]["captain_player_id"],
        vice_captain_player_id=mapping["selection"]["vice_captain_player_id"],
    )
    chip_inventory = ChipInventoryState(
        as_of_gameweek=mapping["chip_inventory"]["as_of_gameweek"],
        entries=tuple(
            ChipInventoryEntry(**dict(item))
            for item in mapping["chip_inventory"]["entries"]
        ),
    )
    free_transfers = FreeTransferState(**dict(mapping["free_transfers"]))
    predecessor_raw = mapping.get("predecessor")
    predecessor = (
        None
        if predecessor_raw is None
        else SquadStatePredecessor(**dict(predecessor_raw))
    )
    shadow_raw = mapping.get("shadow_optimal")
    shadow = None
    if shadow_raw is not None:
        shadow_payload = dict(shadow_raw)
        shadow_payload.pop("diagnostic_only", None)
        shadow_payload["player_ids"] = tuple(shadow_payload["player_ids"])
        shadow = ShadowOptimalDiagnostic(**shadow_payload)

    state = SquadState(
        season=mapping["season"],
        gameweek=mapping["gameweek"],
        as_of_utc=mapping["as_of_utc"],
        state_version=mapping["state_version"],
        state_kind=mapping["state_kind"],
        state_status=mapping["state_status"],
        source_phase_id=mapping["source_phase_id"],
        source_run_id=mapping["source_run_id"],
        players=players,
        selection=selection,
        bank_units=mapping["bank_units"],
        chip_inventory=chip_inventory,
        free_transfers=free_transfers,
        predecessor=predecessor,
        shadow_optimal=shadow,
        contract_version=mapping.get("contract_version", CONTRACT_VERSION),
    )

    declared_state_id = mapping.get("state_id")
    if declared_state_id is not None and str(declared_state_id) != state.state_id:
        raise SquadStateError(
            "serialized state_id does not match canonical state identity: declared=%s actual=%s."
            % (declared_state_id, state.state_id)
        )

    declared_fingerprint = mapping.get("owned_state_fingerprint")
    if declared_fingerprint is not None:
        declared = str(declared_fingerprint).strip().lower()
        if declared != state.owned_state_fingerprint:
            raise SquadStateError(
                "serialized owned_state_fingerprint does not match canonical owned state."
            )

    if mapping.get("writes_database") not in (None, False):
        raise SquadStateError("serialized squad state may not claim writes_database=true.")

    return state


def load_squad_state_json(path: Path) -> SquadState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SquadStateError("Squad-state JSON must contain one object.")
    return squad_state_from_mapping(payload)


def save_squad_state_json(state: SquadState, path: Path) -> Path:
    """Artifact-only JSON I/O. This contract intentionally has no DB write surface."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output
