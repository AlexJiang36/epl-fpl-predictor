from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "fpl_squad_transfer_rules_v1"
VALID_POSITIONS = ("GKP", "DEF", "MID", "FWD")
DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "config" / "fpl"


class SquadTransferRulesError(ValueError):
    """Raised when a squad/transfer rules document or input is invalid."""


class SquadTransferRulesNotFoundError(FileNotFoundError):
    """Raised when no squad/transfer rules document exists for a season."""


@dataclass(frozen=True)
class SquadTransferRules:
    effective_season: str
    rules_version: str
    schema_version: str
    path: Path
    sha256: str
    data: Dict[str, Any]

    @property
    def units(self) -> Dict[str, Any]:
        return self.data["units"]

    @property
    def squad(self) -> Dict[str, Any]:
        return self.data["squad"]

    @property
    def lineup(self) -> Dict[str, Any]:
        return self.data["lineup"]

    @property
    def transfers(self) -> Dict[str, Any]:
        return self.data["transfers"]

    @property
    def initial_budget_units(self) -> int:
        return int(self.units["initial_budget_units"])

    @property
    def position_quotas(self) -> Dict[str, int]:
        return {
            position: int(self.squad["position_quotas"][position])
            for position in VALID_POSITIONS
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SquadTransferRulesError("%s must be an object." % label)
    return value


def require_list(value: Any, label: str) -> List[Any]:
    if not isinstance(value, list):
        raise SquadTransferRulesError("%s must be a list." % label)
    return value


def require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SquadTransferRulesError("%s must be a non-empty string." % label)
    return value.strip()


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SquadTransferRulesError("%s must be boolean." % label)
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SquadTransferRulesError("%s must be an integer." % label)
    return value


def require_nonnegative_int(value: Any, label: str) -> int:
    parsed = require_int(value, label)
    if parsed < 0:
        raise SquadTransferRulesError("%s must be non-negative." % label)
    return parsed


def require_positive_int(value: Any, label: str) -> int:
    parsed = require_int(value, label)
    if parsed <= 0:
        raise SquadTransferRulesError("%s must be positive." % label)
    return parsed


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SquadTransferRulesError("%s must be numeric." % label)
    return float(value)


def require_exact_keys(
    mapping: Mapping[str, Any],
    expected_keys: Sequence[str],
    label: str,
) -> None:
    expected = set(expected_keys)
    actual = set(mapping.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: List[str] = []
        if missing:
            details.append("missing=%s" % missing)
        if extra:
            details.append("extra=%s" % extra)
        raise SquadTransferRulesError(
            "%s keys are invalid (%s)." % (label, "; ".join(details))
        )


def validate_season_format(season: str) -> None:
    if re.fullmatch(r"\d{4}_\d{2}", season) is None:
        raise SquadTransferRulesError(
            "effective_season must use YYYY_YY format, for example 2025_26."
        )
    start_year = int(season[:4])
    expected_suffix = str((start_year + 1) % 100).zfill(2)
    if season[-2:] != expected_suffix:
        raise SquadTransferRulesError(
            "effective_season %s is not a consecutive season label." % season
        )


def validate_position_counts(value: Any, label: str) -> Dict[str, int]:
    mapping = require_mapping(value, label)
    require_exact_keys(mapping, VALID_POSITIONS, label)
    return {
        position: require_nonnegative_int(
            mapping[position], "%s.%s" % (label, position)
        )
        for position in VALID_POSITIONS
    }


def validate_rules_document(
    document: Mapping[str, Any],
    expected_season: Optional[str] = None,
) -> None:
    required_top_level = {
        "schema_version",
        "competition",
        "effective_season",
        "rules_version",
        "status",
        "source_notes",
        "positions",
        "units",
        "squad",
        "lineup",
        "transfers",
        "deterministic_examples",
    }
    missing = sorted(required_top_level - set(document.keys()))
    if missing:
        raise SquadTransferRulesError(
            "Squad/transfer rules document is missing top-level keys: %s." % missing
        )

    schema_version = require_nonempty_string(
        document["schema_version"], "schema_version"
    )
    if schema_version != SCHEMA_VERSION:
        raise SquadTransferRulesError(
            "Unsupported schema_version=%s; expected %s."
            % (schema_version, SCHEMA_VERSION)
        )

    if require_nonempty_string(document["competition"], "competition") != "FPL":
        raise SquadTransferRulesError("competition must be FPL.")

    season = require_nonempty_string(document["effective_season"], "effective_season")
    validate_season_format(season)
    if expected_season is not None and season != expected_season:
        raise SquadTransferRulesError(
            "Rules effective_season=%s does not match requested season=%s."
            % (season, expected_season)
        )

    require_nonempty_string(document["rules_version"], "rules_version")
    require_nonempty_string(document["status"], "status")
    require_list(document["source_notes"], "source_notes")

    positions = require_list(document["positions"], "positions")
    if positions != list(VALID_POSITIONS):
        raise SquadTransferRulesError(
            "positions must be exactly %s in canonical order."
            % (list(VALID_POSITIONS),)
        )

    units = require_mapping(document["units"], "units")
    required_unit_keys = {
        "price_unit",
        "one_price_unit_gbp_millions",
        "initial_budget_units",
    }
    missing_units = sorted(required_unit_keys - set(units.keys()))
    if missing_units:
        raise SquadTransferRulesError("units is missing keys: %s." % missing_units)
    require_nonempty_string(units["price_unit"], "units.price_unit")
    unit_value = require_number(
        units["one_price_unit_gbp_millions"],
        "units.one_price_unit_gbp_millions",
    )
    if abs(unit_value - 0.1) > 1e-12:
        raise SquadTransferRulesError(
            "one_price_unit_gbp_millions must be 0.1 for FPL tenths pricing."
        )
    require_positive_int(units["initial_budget_units"], "units.initial_budget_units")

    squad = require_mapping(document["squad"], "squad")
    required_squad_keys = {
        "size",
        "position_quotas",
        "max_players_per_club",
        "duplicate_players_allowed",
    }
    missing_squad = sorted(required_squad_keys - set(squad.keys()))
    if missing_squad:
        raise SquadTransferRulesError("squad is missing keys: %s." % missing_squad)
    squad_size = require_positive_int(squad["size"], "squad.size")
    quotas = validate_position_counts(
        squad["position_quotas"], "squad.position_quotas"
    )
    if sum(quotas.values()) != squad_size:
        raise SquadTransferRulesError(
            "squad.position_quotas must sum to squad.size."
        )
    require_positive_int(
        squad["max_players_per_club"], "squad.max_players_per_club"
    )
    if require_bool(
        squad["duplicate_players_allowed"], "squad.duplicate_players_allowed"
    ):
        raise SquadTransferRulesError(
            "FPL squad policy must not allow duplicate players."
        )

    lineup = require_mapping(document["lineup"], "lineup")
    required_lineup_keys = {
        "starting_size",
        "bench_size",
        "position_bounds",
        "bench",
        "captain_required",
        "vice_captain_required",
        "captain_and_vice_must_differ",
    }
    missing_lineup = sorted(required_lineup_keys - set(lineup.keys()))
    if missing_lineup:
        raise SquadTransferRulesError("lineup is missing keys: %s." % missing_lineup)
    starting_size = require_positive_int(
        lineup["starting_size"], "lineup.starting_size"
    )
    bench_size = require_nonnegative_int(lineup["bench_size"], "lineup.bench_size")
    if starting_size + bench_size != squad_size:
        raise SquadTransferRulesError(
            "lineup.starting_size + lineup.bench_size must equal squad.size."
        )

    bounds = require_mapping(lineup["position_bounds"], "lineup.position_bounds")
    require_exact_keys(bounds, VALID_POSITIONS, "lineup.position_bounds")
    min_total = 0
    max_total = 0
    for position in VALID_POSITIONS:
        position_bounds = require_mapping(
            bounds[position], "lineup.position_bounds.%s" % position
        )
        require_exact_keys(
            position_bounds,
            ("min", "max"),
            "lineup.position_bounds.%s" % position,
        )
        minimum = require_nonnegative_int(
            position_bounds["min"],
            "lineup.position_bounds.%s.min" % position,
        )
        maximum = require_nonnegative_int(
            position_bounds["max"],
            "lineup.position_bounds.%s.max" % position,
        )
        if minimum > maximum:
            raise SquadTransferRulesError(
                "lineup position min cannot exceed max for %s." % position
            )
        if maximum > quotas[position]:
            raise SquadTransferRulesError(
                "lineup max for %s cannot exceed squad quota." % position
            )
        min_total += minimum
        max_total += maximum
    if not (min_total <= starting_size <= max_total):
        raise SquadTransferRulesError(
            "lineup position bounds cannot produce starting_size=%s." % starting_size
        )

    bench = require_mapping(lineup["bench"], "lineup.bench")
    require_exact_keys(
        bench, ("goalkeepers", "outfield_players"), "lineup.bench"
    )
    bench_goalkeepers = require_nonnegative_int(
        bench["goalkeepers"], "lineup.bench.goalkeepers"
    )
    bench_outfield = require_nonnegative_int(
        bench["outfield_players"], "lineup.bench.outfield_players"
    )
    if bench_goalkeepers + bench_outfield != bench_size:
        raise SquadTransferRulesError(
            "lineup bench composition must sum to lineup.bench_size."
        )
    for key in (
        "captain_required",
        "vice_captain_required",
        "captain_and_vice_must_differ",
    ):
        require_bool(lineup[key], "lineup.%s" % key)

    transfers = require_mapping(document["transfers"], "transfers")
    required_transfer_keys = {
        "pre_season",
        "weekly",
        "chip_behavior",
        "special_events",
    }
    missing_transfer_keys = sorted(required_transfer_keys - set(transfers.keys()))
    if missing_transfer_keys:
        raise SquadTransferRulesError(
            "transfers is missing keys: %s." % missing_transfer_keys
        )

    pre_season = require_mapping(transfers["pre_season"], "transfers.pre_season")
    require_bool(
        pre_season.get("unlimited_free_transfers"),
        "transfers.pre_season.unlimited_free_transfers",
    )
    require_nonnegative_int(
        pre_season.get("hit_cost_points"),
        "transfers.pre_season.hit_cost_points",
    )

    weekly = require_mapping(transfers["weekly"], "transfers.weekly")
    weekly_required = {
        "free_transfers_accrued_per_gameweek",
        "maximum_stored_free_transfers",
        "hit_cost_points_per_additional_transfer",
        "incoming_count_must_equal_outgoing_count",
    }
    missing_weekly = sorted(weekly_required - set(weekly.keys()))
    if missing_weekly:
        raise SquadTransferRulesError(
            "transfers.weekly is missing keys: %s." % missing_weekly
        )
    accrued = require_nonnegative_int(
        weekly["free_transfers_accrued_per_gameweek"],
        "transfers.weekly.free_transfers_accrued_per_gameweek",
    )
    cap = require_positive_int(
        weekly["maximum_stored_free_transfers"],
        "transfers.weekly.maximum_stored_free_transfers",
    )
    if accrued > cap:
        raise SquadTransferRulesError(
            "weekly free-transfer accrual cannot exceed storage cap."
        )
    require_nonnegative_int(
        weekly["hit_cost_points_per_additional_transfer"],
        "transfers.weekly.hit_cost_points_per_additional_transfer",
    )
    require_bool(
        weekly["incoming_count_must_equal_outgoing_count"],
        "transfers.weekly.incoming_count_must_equal_outgoing_count",
    )

    chip_behavior = require_mapping(
        transfers["chip_behavior"], "transfers.chip_behavior"
    )
    require_exact_keys(chip_behavior, ("wildcard", "free_hit"), "chip_behavior")
    for chip_name in ("wildcard", "free_hit"):
        chip = require_mapping(
            chip_behavior[chip_name],
            "transfers.chip_behavior.%s" % chip_name,
        )
        required_chip_keys = {
            "unlimited_free_transfers",
            "preserve_saved_free_transfers",
            "weekly_accrual_after_chip",
            "remove_transfer_hits",
            "temporary_squad",
        }
        missing_chip = sorted(required_chip_keys - set(chip.keys()))
        if missing_chip:
            raise SquadTransferRulesError(
                "transfers.chip_behavior.%s is missing keys: %s."
                % (chip_name, missing_chip)
            )
        for key in (
            "unlimited_free_transfers",
            "preserve_saved_free_transfers",
            "remove_transfer_hits",
            "temporary_squad",
        ):
            require_bool(
                chip[key], "transfers.chip_behavior.%s.%s" % (chip_name, key)
            )
        require_nonnegative_int(
            chip["weekly_accrual_after_chip"],
            "transfers.chip_behavior.%s.weekly_accrual_after_chip" % chip_name,
        )

    special_events = require_list(
        transfers["special_events"], "transfers.special_events"
    )
    event_ids: List[str] = []
    for index, raw_event in enumerate(special_events):
        label = "transfers.special_events[%s]" % index
        event = require_mapping(raw_event, label)
        required_event_keys = {
            "event_id",
            "after_gameweek",
            "applies_for_gameweek",
            "operation",
            "target_free_transfers",
            "carry_forward",
        }
        missing_event = sorted(required_event_keys - set(event.keys()))
        if missing_event:
            raise SquadTransferRulesError(
                "%s is missing keys: %s." % (label, missing_event)
            )
        event_id = require_nonempty_string(event["event_id"], "%s.event_id" % label)
        if event_id in event_ids:
            raise SquadTransferRulesError(
                "Duplicate transfer special-event id: %s." % event_id
            )
        event_ids.append(event_id)
        after_gw = require_nonnegative_int(
            event["after_gameweek"], "%s.after_gameweek" % label
        )
        applies_gw = require_positive_int(
            event["applies_for_gameweek"], "%s.applies_for_gameweek" % label
        )
        if applies_gw != after_gw + 1:
            raise SquadTransferRulesError(
                "%s must apply to the gameweek immediately after after_gameweek."
                % label
            )
        if require_nonempty_string(event["operation"], "%s.operation" % label) != "top_up_to":
            raise SquadTransferRulesError(
                "%s.operation must currently be top_up_to." % label
            )
        target = require_nonnegative_int(
            event["target_free_transfers"],
            "%s.target_free_transfers" % label,
        )
        if target > cap:
            raise SquadTransferRulesError(
                "%s target_free_transfers cannot exceed weekly storage cap." % label
            )
        require_bool(event["carry_forward"], "%s.carry_forward" % label)

    examples = require_mapping(
        document["deterministic_examples"], "deterministic_examples"
    )
    required_example_keys = {
        "base_squad",
        "squad_cases",
        "lineup_cases",
        "transfer_cases",
        "transfer_legality_cases",
    }
    missing_examples = sorted(required_example_keys - set(examples.keys()))
    if missing_examples:
        raise SquadTransferRulesError(
            "deterministic_examples is missing keys: %s." % missing_examples
        )
    if len(require_list(examples["base_squad"], "deterministic_examples.base_squad")) != squad_size:
        raise SquadTransferRulesError(
            "deterministic_examples.base_squad must contain squad.size players."
        )
    for key in (
        "squad_cases",
        "lineup_cases",
        "transfer_cases",
        "transfer_legality_cases",
    ):
        cases = require_list(examples[key], "deterministic_examples.%s" % key)
        if not cases:
            raise SquadTransferRulesError(
                "deterministic_examples.%s must not be empty." % key
            )


def config_path_for_season(
    season: str,
    config_root: Optional[Path] = None,
) -> Path:
    validate_season_format(season)
    root = config_root or DEFAULT_CONFIG_ROOT
    return root / ("squad_transfer_rules_%s.json" % season)


def load_squad_transfer_rules(
    season: str,
    config_path: Optional[Path] = None,
    config_root: Optional[Path] = None,
) -> SquadTransferRules:
    path = config_path or config_path_for_season(season, config_root=config_root)
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise SquadTransferRulesNotFoundError(
            "No FPL squad/transfer rules registry exists for season=%s at %s."
            % (season, path)
        )
    if not path.is_file():
        raise SquadTransferRulesNotFoundError(
            "Squad/transfer rules path is not a file: %s." % path
        )

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SquadTransferRulesError(
            "Invalid JSON in squad/transfer rules file %s: %s." % (path, exc)
        )

    document_mapping = require_mapping(document, "squad/transfer rules document")
    validate_rules_document(document_mapping, expected_season=season)
    return SquadTransferRules(
        effective_season=str(document_mapping["effective_season"]),
        rules_version=str(document_mapping["rules_version"]),
        schema_version=str(document_mapping["schema_version"]),
        path=path,
        sha256=sha256_file(path),
        data=dict(document_mapping),
    )


def normalize_player(player: Mapping[str, Any], label: str) -> Dict[str, Any]:
    mapping = require_mapping(player, label)
    required = {"player_id", "position", "club_id", "price_units"}
    missing = sorted(required - set(mapping.keys()))
    if missing:
        raise SquadTransferRulesError("%s is missing keys: %s." % (label, missing))
    player_id = mapping["player_id"]
    if isinstance(player_id, bool) or not isinstance(player_id, (int, str)):
        raise SquadTransferRulesError(
            "%s.player_id must be an integer or non-empty string." % label
        )
    if isinstance(player_id, str) and not player_id.strip():
        raise SquadTransferRulesError("%s.player_id must not be empty." % label)
    position = require_nonempty_string(mapping["position"], "%s.position" % label)
    if position not in VALID_POSITIONS:
        raise SquadTransferRulesError(
            "%s.position=%s is invalid; expected one of %s."
            % (label, position, VALID_POSITIONS)
        )
    club_id = mapping["club_id"]
    if isinstance(club_id, bool) or not isinstance(club_id, (int, str)):
        raise SquadTransferRulesError(
            "%s.club_id must be an integer or non-empty string." % label
        )
    if isinstance(club_id, str) and not club_id.strip():
        raise SquadTransferRulesError("%s.club_id must not be empty." % label)
    price_units = require_nonnegative_int(
        mapping["price_units"], "%s.price_units" % label
    )
    return {
        "player_id": player_id,
        "position": position,
        "club_id": club_id,
        "price_units": price_units,
    }


def normalize_players(players: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(players, (str, bytes)) or not isinstance(players, Sequence):
        raise SquadTransferRulesError("players must be a sequence of objects.")
    return [
        normalize_player(player, "players[%s]" % index)
        for index, player in enumerate(players)
    ]


def count_positions(players: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {position: 0 for position in VALID_POSITIONS}
    for player in players:
        counts[str(player["position"])] += 1
    return counts


def count_clubs(players: Sequence[Mapping[str, Any]]) -> Dict[Any, int]:
    counts: Dict[Any, int] = {}
    for player in players:
        club_id = player["club_id"]
        counts[club_id] = counts.get(club_id, 0) + 1
    return counts


def validate_squad(
    rules: SquadTransferRules,
    players: Sequence[Mapping[str, Any]],
    budget_limit_units: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate a full FPL squad using only the loaded versioned policy."""

    normalized = normalize_players(players)
    errors: List[str] = []
    squad_policy = rules.squad
    expected_size = int(squad_policy["size"])
    budget_limit = (
        rules.initial_budget_units
        if budget_limit_units is None
        else require_nonnegative_int(budget_limit_units, "budget_limit_units")
    )

    player_ids = [player["player_id"] for player in normalized]
    if len(normalized) != expected_size:
        errors.append(
            "squad_size_invalid: expected %s, got %s"
            % (expected_size, len(normalized))
        )
    if len(set(player_ids)) != len(player_ids):
        errors.append("duplicate_player_ids")

    actual_positions = count_positions(normalized)
    expected_positions = rules.position_quotas
    if actual_positions != expected_positions:
        errors.append(
            "position_quotas_invalid: expected %s, got %s"
            % (expected_positions, actual_positions)
        )

    club_counts = count_clubs(normalized)
    max_per_club = int(squad_policy["max_players_per_club"])
    club_violations = {
        str(club_id): count
        for club_id, count in club_counts.items()
        if count > max_per_club
    }
    if club_violations:
        errors.append(
            "club_limit_exceeded: max=%s violations=%s"
            % (max_per_club, club_violations)
        )

    total_price_units = sum(int(player["price_units"]) for player in normalized)
    if total_price_units > budget_limit:
        errors.append(
            "budget_exceeded: total_price_units=%s budget_limit_units=%s"
            % (total_price_units, budget_limit)
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "player_count": len(normalized),
        "unique_player_count": len(set(player_ids)),
        "position_counts": actual_positions,
        "club_counts": {str(key): value for key, value in club_counts.items()},
        "total_price_units": total_price_units,
        "budget_limit_units": budget_limit,
        "remaining_budget_units": budget_limit - total_price_units,
    }


def validate_lineup(
    rules: SquadTransferRules,
    squad_players: Sequence[Mapping[str, Any]],
    starting_player_ids: Sequence[Any],
    bench_order: Sequence[Any],
    captain_player_id: Optional[Any],
    vice_captain_player_id: Optional[Any],
) -> Dict[str, Any]:
    """Validate starting XI, bench, captain and vice-captain from a full squad."""

    normalized = normalize_players(squad_players)
    errors: List[str] = []
    lineup_policy = rules.lineup

    squad_by_id = {player["player_id"]: player for player in normalized}
    squad_ids = list(squad_by_id.keys())
    starting_ids = list(starting_player_ids)
    bench_ids = list(bench_order)

    if len(starting_ids) != int(lineup_policy["starting_size"]):
        errors.append(
            "starting_size_invalid: expected %s, got %s"
            % (lineup_policy["starting_size"], len(starting_ids))
        )
    if len(bench_ids) != int(lineup_policy["bench_size"]):
        errors.append(
            "bench_size_invalid: expected %s, got %s"
            % (lineup_policy["bench_size"], len(bench_ids))
        )
    if len(set(starting_ids)) != len(starting_ids):
        errors.append("duplicate_starting_player_ids")
    if len(set(bench_ids)) != len(bench_ids):
        errors.append("duplicate_bench_player_ids")
    if set(starting_ids) & set(bench_ids):
        errors.append("starting_and_bench_overlap")

    unknown_starting = [player_id for player_id in starting_ids if player_id not in squad_by_id]
    unknown_bench = [player_id for player_id in bench_ids if player_id not in squad_by_id]
    if unknown_starting:
        errors.append("starting_players_not_in_squad: %s" % unknown_starting)
    if unknown_bench:
        errors.append("bench_players_not_in_squad: %s" % unknown_bench)

    if set(starting_ids) | set(bench_ids) != set(squad_ids):
        errors.append("starting_and_bench_do_not_partition_squad")

    starting_players = [
        squad_by_id[player_id]
        for player_id in starting_ids
        if player_id in squad_by_id
    ]
    starting_counts = count_positions(starting_players)
    bounds = lineup_policy["position_bounds"]
    for position in VALID_POSITIONS:
        minimum = int(bounds[position]["min"])
        maximum = int(bounds[position]["max"])
        actual = starting_counts[position]
        if actual < minimum or actual > maximum:
            errors.append(
                "formation_invalid_%s: expected %s-%s, got %s"
                % (position, minimum, maximum, actual)
            )

    bench_players = [
        squad_by_id[player_id]
        for player_id in bench_ids
        if player_id in squad_by_id
    ]
    bench_counts = count_positions(bench_players)
    expected_bench_goalkeepers = int(lineup_policy["bench"]["goalkeepers"])
    expected_bench_outfield = int(lineup_policy["bench"]["outfield_players"])
    actual_bench_goalkeepers = bench_counts["GKP"]
    actual_bench_outfield = len(bench_players) - actual_bench_goalkeepers
    if actual_bench_goalkeepers != expected_bench_goalkeepers:
        errors.append(
            "bench_goalkeeper_count_invalid: expected %s, got %s"
            % (expected_bench_goalkeepers, actual_bench_goalkeepers)
        )
    if actual_bench_outfield != expected_bench_outfield:
        errors.append(
            "bench_outfield_count_invalid: expected %s, got %s"
            % (expected_bench_outfield, actual_bench_outfield)
        )

    if bool(lineup_policy["captain_required"]):
        if captain_player_id is None:
            errors.append("captain_missing")
        elif captain_player_id not in set(starting_ids):
            errors.append("captain_not_in_starting_lineup")
    if bool(lineup_policy["vice_captain_required"]):
        if vice_captain_player_id is None:
            errors.append("vice_captain_missing")
        elif vice_captain_player_id not in set(starting_ids):
            errors.append("vice_captain_not_in_starting_lineup")
    if (
        bool(lineup_policy["captain_and_vice_must_differ"])
        and captain_player_id is not None
        and vice_captain_player_id is not None
        and captain_player_id == vice_captain_player_id
    ):
        errors.append("captain_and_vice_captain_must_differ")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "starting_player_count": len(starting_ids),
        "bench_player_count": len(bench_ids),
        "starting_position_counts": starting_counts,
        "bench_position_counts": bench_counts,
        "captain_player_id": captain_player_id,
        "vice_captain_player_id": vice_captain_player_id,
    }


def apply_example_mutations(
    base_players: Sequence[Mapping[str, Any]],
    mutations: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    players = copy.deepcopy(normalize_players(base_players))
    by_id = {player["player_id"]: player for player in players}
    for index, raw_mutation in enumerate(mutations):
        mutation = require_mapping(raw_mutation, "mutations[%s]" % index)
        operation = require_nonempty_string(
            mutation.get("operation"), "mutations[%s].operation" % index
        )
        player_id = mutation.get("player_id")
        if player_id not in by_id:
            raise SquadTransferRulesError(
                "Mutation references unknown player_id=%r." % player_id
            )
        if operation == "set_price_units":
            by_id[player_id]["price_units"] = require_nonnegative_int(
                mutation.get("value"), "mutation.value"
            )
        elif operation == "set_club_id":
            value = mutation.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise SquadTransferRulesError(
                    "set_club_id mutation value must be integer or string."
                )
            by_id[player_id]["club_id"] = value
        elif operation == "set_position":
            value = require_nonempty_string(mutation.get("value"), "mutation.value")
            if value not in VALID_POSITIONS:
                raise SquadTransferRulesError(
                    "set_position mutation value is invalid: %s." % value
                )
            by_id[player_id]["position"] = value
        elif operation == "set_player_id":
            by_id[player_id]["player_id"] = mutation.get("value")
        else:
            raise SquadTransferRulesError(
                "Unsupported deterministic-example mutation operation=%s." % operation
            )
    return players


def validate_squad_examples(rules: SquadTransferRules) -> Dict[str, Any]:
    examples = rules.data["deterministic_examples"]
    base_squad = examples["base_squad"]
    results: List[Dict[str, Any]] = []
    for raw_case in examples["squad_cases"]:
        case = require_mapping(raw_case, "squad deterministic case")
        name = require_nonempty_string(case.get("name"), "squad case name")
        mutations = require_list(case.get("mutations", []), "%s.mutations" % name)
        expected_valid = require_bool(
            case.get("expected_valid"), "%s.expected_valid" % name
        )
        players = apply_example_mutations(base_squad, mutations)
        result = validate_squad(rules, players)
        passed = bool(result["valid"]) == expected_valid
        results.append(
            {
                "name": name,
                "expected_valid": expected_valid,
                "actual_valid": bool(result["valid"]),
                "passed": passed,
                "errors": result["errors"],
            }
        )
    return {
        "case_count": len(results),
        "passed_count": sum(1 for result in results if result["passed"]),
        "all_passed": all(result["passed"] for result in results),
        "cases": results,
    }


def validate_lineup_examples(rules: SquadTransferRules) -> Dict[str, Any]:
    examples = rules.data["deterministic_examples"]
    base_squad = examples["base_squad"]
    results: List[Dict[str, Any]] = []
    for raw_case in examples["lineup_cases"]:
        case = require_mapping(raw_case, "lineup deterministic case")
        name = require_nonempty_string(case.get("name"), "lineup case name")
        expected_valid = require_bool(
            case.get("expected_valid"), "%s.expected_valid" % name
        )
        result = validate_lineup(
            rules=rules,
            squad_players=base_squad,
            starting_player_ids=require_list(
                case.get("starting_player_ids"), "%s.starting_player_ids" % name
            ),
            bench_order=require_list(
                case.get("bench_order"), "%s.bench_order" % name
            ),
            captain_player_id=case.get("captain_player_id"),
            vice_captain_player_id=case.get("vice_captain_player_id"),
        )
        passed = bool(result["valid"]) == expected_valid
        results.append(
            {
                "name": name,
                "expected_valid": expected_valid,
                "actual_valid": bool(result["valid"]),
                "passed": passed,
                "errors": result["errors"],
            }
        )
    return {
        "case_count": len(results),
        "passed_count": sum(1 for result in results if result["passed"]),
        "all_passed": all(result["passed"] for result in results),
        "cases": results,
    }
