from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.rules.squad import (
    require_bool,
    require_exact_keys,
    require_int,
    require_list,
    require_mapping,
    require_nonempty_string,
    require_nonnegative_int,
    require_positive_int,
    validate_season_format,
)


SCHEMA_VERSION = "fpl_chip_rules_v1"
VALID_CHIP_IDS = (
    "wildcard",
    "free_hit",
    "triple_captain",
    "bench_boost",
)
VALID_WINDOW_IDS = ("first_half", "second_half")
VALID_RESET_BEHAVIORS = ("expire_unused", "end_of_season")
VALID_ACTIVATION_SURFACES = ("confirm_transfers", "save_pick_team")
VALIDATION_VERSION = "day75a_v1"

DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "config" / "fpl"
APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_SOURCES = (
    APP_ROOT / "api" / "routes" / "chips.py",
    APP_ROOT / "utils" / "bench_boost_scenario.py",
    APP_ROOT / "utils" / "triple_captain_scenario.py",
    APP_ROOT / "utils" / "wildcard_builder.py",
)


class ChipRulesError(ValueError):
    """Raised when a chip policy document or chip state is invalid."""


class ChipRulesNotFoundError(FileNotFoundError):
    """Raised when no chip policy document exists for a season."""


@dataclass(frozen=True)
class ChipRules:
    effective_season: str
    rules_version: str
    schema_version: str
    path: Path
    sha256: str
    data: Dict[str, Any]

    @property
    def season(self) -> Dict[str, Any]:
        return self.data["season"]

    @property
    def global_rules(self) -> Dict[str, Any]:
        return self.data["global_rules"]

    @property
    def inventory_windows(self) -> List[Dict[str, Any]]:
        return self.data["inventory_windows"]

    @property
    def chips(self) -> Dict[str, Dict[str, Any]]:
        return self.data["chips"]

    @property
    def cross_registry_contract(self) -> Dict[str, Any]:
        return self.data["cross_registry_contract"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_path_for_season(
    season: str,
    config_root: Optional[Path] = None,
) -> Path:
    validate_season_format(season)
    root = Path(config_root) if config_root is not None else DEFAULT_CONFIG_ROOT
    return root / ("chip_rules_%s.json" % season)


def normalize_chip_id(chip_id: str) -> str:
    normalized = require_nonempty_string(chip_id, "chip_id").lower()
    if normalized not in VALID_CHIP_IDS:
        raise ChipRulesError(
            "chip_id=%s is unsupported; expected one of %s."
            % (normalized, VALID_CHIP_IDS)
        )
    return normalized


def validate_gameweek(value: Any, label: str, first_gw: int, last_gw: int) -> int:
    gameweek = require_int(value, label)
    if gameweek < first_gw or gameweek > last_gw:
        raise ChipRulesError(
            "%s=%s is outside the supported Gameweek range %s-%s."
            % (label, gameweek, first_gw, last_gw)
        )
    return gameweek


def _window_map(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    windows = require_list(document["inventory_windows"], "inventory_windows")
    mapped: Dict[str, Dict[str, Any]] = {}
    for index, raw_window in enumerate(windows):
        window = require_mapping(
            raw_window, "inventory_windows[%s]" % index
        )
        window_id = require_nonempty_string(
            window.get("window_id"),
            "inventory_windows[%s].window_id" % index,
        )
        if window_id in mapped:
            raise ChipRulesError("Duplicate inventory window_id=%s." % window_id)
        mapped[window_id] = dict(window)
    return mapped


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
        "season",
        "global_rules",
        "inventory_windows",
        "chips",
        "cross_registry_contract",
        "deterministic_examples",
    }
    missing = sorted(required_top_level - set(document.keys()))
    if missing:
        raise ChipRulesError(
            "Chip rules document is missing top-level keys: %s." % missing
        )

    schema_version = require_nonempty_string(
        document["schema_version"], "schema_version"
    )
    if schema_version != SCHEMA_VERSION:
        raise ChipRulesError(
            "Unsupported schema_version=%s; expected %s."
            % (schema_version, SCHEMA_VERSION)
        )

    if require_nonempty_string(document["competition"], "competition") != "FPL":
        raise ChipRulesError("competition must be FPL.")

    season_label = require_nonempty_string(
        document["effective_season"], "effective_season"
    )
    validate_season_format(season_label)
    if expected_season is not None and season_label != expected_season:
        raise ChipRulesError(
            "Rules effective_season=%s does not match requested season=%s."
            % (season_label, expected_season)
        )

    require_nonempty_string(document["rules_version"], "rules_version")
    require_nonempty_string(document["status"], "status")
    source_notes = require_list(document["source_notes"], "source_notes")
    if not source_notes:
        raise ChipRulesError("source_notes must not be empty.")
    for index, note in enumerate(source_notes):
        require_nonempty_string(note, "source_notes[%s]" % index)

    season = require_mapping(document["season"], "season")
    require_exact_keys(
        season,
        (
            "first_gameweek",
            "last_gameweek",
            "split_after_gameweek",
            "second_half_starts_gameweek",
        ),
        "season",
    )
    first_gw = require_positive_int(season["first_gameweek"], "season.first_gameweek")
    last_gw = require_positive_int(season["last_gameweek"], "season.last_gameweek")
    split_after = require_positive_int(
        season["split_after_gameweek"], "season.split_after_gameweek"
    )
    second_starts = require_positive_int(
        season["second_half_starts_gameweek"],
        "season.second_half_starts_gameweek",
    )
    if first_gw != 1:
        raise ChipRulesError("season.first_gameweek must be 1.")
    if last_gw < first_gw:
        raise ChipRulesError("season.last_gameweek must be >= first_gameweek.")
    if split_after + 1 != second_starts:
        raise ChipRulesError(
            "second_half_starts_gameweek must equal split_after_gameweek + 1."
        )
    if second_starts > last_gw:
        raise ChipRulesError("second half must start within the season.")

    global_rules = require_mapping(document["global_rules"], "global_rules")
    require_exact_keys(
        global_rules,
        (
            "maximum_active_chips_per_gameweek",
            "unused_first_half_inventory_carries_to_second_half",
            "refresh_inventory_at_second_half",
            "unsupported_chip_ids",
        ),
        "global_rules",
    )
    max_active = require_positive_int(
        global_rules["maximum_active_chips_per_gameweek"],
        "global_rules.maximum_active_chips_per_gameweek",
    )
    if max_active != 1:
        raise ChipRulesError(
            "FPL chip policy must allow exactly one active chip per Gameweek."
        )
    if require_bool(
        global_rules["unused_first_half_inventory_carries_to_second_half"],
        "global_rules.unused_first_half_inventory_carries_to_second_half",
    ):
        raise ChipRulesError(
            "2025/26 first-half chip inventory must not carry into the second half."
        )
    if not require_bool(
        global_rules["refresh_inventory_at_second_half"],
        "global_rules.refresh_inventory_at_second_half",
    ):
        raise ChipRulesError("2025/26 chip inventory must refresh at Gameweek 20.")
    unsupported = require_list(
        global_rules["unsupported_chip_ids"],
        "global_rules.unsupported_chip_ids",
    )
    if "assistant_manager" not in unsupported:
        raise ChipRulesError(
            "2025/26 unsupported_chip_ids must include assistant_manager."
        )

    windows = _window_map(document)
    if set(windows.keys()) != set(VALID_WINDOW_IDS):
        raise ChipRulesError(
            "inventory_windows must define exactly %s." % (VALID_WINDOW_IDS,)
        )

    for window_id in VALID_WINDOW_IDS:
        window = windows[window_id]
        require_exact_keys(
            window,
            (
                "window_id",
                "available_from_gameweek",
                "available_through_gameweek",
                "expires_after_gameweek",
                "reset_behavior",
            ),
            "inventory_windows.%s" % window_id,
        )
        start = validate_gameweek(
            window["available_from_gameweek"],
            "inventory_windows.%s.available_from_gameweek" % window_id,
            first_gw,
            last_gw,
        )
        end = validate_gameweek(
            window["available_through_gameweek"],
            "inventory_windows.%s.available_through_gameweek" % window_id,
            first_gw,
            last_gw,
        )
        expiry = validate_gameweek(
            window["expires_after_gameweek"],
            "inventory_windows.%s.expires_after_gameweek" % window_id,
            first_gw,
            last_gw,
        )
        if start > end:
            raise ChipRulesError(
                "inventory window %s starts after it ends." % window_id
            )
        if expiry != end:
            raise ChipRulesError(
                "inventory window %s expiry must equal its final Gameweek."
                % window_id
            )
        reset_behavior = require_nonempty_string(
            window["reset_behavior"],
            "inventory_windows.%s.reset_behavior" % window_id,
        )
        if reset_behavior not in VALID_RESET_BEHAVIORS:
            raise ChipRulesError(
                "inventory window %s has invalid reset_behavior=%s."
                % (window_id, reset_behavior)
            )

    first_window = windows["first_half"]
    second_window = windows["second_half"]
    if int(first_window["available_from_gameweek"]) != first_gw:
        raise ChipRulesError("first_half must start at the first Gameweek.")
    if int(first_window["available_through_gameweek"]) != split_after:
        raise ChipRulesError("first_half must end at split_after_gameweek.")
    if int(second_window["available_from_gameweek"]) != second_starts:
        raise ChipRulesError(
            "second_half must start at second_half_starts_gameweek."
        )
    if int(second_window["available_through_gameweek"]) != last_gw:
        raise ChipRulesError("second_half must end at the last Gameweek.")

    chips = require_mapping(document["chips"], "chips")
    if set(chips.keys()) != set(VALID_CHIP_IDS):
        raise ChipRulesError(
            "chips must define exactly %s." % (VALID_CHIP_IDS,)
        )

    required_chip_keys = (
        "display_name",
        "inventory_per_window",
        "playable_from_gameweek_by_window",
        "playable_through_gameweek_by_window",
        "activation_surface",
        "can_cancel_after_confirmation",
        "cannot_be_used_in_gameweek_1",
        "cannot_be_used_in_consecutive_gameweeks",
        "effects",
    )
    common_effect_keys = {
        "captain_multiplier",
        "bench_points_count",
        "unlimited_transfers",
        "remove_transfer_hits",
        "preserve_saved_free_transfers",
        "weekly_free_transfer_accrual_after_chip",
        "temporary_squad",
        "restore_previous_squad_at_next_deadline",
        "restore_previous_bank_at_next_deadline",
        "active_window_bank_changes_discarded",
        "transfers_are_permanent",
    }

    for chip_id in VALID_CHIP_IDS:
        chip = require_mapping(chips[chip_id], "chips.%s" % chip_id)
        missing_chip = sorted(set(required_chip_keys) - set(chip.keys()))
        if missing_chip:
            raise ChipRulesError(
                "chips.%s is missing keys: %s." % (chip_id, missing_chip)
            )
        require_nonempty_string(chip["display_name"], "chips.%s.display_name" % chip_id)
        inventory_per_window = require_positive_int(
            chip["inventory_per_window"],
            "chips.%s.inventory_per_window" % chip_id,
        )
        if inventory_per_window != 1:
            raise ChipRulesError(
                "2025/26 chips.%s.inventory_per_window must be 1." % chip_id
            )

        playable_from = require_mapping(
            chip["playable_from_gameweek_by_window"],
            "chips.%s.playable_from_gameweek_by_window" % chip_id,
        )
        playable_through = require_mapping(
            chip["playable_through_gameweek_by_window"],
            "chips.%s.playable_through_gameweek_by_window" % chip_id,
        )
        require_exact_keys(
            playable_from,
            VALID_WINDOW_IDS,
            "chips.%s.playable_from_gameweek_by_window" % chip_id,
        )
        require_exact_keys(
            playable_through,
            VALID_WINDOW_IDS,
            "chips.%s.playable_through_gameweek_by_window" % chip_id,
        )

        for window_id in VALID_WINDOW_IDS:
            window = windows[window_id]
            start = validate_gameweek(
                playable_from[window_id],
                "chips.%s.playable_from_gameweek_by_window.%s"
                % (chip_id, window_id),
                first_gw,
                last_gw,
            )
            end = validate_gameweek(
                playable_through[window_id],
                "chips.%s.playable_through_gameweek_by_window.%s"
                % (chip_id, window_id),
                first_gw,
                last_gw,
            )
            if start < int(window["available_from_gameweek"]):
                raise ChipRulesError(
                    "chips.%s playable start is before inventory window %s."
                    % (chip_id, window_id)
                )
            if end > int(window["available_through_gameweek"]):
                raise ChipRulesError(
                    "chips.%s playable end is after inventory window %s."
                    % (chip_id, window_id)
                )
            if start > end:
                raise ChipRulesError(
                    "chips.%s has an empty playable range in %s."
                    % (chip_id, window_id)
                )

        activation_surface = require_nonempty_string(
            chip["activation_surface"],
            "chips.%s.activation_surface" % chip_id,
        )
        if activation_surface not in VALID_ACTIVATION_SURFACES:
            raise ChipRulesError(
                "chips.%s has invalid activation_surface=%s."
                % (chip_id, activation_surface)
            )

        require_bool(
            chip["can_cancel_after_confirmation"],
            "chips.%s.can_cancel_after_confirmation" % chip_id,
        )
        cannot_gw1 = require_bool(
            chip["cannot_be_used_in_gameweek_1"],
            "chips.%s.cannot_be_used_in_gameweek_1" % chip_id,
        )
        cannot_consecutive = require_bool(
            chip["cannot_be_used_in_consecutive_gameweeks"],
            "chips.%s.cannot_be_used_in_consecutive_gameweeks" % chip_id,
        )

        expected_gw1_block = chip_id in ("wildcard", "free_hit")
        if cannot_gw1 != expected_gw1_block:
            raise ChipRulesError(
                "chips.%s cannot_be_used_in_gameweek_1 is inconsistent."
                % chip_id
            )
        if cannot_consecutive != (chip_id == "free_hit"):
            raise ChipRulesError(
                "Only Free Hit may set cannot_be_used_in_consecutive_gameweeks."
            )

        effects = require_mapping(chip["effects"], "chips.%s.effects" % chip_id)
        missing_effects = sorted(common_effect_keys - set(effects.keys()))
        if missing_effects:
            raise ChipRulesError(
                "chips.%s.effects is missing keys: %s."
                % (chip_id, missing_effects)
            )
        require_positive_int(
            effects["captain_multiplier"],
            "chips.%s.effects.captain_multiplier" % chip_id,
        )
        for key in (
            "bench_points_count",
            "unlimited_transfers",
            "remove_transfer_hits",
            "preserve_saved_free_transfers",
            "temporary_squad",
            "restore_previous_squad_at_next_deadline",
            "restore_previous_bank_at_next_deadline",
            "active_window_bank_changes_discarded",
            "transfers_are_permanent",
        ):
            require_bool(
                effects[key],
                "chips.%s.effects.%s" % (chip_id, key),
            )
        require_nonnegative_int(
            effects["weekly_free_transfer_accrual_after_chip"],
            "chips.%s.effects.weekly_free_transfer_accrual_after_chip"
            % chip_id,
        )

    if int(chips["triple_captain"]["effects"]["captain_multiplier"]) != 3:
        raise ChipRulesError("Triple Captain multiplier must be 3.")
    if not bool(chips["bench_boost"]["effects"]["bench_points_count"]):
        raise ChipRulesError("Bench Boost must count bench points.")

    for chip_id in ("wildcard", "free_hit"):
        effects = chips[chip_id]["effects"]
        if not bool(effects["unlimited_transfers"]):
            raise ChipRulesError("%s must allow unlimited transfers." % chip_id)
        if not bool(effects["remove_transfer_hits"]):
            raise ChipRulesError("%s must remove transfer hits." % chip_id)
        if not bool(effects["preserve_saved_free_transfers"]):
            raise ChipRulesError(
                "%s must preserve saved free transfers." % chip_id
            )

    if bool(chips["wildcard"]["effects"]["temporary_squad"]):
        raise ChipRulesError("Wildcard squad must be permanent.")
    if not bool(chips["wildcard"]["effects"]["transfers_are_permanent"]):
        raise ChipRulesError("Wildcard transfers must be permanent.")
    if not bool(chips["free_hit"]["effects"]["temporary_squad"]):
        raise ChipRulesError("Free Hit squad must be temporary.")
    if bool(chips["free_hit"]["effects"]["transfers_are_permanent"]):
        raise ChipRulesError("Free Hit transfers must not be permanent.")
    if not bool(
        chips["free_hit"]["effects"]["restore_previous_squad_at_next_deadline"]
    ):
        raise ChipRulesError("Free Hit must restore the previous squad.")

    contract = require_mapping(
        document["cross_registry_contract"], "cross_registry_contract"
    )
    require_exact_keys(
        contract,
        (
            "squad_transfer_rules_schema_version",
            "required_transfer_chip_ids",
            "fields_that_must_match",
        ),
        "cross_registry_contract",
    )
    require_nonempty_string(
        contract["squad_transfer_rules_schema_version"],
        "cross_registry_contract.squad_transfer_rules_schema_version",
    )
    transfer_chip_ids = require_list(
        contract["required_transfer_chip_ids"],
        "cross_registry_contract.required_transfer_chip_ids",
    )
    if transfer_chip_ids != ["wildcard", "free_hit"]:
        raise ChipRulesError(
            "required_transfer_chip_ids must be ['wildcard', 'free_hit']."
        )
    fields = require_list(
        contract["fields_that_must_match"],
        "cross_registry_contract.fields_that_must_match",
    )
    if not fields:
        raise ChipRulesError("fields_that_must_match must not be empty.")

    examples = require_mapping(
        document["deterministic_examples"], "deterministic_examples"
    )
    require_exact_keys(
        examples,
        ("state_examples", "activation_examples", "interaction_examples"),
        "deterministic_examples",
    )
    for key in ("state_examples", "activation_examples", "interaction_examples"):
        values = require_list(
            examples[key], "deterministic_examples.%s" % key
        )
        if not values:
            raise ChipRulesError(
                "deterministic_examples.%s must not be empty." % key
            )


def load_chip_rules(
    season: str,
    config_path: Optional[Path] = None,
    config_root: Optional[Path] = None,
) -> ChipRules:
    requested_season = require_nonempty_string(season, "season")
    validate_season_format(requested_season)
    path = (
        Path(config_path)
        if config_path is not None
        else config_path_for_season(requested_season, config_root=config_root)
    )
    if not path.exists():
        raise ChipRulesNotFoundError(
            "Chip rules file not found for season=%s: %s"
            % (requested_season, path)
        )

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChipRulesError(
            "Chip rules file is not valid JSON: %s (%s)." % (path, exc)
        )

    mapping = require_mapping(document, "chip rules document")
    validate_rules_document(mapping, expected_season=requested_season)
    return ChipRules(
        effective_season=requested_season,
        rules_version=str(mapping["rules_version"]),
        schema_version=str(mapping["schema_version"]),
        path=path,
        sha256=sha256_file(path),
        data=dict(mapping),
    )


def window_for_gameweek(rules: ChipRules, gameweek: int) -> Dict[str, Any]:
    first_gw = int(rules.season["first_gameweek"])
    last_gw = int(rules.season["last_gameweek"])
    gw = validate_gameweek(gameweek, "gameweek", first_gw, last_gw)
    matches = [
        window
        for window in rules.inventory_windows
        if int(window["available_from_gameweek"])
        <= gw
        <= int(window["available_through_gameweek"])
    ]
    if len(matches) != 1:
        raise ChipRulesError(
            "Expected one inventory window for gameweek=%s, found %s."
            % (gw, len(matches))
        )
    return dict(matches[0])


def _normalize_usage_history(
    rules: ChipRules,
    usage_history: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if isinstance(usage_history, (str, bytes)) or not isinstance(
        usage_history, Sequence
    ):
        raise ChipRulesError("usage_history must be a sequence of objects.")

    first_gw = int(rules.season["first_gameweek"])
    last_gw = int(rules.season["last_gameweek"])
    normalized: List[Dict[str, Any]] = []
    for index, raw_entry in enumerate(usage_history):
        entry = require_mapping(raw_entry, "usage_history[%s]" % index)
        require_exact_keys(
            entry,
            ("chip_id", "gameweek"),
            "usage_history[%s]" % index,
        )
        normalized.append(
            {
                "chip_id": normalize_chip_id(entry["chip_id"]),
                "gameweek": validate_gameweek(
                    entry["gameweek"],
                    "usage_history[%s].gameweek" % index,
                    first_gw,
                    last_gw,
                ),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (int(item["gameweek"]), str(item["chip_id"])),
    )


def chip_playable_range(
    rules: ChipRules,
    chip_id: str,
    gameweek: int,
) -> Dict[str, Any]:
    normalized_chip = normalize_chip_id(chip_id)
    window = window_for_gameweek(rules, gameweek)
    window_id = str(window["window_id"])
    policy = rules.chips[normalized_chip]
    return {
        "chip_id": normalized_chip,
        "window_id": window_id,
        "playable_from_gameweek": int(
            policy["playable_from_gameweek_by_window"][window_id]
        ),
        "playable_through_gameweek": int(
            policy["playable_through_gameweek_by_window"][window_id]
        ),
    }


def _history_errors(
    rules: ChipRules,
    usage_history: Sequence[Mapping[str, Any]],
    current_gameweek: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    try:
        history = _normalize_usage_history(rules, usage_history)
    except (ChipRulesError, TypeError) as exc:
        return [], [str(exc)]

    errors: List[str] = []
    if current_gameweek is not None:
        first_gw = int(rules.season["first_gameweek"])
        last_gw = int(rules.season["last_gameweek"])
        try:
            current = validate_gameweek(
                current_gameweek,
                "current_gameweek",
                first_gw,
                last_gw,
            )
        except ChipRulesError as exc:
            return history, [str(exc)]
        future = [
            entry
            for entry in history
            if int(entry["gameweek"]) > current
        ]
        if future:
            errors.append(
                "usage_history contains entries after current_gameweek=%s: %s."
                % (current, future)
            )

    by_gameweek: Dict[int, List[str]] = {}
    by_chip_window: Dict[Tuple[str, str], int] = {}
    free_hit_gameweeks: List[int] = []

    for entry in history:
        chip_id = str(entry["chip_id"])
        gameweek = int(entry["gameweek"])
        by_gameweek.setdefault(gameweek, []).append(chip_id)

        playable = chip_playable_range(rules, chip_id, gameweek)
        if not (
            int(playable["playable_from_gameweek"])
            <= gameweek
            <= int(playable["playable_through_gameweek"])
        ):
            errors.append(
                "%s is not playable in gameweek=%s; allowed range for %s is %s-%s."
                % (
                    chip_id,
                    gameweek,
                    playable["window_id"],
                    playable["playable_from_gameweek"],
                    playable["playable_through_gameweek"],
                )
            )

        key = (chip_id, str(playable["window_id"]))
        by_chip_window[key] = by_chip_window.get(key, 0) + 1
        limit = int(rules.chips[chip_id]["inventory_per_window"])
        if by_chip_window[key] > limit:
            errors.append(
                "%s inventory exceeded in %s: used=%s, limit=%s."
                % (chip_id, key[1], by_chip_window[key], limit)
            )

        if chip_id == "free_hit":
            free_hit_gameweeks.append(gameweek)

    maximum = int(
        rules.global_rules["maximum_active_chips_per_gameweek"]
    )
    for gameweek, chip_ids in sorted(by_gameweek.items()):
        if len(chip_ids) > maximum:
            errors.append(
                "gameweek=%s has %s active chips %s; maximum=%s."
                % (gameweek, len(chip_ids), chip_ids, maximum)
            )

    free_hit_gameweeks = sorted(free_hit_gameweeks)
    for previous, current in zip(
        free_hit_gameweeks, free_hit_gameweeks[1:]
    ):
        if current - previous == 1:
            errors.append(
                "Free Hit cannot be used in consecutive Gameweeks: %s and %s."
                % (previous, current)
            )

    return history, errors


def validate_chip_state(
    rules: ChipRules,
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        state_mapping = require_mapping(state, "state")
        require_exact_keys(
            state_mapping,
            ("current_gameweek", "active_chip", "usage_history"),
            "state",
        )
        first_gw = int(rules.season["first_gameweek"])
        last_gw = int(rules.season["last_gameweek"])
        current_gameweek = validate_gameweek(
            state_mapping["current_gameweek"],
            "state.current_gameweek",
            first_gw,
            last_gw,
        )
        active_raw = state_mapping["active_chip"]
        active_chip = (
            None
            if active_raw is None or str(active_raw).strip() == ""
            else normalize_chip_id(str(active_raw))
        )
        history, errors = _history_errors(
            rules,
            require_list(state_mapping["usage_history"], "state.usage_history"),
            current_gameweek=current_gameweek,
        )

        if active_chip is not None:
            active_entries = [
                entry
                for entry in history
                if int(entry["gameweek"]) == current_gameweek
                and str(entry["chip_id"]) == active_chip
            ]
            if len(active_entries) != 1:
                errors.append(
                    "active_chip=%s must have exactly one matching usage entry "
                    "at current_gameweek=%s."
                    % (active_chip, current_gameweek)
                )

        inventory = derive_chip_inventory(
            rules,
            history,
            as_of_gameweek=current_gameweek,
            validate_history=False,
        )
        return {
            "valid": not errors,
            "errors": errors,
            "current_gameweek": current_gameweek,
            "active_chip": active_chip,
            "usage_history": history,
            "inventory": inventory,
        }
    except (ChipRulesError, TypeError) as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "current_gameweek": None,
            "active_chip": None,
            "usage_history": [],
            "inventory": {},
        }


def validate_chip_activation(
    rules: ChipRules,
    chip_id: str,
    target_gameweek: int,
    usage_history: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    errors: List[str] = []
    try:
        normalized_chip = normalize_chip_id(chip_id)
        first_gw = int(rules.season["first_gameweek"])
        last_gw = int(rules.season["last_gameweek"])
        target = validate_gameweek(
            target_gameweek,
            "target_gameweek",
            first_gw,
            last_gw,
        )
        history, history_errors = _history_errors(
            rules,
            usage_history,
            current_gameweek=target,
        )
        errors.extend(history_errors)

        playable = chip_playable_range(rules, normalized_chip, target)
        if not (
            int(playable["playable_from_gameweek"])
            <= target
            <= int(playable["playable_through_gameweek"])
        ):
            errors.append(
                "%s is not playable in gameweek=%s; allowed range for %s is %s-%s."
                % (
                    normalized_chip,
                    target,
                    playable["window_id"],
                    playable["playable_from_gameweek"],
                    playable["playable_through_gameweek"],
                )
            )

        chips_in_target = [
            entry
            for entry in history
            if int(entry["gameweek"]) == target
        ]
        maximum = int(
            rules.global_rules["maximum_active_chips_per_gameweek"]
        )
        if len(chips_in_target) >= maximum:
            errors.append(
                "gameweek=%s already has an active chip: %s."
                % (target, [entry["chip_id"] for entry in chips_in_target])
            )

        used_same_window = [
            entry
            for entry in history
            if str(entry["chip_id"]) == normalized_chip
            and str(window_for_gameweek(rules, int(entry["gameweek"]))["window_id"])
            == str(playable["window_id"])
        ]
        inventory_limit = int(
            rules.chips[normalized_chip]["inventory_per_window"]
        )
        if len(used_same_window) >= inventory_limit:
            errors.append(
                "%s inventory is exhausted in %s: used=%s, limit=%s."
                % (
                    normalized_chip,
                    playable["window_id"],
                    len(used_same_window),
                    inventory_limit,
                )
            )

        if bool(
            rules.chips[normalized_chip][
                "cannot_be_used_in_consecutive_gameweeks"
            ]
        ):
            adjacent = [
                int(entry["gameweek"])
                for entry in history
                if str(entry["chip_id"]) == normalized_chip
                and abs(int(entry["gameweek"]) - target) == 1
            ]
            if adjacent:
                errors.append(
                    "%s cannot be used in consecutive Gameweeks; adjacent usage=%s."
                    % (normalized_chip, sorted(adjacent))
                )

        inventory = derive_chip_inventory(
            rules,
            history,
            as_of_gameweek=target,
            validate_history=False,
        )
        return {
            "valid": not errors,
            "errors": errors,
            "chip_id": normalized_chip,
            "target_gameweek": target,
            "window_id": playable["window_id"],
            "inventory_before": inventory[normalized_chip],
            "effects": dict(rules.chips[normalized_chip]["effects"]),
        }
    except (ChipRulesError, TypeError) as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "chip_id": str(chip_id),
            "target_gameweek": target_gameweek,
            "window_id": None,
            "inventory_before": {},
            "effects": {},
        }


def derive_chip_inventory(
    rules: ChipRules,
    usage_history: Sequence[Mapping[str, Any]],
    as_of_gameweek: int,
    validate_history: bool = True,
) -> Dict[str, Any]:
    first_gw = int(rules.season["first_gameweek"])
    last_gw = int(rules.season["last_gameweek"])
    as_of = validate_gameweek(
        as_of_gameweek,
        "as_of_gameweek",
        first_gw,
        last_gw,
    )
    if validate_history:
        history, errors = _history_errors(
            rules,
            usage_history,
            current_gameweek=as_of,
        )
        if errors:
            raise ChipRulesError(
                "Cannot derive inventory from invalid history: %s."
                % errors
            )
    else:
        history = [
            dict(entry)
            for entry in usage_history
        ]

    current_window_id = str(window_for_gameweek(rules, as_of)["window_id"])
    inventory: Dict[str, Any] = {}
    for chip_id in VALID_CHIP_IDS:
        windows: Dict[str, Any] = {}
        for window in rules.inventory_windows:
            window_id = str(window["window_id"])
            used = sum(
                1
                for entry in history
                if str(entry["chip_id"]) == chip_id
                and str(
                    window_for_gameweek(rules, int(entry["gameweek"]))[
                        "window_id"
                    ]
                )
                == window_id
            )
            total = int(rules.chips[chip_id]["inventory_per_window"])
            remaining = max(0, total - used)
            if as_of > int(window["available_through_gameweek"]):
                status = "expired"
                available_now = 0
            elif as_of < int(window["available_from_gameweek"]):
                status = "future"
                available_now = 0
            else:
                status = "available" if remaining > 0 else "used"
                available_now = remaining

            windows[window_id] = {
                "total": total,
                "used": used,
                "remaining": remaining,
                "available_now": available_now,
                "status": status,
                "playable_from_gameweek": int(
                    rules.chips[chip_id][
                        "playable_from_gameweek_by_window"
                    ][window_id]
                ),
                "playable_through_gameweek": int(
                    rules.chips[chip_id][
                        "playable_through_gameweek_by_window"
                    ][window_id]
                ),
            }

        inventory[chip_id] = {
            "current_window_id": current_window_id,
            "current_window": windows[current_window_id],
            "windows": windows,
        }
    return inventory


def chip_effects(rules: ChipRules, chip_id: str) -> Dict[str, Any]:
    normalized_chip = normalize_chip_id(chip_id)
    policy = rules.chips[normalized_chip]
    return {
        "chip_id": normalized_chip,
        "display_name": policy["display_name"],
        "activation_surface": policy["activation_surface"],
        "can_cancel_after_confirmation": bool(
            policy["can_cancel_after_confirmation"]
        ),
        "effects": dict(policy["effects"]),
    }


def validate_state_examples(rules: ChipRules) -> Dict[str, Any]:
    examples = rules.data["deterministic_examples"]["state_examples"]
    results: List[Dict[str, Any]] = []
    for example in examples:
        result = validate_chip_state(rules, example["state"])
        expected = bool(example["expected_valid"])
        passed = bool(result["valid"]) == expected
        results.append(
            {
                "example_id": example["example_id"],
                "expected_valid": expected,
                "actual_valid": bool(result["valid"]),
                "passed": passed,
                "errors": result["errors"],
            }
        )
    return {
        "passed": all(item["passed"] for item in results),
        "passed_count": sum(1 for item in results if item["passed"]),
        "total_count": len(results),
        "results": results,
    }


def validate_activation_examples(rules: ChipRules) -> Dict[str, Any]:
    examples = rules.data["deterministic_examples"]["activation_examples"]
    results: List[Dict[str, Any]] = []
    for example in examples:
        result = validate_chip_activation(
            rules,
            chip_id=example["chip_id"],
            target_gameweek=example["target_gameweek"],
            usage_history=example["usage_history"],
        )
        expected = bool(example["expected_valid"])
        passed = bool(result["valid"]) == expected
        results.append(
            {
                "example_id": example["example_id"],
                "expected_valid": expected,
                "actual_valid": bool(result["valid"]),
                "passed": passed,
                "errors": result["errors"],
            }
        )
    return {
        "passed": all(item["passed"] for item in results),
        "passed_count": sum(1 for item in results if item["passed"]),
        "total_count": len(results),
        "results": results,
    }


def validate_interaction_examples(rules: ChipRules) -> Dict[str, Any]:
    examples = rules.data["deterministic_examples"]["interaction_examples"]
    results: List[Dict[str, Any]] = []
    for example in examples:
        chip_id = normalize_chip_id(example["chip_id"])
        actual_effects = rules.chips[chip_id]["effects"]
        mismatches: Dict[str, Any] = {}
        for key, expected_value in example["expected"].items():
            actual_value = actual_effects.get(key)
            if actual_value != expected_value:
                mismatches[key] = {
                    "expected": expected_value,
                    "actual": actual_value,
                }
        results.append(
            {
                "example_id": example["example_id"],
                "chip_id": chip_id,
                "passed": not mismatches,
                "mismatches": mismatches,
            }
        )
    return {
        "passed": all(item["passed"] for item in results),
        "passed_count": sum(1 for item in results if item["passed"]),
        "total_count": len(results),
        "results": results,
    }


def validate_squad_transfer_consistency(
    rules: ChipRules,
    squad_transfer_config_path: Path,
) -> Dict[str, Any]:
    path = Path(squad_transfer_config_path)
    if not path.exists():
        return {
            "passed": False,
            "path": str(path),
            "schema_version": None,
            "mismatches": [],
            "errors": ["Squad/transfer rules file not found: %s" % path],
        }

    try:
        document = require_mapping(
            json.loads(path.read_text(encoding="utf-8")),
            "squad/transfer rules document",
        )
    except (json.JSONDecodeError, ChipRulesError) as exc:
        return {
            "passed": False,
            "path": str(path),
            "schema_version": None,
            "mismatches": [],
            "errors": ["Invalid squad/transfer rules document: %s" % exc],
        }

    expected_schema = str(
        rules.cross_registry_contract[
            "squad_transfer_rules_schema_version"
        ]
    )
    actual_schema = document.get("schema_version")
    errors: List[str] = []
    if actual_schema != expected_schema:
        errors.append(
            "Squad/transfer schema_version=%s, expected=%s."
            % (actual_schema, expected_schema)
        )

    transfers = document.get("transfers")
    if not isinstance(transfers, Mapping):
        errors.append("Squad/transfer document has no transfers object.")
        chip_behavior: Mapping[str, Any] = {}
    else:
        raw_behavior = transfers.get("chip_behavior")
        if not isinstance(raw_behavior, Mapping):
            errors.append(
                "Squad/transfer document has no transfers.chip_behavior object."
            )
            chip_behavior = {}
        else:
            chip_behavior = raw_behavior

    field_map = {
        "unlimited_transfers": "unlimited_free_transfers",
        "remove_transfer_hits": "remove_transfer_hits",
        "preserve_saved_free_transfers": "preserve_saved_free_transfers",
        "weekly_free_transfer_accrual_after_chip": "weekly_accrual_after_chip",
        "temporary_squad": "temporary_squad",
    }
    mismatches: List[Dict[str, Any]] = []
    for chip_id in rules.cross_registry_contract["required_transfer_chip_ids"]:
        transfer_policy = chip_behavior.get(chip_id)
        if not isinstance(transfer_policy, Mapping):
            mismatches.append(
                {
                    "chip_id": chip_id,
                    "field": None,
                    "chip_registry_value": None,
                    "squad_transfer_registry_value": None,
                    "reason": "missing_transfer_chip_policy",
                }
            )
            continue
        chip_effect = rules.chips[chip_id]["effects"]
        for chip_field in rules.cross_registry_contract[
            "fields_that_must_match"
        ]:
            transfer_field = field_map.get(chip_field)
            if transfer_field is None:
                mismatches.append(
                    {
                        "chip_id": chip_id,
                        "field": chip_field,
                        "chip_registry_value": chip_effect.get(chip_field),
                        "squad_transfer_registry_value": None,
                        "reason": "no_cross_registry_field_mapping",
                    }
                )
                continue
            left = chip_effect.get(chip_field)
            right = transfer_policy.get(transfer_field)
            if left != right:
                mismatches.append(
                    {
                        "chip_id": chip_id,
                        "field": chip_field,
                        "chip_registry_value": left,
                        "squad_transfer_registry_value": right,
                        "reason": "value_mismatch",
                    }
                )

    return {
        "passed": not errors and not mismatches,
        "path": str(path),
        "schema_version": actual_schema,
        "sha256": sha256_file(path),
        "mismatches": mismatches,
        "errors": errors,
    }


LEGACY_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "chip_endpoint_or_scenario",
        re.compile(
            r"free-hit|wildcard|triple_captain|bench_boost",
            re.IGNORECASE,
        ),
    ),
    (
        "embedded_squad_rule",
        re.compile(
            r"\bSQUAD_RULES\b|\bSTARTING_RULES\b|max_3_players_per_club",
            re.IGNORECASE,
        ),
    ),
    (
        "embedded_chip_effect",
        re.compile(
            r"extra_captain_bonus\s*=\s*2\s*\*|scenario_type\s*=\s*[\"']"
            r"(?:bench_boost|triple_captain)[\"']",
            re.IGNORECASE,
        ),
    ),
)


def scan_legacy_source(path: Path) -> Dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists():
        return {
            "path": str(source_path),
            "exists": False,
            "uses_chip_registry": False,
            "finding_count": 0,
            "findings": [],
        }

    text = source_path.read_text(encoding="utf-8")
    uses_registry = bool(
        re.search(
            r"from\s+app\.rules\.chips\s+import|"
            r"import\s+app\.rules\.chips|"
            r"\bload_chip_rules\s*\(",
            text,
        )
    )
    findings: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for finding_type, pattern in LEGACY_PATTERNS:
            if pattern.search(line):
                findings.append(
                    {
                        "type": finding_type,
                        "line_number": line_number,
                        "text": line.strip()[:240],
                    }
                )
    return {
        "path": str(source_path),
        "exists": True,
        "uses_chip_registry": uses_registry,
        "finding_count": len(findings),
        "findings": findings,
    }


def scan_legacy_sources(paths: Sequence[Path]) -> Dict[str, Any]:
    sources = [scan_legacy_source(path) for path in paths]
    existing = [source for source in sources if source["exists"]]
    with_findings = [
        source for source in existing if source["finding_count"] > 0
    ]
    integrated = [
        source for source in existing if source["uses_chip_registry"]
    ]
    return {
        "source_count": len(sources),
        "existing_source_count": len(existing),
        "sources_with_chip_logic": len(with_findings),
        "chip_logic_finding_count": sum(
            int(source["finding_count"]) for source in existing
        ),
        "registry_integrated_source_count": len(integrated),
        "all_existing_sources_use_registry": bool(existing)
        and len(integrated) == len(existing),
        "sources": sources,
    }


def build_validation_report(
    rules: ChipRules,
    squad_transfer_config_path: Path,
    legacy_sources: Sequence[Path],
) -> Dict[str, Any]:
    state_examples = validate_state_examples(rules)
    activation_examples = validate_activation_examples(rules)
    interaction_examples = validate_interaction_examples(rules)
    cross_registry = validate_squad_transfer_consistency(
        rules,
        squad_transfer_config_path=squad_transfer_config_path,
    )
    legacy = scan_legacy_sources(legacy_sources)

    blockers: List[str] = []
    if not state_examples["passed"]:
        blockers.append("One or more deterministic chip-state examples failed.")
    if not activation_examples["passed"]:
        blockers.append(
            "One or more deterministic chip-activation examples failed."
        )
    if not interaction_examples["passed"]:
        blockers.append(
            "One or more deterministic chip-interaction examples failed."
        )
    if not cross_registry["passed"]:
        blockers.append(
            "Chip policy conflicts with the Day74B squad/transfer registry."
        )

    warnings: List[str] = []
    if not legacy["all_existing_sources_use_registry"]:
        warnings.append(
            "Existing chip routes and scenario/planner utilities do not yet "
            "consume the Day75A registry; Day75A intentionally creates the "
            "versioned policy and pure validators without migrating legacy callers."
        )
    warnings.append(
        "Chip inventory and activation validation is read-only; production "
        "chip activation and decision writes remain disabled."
    )

    passed = not blockers
    return {
        "validation_version": VALIDATION_VERSION,
        "generated_at_utc": utc_now(),
        "effective_season": rules.effective_season,
        "rules_version": rules.rules_version,
        "schema_version": rules.schema_version,
        "rules_path": str(rules.path),
        "rules_sha256": rules.sha256,
        "passed": passed,
        "audit_only": True,
        "writes_database": False,
        "ready_for_chip_policy_loading": passed,
        "ready_for_chip_inventory_validation": passed,
        "ready_for_chip_state_validation": passed,
        "ready_for_chip_activation_validation": passed,
        "ready_for_chip_transfer_interaction_validation": (
            passed and cross_registry["passed"]
        ),
        "ready_for_chip_planner_migration": passed,
        "legacy_chip_callers_use_registry": bool(
            legacy["all_existing_sources_use_registry"]
        ),
        "ready_for_constant_free_legacy_chip_callers": (
            passed and bool(legacy["all_existing_sources_use_registry"])
        ),
        "ready_for_production_chip_write": False,
        "deterministic_examples": {
            "state": state_examples,
            "activation": activation_examples,
            "interaction": interaction_examples,
        },
        "cross_registry_consistency": cross_registry,
        "legacy_comparison": legacy,
        "blockers": blockers,
        "warnings": warnings,
    }


def write_json(report: Mapping[str, Any], path_value: str) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown(report: Mapping[str, Any], path_value: str) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)

    examples = report["deterministic_examples"]
    legacy = report["legacy_comparison"]
    cross_registry = report["cross_registry_consistency"]
    lines = [
        "# Day75A Chip Rules and Inventory Registry Validation",
        "",
        "- Validation version: `%s`" % report["validation_version"],
        "- Effective season: `%s`" % report["effective_season"],
        "- Rules version: `%s`" % report["rules_version"],
        "- Schema version: `%s`" % report["schema_version"],
        "- Rules SHA256: `%s`" % report["rules_sha256"],
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `%s`" % report["audit_only"],
        "- Writes database: `%s`" % report["writes_database"],
        "",
        "## Readiness",
        "",
        "- Policy loading: `%s`"
        % report["ready_for_chip_policy_loading"],
        "- Inventory validation: `%s`"
        % report["ready_for_chip_inventory_validation"],
        "- State validation: `%s`"
        % report["ready_for_chip_state_validation"],
        "- Activation validation: `%s`"
        % report["ready_for_chip_activation_validation"],
        "- Transfer interaction validation: `%s`"
        % report["ready_for_chip_transfer_interaction_validation"],
        "- Planner migration: `%s`"
        % report["ready_for_chip_planner_migration"],
        "- Legacy callers use registry: `%s`"
        % report["legacy_chip_callers_use_registry"],
        "- Constant-free legacy chip callers: `%s`"
        % report["ready_for_constant_free_legacy_chip_callers"],
        "- Production chip write: `%s`"
        % report["ready_for_production_chip_write"],
        "",
        "## Deterministic examples",
        "",
    ]
    for label in ("state", "activation", "interaction"):
        item = examples[label]
        lines.append(
            "- %s: %s/%s passed"
            % (label, item["passed_count"], item["total_count"])
        )

    lines.extend(
        [
            "",
            "## Day74B cross-registry consistency",
            "",
            "- Passed: `%s`" % cross_registry["passed"],
            "- Path: `%s`" % cross_registry["path"],
            "- Mismatch count: `%s`"
            % len(cross_registry["mismatches"]),
            "- Error count: `%s`" % len(cross_registry["errors"]),
            "",
            "## Legacy comparison",
            "",
            "- Source count: `%s`" % legacy["source_count"],
            "- Existing source count: `%s`"
            % legacy["existing_source_count"],
            "- Sources with chip logic: `%s`"
            % legacy["sources_with_chip_logic"],
            "- Chip-logic finding count: `%s`"
            % legacy["chip_logic_finding_count"],
            "- Registry-integrated source count: `%s`"
            % legacy["registry_integrated_source_count"],
            "",
            "## Blockers",
            "",
        ]
    )
    if report["blockers"]:
        lines.extend("- %s" % item for item in report["blockers"])
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend("- %s" % item for item in report["warnings"])
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(
    report: Mapping[str, Any],
    out_json: str,
    out_md: str,
) -> None:
    examples = report["deterministic_examples"]
    legacy = report["legacy_comparison"]
    cross_registry = report["cross_registry_consistency"]

    print("=== Day75A Chip Rules and Inventory Registry ===")
    print("validation_version:", report["validation_version"])
    print("effective_season:", report["effective_season"])
    print("rules_version:", report["rules_version"])
    print("schema_version:", report["schema_version"])
    print("rules_sha256:", report["rules_sha256"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print(
        "ready_for_chip_policy_loading:",
        report["ready_for_chip_policy_loading"],
    )
    print(
        "ready_for_chip_inventory_validation:",
        report["ready_for_chip_inventory_validation"],
    )
    print(
        "ready_for_chip_state_validation:",
        report["ready_for_chip_state_validation"],
    )
    print(
        "ready_for_chip_activation_validation:",
        report["ready_for_chip_activation_validation"],
    )
    print(
        "ready_for_chip_transfer_interaction_validation:",
        report["ready_for_chip_transfer_interaction_validation"],
    )
    print(
        "ready_for_chip_planner_migration:",
        report["ready_for_chip_planner_migration"],
    )
    print(
        "legacy_chip_callers_use_registry:",
        report["legacy_chip_callers_use_registry"],
    )
    print(
        "ready_for_constant_free_legacy_chip_callers:",
        report["ready_for_constant_free_legacy_chip_callers"],
    )
    print(
        "ready_for_production_chip_write:",
        report["ready_for_production_chip_write"],
    )
    print()
    print("Deterministic examples:")
    for label in ("state", "activation", "interaction"):
        item = examples[label]
        print(
            "- %s: %s/%s passed"
            % (label, item["passed_count"], item["total_count"])
        )
    print()
    print("Day74B cross-registry consistency:")
    print("- passed:", cross_registry["passed"])
    print("- mismatch_count:", len(cross_registry["mismatches"]))
    print("- error_count:", len(cross_registry["errors"]))
    print()
    print("Legacy comparison:")
    print("- source_count:", legacy["source_count"])
    print("- existing_source_count:", legacy["existing_source_count"])
    print(
        "- sources_with_chip_logic:",
        legacy["sources_with_chip_logic"],
    )
    print(
        "- chip_logic_finding_count:",
        legacy["chip_logic_finding_count"],
    )
    print(
        "- registry_integrated_source_count:",
        legacy["registry_integrated_source_count"],
    )
    print()
    print(
        "Blockers:",
        "none" if not report["blockers"] else report["blockers"],
    )
    print(
        "Warnings:",
        "none" if not report["warnings"] else report["warnings"],
    )
    print("saved_json:", out_json)
    print("saved_md:", out_md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the versioned FPL chip inventory and interaction policy."
        )
    )
    parser.add_argument("--season", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--squad-transfer-config-path", required=True)
    parser.add_argument(
        "--legacy-source",
        action="append",
        default=[],
        help=(
            "Legacy chip caller to audit. May be provided multiple times. "
            "Defaults to known chip route/scenario/planner sources."
        ),
    )
    parser.add_argument(
        "--out-json",
        default="/tmp/day75a_chip_rules_validation.json",
    )
    parser.add_argument(
        "--out-md",
        default="/tmp/day75a_chip_rules_validation.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = load_chip_rules(
        season=args.season,
        config_path=Path(args.config_path) if args.config_path else None,
    )
    legacy_sources = (
        tuple(Path(value) for value in args.legacy_source)
        if args.legacy_source
        else DEFAULT_LEGACY_SOURCES
    )
    report = build_validation_report(
        rules,
        squad_transfer_config_path=Path(
            args.squad_transfer_config_path
        ),
        legacy_sources=legacy_sources,
    )
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report, args.out_json, args.out_md)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
