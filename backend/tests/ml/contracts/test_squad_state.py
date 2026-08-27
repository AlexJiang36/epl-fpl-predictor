from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ml.contracts.squad_state import (
    CONTRACT_VERSION,
    WRITES_DATABASE,
    ChipInventoryEntry,
    ChipInventoryState,
    FreeTransferState,
    ShadowOptimalDiagnostic,
    SquadPlayerState,
    SquadSelectionState,
    SquadState,
    SquadStateError,
    SquadStatePredecessor,
    calculate_selling_price_units,
    load_squad_state_json,
    predecessor_reference,
    require_valid_squad_state,
    save_squad_state_json,
    validate_squad_state,
    validate_state_transition,
)


class FakeSquadRules:
    initial_budget_units = 1000
    position_quotas = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    squad = {
        "size": 15,
        "max_players_per_club": 3,
    }
    lineup = {
        "starting_size": 11,
        "bench_size": 4,
        "position_bounds": {
            "GKP": {"min": 1, "max": 1},
            "DEF": {"min": 3, "max": 5},
            "MID": {"min": 2, "max": 5},
            "FWD": {"min": 1, "max": 3},
        },
        "bench": {"goalkeepers": 1, "outfield_players": 3},
        "captain_required": True,
        "vice_captain_required": True,
        "captain_and_vice_must_differ": True,
    }
    transfers = {
        "weekly": {
            "maximum_stored_free_transfers": 5,
        }
    }


class FakeChipRules:
    chips = {
        "wildcard": {"inventory_per_window": 1},
        "free_hit": {"inventory_per_window": 1},
        "triple_captain": {"inventory_per_window": 1},
        "bench_boost": {"inventory_per_window": 1},
    }


def make_players(club_override=None, price_delta=0):
    positions = {
        1: "GKP", 2: "GKP",
        3: "DEF", 4: "DEF", 5: "DEF", 6: "DEF", 7: "DEF",
        8: "MID", 9: "MID", 10: "MID", 11: "MID", 12: "MID",
        13: "FWD", 14: "FWD", 15: "FWD",
    }
    players = []
    for pid in range(1, 16):
        purchase = 50
        current = 50 + (price_delta if pid == 8 else 0)
        club_id = ((pid - 1) % 5) + 1
        if club_override and pid in club_override:
            club_id = club_override[pid]
        players.append(
            SquadPlayerState(
                fpl_player_id=pid,
                player_name="P%s" % pid,
                position=positions[pid],
                club_id=club_id,
                purchase_price_units=purchase,
                current_price_units=current,
                selling_price_units=calculate_selling_price_units(purchase, current),
            )
        )
    return tuple(players)


def make_selection():
    return SquadSelectionState(
        starting_xi_player_ids=(1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14),
        bench_order_player_ids=(2, 7, 12, 15),
        captain_player_id=8,
        vice_captain_player_id=13,
    )


def make_chips(gw):
    return ChipInventoryState(
        as_of_gameweek=gw,
        entries=tuple(
            ChipInventoryEntry(chip_id=chip, remaining=1, available_now=1)
            for chip in ("wildcard", "free_hit", "triple_captain", "bench_boost")
        ),
    )


def make_gw1_frozen(kind="model_team"):
    return SquadState(
        season="2026_27",
        gameweek=1,
        as_of_utc="2026-08-21T03:11:47Z",
        state_version="v1",
        state_kind=kind,
        state_status="frozen",
        source_phase_id="GW01-FREEZE",
        source_run_id="gw1-freeze-run",
        players=make_players(),
        selection=make_selection(),
        bank_units=0,
        chip_inventory=make_chips(1),
        free_transfers=FreeTransferState(available_for_gameweek=2, count=1),
    )


def make_gw2_planning(kind="model_team", predecessor=None, shadow=None):
    if predecessor is None:
        predecessor = predecessor_reference(make_gw1_frozen(kind=kind))
    return SquadState(
        season="2026_27",
        gameweek=2,
        as_of_utc="2026-08-25T08:00:00Z",
        state_version="v1",
        state_kind=kind,
        state_status="planning",
        source_phase_id="GW02-PRE",
        source_run_id="gw2-pre-state-run",
        players=make_players(price_delta=2),
        selection=make_selection(),
        bank_units=0,
        chip_inventory=make_chips(2),
        free_transfers=FreeTransferState(available_for_gameweek=2, count=1),
        predecessor=predecessor,
        shadow_optimal=shadow,
    )


class SquadStateContractTests(unittest.TestCase):
    def setUp(self):
        self.squad_rules = FakeSquadRules()
        self.chip_rules = FakeChipRules()

    def test_valid_model_team_state_is_ready_for_optimization(self):
        state = make_gw2_planning()
        report = validate_squad_state(state, self.squad_rules, self.chip_rules)
        self.assertTrue(report.valid)
        self.assertTrue(report.ready_for_optimization)
        self.assertEqual(report.errors, ())
        self.assertFalse(report.writes_database)
        self.assertIs(require_valid_squad_state(state, self.squad_rules, self.chip_rules), state)

    def test_team_alex_is_a_separate_valid_state_kind(self):
        state = make_gw2_planning(kind="team_alex")
        report = validate_squad_state(state, self.squad_rules, self.chip_rules)
        self.assertTrue(report.valid)
        self.assertEqual(state.state_kind, "team_alex")
        self.assertIn("TEAM-ALEX", state.state_id)

    def test_shadow_optimal_cannot_be_owned_state_kind(self):
        with self.assertRaises(SquadStateError):
            SquadState(
                season="2026_27",
                gameweek=1,
                as_of_utc="2026-08-21T03:11:47Z",
                state_version="v1",
                state_kind="shadow_optimal",
                state_status="frozen",
                source_phase_id="GW01-FREEZE",
                source_run_id="bad",
                players=make_players(),
                selection=make_selection(),
                bank_units=0,
                chip_inventory=make_chips(1),
                free_transfers=FreeTransferState(available_for_gameweek=2, count=1),
            )

    def test_shadow_optimal_does_not_change_owned_state_fingerprint(self):
        base = make_gw2_planning()
        shadow = ShadowOptimalDiagnostic(
            target_gameweek=2,
            player_ids=tuple(range(101, 116)),
            source_run_id="shadow-run",
            objective_value=55.2,
        )
        with_shadow = replace(base, shadow_optimal=shadow)
        self.assertEqual(base.owned_state_fingerprint, with_shadow.owned_state_fingerprint)
        self.assertTrue(with_shadow.shadow_optimal.diagnostic_only)

    def test_selling_price_formula_is_canonical_and_validated(self):
        self.assertEqual(calculate_selling_price_units(50, 54), 52)
        self.assertEqual(calculate_selling_price_units(50, 53), 51)
        self.assertEqual(calculate_selling_price_units(50, 48), 48)
        with self.assertRaises(SquadStateError):
            SquadPlayerState(
                fpl_player_id=99,
                position="MID",
                club_id=1,
                purchase_price_units=50,
                current_price_units=54,
                selling_price_units=54,
            )

    def test_structurally_illegal_owned_squad_is_blocked_before_optimization(self):
        state = make_gw2_planning()
        bad_players = make_players(club_override={1: 1, 2: 1, 3: 1, 4: 1})
        bad = replace(state, players=bad_players)
        report = validate_squad_state(bad, self.squad_rules, self.chip_rules)
        self.assertFalse(report.valid)
        self.assertFalse(report.ready_for_optimization)
        self.assertTrue(any("club_limit_exceeded" in item for item in report.errors))

    def test_illegal_lineup_is_blocked_before_optimization(self):
        state = make_gw2_planning()
        bad_selection = SquadSelectionState(
            starting_xi_player_ids=(1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15),
            bench_order_player_ids=(2, 5, 6, 7),
            captain_player_id=8,
            vice_captain_player_id=13,
        )
        bad = replace(state, selection=bad_selection)
        report = validate_squad_state(bad, self.squad_rules, self.chip_rules)
        self.assertFalse(report.valid)
        self.assertTrue(any("formation_invalid_DEF" in item for item in report.errors))

    def test_free_transfers_are_separate_and_capped(self):
        state = replace(
            make_gw2_planning(),
            free_transfers=FreeTransferState(available_for_gameweek=2, count=6),
        )
        report = validate_squad_state(state, self.squad_rules, self.chip_rules)
        self.assertFalse(report.free_transfers_valid)
        self.assertTrue(any("storage cap" in item for item in report.errors))

    def test_chip_inventory_is_separate_and_validated_against_rules(self):
        state = make_gw2_planning()
        bad_inventory = ChipInventoryState(
            as_of_gameweek=2,
            entries=(ChipInventoryEntry("wildcard", 1, 1),),
        )
        report = validate_squad_state(
            replace(state, chip_inventory=bad_inventory),
            self.squad_rules,
            self.chip_rules,
        )
        self.assertFalse(report.chip_inventory_valid)
        self.assertTrue(any("chip IDs" in item for item in report.errors))

    def test_gw2_requires_immutable_gw1_freeze_predecessor(self):
        gw1 = make_gw1_frozen()
        gw2 = make_gw2_planning(predecessor=predecessor_reference(gw1))
        valid, errors = validate_state_transition(gw1, gw2)
        self.assertTrue(valid)
        self.assertEqual(errors, [])
        report = validate_squad_state(gw2, self.squad_rules, self.chip_rules)
        self.assertTrue(report.predecessor_valid)

    def test_mutable_or_wrong_kind_predecessor_is_blocked(self):
        gw1 = make_gw1_frozen()
        bad_ref = SquadStatePredecessor(
            state_id=gw1.state_id,
            owned_state_fingerprint=gw1.owned_state_fingerprint,
            season="2026_27",
            gameweek=1,
            state_kind="team_alex",
            source_phase_id="GW01-FREEZE",
            immutable=False,
        )
        state = make_gw2_planning(predecessor=bad_ref)
        report = validate_squad_state(state, self.squad_rules, self.chip_rules)
        self.assertFalse(report.predecessor_valid)
        self.assertTrue(any("state_kind" in item or "immutable" in item for item in report.errors))

    def test_frozen_state_carries_next_gameweek_free_transfer_scope(self):
        gw1 = make_gw1_frozen()
        report = validate_squad_state(gw1, self.squad_rules, self.chip_rules)
        self.assertTrue(report.valid)
        bad = replace(
            gw1,
            free_transfers=FreeTransferState(available_for_gameweek=1, count=1),
        )
        bad_report = validate_squad_state(bad, self.squad_rules, self.chip_rules)
        self.assertFalse(bad_report.free_transfers_valid)


    def test_stale_predecessor_fingerprint_is_blocked(self):
        gw1 = make_gw1_frozen()
        ref = predecessor_reference(gw1)
        bad_ref = replace(ref, owned_state_fingerprint="0" * 64)
        gw2 = make_gw2_planning(predecessor=bad_ref)
        valid, errors = validate_state_transition(gw1, gw2)
        self.assertFalse(valid)
        self.assertTrue(any("fingerprint" in item for item in errors))

    def test_retained_purchase_price_and_no_transfer_bank_are_immutable_across_rollover(self):
        gw1 = make_gw1_frozen()
        gw2 = make_gw2_planning(predecessor=predecessor_reference(gw1))

        # Market prices may move, but retained-player purchase prices must not.
        players = list(gw2.players)
        retained = players[0]
        players[0] = replace(
            retained,
            purchase_price_units=retained.purchase_price_units + 2,
            selling_price_units=calculate_selling_price_units(
                retained.purchase_price_units + 2, retained.current_price_units
            ),
        )
        bad_purchase = replace(gw2, players=tuple(players))
        valid, errors = validate_state_transition(gw1, bad_purchase)
        self.assertFalse(valid)
        self.assertTrue(any("purchase_price_units" in item for item in errors))

        bad_bank = replace(gw2, bank_units=1)
        valid, errors = validate_state_transition(gw1, bad_bank)
        self.assertFalse(valid)
        self.assertTrue(any("bank_units" in item for item in errors))

    def test_state_status_must_match_day125a_phase_id(self):
        with self.assertRaises(SquadStateError):
            SquadState(
                season="2026_27",
                gameweek=2,
                as_of_utc="2026-08-25T08:00:00Z",
                state_version="v1",
                state_kind="model_team",
                state_status="planning",
                source_phase_id="GW02-FREEZE",
                source_run_id="bad-phase",
                players=make_players(),
                selection=make_selection(),
                bank_units=0,
                chip_inventory=make_chips(2),
                free_transfers=FreeTransferState(available_for_gameweek=2, count=1),
                predecessor=predecessor_reference(make_gw1_frozen()),
            )

    def test_json_roundtrip_preserves_owned_identity(self):
        shadow = ShadowOptimalDiagnostic(
            target_gameweek=2,
            player_ids=tuple(range(101, 116)),
            source_run_id="shadow-run",
        )
        state = make_gw2_planning(shadow=shadow)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "squad_state.json"
            save_squad_state_json(state, path)
            loaded = load_squad_state_json(path)
        self.assertEqual(state.state_id, loaded.state_id)
        self.assertEqual(state.owned_state_fingerprint, loaded.owned_state_fingerprint)
        self.assertEqual(state.to_dict(), loaded.to_dict())

    def test_json_load_rejects_tampered_serialized_identity(self):
        state = make_gw2_planning()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "squad_state.json"
            save_squad_state_json(state, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["bank_units"] = payload["bank_units"] + 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SquadStateError):
                load_squad_state_json(path)

    def test_contract_has_artifact_only_io_and_no_database_write_flag(self):
        state = make_gw2_planning()
        report = validate_squad_state(state, self.squad_rules, self.chip_rules)
        self.assertEqual(CONTRACT_VERSION, "fpl_squad_state_v1")
        self.assertFalse(WRITES_DATABASE)
        self.assertFalse(state.writes_database)
        self.assertFalse(report.writes_database)


if __name__ == "__main__":
    unittest.main()
