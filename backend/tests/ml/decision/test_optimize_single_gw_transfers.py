from __future__ import annotations

import copy
import unittest

from app.rules.squad import load_squad_transfer_rules
from ml.contracts.gameweek_cycle import build_phase_id
from ml.contracts.squad_state import (
    ChipInventoryState,
    FreeTransferState,
    SquadPlayerState,
    SquadSelectionState,
    SquadState,
)
from ml.decision.free_transfer_ledger import build_ledger_state
from ml.decision.generate_transfer_candidates import (
    CandidatePruningPolicy,
    generate_transfer_candidates,
)
from ml.decision.optimize_single_gw_transfers import (
    SingleGWTransferOptimizerError,
    optimize_single_gw_transfers,
)


SEASON = "2026_27"


class SingleGWTransferOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_squad_transfer_rules(SEASON)
        cls.hit_cost = int(
            cls.rules.transfers["weekly"][
                "hit_cost_points_per_additional_transfer"
            ]
        )

    def frozen_state(
        self,
        *,
        bank_units=0,
        ft_count=1,
        club_overrides=None,
    ):
        club_overrides = club_overrides or {}
        players = []
        specs = (
            [("GKP", 2)]
            + [("DEF", 5)]
            + [("MID", 5)]
            + [("FWD", 3)]
        )
        pid = 1
        for position, count in specs:
            for _ in range(count):
                club_id = int(club_overrides.get(pid, pid))
                players.append(
                    SquadPlayerState(
                        fpl_player_id=pid,
                        player_name="P%s" % pid,
                        position=position,
                        club_id=club_id,
                        purchase_price_units=50,
                        current_price_units=50,
                        selling_price_units=50,
                    )
                )
                pid += 1

        starters = (1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15)
        bench = (2, 6, 7, 12)
        return SquadState(
            season=SEASON,
            gameweek=2,
            as_of_utc="2026-08-28T02:55:45Z",
            state_version="test_gw2_frozen",
            state_kind="model_team",
            state_status="frozen",
            source_phase_id=build_phase_id(2, "freeze"),
            source_run_id="test_gw2_freeze",
            players=tuple(players),
            selection=SquadSelectionState(
                starting_xi_player_ids=starters,
                bench_order_player_ids=bench,
                captain_player_id=8,
                vice_captain_player_id=9,
            ),
            bank_units=bank_units,
            chip_inventory=ChipInventoryState(
                as_of_gameweek=2,
                entries=(),
            ),
            free_transfers=FreeTransferState(
                available_for_gameweek=3,
                count=ft_count,
            ),
        )

    def ledger(self, state, *, available=None):
        count = (
            state.free_transfers.count
            if available is None
            else available
        )
        return build_ledger_state(
            season=SEASON,
            state_kind=state.state_kind,
            gameweek=3,
            available_free_transfers=count,
            rules=self.rules,
        )

    def predictions(self, state, incoming=None, owned_overrides=None):
        owned_overrides = owned_overrides or {}
        rows = []
        for player in state.players:
            rows.append(
                {
                    "fpl_player_id": player.fpl_player_id,
                    "target_gw": 3,
                    "predicted_points": float(
                        owned_overrides.get(player.fpl_player_id, 3.0)
                    ),
                    "selection_eligible": True,
                    "expected_minutes": 80.0,
                    "start_probability": 0.8,
                    "appearance_probability": 0.95,
                    "risk_flags": [],
                }
            )
        for item in incoming or []:
            rows.append(
                {
                    "fpl_player_id": item["fpl_player_id"],
                    "target_gw": 3,
                    "predicted_points": float(item["predicted_points"]),
                    "selection_eligible": True,
                    "expected_minutes": 85.0,
                    "start_probability": 0.9,
                    "appearance_probability": 0.98,
                    "risk_flags": [],
                }
            )
        return rows

    def candidate_report(self, state, incoming, predictions):
        planning_view = {
            "season": SEASON,
            "state_kind": state.state_kind,
            "gameweek": 3,
            "bank_units": state.bank_units,
            "squad": [player.to_dict() for player in state.players],
        }
        market = [
            {
                "fpl_player_id": item["fpl_player_id"],
                "web_name": item["name"],
                "position": item["position"],
                "team_id": item["club_id"],
                "team_short_name": "T%s" % item["club_id"],
                "now_cost": item.get("price_units", 50),
                "status": "a",
                "selection_eligible": True,
            }
            for item in incoming
        ]
        return generate_transfer_candidates(
            planning_view,
            market,
            predictions,
            pruning_policy=CandidatePruningPolicy(
                max_pair_candidates_per_out=None
            ),
            rules=self.rules,
        )

    def run_case(
        self,
        *,
        incoming,
        max_transfers,
        state=None,
        owned_overrides=None,
    ):
        state = state or self.frozen_state()
        predictions = self.predictions(
            state,
            incoming=incoming,
            owned_overrides=owned_overrides,
        )
        candidates = self.candidate_report(
            state,
            incoming,
            predictions,
        )
        return optimize_single_gw_transfers(
            state,
            candidates,
            predictions,
            self.ledger(state),
            max_transfers=max_transfers,
            rules=self.rules,
        )

    def test_current_valuation_overlay_supports_price_drift(self):
        state = self.frozen_state(bank_units=0, ft_count=1)
        current_players = []
        for player in state.players:
            row = player.to_dict()
            if player.fpl_player_id == 8:
                row["current_price_units"] = 52
                row["selling_price_units"] = 51
            current_players.append(row)

        current_owned = {
            "season": SEASON,
            "state_kind": "model_team",
            "gameweek": 3,
            "bank_units": 0,
            "players": current_players,
        }
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "UpgradeMID",
                "position": "MID",
                "club_id": 101,
                "price_units": 51,
                "predicted_points": 10.0,
            }
        ]
        predictions = self.predictions(
            state,
            incoming=incoming,
            owned_overrides={8: 1.0},
        )
        candidates = generate_transfer_candidates(
            current_owned,
            [
                {
                    "fpl_player_id": 101,
                    "web_name": "UpgradeMID",
                    "position": "MID",
                    "team_id": 101,
                    "team_short_name": "T101",
                    "now_cost": 51,
                    "status": "a",
                    "selection_eligible": True,
                }
            ],
            predictions,
            pruning_policy=CandidatePruningPolicy(
                max_pair_candidates_per_out=None
            ),
            rules=self.rules,
        )
        result = optimize_single_gw_transfers(
            state,
            candidates,
            predictions,
            self.ledger(state),
            max_transfers=1,
            current_owned_state=current_owned,
            rules=self.rules,
        )
        self.assertTrue(result["current_valuation_overlay_used"])
        matching = [
            option
            for option in result["ranked_options"]
            if option["action"] == "TRANSFER"
            and option["transfers"][0]["out_fpl_player_id"] == 8
        ]
        self.assertTrue(matching)
        self.assertEqual(
            matching[0]["transfers"][0]["out_selling_price_units"],
            51,
        )


    def test_no_transfer_is_first_class_candidate(self):
        result = self.run_case(incoming=[], max_transfers=0)
        self.assertEqual(result["winner"]["action"], "NO TRANSFER")
        self.assertEqual(result["no_transfer"]["action"], "NO TRANSFER")
        self.assertEqual(result["no_transfer"]["transfer_count"], 0)
        self.assertEqual(result["no_transfer"]["transfer_hit_points"], 0)
        self.assertTrue(result["safety"]["no_transfer_first_class"])

    def test_previous_state_must_be_frozen(self):
        frozen = self.frozen_state()
        mapping = frozen.to_dict()
        mapping["state_status"] = "planning"
        mapping["source_phase_id"] = build_phase_id(2, "pre")
        predictions = self.predictions(frozen)
        candidates = {
            "generator_scope": "stateful_owned_squad_transfer_candidates",
            "season": SEASON,
            "state_kind": "model_team",
            "target_gw": 3,
            "pair_candidates": [],
            "safety": {
                "shadow_optimal_consumed": False,
                "opening_squad_rebuild_used": False,
                "general_transfer_targets_surface_used_as_state": False,
                "missing_predictions_zero_filled": False,
            },
        }
        with self.assertRaises(SingleGWTransferOptimizerError):
            optimize_single_gw_transfers(
                mapping,
                candidates,
                predictions,
                self.ledger(frozen),
                max_transfers=0,
                rules=self.rules,
            )

    def test_target_gw_is_previous_frozen_gw_plus_one(self):
        result = self.run_case(incoming=[], max_transfers=0)
        self.assertEqual(result["target_gw"], 3)

    def test_clear_single_free_transfer_beats_no_transfer(self):
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "UpgradeMID",
                "position": "MID",
                "club_id": 101,
                "price_units": 50,
                "predicted_points": 10.0,
            }
        ]
        result = self.run_case(incoming=incoming, max_transfers=1)
        self.assertEqual(result["winner"]["action"], "TRANSFER")
        self.assertEqual(result["winner"]["transfer_count"], 1)
        self.assertEqual(result["winner"]["transfer_hit_points"], 0)
        self.assertGreater(result["winner"]["net_gain_vs_no_transfer"], 0.0)

    def test_multiple_transfers_are_supported(self):
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "UpgradeMID",
                "position": "MID",
                "club_id": 101,
                "price_units": 50,
                "predicted_points": 12.0,
            },
            {
                "fpl_player_id": 102,
                "name": "UpgradeDEF",
                "position": "DEF",
                "club_id": 102,
                "price_units": 50,
                "predicted_points": 11.0,
            },
        ]
        result = self.run_case(incoming=incoming, max_transfers=2)
        self.assertTrue(
            any(option["transfer_count"] == 2 for option in result["ranked_options"])
        )
        two = next(
            option
            for option in result["ranked_options"]
            if option["transfer_count"] == 2
        )
        self.assertEqual(
            len({item["out_fpl_player_id"] for item in two["transfers"]}),
            2,
        )
        self.assertEqual(
            len({item["in_fpl_player_id"] for item in two["transfers"]}),
            2,
        )

    def test_second_transfer_is_charged_when_only_one_ft_available(self):
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "UpgradeMID",
                "position": "MID",
                "club_id": 101,
                "price_units": 50,
                "predicted_points": 12.0,
            },
            {
                "fpl_player_id": 102,
                "name": "UpgradeDEF",
                "position": "DEF",
                "club_id": 102,
                "price_units": 50,
                "predicted_points": 11.0,
            },
        ]
        result = self.run_case(incoming=incoming, max_transfers=2)
        two = next(
            option
            for option in result["ranked_options"]
            if option["transfer_count"] == 2
        )
        self.assertEqual(two["free_transfers_used"], 1)
        self.assertEqual(two["charged_transfers"], 1)
        self.assertEqual(two["transfer_hit_points"], self.hit_cost)

    def test_no_transfer_can_beat_transfer_after_cost(self):
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "WeakMID",
                "position": "MID",
                "club_id": 101,
                "price_units": 50,
                "predicted_points": 2.0,
            }
        ]
        result = self.run_case(incoming=incoming, max_transfers=1)
        self.assertEqual(result["winner"]["action"], "NO TRANSFER")

    def test_lineup_and_captain_are_reoptimized_after_transfer(self):
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "CaptainUpgrade",
                "position": "MID",
                "club_id": 101,
                "price_units": 50,
                "predicted_points": 20.0,
            }
        ]
        result = self.run_case(incoming=incoming, max_transfers=1)
        self.assertEqual(
            result["winner"]["lineup"]["captain_player_id"],
            101,
        )
        self.assertIn(
            101,
            result["winner"]["lineup"]["starting_player_ids"],
        )

    def test_bank_effect_is_explained_separately(self):
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "CheapUpgrade",
                "position": "MID",
                "club_id": 101,
                "price_units": 45,
                "predicted_points": 10.0,
            }
        ]
        result = self.run_case(incoming=incoming, max_transfers=1)
        winner = result["winner"]
        self.assertEqual(winner["bank_before_units"], 0)
        self.assertEqual(winner["bank_after_units"], 5)
        self.assertEqual(winner["sales_units"], 50)
        self.assertEqual(winner["purchases_units"], 45)

    def test_multi_transfer_club_limit_is_rechecked(self):
        # Two players already at club 500. Each incoming to club 500 is legal
        # alone (count becomes 3), but taking both would produce 4 and must fail.
        state = self.frozen_state(
            club_overrides={1: 500, 3: 500}
        )
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "Club500MID",
                "position": "MID",
                "club_id": 500,
                "price_units": 50,
                "predicted_points": 12.0,
            },
            {
                "fpl_player_id": 102,
                "name": "Club500FWD",
                "position": "FWD",
                "club_id": 500,
                "price_units": 50,
                "predicted_points": 11.0,
            },
        ]
        result = self.run_case(
            incoming=incoming,
            max_transfers=2,
            state=state,
        )
        self.assertGreater(
            result["rejected_plan_summary"].get("illegal_final_squad", 0),
            0,
        )

    def test_candidate_report_target_gw_mismatch_is_rejected(self):
        state = self.frozen_state()
        predictions = self.predictions(state)
        candidates = self.candidate_report(state, [], predictions)
        candidates["target_gw"] = 4
        with self.assertRaises(SingleGWTransferOptimizerError):
            optimize_single_gw_transfers(
                state,
                candidates,
                predictions,
                self.ledger(state),
                max_transfers=0,
                rules=self.rules,
            )

    def test_candidate_report_state_kind_mismatch_is_rejected(self):
        state = self.frozen_state()
        predictions = self.predictions(state)
        candidates = self.candidate_report(state, [], predictions)
        candidates["state_kind"] = "team_alex"
        with self.assertRaises(SingleGWTransferOptimizerError):
            optimize_single_gw_transfers(
                state,
                candidates,
                predictions,
                self.ledger(state),
                max_transfers=0,
                rules=self.rules,
            )

    def test_ft_ledger_mismatch_is_rejected(self):
        state = self.frozen_state(ft_count=1)
        predictions = self.predictions(state)
        candidates = self.candidate_report(state, [], predictions)
        wrong_ledger = build_ledger_state(
            season=SEASON,
            state_kind="model_team",
            gameweek=3,
            available_free_transfers=2,
            rules=self.rules,
        )
        with self.assertRaises(SingleGWTransferOptimizerError):
            optimize_single_gw_transfers(
                state,
                candidates,
                predictions,
                wrong_ledger,
                max_transfers=0,
                rules=self.rules,
            )

    def test_missing_owned_prediction_is_never_zero_filled(self):
        state = self.frozen_state()
        predictions = self.predictions(state)
        predictions = [
            row for row in predictions if row["fpl_player_id"] != 1
        ]
        candidates = self.candidate_report(state, [], predictions)
        with self.assertRaises(SingleGWTransferOptimizerError):
            optimize_single_gw_transfers(
                state,
                candidates,
                predictions,
                self.ledger(state),
                max_transfers=0,
                rules=self.rules,
            )

    def test_no_transfer_and_transfer_use_same_evaluation_structure(self):
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "UpgradeMID",
                "position": "MID",
                "club_id": 101,
                "price_units": 50,
                "predicted_points": 10.0,
            }
        ]
        result = self.run_case(incoming=incoming, max_transfers=1)
        transfer = next(
            option
            for option in result["ranked_options"]
            if option["action"] == "TRANSFER"
        )
        self.assertEqual(
            set(result["no_transfer"].keys()),
            set(transfer.keys()),
        )

    def test_exact_tie_prefers_fewer_transfers(self):
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "EqualMID",
                "position": "MID",
                "club_id": 101,
                "price_units": 50,
                "predicted_points": 3.0,
            }
        ]
        result = self.run_case(incoming=incoming, max_transfers=1)
        self.assertEqual(result["winner"]["action"], "NO TRANSFER")

    def test_output_is_deterministic_when_candidate_edges_are_reversed(self):
        state = self.frozen_state()
        incoming = [
            {
                "fpl_player_id": 101,
                "name": "UpgradeMID",
                "position": "MID",
                "club_id": 101,
                "price_units": 50,
                "predicted_points": 10.0,
            },
            {
                "fpl_player_id": 102,
                "name": "UpgradeDEF",
                "position": "DEF",
                "club_id": 102,
                "price_units": 50,
                "predicted_points": 9.0,
            },
        ]
        predictions = self.predictions(state, incoming=incoming)
        candidates = self.candidate_report(state, incoming, predictions)
        first = optimize_single_gw_transfers(
            state,
            candidates,
            predictions,
            self.ledger(state),
            max_transfers=2,
            rules=self.rules,
        )
        reversed_candidates = copy.deepcopy(candidates)
        reversed_candidates["pair_candidates"] = list(
            reversed(reversed_candidates["pair_candidates"])
        )
        second = optimize_single_gw_transfers(
            state,
            reversed_candidates,
            predictions,
            self.ledger(state),
            max_transfers=2,
            rules=self.rules,
        )
        self.assertEqual(
            first["winner"]["transfer_ids"],
            second["winner"]["transfer_ids"],
        )
        self.assertEqual(
            first["winner"]["net_projected_points"],
            second["winner"]["net_projected_points"],
        )

    def test_max_transfers_is_explicit(self):
        result = self.run_case(incoming=[], max_transfers=0)
        self.assertEqual(result["configured_max_transfers"], 0)

    def test_opening_squad_optimizer_is_never_used(self):
        result = self.run_case(incoming=[], max_transfers=0)
        self.assertFalse(
            result["safety"]["opening_squad_optimizer_used"]
        )
        self.assertFalse(result["safety"]["candidate_generator_bypassed"])
        self.assertFalse(result["safety"]["writes_manager_state"])
        self.assertFalse(result["safety"]["writes_database"])


if __name__ == "__main__":
    unittest.main()
