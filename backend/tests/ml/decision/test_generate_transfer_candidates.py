from __future__ import annotations

import unittest

from app.rules.squad import load_squad_transfer_rules
from ml.decision.generate_transfer_candidates import (
    CandidatePruningPolicy,
    TransferCandidateGeneratorError,
    generate_transfer_candidates,
)


SEASON = "2026_27"


class TransferCandidateGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_squad_transfer_rules(SEASON)
        cls.quotas = {
            pos: int(cls.rules.squad["position_quotas"][pos])
            for pos in ("GKP", "DEF", "MID", "FWD")
        }

    def owned_state(self, *, kind="model_team", bank_units=10):
        rows = []
        pid = 1
        for position in ("GKP", "DEF", "MID", "FWD"):
            for _ in range(self.quotas[position]):
                rows.append(
                    {
                        "player_id": pid,
                        "fpl_player_id": pid,
                        "web_name": "Owned%s" % pid,
                        "position": position,
                        "team_id": pid,
                        "team_short_name": "T%s" % pid,
                        "selling_price_units": 50,
                        "current_price_units": 50,
                    }
                )
                pid += 1
        return {
            "state_version": "test_squad_state_v1",
            "season": SEASON,
            "gameweek": 3,
            "state_kind": kind,
            "bank_units": bank_units,
            "squad": rows,
        }

    def prediction(
        self,
        pid,
        points,
        *,
        eligible=True,
        position=None,
        risk_flags=None,
        expected_minutes=80.0,
        start_probability=0.8,
        appearance_probability=0.95,
    ):
        return {
            "player_id": pid,
            "fpl_player_id": pid,
            "target_gw": 3,
            "predicted_points": points,
            "selection_eligible": eligible,
            "risk_flags": risk_flags or [],
            "expected_minutes": expected_minutes,
            "start_probability": start_probability,
            "appearance_probability": appearance_probability,
            "fallback_used": False,
            "manual_review_required": False,
            "role_proxy": "starter",
            "role_confidence": "high",
        }

    def market(
        self,
        pid,
        position,
        *,
        price=50,
        team_id=100,
        eligible=True,
        name=None,
    ):
        return {
            "player_id": pid,
            "fpl_player_id": pid,
            "web_name": name or "Market%s" % pid,
            "position": position,
            "team_id": team_id,
            "team_short_name": "M%s" % team_id,
            "now_cost": price,
            "status": "a",
            "selection_eligible": eligible,
        }

    def base_inputs(self):
        state = self.owned_state()
        predictions = []
        for row in state["squad"]:
            predictions.append(
                self.prediction(
                    row["player_id"],
                    float(row["player_id"]) / 10.0,
                )
            )
        market = [
            self.market(1001, "GKP", price=45, team_id=1001),
            self.market(1002, "DEF", price=55, team_id=1002),
            self.market(1003, "MID", price=60, team_id=1003),
            self.market(1004, "FWD", price=50, team_id=1004),
        ]
        predictions.extend(
            [
                self.prediction(1001, 4.0),
                self.prediction(1002, 5.0),
                self.prediction(1003, 6.0),
                self.prediction(1004, 5.5),
            ]
        )
        return state, market, predictions

    def test_uses_owned_squad_not_scratch_squad(self):
        state, market, predictions = self.base_inputs()
        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        owned_ids = {str(row["player_id"]) for row in state["squad"]}
        self.assertTrue(report["pair_candidates"])
        self.assertTrue(
            all(str(row["out_key"]) in owned_ids for row in report["pair_candidates"])
        )
        self.assertFalse(report["safety"]["opening_squad_rebuild_used"])

    def test_shadow_optimal_state_is_rejected(self):
        state, market, predictions = self.base_inputs()
        state["state_kind"] = "shadow_optimal"
        with self.assertRaises(TransferCandidateGeneratorError):
            generate_transfer_candidates(
                state, market, predictions, rules=self.rules
            )

    def test_locked_owned_player_is_never_out_candidate(self):
        state, market, predictions = self.base_inputs()
        locked_id = state["squad"][0]["player_id"]
        report = generate_transfer_candidates(
            state,
            market,
            predictions,
            locked_player_ids=[locked_id],
            rules=self.rules,
        )
        self.assertNotIn(
            str(locked_id),
            {str(row["key"]) for row in report["out_candidates"]},
        )
        self.assertTrue(
            any(
                str(row["key"]) == str(locked_id)
                and row["reason"] == "locked_player"
                for row in report["excluded_outgoing"]
            )
        )

    def test_pair_candidates_are_position_compatible(self):
        state, market, predictions = self.base_inputs()
        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        self.assertTrue(
            all(
                row["out_position"] == row["in_position"]
                for row in report["pair_candidates"]
            )
        )

    def test_affordability_uses_selling_price_plus_bank(self):
        state, market, predictions = self.base_inputs()
        # First MID owned player sells for 50 and bank is 10 => max 60.
        mid_out = next(
            row for row in state["squad"] if row["position"] == "MID"
        )
        market.append(
            self.market(2001, "MID", price=61, team_id=2001)
        )
        predictions.append(self.prediction(2001, 20.0))
        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        self.assertFalse(
            any(
                str(row["out_key"]) == str(mid_out["player_id"])
                and str(row["in_key"]) == "2001"
                for row in report["pair_candidates"]
            )
        )
        self.assertGreater(
            report["pair_rejection_summary"].get("unaffordable", 0),
            0,
        )

    def test_affordable_pair_records_bank_after(self):
        state, market, predictions = self.base_inputs()
        mid_out = next(
            row for row in state["squad"] if row["position"] == "MID"
        )
        pair = next(
            row
            for row in generate_transfer_candidates(
                state, market, predictions, rules=self.rules
            )["pair_candidates"]
            if str(row["out_key"]) == str(mid_out["player_id"])
            and str(row["in_key"]) == "1003"
        )
        self.assertEqual(pair["affordable_budget_units"], 60)
        self.assertEqual(pair["bank_after_units"], 0)

    def test_club_limit_is_respected_after_swap(self):
        state, market, predictions = self.base_inputs()
        max_per_club = int(self.rules.squad["max_players_per_club"])

        # Put max_per_club owned players on club 500 without violating quota.
        for index in range(max_per_club):
            state["squad"][index]["team_id"] = 500
            state["squad"][index]["team_short_name"] = "C500"

        target_position = next(
            row["position"]
            for row in state["squad"]
            if row["team_id"] != 500
        )
        out_id = next(
            row["player_id"]
            for row in state["squad"]
            if row["position"] == target_position and row["team_id"] != 500
        )

        market.append(
            self.market(3001, target_position, price=50, team_id=500)
        )
        predictions.append(self.prediction(3001, 30.0))

        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        self.assertFalse(
            any(
                str(row["out_key"]) == str(out_id)
                and str(row["in_key"]) == "3001"
                for row in report["pair_candidates"]
            )
        )
        self.assertGreater(
            report["pair_rejection_summary"].get("club_limit", 0),
            0,
        )

    def test_ineligible_incoming_is_excluded(self):
        state, market, predictions = self.base_inputs()
        market.append(
            self.market(4001, "MID", price=50, team_id=4001, eligible=False)
        )
        predictions.append(self.prediction(4001, 20.0, eligible=False))
        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        self.assertNotIn(
            "4001",
            {str(row["key"]) for row in report["in_candidates"]},
        )

    def test_missing_prediction_is_not_zero_filled(self):
        state, market, predictions = self.base_inputs()
        market.append(
            self.market(5001, "MID", price=50, team_id=5001)
        )
        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        excluded = {
            str(row["key"]): row["reason"]
            for row in report["excluded_incoming"]
        }
        self.assertEqual(
            excluded["5001"],
            "missing_horizon_prediction_not_zero_filled",
        )
        self.assertFalse(report["safety"]["missing_predictions_zero_filled"])

    def test_projected_gain_is_in_minus_out_horizon_points(self):
        state, market, predictions = self.base_inputs()
        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        pair = report["pair_candidates"][0]
        self.assertAlmostEqual(
            pair["projected_gain"],
            pair["in_horizon_predicted_points"]
            - pair["out_horizon_predicted_points"],
        )

    def test_risk_and_role_metadata_are_preserved(self):
        state, market, predictions = self.base_inputs()
        predictions.append(
            self.prediction(
                6001,
                7.5,
                risk_flags=["role_uncertainty"],
                expected_minutes=55.0,
                start_probability=0.55,
                appearance_probability=0.8,
            )
        )
        market.append(
            self.market(6001, "MID", price=50, team_id=6001)
        )
        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        incoming = next(
            row for row in report["in_candidates"] if str(row["key"]) == "6001"
        )
        self.assertEqual(incoming["risk"]["risk_flags"], ["role_uncertainty"])
        self.assertEqual(incoming["role"]["expected_minutes"], 55.0)
        self.assertEqual(incoming["role"]["start_probability"], 0.55)

    def test_pair_pruning_is_explicit_and_configurable(self):
        state, market, predictions = self.base_inputs()
        for pid, pts in ((7001, 10.0), (7002, 9.0), (7003, 8.0)):
            market.append(self.market(pid, "MID", price=50, team_id=pid))
            predictions.append(self.prediction(pid, pts))

        report = generate_transfer_candidates(
            state,
            market,
            predictions,
            pruning_policy=CandidatePruningPolicy(
                max_pair_candidates_per_out=1
            ),
            rules=self.rules,
        )
        per_out = {}
        for row in report["pair_candidates"]:
            per_out.setdefault(str(row["out_key"]), 0)
            per_out[str(row["out_key"])] += 1
        self.assertTrue(all(count <= 1 for count in per_out.values()))
        self.assertGreater(report["pair_pruned_count"], 0)
        self.assertEqual(
            report["pruning_policy"]["max_pair_candidates_per_out"], 1
        )

    def test_minimum_gain_pruning_is_explicit(self):
        state, market, predictions = self.base_inputs()
        report = generate_transfer_candidates(
            state,
            market,
            predictions,
            pruning_policy=CandidatePruningPolicy(
                minimum_projected_gain=100.0,
                max_pair_candidates_per_out=None,
            ),
            rules=self.rules,
        )
        self.assertEqual(report["pair_candidates"], [])
        self.assertGreater(
            report["pair_rejection_summary"].get(
                "below_minimum_projected_gain", 0
            ),
            0,
        )

    def test_multi_gw_rows_sum_when_no_explicit_horizon_points(self):
        state, market, predictions = self.base_inputs()
        market.append(self.market(8001, "MID", price=50, team_id=8001))
        predictions.extend(
            [
                {
                    **self.prediction(8001, 4.0),
                    "target_gw": 3,
                },
                {
                    **self.prediction(8001, 5.0),
                    "target_gw": 4,
                },
            ]
        )
        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        incoming = next(
            row for row in report["in_candidates"] if str(row["key"]) == "8001"
        )
        self.assertEqual(incoming["horizon_predicted_points"], 9.0)
        self.assertEqual(incoming["target_gw_predicted_points"], 4.0)

    def test_already_owned_player_is_not_in_candidate(self):
        state, market, predictions = self.base_inputs()
        owned = state["squad"][0]
        market.append(
            self.market(
                owned["player_id"],
                owned["position"],
                price=50,
                team_id=owned["team_id"],
            )
        )
        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        self.assertNotIn(
            str(owned["player_id"]),
            {str(row["key"]) for row in report["in_candidates"]},
        )

    def test_model_team_and_team_alex_are_separate_stateful_runs(self):
        state, market, predictions = self.base_inputs()
        model = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        alex_state = self.owned_state(kind="team_alex", bank_units=20)
        alex = generate_transfer_candidates(
            alex_state, market, predictions, rules=self.rules
        )
        self.assertEqual(model["state_kind"], "model_team")
        self.assertEqual(alex["state_kind"], "team_alex")
        self.assertEqual(model["bank_units"], 10)
        self.assertEqual(alex["bank_units"], 20)

    def test_general_transfer_targets_is_not_used_as_owned_state(self):
        state, market, predictions = self.base_inputs()
        report = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        self.assertFalse(
            report["safety"]["general_transfer_targets_surface_used_as_state"]
        )
        self.assertEqual(
            report["generator_scope"],
            "stateful_owned_squad_transfer_candidates",
        )

    def test_output_order_is_deterministic(self):
        state, market, predictions = self.base_inputs()
        first = generate_transfer_candidates(
            state, market, predictions, rules=self.rules
        )
        second = generate_transfer_candidates(
            state,
            list(reversed(market)),
            list(reversed(predictions)),
            rules=self.rules,
        )
        first_pairs = [
            (row["out_key"], row["in_key"]) for row in first["pair_candidates"]
        ]
        second_pairs = [
            (row["out_key"], row["in_key"]) for row in second["pair_candidates"]
        ]
        self.assertEqual(first_pairs, second_pairs)


if __name__ == "__main__":
    unittest.main()
