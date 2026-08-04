from __future__ import annotations

import copy
import json
import unittest

from ml.contracts.opening_squad import (
    CONTRACT_VERSION,
    DEFAULT_POLICY_VERSION,
    HORIZON_GW1_GW5,
    HORIZON_GW1_ONLY,
    OpeningSquadObjectiveError,
    OpeningSquadObjectivePolicy,
    RiskPenaltyPolicy,
    RoleWeightPolicy,
    ValueBankPolicy,
    build_default_opening_squad_objective_policy,
    deterministic_objective_sort_key,
    evaluate_opening_squad_objective,
    reconcile_objective_evaluation,
)


PLAYER_LAYOUT = [
    ("G1", "GKP", 50),
    ("G2", "GKP", 45),
    ("D1", "DEF", 55),
    ("D2", "DEF", 50),
    ("D3", "DEF", 50),
    ("D4", "DEF", 45),
    ("D5", "DEF", 45),
    ("M1", "MID", 115),
    ("M2", "MID", 90),
    ("M3", "MID", 75),
    ("M4", "MID", 60),
    ("M5", "MID", 50),
    ("F1", "FWD", 125),
    ("F2", "FWD", 80),
    ("F3", "FWD", 55),
]
STARTERS = [
    "G1",
    "D1",
    "D2",
    "D3",
    "D4",
    "M1",
    "M2",
    "M3",
    "M4",
    "F1",
    "F2",
]
BENCH = ["G2", "D5", "M5", "F3"]
CAPTAIN = "M1"
VICE = "F1"


class OpeningSquadObjectiveContractTests(unittest.TestCase):
    def policy(self, horizon_mode=HORIZON_GW1_GW5, **overrides):
        kwargs = {
            "target_season": "2026_27",
            "horizon_mode": horizon_mode,
        }
        kwargs.update(overrides)
        return OpeningSquadObjectivePolicy(**kwargs)

    def rows(self, gameweeks=(1, 2, 3, 4, 5)):
        rows = []
        for player_index, (player_id, position, now_cost) in enumerate(
            PLAYER_LAYOUT
        ):
            for gw in gameweeks:
                predicted_points = 3.5 + player_index * 0.11 + gw * 0.23
                rows.append(
                    {
                        "target_season": "2026_27",
                        "target_gw": gw,
                        "player_id": player_id,
                        "position": position,
                        "now_cost": now_cost,
                        "predicted_points": predicted_points,
                        "expected_minutes": 72.0,
                        "start_probability": 0.82,
                        "appearance_probability": 0.94,
                        "has_fixture": True,
                        "selection_eligible": True,
                        "eligibility_reason": "eligible_preview_status",
                        "uncertainty_lower": predicted_points - 1.0,
                        "uncertainty_upper": predicted_points + 1.0,
                        "fallback_used": False,
                        "fallback_level": 0,
                        "risk_flags": [],
                        "readiness_status": "preview_only",
                        "production_ready": False,
                    }
                )
        return rows

    def evaluate(self, policy=None, rows=None, bank_units=10):
        return evaluate_opening_squad_objective(
            policy or self.policy(),
            rows or self.rows(),
            starting_player_ids=STARTERS,
            bench_order=BENCH,
            captain_player_id=CAPTAIN,
            vice_captain_player_id=VICE,
            bank_units=bank_units,
        )

    def row(self, rows, player_id, gw):
        return next(
            row
            for row in rows
            if row["player_id"] == player_id and row["target_gw"] == gw
        )

    def contribution(self, result, player_id, gw):
        return next(
            contribution
            for contribution in result["by_player"][player_id][
                "contributions"
            ]
            if contribution["target_gw"] == gw
        )

    def test_default_fast_lane_policy_is_explicit_and_read_only(self):
        policy = build_default_opening_squad_objective_policy("2026_27")
        self.assertEqual(policy.contract_version, CONTRACT_VERSION)
        self.assertEqual(policy.policy_version, DEFAULT_POLICY_VERSION)
        self.assertEqual(policy.horizon_mode, HORIZON_GW1_GW5)
        self.assertEqual(policy.requested_gameweeks, (1, 2, 3, 4, 5))
        self.assertEqual(policy.recommendation_status, "preview_only")
        self.assertFalse(policy.writes_enabled)
        self.assertEqual(policy.value_bank.value_bonus_weight, 0.0)
        self.assertEqual(policy.value_bank.bank_bonus_per_unit, 0.0)

    def test_policy_json_round_trip_is_stable(self):
        policy = self.policy()
        decoded = OpeningSquadObjectivePolicy.from_json(policy.to_json())
        self.assertEqual(decoded.to_dict(), policy.to_dict())
        self.assertEqual(json.loads(decoded.to_json()), json.loads(policy.to_json()))

    def test_gw1_only_mode_uses_only_gw1(self):
        result = self.evaluate(policy=self.policy(HORIZON_GW1_ONLY))
        self.assertEqual(result["effective_gameweeks"], [1])
        self.assertEqual(sorted(result["by_gameweek"]), ["1"])
        self.assertFalse(result["horizon_fallback_used"])

    def test_gw1_gw5_discount_schedule_is_explicit(self):
        policy = self.policy()
        result = self.evaluate(policy=policy)
        self.assertEqual(
            [
                result["by_gameweek"][str(gw)]["discount"]
                for gw in range(1, 6)
            ],
            [policy.gameweek_discounts[gw] for gw in range(1, 6)],
        )

    def test_objective_reconciliation_passes(self):
        result = self.evaluate()
        reconciliation = reconcile_objective_evaluation(result)
        self.assertTrue(result["reconciliation"]["passed"])
        self.assertTrue(reconciliation["passed"])
        self.assertEqual(reconciliation["difference"], 0.0)

    def test_captain_receives_additional_starter_weight(self):
        result = self.evaluate(policy=self.policy(HORIZON_GW1_ONLY))
        captain = self.contribution(result, CAPTAIN, 1)
        normal_starter = self.contribution(result, "M2", 1)
        self.assertEqual(captain["role_weight"], 2.0)
        self.assertEqual(normal_starter["role_weight"], 1.0)
        self.assertAlmostEqual(
            captain["gross_expected_points"],
            captain["predicted_points"] * 2.0,
        )

    def test_vice_captain_default_has_no_deterministic_bonus(self):
        result = self.evaluate(policy=self.policy(HORIZON_GW1_ONLY))
        vice = self.contribution(result, VICE, 1)
        self.assertTrue(vice["is_vice_captain"])
        self.assertEqual(vice["role_weight"], 1.0)
        self.assertIn(
            "contingency_only",
            result["explanation"]["vice_captain_treatment"],
        )

    def test_bench_goalkeeper_and_outfield_weights_are_ordered(self):
        result = self.evaluate(policy=self.policy(HORIZON_GW1_ONLY))
        self.assertEqual(self.contribution(result, "G2", 1)["role_weight"], 0.05)
        self.assertEqual(self.contribution(result, "D5", 1)["role_weight"], 0.12)
        self.assertEqual(self.contribution(result, "M5", 1)["role_weight"], 0.06)
        self.assertEqual(self.contribution(result, "F3", 1)["role_weight"], 0.03)

    def test_minutes_and_start_risk_penalties_are_separate(self):
        rows = self.rows(gameweeks=(1,))
        target = self.row(rows, "D1", 1)
        target["expected_minutes"] = 30.0
        target["start_probability"] = 0.30
        result = self.evaluate(
            policy=self.policy(HORIZON_GW1_ONLY),
            rows=rows,
        )
        contribution = self.contribution(result, "D1", 1)
        self.assertGreater(contribution["minutes_risk_penalty"], 0.0)
        self.assertGreater(contribution["start_risk_penalty"], 0.0)

    def test_fallback_and_uncertainty_penalties_are_separate(self):
        rows = self.rows(gameweeks=(1,))
        target = self.row(rows, "D1", 1)
        target["fallback_used"] = True
        target["fallback_level"] = 2
        target["uncertainty_lower"] = 1.0
        target["uncertainty_upper"] = 6.0
        result = self.evaluate(
            policy=self.policy(HORIZON_GW1_ONLY),
            rows=rows,
        )
        contribution = self.contribution(result, "D1", 1)
        self.assertGreater(contribution["fallback_penalty"], 0.0)
        self.assertGreater(contribution["uncertainty_penalty"], 0.0)
        self.assertTrue(result["manual_review_required"])

    def test_missing_uncertainty_is_penalized_and_disclosed(self):
        rows = self.rows(gameweeks=(1,))
        target = self.row(rows, "D1", 1)
        target["uncertainty_lower"] = None
        target["uncertainty_upper"] = None
        result = self.evaluate(
            policy=self.policy(HORIZON_GW1_ONLY),
            rows=rows,
        )
        contribution = self.contribution(result, "D1", 1)
        self.assertGreater(contribution["uncertainty_penalty"], 0.0)
        self.assertTrue(result["manual_review_required"])
        self.assertIn(
            "player_uncertainty_missing",
            {reason["code"] for reason in result["manual_review_reasons"]},
        )

    def test_missing_fixture_is_disclosed_for_manual_review(self):
        rows = self.rows(gameweeks=(1,))
        target = self.row(rows, "D1", 1)
        target["has_fixture"] = False

        result = self.evaluate(
            policy=self.policy(HORIZON_GW1_ONLY),
            rows=rows,
        )
        contribution = self.contribution(result, "D1", 1)
        reasons = [
            reason
            for reason in result["manual_review_reasons"]
            if reason["code"]
            == "player_fixture_missing_or_unconfirmed"
        ]

        self.assertFalse(contribution["has_fixture"])
        self.assertTrue(result["manual_review_required"])
        self.assertEqual(len(reasons), 1)
        self.assertEqual(reasons[0]["player_id"], "D1")
        self.assertEqual(reasons[0]["target_gw"], 1)
        self.assertEqual(result["recommendation_status"], "preview_only")
        self.assertFalse(result["writes_enabled"])
        self.assertTrue(result["reconciliation"]["passed"])
        self.assertIn(
            "has_fixture=False",
            result["explanation"]["fixture_treatment"],
        )

    def test_incomplete_primary_horizon_falls_back_to_gw1(self):
        result = self.evaluate(rows=self.rows(gameweeks=(1, 2, 3, 4)))
        self.assertEqual(result["requested_gameweeks"], [1, 2, 3, 4, 5])
        self.assertEqual(result["effective_gameweeks"], [1])
        self.assertTrue(result["horizon_fallback_used"])
        self.assertTrue(result["manual_review_required"])
        self.assertEqual(result["recommendation_status"], "preview_only")

    def test_missing_gw1_projection_fails_closed(self):
        rows = [
            row
            for row in self.rows()
            if not (row["player_id"] == "D1" and row["target_gw"] == 1)
        ]
        with self.assertRaisesRegex(
            OpeningSquadObjectiveError,
            "GW1 projections are missing",
        ):
            self.evaluate(rows=rows)

    def test_default_policy_does_not_force_full_budget_spend(self):
        first = self.evaluate(bank_units=0)
        second = self.evaluate(bank_units=35)
        self.assertEqual(first["totals"]["bank_bonus"], 0.0)
        self.assertEqual(second["totals"]["bank_bonus"], 0.0)
        self.assertEqual(
            first["totals"]["objective_value"],
            second["totals"]["objective_value"],
        )

    def test_optional_value_and_bank_behavior_is_configurable(self):
        policy = self.policy(
            HORIZON_GW1_ONLY,
            value_bank=ValueBankPolicy(
                value_bonus_weight=2.0,
                bank_bonus_per_unit=0.01,
                use_bank_as_tie_breaker=True,
            ),
        )
        result = self.evaluate(policy=policy, bank_units=10)
        self.assertGreater(result["totals"]["value_bonus"], 0.0)
        self.assertEqual(result["totals"]["bank_bonus"], 0.1)

    def test_tie_break_key_is_deterministic(self):
        first = self.evaluate()
        second = self.evaluate(rows=list(reversed(self.rows())))
        self.assertEqual(
            deterministic_objective_sort_key(first),
            deterministic_objective_sort_key(second),
        )
        self.assertEqual(first["tie_break_key"], second["tie_break_key"])

    def test_ineligible_projection_row_fails_closed(self):
        rows = self.rows()
        self.row(rows, "D1", 1)["selection_eligible"] = False
        with self.assertRaisesRegex(
            OpeningSquadObjectiveError,
            "not selection eligible",
        ):
            self.evaluate(rows=rows)

    def test_duplicate_player_gameweek_row_fails_closed(self):
        rows = self.rows()
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(
            OpeningSquadObjectiveError,
            "Duplicate projection row",
        ):
            self.evaluate(rows=rows)

    def test_fast_lane_cannot_enable_writes(self):
        with self.assertRaisesRegex(
            OpeningSquadObjectiveError,
            "writes must remain disabled",
        ):
            self.policy(writes_enabled=True)

    def test_custom_role_and_risk_policies_reconcile(self):
        policy = self.policy(
            HORIZON_GW1_ONLY,
            role_weights=RoleWeightPolicy(
                starter_weight=0.95,
                captain_bonus_weight=0.90,
                vice_captain_bonus_weight=0.05,
                bench_goalkeeper_weight=0.08,
                bench_outfield_weights=(0.15, 0.10, 0.05),
            ),
            risk_penalties=RiskPenaltyPolicy(
                expected_minutes_target=70.0,
                minutes_shortfall_penalty_per_minute=0.02,
                start_probability_target=0.85,
                start_probability_shortfall_penalty=1.0,
                fallback_used_penalty=0.5,
                fallback_level_penalty=0.2,
                uncertainty_width_penalty=0.1,
                missing_uncertainty_penalty=0.8,
            ),
        )
        result = self.evaluate(policy=policy)
        self.assertTrue(result["reconciliation"]["passed"])
        self.assertEqual(
            self.contribution(result, CAPTAIN, 1)["role_weight"],
            1.85,
        )


if __name__ == "__main__":
    unittest.main()
