from __future__ import annotations

import copy
import unittest

import pandas as pd

from ml.contracts.opening_squad import build_default_opening_squad_objective_policy
from ml.decision.optimize_opening_lineup import (
    OpeningLineupOptimizerError,
    optimize_complete_lineup,
    validate_day101a_report,
)
from ml.decision.squad_rules import SquadLegalityEngine


TARGET_SEASON = "2026_27"


class Day101BOpeningLineupOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = SquadLegalityEngine.from_season(TARGET_SEASON)
        cls.policy = build_default_opening_squad_objective_policy(
            target_season=TARGET_SEASON,
            horizon_mode="gw1_gw5",
        )

    def make_fixed_squad(self) -> pd.DataFrame:
        positions = (
            ["GKP"] * 2
            + ["DEF"] * 5
            + ["MID"] * 5
            + ["FWD"] * 3
        )
        rows = []
        for index, position in enumerate(positions, start=1):
            team_id = ((index - 1) % 5) + 1
            rows.append(
                {
                    "variant": "primary",
                    "player_id": index,
                    "player_name": "Player %s" % index,
                    "web_name": "P%s" % index,
                    "team_id": team_id,
                    "team_name": "Team %s" % team_id,
                    "team_short_name": "T%s" % team_id,
                    "position": position,
                    "now_cost": 50,
                    "variant_total_cost_units": 750,
                    "variant_bank_units": 250,
                }
            )
        return pd.DataFrame(rows)

    def make_projections(self) -> pd.DataFrame:
        positions = (
            ["GKP"] * 2
            + ["DEF"] * 5
            + ["MID"] * 5
            + ["FWD"] * 3
        )
        rows = []
        for index, position in enumerate(positions, start=1):
            predicted_points = 8.0 - index * 0.20
            expected_minutes = 90.0 - (index % 4) * 5.0
            start_probability = 0.97 - (index % 4) * 0.04
            appearance_probability = min(1.0, start_probability + 0.02)
            rows.append(
                {
                    "player_id": index,
                    "target_gw": 1,
                    "predicted_points": predicted_points,
                    "expected_minutes": expected_minutes,
                    "start_probability": start_probability,
                    "appearance_probability": appearance_probability,
                    "has_fixture": True,
                    "fallback_used": False,
                    "fallback_level": 0,
                    "uncertainty_lower": predicted_points - 0.8,
                    "uncertainty_upper": predicted_points + 0.8,
                    "now_cost": 50,
                    "position": position,
                    "risk_flags": [],
                    "readiness_status": "preview_only",
                    "production_ready": False,
                    "selection_eligible": True,
                    "manual_review_required": False,
                }
            )
        return pd.DataFrame(rows)

    def safe_day101a_report(self):
        return {
            "artifact_type": "opening_squad_optimizer",
            "optimizer_version": "day101a_v1",
            "passed": True,
            "ready_for_day101b": True,
            "stop_point_satisfied": True,
            "preview_only": True,
            "production_approved": False,
            "writes_database": False,
            "writes_predictions_table": False,
            "writes_recommendations": False,
            "writes_squad_state": False,
            "target_season": TARGET_SEASON,
            "target_gw": 1,
            "variants": {
                "primary": {
                    "selected_player_ids": list(range(1, 16)),
                    "squad_legality": {"valid": True},
                    "objective_reconciliation": {"passed": True},
                    "objective_policy": self.policy.to_dict(),
                }
            },
            "scope_boundary": {
                "final_lineup_selected": False,
                "final_captain_selected": False,
                "final_vice_captain_selected": False,
                "final_bench_order_selected": False,
                "provisional_objective_assignment_only": True,
                "day101b_required": True,
            },
        }

    def optimize(self):
        return optimize_complete_lineup(
            squad=self.make_fixed_squad(),
            projections=self.make_projections(),
            policy=self.policy,
            engine=self.engine,
            bank_units=250,
        )

    def test_day101a_source_gate_passes_safe_report(self):
        validate_day101a_report(self.safe_day101a_report())

    def test_day101a_source_gate_fails_if_squad_state_writes_enabled(self):
        report = copy.deepcopy(self.safe_day101a_report())
        report["writes_squad_state"] = True
        with self.assertRaises(OpeningLineupOptimizerError):
            validate_day101a_report(report)

    def test_primary_is_complete_legal_fixed_squad_plan(self):
        plans = self.optimize()
        primary = plans["primary"]

        self.assertEqual(len(primary["starting_player_ids"]), 11)
        self.assertEqual(len(primary["bench_order"]), 4)
        self.assertEqual(
            set(primary["starting_player_ids"]) | set(primary["bench_order"]),
            set(range(1, 16)),
        )
        self.assertEqual(
            set(primary["starting_player_ids"]) & set(primary["bench_order"]),
            set(),
        )
        self.assertTrue(primary["legality"]["valid"])
        self.assertTrue(primary["objective_reconciliation"]["passed"])
        self.assertNotEqual(
            primary["captain_player_id"],
            primary["vice_captain_player_id"],
        )
        self.assertIn(
            primary["captain_player_id"],
            primary["starting_player_ids"],
        )
        self.assertIn(
            primary["vice_captain_player_id"],
            primary["starting_player_ids"],
        )
        self.assertEqual(
            primary["legality"]["lineup"]["bench_slot_contract"]["slot_0"],
            "substitute_goalkeeper",
        )

    def test_lineup_alternative_changes_starting_xi_and_is_legal(self):
        plans = self.optimize()
        primary = plans["primary"]
        alternative = plans["lineup_alternative"]

        self.assertNotEqual(
            set(primary["starting_player_ids"]),
            set(alternative["starting_player_ids"]),
        )
        self.assertTrue(alternative["legality"]["valid"])
        self.assertTrue(alternative["objective_reconciliation"]["passed"])
        self.assertEqual(
            set(alternative["starting_player_ids"]) | set(alternative["bench_order"]),
            set(range(1, 16)),
        )

    def test_captaincy_alternative_preserves_xi_and_bench(self):
        plans = self.optimize()
        primary = plans["primary"]
        alternative = plans["captaincy_alternative"]

        self.assertEqual(
            set(primary["starting_player_ids"]),
            set(alternative["starting_player_ids"]),
        )
        self.assertEqual(
            primary["bench_order"],
            alternative["bench_order"],
        )
        self.assertNotEqual(
            primary["captain_player_id"],
            alternative["captain_player_id"],
        )
        self.assertTrue(alternative["legality"]["valid"])
        self.assertTrue(alternative["objective_reconciliation"]["passed"])

    def test_optimizer_is_deterministic(self):
        first = self.optimize()
        second = self.optimize()

        for name in ("primary", "lineup_alternative", "captaincy_alternative"):
            self.assertEqual(
                first[name]["starting_player_ids"],
                second[name]["starting_player_ids"],
            )
            self.assertEqual(
                first[name]["bench_order"],
                second[name]["bench_order"],
            )
            self.assertEqual(
                first[name]["captain_player_id"],
                second[name]["captain_player_id"],
            )
            self.assertEqual(
                first[name]["vice_captain_player_id"],
                second[name]["vice_captain_player_id"],
            )


if __name__ == "__main__":
    unittest.main()
