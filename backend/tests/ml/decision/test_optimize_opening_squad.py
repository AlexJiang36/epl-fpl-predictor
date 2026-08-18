from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from app.rules.squad import SquadTransferRules
from ml.contracts.opening_squad import build_default_opening_squad_objective_policy
from ml.decision.optimize_opening_squad import (
    OpeningSquadOptimizerError,
    aggregate_availability_risk,
    build_candidate_audit,
    build_player_metadata,
    build_variant_policies,
    normalize_long_horizon,
    normalize_projection_rows,
    projection_records,
    solve_with_relaxations,
    validate_and_evaluate_variant,
    validate_source_report,
)
from ml.decision.squad_rules import SquadLegalityEngine


class Day101AOpeningSquadOptimizerTests(unittest.TestCase):
    def build_rules(self) -> SquadTransferRules:
        data = {
            "units": {"initial_budget_units": 1000},
            "squad": {
                "size": 15,
                "position_quotas": {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
                "max_players_per_club": 3,
            },
            "lineup": {
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
            },
            "transfers": {},
        }
        return SquadTransferRules(
            effective_season="2026_27",
            rules_version="fpl_2026_27_squad_transfer_v1",
            schema_version="fpl_squad_transfer_rules_v1",
            path=Path("/tmp/test_squad_transfer_rules_2026_27.json"),
            sha256="0" * 64,
            data=data,
        )

    def make_frames(self):
        specs = [("GKP", 5), ("DEF", 12), ("MID", 12), ("FWD", 8)]
        projection_rows = []
        long_rows = []
        player_id = 1
        for position, count in specs:
            for idx in range(count):
                team_id = ((player_id - 1) % 14) + 1
                now_cost = 40 + (idx % 5) * 5
                points = 7.2 - 0.07 * player_id + (0.25 if position == "MID" else 0.0)
                expected_minutes = 86.0 - (idx % 4) * 5.0
                start_probability = 0.94 - (idx % 4) * 0.05
                fallback_used = (idx % 5) == 4
                fallback_level = 4 if fallback_used else 0
                projection_rows.append(
                    {
                        "player_id": player_id,
                        "target_gw": 1,
                        "predicted_points": points,
                        "expected_minutes": expected_minutes,
                        "start_probability": start_probability,
                        "appearance_probability": min(1.0, start_probability + 0.04),
                        "has_fixture": True,
                        "fallback_used": fallback_used,
                        "fallback_level": fallback_level,
                        "uncertainty_lower": points - 1.0,
                        "uncertainty_upper": points + 1.0,
                        "now_cost": now_cost,
                        "position": position,
                        "risk_flags": "[]",
                        "readiness_status": "preview_only",
                        "production_ready": False,
                        "selection_eligible": True,
                        "manual_review_required": fallback_used,
                    }
                )
                long_rows.append(
                    {
                        "player_id": player_id,
                        "fpl_player_id": 1000 + player_id,
                        "target_gw": 1,
                        "player_name": "Player %s" % player_id,
                        "web_name": "P%s" % player_id,
                        "team_id": team_id,
                        "team_name": "Team %s" % team_id,
                        "team_short_name": "T%s" % team_id,
                        "position": position,
                        "now_cost": now_cost,
                        "selection_eligible": True,
                        "prediction_available": True,
                        "row_status": "prediction_available",
                        "fixture_eligibility_reason": "eligible_current_player_and_fixture",
                        "manual_review_required": fallback_used,
                    }
                )
                player_id += 1

        # Add two visible but excluded current players to the long horizon.
        for position, reason in (("DEF", "current_player_ineligible"), ("MID", "current_player_ineligible")):
            team_id = ((player_id - 1) % 14) + 1
            long_rows.append(
                {
                    "player_id": player_id,
                    "fpl_player_id": 1000 + player_id,
                    "target_gw": 1,
                    "player_name": "Excluded %s" % player_id,
                    "web_name": "X%s" % player_id,
                    "team_id": team_id,
                    "team_name": "Team %s" % team_id,
                    "team_short_name": "T%s" % team_id,
                    "position": position,
                    "now_cost": 45,
                    "selection_eligible": False,
                    "prediction_available": True,
                    "row_status": "prediction_available_but_ineligible",
                    "fixture_eligibility_reason": reason,
                    "manual_review_required": True,
                }
            )
            player_id += 1
        return pd.DataFrame(projection_rows), pd.DataFrame(long_rows)

    def test_source_report_requires_safe_gw1_fallback_contract(self):
        report = {
            "passed": True,
            "ready_for_day101a": True,
            "stop_point_satisfied": True,
            "preview_only": True,
            "recommendation_status": "preview_only",
            "production_approved": False,
            "writes_database": False,
            "writes_predictions_table": False,
            "writes_recommendations": False,
            "writes_squad_state": False,
            "target_season": "2026_27",
            "start_gw": 1,
            "objective_mode": "gw1_only_fallback",
            "effective_horizon": 1,
            "requested_horizon": 5,
            "future_fixture_context": "manual_review_only",
            "as_of_time_utc": "2026-08-18T04:04:45Z",
        }
        validate_source_report(report)
        unsafe = dict(report)
        unsafe["production_approved"] = True
        with self.assertRaises(OpeningSquadOptimizerError):
            validate_source_report(unsafe)

    def test_candidate_audit_preserves_exclusion_reasons(self):
        raw_projection, raw_long = self.make_frames()
        projection = normalize_projection_rows(raw_projection)
        long_frame = normalize_long_horizon(raw_long)
        audit = build_candidate_audit(long_frame, projection, start_gw=1)
        self.assertEqual(int(audit["candidate_eligible"].sum()), len(projection))
        excluded = audit[~audit["candidate_eligible"]]
        self.assertEqual(len(excluded), 2)
        self.assertTrue(excluded["exclusion_reasons"].str.contains("selection_ineligible").all())

    def test_primary_and_two_alternatives_are_legal_and_distinct(self):
        raw_projection, raw_long = self.make_frames()
        projection = normalize_projection_rows(raw_projection)
        long_frame = normalize_long_horizon(raw_long)
        metadata = build_player_metadata(long_frame, projection, start_gw=1)
        rules = self.build_rules()
        engine = SquadLegalityEngine(rules)
        policies = build_variant_policies("2026_27")
        records = projection_records(projection)

        primary = solve_with_relaxations(
            variant="primary",
            policy=policies["primary"],
            projection_frame=projection,
            metadata=metadata,
            rules=rules,
            effective_gameweeks=[1],
        )
        primary = validate_and_evaluate_variant(
            variant_result=primary,
            policy=policies["primary"],
            projection_records_all=records,
            metadata=metadata,
            engine=engine,
        )
        primary_ids = set(primary["selected_player_ids"])

        alt_a = solve_with_relaxations(
            variant="alternative_a",
            policy=policies["alternative_a"],
            projection_frame=projection,
            metadata=metadata,
            rules=rules,
            effective_gameweeks=[1],
            primary_ids=primary_ids,
        )
        alt_a = validate_and_evaluate_variant(
            variant_result=alt_a,
            policy=policies["alternative_a"],
            projection_records_all=records,
            metadata=metadata,
            engine=engine,
        )
        alt_a_ids = set(alt_a["selected_player_ids"])

        alt_b = solve_with_relaxations(
            variant="alternative_b",
            policy=policies["alternative_b"],
            projection_frame=projection,
            metadata=metadata,
            rules=rules,
            effective_gameweeks=[1],
            primary_ids=primary_ids,
            alternative_a_ids=alt_a_ids,
        )
        alt_b = validate_and_evaluate_variant(
            variant_result=alt_b,
            policy=policies["alternative_b"],
            projection_records_all=records,
            metadata=metadata,
            engine=engine,
        )
        alt_b_ids = set(alt_b["selected_player_ids"])

        for result in (primary, alt_a, alt_b):
            self.assertEqual(len(result["selected_player_ids"]), 15)
            self.assertTrue(result["squad_legality"]["valid"])
            self.assertTrue(result["provisional_plan_legality"]["valid"])
            self.assertTrue(result["objective_reconciliation"]["passed"])
            self.assertTrue(result["objective_evaluation"]["horizon_fallback_used"])
            self.assertEqual(result["objective_evaluation"]["effective_gameweeks"], [1])
            self.assertLessEqual(result["total_cost_units"], 1000)
            self.assertFalse(result["objective_evaluation_plan"]["is_final_day101b_decision"])

        self.assertNotEqual(primary_ids, alt_a_ids)
        self.assertNotEqual(primary_ids, alt_b_ids)
        self.assertNotEqual(alt_a_ids, alt_b_ids)
        self.assertLessEqual(len(primary_ids & alt_a_ids), 14)
        self.assertLessEqual(len(primary_ids & alt_b_ids), 14)
        self.assertLessEqual(len(alt_a_ids & alt_b_ids), 14)

        alt_b_risks = aggregate_availability_risk(
            policies["alternative_b"], projection, [1]
        )
        primary_risk = sum(alt_b_risks[player_id] for player_id in primary_ids)
        alternative_b_risk = sum(alt_b_risks[player_id] for player_id in alt_b_ids)
        self.assertLess(alternative_b_risk, primary_risk * 0.99 + 1e-8)
        self.assertLess(
            alt_b["alternative_design"]["selected_availability_risk_score"],
            alt_b["alternative_design"]["primary_availability_risk_score"],
        )

    def test_primary_solver_is_deterministic(self):
        raw_projection, raw_long = self.make_frames()
        projection = normalize_projection_rows(raw_projection)
        long_frame = normalize_long_horizon(raw_long)
        metadata = build_player_metadata(long_frame, projection, start_gw=1)
        rules = self.build_rules()
        policy = build_variant_policies("2026_27")["primary"]
        first = solve_with_relaxations(
            variant="primary",
            policy=policy,
            projection_frame=projection,
            metadata=metadata,
            rules=rules,
            effective_gameweeks=[1],
        )
        second = solve_with_relaxations(
            variant="primary",
            policy=policy,
            projection_frame=projection,
            metadata=metadata,
            rules=rules,
            effective_gameweeks=[1],
        )
        self.assertEqual(first["selected_player_ids"], second["selected_player_ids"])
        self.assertEqual(
            first["objective_evaluation_plan"],
            second["objective_evaluation_plan"],
        )

    def test_default_contract_itself_allows_gw1_fallback(self):
        policy = build_default_opening_squad_objective_policy("2026_27", "gw1_gw5")
        self.assertTrue(policy.allow_gw1_fallback)
        self.assertEqual(policy.recommendation_status, "preview_only")
        self.assertFalse(policy.writes_enabled)


if __name__ == "__main__":
    unittest.main()
