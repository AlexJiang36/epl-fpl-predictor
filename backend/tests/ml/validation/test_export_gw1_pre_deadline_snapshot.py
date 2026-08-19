from __future__ import annotations

import copy
import unittest

import pandas as pd

from ml.validation.export_gw1_pre_deadline_snapshot import (
    GW1SnapshotError,
    artifact_definitions,
    build_global_prediction_snapshot,
    build_opening_squad_markdown,
    build_validation_markdown,
    build_validation_report,
    build_version_audit,
    evaluation_contract,
    normalize_primary_lineup,
    reconcile_primary_with_final_refresh,
    validate_day101b_report,
)


TARGET_SEASON = "2026_27"


class Day101CGW1SnapshotTests(unittest.TestCase):
    def safe_day101b_report(self):
        return {
            "artifact_type": "opening_lineup_optimizer",
            "optimizer_version": "day101b_v1",
            "passed": True,
            "ready_for_day101c": True,
            "stop_point_satisfied": True,
            "preview_only": True,
            "production_approved": False,
            "writes_database": False,
            "writes_predictions_table": False,
            "writes_recommendations": False,
            "writes_squad_state": False,
            "recommendation_status": "preview_only",
            "blockers": [],
            "target_season": TARGET_SEASON,
            "target_gw": 1,
            "availability_scope": {
                "official_chance_of_playing_next_round_propagated": True,
                "official_availability_consumed_via_adjusted_projection_inputs": True,
                "day101b_additional_official_availability_penalty_applied": False,
                "required_follow_up": None,
            },
            "plans": {
                "primary": {
                    "legality": {"valid": True},
                    "objective_reconciliation": {"passed": True},
                }
            },
            "run_metadata": {
                "versions": {
                    "model_version": "pre_gw1_heuristic_preview",
                    "rules_versions": {"squad_transfer": "rules_v1"},
                }
            },
        }

    def safe_day101a_report(self):
        policy = {
            "contract_version": "opening_squad_objective_v1",
            "policy_version": "fast_lane_policy_v1",
        }
        variants = {}
        for variant in ("primary", "alternative_a", "alternative_b"):
            variants[variant] = {
                "squad_legality": {"valid": True},
                "objective_reconciliation": {"passed": True},
                "objective_policy": policy,
            }
        return {
            "preview_only": True,
            "production_approved": False,
            "writes_database": False,
            "writes_squad_state": False,
            "optimizer_version": "day101a_v1",
            "variants": variants,
        }

    def safe_day97a_report(self):
        return {
            "preview_only": True,
            "production_approved": False,
            "writes_database": False,
            "horizon_version": "day97a_v1",
            "horizon_schema_version": "fpl_player_prediction_horizon_v1",
            "objective_mode": "gw1_only_fallback",
            "run_metadata": {"versions": {"feature_version": "day97a_v1"}},
        }

    def safe_day76d_report(self):
        return {
            "writes_database": False,
            "writes_predictions_table": False,
            "production_approved": False,
            "historical_multi_season_backtest_complete": False,
            "component_model_stack_complete": False,
            "refresh_version": "day76d_v1",
            "run_metadata": {
                "versions": {
                    "feature_version": "day71a_v0_1",
                    "model_version": "day72a_v0_2",
                }
            },
        }

    def make_global_rows(self):
        rows = []
        for player_id in range(1, 6):
            rows.append(
                {
                    "target_season": TARGET_SEASON,
                    "target_gw": 1,
                    "player_id": player_id,
                    "player_name": "Player %s" % player_id,
                    "web_name": "P%s" % player_id,
                    "team_id": 1,
                    "team_name": "Team",
                    "team_short_name": "TM",
                    "position": "MID",
                    "now_cost": 50,
                    "selection_eligible": player_id != 5,
                    "prediction_available": True,
                    "prediction_status": "available_preview",
                    "predicted_points": 5.0 + player_id,
                    "expected_minutes": 80.0,
                    "appearance_probability": 0.95,
                    "start_probability": 0.90,
                    "fallback_used": True,
                    "fallback_level": 1,
                    "risk_flags": [],
                    "manual_review_required": False,
                    "source_prediction_run_id": "prediction_run",
                    "source_fixture_horizon_run_id": "fixture_run",
                    "recommendation_status": "preview_only",
                    "production_ready": False,
                    "production_approved": False,
                    "prediction_write_allowed": False,
                }
            )
        for player_id in range(1, 6):
            row = dict(rows[player_id - 1])
            row["target_gw"] = 2
            row["prediction_available"] = False
            row["prediction_status"] = "missing_future_prediction"
            row["predicted_points"] = None
            row["expected_minutes"] = None
            row["appearance_probability"] = None
            row["start_probability"] = None
            rows.append(row)
        return pd.DataFrame(rows)

    def make_reconciliation_frames(self):
        squad_rows = []
        lineup_rows = []
        preview_rows = []
        standard_rows = []
        horizon_map = {}
        for player_id in range(1, 16):
            position = (
                "GKP" if player_id <= 2
                else "DEF" if player_id <= 7
                else "MID" if player_id <= 12
                else "FWD"
            )
            squad_rows.append(
                {
                    "variant": "primary",
                    "variant_label": "balanced_risk_adjusted",
                    "player_id": player_id,
                    "fpl_player_id": player_id,
                    "player_name": "Player %s" % player_id,
                    "web_name": "P%s" % player_id,
                    "team_id": ((player_id - 1) % 5) + 1,
                    "team_name": "Team %s" % (((player_id - 1) % 5) + 1),
                    "team_short_name": "T%s" % (((player_id - 1) % 5) + 1),
                    "position": position,
                    "now_cost": 50,
                    "gw1_predicted_points": 5.0,
                    "expected_minutes": 80.0,
                    "start_probability": 0.9,
                    "appearance_probability": 0.95,
                    "fallback_used": True,
                    "fallback_level": 1,
                    "risk_flags": "[]",
                    "manual_review_required": False,
                    "variant_total_cost_units": 750,
                    "variant_bank_units": 250,
                    "variant_objective_value": 50.0,
                }
            )
            lineup_rows.append(
                {
                    "plan": "primary",
                    "player_id": player_id,
                    "player_name": "Player %s" % player_id,
                    "web_name": "P%s" % player_id,
                    "team_short_name": "T%s" % (((player_id - 1) % 5) + 1),
                    "position": position,
                    "now_cost": 50,
                    "role": "starter" if player_id <= 11 else "bench_%s" % (player_id - 11),
                    "is_starter": player_id <= 11,
                    "is_captain": player_id == 1,
                    "is_vice_captain": player_id == 2,
                    "formation": "3-5-2",
                    "predicted_points": 5.0,
                    "expected_minutes": 80.0,
                    "start_probability": 0.9,
                    "appearance_probability": 0.95,
                    "fallback_used": True,
                    "fallback_level": 1,
                    "risk_flags": "[]",
                    "manual_review_required": False,
                }
            )
            preview_rows.append(
                {
                    "player_id": player_id,
                    "player_name": "Player %s" % player_id,
                    "web_name": "P%s" % player_id,
                    "team_id": ((player_id - 1) % 5) + 1,
                    "team_name": "Team %s" % (((player_id - 1) % 5) + 1),
                    "team_short_name": "T%s" % (((player_id - 1) % 5) + 1),
                    "position": position,
                    "now_cost": 50,
                    "status": "a",
                    "chance_of_playing_next_round": None,
                    "news": "",
                    "news_added": "",
                    "official_availability_probability": None,
                    "official_availability_workload_factor": 1.0,
                    "official_availability_adjustment_applied": False,
                    "appearance_probability": 0.95,
                    "start_probability": 0.9,
                    "expected_minutes": 80.0,
                    "predicted_points": 5.0,
                    "prediction_confidence": "medium",
                    "fallback_policy_used": "fallback",
                    "fallback_level": 1,
                    "fallback_reason": "pre_gw1",
                    "risk_flags": "",
                    "status_cutoff_valid": True,
                    "status_hard_guardrail_applied": False,
                    "prediction_write_allowed": False,
                    "production_ready": False,
                }
            )
            standard_rows.append(
                {
                    "player_id": player_id,
                    "position": position,
                    "now_cost": 50,
                    "selection_eligible": True,
                    "appearance_probability": 0.95,
                    "start_probability": 0.9,
                    "expected_minutes": 80.0,
                    "predicted_points": 5.0,
                    "risk_flags": "",
                    "prediction_write_allowed": False,
                    "production_ready": False,
                }
            )
            horizon_map[player_id] = {
                "gw1": {
                    "prediction_available": True,
                    "predicted_points": 5.0,
                    "opponent": "OPP",
                    "home_away": "H",
                    "row_status": "available_preview",
                },
                "gw2": {"prediction_available": False, "predicted_points": None, "opponent": "OP2", "home_away": "A", "row_status": "missing_future_prediction"},
                "gw3": {"prediction_available": False, "predicted_points": None, "opponent": "OP3", "home_away": "H", "row_status": "missing_future_prediction"},
                "gw4": {"prediction_available": False, "predicted_points": None, "opponent": "OP4", "home_away": "A", "row_status": "missing_future_prediction"},
                "gw5": {"prediction_available": False, "predicted_points": None, "opponent": "OP5", "home_away": "H", "row_status": "missing_future_prediction"},
            }
        return (
            pd.DataFrame(squad_rows),
            pd.DataFrame(lineup_rows),
            pd.DataFrame(preview_rows),
            pd.DataFrame(standard_rows),
            horizon_map,
        )

    def safe_model_audit(self):
        return {
            "selected_player_count": 15,
            "duplicate_player_count": 0,
            "position_counts": {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
            "expected_position_quotas": {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
            "position_quotas_pass": True,
            "club_counts": {"1": 3, "2": 3, "3": 3, "4": 3, "5": 3},
            "max_players_per_club": 3,
            "club_limits_pass": True,
            "total_cost_units": 1000,
            "bank_units": 0,
            "initial_budget_units": 1000,
            "budget_and_bank_reconcile": True,
            "starting_xi_count": 11,
            "formation": "3-5-2",
            "bench_order": [12, 13, 14, 15],
            "bench_count": 4,
            "captain_player_id": 1,
            "vice_captain_player_id": 2,
            "captain_and_vice_distinct_starting_members": True,
            "legality": {"valid": True},
            "objective_value": 50.0,
            "objective_reconciliation": {"passed": True},
        }

    def test_day101b_source_gate_passes_current_availability_metadata(self):
        validate_day101b_report(self.safe_day101b_report())

    def test_day101b_source_gate_rejects_stale_availability_metadata(self):
        report = copy.deepcopy(self.safe_day101b_report())
        report["availability_scope"][
            "official_chance_of_playing_next_round_propagated"
        ] = False
        with self.assertRaises(GW1SnapshotError):
            validate_day101b_report(report)

    def test_day101b_source_gate_rejects_write_enabled_source(self):
        report = copy.deepcopy(self.safe_day101b_report())
        report["writes_squad_state"] = True
        with self.assertRaises(GW1SnapshotError):
            validate_day101b_report(report)

    def test_day101b_lineup_schema_does_not_require_duplicated_team_identity(self):
        _, lineup, _, _, _ = self.make_reconciliation_frames()
        report = self.safe_day101b_report()
        report["plans"]["primary"].update(
            {
                "starting_player_ids": list(range(1, 12)),
                "bench_order": [12, 13, 14, 15],
                "captain_player_id": 1,
                "vice_captain_player_id": 2,
            }
        )
        normalized = normalize_primary_lineup(lineup, report)
        self.assertEqual(len(normalized), 15)
        self.assertNotIn("team_id", normalized.columns)
        self.assertNotIn("team_name", normalized.columns)

    def test_global_snapshot_keeps_all_gw1_players_including_ineligible(self):
        snapshot, audit = build_global_prediction_snapshot(
            self.make_global_rows(), TARGET_SEASON
        )
        self.assertEqual(len(snapshot), 5)
        self.assertEqual(audit["player_count"], 5)
        self.assertEqual(audit["eligible_player_count"], 4)
        self.assertEqual(audit["ineligible_player_count"], 1)
        self.assertEqual(audit["prediction_available_count"], 5)
        self.assertEqual(snapshot["target_gw"].unique().tolist(), [1])

    def test_global_snapshot_rejects_missing_required_gw1_prediction(self):
        frame = self.make_global_rows()
        frame.loc[
            (frame["target_gw"] == 1) & (frame["player_id"] == 3),
            "prediction_available",
        ] = False
        with self.assertRaises(GW1SnapshotError):
            build_global_prediction_snapshot(frame, TARGET_SEASON)

    def test_global_snapshot_rejects_production_or_write_flags(self):
        frame = self.make_global_rows()
        frame.loc[
            (frame["target_gw"] == 1) & (frame["player_id"] == 2),
            "prediction_write_allowed",
        ] = True
        with self.assertRaises(GW1SnapshotError):
            build_global_prediction_snapshot(frame, TARGET_SEASON)

    def test_final_refresh_reconciliation_passes_exact_source_values(self):
        squad, lineup, preview, standard, horizon_map = self.make_reconciliation_frames()
        exported, audit = reconcile_primary_with_final_refresh(
            squad, lineup, preview, standard, horizon_map
        )
        self.assertEqual(len(exported), 15)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["mismatch_count"], 0)
        self.assertIn("status", exported.columns)
        self.assertIn("chance_of_playing_next_round", exported.columns)
        self.assertIn("prediction_confidence", exported.columns)
        self.assertIn("gw5_predicted_points", exported.columns)

    def test_final_refresh_reconciliation_rejects_price_or_availability_drift(self):
        squad, lineup, preview, standard, horizon_map = self.make_reconciliation_frames()
        preview.loc[preview["player_id"] == 4, "now_cost"] = 55
        preview.loc[preview["player_id"] == 5, "appearance_probability"] = 0.75
        _, audit = reconcile_primary_with_final_refresh(
            squad, lineup, preview, standard, horizon_map
        )
        self.assertFalse(audit["passed"])
        self.assertGreaterEqual(audit["mismatch_count"], 2)

    def test_version_audit_records_required_five_groups(self):
        audit = build_version_audit(
            self.safe_day76d_report(),
            self.safe_day97a_report(),
            self.safe_day101a_report(),
            self.safe_day101b_report(),
        )
        self.assertIn("rules", audit)
        self.assertIn("feature", audit)
        self.assertIn("prediction", audit)
        self.assertIn("objective", audit)
        self.assertIn("artifact", audit)
        self.assertTrue(audit["all_required_version_groups_recorded"])

    def test_validation_requires_as_of_before_fpl_deadline(self):
        version_audit = build_version_audit(
            self.safe_day76d_report(),
            self.safe_day97a_report(),
            self.safe_day101a_report(),
            self.safe_day101b_report(),
        )
        source_metadata = {"source": {"sha256": "abc"}}
        validation = build_validation_report(
            model_audit=self.safe_model_audit(),
            final_refresh_audit={"passed": True, "mismatch_count": 0},
            version_audit=version_audit,
            source_metadata=source_metadata,
            as_of_time="2026-08-20T12:00:00Z",
            fpl_deadline_time="2026-08-21T17:30:00Z",
            day76d_report=self.safe_day76d_report(),
            day97a_report=self.safe_day97a_report(),
            day101a_report=self.safe_day101a_report(),
            day101b_report=self.safe_day101b_report(),
            final_freeze=False,
        )
        self.assertTrue(validation["checks"]["as_of_time_before_fpl_deadline"]["passed"])
        self.assertTrue(validation["passed"])

        late = build_validation_report(
            model_audit=self.safe_model_audit(),
            final_refresh_audit={"passed": True, "mismatch_count": 0},
            version_audit=version_audit,
            source_metadata=source_metadata,
            as_of_time="2026-08-21T18:00:00Z",
            fpl_deadline_time="2026-08-21T17:30:00Z",
            day76d_report=self.safe_day76d_report(),
            day97a_report=self.safe_day97a_report(),
            day101a_report=self.safe_day101a_report(),
            day101b_report=self.safe_day101b_report(),
            final_freeze=False,
        )
        self.assertFalse(late["checks"]["as_of_time_before_fpl_deadline"]["passed"])
        self.assertFalse(late["passed"])

    def test_validation_requires_input_fingerprints(self):
        version_audit = build_version_audit(
            self.safe_day76d_report(),
            self.safe_day97a_report(),
            self.safe_day101a_report(),
            self.safe_day101b_report(),
        )
        validation = build_validation_report(
            model_audit=self.safe_model_audit(),
            final_refresh_audit={"passed": True, "mismatch_count": 0},
            version_audit=version_audit,
            source_metadata={"source": {"sha256": ""}},
            as_of_time="2026-08-20T12:00:00Z",
            fpl_deadline_time="2026-08-21T17:30:00Z",
            day76d_report=self.safe_day76d_report(),
            day97a_report=self.safe_day97a_report(),
            day101a_report=self.safe_day101a_report(),
            day101b_report=self.safe_day101b_report(),
            final_freeze=False,
        )
        self.assertFalse(validation["checks"]["input_fingerprints_recorded"]["passed"])
        self.assertFalse(validation["passed"])

    def test_artifact_definitions_include_every_plan_required_primary_output(self):
        definitions = artifact_definitions()
        self.assertEqual(definitions["gw1_opening_squad_csv"], ("gw1_opening_squad", "csv"))
        self.assertEqual(definitions["gw1_opening_squad_json"], ("gw1_opening_squad", "json"))
        self.assertEqual(definitions["gw1_opening_squad_md"], ("gw1_opening_squad", "md"))
        self.assertEqual(
            definitions["gw1_opening_squad_validation_json"],
            ("gw1_opening_squad_validation", "json"),
        )
        self.assertEqual(
            definitions["gw1_opening_squad_validation_md"],
            ("gw1_opening_squad_validation", "md"),
        )
        self.assertIn("global_player_prediction_snapshot_csv", definitions)

    def test_evaluation_contract_contains_all_six_agreed_metric_groups(self):
        contract = evaluation_contract()
        groups = contract["player_prediction_metrics"]
        self.assertEqual(
            list(groups),
            [
                "1_point_accuracy",
                "2_top_k_ranking_hits",
                "3_top_k_points_capture",
                "4_ranking_quality",
                "5_position_level",
                "6_availability_and_minutes",
            ],
        )
        self.assertEqual(groups["2_top_k_ranking_hits"]["k_values"], [10, 20, 50])
        self.assertEqual(groups["2_top_k_ranking_hits"]["headline_metric"], "top_20_hits")
        self.assertTrue(contract["model_team_metrics_separate"])
        self.assertTrue(contract["do_not_collapse_into_one_score"])

    def test_required_markdown_contains_plan_sections_and_historical_disclosure(self):
        player = {
            "player_id": 1,
            "web_name": "Player",
            "player_name": "Player",
            "team_short_name": "TM",
            "position": "GKP",
            "now_cost": 50,
            "role": "starter",
            "is_starter": True,
            "is_captain": True,
            "is_vice_captain": False,
            "gw1_predicted_points": 5.0,
            "gw2_predicted_points": None,
            "gw3_predicted_points": None,
            "gw4_predicted_points": None,
            "gw5_predicted_points": None,
            "expected_minutes": 90.0,
            "start_probability": 1.0,
            "appearance_probability": 1.0,
            "status": "a",
            "chance_of_playing_next_round": None,
            "prediction_confidence": "medium",
            "fallback_policy_used": "fallback",
            "fallback_level": 1,
            "risk_flags": [],
            "manual_review_required": False,
        }
        players = [dict(player, player_id=i, web_name="P%s" % i, player_name="P%s" % i) for i in range(1, 16)]
        for i, item in enumerate(players):
            item["position"] = "GKP" if i < 2 else "DEF" if i < 7 else "MID" if i < 12 else "FWD"
            item["is_captain"] = i == 0
            item["is_vice_captain"] = i == 1
        alt_players = [
            {"player_name": item["player_name"], "web_name": item["web_name"], "position": item["position"]}
            for item in players
        ]
        payload = {
            "target_season": TARGET_SEASON,
            "snapshot_kind": "pre_deadline_candidate",
            "as_of_time_utc": "2026-08-20T12:00:00Z",
            "fpl_deadline_time_utc": "2026-08-21T17:30:00Z",
            "primary": {
                "players": players,
                "total_cost_units": 1000,
                "bank_units": 0,
                "formation": "3-5-2",
                "captain_name": "P1",
                "vice_captain_name": "P2",
                "starting_player_ids": list(range(1, 12)),
                "bench_names": ["P12", "P13", "P14", "P15"],
            },
            "horizon": {
                "requested_horizon": 5,
                "effective_horizon": 1,
                "objective_mode": "gw1_only_fallback",
                "future_fixture_context": "manual_review_only",
            },
            "key_exclusions": {
                "candidate_count": 514,
                "excluded_count": 76,
                "exclusion_reason_counts": {"selection_ineligible": 76},
            },
            "alternative_a": {
                "total_cost_units": 1000,
                "bank_units": 0,
                "objective_value": 49.0,
                "players": alt_players,
                "alternative_design": {},
                "lineup_scope_note": "provisional",
            },
            "alternative_b": {
                "total_cost_units": 1000,
                "bank_units": 0,
                "objective_value": 48.0,
                "players": alt_players,
                "alternative_design": {},
                "lineup_scope_note": "provisional",
            },
        }
        validation = {
            "passed": True,
            "ready_for_manual_fpl_entry_review": True,
            "ready_for_final_freeze": True,
            "stop_point_satisfied": True,
        }
        markdown = build_opening_squad_markdown(payload, validation)
        self.assertIn("Primary Model Team — 15-player squad", markdown)
        self.assertIn("Starting XI", markdown)
        self.assertIn("Ordered bench", markdown)
        self.assertIn("GW1 and horizon projection status", markdown)
        self.assertIn("Key exclusions", markdown)
        self.assertIn("Two alternative squads", markdown)
        self.assertIn("Final manual-review checklist", markdown)
        self.assertIn("Historical approval is incomplete", markdown)
        self.assertIn("without requiring an API, frontend, or database write", markdown)

    def test_validation_markdown_renders_all_required_checks(self):
        checks = {}
        for key in (
            "player_count_15",
            "position_quotas_pass",
            "club_limits_pass",
            "budget_and_bank_reconcile",
            "starting_xi_and_formation_pass",
            "bench_order_pass",
            "captain_and_vice_pass",
            "no_duplicate_players",
            "player_identity_price_availability_match_final_refresh",
            "objective_total_reconciles_to_player_components",
            "required_versions_recorded",
            "input_fingerprints_recorded",
            "as_of_time_before_fpl_deadline",
            "status_preview_only",
            "production_and_manager_state_writes_disabled",
        ):
            checks[key] = {"passed": True, "detail": "ok"}
        validation = {
            "snapshot_kind": "pre_deadline_candidate",
            "as_of_time_utc": "2026-08-20T12:00:00Z",
            "fpl_deadline_time_utc": "2026-08-21T17:30:00Z",
            "checks": checks,
            "warnings": [],
            "blockers": [],
            "historical_multi_season_backtest_complete": False,
            "historical_approval_incomplete": True,
            "stop_point_satisfied": True,
        }
        markdown = build_validation_markdown(validation)
        self.assertEqual(markdown.count("- [x]"), 15)
        self.assertIn("Historical approval incomplete: `True`", markdown)


if __name__ == "__main__":
    unittest.main()
