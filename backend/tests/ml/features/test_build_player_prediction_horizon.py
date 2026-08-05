from __future__ import annotations

import unittest

import pandas as pd

from ml.features.build_player_prediction_horizon import (
    DEFAULT_HORIZON,
    DEFAULT_START_GW,
    DEFAULT_TARGET_SEASON,
    OBJECTIVE_GW1_ONLY,
    PlayerPredictionHorizonError,
    artifact_definitions,
    build_long_horizon,
    build_optimizer_rows,
    build_player_summary,
    build_validation,
    determine_mode,
    discount_schedule,
    normalize_risk_flags,
    normalize_scope,
    validate_source_reports,
)


class PlayerPredictionHorizonTests(unittest.TestCase):
    def predictions(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "target_season": "2026_27",
                    "target_gw": 1,
                    "run_id": "day76d",
                    "player_id": 1,
                    "fpl_player_id": 101,
                    "player_name": "Player One",
                    "web_name": "One",
                    "team_id": 1,
                    "position": "MID",
                    "now_cost": 75,
                    "has_fixture": True,
                    "selection_eligible": True,
                    "appearance_probability": 0.9,
                    "start_probability": 0.8,
                    "expected_minutes": 75.0,
                    "predicted_points": 5.0,
                    "uncertainty_lower": None,
                    "uncertainty_upper": None,
                    "confidence_score": 0.5,
                    "confidence_label": "medium",
                    "fallback_used": True,
                    "fallback_level": 3,
                    "risk_flags": "['rotation_flag']",
                    "readiness_status": "preview_only",
                    "production_ready": False,
                    "prediction_write_allowed": False,
                    "prediction_source": "pre_gw1_heuristic_preview",
                },
                {
                    "target_season": "2026_27",
                    "target_gw": 1,
                    "run_id": "day76d",
                    "player_id": 2,
                    "fpl_player_id": 102,
                    "player_name": "Player Two",
                    "web_name": "Two",
                    "team_id": 2,
                    "position": "DEF",
                    "now_cost": 50,
                    "has_fixture": True,
                    "selection_eligible": False,
                    "appearance_probability": 0.7,
                    "start_probability": 0.5,
                    "expected_minutes": 55.0,
                    "predicted_points": 3.0,
                    "uncertainty_lower": None,
                    "uncertainty_upper": None,
                    "confidence_score": 0.3,
                    "confidence_label": "low",
                    "fallback_used": True,
                    "fallback_level": 4,
                    "risk_flags": "",
                    "readiness_status": "preview_only",
                    "production_ready": False,
                    "prediction_write_allowed": False,
                    "prediction_source": "pre_gw1_heuristic_preview",
                },
            ]
        )

    def fixture_context(self) -> pd.DataFrame:
        rows = []
        for gw in range(1, 6):
            for player_id, team_id, eligible, position, price in (
                (1, 1, True, "MID", 75),
                (2, 2, False, "DEF", 50),
            ):
                rows.append(
                    {
                        "run_id": "day79b",
                        "target_season": "2026_27",
                        "gameweek": gw,
                        "player_id": player_id,
                        "player_code": 100 + player_id,
                        "player_name": "Player %s" % player_id,
                        "web_name": "P%s" % player_id,
                        "team_id": team_id,
                        "team_name": "Team %s" % team_id,
                        "team_short_name": "T%s" % team_id,
                        "position": position,
                        "price_units": price,
                        "price": price / 10.0,
                        "selection_eligible": eligible,
                        "fixture_id": gw * 10 + team_id,
                        "fixture_count_for_team_gw": 1,
                        "has_fixture": True,
                        "blank_gw_flag": False,
                        "double_gw_flag": False,
                        "opponent_team_id": 3 - team_id,
                        "opponent_team_name": "Opponent",
                        "opponent_team_short_name": "OPP",
                        "is_home": gw % 2 == 1,
                        "kickoff_time_utc": "2026-08-21T19:00:00Z",
                        "kickoff_time_status": "known",
                        "kickoff_change_status": "baseline_no_previous_comparison",
                        "player_fixture_eligible": eligible,
                        "eligibility_reason": (
                            "eligible_current_player_and_fixture"
                            if eligible
                            else "current_player_ineligible"
                        ),
                        "manual_review_required": False,
                    }
                )
        return pd.DataFrame(rows)

    def build(self):
        predictions = self.predictions()
        fixture = self.fixture_context()
        discounts = discount_schedule(1, 5)
        long_frame, diagnostics = build_long_horizon(
            predictions, fixture, 1, 5, discounts
        )
        mode = determine_mode(long_frame, 1, 5)
        optimizer = build_optimizer_rows(long_frame, mode)
        summary = build_player_summary(long_frame, mode)
        validation = build_validation(
            long_frame, optimizer, summary, predictions, 1, 5, mode
        )
        return long_frame, mode, optimizer, summary, validation, diagnostics

    def test_defaults_match_fast_lane_scope(self):
        self.assertEqual(DEFAULT_TARGET_SEASON, "2026_27")
        self.assertEqual(DEFAULT_START_GW, 1)
        self.assertEqual(DEFAULT_HORIZON, 5)

    def test_scope_refuses_horizon_beyond_gw38(self):
        with self.assertRaises(PlayerPredictionHorizonError):
            normalize_scope("2026_27", 37, 3)

    def test_discount_schedule_matches_gw1_gw5_policy(self):
        self.assertEqual(
            discount_schedule(1, 5),
            {1: 1.0, 2: 0.85, 3: 0.7, 4: 0.55, 5: 0.4},
        )

    def test_long_format_preserves_every_player_gameweek(self):
        long_frame, _, _, _, _, _ = self.build()
        self.assertEqual(len(long_frame), 10)
        self.assertEqual(
            len(long_frame[["player_id", "target_gw"]].drop_duplicates()),
            10,
        )

    def test_gw1_prediction_is_preserved(self):
        long_frame, _, _, _, _, _ = self.build()
        row = long_frame[
            (long_frame["player_id"] == 1)
            & (long_frame["target_gw"] == 1)
        ].iloc[0]
        self.assertEqual(row["predicted_points"], 5.0)
        self.assertEqual(row["discounted_predicted_points"], 5.0)
        self.assertTrue(row["prediction_available"])

    def test_future_predictions_are_missing_not_zero(self):
        long_frame, _, _, _, _, _ = self.build()
        future = long_frame[long_frame["target_gw"] > 1]
        self.assertTrue(future["predicted_points"].isna().all())
        self.assertTrue(future["discounted_predicted_points"].isna().all())
        self.assertTrue(
            future["prediction_status"].eq("missing_future_prediction").all()
        )

    def test_fallback_mode_is_explicit(self):
        _, mode, _, _, _, _ = self.build()
        self.assertEqual(mode["objective_mode"], OBJECTIVE_GW1_ONLY)
        self.assertEqual(mode["effective_horizon"], 1)
        self.assertEqual(mode["future_fixture_context"], "manual_review_only")
        self.assertEqual(mode["missing_prediction_gameweeks"], [2, 3, 4, 5])

    def test_optimizer_rows_include_only_safe_available_rows(self):
        _, _, optimizer, _, _, _ = self.build()
        self.assertEqual(len(optimizer), 1)
        self.assertEqual(int(optimizer.iloc[0]["player_id"]), 1)
        self.assertEqual(int(optimizer.iloc[0]["target_gw"]), 1)
        self.assertFalse(optimizer["predicted_points"].isna().any())

    def test_ineligible_player_remains_visible_but_not_optimizer_eligible(self):
        long_frame, _, optimizer, _, _, _ = self.build()
        player_two = long_frame[long_frame["player_id"] == 2]
        self.assertEqual(len(player_two), 5)
        self.assertFalse(player_two["selection_eligible"].any())
        self.assertNotIn(2, optimizer["player_id"].tolist())

    def test_minutes_start_fallback_risk_and_uncertainty_are_preserved(self):
        long_frame, _, _, _, _, _ = self.build()
        row = long_frame[
            (long_frame["player_id"] == 1)
            & (long_frame["target_gw"] == 1)
        ].iloc[0]
        self.assertEqual(row["expected_minutes"], 75.0)
        self.assertEqual(row["start_probability"], 0.8)
        self.assertTrue(row["fallback_used"])
        self.assertEqual(row["fallback_level"], 3)
        self.assertIn("rotation_flag", row["risk_flags"])
        self.assertTrue(pd.isna(row["uncertainty_lower"]))

    def test_summary_uses_available_points_only(self):
        _, _, _, summary, _, _ = self.build()
        row = summary[summary["player_id"] == 1].iloc[0]
        self.assertEqual(row["prediction_gameweek_count"], 1)
        self.assertEqual(row["missing_prediction_gameweek_count"], 4)
        self.assertEqual(row["available_undiscounted_points"], 5.0)
        self.assertFalse(row["full_horizon_prediction_complete"])

    def test_validation_passes_gw1_fallback(self):
        _, _, _, _, validation, _ = self.build()
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["blockers"], [])
        self.assertEqual(validation["prediction_rows_by_gw"]["1"], 2)
        self.assertEqual(validation["prediction_rows_by_gw"]["2"], 0)

    def test_missing_gw1_prediction_fails_closed(self):
        predictions = self.predictions().iloc[:1].copy()
        fixture = self.fixture_context()
        long_frame, _ = build_long_horizon(
            predictions, fixture, 1, 5, discount_schedule(1, 5)
        )
        with self.assertRaises(PlayerPredictionHorizonError):
            determine_mode(long_frame, 1, 5)

    def test_duplicate_player_gameweek_fixture_context_fails_before_builder(self):
        fixture = self.fixture_context()
        duplicate = pd.concat([fixture, fixture.iloc[[0]]], ignore_index=True)
        self.assertTrue(duplicate.duplicated(["player_id", "gameweek"]).any())

    def test_source_report_validation_accepts_read_only_previews(self):
        prediction_report = {
            "target_season": "2026_27",
            "target_gw": 1,
            "stop_point_satisfied": True,
            "writes_database": False,
            "production_approved": False,
        }
        fixture_report = {
            "target_season": "2026_27",
            "start_gw": 1,
            "horizon": 5,
            "ready_for_day97a": True,
            "stop_point_satisfied": True,
            "writes_database": False,
        }
        validate_source_reports(
            prediction_report, fixture_report, "2026_27", 1, 5
        )

    def test_source_report_refuses_production_approved_preview(self):
        prediction_report = {
            "target_season": "2026_27",
            "target_gw": 1,
            "stop_point_satisfied": True,
            "writes_database": False,
            "production_approved": True,
        }
        fixture_report = {
            "target_season": "2026_27",
            "start_gw": 1,
            "horizon": 5,
            "ready_for_day97a": True,
            "stop_point_satisfied": True,
            "writes_database": False,
        }
        with self.assertRaises(PlayerPredictionHorizonError):
            validate_source_reports(
                prediction_report, fixture_report, "2026_27", 1, 5
            )

    def test_artifact_set_is_explicit(self):
        self.assertEqual(
            set(artifact_definitions()),
            {
                "player_prediction_horizon_csv",
                "player_horizon_summary_csv",
                "optimizer_projection_rows_csv",
                "run_metadata_json",
                "player_prediction_horizon_report_json",
                "player_prediction_horizon_report_md",
            },
        )

    def test_risk_flags_normalize_csv_list_string(self):
        self.assertEqual(
            normalize_risk_flags("['a', 'b']"),
            ("a", "b"),
        )


if __name__ == "__main__":
    unittest.main()
