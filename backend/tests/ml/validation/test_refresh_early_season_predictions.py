from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from ml.validation.refresh_early_season_predictions import (
    build_current_player_stats,
    build_current_team_stats,
    blended_team_record,
    fixture_multiplier,
    scoreline_alignment_diagnostics,
    validate_actual_gw_coverage,
)


class EarlySeasonPredictionPipelineTests(unittest.TestCase):
    def test_gw2_current_player_component_uses_single_gw_without_lag(self) -> None:
        actuals = pd.DataFrame(
            [
                {
                    "fpl_player_id": 10,
                    "gw": 1,
                    "minutes": 90,
                    "total_points": 8,
                    "goals_scored": 1,
                    "assists": 0,
                    "clean_sheets": 1,
                },
                {
                    "fpl_player_id": 11,
                    "gw": 1,
                    "minutes": 0,
                    "total_points": 0,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                },
            ]
        )
        result = build_current_player_stats(actuals, target_gw=2)
        self.assertAlmostEqual(result[10]["current_points_per90"], 8.0)
        self.assertAlmostEqual(result[10]["current_expected_minutes"], 90.0)
        self.assertAlmostEqual(result[10]["current_appearance_probability"], 1.0)
        self.assertAlmostEqual(result[11]["current_points_per90"], 0.0)
        self.assertAlmostEqual(result[11]["current_expected_minutes"], 0.0)

    def test_gw3_requires_gw1_and_gw2_actual_coverage(self) -> None:
        incomplete = pd.DataFrame([{"fpl_player_id": 10, "gw": 1}])
        with self.assertRaises(RuntimeError):
            validate_actual_gw_coverage(incomplete, target_gw=3)

        complete = pd.DataFrame(
            [
                {"fpl_player_id": 10, "gw": 1},
                {"fpl_player_id": 10, "gw": 2},
            ]
        )
        report = validate_actual_gw_coverage(complete, target_gw=3)
        self.assertEqual(report["expected_actual_gws"], [1, 2])

    def test_team_blend_uses_requested_weights(self) -> None:
        prior = {
            1: {
                "points_per_match": 2.0,
                "goals_for_per_match": 2.0,
                "goals_against_per_match": 1.0,
                "clean_sheet_rate": 0.5,
                "goal_difference": 38.0,
                "prior_fallback_applied": False,
            }
        }
        current = {
            1: {
                "current_matches": 1,
                "current_points_per_match": 0.0,
                "current_goals_for_per_match": 0.0,
                "current_goals_against_per_match": 2.0,
                "current_clean_sheet_rate": 0.0,
                "current_goal_difference_per_match": -2.0,
            }
        }
        out = blended_team_record(1, prior, current, prior_weight=0.8, current_weight=0.2)
        self.assertAlmostEqual(out["points_per_match"], 1.6)
        self.assertAlmostEqual(out["goals_for_per_match"], 1.6)
        self.assertAlmostEqual(out["goals_against_per_match"], 1.2)
        self.assertAlmostEqual(out["clean_sheet_rate"], 0.4)

    def test_fixture_multiplier_is_bounded(self) -> None:
        args = SimpleNamespace(
            fixture_signal_min=0.75,
            fixture_signal_max=1.25,
            fixture_multiplier_min=0.85,
            fixture_multiplier_max=1.15,
        )
        out = fixture_multiplier(
            position="MID",
            expected_team_goals=4.0,
            expected_opponent_goals=0.1,
            league_mean_expected_goals=1.4,
            args=args,
        )
        self.assertGreaterEqual(out["fixture_multiplier"], 0.85)
        self.assertLessEqual(out["fixture_multiplier"], 1.15)

    def test_current_team_stats_accept_scored_provisional_fixture(self) -> None:
        fixtures = pd.DataFrame(
            [
                {
                    "home_fpl_team_id": 1,
                    "away_fpl_team_id": 2,
                    "home_score": 2,
                    "away_score": 1,
                    "finished": False,
                }
            ]
        )
        out = build_current_team_stats(fixtures)
        self.assertEqual(out[1]["current_matches"], 1)
        self.assertAlmostEqual(out[1]["current_points_per_match"], 3.0)
        self.assertAlmostEqual(out[2]["current_points_per_match"], 0.0)


    def test_scoreline_alignment_diagnostics_reports_gap_and_label_match(self) -> None:
        matches = pd.DataFrame(
            [
                {
                    "fpl_fixture_id": 11,
                    "home_win_probability": 0.60,
                    "draw_probability": 0.25,
                    "away_win_probability": 0.15,
                    "predicted_result_label": "home_win",
                }
            ]
        )
        scorelines = pd.DataFrame(
            [
                {
                    "fpl_fixture_id": 11,
                    "scoreline_home_win_probability": 0.50,
                    "scoreline_draw_probability": 0.30,
                    "scoreline_away_win_probability": 0.20,
                    "scoreline_result_label": "home_win",
                }
            ]
        )
        out = scoreline_alignment_diagnostics(matches, scorelines)
        self.assertEqual(out["rows_compared"], 1)
        self.assertEqual(out["label_mismatch_rows"], 0)
        self.assertAlmostEqual(out["max_abs_probability_gap"], 0.10)


if __name__ == "__main__":
    unittest.main()
