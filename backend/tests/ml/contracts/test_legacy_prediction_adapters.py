from __future__ import annotations

import unittest

from ml.contracts.predictions import (
    OUTPUT_PLAYER_POINTS,
    PredictionContractError,
    adapt_day70a_match_preview,
    adapt_day70c_scoreline_preview,
    adapt_day72a_player_points_preview,
)


class LegacyPredictionAdapterTests(unittest.TestCase):
    def test_day70a_adapter_preserves_preview_safety(self) -> None:
        row = {
            "source_season": "2025_26",
            "target_season": "2026_27",
            "target_gw": 1,
            "prediction_mode": "pre_gw1_prior",
            "prediction_scope": "read_only_preview",
            "model_name": "pre_gw1_match_prior_heuristic_v0",
            "model_version": "day70a_v0",
            "fixture_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "home_win_probability": 0.5,
            "draw_probability": 0.3,
            "away_win_probability": 0.2,
            "predicted_result_label": "H",
            "confidence_score": 0.2,
            "any_team_fallback_applied": True,
            "data_quality_status": "fallback_effective_features",
            "prediction_write_allowed": False,
            "production_ready": False,
            "calibration_status": "not_calibrated_preview_only",
            "guardrail_status": "basic_probability_sanity_only",
        }
        output = adapt_day70a_match_preview(
            row,
            as_of_time="2026-08-04T03:00:00Z",
            run_id="day70a_compat_001",
        )
        self.assertFalse(output.safety.production_ready)
        self.assertTrue(output.safety.fallback_used)

    def test_day70a_adapter_refuses_unsafe_legacy_flag(self) -> None:
        row = {
            "source_season": "2025_26",
            "target_season": "2026_27",
            "target_gw": 1,
            "prediction_mode": "pre_gw1_prior",
            "prediction_scope": "read_only_preview",
            "model_name": "model",
            "model_version": "v1",
            "fixture_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "home_win_probability": 0.5,
            "draw_probability": 0.3,
            "away_win_probability": 0.2,
            "predicted_result_label": "H",
            "prediction_write_allowed": False,
            "production_ready": True,
        }
        with self.assertRaises(PredictionContractError):
            adapt_day70a_match_preview(
                row,
                as_of_time="2026-08-04T03:00:00Z",
                run_id="unsafe_001",
            )

    def test_day70c_adapter(self) -> None:
        row = {
            "source_season": "2025_26",
            "target_season": "2026_27",
            "target_gw": 1,
            "prediction_mode": "pre_gw1_prior",
            "prediction_scope": "read_only_scoreline_preview",
            "model_name": "pre_gw1_scoreline_prior_heuristic_v0",
            "model_version": "day70c_v0",
            "fixture_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "expected_home_goals": 1.6,
            "expected_away_goals": 0.9,
            "scoreline_home_win_probability": 0.55,
            "scoreline_draw_probability": 0.25,
            "scoreline_away_win_probability": 0.20,
            "top_1_scoreline": "1-0",
            "any_team_fallback_applied": False,
            "data_quality_status": "full_prior_features",
            "prediction_write_allowed": False,
            "production_ready": False,
            "calibration_status": "not_calibrated_scoreline_preview_only",
            "guardrail_status": "basic_score_grid_sanity_only",
        }
        output = adapt_day70c_scoreline_preview(
            row,
            as_of_time="2026-08-04T03:00:00Z",
            run_id="day70c_compat_001",
        )
        self.assertEqual(output.predicted_scoreline, "1-0")

    def test_day72a_adapter_produces_optimizer_contract(self) -> None:
        row = {
            "source_seasons": "2025_26",
            "target_season": "2026_27",
            "target_gw": 1,
            "as_of_time": "2026-08-04T03:00:00Z",
            "prediction_mode": "pre_gw1_prior",
            "prediction_scope": "read_only_pre_gw1_player_prediction_preview",
            "run_id": "day72a_202627_001",
            "model_name": "pre_gw1_player_prior_heuristic_v0",
            "model_version": "day72a_v0_1",
            "player_feature_version": "day71a_v0",
            "scoring_rules_version": "fpl_scoring_2026_27_v1",
            "player_id": 10,
            "fpl_player_id": 1010,
            "player_name": "Example Player",
            "web_name": "Example",
            "team_id": 1,
            "team_name": "Example FC",
            "team_short_name": "EXA",
            "position": "MID",
            "price": 7.5,
            "now_cost": 75,
            "fixture_id": 1,
            "fpl_fixture_id": 1001,
            "opponent_team_id": 2,
            "opponent_team_name": "Other FC",
            "opponent_short_name": "OTH",
            "is_home": True,
            "has_fixture": True,
            "appearance_probability": 0.9,
            "start_probability": 0.8,
            "expected_minutes": 78.0,
            "minutes_lower_bound": 40.0,
            "minutes_upper_bound": 90.0,
            "raw_expected_points": 4.9,
            "final_predicted_points": 5.1,
            "predicted_points": 5.1,
            "expected_appearance_points": 1.8,
            "expected_goals": 0.25,
            "expected_goal_points": 1.25,
            "expected_assists": 0.2,
            "expected_assist_points": 0.6,
            "clean_sheet_probability": 0.35,
            "expected_clean_sheet_points": 0.35,
            "expected_bonus": 0.5,
            "expected_other_points": 0.6,
            "fallback_policy_used": "none",
            "fallback_level": 0,
            "fallback_reason": None,
            "risk_flags": "role_uncertainty",
            "data_quality_status": "full_player_prior",
            "prediction_confidence": "medium",
            "prediction_write_allowed": False,
            "production_ready": False,
            "calibration_status": "not_calibrated_preview_only",
            "guardrail_status": "basic_bounds_fixture_multiplier_and_role_uncertainty_v0_1",
            "component_accounting_status": "reconciled",
            "role_contract_version": "role_v1",
        }
        output = adapt_day72a_player_points_preview(row)
        self.assertEqual(output.context.output_type, OUTPUT_PLAYER_POINTS)
        self.assertFalse(output.safety.fallback_used)
        self.assertEqual(output.to_optimizer_row()["predicted_points"], 5.1)
        self.assertEqual(output.extensions["role_contract_version"], "role_v1")

    def test_day72a_adapter_requires_real_as_of_time(self) -> None:
        row = {
            "source_seasons": "2025_26",
            "target_season": "2026_27",
            "target_gw": 1,
            "prediction_mode": "pre_gw1_prior",
            "prediction_scope": "preview",
            "run_id": "run_1",
            "model_name": "model",
            "model_version": "v1",
            "player_feature_version": "feature_v1",
            "scoring_rules_version": "scoring_v1",
        }
        with self.assertRaises(PredictionContractError):
            adapt_day72a_player_points_preview(row)


if __name__ == "__main__":
    unittest.main()
