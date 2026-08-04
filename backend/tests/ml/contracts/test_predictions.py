from __future__ import annotations

import unittest

from ml.contracts.predictions import (
    OUTPUT_DECISION,
    OUTPUT_MATCH,
    OUTPUT_MINUTES,
    OUTPUT_PLAYER_EVENT,
    OUTPUT_PLAYER_POINTS,
    OUTPUT_RANKING,
    OUTPUT_SCORELINE,
    DecisionOutput,
    MatchPredictionOutput,
    MinutesPredictionOutput,
    PlayerEventPredictionOutput,
    PlayerPointsPredictionOutput,
    PredictionContext,
    PredictionContractError,
    RankingOutput,
    SafetyMetadata,
    ScorelinePredictionOutput,
    contract_field_requirements,
)


class PredictionContractTests(unittest.TestCase):
    def context(self, output_type: str) -> PredictionContext:
        return PredictionContext(
            output_type=output_type,
            source_seasons=("2025_26",),
            target_season="2026_27",
            target_gw=1,
            as_of_time="2026-08-03T20:00:00-07:00",
            prediction_mode="pre_gw1_prior",
            prediction_scope="read_only_preview",
            run_id="run_001",
            model_name="model_v1",
            model_version="v1",
            feature_version="feature_v1",
            rules_versions={"scoring": "scoring_v1"},
        )

    def safety(self) -> SafetyMetadata:
        return SafetyMetadata(
            data_quality_status="full_prior_features",
            calibration_status="not_calibrated_preview_only",
            guardrail_status="basic_sanity_only",
            confidence_score=0.7,
            fallback_used=False,
            production_ready=False,
            prediction_write_allowed=False,
        )

    def test_context_normalizes_as_of_to_utc(self) -> None:
        context = self.context(OUTPUT_MATCH)
        self.assertEqual(context.as_of_time, "2026-08-04T03:00:00Z")


    def test_context_from_day75b_mapping(self) -> None:
        context = PredictionContext.from_run_metadata(
            {
                "run_id": "prediction_2026_27_gw01_001",
                "run_type": "prediction",
                "artifact_type": "player_prediction_manifest",
                "source_seasons": ["2025_26"],
                "target_season": "2026_27",
                "target_gw": 1,
                "horizon": 1,
                "as_of_time_utc": "2026-08-04T03:00:00Z",
                "prediction_mode": "pre_gw1_prior",
                "versions": {
                    "feature_version": "feature_v1",
                    "model_version": "model_v1",
                    "rules_versions": {"scoring": "scoring_v1"},
                    "artifact_version": "manifest_v1",
                },
            },
            output_type=OUTPUT_PLAYER_POINTS,
            model_name="player_model",
            model_version="model_v1",
            prediction_scope="read_only_preview",
        )
        self.assertEqual(context.as_of_time, "2026-08-04T03:00:00Z")
        self.assertEqual(context.feature_version, "feature_v1")
        self.assertEqual(context.source_artifact_version, "manifest_v1")
        self.assertEqual(context.contract_version, "fpl_prediction_contract_v1")

    def test_context_rejects_invalid_season(self) -> None:
        with self.assertRaises(PredictionContractError):
            PredictionContext(
                output_type=OUTPUT_MATCH,
                source_seasons=("2025_26",),
                target_season="2026_28",
                target_gw=1,
                as_of_time="2026-08-04T03:00:00Z",
                prediction_mode="pre_gw1_prior",
                prediction_scope="preview",
                run_id="run_1",
                model_name="model",
                model_version="v1",
            )

    def test_write_allowed_requires_production_ready(self) -> None:
        with self.assertRaises(PredictionContractError):
            SafetyMetadata(
                data_quality_status="ok",
                calibration_status="calibrated",
                guardrail_status="validated",
                readiness_status="validated",
                production_ready=False,
                prediction_write_allowed=True,
            )

    def test_fallback_requires_level(self) -> None:
        with self.assertRaises(PredictionContractError):
            SafetyMetadata(
                data_quality_status="fallback",
                calibration_status="none",
                guardrail_status="basic",
                fallback_used=True,
            )

    def test_match_probabilities_must_sum_to_one(self) -> None:
        with self.assertRaises(PredictionContractError):
            MatchPredictionOutput(
                context=self.context(OUTPUT_MATCH),
                safety=self.safety(),
                fixture_id=1,
                home_team_id=1,
                away_team_id=2,
                home_win_probability=0.5,
                draw_probability=0.3,
                away_win_probability=0.3,
                predicted_result_label="H",
            )

    def test_valid_match_output(self) -> None:
        output = MatchPredictionOutput(
            context=self.context(OUTPUT_MATCH),
            safety=self.safety(),
            fixture_id=1,
            home_team_id=1,
            away_team_id=2,
            home_win_probability=0.5,
            draw_probability=0.3,
            away_win_probability=0.2,
            predicted_result_label="H",
        )
        self.assertEqual(output.fixture_id, 1)

    def test_scoreline_requires_complete_probability_triplet(self) -> None:
        with self.assertRaises(PredictionContractError):
            ScorelinePredictionOutput(
                context=self.context(OUTPUT_SCORELINE),
                safety=self.safety(),
                fixture_id=1,
                home_team_id=1,
                away_team_id=2,
                expected_home_goals=1.5,
                expected_away_goals=1.0,
                scoreline_home_win_probability=0.5,
            )

    def test_minutes_start_cannot_exceed_appearance(self) -> None:
        with self.assertRaises(PredictionContractError):
            MinutesPredictionOutput(
                context=self.context(OUTPUT_MINUTES),
                safety=self.safety(),
                player_id=10,
                appearance_probability=0.5,
                start_probability=0.6,
                conditional_minutes_if_appears=70,
                expected_minutes=35,
                minutes_lower_bound=0,
                minutes_upper_bound=70,
            )

    def test_event_output(self) -> None:
        output = PlayerEventPredictionOutput(
            context=self.context(OUTPUT_PLAYER_EVENT),
            safety=self.safety(),
            player_id=10,
            expected_goals=0.3,
            expected_assists=0.2,
            clean_sheet_probability=0.4,
            expected_bonus=0.5,
            expected_other_points=0.1,
        )
        self.assertEqual(output.expected_goals, 0.3)

    def test_player_points_optimizer_row_is_small_and_stable(self) -> None:
        output = PlayerPointsPredictionOutput(
            context=self.context(OUTPUT_PLAYER_POINTS),
            safety=self.safety(),
            player_id=10,
            fpl_player_id=1010,
            player_name="Example Player",
            team_id=1,
            position="MID",
            price=7.5,
            now_cost=75,
            predicted_points=5.2,
            expected_minutes=82.0,
            start_probability=0.8,
            appearance_probability=0.9,
            has_fixture=True,
            fixture_id=1,
            opponent_team_id=2,
            is_home=True,
        )
        row = output.to_optimizer_row()
        self.assertEqual(row["predicted_points"], 5.2)
        self.assertEqual(row["position"], "MID")
        self.assertNotIn("extensions", row)

    def test_player_points_rejects_inconsistent_price(self) -> None:
        with self.assertRaises(PredictionContractError):
            PlayerPointsPredictionOutput(
                context=self.context(OUTPUT_PLAYER_POINTS),
                safety=self.safety(),
                player_id=10,
                team_id=1,
                position="MID",
                price=7.5,
                now_cost=70,
                predicted_points=5.2,
                expected_minutes=82.0,
                start_probability=0.8,
                appearance_probability=0.9,
                has_fixture=False,
            )

    def test_ranking_contract(self) -> None:
        output = RankingOutput(
            context=self.context(OUTPUT_RANKING),
            safety=self.safety(),
            player_id=10,
            position="MID",
            metric_name="horizon_points",
            metric_value=23.4,
            rank=1,
            horizon_gws=(1, 2, 3, 4, 5),
        )
        self.assertEqual(output.rank, 1)

    def test_decision_contract(self) -> None:
        output = DecisionOutput(
            context=self.context(OUTPUT_DECISION),
            safety=self.safety(),
            decision_id="opening_squad_001",
            decision_type="opening_squad",
            objective_name="discounted_horizon_points",
            objective_value=88.2,
            selected_entity_ids=tuple(range(1, 16)),
        )
        self.assertEqual(len(output.selected_entity_ids), 15)

    def test_field_requirement_documentation(self) -> None:
        requirements = contract_field_requirements(OUTPUT_PLAYER_POINTS)
        self.assertIn("predicted_points", requirements["mandatory"])
        self.assertIn("extensions", requirements["optional"])


if __name__ == "__main__":
    unittest.main()
