from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from ml.validation.refresh_pre_gw1_player_predictions import (
    Day76DInputError,
    PREDICTION_SOURCE,
    REQUIRED_LIMITATION_LABELS,
    adapt_current_player_pool,
    adapt_player_identity_mapping,
    adapt_player_priors,
    adapt_standard_predictions,
    adapt_target_fixtures,
    adapt_target_teams,
    adapt_team_identity_mapping,
    build_artifact_first_player_feature_report,
    build_limitation_contract,
    normalize_contract_input_row,
    validate_rollover_report,
)


class FakePlayerContract:
    def __init__(self, player_id: int, eligible: bool) -> None:
        self.player_id = player_id
        self.eligible = eligible

    def to_optimizer_row(self) -> dict:
        return {
            "player_id": self.player_id,
            "selection_eligible": self.eligible,
            "prediction_write_allowed": False,
            "production_ready": False,
        }


class Day76DRefreshTests(unittest.TestCase):
    def rollover_report(self) -> dict:
        return {
            "source_season": "2025_26",
            "target_season": "2026_27",
            "target_gw": 1,
            "as_of_time_utc": "2026-08-04T07:53:42Z",
            "passed": True,
            "audit_only": True,
            "writes_database": False,
            "blockers": [],
            "readiness": {
                "current_player_pool_validated": True,
                "focused_player_identity_coverage_validated": True,
                "gw1_gw5_fixture_scope_validated": True,
                "target_season_rule_versions_validated": True,
                "target_team_transition_validated": True,
                "ready_for_prediction_write": False,
            },
        }

    def player_pool(self) -> pd.DataFrame:
        rows = []
        for team_id in range(1, 21):
            rows.append(
                {
                    "target_player_id": team_id,
                    "target_player_code": 1000 + team_id,
                    "target_player_name": "Player %s" % team_id,
                    "target_web_name": "P%s" % team_id,
                    "target_team_id": team_id,
                    "target_team_name": "Team %s" % team_id,
                    "target_team_short_name": "T%02d" % team_id,
                    "target_position": "GKP" if team_id == 1 else "MID",
                    "target_price": 5.0,
                    "target_status": "a",
                    "current_selection_eligible": team_id != 20,
                }
            )
        return pd.DataFrame(rows)

    def test_rollover_report_accepts_read_only_checkpoint(self) -> None:
        result = validate_rollover_report(
            self.rollover_report(),
            source_season="2025_26",
            target_season="2026_27",
            target_gw=1,
        )
        self.assertEqual(result["target_season"], "2026_27")
        self.assertTrue(result["as_of_time_utc"].endswith("Z"))

    def test_rollover_report_refuses_prediction_write_readiness(self) -> None:
        report = self.rollover_report()
        report["readiness"]["ready_for_prediction_write"] = True
        with self.assertRaisesRegex(Day76DInputError, "must remain false"):
            validate_rollover_report(report)

    def test_current_player_pool_adapter_preserves_players(self) -> None:
        result = adapt_current_player_pool(self.player_pool())
        self.assertEqual(len(result), 20)
        self.assertEqual(result.loc[0, "player_id"], 1)
        self.assertEqual(result.loc[0, "fpl_player_id"], 1)
        self.assertEqual(result.loc[0, "position"], "GKP")
        self.assertFalse(bool(result.loc[19, "current_selection_eligible"]))

    def test_target_team_adapter_requires_twenty_teams(self) -> None:
        result = adapt_target_teams(self.player_pool())
        self.assertEqual(len(result), 20)
        self.assertEqual(result["team_id"].tolist(), list(range(1, 21)))

    def test_fixture_adapter_selects_gw1(self) -> None:
        rows = []
        for fixture_id in range(1, 11):
            rows.append(
                {
                    "target_season": "2026_27",
                    "gameweek": 1,
                    "fixture_id": fixture_id,
                    "kickoff_time_utc": "2026-08-21T19:00:00Z",
                    "home_team_id": fixture_id,
                    "away_team_id": fixture_id + 10,
                    "started": False,
                    "finished": False,
                }
            )
        result = adapt_target_fixtures(pd.DataFrame(rows), 1)
        self.assertEqual(len(result), 10)
        self.assertEqual(result["gw"].unique().tolist(), [1])
        self.assertFalse(result["finished"].any())

    def test_player_mapping_adapter_accepts_only_prior_eligible(self) -> None:
        mapping = pd.DataFrame(
            [
                {
                    "source_season": "2025_26",
                    "target_season": "2026_27",
                    "source_player_id": 10,
                    "source_player_name": "Mapped",
                    "target_player_id": 1,
                    "target_player_name": "Mapped",
                    "mapping_status": "accepted_exact",
                    "mapping_method": "unique_exact",
                    "mapping_reason": "safe",
                    "historical_prior_eligible": True,
                },
                {
                    "source_season": "2025_26",
                    "target_season": "2026_27",
                    "source_player_id": None,
                    "source_player_name": None,
                    "target_player_id": 2,
                    "target_player_name": "New Player",
                    "mapping_status": "unresolved",
                    "mapping_method": "none",
                    "mapping_reason": "no safe source identity",
                    "historical_prior_eligible": False,
                },
            ]
        )
        result = adapt_player_identity_mapping(mapping)
        self.assertTrue(bool(result.loc[0, "is_auto_approved"]))
        self.assertFalse(bool(result.loc[1, "is_auto_approved"]))
        self.assertTrue(bool(result.loc[1, "needs_manual_review"]))
        self.assertEqual(result.loc[1, "raw_player_id"], "unmapped_target_2")

    def test_player_prior_adapter_adds_safe_identity_fields(self) -> None:
        prior = pd.DataFrame(
            [
                {
                    "source_season": "2025_26",
                    "target_season": "2026_27",
                    "player_id": 10,
                    "web_name": "Mapped",
                    "prev_season_minutes": 1000,
                    "prev_season_appearances": 20,
                    "prev_season_starts_proxy": 12,
                    "prev_season_total_points": 80,
                    "prev_season_points_per90": 7.2,
                    "prev_season_goals": 5,
                    "prev_season_assists": 4,
                    "prev_season_clean_sheets": 2,
                    "prev_season_bonus": None,
                }
            ]
        )
        result = adapt_player_priors(prior)
        self.assertEqual(result.loc[0, "raw_player_id"], 10)
        self.assertEqual(result.loc[0, "prior_identity_scope"], "canonical_player_id")

    def test_team_mapping_adapter_preserves_promoted_and_relegated_rows(self) -> None:
        mapping = pd.DataFrame(
            [
                {
                    "source_season": "2025_26",
                    "target_season": "2026_27",
                    "source_team_id": 3,
                    "source_team_name": "Arsenal",
                    "source_team_short_name": "ARS",
                    "target_team_id": 1,
                    "target_team_name": "Arsenal",
                    "target_team_short_name": "ARS",
                    "mapping_status": "accepted_exact_short_name",
                    "historical_prior_eligible": True,
                },
                {
                    "source_season": "2025_26",
                    "target_season": "2026_27",
                    "source_team_id": 5,
                    "source_team_name": "Relegated",
                    "source_team_short_name": "REL",
                    "target_team_id": None,
                    "target_team_name": None,
                    "target_team_short_name": None,
                    "mapping_status": "historical_only",
                    "historical_prior_eligible": False,
                },
                {
                    "source_season": "2025_26",
                    "target_season": "2026_27",
                    "source_team_id": None,
                    "source_team_name": None,
                    "source_team_short_name": None,
                    "target_team_id": 7,
                    "target_team_name": "Promoted",
                    "target_team_short_name": "PRO",
                    "mapping_status": "target_only",
                    "historical_prior_eligible": False,
                },
            ]
        )
        result = adapt_team_identity_mapping(mapping)
        self.assertEqual(result.loc[0, "match_status"], "auto_approved_team_candidate")
        self.assertEqual(result.loc[1, "match_status"], "historical_only_unmatched")
        self.assertEqual(result.loc[2, "match_status"], "target_only_unmatched")

    def test_limitation_contract_contains_required_labels(self) -> None:
        result = build_limitation_contract(
            "fpl_2026_27_scoring_v1",
            "fpl_2026_27_bps_v1",
            True,
        )
        for key, value in REQUIRED_LIMITATION_LABELS.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["prediction_source"], PREDICTION_SOURCE)
        components = {
            item["component"] for item in result["unresolved_point_components"]
        }
        self.assertIn("prev_season_bonus", components)
        self.assertIn("bonus_points_system", components)

    def test_artifact_first_report_uses_artifact_counts_not_database_counts(self) -> None:
        features = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "has_prev_season_player_prior": True,
                    "no_prior_flag": False,
                    "has_fixture": True,
                    "blank_gw_flag": False,
                    "promoted_team_player_flag": False,
                    "missing_team_context_flag": False,
                    "missing_fixture_context_flag": False,
                    "prediction_write_allowed": False,
                    "production_ready": False,
                    "requires_player_feature_manifest_before_prediction": True,
                    "player_mapping_status": "auto_approved_player_candidate",
                    "team_fallback_applied": False,
                    "opponent_fallback_applied": False,
                },
                {
                    "player_id": 2,
                    "has_prev_season_player_prior": False,
                    "no_prior_flag": True,
                    "has_fixture": True,
                    "blank_gw_flag": False,
                    "promoted_team_player_flag": True,
                    "missing_team_context_flag": False,
                    "missing_fixture_context_flag": False,
                    "prediction_write_allowed": False,
                    "production_ready": False,
                    "requires_player_feature_manifest_before_prediction": True,
                    "player_mapping_status": "no_safe_accepted_mapping",
                    "team_fallback_applied": True,
                    "opponent_fallback_applied": False,
                },
            ]
        )
        mapping = pd.DataFrame(
            [
                {"is_auto_approved": True},
                {"is_auto_approved": False},
            ]
        )
        build_summary = {
            "mapping_summary": {
                "mapping_rows": 2,
                "top_mapping_rows": 2,
                "accepted_mapping_rows": 1,
                "ambiguous_or_manual_review_rows": 1,
                "unmatched_rows": 1,
                "duplicate_accepted_candidate_player_id_count": 0,
                "duplicate_accepted_raw_player_id_count": 0,
            },
            "fixture_build_summary": {
                "fixture_rows": 10,
                "duplicate_team_fixture_context_rows": 0,
            },
        }
        args = SimpleNamespace(
            source_season="2025_26",
            target_season="2026_27",
            target_gw=1,
            prediction_mode="auto",
            player_prior_csv="priors.csv",
            player_mapping_csv="mapping.csv",
            match_features_csv="matches.csv",
            out_csv="features.csv",
            out_json="features.json",
            out_md="features.md",
        )
        report = build_artifact_first_player_feature_report(
            args=args,
            mode_result={
                "resolved_prediction_mode": "pre_gw1_prior",
                "errors": [],
                "warnings": [],
            },
            features=features,
            build_summary=build_summary,
            player_priors=pd.DataFrame([{"x": 1}]),
            player_mapping=mapping,
            match_features=pd.DataFrame([{"fixture_id": 1}]),
            target_player_count=2,
            target_team_count=20,
            target_fixture_count=10,
            feature_version="day71a_v0",
            feature_scope="pre_gw1_player_features",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["row_counts"]["target_player_rows"], 2)
        self.assertEqual(report["row_counts"]["players_without_prior"], 1)
        self.assertFalse(report["writes_database"])

    def test_contract_input_normalizes_empty_risk_flags_from_csv(self) -> None:
        normalized = normalize_contract_input_row(
            {
                "player_id": 1,
                "risk_flags": float("nan"),
                "optional_fixture_id": float("nan"),
                "prediction_confidence": "medium",
            }
        )
        self.assertEqual(normalized["risk_flags"], ())
        self.assertIsNone(normalized["optional_fixture_id"])
        self.assertEqual(normalized["prediction_confidence"], "medium")

    def test_standard_adapter_normalizes_nan_before_contract(self) -> None:
        preview = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "eligible": True,
                    "risk_flags": float("nan"),
                }
            ]
        )

        def adapter(row: dict) -> FakePlayerContract:
            self.assertEqual(row["risk_flags"], ())
            return FakePlayerContract(int(row["player_id"]), bool(row["eligible"]))

        result = adapt_standard_predictions(preview, adapter)
        self.assertEqual(len(result), 1)

    def test_standard_adapter_preserves_row_count_and_safety_labels(self) -> None:
        preview = pd.DataFrame(
            [
                {"player_id": 1, "eligible": True},
                {"player_id": 2, "eligible": False},
            ]
        )

        def adapter(row: dict) -> FakePlayerContract:
            return FakePlayerContract(int(row["player_id"]), bool(row["eligible"]))

        result = adapt_standard_predictions(
            preview,
            adapter,
            selection_eligibility={1: True, 2: False},
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["prediction_source"], PREDICTION_SOURCE)
        self.assertFalse(result[0]["production_approved"])
        self.assertFalse(result[0]["prediction_write_allowed"])
        self.assertFalse(result[1]["selection_eligible"])
        self.assertEqual(
            result[1]["eligibility_reason"],
            "day76c_current_selection_ineligible",
        )


if __name__ == "__main__":
    unittest.main()
