from __future__ import annotations

import json
import unittest

from ml.contracts.run_metadata import (
    CONTRACT_VERSION,
    RunMetadata,
    RunMetadataError,
    build_run_id,
    build_run_metadata,
    validate_run_metadata,
)


class RunMetadataContractTests(unittest.TestCase):
    def base_kwargs(self, run_type: str) -> dict:
        kwargs = {
            "run_id": None,
            "run_type": run_type,
            "artifact_type": "test_%s_artifact" % run_type,
            "source_seasons": ["2023_24", "2024_25"],
            "target_season": "2025_26",
            "target_gw": 1,
            "horizon": 1,
            "as_of_time": "2025-08-10T12:00:00Z",
            "prediction_mode": "pre_gw1_prior",
            "created_at": "2025-08-10T12:05:00Z",
            "artifact_version": "artifact_v1",
            "provenance": {
                "producer": "tests.run_metadata",
                "inputs": [],
                "parent_run_ids": [],
                "notes": [],
            },
        }

        if run_type == "feature":
            kwargs["feature_version"] = "feature_v1"
        elif run_type == "model":
            kwargs["target_gw"] = None
            kwargs["horizon"] = None
            kwargs["prediction_mode"] = "historical_training"
            kwargs["feature_version"] = "feature_v1"
            kwargs["model_version"] = "model_v1"
        elif run_type == "prediction":
            kwargs["feature_version"] = "feature_v1"
            kwargs["model_version"] = "model_v1"
            kwargs["rules_versions"] = {"scoring": "scoring_v1"}
            kwargs["manifest_version"] = "manifest_v1"
        elif run_type == "evaluation":
            kwargs["model_version"] = "model_v1"
            kwargs["manifest_version"] = "manifest_v1"
        elif run_type == "optimization":
            kwargs["model_version"] = "model_v1"
            kwargs["rules_versions"] = {
                "squad_transfer": "squad_transfer_v1"
            }
            kwargs["manifest_version"] = "manifest_v1"
        elif run_type == "publishing":
            kwargs["manifest_version"] = "manifest_v1"

        return kwargs

    def test_all_six_run_types_validate(self) -> None:
        for run_type in (
            "feature",
            "model",
            "prediction",
            "evaluation",
            "optimization",
            "publishing",
        ):
            with self.subTest(run_type=run_type):
                metadata = build_run_metadata(**self.base_kwargs(run_type))
                self.assertEqual(validate_run_metadata(metadata), [])
                self.assertEqual(metadata.contract_version, CONTRACT_VERSION)

    def test_json_round_trip_is_stable(self) -> None:
        metadata = build_run_metadata(**self.base_kwargs("prediction"))
        encoded = metadata.to_json()
        decoded = RunMetadata.from_json(encoded)
        self.assertEqual(decoded.to_dict(), metadata.to_dict())
        self.assertEqual(json.loads(decoded.to_json()), json.loads(encoded))

    def test_source_seasons_must_be_oldest_to_newest(self) -> None:
        kwargs = self.base_kwargs("feature")
        kwargs["source_seasons"] = ["2024_25", "2023_24"]
        with self.assertRaisesRegex(
            RunMetadataError,
            "ordered oldest to newest",
        ):
            build_run_metadata(**kwargs)

    def test_source_seasons_must_not_be_future(self) -> None:
        kwargs = self.base_kwargs("feature")
        kwargs["source_seasons"] = ["2026_27"]
        with self.assertRaisesRegex(
            RunMetadataError,
            "must not be later than target_season",
        ):
            build_run_metadata(**kwargs)

    def test_as_of_time_must_not_be_after_creation(self) -> None:
        kwargs = self.base_kwargs("prediction")
        kwargs["as_of_time"] = "2025-08-10T12:06:00Z"
        with self.assertRaisesRegex(
            RunMetadataError,
            "as_of_time_utc must not be later",
        ):
            build_run_metadata(**kwargs)

    def test_horizon_must_not_pass_gameweek_38(self) -> None:
        kwargs = self.base_kwargs("optimization")
        kwargs["target_gw"] = 38
        kwargs["horizon"] = 2
        with self.assertRaisesRegex(
            RunMetadataError,
            "must not exceed Gameweek 38",
        ):
            build_run_metadata(**kwargs)

    def test_prediction_requires_rules_versions(self) -> None:
        kwargs = self.base_kwargs("prediction")
        kwargs["rules_versions"] = {}
        with self.assertRaisesRegex(
            RunMetadataError,
            "rules_versions must be non-empty",
        ):
            build_run_metadata(**kwargs)

    def test_model_run_can_be_season_level(self) -> None:
        metadata = build_run_metadata(**self.base_kwargs("model"))
        self.assertIsNone(metadata.target_gw)
        self.assertIsNone(metadata.horizon)

    def test_targeted_runs_require_horizon(self) -> None:
        kwargs = self.base_kwargs("feature")
        kwargs["horizon"] = None
        with self.assertRaisesRegex(
            RunMetadataError,
            "horizon is required",
        ):
            build_run_metadata(**kwargs)

    def test_run_id_helper_is_deterministic(self) -> None:
        first = build_run_id(
            run_type="prediction",
            target_season="2025_26",
            target_gw=1,
            created_at="2025-08-10T12:05:00Z",
            artifact_type="player_prediction_manifest",
        )
        second = build_run_id(
            run_type="prediction",
            target_season="2025_26",
            target_gw=1,
            created_at="2025-08-10T12:05:00Z",
            artifact_type="player_prediction_manifest",
        )
        self.assertEqual(first, second)
        self.assertIn("2025_26", first)
        self.assertIn("gw1", first)

    def test_provenance_parent_cannot_reference_self(self) -> None:
        kwargs = self.base_kwargs("publishing")
        kwargs["run_id"] = "publishing_run_001"
        kwargs["provenance"] = {
            "producer": "tests.run_metadata",
            "inputs": [],
            "parent_run_ids": ["publishing_run_001"],
            "notes": [],
        }
        with self.assertRaisesRegex(
            RunMetadataError,
            "must not list itself",
        ):
            build_run_metadata(**kwargs)

    def test_timezone_is_normalized_to_utc(self) -> None:
        kwargs = self.base_kwargs("feature")
        kwargs["as_of_time"] = "2025-08-10T05:00:00-07:00"
        metadata = build_run_metadata(**kwargs)
        self.assertEqual(metadata.as_of_time_utc, "2025-08-10T12:00:00Z")


if __name__ == "__main__":
    unittest.main()
