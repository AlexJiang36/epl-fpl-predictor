from __future__ import annotations

import unittest

from ml.contracts.predictions import OUTPUT_PLAYER_POINTS, PREDICTION_CONTRACT_VERSION
from ml.registry.models import (
    ModelRegistry,
    ModelRegistryEntry,
    ModelRegistryError,
    adapt_legacy_model_metadata,
)


class ModelRegistryTests(unittest.TestCase):
    def entry(self) -> ModelRegistryEntry:
        return ModelRegistryEntry(
            model_name="player_model",
            model_version="v1",
            task_type="player_points",
            output_contract_type=OUTPUT_PLAYER_POINTS,
            output_contract_version=PREDICTION_CONTRACT_VERSION,
            supported_prediction_modes=("pre_gw1_prior",),
            feature_versions={"pre_gw1_player_features": "day71a_v0"},
            status="active",
            calibration_status="calibrated",
            guardrail_status="validated",
            production_ready=True,
            is_active=True,
            is_production_default=True,
        )

    def test_production_default_requires_ready(self) -> None:
        with self.assertRaises(ModelRegistryError):
            ModelRegistryEntry(
                model_name="bad",
                model_version="v1",
                task_type="player_points",
                output_contract_type=OUTPUT_PLAYER_POINTS,
                output_contract_version=PREDICTION_CONTRACT_VERSION,
                supported_prediction_modes=("pre_gw1_prior",),
                feature_versions={"features": "v1"},
                status="active",
                production_ready=False,
                is_active=True,
                is_production_default=True,
            )

    def test_registry_compatibility_lookup(self) -> None:
        registry = ModelRegistry((self.entry(),))
        matches = registry.compatible_entries(
            task_type="player_points",
            prediction_mode="pre_gw1_prior",
            feature_versions={"pre_gw1_player_features": "day71a_v0"},
            require_production_ready=True,
        )
        self.assertEqual(len(matches), 1)

    def test_registry_rejects_wrong_feature_version(self) -> None:
        registry = ModelRegistry((self.entry(),))
        matches = registry.compatible_entries(
            task_type="player_points",
            prediction_mode="pre_gw1_prior",
            feature_versions={"pre_gw1_player_features": "wrong"},
        )
        self.assertEqual(matches, tuple())

    def test_production_default_lookup(self) -> None:
        registry = ModelRegistry((self.entry(),))
        default = registry.production_default("player_points", "pre_gw1_prior")
        self.assertIsNotNone(default)
        self.assertEqual(default.model_name, "player_model")

    def test_legacy_adapter_does_not_promote_old_default(self) -> None:
        entry = adapt_legacy_model_metadata(
            {
                "model_name": "gbr_player_v2_1",
                "task_type": "player_points",
                "feature_version": "v2_1",
                "status": "active",
                "is_active": True,
                "is_production_default": True,
                "metrics_summary": {"val_mae": 0.95},
            },
            model_version="legacy_metadata_v1",
            output_contract_type=OUTPUT_PLAYER_POINTS,
            output_contract_version=PREDICTION_CONTRACT_VERSION,
            supported_prediction_modes=("normal_weekly",),
        )
        self.assertFalse(entry.production_ready)
        self.assertFalse(entry.is_active)
        self.assertFalse(entry.is_production_default)
        self.assertTrue(entry.extensions["legacy_is_production_default"])

    def test_duplicate_registry_key_fails(self) -> None:
        registry = ModelRegistry((self.entry(),))
        with self.assertRaises(ModelRegistryError):
            registry.register(self.entry())


if __name__ == "__main__":
    unittest.main()
