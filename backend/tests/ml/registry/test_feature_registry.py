from __future__ import annotations

import unittest

from ml.registry.features import (
    FeatureRegistry,
    FeatureRegistryEntry,
    FeatureRegistryError,
    adapt_legacy_feature_snapshot_metadata,
)


class FeatureRegistryTests(unittest.TestCase):
    def entry(self) -> FeatureRegistryEntry:
        return FeatureRegistryEntry(
            feature_name="pre_gw1_player_features",
            feature_version="day71a_v0",
            task_types=("player_points",),
            prediction_modes=("pre_gw1_prior",),
            required_columns=("player_id", "team_id", "position"),
            optional_columns=("status",),
            status="validated",
            production_ready=True,
        )

    def test_validate_required_columns(self) -> None:
        entry = self.entry()
        missing = entry.validate_columns(("player_id", "position"))
        self.assertEqual(missing, ("team_id",))

    def test_required_optional_overlap_fails(self) -> None:
        with self.assertRaises(FeatureRegistryError):
            FeatureRegistryEntry(
                feature_name="bad",
                feature_version="v1",
                task_types=("player_points",),
                prediction_modes=("pre_gw1_prior",),
                required_columns=("player_id",),
                optional_columns=("player_id",),
            )

    def test_registry_duplicate_fails(self) -> None:
        registry = FeatureRegistry((self.entry(),))
        with self.assertRaises(FeatureRegistryError):
            registry.register(self.entry())

    def test_compatible_entries(self) -> None:
        registry = FeatureRegistry((self.entry(),))
        compatible = registry.compatible_entries("player_points", "pre_gw1_prior")
        self.assertEqual(len(compatible), 1)

    def test_legacy_adapter_remains_unverified(self) -> None:
        entry = adapt_legacy_feature_snapshot_metadata(
            {
                "snapshot_id": "snapshot_1",
                "snapshot_type": "match_features",
                "feature_version": "legacy_v1",
            },
            task_types=("match_result",),
            prediction_modes=("pre_gw1_prior",),
            required_columns=("fixture_id",),
        )
        self.assertFalse(entry.production_ready)
        self.assertEqual(entry.status, "draft")


if __name__ == "__main__":
    unittest.main()
