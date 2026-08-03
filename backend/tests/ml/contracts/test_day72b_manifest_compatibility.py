from __future__ import annotations

import unittest
from types import SimpleNamespace

from ml.validation.export_pre_gw1_player_prediction_manifest import (
    ARTIFACT_TYPE,
    MANIFEST_VERSION,
    build_standard_run_metadata,
)


class Day72BManifestCompatibilityTests(unittest.TestCase):
    def test_standard_metadata_preserves_existing_identity(self) -> None:
        args = SimpleNamespace(
            source_seasons=["2024_25"],
            target_season="2025_26",
            target_gw=1,
            as_of_time="2025-08-10T12:00:00Z",
            player_feature_version="day71a_v0",
            expected_model_version="day72a_v0_1",
            scoring_rules_version="target_season_rules_unresolved",
            role_contract_version="day71b_v1",
            threshold_policy_version="player_role_thresholds_v0",
            scoreline_model_version="day70c_v0",
        )
        mode_result = {"resolved_prediction_mode": "pre_gw1_prior"}
        fingerprints = {
            "prediction_preview_csv": {
                "path": "/tmp/preview.csv",
                "exists": True,
                "size_bytes": 123,
                "sha256": "a" * 64,
            }
        }

        metadata = build_standard_run_metadata(
            args=args,
            mode_result=mode_result,
            created_at="2025-08-10T12:05:00Z",
            manifest_run_id="day72b_existing_run_id",
            artifact_fingerprints=fingerprints,
        )

        self.assertEqual(metadata["run_id"], "day72b_existing_run_id")
        self.assertEqual(metadata["artifact_type"], ARTIFACT_TYPE)
        self.assertEqual(
            metadata["versions"]["manifest_version"],
            MANIFEST_VERSION,
        )
        self.assertEqual(metadata["target_gw"], 1)
        self.assertEqual(metadata["horizon"], 1)
        self.assertEqual(metadata["prediction_mode"], "pre_gw1_prior")
        self.assertEqual(
            metadata["provenance"]["inputs"][0]["name"],
            "prediction_preview_csv",
        )


if __name__ == "__main__":
    unittest.main()
