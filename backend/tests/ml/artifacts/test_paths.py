from __future__ import annotations

from datetime import datetime, timezone
import unittest

from ml.artifacts.paths import (
    ArtifactPathError,
    build_immutable_artifact_key,
    build_immutable_key_from_run_metadata,
    build_pointer_key,
    format_path_timestamp,
    is_immutable_key,
    is_pointer_key,
    legacy_decision_run_key,
    legacy_evaluation_key,
    legacy_feature_snapshot_key,
    legacy_model_metadata_key,
    legacy_offline_dataset_key,
    legacy_run_snapshot_key,
    normalize_season,
)


class ArtifactPathContractTests(unittest.TestCase):
    def test_immutable_key_contains_required_dimensions(self) -> None:
        key = build_immutable_artifact_key(
            artifact_type="player_prediction",
            season="2025_26",
            target_gw=1,
            as_of_time="2026-08-03T12:34:56Z",
            run_id="prediction_2025_26_gw1_abcd1234",
            version="day72b_v1",
            filename="manifest",
            extension="json",
        )

        self.assertEqual(
            key,
            "v1/immutable/player_prediction/season=2025_26/gw=01/"
            "as_of=20260803T123456Z/version=day72b_v1/"
            "run_id=prediction_2025_26_gw1_abcd1234/manifest.json",
        )
        self.assertTrue(is_immutable_key(key))
        self.assertFalse(is_pointer_key(key))

    def test_season_level_key_uses_explicit_scope(self) -> None:
        key = build_immutable_artifact_key(
            artifact_type="model",
            season="2025_26",
            target_gw=None,
            as_of_time="2026-08-03T12:00:00+00:00",
            run_id="model_run_001",
            version="v1",
        )
        self.assertIn("/gw=season/", key)

    def test_pointer_key_is_separate_from_immutable_key(self) -> None:
        key = build_pointer_key(
            pointer_kind="latest",
            artifact_type="player_prediction",
            season="2025_26",
            target_gw=1,
            pointer_name="production_candidate",
        )
        self.assertEqual(
            key,
            "v1/pointers/latest/player_prediction/season=2025_26/gw=01/"
            "production_candidate.json",
        )
        self.assertTrue(is_pointer_key(key))
        self.assertFalse(is_immutable_key(key))

    def test_active_pointer_is_supported(self) -> None:
        key = build_pointer_key(
            pointer_kind="active",
            artifact_type="model",
            season="2025_26",
            target_gw=None,
        )
        self.assertIn("/pointers/active/model/", key)

    def test_unsupported_pointer_kind_fails(self) -> None:
        with self.assertRaises(ArtifactPathError):
            build_pointer_key(
                pointer_kind="archive",
                artifact_type="model",
                season="2025_26",
                target_gw=None,
            )

    def test_timestamp_is_normalized_to_utc(self) -> None:
        value = datetime(
            2026,
            8,
            3,
            5,
            0,
            0,
            tzinfo=timezone.utc,
        )
        self.assertEqual(format_path_timestamp(value), "20260803T050000Z")
        self.assertEqual(
            format_path_timestamp("2026-08-02T22:00:00-07:00"),
            "20260803T050000Z",
        )

    def test_invalid_season_fails(self) -> None:
        with self.assertRaises(ArtifactPathError):
            normalize_season("2025_27")

    def test_unsafe_segment_fails(self) -> None:
        with self.assertRaises(ArtifactPathError):
            build_immutable_artifact_key(
                artifact_type="../prediction",
                season="2025_26",
                target_gw=1,
                as_of_time="2026-08-03T12:00:00Z",
                run_id="run_1",
                version="v1",
            )

    def test_build_key_from_day75b_metadata(self) -> None:
        metadata = {
            "artifact_type": "pre_gw1_player_prediction_manifest",
            "target_season": "2025_26",
            "target_gw": 1,
            "as_of_time_utc": "2026-08-03T12:00:00Z",
            "run_id": "day72b_v1_2025_26_gw1_abc",
            "versions": {
                "artifact_version": "day72b_v1",
            },
        }
        key = build_immutable_key_from_run_metadata(
            metadata,
            filename="manifest",
        )
        self.assertIn(
            "/run_id=day72b_v1_2025_26_gw1_abc/manifest.json",
            key,
        )

    def test_explicit_legacy_compatibility_keys(self) -> None:
        self.assertEqual(
            legacy_feature_snapshot_key("player_features_abc"),
            "feature_snapshots/player_features_abc.json",
        )
        self.assertEqual(
            legacy_decision_run_key("transfer_abc"),
            "decision_runs/transfer_abc.json",
        )
        self.assertEqual(
            legacy_run_snapshot_key("refresh_abc"),
            "run_snapshots/refresh_abc.json",
        )
        self.assertEqual(
            legacy_model_metadata_key("baseline_rollavg_v1"),
            "model_metadata/baseline_rollavg_v1.json",
        )
        self.assertEqual(
            legacy_offline_dataset_key("player_features.csv"),
            "offline_datasets/player_features.csv",
        )
        self.assertEqual(
            legacy_evaluation_key("match_eval.csv"),
            "evaluations/match_eval.csv",
        )


if __name__ == "__main__":
    unittest.main()
