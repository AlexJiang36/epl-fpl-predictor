from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ml.artifacts.paths import (
    ArtifactPathError,
    build_immutable_artifact_key,
    build_pointer_key,
    legacy_decision_run_key,
)
from ml.artifacts.storage import (
    ArtifactNotFoundError,
    ArtifactStorageError,
    ImmutableArtifactExistsError,
    LocalArtifactStorage,
)


class LocalArtifactStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = LocalArtifactStorage(self.temp_dir.name)
        self.immutable_key = build_immutable_artifact_key(
            artifact_type="decision_run",
            season="2025_26",
            target_gw=1,
            as_of_time="2026-08-03T12:00:00Z",
            run_id="decision_run_001",
            version="v1",
        )
        self.pointer_key = build_pointer_key(
            pointer_kind="latest",
            artifact_type="decision_run",
            season="2025_26",
            target_gw=1,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_immutable_json_round_trip(self) -> None:
        payload = {"run_id": "decision_run_001", "score": 10}
        stored = self.storage.write_immutable_json(self.immutable_key, payload)

        self.assertTrue(stored.immutable)
        self.assertEqual(self.storage.read_json(self.immutable_key), payload)
        self.assertTrue(self.storage.exists(self.immutable_key))

    def test_immutable_overwrite_is_forbidden(self) -> None:
        self.storage.write_immutable_text(self.immutable_key, "first")

        with self.assertRaises(ImmutableArtifactExistsError):
            self.storage.write_immutable_text(self.immutable_key, "second")

        self.assertEqual(self.storage.read_text(self.immutable_key), "first")

    def test_immutable_write_requires_immutable_key(self) -> None:
        with self.assertRaises(ArtifactStorageError):
            self.storage.write_immutable_text(self.pointer_key, "not allowed")

    def test_mutable_pointer_can_be_replaced(self) -> None:
        first = {"target_key": "first"}
        second = {"target_key": "second"}

        self.storage.write_mutable_json(self.pointer_key, first)
        self.storage.write_mutable_json(self.pointer_key, second)

        self.assertEqual(self.storage.read_json(self.pointer_key), second)

    def test_mutable_write_requires_pointer_key(self) -> None:
        with self.assertRaises(ArtifactStorageError):
            self.storage.write_mutable_text(self.immutable_key, "not allowed")

    def test_list_filters_by_prefix(self) -> None:
        other_key = build_immutable_artifact_key(
            artifact_type="feature",
            season="2025_26",
            target_gw=1,
            as_of_time="2026-08-03T12:00:00Z",
            run_id="feature_run_001",
            version="v1",
        )
        self.storage.write_immutable_text(self.immutable_key, "decision")
        self.storage.write_immutable_text(other_key, "feature")
        self.storage.write_mutable_text(self.pointer_key, "pointer")

        listed = self.storage.list_keys("v1/immutable/decision_run")
        self.assertEqual(listed, [self.immutable_key])

    def test_fingerprint_returns_sha256_and_size(self) -> None:
        content = b"artifact-data"
        self.storage.write_immutable_bytes(self.immutable_key, content)

        fingerprint = self.storage.fingerprint(self.immutable_key)

        self.assertEqual(fingerprint.algorithm, "sha256")
        self.assertEqual(
            fingerprint.hexdigest,
            hashlib.sha256(content).hexdigest(),
        )
        self.assertEqual(fingerprint.size_bytes, len(content))

    def test_missing_read_fails_explicitly(self) -> None:
        with self.assertRaises(ArtifactNotFoundError):
            self.storage.read_bytes(self.immutable_key)

    def test_path_traversal_is_rejected(self) -> None:
        with self.assertRaises(ArtifactPathError):
            self.storage.read_bytes("../secret.json")

    def test_explicit_legacy_helper_reads_existing_path(self) -> None:
        legacy_key = legacy_decision_run_key("transfer_legacy")
        legacy_path = Path(self.temp_dir.name) / legacy_key
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_payload = {"run_id": "transfer_legacy", "target_gw": 32}
        legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

        loaded = self.storage.read_first_existing_json([legacy_key])

        self.assertEqual(loaded, legacy_payload)

    def test_legacy_read_does_not_make_legacy_path_writable_as_immutable(self) -> None:
        legacy_key = legacy_decision_run_key("transfer_legacy")
        with self.assertRaises(ArtifactStorageError):
            self.storage.write_immutable_json(legacy_key, {"run_id": "x"})


if __name__ == "__main__":
    unittest.main()
