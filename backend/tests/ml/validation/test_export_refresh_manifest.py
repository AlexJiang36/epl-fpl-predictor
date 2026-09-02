from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.validation.export_refresh_manifest import (
    REFRESH_MANIFEST_VERSION,
    RefreshManifestWriter,
    build_initial_manifest,
    manifest_stage_summary,
    stable_fingerprint,
)


class RefreshManifestTests(unittest.TestCase):
    def test_stable_fingerprint_is_order_independent(self) -> None:
        self.assertEqual(
            stable_fingerprint({"a": 1, "b": 2}),
            stable_fingerprint({"b": 2, "a": 1}),
        )

    def test_writer_records_run_ids_warnings_blockers_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            planning = Path(temporary_directory) / "planning"
            writer = RefreshManifestWriter.create(
                planning_root=planning,
                runner_version="runner_v1",
                dag_version="dag_v1",
                season="2026_27",
                target_gw=3,
                requested_phase="auto",
                resume_requested=True,
                final_freeze_requested=False,
                discovered_state={"resolved_action": "PRE"},
                candidate_run_id="candidate_old",
                active_run_id="candidate_old",
            )
            writer.add_warning("warning one")
            writer.add_blocker("blocker one")
            writer.update(
                status="FAILED",
                resolved_action="PRE",
                candidate_run_id="candidate_old",
                active_run_id="candidate_old",
                stage_results=[
                    {
                        "stage": "prediction_mode",
                        "status": "PASS",
                        "duration_seconds": 0.1,
                        "skip_reason": None,
                    },
                    {
                        "stage": "player_model",
                        "status": "FAILED",
                        "duration_seconds": 0.2,
                        "skip_reason": None,
                    },
                ],
                failure={"type": "RuntimeError", "message": "boom"},
            )
            payload = json.loads(writer.path.read_text())
            self.assertEqual(
                payload["manifest_version"],
                REFRESH_MANIFEST_VERSION,
            )
            self.assertEqual(payload["candidate_run_id"], "candidate_old")
            self.assertEqual(payload["active_run_id"], "candidate_old")
            self.assertEqual(payload["warnings"], ["warning one"])
            self.assertEqual(payload["blockers"], ["blocker one"])
            self.assertEqual(payload["status"], "FAILED")
            self.assertEqual(payload["failure"]["message"], "boom")

    def test_stage_summary_counts_reuse_skip_failure_and_duration(self) -> None:
        summary = manifest_stage_summary(
            [
                {
                    "stage": "a",
                    "status": "REUSED",
                    "duration_seconds": 0.1,
                },
                {
                    "stage": "b",
                    "status": "SKIPPED",
                    "duration_seconds": 0.0,
                },
                {
                    "stage": "c",
                    "status": "FAILED",
                    "duration_seconds": 0.2,
                },
            ]
        )
        self.assertEqual(summary["reused_stage_count"], 1)
        self.assertEqual(summary["skipped_stage_count"], 1)
        self.assertEqual(summary["failed_stage"], "c")
        self.assertEqual(summary["latest_successful_stage"], "a")
        self.assertEqual(summary["total_stage_duration_seconds"], 0.3)


if __name__ == "__main__":
    unittest.main()
