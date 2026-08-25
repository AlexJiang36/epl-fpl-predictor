from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.validation.verify_early_season_publish import (
    canonical_match_result_label,
    compare_match_rows,
    compare_player_rows,
    sha256_file,
    validate_receipt_and_snapshot,
)


class EarlySeasonPostPublishVerifyTests(unittest.TestCase):
    def test_player_rows_exact_match(self):
        preview = pd.DataFrame(
            {"fpl_player_id": [1, 2], "predicted_points": [2.5, 4.0]}
        )
        db = pd.DataFrame(
            {"fpl_player_id": [1, 2], "predicted_points": [2.5, 4.0]}
        )
        result = compare_player_rows(preview, db)
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["value_mismatch_rows"], 0)
        self.assertEqual(result["max_abs_predicted_points_diff"], 0.0)

    def test_player_value_change_is_blocked(self):
        preview = pd.DataFrame(
            {"fpl_player_id": [1], "predicted_points": [2.5]}
        )
        db = pd.DataFrame(
            {"fpl_player_id": [1], "predicted_points": [2.6]}
        )
        result = compare_player_rows(preview, db)
        self.assertTrue(result["blockers"])
        self.assertEqual(result["value_mismatch_rows"], 1)

    def test_match_rows_exact_match_with_hda_mapping(self):
        preview = pd.DataFrame(
            {
                "fpl_fixture_id": [11],
                "home_win_probability": [0.6],
                "draw_probability": [0.25],
                "away_win_probability": [0.15],
                "predicted_result_label": ["home_win"],
            }
        )
        db = pd.DataFrame(
            {
                "fpl_fixture_id": [11],
                "pred_home_win": [0.6],
                "pred_draw": [0.25],
                "pred_away_win": [0.15],
                "pred_result": ["H"],
            }
        )
        result = compare_match_rows(preview, db)
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["probability_mismatch_rows"], 0)
        self.assertEqual(result["label_mismatch_rows"], 0)

    def test_match_label_mapping(self):
        self.assertEqual(canonical_match_result_label("home_win"), "H")
        self.assertEqual(canonical_match_result_label("draw"), "D")
        self.assertEqual(canonical_match_result_label("away_win"), "A")

    def test_snapshot_tamper_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            snap_dir = root / "snap"
            run_dir.mkdir()
            snap_dir.mkdir()

            artifacts = {
                "run_manifest.json": json.dumps(
                    {
                        "run_id": "run1",
                        "season": "2026_27",
                        "target_gw": 2,
                        "status": "PASS_PREVIEW",
                    }
                ),
                "player_predictions_preview.csv": "fpl_player_id,predicted_points\n1,2.5\n",
                "match_predictions_preview.csv": "fpl_fixture_id,home_win_probability,draw_probability,away_win_probability,predicted_result_label\n11,0.6,0.25,0.15,home_win\n",
                "scoreline_preview.csv": "fpl_fixture_id\n11\n",
                "bootstrap_snapshot.json": "{}",
                "summary.md": "summary\n",
            }
            hashes = {}
            for name, content in artifacts.items():
                (run_dir / name).write_text(content, encoding="utf-8")
                (snap_dir / name).write_text(content, encoding="utf-8")
                hashes[name] = sha256_file(run_dir / name)

            snapshot_manifest = {
                "snapshot_contract": "immutable_pre_deadline_model_publish_snapshot_v1",
                "source_run_id": "run1",
                "season": "2026_27",
                "target_gw": 2,
                "overwrite_allowed": False,
                "final_deadline_freeze": False,
                "artifact_sha256": hashes,
                "prior_hash_checks": {},
            }
            snapshot_manifest_path = snap_dir / "snapshot_manifest.json"
            snapshot_manifest_path.write_text(
                json.dumps(snapshot_manifest), encoding="utf-8"
            )

            receipt = {
                "receipt_contract": "early_season_prediction_publish_receipt_v1",
                "season": "2026_27",
                "target_gw": 2,
                "source_run_id": "run1",
                "snapshot_dir": str(snap_dir),
                "snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
                "database_prediction_write": True,
                "final_deadline_freeze": False,
            }
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            # Tamper with the copied artifact after the receipt/snapshot manifest exist.
            (snap_dir / "summary.md").write_text("tampered\n", encoding="utf-8")

            result = validate_receipt_and_snapshot(
                run_dir, receipt_path, "2026_27", 2
            )
            self.assertTrue(
                any("artifact hash mismatch" in b.lower() for b in result["blockers"])
            )


if __name__ == "__main__":
    unittest.main()
