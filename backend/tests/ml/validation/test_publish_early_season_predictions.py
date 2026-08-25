from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.validation.publish_early_season_predictions import (
    canonical_match_result_label,
    create_immutable_snapshot,
    validate_preview_artifacts,
)


class EarlySeasonPublishGateTests(unittest.TestCase):
    def make_run_dir(self, root: Path) -> Path:
        run_dir = root / "run"
        run_dir.mkdir()

        prior_dir = root / "prior"
        prior_dir.mkdir()
        prior_player = prior_dir / "prediction_preview_csv.csv"
        prior_team = prior_dir / "effective_match_features_csv.csv"
        prior_player.write_text("x\n1\n", encoding="utf-8")
        prior_team.write_text("x\n1\n", encoding="utf-8")

        import hashlib

        def sha(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        manifest = {
            "pipeline_version": "early_season_prediction_pipeline_v0_1",
            "run_id": "test_run",
            "status": "PASS_PREVIEW",
            "season": "2026_27",
            "target_gw": 2,
            "prediction_mode": "early_season_blend",
            "weights": {"prior": 0.8, "current": 0.2},
            "database_prediction_write": False,
            "preview_only": True,
            "blockers": [],
            "warnings": [],
            "current_player_pool_rows": 2,
            "target_fixture_rows": 1,
            "scoreline_alignment": {
                "label_mismatch_rows": 0,
                "max_abs_probability_gap": 0.1,
            },
            "prior_artifacts": {
                "player_preview": str(prior_player),
                "player_preview_sha256": sha(prior_player),
                "effective_match_features": str(prior_team),
                "effective_match_features_sha256": sha(prior_team),
            },
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "bootstrap_snapshot.json").write_text("{}", encoding="utf-8")
        (run_dir / "summary.md").write_text("# summary\n", encoding="utf-8")

        pd.DataFrame(
            [
                {
                    "target_season": "2026_27",
                    "target_gw": 2,
                    "model_name": "early_season_blend_player_v0",
                    "fpl_player_id": 1,
                    "player_id": 101,
                    "predicted_points": 3.1,
                },
                {
                    "target_season": "2026_27",
                    "target_gw": 2,
                    "model_name": "early_season_blend_player_v0",
                    "fpl_player_id": 2,
                    "player_id": 102,
                    "predicted_points": 2.4,
                },
            ]
        ).to_csv(run_dir / "player_predictions_preview.csv", index=False)

        pd.DataFrame(
            [
                {
                    "target_season": "2026_27",
                    "target_gw": 2,
                    "model_name": "early_season_blend_match_v0",
                    "fpl_fixture_id": 11,
                    "fixture_id": 201,
                    "home_win_probability": 0.5,
                    "draw_probability": 0.25,
                    "away_win_probability": 0.25,
                    "predicted_result_label": "home_win",
                }
            ]
        ).to_csv(run_dir / "match_predictions_preview.csv", index=False)

        pd.DataFrame(
            [
                {
                    "target_season": "2026_27",
                    "target_gw": 2,
                    "model_name": "early_season_blend_scoreline_v0",
                    "fpl_fixture_id": 11,
                }
            ]
        ).to_csv(run_dir / "scoreline_preview.csv", index=False)
        return run_dir

    def test_match_result_label_maps_to_canonical_hda(self) -> None:
        self.assertEqual(canonical_match_result_label("home_win"), "H")
        self.assertEqual(canonical_match_result_label("draw"), "D")
        self.assertEqual(canonical_match_result_label("away_win"), "A")

    def test_valid_preview_passes_gate_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.make_run_dir(Path(tmp))
            result = validate_preview_artifacts(run_dir, "2026_27", 2)
            self.assertEqual(result["blockers"], [])
            self.assertEqual(len(result["players"]), 2)
            self.assertEqual(len(result["matches"]), 1)

    def test_changed_prior_hash_blocks_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self.make_run_dir(root)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            Path(manifest["prior_artifacts"]["player_preview"]).write_text("changed\n", encoding="utf-8")
            result = validate_preview_artifacts(run_dir, "2026_27", 2)
            self.assertTrue(any("SHA256 changed" in x for x in result["blockers"]))

    def test_duplicate_player_external_id_blocks_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.make_run_dir(Path(tmp))
            df = pd.read_csv(run_dir / "player_predictions_preview.csv")
            df.loc[1, "fpl_player_id"] = df.loc[0, "fpl_player_id"]
            df.to_csv(run_dir / "player_predictions_preview.csv", index=False)
            result = validate_preview_artifacts(run_dir, "2026_27", 2)
            self.assertTrue(any("duplicate fpl_player_id" in x for x in result["blockers"]))

    def test_snapshot_is_copy_verified_and_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self.make_run_dir(root)
            validated = validate_preview_artifacts(run_dir, "2026_27", 2)
            snapshot_root = root / "published"
            snapshot_root.mkdir()
            snapshot_dir, manifest = create_immutable_snapshot(
                validated, snapshot_root, "2026_27", 2
            )
            self.assertTrue((snapshot_dir / "snapshot_manifest.json").exists())
            self.assertEqual(manifest["player_rows"], 2)
            with self.assertRaises(RuntimeError):
                create_immutable_snapshot(validated, snapshot_root, "2026_27", 2)


if __name__ == "__main__":
    unittest.main()
