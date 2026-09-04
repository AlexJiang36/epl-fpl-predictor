from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.pipeline.run_fpl_refresh import (
    StageRecorder,
    _final_freeze_has_scoreable_team_alex,
    run_final_freeze_from_candidate,
)


class TeamAlexFinalFreezeHotfixTests(unittest.TestCase):
    def _existing_final(self, root: Path, complete: bool) -> Path:
        final_dir = (
            root
            / "frozen-snapshots"
            / "2026_27"
            / "gw03"
            / "model-team"
            / "gw03_final_freeze_old"
        )
        final_dir.mkdir(parents=True)
        (final_dir / "snapshot_manifest.json").write_text(
            json.dumps(
                {
                    "season": "2026_27",
                    "target_gw": 3,
                    "snapshot_kind": "final_pre_deadline",
                    "final_pre_deadline_snapshot_frozen": True,
                }
            ),
            encoding="utf-8",
        )
        ref_dir = final_dir / "tracks" / "team_alex"
        ref_dir.mkdir(parents=True)
        reference = {"provided": False, "state_kind": "team_alex"}
        if complete:
            copied = ref_dir / "source" / "team_alex_state.json"
            copied.parent.mkdir(parents=True)
            copied.write_text("{}\n", encoding="utf-8")
            reference = {
                "provided": True,
                "state_kind": "team_alex",
                "snapshot_copy_path": str(copied.relative_to(final_dir)),
            }
        (ref_dir / "reference.json").write_text(
            json.dumps(reference), encoding="utf-8"
        )
        return final_dir

    def test_completeness_requires_copied_team_alex_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = self._existing_final(root / "a", complete=False)
            complete = self._existing_final(root / "b", complete=True)
            self.assertFalse(_final_freeze_has_scoreable_team_alex(incomplete))
            self.assertTrue(_final_freeze_has_scoreable_team_alex(complete))

    def test_complete_existing_final_is_reused_even_when_reference_is_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planning = Path(tmp)
            existing = self._existing_final(planning, complete=True)
            ref = planning / "team_alex_reference.json"
            ref.write_text("{}\n", encoding="utf-8")
            result = run_final_freeze_from_candidate(
                planning_root=planning,
                season="2026_27",
                target_gw=3,
                candidate_dir=planning / "missing_candidate",
                recorder=StageRecorder(),
                resume=False,
                team_alex_reference_json=str(ref),
            )
            self.assertEqual(existing, result)

    def test_incomplete_existing_final_is_not_reused_when_reference_is_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planning = Path(tmp)
            self._existing_final(planning, complete=False)
            ref = planning / "team_alex_reference.json"
            ref.write_text(
                json.dumps({"season": "2026_27", "target_gw": 3}),
                encoding="utf-8",
            )
            with patch(
                "ml.pipeline.run_fpl_refresh.validate_final_freeze_candidate",
                side_effect=RuntimeError("reached freeze validation"),
            ):
                with self.assertRaisesRegex(RuntimeError, "reached freeze validation"):
                    run_final_freeze_from_candidate(
                        planning_root=planning,
                        season="2026_27",
                        target_gw=3,
                        candidate_dir=planning / "candidate",
                        recorder=StageRecorder(),
                        resume=False,
                        team_alex_reference_json=str(ref),
                    )

    def test_team_alex_reference_reaches_day127b_exporter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planning = Path(tmp) / "planning"
            planning.mkdir()
            self._existing_final(planning, complete=False)
            candidate = Path(tmp) / "candidate"
            candidate.mkdir()
            prediction_run = Path(tmp) / "prediction_run"
            prediction_run.mkdir()
            previous = Path(tmp) / "previous.json"
            previous.write_text("{}\n", encoding="utf-8")
            current = Path(tmp) / "current.json"
            current.write_text("{}\n", encoding="utf-8")
            decision = candidate / "transfer_decision.json"
            decision.write_text("{}\n", encoding="utf-8")
            (candidate / "free_transfer_ledger_state.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (candidate / "match_model_source").mkdir()
            (candidate / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "inputs": {
                            "prediction_run": str(prediction_run),
                            "previous_frozen_model_team": str(previous),
                            "current_owned_state": str(current),
                        },
                        "resolved_prediction_mode": "early_season_blend",
                        "transfer_decision": str(decision),
                    }
                ),
                encoding="utf-8",
            )
            ref = Path(tmp) / "team_alex_reference.json"
            ref_payload = {
                "season": "2026_27",
                "target_gw": 3,
                "as_of_utc": "2026-09-04T03:41:00Z",
                "path": str(Path(tmp) / "team_alex_state.json"),
            }
            ref.write_text(json.dumps(ref_payload), encoding="utf-8")
            player_csv = Path(tmp) / "players.csv"
            player_csv.write_text("fpl_player_id,predicted_points\n", encoding="utf-8")
            captured = {}
            new_final = Path(tmp) / "new_final"

            def fake_export(**kwargs):
                captured.update(kwargs)
                return {
                    "final_pre_deadline_snapshot_frozen": True,
                    "snapshot_dir": str(new_final),
                    "manifest_path": str(new_final / "snapshot_manifest.json"),
                }

            with patch(
                "ml.pipeline.run_fpl_refresh.validate_final_freeze_candidate",
                return_value={
                    "as_of_utc": "2026-09-04T03:00:00Z",
                    "deadline_utc": "2026-09-04T17:30:00Z",
                },
            ), patch(
                "ml.pipeline.run_fpl_refresh.validate_prediction_run",
                return_value={"player_csv": player_csv},
            ), patch(
                "ml.pipeline.run_fpl_refresh._load_previous_state_for_candidate",
                return_value={},
            ), patch(
                "ml.pipeline.run_fpl_refresh.export_gameweek_pre_deadline_snapshot",
                side_effect=fake_export,
            ):
                result = run_final_freeze_from_candidate(
                    planning_root=planning,
                    season="2026_27",
                    target_gw=3,
                    candidate_dir=candidate,
                    recorder=StageRecorder(),
                    resume=False,
                    team_alex_reference_json=str(ref),
                )

            self.assertEqual(new_final, result)
            self.assertEqual(ref_payload, captured["team_alex_reference"])
            self.assertEqual(
                "2026-09-04T03:41:00Z",
                captured["as_of_time"],
            )
            self.assertEqual(
                "2026-09-04T03:00:00Z",
                captured["player_model_artifact"]["as_of_utc"],
            )
            self.assertEqual(
                "2026-09-04T03:00:00Z",
                captured["match_model_artifact"]["as_of_utc"],
            )
            self.assertTrue(captured["final_freeze"])


if __name__ == "__main__":
    unittest.main()
