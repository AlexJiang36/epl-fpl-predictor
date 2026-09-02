from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ml.eval.run_gameweek_evaluation import (
    LEGACY_TEAM_ALEX_ARCHIVE_CONTRACTS,
    GameweekEvaluationError,
    _discover_frozen_team_alex,
    build_decision_evaluation,
    build_team_evaluation,
    capture_or_reuse_final_actuals,
    discover_frozen_baseline,
    run_gameweek_evaluation,
    sha256_file,
    sha256_path,
)


class GameweekEvaluationIntegrationTests(unittest.TestCase):
    def write_json(self, path: Path, payload) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_csv(self, path: Path, rows) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def player_rows(self):
        positions = ["GKP", "GKP"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
        rows = []
        for index, position in enumerate(positions, start=1):
            rows.append(
                {
                    "fpl_player_id": index,
                    "web_name": "P%s" % index,
                    "player_name": "Player %s" % index,
                    "position": position,
                    "team_short_name": "T%s" % ((index - 1) % 5 + 1),
                    "status": "a",
                    "fixture_count": 1,
                    "predicted_points": 3.0 + index / 10.0,
                    "expected_minutes_total": 75.0,
                    "blended_appearance_probability": 0.9,
                    "blended_start_probability": 0.8,
                }
            )
        return rows

    def selection(self):
        starters = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
        bench = [2, 6, 7, 15]
        return {
            "starting_xi_player_ids": starters,
            "bench_order_player_ids": bench,
            "captain_player_id": 13,
            "vice_captain_player_id": 14,
        }

    def build_actuals(self, planning: Path, season: str = "2026_27", gw: int = 2) -> Path:
        root = planning / "gw-post" / season / ("gw%02d" % gw) / "actuals" / "final_test"
        root.mkdir(parents=True)
        elements = []
        for row in self.player_rows():
            pid = int(row["fpl_player_id"])
            elements.append(
                {
                    "id": pid,
                    "stats": {
                        "total_points": pid % 7,
                        "minutes": 90,
                        "starts": 1,
                    },
                }
            )
        live = self.write_json(root / "event.json", {"elements": elements})
        fixtures = self.write_json(
            root / "fixtures.json",
            [
                {
                    "id": 101,
                    "event": gw,
                    "finished": True,
                    "team_h_score": 2,
                    "team_a_score": 1,
                }
            ],
        )
        bootstrap = self.write_json(
            root / "bootstrap.json",
            {
                "events": [
                    {
                        "id": gw,
                        "finished": True,
                        "data_checked": True,
                        "deadline_time": "2026-08-28T17:30:00Z",
                    }
                ]
            },
        )
        manifest = {
            "contract_version": "test_actuals",
            "status": "FINAL",
            "season": season,
            "gw": gw,
            "captured_at_utc": "20260830T120000Z",
            "event_live_file": live.name,
            "event_live_sha256": sha256_file(live),
            "fixtures_file": fixtures.name,
            "fixtures_sha256": sha256_file(fixtures),
            "bootstrap_file": bootstrap.name,
            "bootstrap_sha256": sha256_file(bootstrap),
            "event_finished": True,
            "event_data_checked": True,
            "all_fixtures_finished": True,
        }
        return self.write_json(root / "gw2_actuals_manifest_test_FINAL.json", manifest)

    def build_generic_final_snapshot(
        self,
        planning: Path,
        season: str = "2026_27",
        gw: int = 2,
        day128b_match_schema: bool = False,
    ) -> Path:
        root = planning / "frozen-snapshots" / season / ("gw%02d" % gw) / "final" / "test_final"
        player_source = root / "tracks" / "player_model" / "source"
        self.write_csv(player_source, self.player_rows())
        self.write_json(
            root / "tracks" / "player_model" / "reference.json",
            {
                "snapshot_copy_path": "tracks/player_model/source",
                "snapshot_copy_sha256": sha256_path(player_source),
            },
        )

        match_source = root / "tracks" / "match_model" / "source"
        match_row = {
            "fpl_fixture_id": 101,
            "home_team_short_name": "AAA",
            "away_team_short_name": "BBB",
            "home_win_probability": 0.6,
            "draw_probability": 0.2,
            "away_win_probability": 0.2,
            "predicted_result_label": "home_win",
            "model_name": "test_match",
        }
        scoreline_row = {
            "fpl_fixture_id": 101,
            "top_1_scoreline": "2-1",
        }
        if day128b_match_schema:
            match_row.update(
                {
                    "expected_home_goals": 1.8,
                    "expected_away_goals": 1.0,
                    "expected_total_goals": 2.8,
                }
            )
            scoreline_row.update(
                {
                    "scoreline_home_win_probability": 0.55,
                    "scoreline_draw_probability": 0.25,
                    "scoreline_away_win_probability": 0.20,
                    "top_1_scoreline_probability": 0.20,
                }
            )
        else:
            scoreline_row.update(
                {
                    "expected_home_goals": 1.8,
                    "expected_away_goals": 1.0,
                    "top_2_scoreline": "1-0",
                    "top_3_scoreline": "1-1",
                    "top_4_scoreline": "2-0",
                    "top_5_scoreline": "0-0",
                }
            )
        self.write_csv(match_source / "match_predictions_preview.csv", [match_row])
        self.write_csv(match_source / "scoreline_preview.csv", [scoreline_row])
        self.write_json(
            match_source / "source_reference.json",
            {"artifact_type": "test_match_bundle"},
        )
        self.write_json(
            root / "tracks" / "match_model" / "reference.json",
            {
                "snapshot_copy_path": "tracks/match_model/source",
                "snapshot_copy_sha256": sha256_path(match_source),
            },
        )

        players = [
            {"fpl_player_id": row["fpl_player_id"], "player_name": row["web_name"]}
            for row in self.player_rows()
        ]
        state = {
            "season": season,
            "gameweek": gw,
            "state_kind": "model_team",
            "state_status": "frozen",
            "players": players,
            "selection": self.selection(),
        }
        self.write_json(root / "tracks" / "model_team" / "model_team_state.json", state)
        self.write_json(
            root / "tracks" / "model_team" / "chosen_plan.json",
            {
                "season": season,
                "target_gw": gw,
                "action": "NO TRANSFER",
                "transfer_count": 0,
                "transfer_hit_points": 0,
                "net_gain_vs_no_transfer": 0.0,
                "lineup": self.selection(),
            },
        )

        alex_source = root / "tracks" / "team_alex" / "source"
        alex_state = {
            "season": season,
            "gameweek": gw,
            "state_kind": "team_alex",
            "state_status": "frozen",
            "players": players,
            "selection": self.selection(),
            "snapshot_kind": "final_pre_deadline",
            "final_pre_deadline_snapshot_frozen": True,
        }
        self.write_json(alex_source, alex_state)
        self.write_json(
            root / "tracks" / "team_alex" / "reference.json",
            {
                "provided": True,
                "snapshot_copy_path": "tracks/team_alex/source",
                "snapshot_copy_sha256": sha256_path(alex_source),
            },
        )

        manifest = {
            "snapshot_version": "test",
            "status": "PASS_FINAL_SNAPSHOT",
            "snapshot_kind": "final_pre_deadline",
            "final_pre_deadline_snapshot_frozen": True,
            "explicit_final_freeze_mode": True,
            "season": season,
            "target_gw": gw,
            "run_id": "test_final",
            "as_of_utc": "2026-08-28T16:00:00Z",
            "fpl_deadline_utc": "2026-08-28T17:30:00Z",
            "as_of_before_deadline": True,
            "package_fingerprint": "test-package-fingerprint",
            "safety": {
                "target_gw_actuals_consumed": False,
                "post_deadline_reconstruction_allowed": False,
                "overwrite_allowed": False,
            },
        }
        self.write_json(root / "snapshot_manifest.json", manifest)
        checksum_rows = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.name == "SHA256SUMS.txt":
                continue
            checksum_rows.append("%s  %s" % (sha256_file(path), path.relative_to(root).as_posix()))
        (root / "SHA256SUMS.txt").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
        return root

    def test_generic_final_snapshot_produces_complete_post_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planning = Path(tmp) / "planning"
            self.build_generic_final_snapshot(planning)
            actual_manifest = self.build_actuals(planning)
            output = run_gameweek_evaluation(
                planning_root=planning,
                season="2026_27",
                gw=2,
                actual_manifest_path=actual_manifest,
            )
            summary = json.loads((output / "evaluation_summary.json").read_text())
            manifest = json.loads((output / "evaluation_manifest.json").read_text())
            self.assertTrue(summary["leakage_safe"])
            self.assertFalse(summary["prediction_regeneration"])
            self.assertFalse(summary["post_result_pre_reconstruction"])
            self.assertEqual(summary["frozen_inputs"]["kind"], "day127b_final_snapshot")
            self.assertIn("player_model", summary)
            self.assertIn("match_model", summary)
            self.assertIn("model_team", summary)
            self.assertIn("team_alex", summary)
            self.assertIn("decision_evaluation", summary)
            self.assertTrue(manifest["immutable"])

    def test_day128b_match_schema_uses_frozen_match_xg_and_partial_scoreline_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planning = Path(tmp) / "planning"
            self.build_generic_final_snapshot(planning, day128b_match_schema=True)
            actual_manifest = self.build_actuals(planning)
            output = run_gameweek_evaluation(
                planning_root=planning,
                season="2026_27",
                gw=2,
                actual_manifest_path=actual_manifest,
            )
            summary = json.loads((output / "evaluation_summary.json").read_text())
            with (output / "match_evaluation_rows.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(summary["match_model"]["exact_score_top1_accuracy"], 1.0)
            self.assertIsNone(summary["match_model"]["exact_score_top3_accuracy"])
            self.assertIsNone(summary["match_model"]["exact_score_top5_accuracy"])
            self.assertEqual(rows[0]["scoreline_rank_count_available"], "1")
            self.assertEqual(float(rows[0]["expected_home_goals"]), 1.8)
            self.assertEqual(float(rows[0]["expected_away_goals"]), 1.0)

    def test_final_snapshot_that_consumed_actuals_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planning = Path(tmp) / "planning"
            root = self.build_generic_final_snapshot(planning)
            manifest_path = root / "snapshot_manifest.json"
            payload = json.loads(manifest_path.read_text())
            payload["safety"]["target_gw_actuals_consumed"] = True
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            # Rebuild checksum so the failure is the leakage guard, not integrity drift.
            checksum_rows = []
            for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
                checksum_rows.append("%s  %s" % (sha256_file(path), path.relative_to(root).as_posix()))
            (root / "SHA256SUMS.txt").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(GameweekEvaluationError, "consumed target-GW actuals"):
                discover_frozen_baseline(planning, "2026_27", 2)

    def test_missing_team_alex_final_baseline_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planning = Path(tmp) / "planning"
            root = self.build_generic_final_snapshot(planning)
            (root / "tracks" / "team_alex" / "reference.json").write_text(
                json.dumps({"provided": False}), encoding="utf-8"
            )
            checksum_rows = []
            for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
                checksum_rows.append("%s  %s" % (sha256_file(path), path.relative_to(root).as_posix()))
            (root / "SHA256SUMS.txt").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(GameweekEvaluationError, "Team Alex"):
                discover_frozen_baseline(planning, "2026_27", 2)

    def test_existing_final_actual_capture_is_idempotently_reused_without_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planning = Path(tmp) / "planning"
            existing = self.build_actuals(planning)
            with patch(
                "ml.eval.run_gameweek_evaluation.fetch_json_bytes",
                side_effect=AssertionError("validated FINAL actuals should be reused"),
            ):
                result = capture_or_reuse_final_actuals(
                    planning_root=planning,
                    season="2026_27",
                    gw=2,
                    reuse_existing=False,
                    reuse_only=False,
                )
            self.assertEqual(result, existing)

    def test_actual_capture_requires_full_fpl_finality(self) -> None:
        live = json.dumps({"elements": [{"id": 1, "stats": {}}]}).encode()
        fixtures = json.dumps(
            [{"id": 101, "event": 2, "finished": False, "team_h_score": 1, "team_a_score": 0}]
        ).encode()
        bootstrap = json.dumps(
            {"events": [{"id": 2, "finished": False, "data_checked": False}]}
        ).encode()
        responses = [
            (live, json.loads(live)),
            (fixtures, json.loads(fixtures)),
            (bootstrap, json.loads(bootstrap)),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "ml.eval.run_gameweek_evaluation.fetch_json_bytes",
            side_effect=responses,
        ):
            with self.assertRaisesRegex(GameweekEvaluationError, "actuals are incomplete"):
                capture_or_reuse_final_actuals(
                    planning_root=Path(tmp),
                    season="2026_27",
                    gw=2,
                )

    def test_transfer_counterfactual_is_scored_when_frozen_lineup_is_available(self) -> None:
        starter_ids = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
        model_result = {
            "primary_captain_result": {
                "effective_captain": "P13",
                "bonus_points": 6,
                "vice_triggered": False,
            },
            "primary_frozen_xi_actual_total": 52,
            "frozen_starter_ids": starter_ids,
            "captain": "P13",
            "vice_captain": "P14",
        }
        by_id = {
            index: {
                "fpl_player_id": index,
                "web_name": "P%s" % index,
                "player_name": "P%s" % index,
                "actual_points": index % 7,
                "actual_minutes": 90,
            }
            for index in range(1, 16)
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_json(
                Path(tmp) / "decision.json",
                {
                    "winner": {
                        "action": "TRANSFER",
                        "transfer_count": 1,
                        "transfer_hit_points": 4,
                        "net_gain_vs_no_transfer": 2.0,
                    },
                    "no_transfer_plan": {
                        "action": "NO TRANSFER",
                        "lineup": {
                            "starting_player_ids": starter_ids,
                            "captain_player_id": 13,
                            "vice_captain_player_id": 14,
                        },
                    },
                },
            )
            result = build_decision_evaluation(path, model_result, by_id)
            transfer = result["transfer_no_transfer"]
            self.assertEqual(
                transfer["counterfactual_status"],
                "evaluated_from_frozen_no_transfer_lineup",
            )
            self.assertEqual(transfer["actual_chosen_total_after_hits"], 48.0)
            self.assertIsNotNone(transfer["actual_no_transfer_total"])
            self.assertAlmostEqual(
                transfer["actual_counterfactual_vs_no_transfer"],
                48.0 - transfer["actual_no_transfer_total"],
            )

    def test_transfer_evaluation_is_explicit_when_counterfactual_missing(self) -> None:
        model_result = {
            "primary_captain_result": {
                "effective_captain": "P13",
                "bonus_points": 6,
                "vice_triggered": False,
            },
            "frozen_starter_ids": [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14],
            "captain": "P13",
            "vice_captain": "P14",
        }
        by_id = {
            index: {
                "fpl_player_id": index,
                "web_name": "P%s" % index,
                "player_name": "P%s" % index,
                "actual_points": index % 7,
            }
            for index in range(1, 16)
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_json(
                Path(tmp) / "decision.json",
                {
                    "winner": {
                        "action": "TRANSFER",
                        "transfer_count": 1,
                        "transfer_hit_points": 0,
                        "net_gain_vs_no_transfer": 1.25,
                    }
                },
            )
            result = build_decision_evaluation(path, model_result, by_id)
            self.assertEqual(result["transfer_no_transfer"]["status"], "evaluated_from_frozen_decision")
            self.assertEqual(
                result["transfer_no_transfer"]["counterfactual_status"],
                "not_available_frozen_counterfactual_lineup_missing",
            )

    def test_legacy_gw2_team_alex_zip_is_sha_pinned_and_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gw_root = Path(tmp) / "frozen-snapshots" / "2026_27" / "gw02"
            archive_path = gw_root / "team-alex" / "TEAM_ALEX_GW2_FREE_HIT_SNAPSHOT.zip"
            archive_path.parent.mkdir(parents=True)
            payload = {
                "artifact_type": "team_alex_free_hit_snapshot",
                "season": "2026_27",
                "gameweek": 2,
                "team": "Team Alex",
                "chip": {
                    "name": "free_hit",
                    "active": True,
                    "persistent_squad_overwrite": False,
                },
                "captain": "P13",
                "vice_captain": "P14",
                "starting_xi": [{"name": "P%s" % i} for i in [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]],
                "bench": [{"name": "P%s" % i} for i in [2, 6, 7, 15]],
            }
            member = "team_alex_gw2_free_hit_snapshot.json"
            with zipfile.ZipFile(str(archive_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
                info = zipfile.ZipInfo(member, date_time=(2026, 8, 28, 3, 46, 0))
                archive.writestr(info, json.dumps(payload))
            contract = {
                "relative_path": "team-alex/TEAM_ALEX_GW2_FREE_HIT_SNAPSHOT.zip",
                "sha256": sha256_file(archive_path),
                "member": member,
                "artifact_type": "team_alex_free_hit_snapshot",
            }
            with patch.dict(LEGACY_TEAM_ALEX_ARCHIVE_CONTRACTS, {("2026_27", 2): contract}, clear=True):
                found = _discover_frozen_team_alex(
                    gw_root,
                    "2026-08-28T17:30:00Z",
                    "2026_27",
                    2,
                )
            self.assertEqual(found, archive_path)

    def test_legacy_gw2_team_alex_zip_rejects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gw_root = Path(tmp) / "frozen-snapshots" / "2026_27" / "gw02"
            archive_path = gw_root / "team-alex" / "TEAM_ALEX_GW2_FREE_HIT_SNAPSHOT.zip"
            archive_path.parent.mkdir(parents=True)
            with zipfile.ZipFile(str(archive_path), "w") as archive:
                info = zipfile.ZipInfo(
                    "team_alex_gw2_free_hit_snapshot.json",
                    date_time=(2026, 8, 28, 3, 46, 0),
                )
                archive.writestr(
                    info,
                    json.dumps(
                        {
                            "artifact_type": "team_alex_free_hit_snapshot",
                            "season": "2026_27",
                            "gameweek": 2,
                            "team": "Team Alex",
                        }
                    ),
                )
            contract = {
                "relative_path": "team-alex/TEAM_ALEX_GW2_FREE_HIT_SNAPSHOT.zip",
                "sha256": "0" * 64,
                "member": "team_alex_gw2_free_hit_snapshot.json",
                "artifact_type": "team_alex_free_hit_snapshot",
            }
            with patch.dict(LEGACY_TEAM_ALEX_ARCHIVE_CONTRACTS, {("2026_27", 2): contract}, clear=True):
                with self.assertRaisesRegex(GameweekEvaluationError, "fingerprint mismatch"):
                    _discover_frozen_team_alex(
                        gw_root,
                        "2026-08-28T17:30:00Z",
                        "2026_27",
                        2,
                    )

    def test_team_evaluation_supports_name_only_frozen_free_hit_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "team_alex.json"
            starters = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
            bench = [2, 6, 7, 15]
            self.write_json(
                state_path,
                {
                    "starting_xi": [{"name": "P%s" % i} for i in starters],
                    "bench": [{"name": "P%s" % i} for i in bench],
                    "captain": "P13",
                    "vice_captain": "P14",
                },
            )
            player_by_id = {}
            for row in self.player_rows():
                pid = row["fpl_player_id"]
                player_by_id[pid] = {
                    **row,
                    "player_name": "P%s" % pid,
                    "web_name": "P%s" % pid,
                    "actual_points": pid % 6,
                    "actual_minutes": 90,
                }
            result = build_team_evaluation(
                "Team Alex / Gliding Tiger",
                state_path,
                player_by_id,
                "team_alex",
            )
            self.assertEqual(result["frozen_starter_ids"], starters)
            self.assertEqual(result["bench_ids"], bench)
            self.assertEqual(result["captain"], "P13")
            self.assertEqual(result["vice_captain"], "P14")



if __name__ == "__main__":
    unittest.main()
