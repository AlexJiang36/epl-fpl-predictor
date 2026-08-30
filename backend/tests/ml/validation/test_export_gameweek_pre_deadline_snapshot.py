from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.rules.squad import load_squad_transfer_rules
from ml.contracts.gameweek_cycle import build_phase_id
from ml.contracts.squad_state import (
    ChipInventoryState,
    FreeTransferState,
    SquadPlayerState,
    SquadSelectionState,
    SquadState,
    squad_state_from_mapping,
)
from ml.decision.free_transfer_ledger import build_ledger_state
from ml.validation.export_gameweek_pre_deadline_snapshot import (
    GameweekPreDeadlineSnapshotError,
    SNAPSHOT_KIND_CANDIDATE,
    SNAPSHOT_KIND_FINAL,
    export_gameweek_pre_deadline_snapshot,
)


SEASON = "2026_27"


class GameweekPreDeadlineSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_squad_transfer_rules(SEASON)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.artifact_root = self.root / "snapshots"
        self.player_source = self.root / "player_model"
        self.match_source = self.root / "match_model"
        self.player_source.mkdir()
        self.match_source.mkdir()
        (self.player_source / "players.json").write_text(
            json.dumps({"rows": [{"player_id": 1, "points": 5.0}]}),
            encoding="utf-8",
        )
        (self.match_source / "matches.json").write_text(
            json.dumps({"rows": [{"fixture_id": 1, "home_win": 0.5}]}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def previous_state(self, *, bank_units=10, ft_count=1):
        players = []
        specs = (
            [("GKP", 2)]
            + [("DEF", 5)]
            + [("MID", 5)]
            + [("FWD", 3)]
        )
        pid = 1
        for position, count in specs:
            for _ in range(count):
                players.append(
                    SquadPlayerState(
                        fpl_player_id=pid,
                        player_name="P%s" % pid,
                        position=position,
                        club_id=pid,
                        purchase_price_units=50,
                        current_price_units=50,
                        selling_price_units=50,
                    )
                )
                pid += 1

        return SquadState(
            season=SEASON,
            gameweek=2,
            as_of_utc="2026-08-28T17:00:00Z",
            state_version="gw2_final_test",
            state_kind="model_team",
            state_status="frozen",
            source_phase_id=build_phase_id(2, "freeze"),
            source_run_id="gw2_final_test",
            players=tuple(players),
            selection=SquadSelectionState(
                starting_xi_player_ids=(1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15),
                bench_order_player_ids=(2, 6, 7, 12),
                captain_player_id=8,
                vice_captain_player_id=9,
            ),
            bank_units=bank_units,
            chip_inventory=ChipInventoryState(
                as_of_gameweek=2,
                entries=(),
            ),
            free_transfers=FreeTransferState(
                available_for_gameweek=3,
                count=ft_count,
            ),
        )

    def ledger(self, state):
        return build_ledger_state(
            season=SEASON,
            state_kind="model_team",
            gameweek=3,
            available_free_transfers=state.free_transfers.count,
            rules=self.rules,
        )

    def no_transfer_plan(self, state):
        return {
            "action": "NO TRANSFER",
            "transfer_count": 0,
            "transfers": [],
            "bank_before_units": state.bank_units,
            "bank_after_units": state.bank_units,
            "free_transfers_before": state.free_transfers.count,
            "free_transfers_used": 0,
            "charged_transfers": 0,
            "transfer_hit_points": 0,
            "free_transfers_next_gameweek": min(
                state.free_transfers.count + 1,
                int(self.rules.transfers["weekly"]["maximum_stored_free_transfers"]),
            ),
            "lineup": {
                "formation": "3-5-2",
                "starting_player_ids": [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14],
                "bench_order": [2, 6, 7, 15],
                "captain_player_id": 8,
                "vice_captain_player_id": 9,
                "starting_xi_predicted_points": 50.0,
                "captain_bonus_predicted_points": 7.0,
                "objective_points_before_transfer_cost": 57.0,
            },
            "projected_points_before_transfer_cost": 57.0,
            "transfer_cost_points": 0,
            "net_projected_points": 57.0,
            "net_gain_vs_no_transfer": 0.0,
        }

    def one_transfer_plan(self, state):
        plan = self.no_transfer_plan(state)
        plan.update(
            {
                "action": "TRANSFER",
                "transfer_count": 1,
                "transfers": [
                    {
                        "out_fpl_player_id": 8,
                        "out_name": "P8",
                        "out_position": "MID",
                        "out_selling_price_units": 50,
                        "in_fpl_player_id": 101,
                        "in_name": "UpgradeMID",
                        "in_position": "MID",
                        "in_club_id": 101,
                        "in_price_units": 55,
                        "candidate_projected_gain": 3.0,
                    }
                ],
                "bank_after_units": state.bank_units - 5,
                "free_transfers_used": 1,
                "free_transfers_next_gameweek": 1,
                "lineup": {
                    "formation": "3-5-2",
                    "starting_player_ids": [1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 101],
                    "bench_order": [2, 6, 7, 15],
                    "captain_player_id": 101,
                    "vice_captain_player_id": 9,
                    "starting_xi_predicted_points": 53.0,
                    "captain_bonus_predicted_points": 10.0,
                    "objective_points_before_transfer_cost": 63.0,
                },
                "projected_points_before_transfer_cost": 63.0,
                "net_projected_points": 63.0,
                "net_gain_vs_no_transfer": 6.0,
            }
        )
        return plan

    def artifact_spec(self, path, kind, run_id):
        return {
            "run_id": run_id,
            "artifact_kind": kind,
            "path": str(path),
            "season": SEASON,
            "target_gw": 3,
            "as_of_utc": "2026-09-01T16:00:00Z",
        }

    def export(self, *, final=False, plan=None, alex=None, run_id=None,
               as_of="2026-09-01T17:00:00Z",
               deadline="2026-09-01T18:00:00Z",
               state=None):
        state = state or self.previous_state()
        return export_gameweek_pre_deadline_snapshot(
            artifact_root=self.artifact_root,
            season=SEASON,
            target_gw=3,
            as_of_time=as_of,
            fpl_deadline_time=deadline,
            player_model_artifact=self.artifact_spec(
                self.player_source, "player_model_predictions", "player_run"
            ),
            match_model_artifact=self.artifact_spec(
                self.match_source, "match_model_predictions", "match_run"
            ),
            previous_model_team_state=state,
            chosen_plan=plan or self.no_transfer_plan(state),
            transfer_ledger_state=self.ledger(state),
            team_alex_reference=alex,
            final_freeze=final,
            run_id=run_id,
        )

    def manifest(self, result):
        return json.loads(
            Path(result["manifest_path"]).read_text(encoding="utf-8")
        )

    def model_team_state(self, result):
        return json.loads(
            Path(result["model_team_state_path"]).read_text(encoding="utf-8")
        )

    def test_candidate_snapshot_writes_four_tracks(self):
        result = self.export(run_id="candidate_a")
        self.assertEqual(result["status"], "PASS_CANDIDATE_SNAPSHOT")
        self.assertEqual(result["snapshot_kind"], SNAPSHOT_KIND_CANDIDATE)
        self.assertEqual(result["track_count"], 4)
        base = Path(result["snapshot_dir"]) / "tracks"
        for track in ("player_model", "match_model", "model_team", "team_alex"):
            self.assertTrue((base / track).is_dir())

    def test_candidate_is_not_silently_final(self):
        result = self.export(run_id="candidate_b")
        manifest = self.manifest(result)
        self.assertFalse(manifest["final_pre_deadline_snapshot_frozen"])
        self.assertFalse(manifest["explicit_final_freeze_mode"])
        self.assertEqual(self.model_team_state(result)["state_status"], "planning")

    def test_explicit_final_freeze_marks_state_frozen(self):
        result = self.export(final=True, run_id="final_a")
        self.assertEqual(result["status"], "PASS_FINAL_FREEZE")
        self.assertEqual(result["snapshot_kind"], SNAPSHOT_KIND_FINAL)
        self.assertTrue(result["final_pre_deadline_snapshot_frozen"])
        self.assertEqual(self.model_team_state(result)["state_status"], "frozen")

    def test_as_of_must_be_strictly_before_deadline(self):
        with self.assertRaises(GameweekPreDeadlineSnapshotError):
            self.export(
                run_id="bad_time",
                as_of="2026-09-01T18:00:00Z",
                deadline="2026-09-01T18:00:00Z",
            )

    def test_post_deadline_as_of_is_rejected(self):
        with self.assertRaises(GameweekPreDeadlineSnapshotError):
            self.export(
                run_id="after_deadline",
                as_of="2026-09-01T18:01:00Z",
                deadline="2026-09-01T18:00:00Z",
            )

    def test_candidate_snapshot_never_overwrites(self):
        self.export(run_id="candidate_same")
        with self.assertRaises(GameweekPreDeadlineSnapshotError):
            self.export(run_id="candidate_same")

    def test_final_snapshot_never_overwrites(self):
        self.export(final=True, run_id="final_same")
        with self.assertRaises(GameweekPreDeadlineSnapshotError):
            self.export(final=True, run_id="final_same")

    def test_previous_owned_state_fingerprint_is_recorded(self):
        state = self.previous_state()
        result = self.export(state=state, run_id="fingerprint")
        self.assertEqual(
            result["previous_owned_state_fingerprint"],
            state.owned_state_fingerprint,
        )
        self.assertEqual(
            self.manifest(result)["model_team"]["previous_owned_state_fingerprint"],
            state.owned_state_fingerprint,
        )

    def test_transfer_ledger_bank_and_free_transfers_are_recorded(self):
        state = self.previous_state(bank_units=10, ft_count=1)
        result = self.export(state=state, run_id="ledger")
        manifest = self.manifest(result)
        self.assertEqual(manifest["model_team"]["bank_before_units"], 10)
        self.assertEqual(manifest["model_team"]["bank_after_units"], 10)
        self.assertEqual(manifest["model_team"]["free_transfers_before"], 1)
        self.assertEqual(manifest["model_team"]["free_transfers_next_gameweek"], 2)
        self.assertEqual(manifest["transfer_ledger_state"]["gameweek"], 3)

    def test_no_transfer_is_preserved_as_decision(self):
        result = self.export(run_id="roll")
        self.assertEqual(
            self.manifest(result)["model_team"]["action"],
            "NO TRANSFER",
        )

    def test_transfer_plan_builds_target_owned_state(self):
        state = self.previous_state()
        result = self.export(
            state=state,
            plan=self.one_transfer_plan(state),
            run_id="transfer",
        )
        target = self.model_team_state(result)
        ids = {row["fpl_player_id"] for row in target["players"]}
        self.assertNotIn(8, ids)
        self.assertIn(101, ids)
        self.assertEqual(target["bank_units"], 5)

    def test_reoptimized_xi_captain_vice_are_persisted(self):
        state = self.previous_state()
        result = self.export(
            state=state,
            plan=self.one_transfer_plan(state),
            run_id="lineup",
        )
        target = self.model_team_state(result)
        self.assertIn(101, target["selection"]["starting_xi_player_ids"])
        self.assertEqual(target["selection"]["captain_player_id"], 101)
        self.assertEqual(target["selection"]["vice_captain_player_id"], 9)

    def test_player_and_match_sources_are_copied_and_hashed(self):
        result = self.export(run_id="source_copy")
        manifest = self.manifest(result)
        player_ref = json.loads(
            (
                Path(result["snapshot_dir"])
                / "tracks/player_model/reference.json"
            ).read_text(encoding="utf-8")
        )
        match_ref = json.loads(
            (
                Path(result["snapshot_dir"])
                / "tracks/match_model/reference.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            player_ref["sha256"],
            player_ref["snapshot_copy_sha256"],
        )
        self.assertEqual(
            match_ref["sha256"],
            match_ref["snapshot_copy_sha256"],
        )
        self.assertTrue(manifest["tracks"]["player_model"]["immutable"])
        self.assertTrue(manifest["tracks"]["match_model"]["immutable"])

    def test_source_artifact_after_snapshot_as_of_is_rejected(self):
        player_spec = self.artifact_spec(
            self.player_source, "player_model_predictions", "late_player"
        )
        player_spec["as_of_utc"] = "2026-09-01T17:30:00Z"
        state = self.previous_state()
        with self.assertRaises(GameweekPreDeadlineSnapshotError):
            export_gameweek_pre_deadline_snapshot(
                artifact_root=self.artifact_root,
                season=SEASON,
                target_gw=3,
                as_of_time="2026-09-01T17:00:00Z",
                fpl_deadline_time="2026-09-01T18:00:00Z",
                player_model_artifact=player_spec,
                match_model_artifact=self.artifact_spec(
                    self.match_source, "match_model_predictions", "match_run"
                ),
                previous_model_team_state=state,
                chosen_plan=self.no_transfer_plan(state),
                transfer_ledger_state=self.ledger(state),
                run_id="late_source",
            )

    def test_source_hash_mismatch_is_rejected(self):
        player_spec = self.artifact_spec(
            self.player_source, "player_model_predictions", "player_bad_hash"
        )
        player_spec["sha256"] = "0" * 64
        state = self.previous_state()
        with self.assertRaises(GameweekPreDeadlineSnapshotError):
            export_gameweek_pre_deadline_snapshot(
                artifact_root=self.artifact_root,
                season=SEASON,
                target_gw=3,
                as_of_time="2026-09-01T17:00:00Z",
                fpl_deadline_time="2026-09-01T18:00:00Z",
                player_model_artifact=player_spec,
                match_model_artifact=self.artifact_spec(
                    self.match_source, "match_model_predictions", "match_run"
                ),
                previous_model_team_state=state,
                chosen_plan=self.no_transfer_plan(state),
                transfer_ledger_state=self.ledger(state),
                run_id="bad_hash",
            )

    def test_previous_state_must_be_target_minus_one_and_frozen(self):
        state = self.previous_state()
        mapping = state.to_dict()
        mapping["gameweek"] = 1
        with self.assertRaises(GameweekPreDeadlineSnapshotError):
            export_gameweek_pre_deadline_snapshot(
                artifact_root=self.artifact_root,
                season=SEASON,
                target_gw=3,
                as_of_time="2026-09-01T17:00:00Z",
                fpl_deadline_time="2026-09-01T18:00:00Z",
                player_model_artifact=self.artifact_spec(
                    self.player_source,
                    "player_model_predictions",
                    "player_run",
                ),
                match_model_artifact=self.artifact_spec(
                    self.match_source,
                    "match_model_predictions",
                    "match_run",
                ),
                previous_model_team_state=mapping,
                chosen_plan=self.no_transfer_plan(state),
                transfer_ledger_state=self.ledger(state),
                run_id="wrong_previous_gw",
            )

    def test_ft_ledger_must_match_chosen_plan(self):
        state = self.previous_state(ft_count=1)
        plan = self.no_transfer_plan(state)
        plan["free_transfers_before"] = 2
        with self.assertRaises(GameweekPreDeadlineSnapshotError):
            self.export(state=state, plan=plan, run_id="ft_mismatch")

    def test_team_alex_reference_is_separate(self):
        alex_file = self.root / "team_alex.json"
        alex_file.write_text(
            json.dumps({"team": "alex", "temporary_squad": True}),
            encoding="utf-8",
        )
        alex = {
            "run_id": "team_alex_gw3",
            "season": SEASON,
            "target_gw": 3,
            "as_of_utc": "2026-09-01T16:30:00Z",
            "path": str(alex_file),
        }
        result = self.export(alex=alex, run_id="alex")
        manifest = self.manifest(result)
        self.assertTrue(result["team_alex_provided"])
        self.assertTrue(manifest["tracks"]["team_alex"]["provided"])
        self.assertFalse(
            manifest["team_alex"]["model_team_logic_consumed_team_alex"]
        )
        self.assertFalse(
            manifest["safety"]["team_alex_fed_into_model_team_logic"]
        )

    def test_team_alex_changes_do_not_change_model_team_track(self):
        alex_a = self.root / "alex_a.json"
        alex_b = self.root / "alex_b.json"
        alex_a.write_text(json.dumps({"captain": 1}), encoding="utf-8")
        alex_b.write_text(json.dumps({"captain": 999}), encoding="utf-8")

        first = self.export(
            alex={
                "run_id": "alex_a",
                "season": SEASON,
                "target_gw": 3,
                "path": str(alex_a),
            },
            run_id="alex_independent_a",
        )
        second = self.export(
            alex={
                "run_id": "alex_b",
                "season": SEASON,
                "target_gw": 3,
                "path": str(alex_b),
            },
            run_id="alex_independent_b",
        )
        first_state = squad_state_from_mapping(
            json.loads(
                Path(first["model_team_state_path"]).read_text(encoding="utf-8")
            )
        )
        second_state = squad_state_from_mapping(
            json.loads(
                Path(second["model_team_state_path"]).read_text(encoding="utf-8")
            )
        )
        self.assertEqual(first_state.season, second_state.season)
        self.assertEqual(first_state.gameweek, second_state.gameweek)
        self.assertEqual(first_state.as_of_utc, second_state.as_of_utc)
        self.assertEqual(first_state.state_kind, second_state.state_kind)
        self.assertEqual(first_state.state_status, second_state.state_status)
        self.assertEqual(first_state.players, second_state.players)
        self.assertEqual(first_state.selection, second_state.selection)
        self.assertEqual(first_state.bank_units, second_state.bank_units)
        self.assertEqual(first_state.chip_inventory, second_state.chip_inventory)
        self.assertEqual(first_state.free_transfers, second_state.free_transfers)
        self.assertEqual(first_state.predecessor, second_state.predecessor)
        self.assertEqual(first_state.shadow_optimal, second_state.shadow_optimal)

    def test_manifest_declares_no_opening_squad_or_actuals_use(self):
        result = self.export(run_id="safety")
        safety = self.manifest(result)["safety"]
        self.assertFalse(safety["opening_squad_optimizer_used"])
        self.assertFalse(safety["target_gw_actuals_consumed"])
        self.assertFalse(safety["writes_database"])
        self.assertFalse(safety["writes_manager_state"])

    def test_failed_export_leaves_no_partial_package(self):
        state = self.previous_state()
        plan = self.no_transfer_plan(state)
        plan["lineup"]["captain_player_id"] = 999
        with self.assertRaises(GameweekPreDeadlineSnapshotError):
            self.export(state=state, plan=plan, run_id="partial_fail")
        expected = (
            self.artifact_root
            / SEASON
            / "gw03"
            / "candidate"
            / "partial_fail"
        )
        self.assertFalse(expected.exists())


if __name__ == "__main__":
    unittest.main()
