from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from ml.pipeline.run_fpl_refresh import (
    DAG_VERSION,
    FAIL_FAST,
    GameweekDAGError,
    StageAdapter,
    StageRecorder,
    build_dag_plan,
    build_market_csv,
    build_top10_by_position,
    execute_stage_adapters,
    parse_args,
    render_dag_plan,
    stage_adapter,
    validate_phase_request,
    validate_stage_adapters,
    build_current_owned_state_for_candidates,
    build_target_gw_horizon_artifacts,
    canonicalize_legacy_gw2_model_team_state,
    resolve_runner_prediction_mode,
    run_pre,
    run_post_pipeline,
    selection_eligibility_from_live_prediction,
)
from ml.contracts.squad_state import (
    ChipInventoryState,
    FreeTransferState,
    SquadPlayerState,
    SquadSelectionState,
    SquadState,
)


class WeeklyRunnerExistingUtilityTests(unittest.TestCase):
    def prediction_frame(self) -> pd.DataFrame:
        rows = []
        pid = 1
        for position in ("GKP", "DEF", "MID", "FWD"):
            for rank in range(12):
                rows.append(
                    {
                        "player_id": pid + 1000,
                        "fpl_player_id": pid,
                        "web_name": "%s_%02d" % (position, rank),
                        "position": position,
                        "team_id": (rank % 20) + 1,
                        "team_name": "Team %02d" % ((rank % 20) + 1),
                        "team_short_name": "T%02d" % ((rank % 20) + 1),
                        "now_cost": 45 + rank,
                        "status": "a",
                        "predicted_points": float(20 - rank),
                        "chance_of_playing_next_round": 100,
                        "expected_minutes_total": 90.0,
                        "blended_start_probability": 1.0,
                        "blended_appearance_probability": 1.0,
                        "news": "",
                    }
                )
                pid += 1
        return pd.DataFrame(rows)

    def test_market_contract_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "predictions.csv"
            output = root / "market.csv"
            self.prediction_frame().to_csv(source, index=False)

            result = build_market_csv(source, output)
            self.assertEqual(result, output)
            market = pd.read_csv(output)
            self.assertEqual(len(market), 48)
            self.assertFalse(market["fpl_player_id"].duplicated().any())
            self.assertFalse(market["predicted_points"].isna().any())

    def test_top10_by_position_contract_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "predictions.csv"
            self.prediction_frame().to_csv(source, index=False)

            csv_path, json_path, md_path, grouped = build_top10_by_position(
                source,
                root / "out",
                10,
            )

            self.assertTrue(csv_path.is_file())
            self.assertTrue(json_path.is_file())
            self.assertTrue(md_path.is_file())
            for position in ("GKP", "DEF", "MID", "FWD"):
                self.assertEqual(len(grouped[position]), 10)
                points = [row["predicted_points"] for row in grouped[position]]
                self.assertEqual(points, sorted(points, reverse=True))


class UnifiedGameweekDAGSkeletonTests(unittest.TestCase):
    def stage_names(self, plan):
        return [row["name"] for row in plan["stages"]]

    def test_pre_dag_has_explicit_stateful_dependency_order(self) -> None:
        plan = build_dag_plan(
            phase="pre",
            season="2026_27",
            target_gw=3,
        )
        names = self.stage_names(plan)

        expected_chain = [
            "owned_squad_state",
            "free_transfer_ledger",
            "transfer_candidates",
            "transfer_decision",
            "lineup_selection",
            "pre_deadline_candidate",
        ]
        indexes = [names.index(name) for name in expected_chain]
        self.assertEqual(indexes, sorted(indexes))

        by_name = {row["name"]: row for row in plan["stages"]}
        self.assertIn(
            "free_transfer_ledger",
            by_name["transfer_candidates"]["depends_on"],
        )
        self.assertIn(
            "transfer_candidates",
            by_name["transfer_decision"]["depends_on"],
        )
        self.assertIn(
            "transfer_decision",
            by_name["lineup_selection"]["depends_on"],
        )

    def test_pre_dag_reuses_day127_contracts_not_third_decision_path(self) -> None:
        plan = build_dag_plan(
            phase="pre",
            season="2026_27",
            target_gw=3,
        )
        by_name = {row["name"]: row for row in plan["stages"]}
        self.assertEqual(
            by_name["free_transfer_ledger"]["implementation"],
            "ml.decision.free_transfer_ledger",
        )
        self.assertEqual(
            by_name["transfer_candidates"]["implementation"],
            "ml.decision.generate_transfer_candidates",
        )
        self.assertEqual(
            by_name["transfer_decision"]["implementation"],
            "ml.decision.optimize_single_gw_transfers",
        )
        self.assertEqual(
            by_name["lineup_selection"]["implementation"],
            "ml.decision.optimize_single_gw_transfers",
        )
        self.assertIn(
            "not a third lineup optimizer",
            by_name["lineup_selection"]["note"],
        )
        self.assertEqual(
            by_name["pre_deadline_candidate"]["implementation"],
            "ml.validation.export_gameweek_pre_deadline_snapshot",
        )

    def test_freeze_phase_requires_explicit_final_freeze(self) -> None:
        with self.assertRaises(GameweekDAGError):
            build_dag_plan(
                phase="freeze",
                season="2026_27",
                target_gw=3,
                final_freeze=False,
            )

    def test_explicit_freeze_dag_reuses_day127b_exporter(self) -> None:
        plan = build_dag_plan(
            phase="freeze",
            season="2026_27",
            target_gw=3,
            final_freeze=True,
        )
        names = self.stage_names(plan)
        self.assertEqual(
            names,
            ["freeze_window_validation", "final_freeze_export"],
        )
        final_stage = plan["stages"][-1]
        self.assertEqual(
            final_stage["implementation"],
            "ml.validation.export_gameweek_pre_deadline_snapshot",
        )

    def test_auto_without_flag_never_contains_final_freeze(self) -> None:
        plan = build_dag_plan(
            phase="auto",
            season="2026_27",
            target_gw=3,
            final_freeze=False,
        )
        names = self.stage_names(plan)
        self.assertNotIn("final_freeze_export", names)
        self.assertFalse(plan["final_freeze_requested"])

    def test_auto_with_explicit_flag_plans_freeze_after_candidate(self) -> None:
        plan = build_dag_plan(
            phase="auto",
            season="2026_27",
            target_gw=3,
            final_freeze=True,
        )
        names = self.stage_names(plan)
        self.assertLess(
            names.index("pre_deadline_candidate"),
            names.index("freeze_window_validation"),
        )
        self.assertLess(
            names.index("freeze_window_validation"),
            names.index("final_freeze_export"),
        )
        self.assertTrue(plan["final_freeze_requested"])

    def test_auto_runs_previous_gw_post_before_target_pre_in_dag(self) -> None:
        plan = build_dag_plan(
            phase="auto",
            season="2026_27",
            target_gw=3,
        )
        names = self.stage_names(plan)
        self.assertEqual(names[0], "previous_gw_post")
        self.assertLess(
            names.index("previous_gw_post"),
            names.index("live_ingest"),
        )

    def test_final_freeze_flag_is_illegal_for_pre_or_post(self) -> None:
        for phase in ("pre", "post", "status"):
            with self.subTest(phase=phase):
                with self.assertRaises(GameweekDAGError):
                    validate_phase_request(phase, final_freeze=True)

    def test_post_dependency_order_is_explicit(self) -> None:
        plan = build_dag_plan(
            phase="post",
            season="2026_27",
            target_gw=3,
        )
        names = self.stage_names(plan)
        self.assertEqual(names, ["actuals_finality", "post_evaluation"])
        self.assertEqual(
            plan["stages"][1]["depends_on"],
            ["actuals_finality"],
        )
        self.assertEqual(
            plan["stages"][0]["implementation"],
            "ml.eval.run_gameweek_evaluation.capture_or_reuse_final_actuals",
        )
        self.assertEqual(
            plan["stages"][1]["implementation"],
            "ml.eval.run_gameweek_evaluation.run_gameweek_evaluation",
        )

    def test_missing_dependency_is_rejected(self) -> None:
        stages = [
            stage_adapter(
                "a",
                "pre",
                depends_on=("missing",),
                implementation="dummy",
            )
        ]
        with self.assertRaises(GameweekDAGError):
            validate_stage_adapters(stages)

    def test_dependency_cycle_is_rejected(self) -> None:
        stages = [
            stage_adapter(
                "a",
                "pre",
                depends_on=("b",),
                implementation="dummy",
            ),
            stage_adapter(
                "b",
                "pre",
                depends_on=("a",),
                implementation="dummy",
            ),
        ]
        with self.assertRaises(GameweekDAGError):
            validate_stage_adapters(stages)

    def test_stage_recorder_records_start_end_duration_and_planned_io(self) -> None:
        recorder = StageRecorder()
        started = time.time() - 0.01
        recorder.add(
            "example",
            "PASS",
            started,
            outputs=["actual.json"],
            dependencies=["dependency"],
            planned_inputs=["input_artifact"],
            planned_outputs=["planned_artifact"],
        )
        row = recorder.rows[0]
        self.assertEqual(row["stage"], "example")
        self.assertEqual(row["status"], "PASS")
        self.assertTrue(row["started_at_utc"].endswith("Z"))
        self.assertTrue(row["ended_at_utc"].endswith("Z"))
        self.assertGreaterEqual(row["duration_seconds"], 0.0)
        self.assertEqual(row["dependencies"], ["dependency"])
        self.assertEqual(row["planned_inputs"], ["input_artifact"])
        self.assertEqual(row["planned_outputs"], ["planned_artifact"])
        self.assertEqual(row["outputs"], ["actual.json"])

    def test_generic_adapter_executor_is_fail_fast(self) -> None:
        stages = [
            stage_adapter("first", "pre", implementation="dummy"),
            stage_adapter(
                "second",
                "pre",
                depends_on=("first",),
                implementation="dummy",
            ),
            stage_adapter(
                "third",
                "pre",
                depends_on=("second",),
                implementation="dummy",
            ),
        ]
        called = []

        def executor(stage: StageAdapter):
            called.append(stage.name)
            if stage.name == "second":
                raise RuntimeError("boom")
            return ["%s.out" % stage.name]

        recorder = StageRecorder()
        with self.assertRaises(RuntimeError):
            execute_stage_adapters(stages, executor, recorder)

        self.assertTrue(FAIL_FAST)
        self.assertEqual(called, ["first", "second"])
        self.assertEqual(
            [row["status"] for row in recorder.rows],
            ["PASS", "FAILED"],
        )

    def test_dry_run_plan_is_validation_only(self) -> None:
        plan = build_dag_plan(
            phase="pre",
            season="2026_27",
            target_gw=3,
            resume=True,
            publish_predictions=True,
        )
        self.assertEqual(plan["dag_version"], DAG_VERSION)
        self.assertTrue(plan["valid"])
        self.assertTrue(plan["resume"])
        self.assertFalse(plan["model_logic_executed"])
        rendered = render_dag_plan(plan)
        self.assertIn("DAG_VALID: True", rendered)
        self.assertIn("MODEL_LOGIC_EXECUTED: False", rendered)
        self.assertIn("prediction_publish", rendered)

    def test_cli_accepts_required_day128a_flags(self) -> None:
        argv = [
            "run_fpl_refresh",
            "--phase",
            "auto",
            "--season",
            "2026_27",
            "--target-gw",
            "3",
            "--dry-run",
            "--resume",
            "--final-freeze",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.phase, "auto")
        self.assertTrue(args.dry_run)
        self.assertTrue(args.resume)
        self.assertTrue(args.final_freeze)


class UnifiedPrePipelineIntegrationTests(unittest.TestCase):
    def previous_state(self) -> SquadState:
        positions = (
            ["GKP"] * 2
            + ["DEF"] * 5
            + ["MID"] * 5
            + ["FWD"] * 3
        )
        players = []
        for index, position in enumerate(positions, start=1):
            price = 50 + index
            players.append(
                SquadPlayerState(
                    fpl_player_id=index,
                    player_name="P%s" % index,
                    position=position,
                    club_id=((index - 1) % 5) + 1,
                    purchase_price_units=price,
                    current_price_units=price,
                    selling_price_units=price,
                )
            )
        return SquadState(
            season="2026_27",
            gameweek=2,
            as_of_utc="2026-08-28T16:00:00Z",
            state_version="test_frozen_v1",
            state_kind="model_team",
            state_status="frozen",
            source_phase_id="GW02-FREEZE",
            source_run_id="gw02_test_freeze",
            players=tuple(players),
            selection=SquadSelectionState(
                starting_xi_player_ids=tuple(range(1, 12)),
                bench_order_player_ids=(12, 13, 14, 15),
                captain_player_id=1,
                vice_captain_player_id=3,
            ),
            bank_units=10,
            chip_inventory=ChipInventoryState(
                as_of_gameweek=2,
                entries=(),
            ),
            free_transfers=FreeTransferState(
                available_for_gameweek=3,
                count=1,
            ),
            predecessor=None,
            shadow_optimal=None,
        )

    def prediction_rows(self):
        previous = self.previous_state()
        rows = []
        for player in previous.players:
            rows.append(
                {
                    "player_id": 1000 + player.fpl_player_id,
                    "fpl_player_id": player.fpl_player_id,
                    "player_name": player.player_name,
                    "web_name": player.player_name,
                    "position": player.position,
                    "team_id": player.club_id,
                    "team_name": "T%s" % player.club_id,
                    "team_short_name": "T%s" % player.club_id,
                    "now_cost": player.current_price_units,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "fixture_count": 1,
                    "predicted_points": 4.0 + player.fpl_player_id / 100.0,
                    "expected_minutes_total": 80.0,
                    "blended_appearance_probability": 0.95,
                    "blended_start_probability": 0.90,
                    "prior_fallback_used": False,
                    "official_availability_adjustment_applied": False,
                }
            )
        # One legal incoming player per position, with club IDs that stay legal.
        for offset, position in enumerate(("GKP", "DEF", "MID", "FWD"), start=1):
            pid = 100 + offset
            rows.append(
                {
                    "player_id": 2000 + pid,
                    "fpl_player_id": pid,
                    "player_name": "IN%s" % pid,
                    "web_name": "IN%s" % pid,
                    "position": position,
                    "team_id": 10 + offset,
                    "team_name": "NEW%s" % offset,
                    "team_short_name": "N%s" % offset,
                    "now_cost": 50,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "fixture_count": 1,
                    "predicted_points": 8.0,
                    "expected_minutes_total": 85.0,
                    "blended_appearance_probability": 0.98,
                    "blended_start_probability": 0.95,
                    "prior_fallback_used": False,
                    "official_availability_adjustment_applied": False,
                }
            )
        return rows

    def make_prediction_run(self, root: Path) -> Path:
        run_dir = root / "prediction_run"
        run_dir.mkdir(parents=True)
        pd.DataFrame(self.prediction_rows()).to_csv(
            run_dir / "player_predictions_preview.csv",
            index=False,
        )
        match = pd.DataFrame(
            [
                {
                    "target_season": "2026_27",
                    "target_gw": 3,
                    "fpl_fixture_id": 301,
                    "home_win_probability": 0.5,
                    "draw_probability": 0.3,
                    "away_win_probability": 0.2,
                    "expected_home_goals": 1.5,
                    "expected_away_goals": 1.0,
                    "expected_total_goals": 2.5,
                }
            ]
        )
        score = pd.DataFrame(
            [
                {
                    "target_season": "2026_27",
                    "target_gw": 3,
                    "fpl_fixture_id": 301,
                    "scoreline_home_win_probability": 0.50,
                    "scoreline_draw_probability": 0.30,
                    "scoreline_away_win_probability": 0.20,
                    "top_1_scoreline": "1-0",
                    "top_1_scoreline_probability": 0.15,
                }
            ]
        )
        match.to_csv(run_dir / "match_predictions_preview.csv", index=False)
        score.to_csv(run_dir / "scoreline_preview.csv", index=False)
        (run_dir / "bootstrap_snapshot.json").write_text(
            json.dumps(
                {
                    "events": [
                        {
                            "id": 3,
                            "deadline_time": "2026-09-12T10:00:00Z",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "PASS_PREVIEW",
                    "season": "2026_27",
                    "target_gw": 3,
                    "prediction_mode": "early_season_blend",
                    "created_at": "2026-09-11T12:00:00Z",
                    "preview_only": True,
                    "database_prediction_write": False,
                    "outputs": {
                        "player_predictions_preview": "player_predictions_preview.csv",
                        "match_predictions_preview": "match_predictions_preview.csv",
                        "scoreline_preview": "scoreline_preview.csv",
                        "bootstrap_snapshot": "bootstrap_snapshot.json",
                    },
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    def args(self):
        return argparse.Namespace(
            season="2026_27",
            target_gw=3,
            prior_season="2025_26",
            stabilization_gw=6,
            planning_root=None,
            repo_root=None,
            resume=True,
            dry_run=False,
            final_freeze=False,
            publish_predictions=False,
            skip_live_refresh=True,
            base_url="",
            api_port=8765,
            top_n=10,
            max_transfers=2,
            position_top_n=10,
        )

    def test_live_eligibility_adapter_is_explicit_and_conservative(self) -> None:
        eligible, reason, flags = selection_eligibility_from_live_prediction(
            {
                "status": "d",
                "fixture_count": 1,
                "chance_of_playing_next_round": 75,
            }
        )
        self.assertTrue(eligible)
        self.assertEqual(reason, "eligible_doubtful_with_visible_risk")
        self.assertIn("uncertain_status", flags)

        blocked, blocked_reason, _ = selection_eligibility_from_live_prediction(
            {
                "status": "s",
                "fixture_count": 1,
                "chance_of_playing_next_round": None,
            }
        )
        self.assertFalse(blocked)
        self.assertIn("hard_unavailable_status", blocked_reason)

    def test_horizon_adapter_uses_only_target_gw_and_never_zero_fills_future(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = self.make_prediction_run(root)
            result = build_target_gw_horizon_artifacts(
                player_csv=run_dir / "player_predictions_preview.csv",
                match_csv=run_dir / "match_predictions_preview.csv",
                scoreline_csv=run_dir / "scoreline_preview.csv",
                target_gw=3,
                out_dir=root / "package",
            )
            self.assertTrue(result["player_horizon_csv"].is_file())
            self.assertTrue(result["fixture_horizon_csv"].is_file())
            self.assertTrue(all(row["target_gw"] == 3 for row in result["horizon_rows"]))
            payload = json.loads(result["player_horizon_json"].read_text())
            self.assertEqual(payload["effective_horizon_gameweeks"], [3])
            self.assertFalse(payload["multi_gw_horizon_used"])
            self.assertFalse(payload["missing_future_predictions_zero_filled"])

    def test_current_owned_state_applies_current_price_drift(self) -> None:
        previous = self.previous_state()
        market = []
        for player in previous.players:
            current = player.current_price_units
            if player.fpl_player_id == 1:
                current += 2
            market.append(
                {
                    "fpl_player_id": player.fpl_player_id,
                    "position": player.position,
                    "club_id": player.club_id,
                    "now_cost": current,
                }
            )
        current_owned = build_current_owned_state_for_candidates(
            previous,
            market,
            target_gw=3,
        )
        p1 = next(
            row
            for row in current_owned["players"]
            if row["fpl_player_id"] == 1
        )
        self.assertEqual(
            p1["current_price_units"],
            previous.players[0].current_price_units + 2,
        )
        self.assertEqual(
            p1["selling_price_units"],
            previous.players[0].purchase_price_units + 1,
        )
        self.assertTrue(current_owned["current_valuation_overlay"])
        self.assertEqual(current_owned["change_count"], 1)
        self.assertEqual(
            current_owned["source_owned_state_fingerprint"],
            previous.owned_state_fingerprint,
        )

    def test_legacy_gw2_wrapper_can_be_narrowly_canonicalized(self) -> None:
        previous = self.previous_state()
        payload = {
            "artifact_type": "fpl_model_team_frozen_state",
            "season": "2026_27",
            "gw": 2,
            "final_pre_deadline_snapshot_frozen": True,
            "final_deadline_freeze": True,
            "freeze_id": "legacy_gw02",
            "frozen_at_utc": "2026-08-28T16:00:00Z",
            "squad": [
                {
                    "fpl_player_id": p.fpl_player_id,
                    "web_name": p.player_name,
                    "position": p.position,
                    "team_id": p.club_id,
                    "purchase_price_units": p.purchase_price_units,
                    "current_price_units": p.current_price_units,
                }
                for p in previous.players
            ],
            "lineup": {
                "starting_player_ids": list(previous.selection.starting_xi_player_ids),
                "bench_order": list(previous.selection.bench_order_player_ids),
                "captain": {"fpl_player_id": previous.selection.captain_player_id},
                "vice_captain": {
                    "fpl_player_id": previous.selection.vice_captain_player_id
                },
            },
            "transfer_decision": {
                "bank_after_units": previous.bank_units,
                "free_transfers_next_gameweek": 1,
            },
        }
        # Avoid depending on the full chip registry in this adapter-unit test.
        with patch(
            "ml.pipeline.run_fpl_refresh._legacy_gw2_chip_inventory",
            return_value=ChipInventoryState(as_of_gameweek=2, entries=()),
        ):
            state = canonicalize_legacy_gw2_model_team_state(
                payload,
                expected_season="2026_27",
                expected_gameweek=2,
            )
        self.assertEqual(state.gameweek, 2)
        self.assertEqual(state.state_status, "frozen")
        self.assertEqual(state.free_transfers.available_for_gameweek, 3)
        self.assertEqual(len(state.players), 15)

    def test_normal_weekly_mode_fails_closed_until_approved_producer_exists(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "normal_weekly"):
            resolve_runner_prediction_mode(
                season="2026_27",
                target_gw=6,
                prior_season="2025_26",
                stabilization_gw=6,
            )

    def test_formal_pre_wires_day126_day127_and_never_calls_legacy_decision_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repo_root = root / "repo"
            (repo_root / "backend").mkdir(parents=True)
            planning_root = root / "planning"
            prediction_run = self.make_prediction_run(root)
            previous = self.previous_state()

            fake_ledger = Mock()
            fake_ledger.to_dict.return_value = {
                "season": "2026_27",
                "state_kind": "model_team",
                "gameweek": 3,
                "available_free_transfers": 1,
            }
            candidate_report = {
                "generator_version": "fpl_transfer_candidate_generator_v1",
                "generator_scope": "stateful_owned_squad_transfer_candidates",
                "season": "2026_27",
                "target_gw": 3,
                "state_kind": "model_team",
                "pair_candidates": [],
                "safety": {
                    "writes_database": False,
                    "writes_manager_state": False,
                    "shadow_optimal_consumed": False,
                    "opening_squad_rebuild_used": False,
                    "general_transfer_targets_used_as_state": False,
                    "missing_predictions_zero_filled": False,
                },
            }
            winner = {
                "action": "NO TRANSFER",
                "transfer_count": 0,
                "transfers": [],
                "bank_before_units": previous.bank_units,
                "bank_after_units": previous.bank_units,
                "free_transfers_before": 1,
                "free_transfers_used": 0,
                "charged_transfers": 0,
                "transfer_hit_points": 0,
                "free_transfers_next_gameweek": 2,
                "lineup": {
                    "starting_player_ids": list(
                        previous.selection.starting_xi_player_ids
                    ),
                    "bench_order": list(
                        previous.selection.bench_order_player_ids
                    ),
                    "captain_player_id": previous.selection.captain_player_id,
                    "vice_captain_player_id": previous.selection.vice_captain_player_id,
                },
                "net_gain_vs_no_transfer": 0.0,
            }
            decision = {
                "season": "2026_27",
                "target_gw": 3,
                "winner": winner,
            }

            snapshot_dir = root / "snapshot"
            snapshot_dir.mkdir()
            snapshot_manifest = snapshot_dir / "snapshot_manifest.json"
            snapshot_manifest.write_text("{}", encoding="utf-8")
            snapshot_result = {
                "status": "PASS_CANDIDATE_SNAPSHOT",
                "snapshot_kind": "pre_deadline_candidate",
                "final_pre_deadline_snapshot_frozen": False,
                "run_id": "candidate",
                "snapshot_dir": str(snapshot_dir),
                "manifest_path": str(snapshot_manifest),
                "package_fingerprint": "abc",
                "track_count": 4,
                "tracks": {},
                "model_team_state_path": str(snapshot_dir / "model_team_state.json"),
                "previous_owned_state_fingerprint": previous.owned_state_fingerprint,
                "bank_after_units": previous.bank_units,
                "free_transfers_next_gameweek": 2,
                "team_alex_provided": False,
            }

            args = self.args()
            recorder = StageRecorder()

            with patch(
                "ml.pipeline.run_fpl_refresh.resolve_runner_prediction_mode",
                return_value={
                    "valid": True,
                    "resolved_prediction_mode": "early_season_blend",
                },
            ), patch(
                "ml.pipeline.run_fpl_refresh.run_prediction_stage",
                return_value=prediction_run,
            ), patch(
                "ml.pipeline.run_fpl_refresh.discover_previous_model_team_state",
                return_value=(previous, root / "gw2_state.json", "canonical"),
            ), patch(
                "ml.pipeline.run_fpl_refresh.build_ledger_state",
                return_value=fake_ledger,
            ) as ledger_mock, patch(
                "ml.pipeline.run_fpl_refresh.generate_transfer_candidates",
                return_value=candidate_report,
            ) as candidates_mock, patch(
                "ml.pipeline.run_fpl_refresh.optimize_single_gw_transfers",
                return_value=decision,
            ) as optimizer_mock, patch(
                "ml.pipeline.run_fpl_refresh.export_gameweek_pre_deadline_snapshot",
                return_value=snapshot_result,
            ) as exporter_mock, patch(
                "ml.pipeline.run_fpl_refresh.run_transfer_stage",
                side_effect=AssertionError("legacy transfer path called"),
            ), patch(
                "ml.pipeline.run_fpl_refresh.run_lineup_stage",
                side_effect=AssertionError("legacy lineup path called"),
            ):
                package = run_pre(
                    args=args,
                    repo_root=repo_root,
                    planning_root=planning_root,
                    python_exe="python",
                    env={},
                    recorder=recorder,
                )

            self.assertTrue(package.is_dir())
            self.assertTrue((package / "run_manifest.json").is_file())
            self.assertTrue((package / "transfer_candidates.json").is_file())
            self.assertTrue((package / "transfer_decision.json").is_file())
            self.assertTrue((package / "player_prediction_horizon.csv").is_file())
            self.assertTrue((package / "fixture_prediction_horizon.csv").is_file())

            ledger_mock.assert_called_once()
            candidates_mock.assert_called_once()
            optimizer_mock.assert_called_once()
            exporter_mock.assert_called_once()
            self.assertIn(
                "current_owned_state",
                optimizer_mock.call_args.kwargs,
            )
            self.assertIn(
                "current_model_team_state",
                exporter_mock.call_args.kwargs,
            )
            self.assertFalse(exporter_mock.call_args.kwargs["final_freeze"])

            manifest = json.loads((package / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "PASS_PRE_CANDIDATE")
            self.assertFalse(manifest["final_deadline_freeze"])
            self.assertFalse(manifest["safety"]["legacy_weekly_transfer_optimizer_used"])
            self.assertFalse(manifest["safety"]["legacy_weekly_lineup_preview_used"])
            stage_names = [row["stage"] for row in manifest["stage_results"]]
            for required in (
                "prediction_mode",
                "player_model",
                "match_model",
                "prediction_horizon",
                "owned_squad_state",
                "free_transfer_ledger",
                "transfer_candidates",
                "transfer_decision",
                "lineup_selection",
                "pre_deadline_candidate",
            ):
                self.assertIn(required, stage_names)

    def test_pre_export_is_candidate_only_even_when_auto_support_exists(self) -> None:
        plan = build_dag_plan(
            phase="pre",
            season="2026_27",
            target_gw=3,
            final_freeze=False,
        )
        names = [row["name"] for row in plan["stages"]]
        self.assertIn("pre_deadline_candidate", names)
        self.assertNotIn("final_freeze_export", names)


class UnifiedPostPipelineIntegrationTests(unittest.TestCase):
    def test_post_runs_idempotent_ingest_then_finality_then_frozen_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repo_root = root / "repo"
            planning_root = root / "planning"
            (repo_root / "backend").mkdir(parents=True)
            planning_root.mkdir()
            actual_manifest = root / "gw2_actuals_manifest_FINAL.json"
            actual_manifest.write_text("{}", encoding="utf-8")
            evaluation_dir = root / "evaluation"
            evaluation_dir.mkdir()

            events = []
            fake_proc = Mock()

            def refresh_side_effect(*args, **kwargs):
                events.append("ingest")

            def capture_side_effect(**kwargs):
                events.append("finality")
                return actual_manifest

            def evaluation_side_effect(**kwargs):
                events.append("evaluation")
                self.assertEqual(kwargs["actual_manifest_path"], actual_manifest)
                return evaluation_dir

            recorder = StageRecorder()
            with patch(
                "ml.pipeline.run_fpl_refresh.discover_final_evaluation",
                return_value=None,
            ), patch(
                "ml.pipeline.run_fpl_refresh.start_private_api",
                return_value=(fake_proc, "http://127.0.0.1:8765"),
            ), patch(
                "ml.pipeline.run_fpl_refresh.refresh_live_data",
                side_effect=refresh_side_effect,
            ) as refresh_mock, patch(
                "ml.pipeline.run_fpl_refresh.capture_or_reuse_final_actuals",
                side_effect=capture_side_effect,
            ) as capture_mock, patch(
                "ml.pipeline.run_fpl_refresh.run_gameweek_evaluation",
                side_effect=evaluation_side_effect,
            ) as evaluation_mock:
                result = run_post_pipeline(
                    repo_root=repo_root,
                    planning_root=planning_root,
                    season="2026_27",
                    gw=2,
                    python_exe="python",
                    env={},
                    recorder=recorder,
                    resume=False,
                    skip_live_refresh=False,
                    base_url="",
                    api_port=8765,
                )

            self.assertEqual(result, evaluation_dir)
            self.assertEqual(events, ["ingest", "finality", "evaluation"])
            refresh_mock.assert_called_once()
            self.assertFalse(capture_mock.call_args.kwargs["reuse_only"])
            evaluation_mock.assert_called_once()
            fake_proc.terminate.assert_called_once()
            stage_names = [row["stage"] for row in recorder.rows]
            self.assertIn("actuals_finality", stage_names)
            self.assertIn("post_evaluation", stage_names)
            self.assertLess(
                stage_names.index("actuals_finality"),
                stage_names.index("post_evaluation"),
            )

    def test_post_skip_live_refresh_is_reuse_only_and_never_fetches_actuals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repo_root = root / "repo"
            planning_root = root / "planning"
            (repo_root / "backend").mkdir(parents=True)
            planning_root.mkdir()
            actual_manifest = root / "existing_final_actuals.json"
            actual_manifest.write_text("{}", encoding="utf-8")
            evaluation_dir = root / "evaluation"
            evaluation_dir.mkdir()

            recorder = StageRecorder()
            with patch(
                "ml.pipeline.run_fpl_refresh.discover_final_evaluation",
                return_value=None,
            ), patch(
                "ml.pipeline.run_fpl_refresh.start_private_api",
                side_effect=AssertionError("POST reuse-only path started an API"),
            ), patch(
                "ml.pipeline.run_fpl_refresh.refresh_live_data",
                side_effect=AssertionError("POST reuse-only path ingested live data"),
            ), patch(
                "ml.pipeline.run_fpl_refresh.capture_or_reuse_final_actuals",
                return_value=actual_manifest,
            ) as capture_mock, patch(
                "ml.pipeline.run_fpl_refresh.run_gameweek_evaluation",
                return_value=evaluation_dir,
            ):
                result = run_post_pipeline(
                    repo_root=repo_root,
                    planning_root=planning_root,
                    season="2026_27",
                    gw=2,
                    python_exe="python",
                    env={},
                    recorder=recorder,
                    resume=False,
                    skip_live_refresh=True,
                    base_url="",
                    api_port=8765,
                )

            self.assertEqual(result, evaluation_dir)
            self.assertTrue(capture_mock.call_args.kwargs["reuse_only"])
            self.assertFalse(capture_mock.call_args.kwargs["reuse_existing"])


if __name__ == "__main__":
    unittest.main()
