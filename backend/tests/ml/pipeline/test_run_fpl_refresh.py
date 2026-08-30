from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
