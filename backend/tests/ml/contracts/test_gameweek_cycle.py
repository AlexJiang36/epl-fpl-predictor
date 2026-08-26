from __future__ import annotations

import unittest

from ml.contracts.gameweek_cycle import (
    ActualsStatus,
    ArtifactTrack,
    GameweekCycleError,
    GameweekCycleState,
    GameweekPhase,
    LifecycleArtifactRef,
    build_execution_run_id,
    build_phase_id,
    validate_actuals_window,
    validate_freeze_window,
    validate_output_kinds,
    validate_phase_artifacts,
    validate_pre_window,
    validate_transition,
)


class GameweekCycleContractTests(unittest.TestCase):
    def make_pre(self) -> GameweekCycleState:
        return GameweekCycleState(
            season="2026_27",
            target_gw=2,
            phase=GameweekPhase.PRE.value,
        )

    def make_freeze(self, fingerprint: str = "freeze-sha") -> GameweekCycleState:
        return GameweekCycleState(
            season="2026_27",
            target_gw=2,
            phase=GameweekPhase.FREEZE.value,
            freeze_run_id="GW02-FREEZE-001",
            freeze_fingerprint=fingerprint,
        )

    def make_post(self, actuals_status: str = "provisional") -> GameweekCycleState:
        return GameweekCycleState(
            season="2026_27",
            target_gw=2,
            phase=GameweekPhase.POST.value,
            target_actuals_status=actuals_status,
            freeze_run_id="GW02-FREEZE-001",
            freeze_fingerprint="freeze-sha",
            post_eval_run_id="GW02-POST-EVAL-001",
        )

    def test_phase_ids_follow_contract(self):
        self.assertEqual(build_phase_id(2, "pre"), "GW02-PRE")
        self.assertEqual(build_phase_id(2, "freeze"), "GW02-FREEZE")
        self.assertEqual(build_phase_id(1, "post"), "GW01-POST-EVAL")
        self.assertEqual(
            build_execution_run_id(2, "pre", "20260825T120000Z"),
            "GW02-PRE-20260825T120000Z",
        )

    def test_pre_can_rerun_and_advance_to_freeze(self):
        pre = self.make_pre()
        validate_transition(pre, self.make_pre(), deadline_passed=False, target_results_observed=False)
        validate_transition(pre, self.make_freeze(), deadline_passed=False, target_results_observed=False)

    def test_pre_rerun_is_blocked_after_deadline_or_results(self):
        pre = self.make_pre()
        with self.assertRaises(GameweekCycleError):
            validate_transition(
                pre,
                self.make_pre(),
                deadline_passed=True,
                target_results_observed=False,
            )
        with self.assertRaises(GameweekCycleError):
            validate_transition(
                pre,
                self.make_pre(),
                deadline_passed=False,
                target_results_observed=True,
            )

    def test_post_entry_requires_deadline_and_results(self):
        freeze = self.make_freeze()
        post = self.make_post()
        with self.assertRaises(GameweekCycleError):
            validate_transition(
                freeze,
                post,
                deadline_passed=False,
                target_results_observed=True,
            )
        with self.assertRaises(GameweekCycleError):
            validate_transition(
                freeze,
                post,
                deadline_passed=True,
                target_results_observed=False,
            )
        validate_transition(
            freeze,
            post,
            deadline_passed=True,
            target_results_observed=True,
        )

    def test_phase_output_kinds_are_explicitly_validated(self):
        self.assertEqual(
            validate_output_kinds(
                "pre",
                ["player_predictions_pre", "phase_manifest", "pre_publish_receipt"],
            ),
            ("player_predictions_pre", "phase_manifest", "pre_publish_receipt"),
        )
        with self.assertRaises(GameweekCycleError):
            validate_output_kinds("pre", ["player_model_evaluation"])

    def test_pre_cannot_jump_directly_to_post(self):
        with self.assertRaises(GameweekCycleError):
            validate_transition(self.make_pre(), self.make_post(), deadline_passed=True, target_results_observed=True)

    def test_post_cannot_return_to_pre(self):
        with self.assertRaises(GameweekCycleError):
            validate_transition(self.make_post(), self.make_pre(), deadline_passed=True, target_results_observed=True)

    def test_freeze_rerun_requires_same_immutable_identity(self):
        freeze = self.make_freeze()
        validate_transition(freeze, self.make_freeze(), deadline_passed=True, target_results_observed=True)
        with self.assertRaises(GameweekCycleError):
            validate_transition(freeze, self.make_freeze(fingerprint="changed"), deadline_passed=True, target_results_observed=True)

    def test_post_rerun_may_progress_provisional_to_final_but_not_backward(self):
        provisional = self.make_post(ActualsStatus.PROVISIONAL.value)
        final = self.make_post(ActualsStatus.FINAL.value)
        validate_transition(provisional, final, deadline_passed=True, target_results_observed=True)
        with self.assertRaises(GameweekCycleError):
            validate_transition(final, provisional, deadline_passed=True, target_results_observed=True)

    def test_target_actuals_are_blocked_before_post(self):
        with self.assertRaises(GameweekCycleError):
            validate_actuals_window("pre", 2, [1, 2])
        with self.assertRaises(GameweekCycleError):
            validate_actuals_window("freeze", 2, [1, 2])
        self.assertEqual(validate_actuals_window("post", 2, [1, 2]), (1, 2))

    def test_post_requires_target_actuals_and_blocks_future_actuals(self):
        with self.assertRaises(GameweekCycleError):
            validate_actuals_window("post", 2, [1])
        with self.assertRaises(GameweekCycleError):
            validate_actuals_window("post", 2, [1, 2, 3])

    def test_freeze_cannot_be_created_late_or_after_results(self):
        validate_freeze_window(False, False)
        with self.assertRaises(GameweekCycleError):
            validate_freeze_window(True, False)
        with self.assertRaises(GameweekCycleError):
            validate_freeze_window(False, True)

    def test_freeze_requires_all_four_immutable_hashed_tracks(self):
        state = self.make_freeze()
        kinds = {
            ArtifactTrack.PLAYER_MODEL.value: "player_predictions_final",
            ArtifactTrack.MATCH_MODEL.value: "match_predictions_final",
            ArtifactTrack.MODEL_TEAM.value: "model_team_final",
            ArtifactTrack.TEAM_ALEX.value: "team_alex_final",
        }
        artifacts = [
            LifecycleArtifactRef(
                track=track,
                phase="freeze",
                artifact_kind=kind,
                cycle_phase_id="GW02-FREEZE",
                artifact_run_id="%s-run" % track,
                artifact_path="/tmp/%s.json" % track,
                sha256="abc123",
                immutable=True,
            )
            for track, kind in kinds.items()
        ]
        found = validate_phase_artifacts(state, artifacts)
        self.assertEqual(set(found), set(kinds))

        bad = list(artifacts)
        bad[0] = LifecycleArtifactRef(
            track=ArtifactTrack.PLAYER_MODEL.value,
            phase="freeze",
            artifact_kind="player_predictions_final",
            cycle_phase_id="GW02-FREEZE",
            artifact_run_id="player-run",
            artifact_path="/tmp/player.json",
            sha256="abc123",
            immutable=False,
        )
        with self.assertRaises(GameweekCycleError):
            validate_phase_artifacts(state, bad)

    def test_post_artifacts_must_reference_exact_freeze_lineage(self):
        state = self.make_post()
        kinds = {
            ArtifactTrack.PLAYER_MODEL.value: "player_model_evaluation",
            ArtifactTrack.MATCH_MODEL.value: "match_model_evaluation",
            ArtifactTrack.MODEL_TEAM.value: "model_team_evaluation",
            ArtifactTrack.TEAM_ALEX.value: "team_alex_evaluation",
        }
        artifacts = [
            LifecycleArtifactRef(
                track=track,
                phase="post",
                artifact_kind=kind,
                cycle_phase_id="GW02-POST-EVAL",
                artifact_run_id="%s-eval" % track,
                artifact_path="/tmp/%s-eval.json" % track,
                source_freeze_run_id="GW02-FREEZE-001",
                source_freeze_fingerprint="freeze-sha",
            )
            for track, kind in kinds.items()
        ]
        validate_phase_artifacts(state, artifacts)

        wrong = list(artifacts)
        wrong[0] = LifecycleArtifactRef(
            track=ArtifactTrack.PLAYER_MODEL.value,
            phase="post",
            artifact_kind="player_model_evaluation",
            cycle_phase_id="GW02-POST-EVAL",
            artifact_run_id="player-eval",
            artifact_path="/tmp/player-eval.json",
            source_freeze_run_id="other-freeze",
            source_freeze_fingerprint="freeze-sha",
        )
        with self.assertRaises(GameweekCycleError):
            validate_phase_artifacts(state, wrong)


if __name__ == "__main__":
    unittest.main()
