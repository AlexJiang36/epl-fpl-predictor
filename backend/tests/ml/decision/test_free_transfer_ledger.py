from __future__ import annotations

import copy
import unittest
from pathlib import Path

from app.rules.squad import SquadTransferRules, load_squad_transfer_rules
from ml.decision.free_transfer_ledger import (
    FreeTransferLedgerError,
    TransferFinancialEffect,
    build_ledger_state,
    run_ledger_scenario,
    transition_free_transfer_ledger,
    transfer_policy_summary,
)


SEASON = "2026_27"


class FreeTransferLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_squad_transfer_rules(SEASON)
        cls.summary = transfer_policy_summary(cls.rules)
        cls.cap = int(cls.summary["maximum_stored_free_transfers"])
        cls.accrual = int(cls.summary["free_transfers_accrued_per_gameweek"])
        cls.hit_cost = int(
            cls.summary["hit_cost_points_per_additional_transfer"]
        )

    def state(
        self,
        *,
        kind: str = "model_team",
        gw: int = 2,
        available: int = 1,
    ):
        return build_ledger_state(
            season=SEASON,
            state_kind=kind,
            gameweek=gw,
            available_free_transfers=available,
            rules=self.rules,
        )

    def test_target_season_policy_is_consumed(self) -> None:
        state = self.state(available=min(1, self.cap))
        transition = transition_free_transfer_ledger(
            state,
            transfer_count=0,
        )
        self.assertEqual(transition.season, SEASON)
        self.assertEqual(transition.storage_cap, self.cap)
        self.assertEqual(transition.weekly_accrual_applied, self.accrual)

    def test_roll_is_first_class_and_accrues_to_cap(self) -> None:
        state = self.state(available=self.cap)
        transition = transition_free_transfer_ledger(
            state,
            transfer_count=0,
            rules=self.rules,
        )
        self.assertEqual(transition.action, "ROLL")
        self.assertEqual(transition.transfer_count, 0)
        self.assertEqual(transition.free_transfers_used, 0)
        self.assertEqual(transition.charged_transfers, 0)
        self.assertEqual(transition.hit_points, 0)
        self.assertEqual(
            transition.available_free_transfers_next_gameweek,
            self.cap,
        )

    def test_zero_available_transfer_is_charged(self) -> None:
        state = self.state(available=0)
        transition = transition_free_transfer_ledger(
            state,
            transfer_count=1,
            rules=self.rules,
        )
        self.assertEqual(transition.free_transfers_used, 0)
        self.assertEqual(transition.charged_transfers, 1)
        self.assertEqual(transition.hit_points, self.hit_cost)
        self.assertEqual(
            transition.next_state.available_free_transfers,
            min(self.cap, self.accrual),
        )

    def test_multiple_transfers_use_saved_free_transfers_then_hits(self) -> None:
        available = min(2, self.cap)
        state = self.state(available=available)
        transfer_count = available + 2
        transition = transition_free_transfer_ledger(
            state,
            transfer_count=transfer_count,
            rules=self.rules,
        )
        self.assertEqual(transition.free_transfers_used, available)
        self.assertEqual(transition.charged_transfers, 2)
        self.assertEqual(transition.hit_points, 2 * self.hit_cost)
        self.assertEqual(
            transition.next_state.cumulative_transfer_count,
            transfer_count,
        )

    def test_wildcard_uses_target_policy_chip_behavior(self) -> None:
        available = min(2, self.cap)
        state = self.state(available=available)
        policy = self.rules.transfers["chip_behavior"]["wildcard"]
        transition = transition_free_transfer_ledger(
            state,
            transfer_count=7,
            chip="wildcard",
            rules=self.rules,
        )
        expected_base = (
            available if bool(policy["preserve_saved_free_transfers"]) else 0
        )
        expected_next = min(
            self.cap,
            expected_base + int(policy["weekly_accrual_after_chip"]),
        )
        self.assertEqual(transition.action, "WILDCARD")
        self.assertTrue(transition.unlimited_free_transfers)
        self.assertEqual(transition.charged_transfers, 0)
        self.assertEqual(transition.hit_points, 0)
        self.assertEqual(
            transition.available_free_transfers_next_gameweek,
            expected_next,
        )

    def test_free_hit_uses_policy_and_allows_only_temporary_financial_effect(self) -> None:
        available = min(2, self.cap)
        state = self.state(kind="team_alex", available=available)
        policy = self.rules.transfers["chip_behavior"]["free_hit"]
        effect = TransferFinancialEffect(
            bank_before_units=60,
            bank_after_units=0,
            persistent=False,
            note="Temporary Free Hit active-window bank.",
        )
        transition = transition_free_transfer_ledger(
            state,
            transfer_count=12,
            chip="free_hit",
            financial_effect=effect,
            rules=self.rules,
        )
        expected_base = (
            available if bool(policy["preserve_saved_free_transfers"]) else 0
        )
        expected_next = min(
            self.cap,
            expected_base + int(policy["weekly_accrual_after_chip"]),
        )
        self.assertEqual(transition.action, "FREE_HIT")
        self.assertEqual(transition.state_kind, "team_alex")
        self.assertTrue(transition.unlimited_free_transfers)
        self.assertEqual(transition.hit_points, 0)
        self.assertEqual(
            transition.available_free_transfers_next_gameweek,
            expected_next,
        )
        self.assertFalse(transition.financial_effect.persistent)

    def test_free_hit_rejects_persistent_financial_effect(self) -> None:
        state = self.state(kind="team_alex")
        with self.assertRaises(FreeTransferLedgerError):
            transition_free_transfer_ledger(
                state,
                transfer_count=10,
                chip="free_hit",
                financial_effect=TransferFinancialEffect(
                    bank_before_units=60,
                    bank_after_units=0,
                    persistent=True,
                ),
                rules=self.rules,
            )

    def test_bench_boost_is_transfer_neutral(self) -> None:
        available = min(1, self.cap)
        state = self.state(available=available)
        count = available + 1
        transition = transition_free_transfer_ledger(
            state,
            transfer_count=count,
            chip="bench_boost",
            rules=self.rules,
        )
        self.assertEqual(transition.chip, "bench_boost")
        self.assertEqual(transition.action, "TRANSFER")
        self.assertFalse(transition.unlimited_free_transfers)
        self.assertEqual(transition.free_transfers_used, available)
        self.assertEqual(transition.charged_transfers, 1)
        self.assertEqual(transition.hit_points, self.hit_cost)

    def test_model_team_and_team_alex_ledgers_are_independent(self) -> None:
        model = self.state(kind="model_team", available=min(1, self.cap))
        alex = self.state(kind="team_alex", available=min(1, self.cap))
        model_transition = transition_free_transfer_ledger(
            model,
            transfer_count=1,
            rules=self.rules,
        )
        self.assertEqual(model_transition.state_kind, "model_team")
        self.assertEqual(alex.state_kind, "team_alex")
        self.assertEqual(alex.transition_count, 0)
        self.assertEqual(alex.available_free_transfers, min(1, self.cap))

    def test_financial_effect_is_separate_from_transfer_count_state(self) -> None:
        state = self.state(available=min(1, self.cap))
        first = transition_free_transfer_ledger(
            state,
            transfer_count=1,
            financial_effect=TransferFinancialEffect(
                bank_before_units=0,
                bank_after_units=10,
                sales_units=80,
                purchases_units=70,
                persistent=True,
            ),
            rules=self.rules,
        )
        second = transition_free_transfer_ledger(
            state,
            transfer_count=1,
            financial_effect=TransferFinancialEffect(
                bank_before_units=0,
                bank_after_units=20,
                sales_units=90,
                purchases_units=70,
                persistent=True,
            ),
            rules=self.rules,
        )
        self.assertEqual(first.next_state, second.next_state)
        self.assertNotEqual(
            first.financial_effect.bank_after_units,
            second.financial_effect.bank_after_units,
        )
        self.assertFalse(first.price_bank_effect_applied_to_ledger)

    def test_multi_gameweek_roll_then_transfer_scenario(self) -> None:
        initial = self.state(available=min(1, self.cap))
        result = run_ledger_scenario(
            initial,
            [
                {"transfer_count": 0},
                {"transfer_count": 2},
            ],
            rules=self.rules,
        )
        self.assertEqual(result["transition_count"], 2)
        self.assertEqual(result["final_state"]["gameweek"], 4)
        self.assertEqual(result["final_state"]["transition_count"], 2)
        self.assertFalse(result["writes_manager_state"])

    def test_special_event_top_up_is_preserved_by_ledger(self) -> None:
        data = copy.deepcopy(self.rules.data)
        target = min(self.cap, max(self.accrual, min(3, self.cap)))
        data["transfers"]["special_events"] = [
            {
                "event_id": "test_top_up",
                "after_gameweek": 2,
                "applies_for_gameweek": 3,
                "operation": "top_up_to",
                "target_free_transfers": target,
                "carry_forward": True,
            }
        ]
        synthetic = SquadTransferRules(
            effective_season=self.rules.effective_season,
            rules_version=self.rules.rules_version,
            schema_version=self.rules.schema_version,
            path=Path("/tmp/test_rules.json"),
            sha256="test",
            data=data,
        )
        state = build_ledger_state(
            season=SEASON,
            state_kind="model_team",
            gameweek=2,
            available_free_transfers=0,
            rules=synthetic,
        )
        transition = transition_free_transfer_ledger(
            state,
            transfer_count=0,
            rules=synthetic,
        )
        self.assertEqual(transition.special_event_applied, "test_top_up")
        self.assertEqual(
            transition.available_free_transfers_next_gameweek,
            target,
        )

    def test_live_gw2_model_team_shape_produces_policy_driven_gw3_ft(self) -> None:
        available = min(1, self.cap)
        state = self.state(kind="model_team", gw=2, available=available)
        transition = transition_free_transfer_ledger(
            state,
            transfer_count=1,
            rules=self.rules,
        )
        expected_remaining = max(0, available - 1)
        expected_next = min(self.cap, expected_remaining + self.accrual)
        self.assertEqual(
            transition.next_state.available_free_transfers,
            expected_next,
        )

    def test_live_gw2_team_alex_free_hit_keeps_ledger_independent(self) -> None:
        available = min(1, self.cap)
        model = self.state(kind="model_team", gw=2, available=available)
        alex = self.state(kind="team_alex", gw=2, available=available)
        alex_transition = transition_free_transfer_ledger(
            alex,
            transfer_count=12,
            chip="free_hit",
            rules=self.rules,
        )
        self.assertEqual(alex_transition.state_kind, "team_alex")
        self.assertEqual(model.gameweek, 2)
        self.assertEqual(model.transition_count, 0)

    def test_wrong_gameweek_sequence_is_rejected(self) -> None:
        state = self.state(gw=2)
        with self.assertRaises(FreeTransferLedgerError):
            transition_free_transfer_ledger(
                state,
                transfer_count=0,
                completed_gameweek=3,
                rules=self.rules,
            )

    def test_state_above_policy_cap_is_rejected(self) -> None:
        with self.assertRaises(FreeTransferLedgerError):
            build_ledger_state(
                season=SEASON,
                state_kind="model_team",
                gameweek=2,
                available_free_transfers=self.cap + 1,
                rules=self.rules,
            )


if __name__ == "__main__":
    unittest.main()
