from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from app.rules.squad import load_squad_transfer_rules
from ml.decision.optimize_weekly_transfers import (
    best_lineup_for_predicted_points,
    fpl_selling_price_units,
    optimize_roll_vs_one_transfer,
    recommended_state_preview,
)


class WeeklyTransferOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_squad_transfer_rules("2026_27")

    def test_selling_price_uses_half_profit_rounded_down(self):
        self.assertEqual(fpl_selling_price_units(50, 50), 50)
        self.assertEqual(fpl_selling_price_units(50, 49), 49)
        self.assertEqual(fpl_selling_price_units(50, 51), 50)
        self.assertEqual(fpl_selling_price_units(50, 52), 51)
        self.assertEqual(fpl_selling_price_units(50, 53), 51)

    def test_best_lineup_respects_legal_formation(self):
        rows = []
        pid = 1
        for position, count in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            for index in range(count):
                rows.append(
                    {
                        "fpl_player_id": pid,
                        "web_name": "%s%s" % (position, index),
                        "position": position,
                        "predicted_points": float(20 - pid),
                    }
                )
                pid += 1
        result = best_lineup_for_predicted_points(pd.DataFrame(rows))
        self.assertTrue(result["formation"].startswith("1-"))
        self.assertEqual(len(result["starting_player_ids"]), 11)
        self.assertIn(result["captain_player_id"], result["starting_player_ids"])

    def _base_market_and_squad(self):
        rows = []
        pid = 1
        team = 1
        for position, count in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            for _ in range(count):
                rows.append(
                    {
                        "player_id": pid,
                        "fpl_player_id": pid,
                        "web_name": "P%s" % pid,
                        "position": position,
                        "team_id": team,
                        "team_name": "T%s" % team,
                        "team_short_name": "T%s" % team,
                        "now_cost": 50,
                        "status": "a",
                        "predicted_points": 3.0,
                    }
                )
                pid += 1
                team += 1
                if team > 10:
                    team = 1
        squad = pd.DataFrame(rows)
        market = squad.copy()
        extra = {
            "player_id": 99,
            "fpl_player_id": 99,
            "web_name": "UpgradeMID",
            "position": "MID",
            "team_id": 20,
            "team_name": "Upgrade",
            "team_short_name": "UPG",
            "now_cost": 50,
            "status": "a",
            "predicted_points": 8.0,
        }
        market = pd.concat([market, pd.DataFrame([extra])], ignore_index=True)
        purchase = {int(v): 50 for v in squad["fpl_player_id"]}
        return market, squad, purchase

    def test_optimizer_includes_roll_and_all_legal_single_moves(self):
        market, squad, purchase = self._base_market_and_squad()
        result = optimize_roll_vs_one_transfer(
            rules=self.rules,
            current_squad=squad,
            market=market,
            purchase_price_by_id=purchase,
            bank_units=0,
            available_free_transfers=1,
            target_gw=2,
        )
        ranked = result["ranked_options"]
        self.assertEqual(int((ranked["action"] == "ROLL").sum()), 1)
        self.assertGreater(result["one_transfer_option_count"], 0)

    def test_clear_upgrade_beats_roll(self):
        market, squad, purchase = self._base_market_and_squad()
        result = optimize_roll_vs_one_transfer(
            rules=self.rules,
            current_squad=squad,
            market=market,
            purchase_price_by_id=purchase,
            bank_units=0,
            available_free_transfers=1,
            target_gw=2,
        )
        self.assertEqual(result["winner"]["action"], "TRANSFER")
        self.assertEqual(int(result["winner"]["in_fpl_player_id"]), 99)
        self.assertGreater(float(result["winner"]["net_gain_vs_roll"]), 0.0)

    def test_roll_keeps_extra_free_transfer_for_next_gw(self):
        market, squad, purchase = self._base_market_and_squad()
        result = optimize_roll_vs_one_transfer(
            rules=self.rules,
            current_squad=squad,
            market=market,
            purchase_price_by_id=purchase,
            bank_units=0,
            available_free_transfers=1,
            target_gw=2,
        )
        self.assertEqual(int(result["roll"]["free_transfers_next_gameweek"]), 2)
        transfer = result["ranked_options"][
            result["ranked_options"]["action"] == "TRANSFER"
        ].iloc[0]
        self.assertEqual(int(transfer["free_transfers_next_gameweek"]), 1)

    def test_unavailable_incoming_is_excluded(self):
        market, squad, purchase = self._base_market_and_squad()
        market.loc[market["fpl_player_id"] == 99, "status"] = "u"
        result = optimize_roll_vs_one_transfer(
            rules=self.rules,
            current_squad=squad,
            market=market,
            purchase_price_by_id=purchase,
            bank_units=0,
            available_free_transfers=1,
            target_gw=2,
        )
        ids = set(
            pd.to_numeric(
                result["ranked_options"]["in_fpl_player_id"], errors="coerce"
            ).dropna().astype(int)
        )
        self.assertNotIn(99, ids)

    def test_recommended_state_preview_is_reusable_15_player_state(self):
        market, squad, purchase = self._base_market_and_squad()
        result = optimize_roll_vs_one_transfer(
            rules=self.rules,
            current_squad=squad,
            market=market,
            purchase_price_by_id=purchase,
            bank_units=0,
            available_free_transfers=1,
            target_gw=2,
        )
        state = recommended_state_preview(
            winner=result["winner"],
            current_squad=squad,
            market=market,
            purchase_prices=purchase,
            target_gw=2,
        )
        self.assertEqual(len(state["squad"]), 15)
        self.assertEqual(len({p["fpl_player_id"] for p in state["squad"]}), 15)
        self.assertEqual(state["next_gameweek"], 3)
        self.assertFalse(state["finalized"])


if __name__ == "__main__":
    unittest.main()
