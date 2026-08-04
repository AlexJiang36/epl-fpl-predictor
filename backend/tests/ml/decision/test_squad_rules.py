from __future__ import annotations

import copy
import unittest

from ml.decision.squad_rules import ENGINE_VERSION, SquadLegalityEngine


TARGET_SEASON = "2026_27"


class SquadLegalityEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = SquadLegalityEngine.from_season(TARGET_SEASON)
        cls.rules = cls.engine.rules

    def base_squad(self):
        players = copy.deepcopy(
            self.rules.data["deterministic_examples"]["base_squad"]
        )
        for player in players:
            player.update(
                {
                    "selection_eligible": True,
                    "eligibility_reason": "eligible_preview_status",
                    "status_cutoff_valid": True,
                    "status_hard_guardrail_applied": False,
                }
            )
        return players

    def lineup_case(self, name):
        cases = self.rules.data["deterministic_examples"]["lineup_cases"]
        return copy.deepcopy(
            next(case for case in cases if case["name"] == name)
        )

    def correct_bank(self, players):
        return self.rules.initial_budget_units - sum(
            int(player["price_units"]) for player in players
        )

    def validate(self, players=None, case_name="legal_1_4_4_2_lineup", bank=None):
        squad = self.base_squad() if players is None else players
        case = self.lineup_case(case_name)
        declared_bank = self.correct_bank(squad) if bank is None else bank
        return self.engine.validate_plan(
            squad,
            starting_player_ids=case["starting_player_ids"],
            bench_order=case["bench_order"],
            captain_player_id=case["captain_player_id"],
            vice_captain_player_id=case["vice_captain_player_id"],
            declared_bank_units=declared_bank,
        )

    def assert_issue(self, result, code):
        self.assertIn(code, result["issue_codes"])
        issue = next(
            issue for issue in result["issues"] if issue["code"] == code
        )
        self.assertTrue(issue["scope"])
        self.assertTrue(issue["message"])
        self.assertTrue(issue["constraint"])
        return issue

    def test_loads_versioned_2026_27_policy(self):
        metadata = self.engine.rules_metadata()
        self.assertEqual(metadata["effective_season"], TARGET_SEASON)
        self.assertEqual(
            metadata["rules_version"],
            "fpl_2026_27_squad_transfer_v1",
        )
        self.assertEqual(ENGINE_VERSION, "day100a_v1")
        self.assertTrue(metadata["rules_sha256"])

    def test_legal_non_3_4_3_plan_passes(self):
        result = self.validate(case_name="legal_1_4_4_2_lineup")

        self.assertTrue(result["valid"])
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["lineup"]["formation"], "4-4-2")
        self.assertEqual(result["squad"]["computed_bank_units"], 10)
        self.assertTrue(result["component_validity"]["bench"])
        self.assertTrue(result["component_validity"]["captaincy"])

    def test_second_legal_formation_passes(self):
        result = self.validate(case_name="legal_1_3_5_2_lineup")

        self.assertTrue(result["valid"])
        self.assertEqual(result["lineup"]["formation"], "3-5-2")

    def test_invalid_squad_size_returns_structured_issue(self):
        players = self.base_squad()[:-1]
        result = self.validate(players=players)

        issue = self.assert_issue(result, "squad_size_invalid")
        self.assertEqual(issue["expected"], int(self.rules.squad["size"]))
        self.assertEqual(issue["actual"], len(players))

    def test_invalid_position_quota_returns_structured_issue(self):
        players = self.base_squad()
        next(player for player in players if player["player_id"] == "F3")[
            "position"
        ] = "MID"

        result = self.validate(players=players)
        issue = self.assert_issue(result, "position_quotas_invalid")

        self.assertEqual(issue["expected"], self.rules.position_quotas)
        self.assertNotEqual(issue["expected"], issue["actual"])

    def test_budget_and_bank_are_validated_independently(self):
        players = self.base_squad()
        next(player for player in players if player["player_id"] == "F3")[
            "price_units"
        ] += 20

        result = self.validate(players=players, bank=10)

        self.assert_issue(result, "budget_exceeded")
        bank_issue = self.assert_issue(result, "declared_bank_mismatch")
        self.assertEqual(bank_issue["expected"], -10)
        self.assertEqual(bank_issue["actual"], 10)

    def test_declared_bank_mismatch_fails(self):
        result = self.validate(bank=11)

        issue = self.assert_issue(result, "declared_bank_mismatch")
        self.assertEqual(issue["expected"], 10)
        self.assertEqual(issue["actual"], 11)

    def test_club_limit_returns_violating_players(self):
        players = self.base_squad()
        next(player for player in players if player["player_id"] == "D4")[
            "club_id"
        ] = "AAA"

        result = self.validate(players=players)
        issue = self.assert_issue(result, "club_limit_exceeded")

        self.assertGreater(len(issue["player_ids"]), 3)
        self.assertIn("AAA", issue["actual"])

    def test_duplicate_player_id_fails(self):
        players = self.base_squad()
        next(player for player in players if player["player_id"] == "F3")[
            "player_id"
        ] = "F2"

        result = self.validate(players=players)
        issue = self.assert_issue(result, "duplicate_player_ids")

        self.assertEqual(issue["player_ids"], ["F2"])

    def test_ineligible_player_fails_closed(self):
        players = self.base_squad()
        player = next(
            player for player in players if player["player_id"] == "M5"
        )
        player["selection_eligible"] = False
        player["eligibility_reason"] = "status_hard_guardrail_applied"

        result = self.validate(players=players)
        issue = self.assert_issue(result, "player_ineligible")

        self.assertEqual(issue["player_ids"], ["M5"])
        self.assertEqual(
            issue["details"]["eligibility_reason"],
            "status_hard_guardrail_applied",
        )
        self.assertFalse(result["component_validity"]["eligibility"])

    def test_missing_eligibility_fails_closed(self):
        players = self.base_squad()
        player = next(
            player for player in players if player["player_id"] == "M5"
        )
        del player["selection_eligible"]

        result = self.validate(players=players)
        self.assert_issue(result, "player_eligibility_missing")

    def test_eligible_player_cannot_have_invalid_status_cutoff(self):
        players = self.base_squad()
        player = next(
            player for player in players if player["player_id"] == "M5"
        )
        player["status_cutoff_valid"] = False

        result = self.validate(players=players)
        self.assert_issue(result, "eligible_player_status_cutoff_invalid")

    def test_formation_below_defender_minimum_fails(self):
        players = self.base_squad()
        result = self.engine.validate_plan(
            players,
            starting_player_ids=[
                "G1",
                "D1",
                "D2",
                "M1",
                "M2",
                "M3",
                "M4",
                "M5",
                "F1",
                "F2",
                "F3",
            ],
            bench_order=["G2", "D3", "D4", "D5"],
            captain_player_id="M1",
            vice_captain_player_id="F1",
            declared_bank_units=self.correct_bank(players),
        )

        issue = self.assert_issue(result, "formation_invalid_DEF")
        self.assertEqual(issue["expected"]["min"], 3)
        self.assertEqual(issue["actual"], 2)

    def test_captain_must_be_starting_player(self):
        result = self.validate(case_name="illegal_captain_on_bench")

        issue = self.assert_issue(result, "captain_not_in_starting_lineup")
        self.assertEqual(issue["player_ids"], ["M5"])

    def test_captain_and_vice_must_differ(self):
        players = self.base_squad()
        case = self.lineup_case("legal_1_4_4_2_lineup")

        result = self.engine.validate_plan(
            players,
            starting_player_ids=case["starting_player_ids"],
            bench_order=case["bench_order"],
            captain_player_id="M1",
            vice_captain_player_id="M1",
            declared_bank_units=self.correct_bank(players),
        )

        self.assert_issue(
            result,
            "captain_and_vice_captain_must_differ",
        )

    def test_substitute_goalkeeper_must_use_slot_zero(self):
        players = self.base_squad()
        case = self.lineup_case("legal_1_4_4_2_lineup")
        reordered_bench = ["D5", "G2", "M5", "F3"]

        result = self.engine.validate_plan(
            players,
            starting_player_ids=case["starting_player_ids"],
            bench_order=reordered_bench,
            captain_player_id=case["captain_player_id"],
            vice_captain_player_id=case["vice_captain_player_id"],
            declared_bank_units=self.correct_bank(players),
        )

        self.assert_issue(result, "bench_goalkeeper_slot_invalid")
        self.assert_issue(result, "bench_outfield_slots_contain_goalkeeper")

    def test_starting_and_bench_must_partition_squad(self):
        players = self.base_squad()
        case = self.lineup_case("legal_1_4_4_2_lineup")
        invalid_bench = ["G2", "D5", "M5", "UNKNOWN"]

        result = self.engine.validate_plan(
            players,
            starting_player_ids=case["starting_player_ids"],
            bench_order=invalid_bench,
            captain_player_id=case["captain_player_id"],
            vice_captain_player_id=case["vice_captain_player_id"],
            declared_bank_units=self.correct_bank(players),
        )

        self.assert_issue(result, "bench_players_not_in_squad")
        partition_issue = self.assert_issue(
            result,
            "starting_and_bench_do_not_partition_squad",
        )
        self.assertIn("F3", partition_issue["details"]["missing_from_plan"])
        self.assertIn("UNKNOWN", partition_issue["details"]["extra_in_plan"])


if __name__ == "__main__":
    unittest.main()
