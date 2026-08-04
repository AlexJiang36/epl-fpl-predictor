from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ml.validation.check_target_season_rollover import (
    build_player_identity_mapping,
    load_rollover_config,
    parse_bootstrap,
    parse_fixtures,
    run_checkpoint,
    validate_current_player_pool,
    validate_fixture_scope,
    validate_rule_registries,
    validate_source_player_pool,
    validate_team_transition,
)


TEST_FILE = Path(__file__).resolve()
BACKEND_ROOT = TEST_FILE.parents[3]
REPOSITORY_ROOT = BACKEND_ROOT.parent
CONFIG_ROOT = REPOSITORY_ROOT / "config" / "fpl"
ROLLOVER_CONFIG_PATH = CONFIG_ROOT / "target_season_rollover_2026_27.json"


class TargetSeasonRolloverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_rollover_config(ROLLOVER_CONFIG_PATH)

    def make_source_and_target(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
        source_short_names = self.config["source_team_short_names"]
        target_short_names = self.config["target_team_short_names"]
        source_teams = pd.DataFrame(
            [
                {
                    "source_team_id": index,
                    "source_fpl_team_id": index,
                    "source_team_name": "Source %s" % short_name,
                    "source_team_short_name": short_name,
                }
                for index, short_name in enumerate(source_short_names, start=1)
            ]
        )

        target_team_rows: List[Dict[str, Any]] = []
        target_team_id_by_short: Dict[str, int] = {}
        for index, short_name in enumerate(target_short_names, start=101):
            target_team_id_by_short[short_name] = index
            target_team_rows.append(
                {
                    "id": index,
                    "code": 1000 + index,
                    "name": "Target %s" % short_name,
                    "short_name": short_name,
                }
            )

        source_player_rows: List[Dict[str, Any]] = []
        target_elements: List[Dict[str, Any]] = []
        position_id = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
        source_team_id_by_short = {
            row.source_team_short_name: row.source_team_id
            for row in source_teams.itertuples(index=False)
        }
        next_source_id = 1
        next_target_id = 1001

        for change in self.config["position_changes"]:
            first_name, second_name = change["player_name"].split(" ", 1)
            short_name = change["team_short_name"]
            source_player_rows.append(
                {
                    "source_player_id": next_source_id,
                    "source_fpl_player_id": next_source_id,
                    "source_player_name": change["player_name"],
                    "source_web_name": second_name,
                    "source_position": change["old_position"],
                    "source_team_id": source_team_id_by_short[short_name],
                    "source_team_name": "Source %s" % short_name,
                    "source_team_short_name": short_name,
                }
            )
            target_elements.append(
                {
                    "id": next_target_id,
                    "code": 5000 + next_target_id,
                    "first_name": first_name,
                    "second_name": second_name,
                    "web_name": second_name,
                    "team": target_team_id_by_short[short_name],
                    "element_type": position_id[change["new_position"]],
                    "now_cost": 55,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "news": "",
                    "news_added": None,
                }
            )
            next_source_id += 1
            next_target_id += 1

        # One stable player for each unchanged team. These also prove exact identity coverage.
        for short_name in self.config["unchanged_team_short_names"]:
            player_name = "Stable %s" % short_name
            source_player_rows.append(
                {
                    "source_player_id": next_source_id,
                    "source_fpl_player_id": next_source_id,
                    "source_player_name": player_name,
                    "source_web_name": player_name,
                    "source_position": "MID",
                    "source_team_id": source_team_id_by_short[short_name],
                    "source_team_name": "Source %s" % short_name,
                    "source_team_short_name": short_name,
                }
            )
            target_elements.append(
                {
                    "id": next_target_id,
                    "code": 5000 + next_target_id,
                    "first_name": "Stable",
                    "second_name": short_name,
                    "web_name": player_name,
                    "team": target_team_id_by_short[short_name],
                    "element_type": position_id["MID"],
                    "now_cost": 50,
                    "status": "d" if short_name == "ARS" else "a",
                    "chance_of_playing_next_round": 75 if short_name == "ARS" else 100,
                    "news": "Synthetic test row",
                    "news_added": "2026-08-01T00:00:00Z",
                }
            )
            next_source_id += 1
            next_target_id += 1

        # Promoted-team player deliberately has no source identity and must remain prior-ineligible.
        target_elements.append(
            {
                "id": next_target_id,
                "code": 5000 + next_target_id,
                "first_name": "New",
                "second_name": "CoventryPlayer",
                "web_name": "CoventryPlayer",
                "team": target_team_id_by_short["COV"],
                "element_type": position_id["FWD"],
                "now_cost": 45,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "news": "",
                "news_added": None,
            }
        )

        bootstrap = {
            "teams": target_team_rows,
            "element_types": [
                {"id": 1, "singular_name_short": "GKP"},
                {"id": 2, "singular_name_short": "DEF"},
                {"id": 3, "singular_name_short": "MID"},
                {"id": 4, "singular_name_short": "FWD"},
            ],
            "elements": target_elements,
            "events": [
                {"id": gameweek, "deadline_time": "2026-08-%02dT18:30:00Z" % (14 + gameweek * 7)}
                for gameweek in range(1, 6)
            ],
        }

        fixture_rows: List[Dict[str, Any]] = []
        target_ids = [target_team_id_by_short[short_name] for short_name in target_short_names]
        fixture_id = 1
        for gameweek in range(1, 6):
            for index in range(0, len(target_ids), 2):
                fixture_rows.append(
                    {
                        "id": fixture_id,
                        "event": gameweek,
                        "team_h": target_ids[index],
                        "team_a": target_ids[index + 1],
                        "kickoff_time": "2026-09-%02dT15:00:00Z" % (gameweek * 2 + index // 2),
                        "started": False,
                        "finished": False,
                    }
                )
                fixture_id += 1

        return source_teams, pd.DataFrame(source_player_rows), bootstrap, fixture_rows

    def parsed_target_data(self):
        source_teams, source_players, bootstrap, fixtures_raw = self.make_source_and_target()
        target_teams, target_players, _ = parse_bootstrap(bootstrap)
        fixtures = parse_fixtures(fixtures_raw, target_teams, 1, 5)
        return source_teams, source_players, target_teams, target_players, fixtures

    def test_2026_27_rule_registries_and_examples_pass(self) -> None:
        result = validate_rule_registries(CONFIG_ROOT, self.config)
        self.assertTrue(result["passed"], result["blockers"])
        self.assertEqual(result["actual_versions"]["scoring"], "fpl_2026_27_scoring_v1")
        self.assertEqual(result["actual_versions"]["squad_transfer"], "fpl_2026_27_squad_transfer_v1")
        self.assertEqual(result["actual_versions"]["chips"], "fpl_2026_27_chips_v1")
        self.assertEqual(result["actual_versions"]["bonus_points_system"], "fpl_2026_27_bps_v1")
        self.assertEqual(result["special_transfer_events"], [])

    def test_target_team_transition_is_exact(self) -> None:
        source_teams, _, target_teams, _, _ = self.parsed_target_data()
        mapping, result = validate_team_transition(source_teams, target_teams, self.config)
        self.assertTrue(result["passed"], result["blockers"])
        self.assertEqual(result["unchanged_team_count"], 17)
        self.assertEqual(result["promoted_team_short_names"], ["COV", "HUL", "IPS"])
        self.assertEqual(result["relegated_team_short_names"], ["BUR", "WHU", "WOL"])
        self.assertEqual(len(mapping), 23)

    def test_source_player_pool_requires_safe_identity_fields(self) -> None:
        source_teams, source_players, _, _, _ = self.parsed_target_data()
        result = validate_source_player_pool(source_players, source_teams)
        self.assertTrue(result["passed"], result["blockers"])
        broken = source_players.copy()
        broken.loc[0, "source_position"] = None
        failed = validate_source_player_pool(broken, source_teams)
        self.assertFalse(failed["passed"])
        self.assertTrue(any("source_position" in blocker for blocker in failed["blockers"]))

    def test_current_player_pool_requires_price_position_and_availability(self) -> None:
        _, _, target_teams, target_players, _ = self.parsed_target_data()
        result = validate_current_player_pool(target_players, target_teams)
        self.assertTrue(result["passed"], result["blockers"])
        broken = target_players.copy()
        broken.loc[0, "target_price_units"] = None
        broken.loc[1, "target_position"] = None
        broken.loc[2, "target_status"] = None
        failed = validate_current_player_pool(broken, target_teams)
        self.assertFalse(failed["passed"])
        self.assertGreaterEqual(len(failed["blockers"]), 3)

    def test_gw1_gw5_fixture_scope_requires_ten_and_timezone(self) -> None:
        _, _, target_teams, _, fixtures = self.parsed_target_data()
        result = validate_fixture_scope(fixtures, target_teams, self.config)
        self.assertTrue(result["passed"], result["blockers"])
        broken = fixtures.iloc[1:].copy()
        failed = validate_fixture_scope(broken, target_teams, self.config)
        self.assertFalse(failed["passed"])
        self.assertTrue(any("GW1" in blocker for blocker in failed["blockers"]))

    def test_player_mapping_surfaces_all_official_position_changes(self) -> None:
        _, source_players, _, target_players, _ = self.parsed_target_data()
        mapping, result = build_player_identity_mapping(source_players, target_players, self.config)
        self.assertTrue(result["passed"], result["blockers"])
        self.assertEqual(result["verified_official_position_change_count"], 11)
        self.assertEqual(result["present_expected_official_position_change_count"], 11)
        self.assertEqual(result["expected_official_position_change_count"], 11)
        self.assertEqual(result["target_absent_expected_position_change_count"], 0)
        changed = mapping[mapping["position_change_status"] == "verified_official"]
        self.assertEqual(len(changed), 11)
        self.assertTrue(changed["historical_prior_eligible"].all())

    def test_configured_position_change_target_absent_is_warning_not_blocker(self) -> None:
        _, source_players, _, target_players, _ = self.parsed_target_data()
        target_players = target_players[
            target_players["target_player_name"] != "Eric Moreira"
        ].reset_index(drop=True)
        mapping, result = build_player_identity_mapping(source_players, target_players, self.config)
        self.assertTrue(result["passed"], result["blockers"])
        self.assertEqual(result["verified_official_position_change_count"], 10)
        self.assertEqual(result["present_expected_official_position_change_count"], 10)
        self.assertEqual(result["expected_official_position_change_count"], 11)
        self.assertEqual(result["target_absent_expected_position_change_count"], 1)
        self.assertEqual(
            result["target_absent_expected_position_changes"],
            ["Eric Moreira MID->DEF"],
        )
        self.assertEqual(result["unverified_present_expected_position_change_count"], 0)
        self.assertFalse((mapping["target_player_name"] == "Eric Moreira").any())
        self.assertTrue(any("absent from the current target player pool" in item for item in result["warnings"]))

    def test_configured_position_change_target_present_but_unmapped_fails_closed(self) -> None:
        _, source_players, _, target_players, _ = self.parsed_target_data()
        source_players = source_players[
            source_players["source_player_name"] != "Eric Moreira"
        ].reset_index(drop=True)
        _, result = build_player_identity_mapping(source_players, target_players, self.config)
        self.assertFalse(result["passed"])
        self.assertEqual(result["present_expected_official_position_change_count"], 11)
        self.assertEqual(result["verified_official_position_change_count"], 10)
        self.assertEqual(result["target_absent_expected_position_change_count"], 0)
        self.assertEqual(result["unverified_present_expected_position_change_count"], 1)
        self.assertIn("Eric Moreira MID->DEF", result["unverified_present_expected_position_changes"])
        self.assertTrue(any("present in the target pool" in item for item in result["blockers"]))

    def test_unresolved_target_player_is_retained_but_prior_ineligible(self) -> None:
        _, source_players, _, target_players, _ = self.parsed_target_data()
        mapping, result = build_player_identity_mapping(source_players, target_players, self.config)
        unresolved = mapping[mapping["target_player_name"] == "New CoventryPlayer"].iloc[0]
        self.assertEqual(unresolved["mapping_status"], "unresolved_no_exact_identity")
        self.assertFalse(bool(unresolved["historical_prior_eligible"]))
        self.assertTrue(result["passed"], result["blockers"])
        self.assertGreater(result["unresolved_or_blocked_mapping_count"], 0)

    def test_unlisted_exact_position_change_fails_closed(self) -> None:
        _, source_players, _, target_players, _ = self.parsed_target_data()
        stable_index = target_players[target_players["target_player_name"] == "Stable LIV"].index[0]
        target_players.loc[stable_index, "target_position"] = "FWD"
        _, result = build_player_identity_mapping(source_players, target_players, self.config)
        self.assertFalse(result["passed"])
        self.assertEqual(result["unlisted_position_change_count"], 1)

    def test_target_scope_mismatch_fails_closed(self) -> None:
        source_teams, source_players, bootstrap, fixtures_raw = self.make_source_and_target()
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = run_checkpoint(
                config=self.config,
                source_teams=source_teams,
                source_players=source_players,
                bootstrap=bootstrap,
                fixtures_raw=fixtures_raw,
                config_root=CONFIG_ROOT,
                artifact_root=Path(temporary_directory),
                source_season="2025_26",
                target_season="2025_26",
                target_gw=1,
                as_of_time="2026-08-04T00:00:00Z",
            )
        self.assertFalse(report["passed"])
        self.assertFalse(report["readiness"]["target_scope_confirmed"])
        self.assertTrue(any("target_season" in blocker for blocker in report["blockers"]))

    def test_full_checkpoint_writes_immutable_json_markdown_and_csv_artifacts(self) -> None:
        source_teams, source_players, bootstrap, fixtures_raw = self.make_source_and_target()
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_root = Path(temporary_directory)
            report = run_checkpoint(
                config=self.config,
                source_teams=source_teams,
                source_players=source_players,
                bootstrap=bootstrap,
                fixtures_raw=fixtures_raw,
                config_root=CONFIG_ROOT,
                artifact_root=artifact_root,
                source_season="2025_26",
                target_season="2026_27",
                target_gw=1,
                as_of_time="2026-08-04T00:00:00Z",
            )
            self.assertTrue(report["passed"], report["blockers"])
            self.assertTrue(report["readiness"]["stop_point_satisfied"])
            self.assertFalse(report["readiness"]["ready_for_prediction_write"])
            self.assertEqual(len(report["artifacts"]["keys"]), 8)
            for key in report["artifacts"]["keys"].values():
                self.assertTrue((artifact_root / key).is_file(), key)
            report_json_path = artifact_root / report["artifacts"]["keys"]["rollover_report_json"]
            written = json.loads(report_json_path.read_text(encoding="utf-8"))
            self.assertTrue(written["passed"])
            self.assertEqual(written["run_metadata"]["target_season"], "2026_27")
            self.assertEqual(written["run_metadata"]["target_gw"], 1)
            self.assertEqual(written["run_metadata"]["horizon"], 5)
            player_pool_path = artifact_root / report["artifacts"]["keys"]["current_player_pool_csv"]
            written_players = pd.read_csv(player_pool_path)
            self.assertTrue((written_players["target_season"] == "2026_27").all())
            self.assertTrue((written_players["target_gw"] == 1).all())

    def test_rule_rollover_metadata_is_explicit_not_silent(self) -> None:
        self.assertEqual(
            self.config["mapping_policy"]["configured_position_change_target_absent_policy"],
            "warning_not_blocker_until_player_appears_in_current_pool",
        )
        self.assertEqual(
            self.config["mapping_policy"]["configured_position_change_target_present_unverified_policy"],
            "blocker",
        )
        for filename, source_version in (
            ("scoring_rules_2026_27.json", "fpl_2025_26_scoring_v1"),
            ("squad_transfer_rules_2026_27.json", "fpl_2025_26_squad_transfer_v1"),
            ("chip_rules_2026_27.json", "fpl_2025_26_chips_v1"),
        ):
            document = json.loads((CONFIG_ROOT / filename).read_text(encoding="utf-8"))
            self.assertEqual(document["rollover_policy"]["copy_mode"], "explicit_reviewed_copy")
            self.assertEqual(document["rollover_policy"]["source_rules_version"], source_version)
            self.assertEqual(document["effective_season"], "2026_27")
            self.assertNotIn("2025_26", document["rules_version"])

    def test_scoring_registry_records_2026_27_bps_model_limitation(self) -> None:
        document = json.loads((CONFIG_ROOT / "scoring_rules_2026_27.json").read_text(encoding="utf-8"))
        policy = document["rollover_policy"]["bonus_points_system"]
        self.assertEqual(policy["policy_version"], "fpl_2026_27_bps_v1")
        self.assertFalse(policy["full_event_level_calculator_included"])
        self.assertEqual(len(policy["changes"]), 6)
        self.assertTrue(document["scoring"]["bonus"]["target_season_changes_recorded"])

    def test_squad_registry_removes_2025_26_afcon_event(self) -> None:
        document = json.loads((CONFIG_ROOT / "squad_transfer_rules_2026_27.json").read_text(encoding="utf-8"))
        self.assertEqual(document["transfers"]["special_events"], [])
        names = [case["name"] for case in document["deterministic_examples"]["transfer_cases"]]
        self.assertIn("no_2026_27_special_top_up_after_gameweek_15", names)
        self.assertFalse(any("afcon_event_tops_up" in name for name in names))


if __name__ == "__main__":
    unittest.main()
