from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ml.features.build_fixture_horizon import (
    DEFAULT_HORIZON,
    DEFAULT_START_GW,
    FixtureHorizonInputError,
    add_run_fields,
    analyze_horizon,
    annotate_kickoff_changes,
    artifact_definitions,
    build_player_fixture_eligibility,
    build_team_fixture_horizon,
    build_team_gameweek_horizon,
    normalize_current_players,
    normalize_fixture_scope,
    normalize_target_teams,
    validate_requested_scope,
    validate_rollover_report,
)


class FixtureHorizonTests(unittest.TestCase):
    def teams_mapping(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for team_id in range(1, 21):
            rows.append(
                {
                    "target_season": "2026_27",
                    "target_team_id": team_id,
                    "target_team_name": "Team %s" % team_id,
                    "target_team_short_name": "T%02d" % team_id,
                }
            )
        return pd.DataFrame(rows)

    def current_players(self, players_per_team: int = 2) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        player_id = 1
        for team_id in range(1, 21):
            for index in range(players_per_team):
                rows.append(
                    {
                        "target_season": "2026_27",
                        "target_player_id": player_id,
                        "target_player_code": 100000 + player_id,
                        "target_player_name": "Player %s" % player_id,
                        "target_web_name": "P%s" % player_id,
                        "target_team_id": team_id,
                        "target_team_name": "Team %s" % team_id,
                        "target_team_short_name": "T%02d" % team_id,
                        "target_position": ("GKP", "DEF", "MID", "FWD")[
                            player_id % 4
                        ],
                        "target_price_units": 50,
                        "target_price": 5.0,
                        "target_status": "a",
                        "current_selection_eligible": not (
                            team_id == 20 and index == 1
                        ),
                    }
                )
                player_id += 1
        return pd.DataFrame(rows)

    def fixtures(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        fixture_id = 1
        for gameweek in range(1, 6):
            for home_id in range(1, 20, 2):
                away_id = home_id + 1
                rows.append(
                    {
                        "target_season": "2026_27",
                        "gameweek": gameweek,
                        "fixture_id": fixture_id,
                        "kickoff_time_utc": "2026-08-%02dT15:00:00Z"
                        % (20 + gameweek),
                        "home_team_id": home_id,
                        "home_team_name": "Team %s" % home_id,
                        "home_team_short_name": "T%02d" % home_id,
                        "away_team_id": away_id,
                        "away_team_name": "Team %s" % away_id,
                        "away_team_short_name": "T%02d" % away_id,
                        "started": False,
                        "finished": False,
                    }
                )
                fixture_id += 1
        return pd.DataFrame(rows)

    def rollover_report(self) -> Dict[str, Any]:
        return {
            "target_season": "2026_27",
            "as_of_time_utc": "2026-08-04T07:53:42Z",
            "passed": True,
            "audit_only": True,
            "writes_database": False,
            "blockers": [],
            "readiness": {
                "current_player_pool_validated": True,
                "target_team_transition_validated": True,
                "gw1_gw5_fixture_scope_validated": True,
                "ready_for_prediction_write": False,
            },
            "run_metadata": {"run_id": "day76c_parent_run"},
        }

    def normalized_inputs(self):
        teams = normalize_target_teams(self.teams_mapping(), "2026_27")
        players = normalize_current_players(
            self.current_players(),
            "2026_27",
            teams["team_id"].tolist(),
        )
        fixtures = normalize_fixture_scope(
            self.fixtures(),
            "2026_27",
            1,
            5,
            teams["team_id"].tolist(),
        )
        fixtures, change_log, comparison = annotate_kickoff_changes(
            fixtures, pd.DataFrame()
        )
        return teams, players, fixtures, change_log, comparison

    def test_defaults_match_fast_lane_scope(self) -> None:
        self.assertEqual(DEFAULT_START_GW, 1)
        self.assertEqual(DEFAULT_HORIZON, 5)
        self.assertEqual(validate_requested_scope("2026_27", 1, 5), (1, 5))

    def test_scope_refuses_horizon_beyond_gw38(self) -> None:
        with self.assertRaisesRegex(FixtureHorizonInputError, "must not exceed"):
            validate_requested_scope("2026_27", 37, 3)

    def test_rollover_report_accepts_read_only_checkpoint(self) -> None:
        result = validate_rollover_report(
            self.rollover_report(), target_season="2026_27"
        )
        self.assertEqual(result["target_season"], "2026_27")
        self.assertEqual(result["parent_run_id"], "day76c_parent_run")

    def test_rollover_report_refuses_prediction_write_readiness(self) -> None:
        report = self.rollover_report()
        report["readiness"]["ready_for_prediction_write"] = True
        with self.assertRaisesRegex(
            FixtureHorizonInputError, "must remain false"
        ):
            validate_rollover_report(report)

    def test_target_team_scope_requires_twenty_teams(self) -> None:
        with self.assertRaisesRegex(FixtureHorizonInputError, "exactly 20"):
            normalize_target_teams(self.teams_mapping().iloc[:-1], "2026_27")

    def test_fixture_horizon_has_fifty_rows(self) -> None:
        teams, _, fixtures, _, _ = self.normalized_inputs()
        self.assertEqual(len(teams), 20)
        self.assertEqual(len(fixtures), 50)
        self.assertTrue(fixtures["fixture_identity_valid"].all())

    def test_duplicate_fixture_id_is_detected(self) -> None:
        teams = normalize_target_teams(self.teams_mapping(), "2026_27")
        source = self.fixtures()
        source.loc[1, "fixture_id"] = source.loc[0, "fixture_id"]
        fixtures = normalize_fixture_scope(
            source, "2026_27", 1, 5, teams["team_id"].tolist()
        )
        self.assertGreater(int(fixtures["duplicate_fixture_id_flag"].sum()), 0)

    def test_duplicate_natural_fixture_is_detected(self) -> None:
        teams = normalize_target_teams(self.teams_mapping(), "2026_27")
        source = pd.concat([self.fixtures(), self.fixtures().iloc[[0]]])
        source.iloc[-1, source.columns.get_loc("fixture_id")] = 999
        fixtures = normalize_fixture_scope(
            source, "2026_27", 1, 5, teams["team_id"].tolist()
        )
        self.assertGreater(int(fixtures["duplicate_natural_key_flag"].sum()), 0)

    def test_missing_kickoff_is_visible_without_erasing_fixture(self) -> None:
        teams = normalize_target_teams(self.teams_mapping(), "2026_27")
        source = self.fixtures()
        source.loc[0, "kickoff_time_utc"] = None
        fixtures = normalize_fixture_scope(
            source, "2026_27", 1, 5, teams["team_id"].tolist()
        )
        row = fixtures[fixtures["fixture_id"] == 1].iloc[0]
        self.assertEqual(row["kickoff_time_status"], "missing")
        self.assertFalse(bool(row["kickoff_time_known"]))

    def test_kickoff_change_comparison_is_visible(self) -> None:
        teams = normalize_target_teams(self.teams_mapping(), "2026_27")
        fixtures = normalize_fixture_scope(
            self.fixtures(), "2026_27", 1, 5, teams["team_id"].tolist()
        )
        previous = fixtures[["fixture_id", "kickoff_time_utc"]].copy()
        previous.loc[0, "kickoff_time_utc"] = "2026-08-20T15:00:00Z"
        previous = previous.rename(
            columns={"kickoff_time_utc": "previous_kickoff_time_utc"}
        )
        previous["previous_kickoff_status"] = "known"
        fixtures, _, summary = annotate_kickoff_changes(fixtures, previous)
        changed = fixtures[fixtures["fixture_id"] == 1].iloc[0]
        self.assertEqual(changed["kickoff_change_status"], "changed")
        self.assertEqual(summary["changed_fixture_count"], 1)

    def test_team_fixture_expansion_has_home_and_away_rows(self) -> None:
        _, _, fixtures, _, _ = self.normalized_inputs()
        team_fixtures = build_team_fixture_horizon(fixtures)
        self.assertEqual(len(team_fixtures), 100)
        fixture_one = team_fixtures[team_fixtures["fixture_id"] == 1]
        self.assertEqual(len(fixture_one), 2)
        self.assertEqual(set(fixture_one["is_home"]), {True, False})

    def test_team_gameweek_detects_blank_and_double(self) -> None:
        teams, _, fixtures, _, _ = self.normalized_inputs()
        changed = fixtures[
            ~(
                (fixtures["gameweek"] == 1)
                & (fixtures["fixture_id"] == 1)
            )
        ].copy()
        extra = changed[changed["gameweek"] == 1].iloc[[0]].copy()
        extra["fixture_id"] = 999
        extra["home_team_id"] = 1
        extra["home_team_name"] = "Team 1"
        extra["home_team_short_name"] = "T01"
        extra["away_team_id"] = 3
        extra["away_team_name"] = "Team 3"
        extra["away_team_short_name"] = "T03"
        changed = pd.concat([changed, extra], ignore_index=True)
        team_fixtures = build_team_fixture_horizon(changed)
        team_gameweeks = build_team_gameweek_horizon(
            teams, team_fixtures, 1, 5
        )
        team_two = team_gameweeks[
            (team_gameweeks["team_id"] == 2)
            & (team_gameweeks["gameweek"] == 1)
        ].iloc[0]
        team_three = team_gameweeks[
            (team_gameweeks["team_id"] == 3)
            & (team_gameweeks["gameweek"] == 1)
        ].iloc[0]
        self.assertTrue(bool(team_two["blank_gw_flag"]))
        self.assertTrue(bool(team_three["double_gw_flag"]))

    def test_player_fixture_context_covers_every_player_gameweek(self) -> None:
        teams, players, fixtures, _, _ = self.normalized_inputs()
        team_fixtures = build_team_fixture_horizon(fixtures)
        team_gameweeks = build_team_gameweek_horizon(
            teams, team_fixtures, 1, 5
        )
        player_context = build_player_fixture_eligibility(
            players, team_fixtures, team_gameweeks, 1, 5
        )
        self.assertEqual(len(player_context), len(players) * 5)
        self.assertEqual(
            len(player_context[["player_id", "gameweek"]].drop_duplicates()),
            len(players) * 5,
        )

    def test_current_ineligible_player_remains_ineligible(self) -> None:
        teams, players, fixtures, _, _ = self.normalized_inputs()
        team_fixtures = build_team_fixture_horizon(fixtures)
        team_gameweeks = build_team_gameweek_horizon(
            teams, team_fixtures, 1, 5
        )
        player_context = build_player_fixture_eligibility(
            players, team_fixtures, team_gameweeks, 1, 5
        )
        ineligible_id = int(
            players[~players["selection_eligible"]].iloc[0]["player_id"]
        )
        rows = player_context[player_context["player_id"] == ineligible_id]
        self.assertFalse(rows["player_fixture_eligible"].any())
        self.assertEqual(set(rows["eligibility_reason"]), {"current_player_ineligible"})

    def test_blank_player_context_has_explicit_sentinel(self) -> None:
        teams, players, fixtures, _, _ = self.normalized_inputs()
        fixtures = fixtures[
            ~(
                (fixtures["gameweek"] == 1)
                & (fixtures["fixture_id"] == 1)
            )
        ].copy()
        team_fixtures = build_team_fixture_horizon(fixtures)
        team_gameweeks = build_team_gameweek_horizon(
            teams, team_fixtures, 1, 5
        )
        player_context = build_player_fixture_eligibility(
            players, team_fixtures, team_gameweeks, 1, 5
        )
        blank_rows = player_context[
            (player_context["team_id"].isin([1, 2]))
            & (player_context["gameweek"] == 1)
        ]
        self.assertTrue(blank_rows["blank_gw_flag"].all())
        self.assertTrue(blank_rows["fixture_id"].isna().all())
        eligible_source = blank_rows[blank_rows["selection_eligible"]]
        self.assertEqual(set(eligible_source["eligibility_reason"]), {"blank_gameweek"})

    def test_full_balanced_horizon_passes(self) -> None:
        teams, players, fixtures, _, comparison = self.normalized_inputs()
        frames, validation = analyze_horizon(
            fixtures, teams, players, 1, 5, comparison
        )
        self.assertTrue(validation["passed"])
        self.assertTrue(validation["horizon_complete_for_consumption"])
        self.assertEqual(len(frames["fixture_horizon"]), 50)
        self.assertEqual(len(frames["team_fixture_horizon"]), 100)
        self.assertEqual(len(frames["team_gameweek_horizon"]), 100)
        self.assertEqual(len(frames["player_fixture_eligibility"]), 200)
        self.assertEqual(validation["context"]["blank_team_gameweek_count"], 0)
        self.assertEqual(validation["context"]["double_team_gameweek_count"], 0)

    def test_run_fields_and_artifact_set_are_explicit(self) -> None:
        frame = pd.DataFrame([{"fixture_id": 1}])
        enriched = add_run_fields(
            frame,
            "fixture_horizon_2026_27_gw1_test",
            "2026_27",
            1,
            5,
            "2026-08-04T07:53:42Z",
            "day76c_parent_run",
        )
        self.assertEqual(enriched.iloc[0]["horizon"], 5)
        self.assertEqual(enriched.iloc[0]["end_gw"], 5)
        definitions = artifact_definitions()
        self.assertEqual(len(definitions), 8)
        self.assertIn("player_fixture_eligibility_csv", definitions)
        self.assertIn("fixture_horizon_report_json", definitions)


if __name__ == "__main__":
    unittest.main()
