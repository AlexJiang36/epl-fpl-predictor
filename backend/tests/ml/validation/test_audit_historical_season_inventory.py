from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml.validation.audit_historical_season_inventory import (
    classify_season,
    normalize_season_key,
    prepared_source_full_enough_for_staging,
    profile_csv,
    raw_source_full_enough_for_staging,
    write_markdown,
)


def _coverage(gws: int) -> dict:
    return {
        "available": True,
        "min_gw": 1 if gws else None,
        "max_gw": gws if gws else None,
        "distinct_gw_count": gws,
        "missing_gws_1_to_38": list(range(gws + 1, 39)),
    }


def _source_profile(
    *,
    rows: int,
    gws: int,
    min_gw: int = 1,
    max_gw: int = 38,
    missing_gws: list = None,
) -> dict:
    return {
        "row_count": rows,
        "missing_required_fields": [],
        "required_nulls": {},
        "gw_coverage": {
            "distinct_gw_count": gws,
            "min_gw": min_gw,
            "max_gw": max_gw,
            "missing_gws_1_to_38": (
                [] if missing_gws is None else missing_gws
            ),
        },
        "duplicate_rows": 0,
    }


def _table(
    rows: int,
    *,
    gws: int = 0,
    finished: int = 0,
    scored: int = 0,
) -> dict:
    return {
        "exists": True,
        "row_count": rows,
        "duplicate_groups": 0,
        "critical_nulls": {},
        "gw_coverage": _coverage(gws) if gws else None,
        "fixture_completion": {
            "finished_true_count": finished,
            "rows_with_complete_scores": scored,
        },
    }


def _database(
    *,
    canonical_full: bool = False,
    canonical_partial: bool = False,
    staging_full: bool = False,
    unmapped_teams: int = 0,
    unmapped_players: int = 0,
) -> dict:
    if canonical_full:
        canonical = {
            "teams": _table(20),
            "players": _table(700),
            "fixtures": _table(
                380,
                gws=38,
                finished=380,
                scored=380,
            ),
            "player_gw_stats": _table(28000, gws=38),
        }
    elif canonical_partial:
        canonical = {
            "teams": _table(20),
            "players": _table(650),
            "fixtures": _table(
                380,
                gws=38,
                finished=20,
                scored=20,
            ),
            "player_gw_stats": _table(1200, gws=2),
        }
    else:
        canonical = {
            "teams": _table(0),
            "players": _table(0),
            "fixtures": _table(0),
            "player_gw_stats": _table(0),
        }

    if staging_full:
        staging = {
            "historical_teams": _table(20),
            "historical_players": _table(800),
            "historical_fixtures": _table(
                380,
                gws=38,
                finished=380,
                scored=380,
            ),
            "historical_player_gw_stats": _table(27000, gws=38),
        }
    else:
        staging = {
            "historical_teams": _table(0),
            "historical_players": _table(0),
            "historical_fixtures": _table(0),
            "historical_player_gw_stats": _table(0),
        }

    return {
        "canonical": canonical,
        "staging": staging,
        "mapping": {
            "teams": {
                "available": staging_full,
                "mapped": max(0, 20 - unmapped_teams),
                "unmapped": unmapped_teams,
            },
            "players": {
                "available": staging_full,
                "mapped": max(0, 800 - unmapped_players),
                "unmapped": unmapped_players,
            },
        },
    }


class HistoricalSeasonInventoryTests(unittest.TestCase):
    def test_normalize_season_key_accepts_project_and_folder_forms(self) -> None:
        self.assertEqual(normalize_season_key("2024_25"), "2024_25")
        self.assertEqual(normalize_season_key("2024-25"), "2024_25")
        self.assertIsNone(normalize_season_key("2024"))

    def test_complete_clean_canonical_season_is_training_ready(self) -> None:
        result = classify_season(
            "2025_26",
            _database(canonical_full=True),
            {"season_dir_exists": False},
            {"season_dir_exists": False},
            "2026_27",
        )
        self.assertEqual(result["classification"], "training-ready")
        self.assertEqual(result["blockers"], [])

    def test_partial_current_canonical_season_is_evaluation_only(self) -> None:
        result = classify_season(
            "2026_27",
            _database(canonical_partial=True),
            {"season_dir_exists": False},
            {"season_dir_exists": False},
            "2026_27",
        )
        self.assertEqual(result["classification"], "evaluation-only")
        self.assertEqual(result["scope_role"], "current_in_progress")
        self.assertTrue(
            any("player-GW coverage incomplete" in item for item in result["blockers"])
        )

    def test_complete_staging_with_unmapped_identity_is_mapping_required(self) -> None:
        result = classify_season(
            "2024_25",
            _database(
                staging_full=True,
                unmapped_teams=20,
                unmapped_players=804,
            ),
            {"season_dir_exists": True},
            {"season_dir_exists": True},
            "2026_27",
        )
        self.assertEqual(result["classification"], "mapping-required")
        self.assertTrue(
            any("identity mapping incomplete" in item for item in result["blockers"])
        )

    def test_raw_folder_without_adapter_contract_is_unusable(self) -> None:
        result = classify_season(
            "2016_17",
            _database(),
            {
                "season_dir_exists": True,
                "adapter_compatible": False,
            },
            {"season_dir_exists": False},
            "2026_27",
        )
        self.assertEqual(result["classification"], "unusable")

    def test_profile_csv_reports_duplicates_nulls_and_gw_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "player_gw_stats.csv"
            path.write_text(
                "season,element,gw,minutes,total_points\n"
                "2024_25,1,1,90,6\n"
                "2024_25,1,1,10,1\n"
                "2024_25,2,2,,2\n",
                encoding="utf-8",
            )
            result = profile_csv(
                path,
                duplicate_key_candidates=[["element", "gw"]],
                required_fields=["season", "gw", "minutes", "total_points"],
                gw_candidates=["gw"],
            )
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["duplicate_rows"], 1)
        self.assertEqual(result["required_nulls"]["minutes"], 1)
        self.assertEqual(result["gw_coverage"]["distinct_gw_count"], 2)

    def test_source_full_helpers_require_full_fixture_and_gw_coverage(self) -> None:
        fixture_profile = _source_profile(rows=380, gws=38)
        player_profile = _source_profile(rows=27000, gws=38)
        raw = {
            "adapter_compatible": True,
            "fixture_profile": fixture_profile,
            "player_gw_profile": player_profile,
        }
        prepared = {
            "prepared_core_available": True,
            "files": {
                "fixtures": fixture_profile,
                "player_gw_stats": player_profile,
            },
        }
        self.assertTrue(raw_source_full_enough_for_staging(raw))
        self.assertTrue(prepared_source_full_enough_for_staging(prepared))

    def test_nonstandard_restart_gw_labels_require_mapping(self) -> None:
        fixture_profile = _source_profile(
            rows=380,
            gws=38,
            min_gw=1,
            max_gw=47,
            missing_gws=list(range(30, 39)),
        )
        player_profile = _source_profile(
            rows=22560,
            gws=38,
            min_gw=1,
            max_gw=47,
            missing_gws=list(range(30, 39)),
        )
        raw = {
            "season_dir_exists": True,
            "adapter_compatible": True,
            "fixture_profile": fixture_profile,
            "player_gw_profile": player_profile,
            "individual_gw_files": {
                "count": 47,
                "min_gw": 1,
                "max_gw": 47,
                "missing_gws_1_to_38": [],
            },
        }

        self.assertFalse(raw_source_full_enough_for_staging(raw))
        result = classify_season(
            "2019_20",
            _database(),
            raw,
            {"season_dir_exists": False},
            "2026_27",
        )
        self.assertEqual(result["classification"], "mapping-required")
        self.assertTrue(
            any(
                "timeline normalization" in blocker
                for blocker in result["blockers"]
            )
        )

    def test_empty_event_with_complete_fixture_source_requires_mapping(self) -> None:
        fixture_profile = _source_profile(
            rows=380,
            gws=37,
            missing_gws=[7],
        )
        player_profile = _source_profile(
            rows=26505,
            gws=37,
            missing_gws=[7],
        )
        raw = {
            "season_dir_exists": True,
            "adapter_compatible": True,
            "fixture_profile": fixture_profile,
            "player_gw_profile": player_profile,
            "individual_gw_files": {
                "count": 38,
                "min_gw": 1,
                "max_gw": 38,
                "missing_gws_1_to_38": [],
            },
        }

        self.assertFalse(raw_source_full_enough_for_staging(raw))
        result = classify_season(
            "2022_23",
            _database(),
            raw,
            {"season_dir_exists": False},
            "2026_27",
        )
        self.assertEqual(result["classification"], "mapping-required")
        self.assertTrue(
            any(
                "missing standard labels=[7]" in blocker
                for blocker in result["blockers"]
            )
        )


    def test_write_markdown_keeps_coverage_formatter_callable(self) -> None:
        report = {
            "audit_version": "test",
            "created_at": "2026-09-03T00:00:00+00:00",
            "read_only": True,
            "classification_summary": {
                "training-ready": [],
                "evaluation-only": [],
                "mapping-required": ["2022_23"],
                "unusable": [],
            },
            "seasons": {
                "2022_23": {
                    "classification": {
                        "classification": "mapping-required",
                        "scope_role": "historical",
                        "reasons": ["raw source exists"],
                        "blockers": ["GW timeline normalization required"],
                    },
                    "database": {
                        "canonical": {
                            "teams": _table(0),
                            "players": _table(0),
                            "fixtures": _table(0),
                            "player_gw_stats": _table(0),
                            "gameweeks": _table(0),
                        },
                        "staging": {
                            "historical_teams": _table(0),
                            "historical_players": _table(0),
                            "historical_fixtures": _table(0),
                            "historical_player_gw_stats": _table(0),
                        },
                        "field_availability": {
                            "canonical_players": {},
                            "staging_players": {},
                            "canonical_player_gw_scoring": {},
                            "staging_player_gw_scoring": {},
                        },
                    },
                    "raw_source": {
                        "season_dir_exists": True,
                        "adapter_compatible": True,
                        "player_price_fields": [],
                        "player_status_fields": [],
                        "fixture_profile": _source_profile(
                            rows=380,
                            gws=37,
                            missing_gws=[7],
                        ),
                        "player_gw_profile": {
                            **_source_profile(
                                rows=26505,
                                gws=37,
                                missing_gws=[7],
                            ),
                            "price_fields": ["value"],
                            "scoring_fields": ["minutes", "total_points"],
                        },
                        "individual_gw_files": {
                            "count": 38,
                            "min_gw": 1,
                            "max_gw": 38,
                            "missing_gws_1_to_38": [],
                        },
                    },
                    "prepared_source": {
                        "season_dir_exists": False,
                        "prepared_core_available": False,
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.md"
            write_markdown(report, str(path))
            text = path.read_text(encoding="utf-8")

        self.assertIn("missing standard labels=[7]", text)
        self.assertIn("raw player-GW", text)



if __name__ == "__main__":
    unittest.main()
