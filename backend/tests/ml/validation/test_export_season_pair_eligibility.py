from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from ml.validation.export_season_pair_eligibility import (
    BACKTEST_MODES,
    EARLY_SEASON_BACKTEST,
    ELIGIBILITY_VERSION,
    PRE_GW1_BACKTEST,
    build_report,
    enumerate_season_pairs,
    evaluate_pair,
    index_pair_artifacts,
    mode_decision,
    normalize_season_key,
    profile_pair_csv,
    source_prior_availability,
    summarize_pair_identity,
    target_availability,
)


def gw_coverage(count: int) -> Dict[str, Any]:
    covered = set(range(1, min(count, 38) + 1))
    return {
        "available": True,
        "min_gw": min(covered) if covered else None,
        "max_gw": max(covered) if covered else None,
        "distinct_gw_count": len(covered),
        "missing_gws_1_to_38": [
            gw for gw in range(1, 39) if gw not in covered
        ],
    }


def table(
    rows: int,
    gw_count: int = 0,
    finished: int = 0,
    scored: int = 0,
    duplicate_groups: int = 0,
) -> Dict[str, Any]:
    return {
        "row_count": rows,
        "duplicate_groups": duplicate_groups,
        "gw_coverage": gw_coverage(gw_count),
        "fixture_completion": {
            "finished_true_count": finished,
            "rows_with_complete_scores": scored,
        },
    }


def season_info(
    classification: str,
    role: str,
    canonical: bool = False,
    staging: bool = False,
    current_actual_gws: int = 38,
) -> Dict[str, Any]:
    canonical_tables = {
        "teams": table(20 if canonical else 0),
        "players": table(841 if canonical else 0),
        "fixtures": table(
            380 if canonical else 0,
            38 if canonical else 0,
            (
                min(current_actual_gws, 38) * 10
                if canonical
                else 0
            ),
            (
                min(current_actual_gws, 38) * 10
                if canonical
                else 0
            ),
        ),
        "player_gw_stats": table(
            1000 if canonical else 0,
            current_actual_gws if canonical else 0,
        ),
        "gameweeks": table(38 if canonical else 0),
    }
    staging_tables = {
        "historical_teams": table(20 if staging else 0),
        "historical_players": table(804 if staging else 0),
        "historical_fixtures": table(
            380 if staging else 0,
            38 if staging else 0,
        ),
        "historical_player_gw_stats": table(
            27000 if staging else 0,
            38 if staging else 0,
        ),
    }
    return {
        "classification": {
            "classification": classification,
            "scope_role": role,
        },
        "database": {
            "canonical": canonical_tables,
            "staging": staging_tables,
        },
    }


def inventory_fixture() -> Dict[str, Any]:
    return {
        "audit_version": "fpl_historical_season_inventory_v2",
        "seasons": {
            "2023_24": season_info(
                "mapping-required",
                "historical_source_only",
                canonical=False,
                staging=False,
            ),
            "2024_25": season_info(
                "mapping-required",
                "historical",
                canonical=False,
                staging=True,
            ),
            "2025_26": season_info(
                "training-ready",
                "historical",
                canonical=True,
                staging=False,
            ),
            "2026_27": season_info(
                "evaluation-only",
                "current_in_progress",
                canonical=True,
                staging=False,
                current_actual_gws=2,
            ),
        },
    }


def safe_pair_artifacts(
    source: str = "2024_25",
    target: str = "2025_26",
) -> List[Dict[str, Any]]:
    return [
        {
            "kind": "player_mapping",
            "source_season": source,
            "target_season": target,
            "row_count": 3880,
            "top_row_count": 804,
            "auto_approved_count": 529,
            "manual_review_count": 275,
            "unmatched_count": 50,
            "duplicate_accepted_raw_id_count": 0,
            "duplicate_accepted_candidate_id_count": 0,
            "unsafe_accepted_count": 0,
            "accepted_missing_safety_evidence_count": 0,
            "path": "player_mapping.csv",
        },
        {
            "kind": "team_mapping",
            "source_season": source,
            "target_season": target,
            "row_count": 20,
            "top_row_count": 20,
            "auto_approved_count": 17,
            "manual_review_count": 3,
            "unmatched_count": 3,
            "duplicate_accepted_raw_id_count": 0,
            "duplicate_accepted_candidate_id_count": 0,
            "unsafe_accepted_count": 0,
            "accepted_missing_safety_evidence_count": 0,
            "path": "team_mapping.csv",
        },
        {
            "kind": "player_prior",
            "source_season": source,
            "target_season": target,
            "row_count": 804,
            "path": "player_prior.csv",
        },
        {
            "kind": "team_prior",
            "source_season": source,
            "target_season": target,
            "row_count": 20,
            "path": "team_prior.csv",
        },
        {
            "kind": "pre_gw1_player_features",
            "source_season": source,
            "target_season": target,
            "row_count": 841,
            "duplicate_player_id_count": 0,
            "explicit_prior_fallback_partition_complete": True,
            "prior_without_safe_mapping_count": 0,
            "path": "pre_gw1_player_features.csv",
        },
        {
            "kind": "pre_gw1_match_features_with_fallback",
            "source_season": source,
            "target_season": target,
            "row_count": 10,
            "both_teams_effective_true_count": 10,
            "covered_target_team_id_count": 20,
            "fallback_applied_target_team_id_count": 3,
            "path": "pre_gw1_match_features_with_fallback.csv",
        },
    ]


class SeasonPairEligibilityTests(unittest.TestCase):
    def test_normalize_season_key_accepts_project_and_folder_forms(self) -> None:
        self.assertEqual(normalize_season_key("2024_25"), "2024_25")
        self.assertEqual(normalize_season_key("2024-25"), "2024_25")
        self.assertEqual(normalize_season_key("/data/2024-25/gws"), "2024_25")

    def test_enumerate_pairs_includes_adjacent_and_multi_lookback(self) -> None:
        pairs = enumerate_season_pairs(
            ["2023_24", "2024_25", "2025_26", "2026_27"]
        )
        self.assertEqual(len(pairs), 6)
        self.assertEqual(
            sum(1 for row in pairs if row["pair_kind"] == "adjacent"),
            3,
        )
        self.assertEqual(
            sum(1 for row in pairs if row["pair_kind"] == "multi-lookback"),
            3,
        )

    def test_staging_source_is_prior_buildable(self) -> None:
        availability = source_prior_availability(
            season_info(
                "mapping-required",
                "historical",
                staging=True,
            )
        )
        self.assertTrue(availability["available"])
        self.assertEqual(
            availability["source_kind"],
            "historical_staging_ready",
        )

    def test_raw_only_mapping_required_source_is_not_prior_buildable(self) -> None:
        availability = source_prior_availability(
            season_info(
                "mapping-required",
                "historical_source_only",
                canonical=False,
                staging=False,
            )
        )
        self.assertFalse(availability["available"])
        self.assertEqual(
            availability["source_kind"],
            "not_prior_buildable_yet",
        )

    def test_target_current_actuals_allow_pre_gw1_but_block_full_early_season(self) -> None:
        target = target_availability(
            season_info(
                "evaluation-only",
                "current_in_progress",
                canonical=True,
                current_actual_gws=2,
            )
        )
        self.assertTrue(target["pre_gw1_target_ready"])
        self.assertFalse(target["early_season_target_ready"])

    def test_partial_identity_mapping_requires_explicit_full_target_fallback(self) -> None:
        inv = inventory_fixture()
        source_prior = source_prior_availability(inv["seasons"]["2024_25"])
        target = target_availability(inv["seasons"]["2025_26"])

        partial_without_fallback = safe_pair_artifacts()[:4]
        identity = summarize_pair_identity(
            partial_without_fallback,
            source_prior,
            target,
        )
        self.assertFalse(identity["player_mapping_safe"])
        self.assertFalse(identity["team_mapping_safe"])
        self.assertTrue(
            any(
                "fallback coverage is not proven" in item
                for item in identity["player_identity_blockers"]
            )
        )
        self.assertTrue(
            any(
                "fallback coverage is not proven" in item
                for item in identity["team_identity_blockers"]
            )
        )

    def test_safe_auto_mappings_plus_fallback_allow_historical_pair(self) -> None:
        inv = inventory_fixture()
        artifacts = safe_pair_artifacts()
        pair = {
            "source_season": "2024_25",
            "target_season": "2025_26",
            "pair_kind": "adjacent",
            "lookback_seasons": 1,
        }
        row = evaluate_pair(
            pair,
            inv,
            index_pair_artifacts(artifacts),
        )
        self.assertEqual(row["pair_status"], "ALLOWED")
        self.assertEqual(set(row["allowed_modes"]), set(BACKTEST_MODES))
        self.assertTrue(
            row["identity_and_fallback"][
                "player_partial_coverage_safe_via_fallback"
            ]
        )
        self.assertTrue(
            row["identity_and_fallback"][
                "team_partial_coverage_safe_via_fallback"
            ]
        )
        self.assertFalse(row["execution_approval"]["approved"])

    def test_missing_pair_mapping_blocks_even_when_source_and_target_data_are_ready(self) -> None:
        inv = inventory_fixture()
        pair = {
            "source_season": "2025_26",
            "target_season": "2026_27",
            "pair_kind": "adjacent",
            "lookback_seasons": 1,
        }
        row = evaluate_pair(pair, inv, {})
        self.assertEqual(
            row["pair_status"],
            "BLOCKED_PENDING_IDENTITY",
        )
        self.assertFalse(
            row["mode_results"][PRE_GW1_BACKTEST]["allowed"]
        )
        self.assertTrue(
            any(
                "player identity mapping evidence is missing" in item
                for item in row["blockers"]
            )
        )

    def test_current_target_full_early_season_mode_records_actual_blocker(self) -> None:
        inv = inventory_fixture()
        artifacts = safe_pair_artifacts(
            source="2025_26",
            target="2026_27",
        )
        # Match source counts to canonical 2025/26 source.
        for item in artifacts:
            if item["kind"] == "player_mapping":
                item["top_row_count"] = 841
                item["auto_approved_count"] = 600
            if item["kind"] == "pre_gw1_player_features":
                item["row_count"] = 841
        pair = {
            "source_season": "2025_26",
            "target_season": "2026_27",
            "pair_kind": "adjacent",
            "lookback_seasons": 1,
        }
        row = evaluate_pair(
            pair,
            inv,
            index_pair_artifacts(artifacts),
        )
        self.assertTrue(
            row["mode_results"][PRE_GW1_BACKTEST]["allowed"]
        )
        self.assertFalse(
            row["mode_results"][EARLY_SEASON_BACKTEST]["allowed"]
        )
        self.assertTrue(
            any(
                "GW1-GW5" in item
                for item in row["mode_results"][EARLY_SEASON_BACKTEST]["blockers"]
            )
        )

    def test_profile_player_mapping_csv_uses_top_rows_not_all_candidate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "player_identity_mapping_candidates.csv"
            fieldnames = [
                "source_season",
                "target_season",
                "raw_player_id",
                "candidate_rank",
                "candidate_player_id",
                "match_status",
                "is_auto_approved",
                "needs_manual_review",
                "identity_auto_approval_safe",
            ]
            rows = [
                {
                    "source_season": "2024_25",
                    "target_season": "2025_26",
                    "raw_player_id": "1",
                    "candidate_rank": "1",
                    "candidate_player_id": "10",
                    "match_status": "auto_approved_candidate",
                    "is_auto_approved": "True",
                    "needs_manual_review": "False",
                    "identity_auto_approval_safe": "True",
                },
                {
                    "source_season": "2024_25",
                    "target_season": "2025_26",
                    "raw_player_id": "1",
                    "candidate_rank": "2",
                    "candidate_player_id": "11",
                    "match_status": "candidate",
                    "is_auto_approved": "False",
                    "needs_manual_review": "True",
                    "identity_auto_approval_safe": "False",
                },
                {
                    "source_season": "2024_25",
                    "target_season": "2025_26",
                    "raw_player_id": "2",
                    "candidate_rank": "",
                    "candidate_player_id": "",
                    "match_status": "unmatched",
                    "is_auto_approved": "False",
                    "needs_manual_review": "True",
                    "identity_auto_approval_safe": "",
                },
            ]
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            profile = profile_pair_csv(path)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile["row_count"], 3)
        self.assertEqual(profile["top_row_count"], 2)
        self.assertEqual(profile["auto_approved_count"], 1)
        self.assertEqual(profile["manual_review_count"], 1)
        self.assertEqual(profile["unmatched_count"], 1)


    def test_partial_player_mapping_requires_explicit_no_prior_partition(self) -> None:
        inv = inventory_fixture()
        source_prior = source_prior_availability(inv["seasons"]["2024_25"])
        target = target_availability(inv["seasons"]["2025_26"])
        artifacts = safe_pair_artifacts()
        for item in artifacts:
            if item["kind"] == "pre_gw1_player_features":
                item["explicit_prior_fallback_partition_complete"] = False

        identity = summarize_pair_identity(artifacts, source_prior, target)

        self.assertFalse(identity["player_mapping_safe"])
        self.assertTrue(
            any(
                "fallback coverage is not proven" in item
                for item in identity["player_identity_blockers"]
            )
        )

    def test_partial_team_mapping_requires_all_target_teams_and_transition_fallbacks(self) -> None:
        inv = inventory_fixture()
        source_prior = source_prior_availability(inv["seasons"]["2024_25"])
        target = target_availability(inv["seasons"]["2025_26"])
        artifacts = safe_pair_artifacts()
        for item in artifacts:
            if item["kind"] == "pre_gw1_match_features_with_fallback":
                item["covered_target_team_id_count"] = 17
                item["fallback_applied_target_team_id_count"] = 0

        identity = summarize_pair_identity(artifacts, source_prior, target)

        self.assertFalse(identity["team_mapping_safe"])
        self.assertTrue(
            any(
                "fallback coverage is not proven" in item
                for item in identity["team_identity_blockers"]
            )
        )

    def test_auto_approved_mappings_require_explicit_safety_evidence(self) -> None:
        inv = inventory_fixture()
        source_prior = source_prior_availability(inv["seasons"]["2024_25"])
        target = target_availability(inv["seasons"]["2025_26"])
        artifacts = safe_pair_artifacts()
        for item in artifacts:
            if item["kind"] == "player_mapping":
                item["accepted_missing_safety_evidence_count"] = 1

        identity = summarize_pair_identity(artifacts, source_prior, target)

        self.assertFalse(identity["player_mapping_safe"])
        self.assertTrue(
            any(
                "missing explicit auto-approval safety evidence" in item
                for item in identity["player_identity_blockers"]
            )
        )

    def test_profile_player_features_proves_explicit_prior_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pre_gw1_player_features_2024_25_to_2025_26.csv"
            fieldnames = [
                "source_season",
                "target_season",
                "player_id",
                "has_prev_season_player_prior",
                "no_prior_flag",
                "player_mapping_status",
            ]
            rows = [
                {
                    "source_season": "2024_25",
                    "target_season": "2025_26",
                    "player_id": "1",
                    "has_prev_season_player_prior": "True",
                    "no_prior_flag": "False",
                    "player_mapping_status": "auto_approved_candidate",
                },
                {
                    "source_season": "2024_25",
                    "target_season": "2025_26",
                    "player_id": "2",
                    "has_prev_season_player_prior": "False",
                    "no_prior_flag": "True",
                    "player_mapping_status": "no_safe_accepted_mapping",
                },
            ]
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            profile = profile_pair_csv(path)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertTrue(profile["explicit_prior_fallback_partition_complete"])
        self.assertEqual(profile["prior_without_safe_mapping_count"], 0)
        self.assertEqual(profile["rows_with_explicit_no_prior_flag"], 1)


    def test_build_report_explicitly_decides_every_pair(self) -> None:
        inv = inventory_fixture()
        report = build_report(inv, safe_pair_artifacts())
        self.assertEqual(report["eligibility_version"], ELIGIBILITY_VERSION)
        self.assertEqual(report["season_count"], 4)
        self.assertEqual(report["pair_count"], 6)
        self.assertEqual(report["adjacent_pair_count"], 3)
        self.assertEqual(report["multi_lookback_pair_count"], 3)

        for row in report["pairs"]:
            self.assertEqual(
                set(row["mode_results"].keys()),
                set(BACKTEST_MODES),
            )
            for mode in BACKTEST_MODES:
                self.assertIn(
                    row["mode_results"][mode]["status"],
                    {"ALLOWED", "BLOCKED"},
                )
                if not row["mode_results"][mode]["allowed"]:
                    self.assertTrue(row["mode_results"][mode]["blockers"])


if __name__ == "__main__":
    unittest.main()
