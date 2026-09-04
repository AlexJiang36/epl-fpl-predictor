from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


ELIGIBILITY_VERSION = "fpl_season_pair_eligibility_v2"

PRE_GW1_BACKTEST = "pre_gw1_backtest"
EARLY_SEASON_BACKTEST = "early_season_gw2_gw5_backtest"
BACKTEST_MODES = [PRE_GW1_BACKTEST, EARLY_SEASON_BACKTEST]

EXPECTED_TEAMS = 20
EXPECTED_FIXTURES = 380
EXPECTED_GWS = 38
PRE_GW1_REQUIRED_TARGET_ACTUAL_GWS = [1]
EARLY_SEASON_REQUIRED_TARGET_ACTUAL_GWS = [1, 2, 3, 4, 5]
PRE_GW1_REQUIRED_FINISHED_FIXTURES = 10
EARLY_SEASON_REQUIRED_FINISHED_FIXTURES = 50

BOOL_TRUE = {"1", "true", "t", "yes", "y"}

PAIR_ARTIFACT_HINTS = (
    "prior",
    "mapping",
    "identity",
    "pre_gw1",
)

DEFAULT_ARTIFACT_SUBDIR = "datasets/prepared-fpl-historical"
DEFAULT_DAY130A_RELATIVE = (
    "private-planning/historical-audits/day130a/historical_season_inventory.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_season_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    match = re.search(r"(20\d{2})[-_/](\d{2})", text)
    if match:
        return "%s_%s" % (match.group(1), match.group(2))
    if re.fullmatch(r"20\d{2}_\d{2}", text):
        return text
    return None


def season_start_year(season: str) -> int:
    return int(str(season).split("_", 1)[0])


def chronological_seasons(inventory: Mapping[str, Any]) -> List[str]:
    return sorted(
        [str(item) for item in inventory.get("seasons", {}).keys()],
        key=season_start_year,
    )


def enumerate_season_pairs(seasons: Sequence[str]) -> List[Dict[str, Any]]:
    ordered = sorted([str(item) for item in seasons], key=season_start_year)
    rows: List[Dict[str, Any]] = []
    for target_index in range(1, len(ordered)):
        target = ordered[target_index]
        for source_index in range(0, target_index):
            source = ordered[source_index]
            lookback = target_index - source_index
            rows.append(
                {
                    "source_season": source,
                    "target_season": target,
                    "lookback_seasons": lookback,
                    "pair_kind": "adjacent" if lookback == 1 else "multi-lookback",
                }
            )
    return rows


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in BOOL_TRUE:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def nullable_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def get_nested(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def row_count(table: Mapping[str, Any]) -> int:
    return int(table.get("row_count") or 0)


def coverage_gws(table: Mapping[str, Any]) -> Set[int]:
    coverage = table.get("gw_coverage") or {}
    if not coverage.get("available"):
        return set()

    missing = {
        int(item)
        for item in (coverage.get("missing_gws_1_to_38") or [])
        if nullable_int(item) is not None
    }
    if int(coverage.get("distinct_gw_count") or 0) == 0:
        return set()

    result: Set[int] = set()
    for gw in range(1, EXPECTED_GWS + 1):
        if gw not in missing:
            result.add(gw)
    return result


def fixture_finished_count(table: Mapping[str, Any]) -> int:
    completion = table.get("fixture_completion") or {}
    return int(completion.get("finished_true_count") or 0)


def fixture_scored_count(table: Mapping[str, Any]) -> int:
    completion = table.get("fixture_completion") or {}
    return int(completion.get("rows_with_complete_scores") or 0)


def canonical_table(season_info: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return (
        get_nested(season_info, ["database", "canonical", name], {})
        or {}
    )


def staging_table(season_info: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return (
        get_nested(season_info, ["database", "staging", name], {})
        or {}
    )


def season_classification(season_info: Mapping[str, Any]) -> Optional[str]:
    return get_nested(season_info, ["classification", "classification"])


def season_scope_role(season_info: Mapping[str, Any]) -> Optional[str]:
    return get_nested(season_info, ["classification", "scope_role"])


def source_prior_availability(season_info: Mapping[str, Any]) -> Dict[str, Any]:
    classification = season_classification(season_info)

    canonical_teams = canonical_table(season_info, "teams")
    canonical_players = canonical_table(season_info, "players")
    canonical_fixtures = canonical_table(season_info, "fixtures")
    canonical_player_gw = canonical_table(season_info, "player_gw_stats")

    staging_teams = staging_table(season_info, "historical_teams")
    staging_players = staging_table(season_info, "historical_players")
    staging_fixtures = staging_table(season_info, "historical_fixtures")
    staging_player_gw = staging_table(season_info, "historical_player_gw_stats")

    canonical_full = (
        classification == "training-ready"
        and row_count(canonical_teams) == EXPECTED_TEAMS
        and row_count(canonical_players) > 0
        and row_count(canonical_fixtures) == EXPECTED_FIXTURES
        and row_count(canonical_player_gw) > 0
        and len(coverage_gws(canonical_fixtures)) == EXPECTED_GWS
        and len(coverage_gws(canonical_player_gw)) == EXPECTED_GWS
    )

    staging_full = (
        row_count(staging_teams) == EXPECTED_TEAMS
        and row_count(staging_players) > 0
        and row_count(staging_fixtures) == EXPECTED_FIXTURES
        and row_count(staging_player_gw) > 0
        and len(coverage_gws(staging_fixtures)) == EXPECTED_GWS
        and len(coverage_gws(staging_player_gw)) == EXPECTED_GWS
        and int(staging_teams.get("duplicate_groups") or 0) == 0
        and int(staging_players.get("duplicate_groups") or 0) == 0
        and int(staging_fixtures.get("duplicate_groups") or 0) == 0
        and int(staging_player_gw.get("duplicate_groups") or 0) == 0
    )

    if canonical_full:
        source_kind = "canonical_training_ready"
    elif staging_full:
        source_kind = "historical_staging_ready"
    elif classification == "unusable":
        source_kind = "unusable"
    else:
        source_kind = "not_prior_buildable_yet"

    return {
        "available": bool(canonical_full or staging_full),
        "source_kind": source_kind,
        "canonical_training_ready": canonical_full,
        "historical_staging_ready": staging_full,
        "classification": classification,
        "scope_role": season_scope_role(season_info),
        "source_team_rows": (
            row_count(canonical_teams)
            if canonical_full
            else row_count(staging_teams)
        ),
        "source_player_rows": (
            row_count(canonical_players)
            if canonical_full
            else row_count(staging_players)
        ),
        "source_fixture_rows": (
            row_count(canonical_fixtures)
            if canonical_full
            else row_count(staging_fixtures)
        ),
        "source_player_gw_rows": (
            row_count(canonical_player_gw)
            if canonical_full
            else row_count(staging_player_gw)
        ),
    }


def target_availability(season_info: Mapping[str, Any]) -> Dict[str, Any]:
    teams = canonical_table(season_info, "teams")
    players = canonical_table(season_info, "players")
    fixtures = canonical_table(season_info, "fixtures")
    player_gw = canonical_table(season_info, "player_gw_stats")

    fixture_gws = coverage_gws(fixtures)
    actual_gws = coverage_gws(player_gw)
    finished_count = fixture_finished_count(fixtures)
    scored_count = fixture_scored_count(fixtures)

    canonical_universe_available = (
        row_count(teams) == EXPECTED_TEAMS
        and row_count(players) > 0
        and row_count(fixtures) > 0
    )

    pre_gw1_target_ready = (
        canonical_universe_available
        and 1 in fixture_gws
        and set(PRE_GW1_REQUIRED_TARGET_ACTUAL_GWS).issubset(actual_gws)
        and finished_count >= PRE_GW1_REQUIRED_FINISHED_FIXTURES
        and scored_count >= PRE_GW1_REQUIRED_FINISHED_FIXTURES
    )

    early_season_target_ready = (
        canonical_universe_available
        and set(EARLY_SEASON_REQUIRED_TARGET_ACTUAL_GWS).issubset(fixture_gws)
        and set(EARLY_SEASON_REQUIRED_TARGET_ACTUAL_GWS).issubset(actual_gws)
        and finished_count >= EARLY_SEASON_REQUIRED_FINISHED_FIXTURES
        and scored_count >= EARLY_SEASON_REQUIRED_FINISHED_FIXTURES
    )

    return {
        "classification": season_classification(season_info),
        "scope_role": season_scope_role(season_info),
        "canonical_universe_available": canonical_universe_available,
        "team_rows": row_count(teams),
        "player_rows": row_count(players),
        "fixture_rows": row_count(fixtures),
        "fixture_gws": sorted(fixture_gws),
        "finished_fixture_rows": finished_count,
        "scored_fixture_rows": scored_count,
        "player_gw_rows": row_count(player_gw),
        "actual_gws": sorted(actual_gws),
        "pre_gw1_target_ready": pre_gw1_target_ready,
        "early_season_target_ready": early_season_target_ready,
    }


def artifact_kind(path: Path, columns: Sequence[str]) -> str:
    name = path.name.lower()
    cols = set(columns)

    if "player" in name and "mapping" in name:
        return "player_mapping"
    if "team" in name and "mapping" in name:
        return "team_mapping"
    if "player" in name and "prior" in name:
        return "player_prior"
    if "team" in name and "prior" in name:
        return "team_prior"
    if "pre_gw1_player_features" in name:
        return "pre_gw1_player_features"
    if "pre_gw1_match_features_with_fallback" in name:
        return "pre_gw1_match_features_with_fallback"
    if "pre_gw1_match_features" in name:
        return "pre_gw1_match_features"
    if "pre_gw1_player_prediction" in name:
        return "pre_gw1_player_prediction"
    if "pre_gw1_match_prediction" in name:
        return "pre_gw1_match_prediction"
    if "pre_gw1_scoreline" in name:
        return "pre_gw1_scoreline"

    if {"raw_player_id", "candidate_player_id", "match_status"} <= cols:
        return "player_mapping"
    if {"raw_team_id", "candidate_team_id", "match_status"} <= cols:
        return "team_mapping"
    if {"raw_player_id", "source_season", "target_season"} <= cols:
        return "player_prior_or_feature"
    if {"raw_team_id", "source_season", "target_season"} <= cols:
        return "team_prior_or_feature"
    return "other_pair_artifact"


def first_singleton(values: Set[str]) -> Optional[str]:
    if len(values) == 1:
        return next(iter(values))
    return None


def bool_count(rows: Iterable[Mapping[str, Any]], field: str, expected: bool) -> int:
    result = 0
    for row in rows:
        value = parse_bool(row.get(field))
        if value is expected:
            result += 1
    return result


def duplicate_non_null(values: Iterable[Any]) -> int:
    seen: Set[str] = set()
    duplicates = 0
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if text in seen:
            duplicates += 1
        else:
            seen.add(text)
    return duplicates


def profile_pair_csv(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            columns = list(reader.fieldnames or [])
            if not columns:
                return None
            rows = list(reader)
    except Exception as exc:
        return {
            "path": str(path),
            "error": "%s: %s" % (type(exc).__name__, exc),
        }

    source_values: Set[str] = set()
    target_values: Set[str] = set()
    for row in rows:
        source = normalize_season_key(row.get("source_season"))
        if source is None:
            source = normalize_season_key(row.get("source_seasons"))
        target = normalize_season_key(row.get("target_season"))
        if source:
            source_values.add(source)
        if target:
            target_values.add(target)

    kind = artifact_kind(path, columns)
    result: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "kind": kind,
        "columns": columns,
        "row_count": len(rows),
        "source_season": first_singleton(source_values),
        "target_season": first_singleton(target_values),
    }

    if kind in {"player_mapping", "team_mapping"} and "candidate_rank" in columns:
        top_rows = [
            row
            for row in rows
            if nullable_int(row.get("candidate_rank")) in {None, 1}
        ]
        accepted_statuses = (
            {"auto_approved_candidate"}
            if kind == "player_mapping"
            else {"auto_approved_team_candidate"}
        )
        accepted = [
            row
            for row in top_rows
            if str(row.get("match_status") or "").strip() in accepted_statuses
            or parse_bool(row.get("is_auto_approved")) is True
        ]

        if kind == "player_mapping":
            candidate_id_field = "candidate_player_id"
            raw_id_field = "raw_player_id"
            safety_fields = [
                field
                for field in (
                    "identity_auto_approval_safe",
                    "safe_name_match_for_auto_approval",
                )
                if field in columns
            ]
        else:
            candidate_id_field = "candidate_team_id"
            raw_id_field = "raw_team_id"
            safety_fields = [
                field
                for field in ("safe_team_match_for_auto_approval",)
                if field in columns
            ]

        accepted_with_explicit_safety = 0
        unsafe_accepted_count = 0
        accepted_missing_safety_evidence_count = 0
        for row in accepted:
            safety_values = [parse_bool(row.get(field)) for field in safety_fields]
            if any(value is False for value in safety_values):
                unsafe_accepted_count += 1
            elif safety_values and any(value is True for value in safety_values):
                accepted_with_explicit_safety += 1
            else:
                accepted_missing_safety_evidence_count += 1

        result.update(
            {
                "top_row_count": len(top_rows),
                "auto_approved_count": len(accepted),
                "accepted_with_explicit_safety_count": accepted_with_explicit_safety,
                "accepted_missing_safety_evidence_count": (
                    accepted_missing_safety_evidence_count
                ),
                "manual_review_count": bool_count(
                    top_rows, "needs_manual_review", True
                ),
                "unmatched_count": sum(
                    1
                    for row in top_rows
                    if "unmatched"
                    in str(row.get("match_status") or "").strip().lower()
                ),
                "ambiguous_count": sum(
                    1
                    for row in top_rows
                    if "ambiguous"
                    in str(row.get("match_status") or "").strip().lower()
                ),
                "duplicate_accepted_raw_id_count": duplicate_non_null(
                    row.get(raw_id_field) for row in accepted
                ),
                "duplicate_accepted_candidate_id_count": duplicate_non_null(
                    row.get(candidate_id_field) for row in accepted
                ),
                "unsafe_accepted_count": unsafe_accepted_count,
            }
        )

    if "player_id" in columns:
        result["duplicate_player_id_count"] = duplicate_non_null(
            row.get("player_id") for row in rows
        )
    if "fixture_id" in columns:
        result["duplicate_fixture_id_count"] = duplicate_non_null(
            row.get("fixture_id") for row in rows
        )

    if "has_prev_season_player_prior" in columns:
        result["rows_with_safe_or_available_prior"] = bool_count(
            rows, "has_prev_season_player_prior", True
        )
        if "no_prior_flag" in columns:
            result["rows_with_explicit_no_prior_flag"] = bool_count(
                rows, "no_prior_flag", True
            )
            inconsistent = 0
            unknown = 0
            prior_without_safe_mapping = 0
            for row in rows:
                has_prior = parse_bool(row.get("has_prev_season_player_prior"))
                no_prior = parse_bool(row.get("no_prior_flag"))
                if has_prior is None or no_prior is None:
                    unknown += 1
                    continue
                if has_prior == no_prior:
                    inconsistent += 1
                if (
                    has_prior is True
                    and str(row.get("player_mapping_status") or "").strip()
                    == "no_safe_accepted_mapping"
                ):
                    prior_without_safe_mapping += 1
            result["prior_flag_unknown_count"] = unknown
            result["prior_flag_inconsistency_count"] = inconsistent
            result["prior_without_safe_mapping_count"] = (
                prior_without_safe_mapping
            )
            result["explicit_prior_fallback_partition_complete"] = bool(
                unknown == 0
                and inconsistent == 0
                and prior_without_safe_mapping == 0
                and (
                    int(result["rows_with_safe_or_available_prior"])
                    + int(result["rows_with_explicit_no_prior_flag"])
                    == len(rows)
                )
            )

    if {"home_team_id", "away_team_id"} <= set(columns):
        covered_team_ids: Set[str] = set()
        fallback_team_ids: Set[str] = set()
        for row in rows:
            for side in ("home", "away"):
                team_id = str(row.get("%s_team_id" % side) or "").strip()
                if team_id:
                    covered_team_ids.add(team_id)
                    if parse_bool(row.get("%s_team_fallback_applied" % side)) is True:
                        fallback_team_ids.add(team_id)
        result["covered_target_team_id_count"] = len(covered_team_ids)
        result["fallback_applied_target_team_id_count"] = len(
            fallback_team_ids
        )

    if "both_teams_have_effective_team_features" in columns:
        result["both_teams_effective_true_count"] = bool_count(
            rows, "both_teams_have_effective_team_features", True
        )
    if "fallback_policy_used" in columns:
        result["rows_with_fallback_policy"] = sum(
            1
            for row in rows
            if str(row.get("fallback_policy_used") or "").strip()
        )

    return result


def infer_pair_from_path(
    path: Path,
    seasons: Sequence[str],
) -> Tuple[Optional[str], Optional[str]]:
    text = str(path)
    present: Set[str] = set()
    for season in seasons:
        if season in text or season.replace("_", "-") in text:
            present.add(season)
    ordered = sorted(present, key=season_start_year)
    if len(ordered) >= 2:
        return ordered[0], ordered[-1]
    return None, None


def discover_pair_artifacts(
    artifact_roots: Sequence[Path],
    seasons: Sequence[str],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for root in artifact_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(str(root)):
            dirnames[:] = [
                item
                for item in dirnames
                if item not in {".git", "__pycache__", ".venv", "node_modules"}
            ]
            for filename in filenames:
                lower = filename.lower()
                if not lower.endswith(".csv"):
                    continue
                if not any(hint in lower for hint in PAIR_ARTIFACT_HINTS):
                    continue

                path = Path(dirpath) / filename
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)

                profile = profile_pair_csv(path)
                if not profile:
                    continue

                source = profile.get("source_season")
                target = profile.get("target_season")
                if not source or not target:
                    inferred_source, inferred_target = infer_pair_from_path(
                        path, seasons
                    )
                    source = source or inferred_source
                    target = target or inferred_target
                profile["source_season"] = source
                profile["target_season"] = target
                profile["artifact_root"] = str(root)
                results.append(profile)

    return sorted(results, key=lambda item: item.get("path", ""))


def index_pair_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    result: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in artifacts:
        source = item.get("source_season")
        target = item.get("target_season")
        if not source or not target:
            continue
        key = (str(source), str(target))
        result.setdefault(key, []).append(dict(item))
    return result


def choose_artifact(
    artifacts: Sequence[Mapping[str, Any]],
    kind: str,
) -> Optional[Dict[str, Any]]:
    candidates = [
        dict(item)
        for item in artifacts
        if item.get("kind") == kind and not item.get("error")
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            int(item.get("row_count") or 0),
            str(item.get("path") or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def summarize_pair_identity(
    artifacts: Sequence[Mapping[str, Any]],
    source_prior: Mapping[str, Any],
    target: Mapping[str, Any],
) -> Dict[str, Any]:
    player_mapping = choose_artifact(artifacts, "player_mapping")
    team_mapping = choose_artifact(artifacts, "team_mapping")
    player_prior = choose_artifact(artifacts, "player_prior")
    team_prior = choose_artifact(artifacts, "team_prior")
    player_features = choose_artifact(artifacts, "pre_gw1_player_features")
    match_fallback = choose_artifact(
        artifacts, "pre_gw1_match_features_with_fallback"
    )

    source_player_count = int(source_prior.get("source_player_rows") or 0)
    source_team_count = int(source_prior.get("source_team_rows") or 0)
    target_player_count = int(target.get("player_rows") or 0)

    player_mapping_safe = False
    player_partial_coverage_safe_via_fallback = False
    player_blockers: List[str] = []

    if player_mapping is None:
        player_blockers.append("pair-specific player identity mapping evidence is missing")
    else:
        if int(player_mapping.get("duplicate_accepted_raw_id_count") or 0) > 0:
            player_blockers.append("accepted player mappings contain duplicate raw-player identities")
        if int(player_mapping.get("duplicate_accepted_candidate_id_count") or 0) > 0:
            player_blockers.append("accepted player mappings contain duplicate target-player identities")
        if int(player_mapping.get("unsafe_accepted_count") or 0) > 0:
            player_blockers.append("accepted player mappings contain unsafe auto-approvals")
        if int(player_mapping.get("accepted_missing_safety_evidence_count") or 0) > 0:
            player_blockers.append(
                "accepted player mappings are missing explicit auto-approval safety evidence"
            )

        accepted = int(player_mapping.get("auto_approved_count") or 0)
        top_rows = int(player_mapping.get("top_row_count") or 0)
        full_mapping = (
            source_player_count > 0
            and top_rows == source_player_count
            and accepted == source_player_count
        )

        fallback_proof = (
            player_features is not None
            and target_player_count > 0
            and int(player_features.get("row_count") or 0) == target_player_count
            and int(player_features.get("duplicate_player_id_count") or 0) == 0
            and player_features.get(
                "explicit_prior_fallback_partition_complete"
            ) is True
        )
        partial_mapping = accepted > 0 and top_rows == source_player_count
        player_partial_coverage_safe_via_fallback = bool(
            partial_mapping and fallback_proof
        )

        player_mapping_safe = bool(
            not player_blockers
            and (full_mapping or player_partial_coverage_safe_via_fallback)
        )
        if not player_mapping_safe and not player_blockers:
            if accepted == 0:
                player_blockers.append("no safely auto-approved player mappings are available")
            elif top_rows != source_player_count:
                player_blockers.append(
                    "player mapping audit does not provide exactly one top/unmatched row per source player"
                )
            elif not fallback_proof:
                player_blockers.append(
                    "player mapping coverage is partial and complete target-player fallback coverage is not proven"
                )

    team_mapping_safe = False
    team_partial_coverage_safe_via_fallback = False
    team_blockers: List[str] = []

    if team_mapping is None:
        team_blockers.append("pair-specific team promotion/relegation mapping evidence is missing")
    else:
        if int(team_mapping.get("duplicate_accepted_raw_id_count") or 0) > 0:
            team_blockers.append("accepted team mappings contain duplicate raw-team identities")
        if int(team_mapping.get("duplicate_accepted_candidate_id_count") or 0) > 0:
            team_blockers.append("accepted team mappings contain duplicate target-team identities")
        if int(team_mapping.get("unsafe_accepted_count") or 0) > 0:
            team_blockers.append("accepted team mappings contain unsafe auto-approvals")
        if int(team_mapping.get("accepted_missing_safety_evidence_count") or 0) > 0:
            team_blockers.append(
                "accepted team mappings are missing explicit auto-approval safety evidence"
            )

        accepted = int(team_mapping.get("auto_approved_count") or 0)
        top_rows = int(team_mapping.get("top_row_count") or 0)
        full_mapping = (
            source_team_count > 0
            and top_rows == source_team_count
            and accepted == source_team_count
        )
        transition_gap_count = max(0, top_rows - accepted)
        fallback_proof = (
            match_fallback is not None
            and int(match_fallback.get("row_count") or 0) >= 10
            and int(match_fallback.get("both_teams_effective_true_count") or 0)
            == int(match_fallback.get("row_count") or 0)
            and int(match_fallback.get("covered_target_team_id_count") or 0)
            == EXPECTED_TEAMS
            and (
                transition_gap_count == 0
                or int(
                    match_fallback.get(
                        "fallback_applied_target_team_id_count"
                    )
                    or 0
                )
                >= transition_gap_count
            )
        )
        partial_mapping = accepted > 0 and top_rows == source_team_count
        team_partial_coverage_safe_via_fallback = bool(
            partial_mapping and fallback_proof
        )

        team_mapping_safe = bool(
            not team_blockers
            and (full_mapping or team_partial_coverage_safe_via_fallback)
        )
        if not team_mapping_safe and not team_blockers:
            if accepted == 0:
                team_blockers.append("no safely auto-approved team mappings are available")
            elif top_rows != source_team_count:
                team_blockers.append(
                    "team mapping audit does not provide exactly one top/unmatched row per source team"
                )
            elif not fallback_proof:
                team_blockers.append(
                    "team mapping coverage is partial and promoted/relegated-team fallback coverage is not proven"
                )

    player_auto_count = int((player_mapping or {}).get("auto_approved_count") or 0)
    player_top_count = int((player_mapping or {}).get("top_row_count") or 0)
    team_auto_count = int((team_mapping or {}).get("auto_approved_count") or 0)
    team_top_count = int((team_mapping or {}).get("top_row_count") or 0)

    return {
        "player_prior_artifact": player_prior,
        "team_prior_artifact": team_prior,
        "player_prior_available": bool(player_prior is not None or source_prior.get("available")),
        "team_prior_available": bool(team_prior is not None or source_prior.get("available")),
        "player_mapping": player_mapping,
        "team_mapping": team_mapping,
        "player_mapping_auto_approved_count": player_auto_count,
        "player_mapping_top_row_count": player_top_count,
        "player_mapping_coverage_rate": (
            round(player_auto_count / float(player_top_count), 4)
            if player_top_count > 0
            else None
        ),
        "team_mapping_auto_approved_count": team_auto_count,
        "team_mapping_top_row_count": team_top_count,
        "team_mapping_coverage_rate": (
            round(team_auto_count / float(team_top_count), 4)
            if team_top_count > 0
            else None
        ),
        "team_transition_gap_count": (
            max(0, team_top_count - team_auto_count)
            if team_top_count > 0
            else None
        ),
        "player_fallback_evidence": player_features,
        "team_fallback_evidence": match_fallback,
        "player_mapping_safe": player_mapping_safe,
        "team_mapping_safe": team_mapping_safe,
        "player_partial_coverage_safe_via_fallback": (
            player_partial_coverage_safe_via_fallback
        ),
        "team_partial_coverage_safe_via_fallback": (
            team_partial_coverage_safe_via_fallback
        ),
        "player_identity_blockers": player_blockers,
        "team_identity_blockers": team_blockers,
    }


def unique_ordered(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def base_pair_blockers(
    source_prior: Mapping[str, Any],
    target: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> List[str]:
    blockers: List[str] = []
    if not source_prior.get("available"):
        classification = source_prior.get("classification")
        if classification == "unusable":
            blockers.append("source season is classified unusable by Day130A")
        else:
            blockers.append(
                "source season is not yet prior-buildable from canonical or clean historical staging data"
            )

    if not target.get("canonical_universe_available"):
        blockers.append(
            "target canonical team/player/fixture universe is unavailable"
        )

    blockers.extend(identity.get("team_identity_blockers") or [])
    blockers.extend(identity.get("player_identity_blockers") or [])
    return unique_ordered(blockers)


def mode_decision(
    mode: str,
    source_prior: Mapping[str, Any],
    target: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers = base_pair_blockers(source_prior, target, identity)

    if mode == PRE_GW1_BACKTEST:
        if not target.get("pre_gw1_target_ready"):
            blockers.append(
                "target season lacks complete GW1 fixture/result/player-actual evidence required to evaluate a Pre-GW1 backtest"
            )
        required_actual_gws = list(PRE_GW1_REQUIRED_TARGET_ACTUAL_GWS)
    elif mode == EARLY_SEASON_BACKTEST:
        if not target.get("early_season_target_ready"):
            blockers.append(
                "target season lacks complete GW1-GW5 fixture/result/player-actual evidence required for the full GW2-GW5 early-season backtest"
            )
        required_actual_gws = list(EARLY_SEASON_REQUIRED_TARGET_ACTUAL_GWS)
    else:
        raise ValueError("Unsupported backtest mode: %s" % mode)

    blockers = unique_ordered(blockers)
    allowed = len(blockers) == 0

    notes: List[str] = []
    if identity.get("player_partial_coverage_safe_via_fallback"):
        notes.append(
            "player identity coverage is partial; eligibility depends only on safely auto-approved mappings plus explicit full-target fallback evidence"
        )
    if identity.get("team_partial_coverage_safe_via_fallback"):
        notes.append(
            "team continuity is partial because promotion/relegation is expected; eligibility depends on safely auto-approved team mappings plus explicit fallback evidence"
        )

    return {
        "mode": mode,
        "allowed": allowed,
        "status": "ALLOWED" if allowed else "BLOCKED",
        "required_target_actual_gws": required_actual_gws,
        "blockers": blockers,
        "notes": notes,
    }


def evaluate_pair(
    pair: Mapping[str, Any],
    inventory: Mapping[str, Any],
    artifact_index: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    source = str(pair["source_season"])
    target_season = str(pair["target_season"])

    source_info = inventory["seasons"][source]
    target_info = inventory["seasons"][target_season]

    source_prior = source_prior_availability(source_info)
    target = target_availability(target_info)
    artifacts = list(artifact_index.get((source, target_season), []))
    identity = summarize_pair_identity(artifacts, source_prior, target)

    mode_results = {
        mode: mode_decision(mode, source_prior, target, identity)
        for mode in BACKTEST_MODES
    }
    allowed_modes = [
        mode for mode in BACKTEST_MODES if mode_results[mode]["allowed"]
    ]
    blocked_modes = [
        mode for mode in BACKTEST_MODES if not mode_results[mode]["allowed"]
    ]

    data_ready_for_day131_identity_audit = bool(
        source_prior.get("available")
        and target.get("canonical_universe_available")
    )

    all_blockers: List[str] = []
    for mode in BACKTEST_MODES:
        all_blockers.extend(mode_results[mode]["blockers"])

    return {
        "source_season": source,
        "target_season": target_season,
        "pair_kind": pair["pair_kind"],
        "lookback_seasons": int(pair["lookback_seasons"]),
        "source_prior_availability": source_prior,
        "target_availability": target,
        "pair_artifact_count": len(artifacts),
        "pair_artifacts": artifacts,
        "identity_and_fallback": identity,
        "mode_results": mode_results,
        "allowed_modes": allowed_modes,
        "blocked_modes": blocked_modes,
        "pair_status": (
            "ALLOWED"
            if allowed_modes
            else (
                "BLOCKED_PENDING_IDENTITY"
                if data_ready_for_day131_identity_audit
                else "BLOCKED_DATA"
            )
        ),
        "data_ready_for_day131_identity_audit": (
            data_ready_for_day131_identity_audit
        ),
        "blockers": unique_ordered(all_blockers),
        "execution_approval": {
            "approved": False,
            "reason": (
                "Day130B records data/pair eligibility only. Leakage-safe as-of "
                "snapshot and temporal-leakage gates remain later milestones."
            ),
        },
    }


def build_report(
    inventory: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    seasons = chronological_seasons(inventory)
    pairs = enumerate_season_pairs(seasons)
    artifact_index = index_pair_artifacts(artifacts)
    evaluated = [
        evaluate_pair(pair, inventory, artifact_index) for pair in pairs
    ]

    allowed_pre = [
        "%s->%s" % (row["source_season"], row["target_season"])
        for row in evaluated
        if PRE_GW1_BACKTEST in row["allowed_modes"]
    ]
    allowed_early = [
        "%s->%s" % (row["source_season"], row["target_season"])
        for row in evaluated
        if EARLY_SEASON_BACKTEST in row["allowed_modes"]
    ]

    return {
        "eligibility_version": ELIGIBILITY_VERSION,
        "created_at": utc_now(),
        "read_only": True,
        "day130a_inventory_version": inventory.get("audit_version"),
        "season_count": len(seasons),
        "pair_count": len(evaluated),
        "adjacent_pair_count": sum(
            1 for row in evaluated if row["pair_kind"] == "adjacent"
        ),
        "multi_lookback_pair_count": sum(
            1 for row in evaluated if row["pair_kind"] == "multi-lookback"
        ),
        "artifact_count": len(artifacts),
        "allowed_pair_counts": {
            PRE_GW1_BACKTEST: len(allowed_pre),
            EARLY_SEASON_BACKTEST: len(allowed_early),
        },
        "allowed_pairs": {
            PRE_GW1_BACKTEST: allowed_pre,
            EARLY_SEASON_BACKTEST: allowed_early,
        },
        "pair_status_counts": {
            status: sum(1 for row in evaluated if row["pair_status"] == status)
            for status in [
                "ALLOWED",
                "BLOCKED_PENDING_IDENTITY",
                "BLOCKED_DATA",
            ]
        },
        "pairs": evaluated,
        "notes": [
            "Every chronological source<target pair is enumerated, including adjacent and multi-lookback pairs.",
            "ALLOWED is a Day130B data/identity/fallback eligibility decision, not permission to run a leakage-sensitive backtest yet.",
            "Only safely auto-approved identity mappings may count as mapped evidence; manual-review mappings are never silently consumed.",
            "Partial identity coverage can be eligible only when complete target-row evidence proves explicit no-prior handling for players and explicit promotion/relegation fallback handling for teams.",
            "A full early-season backtest means the GW2-GW5 bridge and therefore requires target actual evidence through GW5.",
            "Day131A may audit identity continuity for pairs whose source priors and target canonical universe are structurally available even when Day130B blocks the backtest pending mapping evidence.",
        ],
    }


def write_json(report: Mapping[str, Any], out_json: Optional[str]) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print("saved_json:", path)


def write_csv(report: Mapping[str, Any], out_csv: Optional[str]) -> None:
    if not out_csv:
        return
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_season",
        "target_season",
        "pair_kind",
        "lookback_seasons",
        "pair_status",
        "pre_gw1_allowed",
        "early_season_allowed",
        "allowed_modes",
        "source_prior_available",
        "source_prior_kind",
        "target_canonical_universe_available",
        "target_actual_gws",
        "team_mapping_safe",
        "player_mapping_safe",
        "team_fallback_proven",
        "player_fallback_proven",
        "day131_identity_audit_ready",
        "blockers",
    ]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["pairs"]:
            identity = row["identity_and_fallback"]
            writer.writerow(
                {
                    "source_season": row["source_season"],
                    "target_season": row["target_season"],
                    "pair_kind": row["pair_kind"],
                    "lookback_seasons": row["lookback_seasons"],
                    "pair_status": row["pair_status"],
                    "pre_gw1_allowed": row["mode_results"][PRE_GW1_BACKTEST]["allowed"],
                    "early_season_allowed": row["mode_results"][EARLY_SEASON_BACKTEST]["allowed"],
                    "allowed_modes": "|".join(row["allowed_modes"]),
                    "source_prior_available": row["source_prior_availability"]["available"],
                    "source_prior_kind": row["source_prior_availability"]["source_kind"],
                    "target_canonical_universe_available": row["target_availability"]["canonical_universe_available"],
                    "target_actual_gws": "|".join(
                        str(item) for item in row["target_availability"]["actual_gws"]
                    ),
                    "team_mapping_safe": identity["team_mapping_safe"],
                    "player_mapping_safe": identity["player_mapping_safe"],
                    "team_fallback_proven": identity["team_partial_coverage_safe_via_fallback"],
                    "player_fallback_proven": identity["player_partial_coverage_safe_via_fallback"],
                    "day131_identity_audit_ready": row["data_ready_for_day131_identity_audit"],
                    "blockers": " | ".join(row["blockers"]),
                }
            )
    print("saved_csv:", path)


def write_markdown(report: Mapping[str, Any], out_md: Optional[str]) -> None:
    if not out_md:
        return
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Safe Season-Pair Eligibility Matrix")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- eligibility_version: `%s`" % report["eligibility_version"])
    lines.append("- Day130A inventory version: `%s`" % report.get("day130a_inventory_version"))
    lines.append("- seasons: `%s`" % report["season_count"])
    lines.append("- pairs: `%s`" % report["pair_count"])
    lines.append("- adjacent pairs: `%s`" % report["adjacent_pair_count"])
    lines.append("- multi-lookback pairs: `%s`" % report["multi_lookback_pair_count"])
    lines.append("- Pre-GW1 allowed pairs: `%s`" % report["allowed_pair_counts"][PRE_GW1_BACKTEST])
    lines.append("- Early-season GW2-GW5 allowed pairs: `%s`" % report["allowed_pair_counts"][EARLY_SEASON_BACKTEST])
    lines.append("")
    lines.append(
        "> `ALLOWED` here means Day130B data/identity/fallback eligibility only. "
        "It does not bypass the later as-of snapshot or leakage-validation gates."
    )
    lines.append("")
    lines.append("## Matrix")
    lines.append("")
    lines.append("| Source | Target | Kind | Lookback | Pre-GW1 | Early GW2-GW5 | Pair status |")
    lines.append("|---|---|---|---:|---|---|---|")
    for row in report["pairs"]:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["source_season"],
                row["target_season"],
                row["pair_kind"],
                row["lookback_seasons"],
                row["mode_results"][PRE_GW1_BACKTEST]["status"],
                row["mode_results"][EARLY_SEASON_BACKTEST]["status"],
                row["pair_status"],
            )
        )
    lines.append("")

    lines.append("## Allowed Pairs")
    lines.append("")
    for mode in BACKTEST_MODES:
        lines.append("### `%s`" % mode)
        allowed = report["allowed_pairs"][mode]
        if allowed:
            for item in allowed:
                lines.append("- `%s`" % item)
        else:
            lines.append("- None")
        lines.append("")

    lines.append("## Pair Details")
    lines.append("")
    for row in report["pairs"]:
        lines.append(
            "### `%s -> %s`"
            % (row["source_season"], row["target_season"])
        )
        lines.append("")
        lines.append("- pair_kind: `%s`" % row["pair_kind"])
        lines.append("- lookback_seasons: `%s`" % row["lookback_seasons"])
        lines.append("- pair_status: `%s`" % row["pair_status"])
        lines.append(
            "- source prior: `%s` (%s)"
            % (
                row["source_prior_availability"]["available"],
                row["source_prior_availability"]["source_kind"],
            )
        )
        lines.append(
            "- target actual GWs: `%s`"
            % ", ".join(
                str(item) for item in row["target_availability"]["actual_gws"]
            )
        )
        identity = row["identity_and_fallback"]
        lines.append("- team_mapping_safe: `%s`" % identity["team_mapping_safe"])
        lines.append(
            "- team mapping coverage: `%s/%s` (%s), transition gaps=%s"
            % (
                identity["team_mapping_auto_approved_count"],
                identity["team_mapping_top_row_count"],
                identity["team_mapping_coverage_rate"],
                identity["team_transition_gap_count"],
            )
        )
        lines.append("- player_mapping_safe: `%s`" % identity["player_mapping_safe"])
        lines.append(
            "- player mapping coverage: `%s/%s` (%s)"
            % (
                identity["player_mapping_auto_approved_count"],
                identity["player_mapping_top_row_count"],
                identity["player_mapping_coverage_rate"],
            )
        )
        lines.append(
            "- prior availability: player=`%s`, team=`%s`"
            % (
                identity["player_prior_available"],
                identity["team_prior_available"],
            )
        )
        lines.append(
            "- team_partial_coverage_safe_via_fallback: `%s`"
            % identity["team_partial_coverage_safe_via_fallback"]
        )
        lines.append(
            "- player_partial_coverage_safe_via_fallback: `%s`"
            % identity["player_partial_coverage_safe_via_fallback"]
        )
        lines.append(
            "- Day131A identity-audit-ready: `%s`"
            % row["data_ready_for_day131_identity_audit"]
        )
        lines.append("- allowed_modes: `%s`" % (
            ", ".join(row["allowed_modes"]) if row["allowed_modes"] else "none"
        ))
        if row["blockers"]:
            lines.append("- blockers:")
            for blocker in row["blockers"]:
                lines.append("  - %s" % blocker)
        else:
            lines.append("- blockers: none")
        lines.append("")

    lines.append("## Safety Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append("- %s" % note)
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print("saved_md:", path)


def print_summary(report: Mapping[str, Any]) -> None:
    print("=== Safe Season-Pair Eligibility Matrix ===")
    print("eligibility_version:", report["eligibility_version"])
    print("seasons:", report["season_count"])
    print("pairs:", report["pair_count"])
    print("adjacent_pairs:", report["adjacent_pair_count"])
    print("multi_lookback_pairs:", report["multi_lookback_pair_count"])
    print(
        "pre_gw1_allowed:",
        report["allowed_pair_counts"][PRE_GW1_BACKTEST],
    )
    print(
        "early_season_allowed:",
        report["allowed_pair_counts"][EARLY_SEASON_BACKTEST],
    )
    print(
        "blocked_pending_identity:",
        report["pair_status_counts"]["BLOCKED_PENDING_IDENTITY"],
    )
    print(
        "blocked_data:",
        report["pair_status_counts"]["BLOCKED_DATA"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a read-only source-season -> target-season eligibility matrix "
            "for Pre-GW1 and GW2-GW5 early-season historical backtests."
        )
    )
    parser.add_argument(
        "--inventory-json",
        default=None,
        help=(
            "Day130A historical season inventory JSON. If omitted, use "
            "../private-planning/historical-audits/day130a/historical_season_inventory.json "
            "relative to the repository root."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help=(
            "Optional root containing pair-specific historical prior/mapping artifacts. "
            "May be repeated. Defaults to ../datasets/prepared-fpl-historical."
        ),
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-csv", default=None)
    return parser.parse_args()


def resolve_repo_root() -> Path:
    path = Path(__file__).resolve()
    return path.parents[3]


def load_inventory(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("Day130A inventory not found: %s" % path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report.get("seasons"), Mapping):
        raise ValueError("Day130A inventory is missing seasons mapping.")
    return report


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    project_root = repo_root.parent

    inventory_path = (
        Path(args.inventory_json)
        if args.inventory_json
        else project_root / DEFAULT_DAY130A_RELATIVE
    )
    artifact_roots = (
        [Path(value) for value in args.artifact_root]
        if args.artifact_root
        else [project_root / DEFAULT_ARTIFACT_SUBDIR]
    )

    inventory = load_inventory(inventory_path)
    seasons = chronological_seasons(inventory)
    artifacts = discover_pair_artifacts(artifact_roots, seasons)
    report = build_report(inventory, artifacts)
    report["inventory_path"] = str(inventory_path)
    report["artifact_roots"] = [str(path) for path in artifact_roots]

    print_summary(report)
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    write_csv(report, args.out_csv)


if __name__ == "__main__":
    main()
