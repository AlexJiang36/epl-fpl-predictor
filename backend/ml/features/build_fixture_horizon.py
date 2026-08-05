from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


HORIZON_VERSION = "day79b_v1"
HORIZON_SCHEMA_VERSION = "fpl_fixture_horizon_v1"
ARTIFACT_TYPE = "fixture_horizon"
DEFAULT_TARGET_SEASON = "2026_27"
DEFAULT_START_GW = 1
DEFAULT_HORIZON = 5
DEFAULT_ARTIFACT_ROOT = "/private/tmp/fpl-artifacts"
PREDICTION_MODE = "pre_gw1_prior"

ROLLOVER_FILENAMES = {
    "fixture_scope": "gw1_gw5_fixtures.csv",
    "current_player_pool": "current_player_pool.csv",
    "team_identity_mapping": "team_identity_mapping.csv",
}

FIXTURE_REQUIRED_COLUMNS = (
    "target_season",
    "gameweek",
    "fixture_id",
    "kickoff_time_utc",
    "home_team_id",
    "home_team_name",
    "home_team_short_name",
    "away_team_id",
    "away_team_name",
    "away_team_short_name",
    "started",
    "finished",
)

PLAYER_REQUIRED_COLUMNS = (
    "target_season",
    "target_player_id",
    "target_player_code",
    "target_player_name",
    "target_web_name",
    "target_team_id",
    "target_team_name",
    "target_team_short_name",
    "target_position",
    "target_price_units",
    "target_price",
    "target_status",
    "current_selection_eligible",
)

TEAM_MAPPING_REQUIRED_COLUMNS = (
    "target_season",
    "target_team_id",
    "target_team_name",
    "target_team_short_name",
)


class FixtureHorizonInputError(RuntimeError):
    """Raised when Day79B cannot produce a safe fixture horizon."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any, label: str) -> datetime:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        raise FixtureHorizonInputError("%s is required." % label)
    normalized = str(value).strip()
    if not normalized:
        raise FixtureHorizonInputError("%s is required." % label)
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise FixtureHorizonInputError(
            "%s must be a valid ISO-8601 timestamp." % label
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FixtureHorizonInputError(
            "%s must include timezone information." % label
        )
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_optional_utc(value: Any) -> Tuple[Optional[str], str]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None, "missing"
    text_value = str(value).strip()
    if not text_value:
        return None, "missing"
    try:
        return format_utc(parse_utc(text_value, "kickoff_time_utc")), "known"
    except FixtureHorizonInputError:
        return text_value, "invalid"


def nullable_int(value: Any) -> Optional[int]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def nullable_float(value: Any) -> Optional[float]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def text_value(value: Any) -> Optional[str]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    result = str(value).strip()
    return result or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FixtureHorizonInputError("JSON file does not exist: %s" % path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FixtureHorizonInputError("JSON root must be an object: %s" % path)
    return value


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FixtureHorizonInputError("%s CSV does not exist: %s" % (label, path))
    return pd.read_csv(path, low_memory=False)


def require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise FixtureHorizonInputError(
            "%s is missing required columns: %s"
            % (label, ", ".join(missing))
        )


def validate_requested_scope(
    target_season: str,
    start_gw: int,
    horizon: int,
) -> Tuple[int, int]:
    if not isinstance(target_season, str) or not target_season.strip():
        raise FixtureHorizonInputError("target_season is required.")
    if isinstance(start_gw, bool) or start_gw < 1 or start_gw > 38:
        raise FixtureHorizonInputError("start_gw must be between 1 and 38.")
    if isinstance(horizon, bool) or horizon < 1:
        raise FixtureHorizonInputError("horizon must be at least 1.")
    end_gw = start_gw + horizon - 1
    if end_gw > 38:
        raise FixtureHorizonInputError(
            "start_gw + horizon - 1 must not exceed Gameweek 38."
        )
    return start_gw, end_gw


def validate_rollover_report(
    report: Mapping[str, Any],
    target_season: Optional[str] = None,
) -> Dict[str, Any]:
    blockers: List[str] = []
    actual_target = str(report.get("target_season") or "")

    if report.get("passed") is not True:
        blockers.append("Day76C rollover report must have passed=true.")
    if report.get("audit_only") is not True:
        blockers.append("Day76C rollover report must have audit_only=true.")
    if report.get("writes_database") is not False:
        blockers.append("Day76C rollover report must have writes_database=false.")
    if report.get("blockers"):
        blockers.append("Day76C rollover report contains blockers.")

    readiness = report.get("readiness") or {}
    if not isinstance(readiness, Mapping):
        blockers.append("Day76C readiness must be a mapping.")
        readiness = {}
    for key in (
        "current_player_pool_validated",
        "target_team_transition_validated",
        "gw1_gw5_fixture_scope_validated",
    ):
        if readiness.get(key) is not True:
            blockers.append("Day76C readiness.%s must be true." % key)
    if readiness.get("ready_for_prediction_write") is not False:
        blockers.append("Day76C ready_for_prediction_write must remain false.")

    if target_season is not None and actual_target != target_season:
        blockers.append("target_season does not match Day76C rollover report.")

    try:
        as_of_time = format_utc(
            parse_utc(report.get("as_of_time_utc"), "as_of_time_utc")
        )
    except FixtureHorizonInputError as exc:
        blockers.append(str(exc))
        as_of_time = ""

    if blockers:
        raise FixtureHorizonInputError(" ".join(blockers))

    parent_run_id = (
        (report.get("run_metadata") or {}).get("run_id")
        if isinstance(report.get("run_metadata") or {}, Mapping)
        else None
    )
    return {
        "target_season": actual_target,
        "as_of_time_utc": as_of_time,
        "parent_run_id": text_value(parent_run_id),
    }


def resolve_rollover_inputs(report_path: Path) -> Dict[str, Path]:
    run_dir = report_path.expanduser().resolve().parent
    result = {"rollover_report": report_path.expanduser().resolve()}
    for key, filename in ROLLOVER_FILENAMES.items():
        path = run_dir / filename
        if not path.is_file():
            raise FixtureHorizonInputError("Missing Day76C artifact: %s" % path)
        result[key] = path
    return result


def normalize_target_teams(
    team_mapping: pd.DataFrame,
    target_season: str,
    expected_team_count: int = 20,
) -> pd.DataFrame:
    require_columns(
        team_mapping,
        TEAM_MAPPING_REQUIRED_COLUMNS,
        "Day76C team identity mapping",
    )
    result = team_mapping[
        team_mapping["target_season"].astype(str) == str(target_season)
    ].copy()
    result["team_id"] = result["target_team_id"].apply(nullable_int)
    result["team_name"] = result["target_team_name"].map(text_value)
    result["team_short_name"] = result["target_team_short_name"].map(text_value)
    result = result[result["team_id"].notna()].copy()
    result = result[["team_id", "team_name", "team_short_name"]]

    conflicts = (
        result.groupby("team_id")[["team_name", "team_short_name"]]
        .nunique(dropna=False)
        .max(axis=1)
    )
    if int((conflicts > 1).sum()):
        raise FixtureHorizonInputError(
            "Day76C team mapping contains conflicting target-team identities."
        )
    result = result.drop_duplicates(subset=["team_id"]).sort_values("team_id")
    if len(result) != expected_team_count:
        raise FixtureHorizonInputError(
            "Day79B requires exactly %s target teams; found %s."
            % (expected_team_count, len(result))
        )
    if result["team_name"].isna().any() or result["team_short_name"].isna().any():
        raise FixtureHorizonInputError(
            "Target-team identities must include names and short names."
        )
    return result.reset_index(drop=True)


def normalize_current_players(
    current_players: pd.DataFrame,
    target_season: str,
    valid_team_ids: Sequence[int],
) -> pd.DataFrame:
    require_columns(
        current_players,
        PLAYER_REQUIRED_COLUMNS,
        "Day76C current player pool",
    )
    result = current_players[
        current_players["target_season"].astype(str) == str(target_season)
    ].copy()
    result["player_id"] = result["target_player_id"].apply(nullable_int)
    result["player_code"] = result["target_player_code"].apply(nullable_int)
    result["player_name"] = result["target_player_name"].map(text_value)
    result["web_name"] = result["target_web_name"].map(text_value)
    result["team_id"] = result["target_team_id"].apply(nullable_int)
    result["team_name"] = result["target_team_name"].map(text_value)
    result["team_short_name"] = result["target_team_short_name"].map(text_value)
    result["position"] = result["target_position"].astype(str).str.upper()
    result["price_units"] = result["target_price_units"].apply(nullable_int)
    result["price"] = pd.to_numeric(result["target_price"], errors="coerce")
    result["player_status"] = result["target_status"].fillna("").astype(str)
    result["selection_eligible"] = result[
        "current_selection_eligible"
    ].apply(bool_value)

    if result.empty:
        raise FixtureHorizonInputError("Current target player pool is empty.")
    if result["player_id"].isna().any():
        raise FixtureHorizonInputError("Current player pool has missing player IDs.")
    if result["player_id"].duplicated().any():
        raise FixtureHorizonInputError("Current player pool has duplicate player IDs.")
    valid_ids = set(int(value) for value in valid_team_ids)
    orphan_count = int((~result["team_id"].isin(valid_ids)).sum())
    if orphan_count:
        raise FixtureHorizonInputError(
            "Current player pool has %s player row(s) with unknown team IDs."
            % orphan_count
        )

    return result[
        [
            "player_id",
            "player_code",
            "player_name",
            "web_name",
            "team_id",
            "team_name",
            "team_short_name",
            "position",
            "price_units",
            "price",
            "player_status",
            "selection_eligible",
        ]
    ].sort_values("player_id").reset_index(drop=True)


def normalize_fixture_scope(
    fixtures: pd.DataFrame,
    target_season: str,
    start_gw: int,
    horizon: int,
    valid_team_ids: Sequence[int],
) -> pd.DataFrame:
    require_columns(fixtures, FIXTURE_REQUIRED_COLUMNS, "Day76C fixture scope")
    _, end_gw = validate_requested_scope(target_season, start_gw, horizon)
    result = fixtures[
        (fixtures["target_season"].astype(str) == str(target_season))
        & (pd.to_numeric(fixtures["gameweek"], errors="coerce") >= start_gw)
        & (pd.to_numeric(fixtures["gameweek"], errors="coerce") <= end_gw)
    ].copy()
    if result.empty:
        raise FixtureHorizonInputError(
            "No fixtures found for %s GW%s-GW%s."
            % (target_season, start_gw, end_gw)
        )

    result["gameweek"] = result["gameweek"].apply(nullable_int)
    result["fixture_id"] = result["fixture_id"].apply(nullable_int)
    result["home_team_id"] = result["home_team_id"].apply(nullable_int)
    result["away_team_id"] = result["away_team_id"].apply(nullable_int)
    result["home_team_name"] = result["home_team_name"].map(text_value)
    result["home_team_short_name"] = result["home_team_short_name"].map(text_value)
    result["away_team_name"] = result["away_team_name"].map(text_value)
    result["away_team_short_name"] = result["away_team_short_name"].map(text_value)
    result["started"] = result["started"].apply(bool_value)
    result["finished"] = result["finished"].apply(bool_value)

    normalized_kickoffs: List[Optional[str]] = []
    kickoff_statuses: List[str] = []
    for value in result["kickoff_time_utc"].tolist():
        normalized, status = normalize_optional_utc(value)
        normalized_kickoffs.append(normalized)
        kickoff_statuses.append(status)
    result["kickoff_time_utc"] = normalized_kickoffs
    result["kickoff_time_status"] = kickoff_statuses
    result["kickoff_time_known"] = result["kickoff_time_status"] == "known"
    result["kickoff_time_valid"] = result["kickoff_time_status"].isin(
        ["known", "missing"]
    )

    result["duplicate_fixture_id_flag"] = result["fixture_id"].duplicated(
        keep=False
    )
    natural_key = ["target_season", "gameweek", "home_team_id", "away_team_id"]
    result["duplicate_natural_key_flag"] = result.duplicated(
        subset=natural_key,
        keep=False,
    )
    pairing_values = result.apply(
        lambda row: (
            str(row["target_season"]),
            row["gameweek"],
            min(row["home_team_id"], row["away_team_id"])
            if row["home_team_id"] is not None and row["away_team_id"] is not None
            else None,
            max(row["home_team_id"], row["away_team_id"])
            if row["home_team_id"] is not None and row["away_team_id"] is not None
            else None,
        ),
        axis=1,
    )
    result["_pairing_key"] = pairing_values
    result["duplicate_pairing_flag"] = result["_pairing_key"].duplicated(
        keep=False
    )

    valid_ids = set(int(value) for value in valid_team_ids)
    result["missing_fixture_identity_flag"] = (
        result["fixture_id"].isna()
        | result["gameweek"].isna()
        | result["home_team_id"].isna()
        | result["away_team_id"].isna()
    )
    result["unknown_team_flag"] = (
        ~result["home_team_id"].isin(valid_ids)
        | ~result["away_team_id"].isin(valid_ids)
    )
    result["self_fixture_flag"] = (
        result["home_team_id"].notna()
        & (result["home_team_id"] == result["away_team_id"])
    )
    result["fixture_identity_valid"] = ~(
        result["missing_fixture_identity_flag"]
        | result["unknown_team_flag"]
        | result["self_fixture_flag"]
        | result["duplicate_fixture_id_flag"]
        | result["duplicate_natural_key_flag"]
        | result["duplicate_pairing_flag"]
    )
    result = result.drop(columns=["_pairing_key"])
    return result.sort_values(
        ["gameweek", "kickoff_time_known", "kickoff_time_utc", "fixture_id"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def load_previous_fixture_horizon(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    previous = load_csv(path, "previous fixture horizon")
    required = ("fixture_id", "kickoff_time_utc")
    require_columns(previous, required, "previous fixture horizon")
    result = previous.copy()
    result["fixture_id"] = result["fixture_id"].apply(nullable_int)
    if result["fixture_id"].isna().any():
        raise FixtureHorizonInputError(
            "Previous fixture horizon contains missing fixture IDs."
        )
    if result["fixture_id"].duplicated().any():
        raise FixtureHorizonInputError(
            "Previous fixture horizon contains duplicate fixture IDs."
        )
    normalized_values: List[Optional[str]] = []
    statuses: List[str] = []
    for value in result["kickoff_time_utc"].tolist():
        normalized, status = normalize_optional_utc(value)
        normalized_values.append(normalized)
        statuses.append(status)
    result["previous_kickoff_time_utc"] = normalized_values
    result["previous_kickoff_status"] = statuses
    keep = ["fixture_id", "previous_kickoff_time_utc", "previous_kickoff_status"]
    for column in (
        "target_season",
        "gameweek",
        "home_team_id",
        "away_team_id",
        "home_team_short_name",
        "away_team_short_name",
    ):
        if column in result.columns:
            keep.append(column)
    return result[keep].copy()


def annotate_kickoff_changes(
    fixtures: pd.DataFrame,
    previous: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    result = fixtures.copy()
    comparison_available = not previous.empty
    if not comparison_available:
        result["previous_kickoff_time_utc"] = None
        result["previous_kickoff_status"] = "not_compared"
        result["kickoff_change_status"] = "baseline_no_previous_comparison"
        change_log = result[
            [
                "target_season",
                "gameweek",
                "fixture_id",
                "home_team_id",
                "home_team_short_name",
                "away_team_id",
                "away_team_short_name",
                "previous_kickoff_time_utc",
                "kickoff_time_utc",
                "kickoff_change_status",
            ]
        ].copy()
        return result, change_log, {
            "comparison_available": False,
            "changed_fixture_count": 0,
            "newly_known_fixture_count": 0,
            "missing_current_fixture_count": 0,
            "new_fixture_count": 0,
            "removed_fixture_count": 0,
            "removed_fixture_ids": [],
        }

    previous_lookup = previous.set_index("fixture_id").to_dict(orient="index")
    current_ids = set(
        int(value) for value in result["fixture_id"].dropna().astype(int).tolist()
    )
    previous_ids = set(int(value) for value in previous["fixture_id"].tolist())
    statuses: List[str] = []
    previous_values: List[Optional[str]] = []
    previous_statuses: List[str] = []
    for row in result.itertuples(index=False):
        fixture_id = int(row.fixture_id) if row.fixture_id is not None else None
        previous_row = previous_lookup.get(fixture_id)
        if previous_row is None:
            previous_values.append(None)
            previous_statuses.append("not_found")
            statuses.append("new_fixture")
            continue
        old_value = previous_row.get("previous_kickoff_time_utc")
        old_status = str(previous_row.get("previous_kickoff_status") or "missing")
        previous_values.append(old_value)
        previous_statuses.append(old_status)
        new_status = str(row.kickoff_time_status)
        new_value = row.kickoff_time_utc
        if new_status == "invalid":
            statuses.append("invalid_current")
        elif new_status == "missing" and old_status == "known":
            statuses.append("missing_current")
        elif new_status == "known" and old_status in {"missing", "invalid"}:
            statuses.append("newly_known")
        elif new_status == "known" and old_status == "known":
            statuses.append("unchanged" if new_value == old_value else "changed")
        elif new_status == old_status:
            statuses.append("unchanged")
        else:
            statuses.append("status_changed")

    result["previous_kickoff_time_utc"] = previous_values
    result["previous_kickoff_status"] = previous_statuses
    result["kickoff_change_status"] = statuses

    change_log = result[
        [
            "target_season",
            "gameweek",
            "fixture_id",
            "home_team_id",
            "home_team_short_name",
            "away_team_id",
            "away_team_short_name",
            "previous_kickoff_time_utc",
            "kickoff_time_utc",
            "kickoff_change_status",
        ]
    ].copy()
    removed_ids = sorted(previous_ids - current_ids)
    if removed_ids:
        removed_rows: List[Dict[str, Any]] = []
        previous_by_id = previous.set_index("fixture_id")
        for fixture_id in removed_ids:
            raw = previous_by_id.loc[fixture_id].to_dict()
            removed_rows.append(
                {
                    "target_season": raw.get("target_season"),
                    "gameweek": raw.get("gameweek"),
                    "fixture_id": fixture_id,
                    "home_team_id": raw.get("home_team_id"),
                    "home_team_short_name": raw.get("home_team_short_name"),
                    "away_team_id": raw.get("away_team_id"),
                    "away_team_short_name": raw.get("away_team_short_name"),
                    "previous_kickoff_time_utc": raw.get(
                        "previous_kickoff_time_utc"
                    ),
                    "kickoff_time_utc": None,
                    "kickoff_change_status": "removed_from_current_horizon",
                }
            )
        change_log = pd.concat(
            [change_log, pd.DataFrame(removed_rows)],
            ignore_index=True,
        )

    return result, change_log, {
        "comparison_available": True,
        "changed_fixture_count": int(
            (result["kickoff_change_status"] == "changed").sum()
        ),
        "newly_known_fixture_count": int(
            (result["kickoff_change_status"] == "newly_known").sum()
        ),
        "missing_current_fixture_count": int(
            (result["kickoff_change_status"] == "missing_current").sum()
        ),
        "new_fixture_count": int(
            (result["kickoff_change_status"] == "new_fixture").sum()
        ),
        "removed_fixture_count": len(removed_ids),
        "removed_fixture_ids": removed_ids,
    }


def fixture_validation(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    start_gw: int,
    horizon: int,
    change_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    _, end_gw = validate_requested_scope(
        str(fixtures["target_season"].iloc[0]), start_gw, horizon
    )
    blockers: List[str] = []
    warnings: List[str] = []

    counts = {
        "missing_fixture_identity_count": int(
            fixtures["missing_fixture_identity_flag"].sum()
        ),
        "unknown_team_reference_count": int(fixtures["unknown_team_flag"].sum()),
        "self_fixture_count": int(fixtures["self_fixture_flag"].sum()),
        "duplicate_fixture_id_row_count": int(
            fixtures["duplicate_fixture_id_flag"].sum()
        ),
        "duplicate_natural_key_row_count": int(
            fixtures["duplicate_natural_key_flag"].sum()
        ),
        "duplicate_pairing_row_count": int(
            fixtures["duplicate_pairing_flag"].sum()
        ),
        "missing_kickoff_count": int(
            (fixtures["kickoff_time_status"] == "missing").sum()
        ),
        "invalid_kickoff_count": int(
            (fixtures["kickoff_time_status"] == "invalid").sum()
        ),
        "started_fixture_count": int(fixtures["started"].sum()),
        "finished_fixture_count": int(fixtures["finished"].sum()),
    }

    blocker_messages = (
        ("missing_fixture_identity_count", "Fixture rows contain missing identities."),
        ("unknown_team_reference_count", "Fixture rows reference unknown teams."),
        ("self_fixture_count", "Fixture rows contain self-fixtures."),
        ("duplicate_fixture_id_row_count", "Duplicate fixture IDs were detected."),
        (
            "duplicate_natural_key_row_count",
            "Duplicate season/GW/home/away fixture rows were detected.",
        ),
        (
            "duplicate_pairing_row_count",
            "Duplicate team pairings were detected within a Gameweek.",
        ),
    )
    for key, message in blocker_messages:
        if counts[key]:
            blockers.append("%s count=%s" % (message, counts[key]))

    gameweeks_present = sorted(
        int(value) for value in fixtures["gameweek"].dropna().unique().tolist()
    )
    missing_gameweeks = [
        gameweek
        for gameweek in range(start_gw, end_gw + 1)
        if gameweek not in gameweeks_present
    ]
    if missing_gameweeks:
        blockers.append(
            "Requested Gameweeks have no fixture rows: %s."
            % ", ".join(str(value) for value in missing_gameweeks)
        )

    if counts["missing_kickoff_count"]:
        warnings.append(
            "%s fixture row(s) have missing kickoff times."
            % counts["missing_kickoff_count"]
        )
    if counts["invalid_kickoff_count"]:
        warnings.append(
            "%s fixture row(s) have invalid kickoff times."
            % counts["invalid_kickoff_count"]
        )
    if change_summary.get("changed_fixture_count"):
        warnings.append(
            "%s fixture kickoff time(s) changed from the previous horizon."
            % change_summary["changed_fixture_count"]
        )
    if change_summary.get("newly_known_fixture_count"):
        warnings.append(
            "%s fixture kickoff time(s) became known."
            % change_summary["newly_known_fixture_count"]
        )
    if change_summary.get("missing_current_fixture_count"):
        warnings.append(
            "%s fixture kickoff time(s) became missing."
            % change_summary["missing_current_fixture_count"]
        )
    if change_summary.get("removed_fixture_count"):
        warnings.append(
            "%s previous fixture row(s) are absent from the current horizon."
            % change_summary["removed_fixture_count"]
        )
    if counts["started_fixture_count"] or counts["finished_fixture_count"]:
        warnings.append(
            "The requested horizon contains started or finished fixture rows."
        )
    if not change_summary.get("comparison_available"):
        warnings.append(
            "No previous fixture horizon was supplied; kickoff-change status is a baseline."
        )

    return {
        "passed": not blockers,
        "fixture_count": int(len(fixtures)),
        "target_team_count": int(len(teams)),
        "gameweeks_present": gameweeks_present,
        "missing_gameweeks": missing_gameweeks,
        **counts,
        "blockers": blockers,
        "warnings": warnings,
    }


def build_team_fixture_horizon(fixtures: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fixture in fixtures.to_dict(orient="records"):
        common = {
            "target_season": fixture["target_season"],
            "gameweek": fixture["gameweek"],
            "fixture_id": fixture["fixture_id"],
            "kickoff_time_utc": fixture["kickoff_time_utc"],
            "kickoff_time_status": fixture["kickoff_time_status"],
            "kickoff_time_known": fixture["kickoff_time_known"],
            "previous_kickoff_time_utc": fixture.get(
                "previous_kickoff_time_utc"
            ),
            "kickoff_change_status": fixture.get("kickoff_change_status"),
            "started": fixture["started"],
            "finished": fixture["finished"],
            "fixture_identity_valid": fixture["fixture_identity_valid"],
        }
        rows.append(
            {
                **common,
                "team_id": fixture["home_team_id"],
                "team_name": fixture["home_team_name"],
                "team_short_name": fixture["home_team_short_name"],
                "opponent_team_id": fixture["away_team_id"],
                "opponent_team_name": fixture["away_team_name"],
                "opponent_team_short_name": fixture["away_team_short_name"],
                "is_home": True,
            }
        )
        rows.append(
            {
                **common,
                "team_id": fixture["away_team_id"],
                "team_name": fixture["away_team_name"],
                "team_short_name": fixture["away_team_short_name"],
                "opponent_team_id": fixture["home_team_id"],
                "opponent_team_name": fixture["home_team_name"],
                "opponent_team_short_name": fixture["home_team_short_name"],
                "is_home": False,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["team_id", "gameweek", "kickoff_time_known", "kickoff_time_utc", "fixture_id"],
        ascending=[True, True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    result["fixture_sequence_for_team_gw"] = (
        result.groupby(["team_id", "gameweek"]).cumcount() + 1
    )
    result["fixture_count_for_team_gw"] = result.groupby(
        ["team_id", "gameweek"]
    )["fixture_id"].transform("size")
    result["blank_gw_flag"] = False
    result["double_gw_flag"] = result["fixture_count_for_team_gw"] > 1
    result["team_gameweek_status"] = result["fixture_count_for_team_gw"].apply(
        lambda count: "double" if int(count) > 1 else "single"
    )
    result["manual_review_required"] = (
        result["kickoff_time_status"] != "known"
    ) | result["kickoff_change_status"].isin(
        [
            "changed",
            "newly_known",
            "missing_current",
            "invalid_current",
            "status_changed",
            "new_fixture",
        ]
    )
    return result


def _json_list(values: Iterable[Any]) -> str:
    cleaned: List[Any] = []
    for value in values:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            continue
        if isinstance(value, (bool, int, float, str)):
            cleaned.append(value)
        else:
            cleaned.append(str(value))
    return json.dumps(cleaned, ensure_ascii=False)


def build_team_gameweek_horizon(
    teams: pd.DataFrame,
    team_fixtures: pd.DataFrame,
    start_gw: int,
    horizon: int,
) -> pd.DataFrame:
    _, end_gw = validate_requested_scope(
        DEFAULT_TARGET_SEASON, start_gw, horizon
    )
    fixture_groups: Dict[Tuple[int, int], pd.DataFrame] = {
        (int(team_id), int(gameweek)): group.copy()
        for (team_id, gameweek), group in team_fixtures.groupby(
            ["team_id", "gameweek"]
        )
    }
    rows: List[Dict[str, Any]] = []
    for team in teams.to_dict(orient="records"):
        team_id = int(team["team_id"])
        for gameweek in range(start_gw, end_gw + 1):
            group = fixture_groups.get((team_id, gameweek), pd.DataFrame())
            fixture_count = int(len(group))
            blank = fixture_count == 0
            double = fixture_count > 1
            rows.append(
                {
                    "target_season": (
                        str(team_fixtures["target_season"].iloc[0])
                        if not team_fixtures.empty
                        else None
                    ),
                    "gameweek": gameweek,
                    "team_id": team_id,
                    "team_name": team["team_name"],
                    "team_short_name": team["team_short_name"],
                    "fixture_count": fixture_count,
                    "has_fixture": fixture_count > 0,
                    "blank_gw_flag": blank,
                    "double_gw_flag": double,
                    "team_gameweek_status": (
                        "blank" if blank else "double" if double else "single"
                    ),
                    "fixture_ids": _json_list(
                        group.get("fixture_id", pd.Series(dtype=object)).tolist()
                    ),
                    "opponent_team_ids": _json_list(
                        group.get(
                            "opponent_team_id", pd.Series(dtype=object)
                        ).tolist()
                    ),
                    "opponent_team_short_names": _json_list(
                        group.get(
                            "opponent_team_short_name", pd.Series(dtype=object)
                        ).tolist()
                    ),
                    "is_home_values": _json_list(
                        group.get("is_home", pd.Series(dtype=object)).tolist()
                    ),
                    "kickoff_times_utc": _json_list(
                        group.get(
                            "kickoff_time_utc", pd.Series(dtype=object)
                        ).tolist()
                    ),
                    "kickoff_known_count": int(
                        group.get(
                            "kickoff_time_known", pd.Series(dtype=bool)
                        ).sum()
                    ),
                    "kickoff_missing_or_invalid_count": int(
                        (
                            group.get(
                                "kickoff_time_status",
                                pd.Series(dtype=object),
                            )
                            != "known"
                        ).sum()
                    ),
                    "kickoff_changed_or_review_count": int(
                        group.get(
                            "manual_review_required", pd.Series(dtype=bool)
                        ).sum()
                    ),
                    "fixture_identity_complete": bool(
                        group.get(
                            "fixture_identity_valid", pd.Series(dtype=bool)
                        ).all()
                    )
                    if fixture_count
                    else True,
                    "manual_review_required": bool(
                        group.get(
                            "manual_review_required", pd.Series(dtype=bool)
                        ).any()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["gameweek", "team_id"]
    ).reset_index(drop=True)


def _eligibility_reason(
    selection_eligible: bool,
    has_fixture: bool,
    fixture_identity_valid: bool,
) -> str:
    if not selection_eligible:
        return "current_player_ineligible"
    if not has_fixture:
        return "blank_gameweek"
    if not fixture_identity_valid:
        return "fixture_identity_invalid"
    return "eligible_current_player_and_fixture"


def build_player_fixture_eligibility(
    players: pd.DataFrame,
    team_fixtures: pd.DataFrame,
    team_gameweeks: pd.DataFrame,
    start_gw: int,
    horizon: int,
) -> pd.DataFrame:
    _, end_gw = validate_requested_scope(
        DEFAULT_TARGET_SEASON, start_gw, horizon
    )
    fixture_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {
        (int(team_id), int(gameweek)): group.to_dict(orient="records")
        for (team_id, gameweek), group in team_fixtures.groupby(
            ["team_id", "gameweek"]
        )
    }
    team_gw_lookup = {
        (int(row.team_id), int(row.gameweek)): row._asdict()
        for row in team_gameweeks.itertuples(index=False)
    }
    target_season = (
        str(team_fixtures["target_season"].iloc[0])
        if not team_fixtures.empty
        else None
    )
    rows: List[Dict[str, Any]] = []
    for player in players.to_dict(orient="records"):
        team_id = int(player["team_id"])
        for gameweek in range(start_gw, end_gw + 1):
            fixtures = fixture_groups.get((team_id, gameweek), [])
            summary = team_gw_lookup[(team_id, gameweek)]
            if not fixtures:
                fixtures = [
                    {
                        "fixture_id": None,
                        "fixture_sequence_for_team_gw": 0,
                        "opponent_team_id": None,
                        "opponent_team_name": None,
                        "opponent_team_short_name": None,
                        "is_home": None,
                        "kickoff_time_utc": None,
                        "kickoff_time_status": "not_applicable_blank",
                        "previous_kickoff_time_utc": None,
                        "kickoff_change_status": "not_applicable_blank",
                        "fixture_identity_valid": True,
                        "manual_review_required": False,
                        "started": False,
                        "finished": False,
                    }
                ]
            for fixture in fixtures:
                has_fixture = fixture["fixture_id"] is not None
                selection_eligible = bool(player["selection_eligible"])
                fixture_identity_valid = bool(
                    fixture.get("fixture_identity_valid", True)
                )
                player_fixture_eligible = bool(
                    selection_eligible and has_fixture and fixture_identity_valid
                )
                rows.append(
                    {
                        "target_season": target_season,
                        "gameweek": gameweek,
                        "player_id": player["player_id"],
                        "player_code": player["player_code"],
                        "player_name": player["player_name"],
                        "web_name": player["web_name"],
                        "team_id": team_id,
                        "team_name": player["team_name"],
                        "team_short_name": player["team_short_name"],
                        "position": player["position"],
                        "price_units": player["price_units"],
                        "price": player["price"],
                        "player_status": player["player_status"],
                        "selection_eligible": selection_eligible,
                        "fixture_id": fixture["fixture_id"],
                        "fixture_sequence_for_team_gw": fixture[
                            "fixture_sequence_for_team_gw"
                        ],
                        "fixture_count_for_team_gw": summary["fixture_count"],
                        "has_fixture": has_fixture,
                        "blank_gw_flag": summary["blank_gw_flag"],
                        "double_gw_flag": summary["double_gw_flag"],
                        "opponent_team_id": fixture.get("opponent_team_id"),
                        "opponent_team_name": fixture.get("opponent_team_name"),
                        "opponent_team_short_name": fixture.get(
                            "opponent_team_short_name"
                        ),
                        "is_home": fixture.get("is_home"),
                        "kickoff_time_utc": fixture.get("kickoff_time_utc"),
                        "kickoff_time_status": fixture.get(
                            "kickoff_time_status"
                        ),
                        "previous_kickoff_time_utc": fixture.get(
                            "previous_kickoff_time_utc"
                        ),
                        "kickoff_change_status": fixture.get(
                            "kickoff_change_status"
                        ),
                        "fixture_identity_valid": fixture_identity_valid,
                        "player_fixture_eligible": player_fixture_eligible,
                        "eligibility_reason": _eligibility_reason(
                            selection_eligible,
                            has_fixture,
                            fixture_identity_valid,
                        ),
                        "manual_review_required": bool(
                            fixture.get("manual_review_required", False)
                        ),
                        "started": bool(fixture.get("started", False)),
                        "finished": bool(fixture.get("finished", False)),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        [
            "gameweek",
            "player_id",
            "fixture_sequence_for_team_gw",
        ]
    ).reset_index(drop=True)


def build_gameweek_summary(
    fixtures: pd.DataFrame,
    team_gameweeks: pd.DataFrame,
    start_gw: int,
    horizon: int,
) -> Dict[str, Any]:
    _, end_gw = validate_requested_scope(
        str(fixtures["target_season"].iloc[0]), start_gw, horizon
    )
    result: Dict[str, Any] = {}
    for gameweek in range(start_gw, end_gw + 1):
        fixture_subset = fixtures[fixtures["gameweek"] == gameweek]
        team_subset = team_gameweeks[team_gameweeks["gameweek"] == gameweek]
        result[str(gameweek)] = {
            "fixture_count": int(len(fixture_subset)),
            "team_fixture_side_count": int(
                (team_subset["fixture_count"]).sum()
            ),
            "single_team_count": int(
                (team_subset["fixture_count"] == 1).sum()
            ),
            "blank_team_count": int(team_subset["blank_gw_flag"].sum()),
            "double_team_count": int(team_subset["double_gw_flag"].sum()),
            "missing_or_invalid_kickoff_count": int(
                (fixture_subset["kickoff_time_status"] != "known").sum()
            ),
            "kickoff_review_count": int(
                fixture_subset["kickoff_change_status"].isin(
                    [
                        "changed",
                        "newly_known",
                        "missing_current",
                        "invalid_current",
                        "status_changed",
                        "new_fixture",
                    ]
                ).sum()
            ),
        }
    return result


def analyze_horizon(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    players: pd.DataFrame,
    start_gw: int,
    horizon: int,
    change_summary: Mapping[str, Any],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    validation = fixture_validation(
        fixtures=fixtures,
        teams=teams,
        start_gw=start_gw,
        horizon=horizon,
        change_summary=change_summary,
    )
    team_fixtures = build_team_fixture_horizon(fixtures)
    team_gameweeks = build_team_gameweek_horizon(
        teams=teams,
        team_fixtures=team_fixtures,
        start_gw=start_gw,
        horizon=horizon,
    )
    player_context = build_player_fixture_eligibility(
        players=players,
        team_fixtures=team_fixtures,
        team_gameweeks=team_gameweeks,
        start_gw=start_gw,
        horizon=horizon,
    )

    expected_team_gameweeks = len(teams) * horizon
    observed_player_gameweeks = len(
        player_context[["player_id", "gameweek"]].drop_duplicates()
    )
    expected_player_gameweeks = len(players) * horizon
    blank_count = int(team_gameweeks["blank_gw_flag"].sum())
    double_count = int(team_gameweeks["double_gw_flag"].sum())
    if blank_count:
        validation["warnings"].append(
            "%s team-Gameweek blank(s) detected." % blank_count
        )
    if double_count:
        validation["warnings"].append(
            "%s team-Gameweek double(s) detected." % double_count
        )

    context_checks = {
        "team_fixture_row_count": int(len(team_fixtures)),
        "team_gameweek_row_count": int(len(team_gameweeks)),
        "expected_team_gameweek_row_count": int(expected_team_gameweeks),
        "player_fixture_context_row_count": int(len(player_context)),
        "player_gameweek_context_count": int(observed_player_gameweeks),
        "expected_player_gameweek_context_count": int(expected_player_gameweeks),
        "blank_team_gameweek_count": blank_count,
        "double_team_gameweek_count": double_count,
        "all_team_gameweeks_represented": len(team_gameweeks)
        == expected_team_gameweeks,
        "all_player_gameweeks_represented": observed_player_gameweeks
        == expected_player_gameweeks,
        "balanced_single_fixture_schedule": blank_count == 0
        and double_count == 0,
        "current_selection_eligible_player_count": int(
            players["selection_eligible"].sum()
        ),
        "current_selection_ineligible_player_count": int(
            (~players["selection_eligible"]).sum()
        ),
        "player_fixture_eligible_row_count": int(
            player_context["player_fixture_eligible"].sum()
        ),
        "player_blank_context_row_count": int(
            player_context["blank_gw_flag"].sum()
        ),
        "player_double_context_row_count": int(
            player_context["double_gw_flag"].sum()
        ),
    }
    if not context_checks["all_team_gameweeks_represented"]:
        validation["blockers"].append(
            "Not every team-Gameweek is represented in the horizon."
        )
    if not context_checks["all_player_gameweeks_represented"]:
        validation["blockers"].append(
            "Not every player-Gameweek is represented in eligibility context."
        )
    validation["passed"] = not validation["blockers"]
    validation["gameweeks"] = build_gameweek_summary(
        fixtures=fixtures,
        team_gameweeks=team_gameweeks,
        start_gw=start_gw,
        horizon=horizon,
    )
    validation["context"] = context_checks
    validation["horizon_complete_for_consumption"] = bool(
        validation["passed"]
        and context_checks["all_team_gameweeks_represented"]
        and context_checks["all_player_gameweeks_represented"]
    )

    return {
        "fixture_horizon": fixtures,
        "team_fixture_horizon": team_fixtures,
        "team_gameweek_horizon": team_gameweeks,
        "player_fixture_eligibility": player_context,
    }, validation


def add_run_fields(
    dataframe: pd.DataFrame,
    run_id: str,
    target_season: str,
    start_gw: int,
    horizon: int,
    as_of_time: str,
    parent_run_id: Optional[str],
) -> pd.DataFrame:
    result = dataframe.copy()
    fields = [
        ("run_id", run_id),
        ("artifact_type", ARTIFACT_TYPE),
        ("horizon_version", HORIZON_VERSION),
        ("horizon_schema_version", HORIZON_SCHEMA_VERSION),
        ("target_season", target_season),
        ("start_gw", start_gw),
        ("horizon", horizon),
        ("end_gw", start_gw + horizon - 1),
        ("as_of_time_utc", as_of_time),
        ("source_rollover_run_id", parent_run_id),
    ]
    for index, (column, value) in enumerate(fields):
        if column in result.columns:
            result[column] = value
        else:
            result.insert(index, column, value)
    return result


def provenance_metadata(paths: Mapping[str, Path]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for key, path in paths.items():
        if path.is_file():
            result[key] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return result


def build_markdown_report(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    context = validation["context"]
    lines = [
        "# Day79B Fixture Horizon Report",
        "",
        "## Scope",
        "",
        "```text",
        "target_season: %s" % report["target_season"],
        "start_gw: %s" % report["start_gw"],
        "end_gw: %s" % report["end_gw"],
        "horizon: %s" % report["horizon"],
        "as_of_time_utc: %s" % report["as_of_time_utc"],
        "run_id: %s" % report["run_metadata"]["run_id"],
        "preview_only: true",
        "writes_database: false",
        "```",
        "",
        "## Output row counts",
        "",
        "```text",
        "fixture_rows: %s" % validation["fixture_count"],
        "team_fixture_rows: %s" % context["team_fixture_row_count"],
        "team_gameweek_rows: %s" % context["team_gameweek_row_count"],
        "player_fixture_eligibility_rows: %s"
        % context["player_fixture_context_row_count"],
        "player_gameweek_contexts: %s"
        % context["player_gameweek_context_count"],
        "blank_team_gameweeks: %s" % context["blank_team_gameweek_count"],
        "double_team_gameweeks: %s" % context["double_team_gameweek_count"],
        "```",
        "",
        "## Gameweek completeness",
        "",
        "| GW | Fixtures | Single teams | Blanks | Doubles | Kickoff missing/invalid | Kickoff review |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for gameweek, summary in sorted(
        validation["gameweeks"].items(), key=lambda item: int(item[0])
    ):
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                gameweek,
                summary["fixture_count"],
                summary["single_team_count"],
                summary["blank_team_count"],
                summary["double_team_count"],
                summary["missing_or_invalid_kickoff_count"],
                summary["kickoff_review_count"],
            )
        )
    lines.extend(["", "## Kickoff comparison", "", "```text"])
    for key, value in report["kickoff_comparison"].items():
        lines.append("%s: %s" % (key, value))
    lines.extend(["```", "", "## Blockers", ""])
    if report["blockers"]:
        lines.extend("- %s" % item for item in report["blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend("- %s" % item for item in report["warnings"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            "- Focused live 2026/27 horizon for the GW1 Fast Lane.",
            "- Broad historical use across every season pair is deferred.",
            "- Complete postponed-fixture scenario generation is deferred.",
            "- The builder is independent from prediction and optimization logic.",
            "- Database, prediction, squad-state, and publishing writes remain disabled.",
            "",
            "## Stop point",
            "",
            "> A validated 2026/27 GW1-GW5 fixture horizon can be consumed by predictions and optimization.",
            "",
            "Stop point satisfied: `%s`" % report["stop_point_satisfied"],
            "",
        ]
    )
    return "\n".join(lines)


def artifact_definitions() -> Dict[str, Tuple[str, str]]:
    return {
        "fixture_horizon_csv": ("fixture_horizon", "csv"),
        "team_fixture_horizon_csv": ("team_fixture_horizon", "csv"),
        "team_gameweek_horizon_csv": ("team_gameweek_horizon", "csv"),
        "player_fixture_eligibility_csv": (
            "player_fixture_eligibility",
            "csv",
        ),
        "fixture_change_log_csv": ("fixture_change_log", "csv"),
        "run_metadata_json": ("run_metadata", "json"),
        "fixture_horizon_report_json": ("fixture_horizon_report", "json"),
        "fixture_horizon_report_md": ("fixture_horizon_report", "md"),
    }


def write_immutable_outputs(
    artifact_root: Path,
    target_season: str,
    start_gw: int,
    as_of_time: str,
    run_id: str,
    frames: Mapping[str, pd.DataFrame],
    change_log: pd.DataFrame,
    run_metadata: Mapping[str, Any],
    report: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    from ml.artifacts.paths import build_immutable_artifact_key
    from ml.artifacts.storage import LocalArtifactStorage

    storage = LocalArtifactStorage(artifact_root)
    definitions = artifact_definitions()
    keys = {
        name: build_immutable_artifact_key(
            artifact_type=ARTIFACT_TYPE,
            season=target_season,
            target_gw=start_gw,
            as_of_time=as_of_time,
            run_id=run_id,
            version=HORIZON_VERSION,
            filename=filename,
            extension=extension,
        )
        for name, (filename, extension) in definitions.items()
    }
    payloads = {
        "fixture_horizon_csv": frames["fixture_horizon"].to_csv(index=False),
        "team_fixture_horizon_csv": frames["team_fixture_horizon"].to_csv(
            index=False
        ),
        "team_gameweek_horizon_csv": frames["team_gameweek_horizon"].to_csv(
            index=False
        ),
        "player_fixture_eligibility_csv": frames[
            "player_fixture_eligibility"
        ].to_csv(index=False),
        "fixture_change_log_csv": change_log.to_csv(index=False),
        "run_metadata_json": json.dumps(
            run_metadata,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
    }
    stored: Dict[str, Dict[str, Any]] = {}
    for name, content in payloads.items():
        artifact = storage.write_immutable_text(keys[name], content)
        stored[name] = artifact.to_dict()

    report["artifacts"] = {
        "root": str(Path(artifact_root).expanduser().resolve()),
        "keys": keys,
        "stored_before_report": dict(stored),
    }
    report_json = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ) + "\n"
    report_md = build_markdown_report(report)
    for name, content in (
        ("fixture_horizon_report_json", report_json),
        ("fixture_horizon_report_md", report_md),
    ):
        artifact = storage.write_immutable_text(keys[name], content)
        stored[name] = artifact.to_dict()
    return stored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only fixture horizon from Day76C immutable artifacts. "
            "The default scope is 2026/27 GW1-GW5."
        )
    )
    parser.add_argument("--rollover-report-json", required=True)
    parser.add_argument("--target-season", default="")
    parser.add_argument("--start-gw", type=int, default=DEFAULT_START_GW)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--as-of-time", default="")
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--previous-fixture-horizon-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_target = args.target_season or None
    validate_requested_scope(
        requested_target or DEFAULT_TARGET_SEASON,
        args.start_gw,
        args.horizon,
    )

    rollover_path = Path(args.rollover_report_json).expanduser().resolve()
    rollover_report = load_json(rollover_path)
    validated = validate_rollover_report(
        rollover_report,
        target_season=requested_target,
    )
    target_season = requested_target or validated["target_season"]
    start_gw, end_gw = validate_requested_scope(
        target_season,
        args.start_gw,
        args.horizon,
    )
    as_of_time = format_utc(
        parse_utc(
            args.as_of_time or validated["as_of_time_utc"],
            "as_of_time",
        )
    )

    input_paths = resolve_rollover_inputs(rollover_path)
    previous_path = (
        Path(args.previous_fixture_horizon_csv).expanduser().resolve()
        if args.previous_fixture_horizon_csv
        else None
    )
    if previous_path is not None:
        input_paths["previous_fixture_horizon"] = previous_path

    fixture_source = load_csv(input_paths["fixture_scope"], "fixture scope")
    current_player_source = load_csv(
        input_paths["current_player_pool"], "current player pool"
    )
    team_mapping_source = load_csv(
        input_paths["team_identity_mapping"], "team identity mapping"
    )

    teams = normalize_target_teams(team_mapping_source, target_season)
    players = normalize_current_players(
        current_player_source,
        target_season,
        teams["team_id"].astype(int).tolist(),
    )
    fixtures = normalize_fixture_scope(
        fixture_source,
        target_season,
        start_gw,
        args.horizon,
        teams["team_id"].astype(int).tolist(),
    )
    previous = load_previous_fixture_horizon(previous_path)
    fixtures, change_log, change_summary = annotate_kickoff_changes(
        fixtures,
        previous,
    )
    frames, validation = analyze_horizon(
        fixtures=fixtures,
        teams=teams,
        players=players,
        start_gw=start_gw,
        horizon=args.horizon,
        change_summary=change_summary,
    )
    if validation["blockers"]:
        raise FixtureHorizonInputError(" ".join(validation["blockers"]))

    from ml.contracts.run_metadata import (
        build_run_metadata,
        provenance_inputs_from_file_metadata,
    )

    created_at = utc_now()
    input_metadata = provenance_metadata(input_paths)
    rule_versions = (
        (rollover_report.get("target_season_rules") or {}).get(
            "actual_versions"
        )
        or {}
    )
    parent_run_id = validated["parent_run_id"] or rollover_path.parent.name.replace(
        "run_id=", ""
    )
    run_metadata = build_run_metadata(
        run_id=None,
        run_type="feature",
        artifact_type=ARTIFACT_TYPE,
        source_seasons=[target_season],
        target_season=target_season,
        target_gw=start_gw,
        horizon=args.horizon,
        as_of_time=as_of_time,
        prediction_mode=PREDICTION_MODE,
        created_at=created_at,
        feature_version=HORIZON_VERSION,
        artifact_version=HORIZON_SCHEMA_VERSION,
        rules_versions={
            str(key): str(value)
            for key, value in rule_versions.items()
            if value
        },
        additional_versions={
            "horizon_version": HORIZON_VERSION,
            "source_rollover_version": str(
                rollover_report.get("rollover_version") or "day76c_unknown"
            ),
        },
        provenance={
            "producer": "ml.features.build_fixture_horizon",
            "inputs": provenance_inputs_from_file_metadata(input_metadata),
            "parent_run_ids": [str(parent_run_id)],
            "notes": [
                "Artifact-first Day79B fixture horizon using Day76C immutable inputs.",
                "Blank, double, duplicate, missing-kickoff, and kickoff-change states remain explicit.",
                "No database, prediction, optimization, squad-state, or publishing writes are performed.",
            ],
        },
    ).to_dict()
    run_id = str(run_metadata["run_id"])

    frames_with_metadata = {
        name: add_run_fields(
            dataframe=frame,
            run_id=run_id,
            target_season=target_season,
            start_gw=start_gw,
            horizon=args.horizon,
            as_of_time=as_of_time,
            parent_run_id=str(parent_run_id),
        )
        for name, frame in frames.items()
    }
    change_log = add_run_fields(
        dataframe=change_log,
        run_id=run_id,
        target_season=target_season,
        start_gw=start_gw,
        horizon=args.horizon,
        as_of_time=as_of_time,
        parent_run_id=str(parent_run_id),
    )

    report: Dict[str, Any] = {
        "created_at_utc": created_at,
        "horizon_version": HORIZON_VERSION,
        "horizon_schema_version": HORIZON_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "target_season": target_season,
        "start_gw": start_gw,
        "end_gw": end_gw,
        "horizon": args.horizon,
        "as_of_time_utc": as_of_time,
        "passed": True,
        "audit_only": True,
        "preview_only": True,
        "writes_database": False,
        "writes_predictions_table": False,
        "writes_squad_state": False,
        "production_approved": False,
        "run_metadata": run_metadata,
        "validation": validation,
        "kickoff_comparison": change_summary,
        "source_artifacts": input_metadata,
        "blockers": list(validation["blockers"]),
        "warnings": sorted(set(validation["warnings"])),
        "deferred_hardening": [
            "broad_historical_use_across_every_season_pair",
            "complete_postponed_fixture_scenario_generation",
        ],
        "ready_for_day97a": bool(
            validation["horizon_complete_for_consumption"]
        ),
        "ready_for_predictions_and_optimization": bool(
            validation["horizon_complete_for_consumption"]
        ),
        "stop_point_satisfied": bool(
            validation["horizon_complete_for_consumption"]
        ),
    }
    stored = write_immutable_outputs(
        artifact_root=Path(args.artifact_root),
        target_season=target_season,
        start_gw=start_gw,
        as_of_time=as_of_time,
        run_id=run_id,
        frames=frames_with_metadata,
        change_log=change_log,
        run_metadata=run_metadata,
        report=report,
    )

    print("Day79B fixture horizon complete.")
    print("run_id:", run_id)
    print("target_season:", target_season)
    print("start_gw:", start_gw)
    print("end_gw:", end_gw)
    print("horizon:", args.horizon)
    print("fixture_rows:", len(frames_with_metadata["fixture_horizon"]))
    print(
        "team_fixture_rows:",
        len(frames_with_metadata["team_fixture_horizon"]),
    )
    print(
        "team_gameweek_rows:",
        len(frames_with_metadata["team_gameweek_horizon"]),
    )
    print(
        "player_fixture_eligibility_rows:",
        len(frames_with_metadata["player_fixture_eligibility"]),
    )
    print(
        "blank_team_gameweeks:",
        validation["context"]["blank_team_gameweek_count"],
    )
    print(
        "double_team_gameweeks:",
        validation["context"]["double_team_gameweek_count"],
    )
    print("immutable_artifacts:", len(stored))
    print("writes_database: false")
    print("production_approved: false")
    print("stop_point_satisfied: true")


if __name__ == "__main__":
    main()
