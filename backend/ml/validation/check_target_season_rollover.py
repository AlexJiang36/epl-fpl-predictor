from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from app.rules.chips import (
    load_chip_rules,
    validate_activation_examples,
    validate_interaction_examples,
    validate_squad_transfer_consistency,
    validate_state_examples,
)
from app.rules.scoring import load_scoring_rules, validate_deterministic_examples
from app.rules.squad import (
    load_squad_transfer_rules,
    validate_lineup_examples,
    validate_squad_examples,
)
from app.rules.transfers import (
    validate_transfer_examples,
    validate_transfer_legality_examples,
)
from ml.artifacts.paths import build_immutable_artifact_key
from ml.artifacts.storage import LocalArtifactStorage
from ml.contracts.run_metadata import build_run_metadata


ROLLOVER_SCHEMA_VERSION = "fpl_target_season_rollover_v1"
ROLLOVER_VERSION = "day76c_v1"
ARTIFACT_TYPE = "target_season_rollover"
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
VALID_POSITIONS = ("GKP", "DEF", "MID", "FWD")
KNOWN_PLAYER_STATUSES = {"a", "d", "i", "u", "s", "n"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("%s must be a non-empty timezone-aware timestamp." % label)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError("%s is not a valid ISO-8601 timestamp: %r." % (label, value))
    if parsed.tzinfo is None:
        raise ValueError("%s must include a timezone offset." % label)
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def nonempty_text(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def nullable_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be a JSON object." % label)
    return value


def required_list(value: Any, label: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError("%s must be a JSON array." % label)
    return value


def load_json_file(path_value: Any, label: str) -> Any:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in %s %s: %s" % (label, path, exc))


def fetch_json(url: str, timeout_seconds: int) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "epl-fpl-predictor-day76c/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Unable to fetch %s: %s" % (url, exc))
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Endpoint %s did not return valid UTF-8 JSON: %s" % (url, exc))


def load_rollover_config(path_value: Any) -> Dict[str, Any]:
    document = dict(required_mapping(load_json_file(path_value, "rollover config"), "rollover config"))
    required_keys = {
        "schema_version",
        "rollover_version",
        "competition",
        "source_season",
        "target_season",
        "target_gw",
        "fixture_scope",
        "source_team_short_names",
        "target_team_short_names",
        "promoted_team_short_names",
        "relegated_team_short_names",
        "unchanged_team_short_names",
        "position_changes",
        "required_rule_versions",
        "mapping_policy",
        "official_sources",
        "safety",
    }
    missing = sorted(required_keys - set(document))
    if missing:
        raise ValueError("Rollover config is missing keys: %s" % missing)
    if document["schema_version"] != ROLLOVER_SCHEMA_VERSION:
        raise ValueError("Unsupported rollover schema_version=%r." % document["schema_version"])
    if document["rollover_version"] != ROLLOVER_VERSION:
        raise ValueError("Unsupported rollover_version=%r." % document["rollover_version"])
    if document["competition"] != "FPL":
        raise ValueError("rollover competition must be FPL.")
    return document


def load_source_from_database() -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text
    from app.core.db import SessionLocal

    database = SessionLocal()
    try:
        teams = pd.read_sql(
            text(
                """
                SELECT
                    t.id AS source_team_id,
                    t.fpl_team_id AS source_fpl_team_id,
                    t.name AS source_team_name,
                    t.short_name AS source_team_short_name
                FROM teams AS t
                ORDER BY t.id
                """
            ),
            database.bind,
        )
        players = pd.read_sql(
            text(
                """
                SELECT
                    p.id AS source_player_id,
                    p.fpl_player_id AS source_fpl_player_id,
                    NULLIF(
                        BTRIM(
                            CONCAT_WS(
                                ' ',
                                NULLIF(BTRIM(p.first_name), ''),
                                NULLIF(BTRIM(p.second_name), '')
                            )
                        ),
                        ''
                    ) AS source_player_name,
                    p.web_name AS source_web_name,
                    p.position AS source_position,
                    t.id AS source_team_id,
                    t.name AS source_team_name,
                    t.short_name AS source_team_short_name
                FROM players AS p
                JOIN teams AS t ON t.id = p.team_id
                ORDER BY p.id
                """
            ),
            database.bind,
        )
    finally:
        database.close()
    return normalize_source_teams(teams), normalize_source_players(players, teams)


def first_existing_column(dataframe: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in dataframe.columns:
            return name
    return None


def normalized_column(
    dataframe: pd.DataFrame,
    names: Sequence[str],
    *,
    required: bool = True,
) -> pd.Series:
    column = first_existing_column(dataframe, names)
    if column is None:
        if required:
            raise ValueError("CSV is missing one of the required columns: %s" % list(names))
        return pd.Series([None] * len(dataframe), index=dataframe.index)
    return dataframe[column]


def normalize_source_teams(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "source_team_id": normalized_column(dataframe, ["source_team_id", "team_id", "id"]).map(nullable_int),
            "source_fpl_team_id": normalized_column(
                dataframe,
                ["source_fpl_team_id", "fpl_team_id"],
                required=False,
            ).map(nullable_int),
            "source_team_name": normalized_column(
                dataframe,
                ["source_team_name", "team_name", "name"],
            ).map(nonempty_text),
            "source_team_short_name": normalized_column(
                dataframe,
                ["source_team_short_name", "team_short_name", "short_name"],
            ).map(lambda value: (nonempty_text(value) or "").upper() or None),
        }
    )
    return result.drop_duplicates().reset_index(drop=True)


def normalize_source_players(
    dataframe: pd.DataFrame,
    source_teams: pd.DataFrame,
) -> pd.DataFrame:
    first_name = normalized_column(dataframe, ["first_name", "source_first_name"], required=False)
    second_name = normalized_column(dataframe, ["second_name", "source_second_name"], required=False)
    explicit_name = normalized_column(
        dataframe,
        ["source_player_name", "player_name", "full_name", "name"],
        required=False,
    )
    combined_names: List[Optional[str]] = []
    for explicit, first, second in zip(explicit_name.tolist(), first_name.tolist(), second_name.tolist()):
        explicit_text = nonempty_text(explicit)
        if explicit_text:
            combined_names.append(explicit_text)
        else:
            combined_names.append(nonempty_text(" ".join(filter(None, [nonempty_text(first), nonempty_text(second)]))))

    result = pd.DataFrame(
        {
            "source_player_id": normalized_column(dataframe, ["source_player_id", "player_id", "id"]).map(nullable_int),
            "source_fpl_player_id": normalized_column(
                dataframe,
                ["source_fpl_player_id", "fpl_player_id"],
                required=False,
            ).map(nullable_int),
            "source_player_name": combined_names,
            "source_web_name": normalized_column(
                dataframe,
                ["source_web_name", "web_name"],
                required=False,
            ).map(nonempty_text),
            "source_position": normalized_column(
                dataframe,
                ["source_position", "position"],
            ).map(lambda value: (nonempty_text(value) or "").upper() or None),
            "source_team_id": normalized_column(
                dataframe,
                ["source_team_id", "team_id"],
                required=False,
            ).map(nullable_int),
            "source_team_name": normalized_column(
                dataframe,
                ["source_team_name", "team_name"],
                required=False,
            ).map(nonempty_text),
            "source_team_short_name": normalized_column(
                dataframe,
                ["source_team_short_name", "team_short_name", "short_name"],
                required=False,
            ).map(lambda value: (nonempty_text(value) or "").upper() or None),
        }
    )

    team_by_id = {
        row.source_team_id: row
        for row in source_teams.itertuples(index=False)
        if row.source_team_id is not None
    }
    for index, row in result.iterrows():
        team = team_by_id.get(row["source_team_id"])
        if team is None:
            continue
        if not row["source_team_name"]:
            result.at[index, "source_team_name"] = team.source_team_name
        if not row["source_team_short_name"]:
            result.at[index, "source_team_short_name"] = team.source_team_short_name
    return result.drop_duplicates().reset_index(drop=True)


def load_source_csvs(player_csv: Any, team_csv: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    team_path = Path(team_csv).expanduser().resolve()
    player_path = Path(player_csv).expanduser().resolve()
    if not team_path.is_file():
        raise FileNotFoundError("source team CSV does not exist: %s" % team_path)
    if not player_path.is_file():
        raise FileNotFoundError("source player CSV does not exist: %s" % player_path)
    raw_teams = pd.read_csv(team_path, low_memory=False)
    teams = normalize_source_teams(raw_teams)
    players = normalize_source_players(pd.read_csv(player_path, low_memory=False), teams)
    return teams, players


def parse_bootstrap(document: Any) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    bootstrap = dict(required_mapping(document, "bootstrap-static response"))
    raw_teams = required_list(bootstrap.get("teams"), "bootstrap.teams")
    raw_positions = required_list(bootstrap.get("element_types"), "bootstrap.element_types")
    raw_players = required_list(bootstrap.get("elements"), "bootstrap.elements")

    position_by_id: Dict[int, str] = {}
    for raw_position in raw_positions:
        position = required_mapping(raw_position, "bootstrap element_type")
        position_id = nullable_int(position.get("id"))
        short_name = (nonempty_text(position.get("singular_name_short")) or "").upper()
        if position_id is not None:
            position_by_id[position_id] = short_name

    team_rows: List[Dict[str, Any]] = []
    team_by_id: Dict[int, Dict[str, Any]] = {}
    for raw_team in raw_teams:
        team = required_mapping(raw_team, "bootstrap team")
        team_id = nullable_int(team.get("id"))
        row = {
            "target_team_id": team_id,
            "target_team_code": nullable_int(team.get("code")),
            "target_team_name": nonempty_text(team.get("name")),
            "target_team_short_name": (nonempty_text(team.get("short_name")) or "").upper() or None,
        }
        team_rows.append(row)
        if team_id is not None:
            team_by_id[team_id] = row

    player_rows: List[Dict[str, Any]] = []
    for raw_player in raw_players:
        player = required_mapping(raw_player, "bootstrap player")
        team_id = nullable_int(player.get("team"))
        team = team_by_id.get(team_id, {})
        first_name = nonempty_text(player.get("first_name"))
        second_name = nonempty_text(player.get("second_name"))
        full_name = nonempty_text(" ".join(filter(None, [first_name, second_name])))
        status = (nonempty_text(player.get("status")) or "").lower() or None
        chance_next = nullable_int(player.get("chance_of_playing_next_round"))
        now_cost = nullable_int(player.get("now_cost"))
        position = position_by_id.get(nullable_int(player.get("element_type")) or -1)
        player_rows.append(
            {
                "target_player_id": nullable_int(player.get("id")),
                "target_player_code": nullable_int(player.get("code")),
                "target_player_name": full_name,
                "target_web_name": nonempty_text(player.get("web_name")),
                "target_team_id": team_id,
                "target_team_name": team.get("target_team_name"),
                "target_team_short_name": team.get("target_team_short_name"),
                "target_position": position,
                "target_price_units": now_cost,
                "target_price": (float(now_cost) / 10.0 if now_cost is not None else None),
                "target_status": status,
                "chance_of_playing_next_round": chance_next,
                "news": nonempty_text(player.get("news")),
                "news_added": nonempty_text(player.get("news_added")),
                "current_selection_eligible": status in {"a", "d"},
            }
        )

    event_rows = required_list(bootstrap.get("events"), "bootstrap.events")
    event_deadlines: Dict[str, Any] = {}
    for raw_event in event_rows:
        event = required_mapping(raw_event, "bootstrap event")
        event_id = nullable_int(event.get("id"))
        if event_id is not None and 1 <= event_id <= 5:
            event_deadlines[str(event_id)] = event.get("deadline_time")

    return (
        pd.DataFrame(team_rows),
        pd.DataFrame(player_rows),
        {"event_deadlines_gw1_gw5": event_deadlines},
    )


def parse_fixtures(document: Any, target_teams: pd.DataFrame, first_gw: int, last_gw: int) -> pd.DataFrame:
    raw_fixtures = required_list(document, "fixtures response")
    team_by_id = {
        int(row["target_team_id"]): row.to_dict()
        for _, row in target_teams.iterrows()
        if row["target_team_id"] is not None and not pd.isna(row["target_team_id"])
    }
    rows: List[Dict[str, Any]] = []
    for raw_fixture in raw_fixtures:
        fixture = required_mapping(raw_fixture, "fixture")
        gameweek = nullable_int(fixture.get("event"))
        if gameweek is None or gameweek < first_gw or gameweek > last_gw:
            continue
        home_id = nullable_int(fixture.get("team_h"))
        away_id = nullable_int(fixture.get("team_a"))
        home = team_by_id.get(home_id, {})
        away = team_by_id.get(away_id, {})
        kickoff = nonempty_text(fixture.get("kickoff_time"))
        rows.append(
            {
                "target_season": None,
                "gameweek": gameweek,
                "fixture_id": nullable_int(fixture.get("id")),
                "kickoff_time_utc": kickoff,
                "home_team_id": home_id,
                "home_team_name": home.get("target_team_name"),
                "home_team_short_name": home.get("target_team_short_name"),
                "away_team_id": away_id,
                "away_team_name": away.get("target_team_name"),
                "away_team_short_name": away.get("target_team_short_name"),
                "started": bool(fixture.get("started", False)),
                "finished": bool(fixture.get("finished", False)),
            }
        )
    return pd.DataFrame(rows)


def duplicate_nonnull_count(series: pd.Series) -> int:
    values = series.dropna()
    return int(values.duplicated(keep=False).sum())



def validate_source_player_pool(
    players: pd.DataFrame,
    source_teams: pd.DataFrame,
) -> Dict[str, Any]:
    blockers: List[str] = []
    name_available = players["source_player_name"].map(nonempty_text).notna() | players[
        "source_web_name"
    ].map(nonempty_text).notna()
    checks = {
        "missing_source_player_id": players["source_player_id"].isna(),
        "missing_source_player_identity_name": ~name_available,
        "missing_source_team": players["source_team_short_name"].isna(),
        "missing_or_invalid_source_position": ~players["source_position"].isin(VALID_POSITIONS),
    }
    invalid_samples: Dict[str, List[Any]] = {}
    for label, mask in checks.items():
        count = int(mask.sum())
        if count:
            blockers.append("Source player pool has %s row(s) with %s." % (count, label))
            invalid_samples[label] = players.loc[mask, "source_player_name"].head(10).fillna("<missing>").tolist()

    duplicate_ids = duplicate_nonnull_count(players["source_player_id"])
    if duplicate_ids:
        blockers.append("Source player pool has %s duplicate source player ID row(s)." % duplicate_ids)

    valid_team_short_names = set(
        source_teams["source_team_short_name"].dropna().astype(str).tolist()
    )
    orphan_mask = ~players["source_team_short_name"].isin(valid_team_short_names)
    orphan_count = int(orphan_mask.sum())
    if orphan_count:
        blockers.append("Source player pool has %s row(s) referencing an unknown source team." % orphan_count)

    return {
        "passed": not blockers,
        "player_count": int(len(players)),
        "duplicate_player_id_row_count": duplicate_ids,
        "position_counts": {
            str(key): int(value)
            for key, value in players["source_position"].fillna("<missing>").value_counts().sort_index().items()
        },
        "invalid_samples": invalid_samples,
        "blockers": blockers,
    }

def validate_current_player_pool(players: pd.DataFrame, target_teams: pd.DataFrame) -> Dict[str, Any]:
    blockers: List[str] = []
    invalid_samples: Dict[str, List[Any]] = {}

    checks = {
        "missing_player_id": players["target_player_id"].isna(),
        "missing_player_name": players["target_player_name"].isna() | (players["target_player_name"].astype(str).str.strip() == ""),
        "missing_team": players["target_team_short_name"].isna(),
        "missing_position": ~players["target_position"].isin(VALID_POSITIONS),
        "missing_or_invalid_price": players["target_price_units"].isna() | (players["target_price_units"].fillna(0) <= 0),
        "missing_or_unknown_availability": ~players["target_status"].isin(KNOWN_PLAYER_STATUSES),
    }
    for label, mask in checks.items():
        count = int(mask.sum())
        if count:
            blockers.append("Current player pool has %s row(s) with %s." % (count, label))
            invalid_samples[label] = players.loc[mask, "target_player_name"].head(10).fillna("<missing>").tolist()

    duplicate_ids = duplicate_nonnull_count(players["target_player_id"])
    if duplicate_ids:
        blockers.append("Current player pool has %s duplicate target player ID row(s)." % duplicate_ids)

    valid_team_ids = set(target_teams["target_team_id"].dropna().astype(int).tolist())
    orphan_mask = ~players["target_team_id"].isin(valid_team_ids)
    orphan_count = int(orphan_mask.sum())
    if orphan_count:
        blockers.append("Current player pool has %s row(s) referencing an unknown target team." % orphan_count)

    return {
        "passed": not blockers,
        "player_count": int(len(players)),
        "duplicate_player_id_row_count": duplicate_ids,
        "current_selection_eligible_count": int(players["current_selection_eligible"].fillna(False).sum()),
        "status_counts": {str(key): int(value) for key, value in players["target_status"].fillna("<missing>").value_counts().sort_index().items()},
        "position_counts": {str(key): int(value) for key, value in players["target_position"].fillna("<missing>").value_counts().sort_index().items()},
        "invalid_samples": invalid_samples,
        "blockers": blockers,
    }


def validate_team_transition(
    source_teams: pd.DataFrame,
    target_teams: pd.DataFrame,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    blockers: List[str] = []
    expected_source = set(config["source_team_short_names"])
    expected_target = set(config["target_team_short_names"])
    expected_promoted = set(config["promoted_team_short_names"])
    expected_relegated = set(config["relegated_team_short_names"])
    expected_unchanged = set(config["unchanged_team_short_names"])

    actual_source = set(source_teams["source_team_short_name"].dropna().tolist())
    actual_target = set(target_teams["target_team_short_name"].dropna().tolist())
    duplicate_source = duplicate_nonnull_count(source_teams["source_team_short_name"])
    duplicate_target = duplicate_nonnull_count(target_teams["target_team_short_name"])

    if len(source_teams) != 20 or actual_source != expected_source:
        blockers.append("Source-season team set does not match the configured 20-team 2025/26 set.")
    if len(target_teams) != 20 or actual_target != expected_target:
        blockers.append("Target-season team set does not match the configured 20-team 2026/27 set.")
    if duplicate_source:
        blockers.append("Source teams contain duplicate short-name rows.")
    if duplicate_target:
        blockers.append("Target teams contain duplicate short-name rows.")

    actual_promoted = actual_target - actual_source
    actual_relegated = actual_source - actual_target
    actual_unchanged = actual_source & actual_target
    if actual_promoted != expected_promoted:
        blockers.append("Promoted-team transition mismatch: expected=%s actual=%s." % (sorted(expected_promoted), sorted(actual_promoted)))
    if actual_relegated != expected_relegated:
        blockers.append("Relegated-team transition mismatch: expected=%s actual=%s." % (sorted(expected_relegated), sorted(actual_relegated)))
    if actual_unchanged != expected_unchanged:
        blockers.append("Unchanged-team identity set mismatch.")

    source_by_short = {row.source_team_short_name: row for row in source_teams.itertuples(index=False)}
    target_by_short = {row.target_team_short_name: row for row in target_teams.itertuples(index=False)}
    rows: List[Dict[str, Any]] = []
    for short_name in sorted(expected_unchanged):
        source = source_by_short.get(short_name)
        target = target_by_short.get(short_name)
        rows.append(
            {
                "source_season": config["source_season"],
                "target_season": config["target_season"],
                "source_team_id": getattr(source, "source_team_id", None),
                "source_team_name": getattr(source, "source_team_name", None),
                "source_team_short_name": short_name,
                "target_team_id": getattr(target, "target_team_id", None),
                "target_team_name": getattr(target, "target_team_name", None),
                "target_team_short_name": short_name,
                "mapping_status": "accepted_exact_short_name",
                "historical_prior_eligible": source is not None and target is not None,
            }
        )
    for short_name in sorted(expected_promoted):
        target = target_by_short.get(short_name)
        rows.append(
            {
                "source_season": config["source_season"],
                "target_season": config["target_season"],
                "source_team_id": None,
                "source_team_name": None,
                "source_team_short_name": None,
                "target_team_id": getattr(target, "target_team_id", None),
                "target_team_name": getattr(target, "target_team_name", None),
                "target_team_short_name": short_name,
                "mapping_status": "target_only_promoted",
                "historical_prior_eligible": False,
            }
        )
    for short_name in sorted(expected_relegated):
        source = source_by_short.get(short_name)
        rows.append(
            {
                "source_season": config["source_season"],
                "target_season": config["target_season"],
                "source_team_id": getattr(source, "source_team_id", None),
                "source_team_name": getattr(source, "source_team_name", None),
                "source_team_short_name": short_name,
                "target_team_id": None,
                "target_team_name": None,
                "target_team_short_name": None,
                "mapping_status": "source_only_relegated",
                "historical_prior_eligible": False,
            }
        )

    return pd.DataFrame(rows), {
        "passed": not blockers,
        "source_team_count": int(len(source_teams)),
        "target_team_count": int(len(target_teams)),
        "unchanged_team_count": len(actual_unchanged),
        "promoted_team_short_names": sorted(actual_promoted),
        "relegated_team_short_names": sorted(actual_relegated),
        "duplicate_source_short_name_row_count": duplicate_source,
        "duplicate_target_short_name_row_count": duplicate_target,
        "blockers": blockers,
    }


def validate_fixture_scope(
    fixtures: pd.DataFrame,
    target_teams: pd.DataFrame,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: List[str] = []
    fixture_scope = required_mapping(config["fixture_scope"], "fixture_scope")
    first_gw = int(fixture_scope["first_gameweek"])
    last_gw = int(fixture_scope["last_gameweek"])
    expected_per_gw = int(fixture_scope["expected_fixtures_per_gameweek"])
    team_ids = set(target_teams["target_team_id"].dropna().astype(int).tolist())
    gw_summary: Dict[str, Any] = {}

    if fixtures.empty:
        blockers.append("No target-season fixtures were found for GW%s-GW%s." % (first_gw, last_gw))
    duplicate_fixture_ids = duplicate_nonnull_count(fixtures["fixture_id"]) if not fixtures.empty else 0
    if duplicate_fixture_ids:
        blockers.append("Fixture scope contains duplicate fixture ID rows.")

    for gameweek in range(first_gw, last_gw + 1):
        subset = fixtures[fixtures["gameweek"] == gameweek]
        participants: List[int] = []
        invalid_kickoffs: List[Any] = []
        for row in subset.itertuples(index=False):
            if row.home_team_id is not None:
                participants.append(int(row.home_team_id))
            if row.away_team_id is not None:
                participants.append(int(row.away_team_id))
            try:
                parse_utc(row.kickoff_time_utc, "fixture kickoff_time")
            except ValueError:
                invalid_kickoffs.append(row.fixture_id)
        unknown_participants = sorted(set(participants) - team_ids)
        missing_teams = sorted(team_ids - set(participants))
        duplicate_participant_rows = len(participants) - len(set(participants))

        if len(subset) != expected_per_gw:
            blockers.append("GW%s has %s fixtures; expected %s." % (gameweek, len(subset), expected_per_gw))
        if invalid_kickoffs:
            blockers.append("GW%s has invalid or missing kickoff timestamps for fixture IDs %s." % (gameweek, invalid_kickoffs[:10]))
        if unknown_participants:
            blockers.append("GW%s references unknown target team IDs %s." % (gameweek, unknown_participants))
        if missing_teams:
            blockers.append("GW%s does not schedule every target team exactly once." % gameweek)
        if duplicate_participant_rows:
            blockers.append("GW%s schedules one or more target teams more than once." % gameweek)

        gw_summary[str(gameweek)] = {
            "fixture_count": int(len(subset)),
            "participant_count": len(participants),
            "unique_participant_count": len(set(participants)),
            "invalid_kickoff_count": len(invalid_kickoffs),
            "missing_team_count": len(missing_teams),
        }

    return {
        "passed": not blockers,
        "fixture_count": int(len(fixtures)),
        "duplicate_fixture_id_row_count": duplicate_fixture_ids,
        "gameweeks": gw_summary,
        "blockers": blockers,
    }


def index_rows_by_name(dataframe: pd.DataFrame, column: str) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for row_index, value in dataframe[column].items():
        key = normalized_name(value)
        if key:
            index[key].append(int(row_index))
    return dict(index)


def build_player_identity_mapping(
    source_players: pd.DataFrame,
    target_players: pd.DataFrame,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    blockers: List[str] = []
    warnings: List[str] = []
    source_full_index = index_rows_by_name(source_players, "source_player_name")
    source_web_index = index_rows_by_name(source_players, "source_web_name")
    unchanged_teams = set(config["unchanged_team_short_names"])

    expected_position_changes: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    expected_position_change_lookup: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for change in config["position_changes"]:
        team_short_name = str(change["team_short_name"]).upper()
        canonical_key = (normalized_name(change["player_name"]), team_short_name)
        expected_position_changes[canonical_key] = change
        alias_values = [change.get("player_name"), change.get("web_name")]
        alias_values.extend(change.get("aliases", []))
        for alias in alias_values:
            alias_key = normalized_name(alias)
            if alias_key:
                expected_position_change_lookup[(alias_key, team_short_name)] = canonical_key

    rows: List[Dict[str, Any]] = []
    accepted_source_use: Dict[Any, List[int]] = defaultdict(list)
    verified_position_change_keys: set = set()
    present_expected_position_change_keys: set = set()
    unlisted_position_changes: List[str] = []

    for target_index, target in target_players.iterrows():
        full_key = normalized_name(target["target_player_name"])
        web_key = normalized_name(target["target_web_name"])
        selected_source_index: Optional[int] = None
        mapping_status = "unresolved_no_exact_identity"
        mapping_method: Optional[str] = None
        mapping_reason = "No unique exact source-season identity matched."

        full_candidates = source_full_index.get(full_key, []) if full_key else []
        if len(full_candidates) == 1:
            selected_source_index = full_candidates[0]
            mapping_status = "accepted_exact"
            mapping_method = "unique_exact_normalized_full_name"
            mapping_reason = "Unique exact normalized full-name match."
        elif len(full_candidates) > 1:
            mapping_status = "ambiguous_exact_full_name"
            mapping_reason = "Multiple source players share the exact normalized full name."
        else:
            web_candidates = source_web_index.get(web_key, []) if web_key else []
            compatible = [
                candidate
                for candidate in web_candidates
                if source_players.loc[candidate, "source_team_short_name"] == target["target_team_short_name"]
                and target["target_team_short_name"] in unchanged_teams
            ]
            if len(compatible) == 1:
                selected_source_index = compatible[0]
                mapping_status = "accepted_exact"
                mapping_method = "unique_exact_web_name_with_unchanged_team"
                mapping_reason = "Unique exact web-name match within the same unchanged team."
            elif len(compatible) > 1:
                mapping_status = "ambiguous_exact_web_name"
                mapping_reason = "Multiple same-team source players share the exact web name."

        source: Optional[pd.Series] = None
        if selected_source_index is not None:
            source = source_players.loc[selected_source_index]
            accepted_source_use[source["source_player_id"]].append(int(target_index))

        old_position = source["source_position"] if source is not None else None
        new_position = target["target_position"]
        position_changed = bool(source is not None and old_position != new_position)
        position_change_status = "not_mapped" if source is None else "unchanged"
        target_team_short_name = str(target["target_team_short_name"] or "").upper()
        expected_key = expected_position_change_lookup.get(
            (full_key, target_team_short_name)
        ) or expected_position_change_lookup.get((web_key, target_team_short_name))
        expected_change = expected_position_changes.get(expected_key) if expected_key else None
        if expected_key is not None:
            present_expected_position_change_keys.add(expected_key)
        if position_changed:
            if (
                expected_change is not None
                and expected_change["old_position"] == old_position
                and expected_change["new_position"] == new_position
            ):
                position_change_status = "verified_official"
                verified_position_change_keys.add(expected_key)
            else:
                position_change_status = "unlisted_position_change"
                unlisted_position_changes.append(str(target["target_player_name"]))
        elif expected_change is not None and source is not None:
            position_change_status = "expected_change_not_observed"

        historical_prior_eligible = mapping_status == "accepted_exact"
        rows.append(
            {
                "source_season": config["source_season"],
                "target_season": config["target_season"],
                "source_player_id": (source["source_player_id"] if source is not None else None),
                "source_fpl_player_id": (source["source_fpl_player_id"] if source is not None else None),
                "source_player_name": (source["source_player_name"] if source is not None else None),
                "source_web_name": (source["source_web_name"] if source is not None else None),
                "source_team_short_name": (source["source_team_short_name"] if source is not None else None),
                "source_position": old_position,
                "target_player_id": target["target_player_id"],
                "target_player_code": target["target_player_code"],
                "target_player_name": target["target_player_name"],
                "target_web_name": target["target_web_name"],
                "target_team_id": target["target_team_id"],
                "target_team_short_name": target["target_team_short_name"],
                "target_position": new_position,
                "target_price_units": target["target_price_units"],
                "target_price": target["target_price"],
                "target_status": target["target_status"],
                "current_selection_eligible": bool(target["current_selection_eligible"]),
                "mapping_status": mapping_status,
                "mapping_method": mapping_method,
                "mapping_reason": mapping_reason,
                "historical_prior_eligible": historical_prior_eligible,
                "position_changed": position_changed,
                "position_change_status": position_change_status,
            }
        )

    mapping = pd.DataFrame(rows)
    duplicate_source_ids = {
        source_id: target_indexes
        for source_id, target_indexes in accepted_source_use.items()
        if source_id is not None and len(target_indexes) > 1
    }
    if duplicate_source_ids:
        blockers.append("Accepted player mappings reuse %s source player ID(s)." % len(duplicate_source_ids))
        duplicate_targets = {index for indexes in duplicate_source_ids.values() for index in indexes}
        for index in duplicate_targets:
            mapping.at[index, "mapping_status"] = "blocked_duplicate_source_identity"
            mapping.at[index, "historical_prior_eligible"] = False

    ambiguous_count = int(mapping["mapping_status"].astype(str).str.startswith("ambiguous_").sum())
    if ambiguous_count:
        blockers.append("Player mapping contains %s ambiguous exact-identity row(s)." % ambiguous_count)

    if unlisted_position_changes:
        blockers.append("Exact player mappings contain unlisted target-season position changes: %s." % sorted(unlisted_position_changes)[:20])

    missing_expected: List[str] = []
    target_absent_expected: List[str] = []
    for key, expected in expected_position_changes.items():
        label = "%s %s->%s" % (
            expected["player_name"],
            expected["old_position"],
            expected["new_position"],
        )
        if key not in present_expected_position_change_keys:
            target_absent_expected.append(label)
        elif key not in verified_position_change_keys:
            missing_expected.append(label)

    if missing_expected:
        blockers.append(
            "Configured official position changes for players present in the target pool "
            "were not all verified: %s." % missing_expected
        )
    if target_absent_expected:
        warnings.append(
            "%s configured official position-change player(s) are absent from the current "
            "target player pool and cannot yet be verified: %s. This is non-blocking because "
            "no target row can inherit a stale position; rerun the checkpoint if they are added."
            % (len(target_absent_expected), target_absent_expected)
        )

    unresolved_count = int((~mapping["historical_prior_eligible"].astype(bool)).sum())
    if unresolved_count:
        warnings.append(
            "%s target player(s) have no safe exact source identity and remain ineligible for historical-prior joins."
            % unresolved_count
        )

    accepted_count = int(mapping["historical_prior_eligible"].astype(bool).sum())
    return mapping, {
        "passed": not blockers,
        "target_player_count": int(len(mapping)),
        "accepted_mapping_count": accepted_count,
        "unresolved_or_blocked_mapping_count": unresolved_count,
        "accepted_mapping_rate": round(accepted_count / len(mapping), 6) if len(mapping) else 0.0,
        "ambiguous_mapping_count": ambiguous_count,
        "duplicate_accepted_source_player_count": len(duplicate_source_ids),
        "verified_official_position_change_count": len(verified_position_change_keys),
        "present_expected_official_position_change_count": len(present_expected_position_change_keys),
        "expected_official_position_change_count": len(expected_position_changes),
        "target_absent_expected_position_change_count": len(target_absent_expected),
        "target_absent_expected_position_changes": target_absent_expected,
        "unverified_present_expected_position_change_count": len(missing_expected),
        "unverified_present_expected_position_changes": missing_expected,
        "unlisted_position_change_count": len(unlisted_position_changes),
        "blockers": blockers,
        "warnings": warnings,
    }


def validate_rule_registries(config_root: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    blockers: List[str] = []
    target_season = str(config["target_season"])
    expected_versions = required_mapping(config["required_rule_versions"], "required_rule_versions")

    scoring_path = config_root / ("scoring_rules_%s.json" % target_season)
    squad_path = config_root / ("squad_transfer_rules_%s.json" % target_season)
    chip_path = config_root / ("chip_rules_%s.json" % target_season)

    scoring_rules = load_scoring_rules(target_season, config_path=scoring_path)
    scoring_results, scoring_blockers = validate_deterministic_examples(scoring_rules)
    blockers.extend(scoring_blockers)

    squad_rules = load_squad_transfer_rules(target_season, config_path=squad_path)
    squad_examples = validate_squad_examples(squad_rules)
    lineup_examples = validate_lineup_examples(squad_rules)
    transfer_examples = validate_transfer_examples(squad_rules)
    transfer_legality_examples = validate_transfer_legality_examples(squad_rules)

    chip_rules = load_chip_rules(target_season, config_path=chip_path)
    chip_state_examples = validate_state_examples(chip_rules)
    chip_activation_examples = validate_activation_examples(chip_rules)
    chip_interaction_examples = validate_interaction_examples(chip_rules)
    cross_registry = validate_squad_transfer_consistency(chip_rules, squad_path)

    documents = {
        "scoring": scoring_rules.data,
        "squad_transfer": squad_rules.data,
        "chips": chip_rules.data,
    }
    actual_versions = {
        "scoring": scoring_rules.rules_version,
        "squad_transfer": squad_rules.rules_version,
        "chips": chip_rules.rules_version,
        "bonus_points_system": scoring_rules.data.get("scoring", {}).get("bonus", {}).get("bps_policy_version"),
    }
    for key, expected in expected_versions.items():
        if actual_versions.get(key) != expected:
            blockers.append("Rule version mismatch for %s: expected=%s actual=%s." % (key, expected, actual_versions.get(key)))

    for key, document in documents.items():
        policy = document.get("rollover_policy")
        if not isinstance(policy, Mapping):
            blockers.append("%s rules do not contain explicit rollover_policy metadata." % key)
        elif policy.get("copy_mode") != "explicit_reviewed_copy":
            blockers.append("%s rules were not marked as an explicit reviewed copy." % key)

    bps_policy = scoring_rules.data.get("rollover_policy", {}).get("bonus_points_system", {})
    bps_changes = bps_policy.get("changes") if isinstance(bps_policy, Mapping) else None
    expected_bps_metrics = {
        "times_tackled",
        "clearances_blocks_interceptions",
        "goalkeeper_any_save",
        "goalkeeper_inside_box_save",
        "goalkeeper_big_chance_save",
        "penalty_save",
    }
    actual_bps_metrics = {
        change.get("metric") for change in (bps_changes or []) if isinstance(change, Mapping)
    }
    if actual_bps_metrics != expected_bps_metrics:
        blockers.append("The 2026/27 BPS policy does not record the complete required change set.")
    if bps_policy.get("full_event_level_calculator_included") is not False:
        blockers.append("BPS policy must explicitly state that a full event-level calculator is not yet included.")

    special_events = squad_rules.data.get("transfers", {}).get("special_events")
    if special_events != []:
        blockers.append("2026/27 squad/transfer rules must not inherit the 2025/26 AFCON special transfer event.")

    chip_policy = chip_rules.data.get("rollover_policy", {}).get("verified_target_season_facts", {})
    if chip_policy.get("sets_per_season") != 2:
        blockers.append("2026/27 chip policy must explicitly record two chip sets.")
    if chip_policy.get("first_half_last_gameweek") != 19 or chip_policy.get("second_half_first_gameweek") != 20:
        blockers.append("2026/27 chip-window split must be explicitly recorded as GW19/GW20.")

    validations = {
        "scoring_examples": {
            "passed": not scoring_blockers,
            "passed_count": sum(1 for row in scoring_results if row.get("passed")),
            "total_count": len(scoring_results),
        },
        "squad_examples": squad_examples,
        "lineup_examples": lineup_examples,
        "transfer_examples": transfer_examples,
        "transfer_legality_examples": transfer_legality_examples,
        "chip_state_examples": chip_state_examples,
        "chip_activation_examples": chip_activation_examples,
        "chip_interaction_examples": chip_interaction_examples,
        "cross_registry_consistency": cross_registry,
    }
    for label, result in validations.items():
        passed = result.get("all_passed") if "all_passed" in result else result.get("passed")
        if passed is not True:
            blockers.append("Rule validation failed: %s." % label)

    return {
        "passed": not blockers,
        "paths": {
            "scoring": str(scoring_path),
            "squad_transfer": str(squad_path),
            "chips": str(chip_path),
        },
        "actual_versions": actual_versions,
        "validations": validations,
        "bps_policy": bps_policy,
        "special_transfer_events": special_events,
        "blockers": blockers,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# Day76C — 2026/27 Target-Season Rollover Checkpoint",
        "",
        "- Created at: `%s`" % report["created_at_utc"],
        "- As of: `%s`" % report["as_of_time_utc"],
        "- Source season: `%s`" % report["source_season"],
        "- Target season: `%s`" % report["target_season"],
        "- Target Gameweek: `%s`" % report["target_gw"],
        "- Rollover version: `%s`" % report["rollover_version"],
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `True`",
        "- Writes database: `False`",
        "- Ready for prediction write: `False`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in report["readiness"].items():
        lines.append("- %s: `%s`" % (key, value))

    source_pool = report["source_player_pool"]
    lines.extend(
        [
            "",
            "## Source player pool",
            "",
            "- Players: `%s`" % source_pool["player_count"],
            "- Duplicate player-ID rows: `%s`" % source_pool["duplicate_player_id_row_count"],
            "- Position counts: `%s`" % source_pool["position_counts"],
        ]
    )

    lines.extend(["", "## Current player pool", ""])
    pool = report["current_player_pool"]
    lines.extend(
        [
            "- Players: `%s`" % pool["player_count"],
            "- Current-selection eligible: `%s`" % pool["current_selection_eligible_count"],
            "- Duplicate player-ID rows: `%s`" % pool["duplicate_player_id_row_count"],
            "- Status counts: `%s`" % pool["status_counts"],
            "- Position counts: `%s`" % pool["position_counts"],
        ]
    )

    transition = report["team_transition"]
    lines.extend(
        [
            "",
            "## Team transition",
            "",
            "- Source teams: `%s`" % transition["source_team_count"],
            "- Target teams: `%s`" % transition["target_team_count"],
            "- Unchanged teams: `%s`" % transition["unchanged_team_count"],
            "- Promoted: `%s`" % transition["promoted_team_short_names"],
            "- Relegated: `%s`" % transition["relegated_team_short_names"],
        ]
    )

    fixtures = report["fixture_scope"]
    lines.extend(["", "## GW1–GW5 fixtures", "", "- Fixtures: `%s`" % fixtures["fixture_count"]])
    for gameweek, summary in fixtures["gameweeks"].items():
        lines.append(
            "- GW%s: `%s` fixtures, `%s` unique teams, `%s` invalid kickoffs"
            % (gameweek, summary["fixture_count"], summary["unique_participant_count"], summary["invalid_kickoff_count"])
        )

    player_mapping = report["player_identity_mapping"]
    lines.extend(
        [
            "",
            "## Player identity mapping",
            "",
            "- Target players: `%s`" % player_mapping["target_player_count"],
            "- Safe accepted mappings: `%s`" % player_mapping["accepted_mapping_count"],
            "- Unresolved or blocked mappings: `%s`" % player_mapping["unresolved_or_blocked_mapping_count"],
            "- Accepted mapping rate: `%s`" % player_mapping["accepted_mapping_rate"],
            "- Verified official position changes present in target pool: `%s/%s`"
            % (
                player_mapping["verified_official_position_change_count"],
                player_mapping["present_expected_official_position_change_count"],
            ),
            "- Configured official position changes: `%s`"
            % player_mapping["expected_official_position_change_count"],
            "- Configured changes absent from current target pool: `%s`"
            % player_mapping["target_absent_expected_position_change_count"],
            "- Target-absent configured changes: `%s`"
            % player_mapping["target_absent_expected_position_changes"],
            "- Fuzzy auto-approval used: `False`",
            "- Unresolved rows historical-prior eligible: `False`",
        ]
    )

    rules = report["target_season_rules"]
    lines.extend(
        [
            "",
            "## Target-season rules",
            "",
            "- Versions: `%s`" % rules["actual_versions"],
            "- AFCON/special transfer events: `%s`" % rules["special_transfer_events"],
            "- BPS policy version: `%s`" % rules["actual_versions"].get("bonus_points_system"),
            "- Full event-level BPS calculator included: `%s`"
            % rules["bps_policy"].get("full_event_level_calculator_included"),
            "- Model limitation: %s" % rules["bps_policy"].get("model_limitation"),
        ]
    )

    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend("- %s" % item for item in report["blockers"])
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend("- %s" % item for item in report["warnings"])
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Stop-point assessment",
            "",
            "> The project has a validated current player pool, fixture scope, target-season policies, and focused identity coverage for 2025/26 → 2026/27.",
            "",
            "Stop point satisfied: `%s`" % report["readiness"]["stop_point_satisfied"],
            "",
        ]
    )
    return "\n".join(lines)


def artifact_keys(metadata: Mapping[str, Any]) -> Dict[str, str]:
    common = {
        "artifact_type": metadata["artifact_type"],
        "season": metadata["target_season"],
        "target_gw": metadata["target_gw"],
        "as_of_time": metadata["as_of_time_utc"],
        "run_id": metadata["run_id"],
        "version": ROLLOVER_VERSION,
    }
    definitions = {
        "bootstrap_snapshot_json": ("bootstrap_snapshot", "json"),
        "fixtures_snapshot_json": ("fixtures_snapshot", "json"),
        "current_player_pool_csv": ("current_player_pool", "csv"),
        "fixture_scope_csv": ("gw1_gw5_fixtures", "csv"),
        "team_identity_mapping_csv": ("team_identity_mapping", "csv"),
        "player_identity_mapping_csv": ("player_identity_mapping", "csv"),
        "rollover_report_json": ("rollover_report", "json"),
        "rollover_report_md": ("rollover_report", "md"),
    }
    return {
        name: build_immutable_artifact_key(filename=filename, extension=extension, **common)
        for name, (filename, extension) in definitions.items()
    }


def write_artifacts(
    artifact_root: Path,
    keys: Mapping[str, str],
    bootstrap: Mapping[str, Any],
    fixtures_raw: Sequence[Any],
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    team_mapping: pd.DataFrame,
    player_mapping: pd.DataFrame,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    storage = LocalArtifactStorage(artifact_root)
    payloads = {
        "bootstrap_snapshot_json": json.dumps(bootstrap, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "fixtures_snapshot_json": json.dumps(fixtures_raw, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "current_player_pool_csv": players.to_csv(index=False),
        "fixture_scope_csv": fixtures.to_csv(index=False),
        "team_identity_mapping_csv": team_mapping.to_csv(index=False),
        "player_identity_mapping_csv": player_mapping.to_csv(index=False),
    }
    stored: Dict[str, Any] = {}
    for name, payload in payloads.items():
        artifact = storage.write_immutable_text(keys[name], payload)
        stored[name] = artifact.to_dict()

    report["artifacts"] = {
        "root": str(Path(artifact_root).expanduser().resolve()),
        "keys": dict(keys),
        "stored_before_report": stored,
    }
    report_json = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    report_md = build_markdown_report(report)
    json_artifact = storage.write_immutable_text(keys["rollover_report_json"], report_json)
    md_artifact = storage.write_immutable_text(keys["rollover_report_md"], report_md)
    stored["rollover_report_json"] = json_artifact.to_dict()
    stored["rollover_report_md"] = md_artifact.to_dict()
    return stored


def run_checkpoint(
    *,
    config: Mapping[str, Any],
    source_teams: pd.DataFrame,
    source_players: pd.DataFrame,
    bootstrap: Mapping[str, Any],
    fixtures_raw: Sequence[Any],
    config_root: Path,
    artifact_root: Path,
    source_season: str,
    target_season: str,
    target_gw: int,
    as_of_time: str,
) -> Dict[str, Any]:
    created_at = utc_now()
    blockers: List[str] = []
    warnings: List[str] = []

    if source_season != config["source_season"]:
        blockers.append("source_season=%s does not match rollover config %s." % (source_season, config["source_season"]))
    if target_season != config["target_season"]:
        blockers.append("target_season=%s does not match rollover config %s." % (target_season, config["target_season"]))
    if int(target_gw) != int(config["target_gw"]):
        blockers.append("target_gw=%s does not match rollover config %s." % (target_gw, config["target_gw"]))

    target_teams, target_players, bootstrap_context = parse_bootstrap(bootstrap)
    target_players.insert(0, "target_season", target_season)
    target_players.insert(1, "target_gw", target_gw)
    target_players.insert(2, "as_of_time_utc", format_utc(parse_utc(as_of_time, "as_of_time")))
    fixture_scope_config = required_mapping(config["fixture_scope"], "fixture_scope")
    fixtures = parse_fixtures(
        fixtures_raw,
        target_teams,
        int(fixture_scope_config["first_gameweek"]),
        int(fixture_scope_config["last_gameweek"]),
    )
    if not fixtures.empty:
        fixtures["target_season"] = target_season

    source_pool_result = validate_source_player_pool(source_players, source_teams)
    pool_result = validate_current_player_pool(target_players, target_teams)
    team_mapping, team_result = validate_team_transition(source_teams, target_teams, config)
    fixture_result = validate_fixture_scope(fixtures, target_teams, config)
    player_mapping, player_result = build_player_identity_mapping(source_players, target_players, config)
    rules_result = validate_rule_registries(config_root, config)

    for result in (source_pool_result, pool_result, team_result, fixture_result, player_result, rules_result):
        blockers.extend(result.get("blockers", []))
        warnings.extend(result.get("warnings", []))

    readiness = {
        "target_scope_confirmed": source_season == "2025_26" and target_season == "2026_27" and target_gw == 1,
        "source_player_pool_validated": bool(source_pool_result["passed"]),
        "current_player_pool_validated": bool(pool_result["passed"]),
        "target_team_transition_validated": bool(team_result["passed"]),
        "gw1_gw5_fixture_scope_validated": bool(fixture_result["passed"]),
        "focused_player_identity_coverage_validated": bool(player_result["passed"]),
        "target_season_rule_versions_validated": bool(rules_result["passed"]),
        "ready_for_prediction_write": False,
    }
    readiness["stop_point_satisfied"] = all(
        value
        for key, value in readiness.items()
        if key not in {"ready_for_prediction_write", "stop_point_satisfied"}
    ) and not blockers

    metadata = build_run_metadata(
        run_id=None,
        run_type="feature",
        artifact_type=ARTIFACT_TYPE,
        source_seasons=[source_season],
        target_season=target_season,
        target_gw=target_gw,
        horizon=int(fixture_scope_config["last_gameweek"]),
        as_of_time=as_of_time,
        prediction_mode="pre_gw1_prior",
        created_at=created_at,
        feature_version=ROLLOVER_VERSION,
        artifact_version=ROLLOVER_SCHEMA_VERSION,
        rules_versions={
            "scoring": str(config["required_rule_versions"]["scoring"]),
            "squad_transfer": str(config["required_rule_versions"]["squad_transfer"]),
            "chips": str(config["required_rule_versions"]["chips"]),
        },
        additional_versions={
            "rollover_version": ROLLOVER_VERSION,
            "bonus_points_system": str(config["required_rule_versions"]["bonus_points_system"]),
        },
        provenance={
            "producer": "ml.validation.check_target_season_rollover",
            "inputs": [],
            "parent_run_ids": [],
            "notes": [
                "Current target data comes from the official FPL bootstrap-static and fixtures APIs or caller-supplied snapshots.",
                "Source-season player and team identities come from the pre-rollover 2025/26 database or caller-supplied CSV snapshots.",
                "Unresolved target player identities are retained but remain historical-prior ineligible.",
            ],
        },
    ).to_dict()

    keys = artifact_keys(metadata)
    report: Dict[str, Any] = {
        "schema_version": ROLLOVER_SCHEMA_VERSION,
        "rollover_version": ROLLOVER_VERSION,
        "created_at_utc": created_at,
        "as_of_time_utc": format_utc(parse_utc(as_of_time, "as_of_time")),
        "source_season": source_season,
        "target_season": target_season,
        "target_gw": target_gw,
        "fixture_scope": fixture_result,
        "bootstrap_context": bootstrap_context,
        "source_player_pool": source_pool_result,
        "current_player_pool": pool_result,
        "team_transition": team_result,
        "player_identity_mapping": player_result,
        "target_season_rules": rules_result,
        "readiness": readiness,
        "audit_only": True,
        "writes_database": False,
        "passed": not blockers and bool(readiness["stop_point_satisfied"]),
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "run_metadata": metadata,
        "artifacts": {"root": str(artifact_root), "keys": keys},
    }

    stored = write_artifacts(
        artifact_root=artifact_root,
        keys=keys,
        bootstrap=bootstrap,
        fixtures_raw=fixtures_raw,
        players=target_players,
        fixtures=fixtures,
        team_mapping=team_mapping,
        player_mapping=player_mapping,
        report=report,
    )
    report["stored_artifacts"] = stored
    return report


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[3]
    default_config_root = repository_root / "config" / "fpl"
    parser = argparse.ArgumentParser(
        description=(
            "Validate the 2025/26 to 2026/27 FPL target-season rollover. "
            "The checkpoint is read-only and writes immutable local audit artifacts only."
        )
    )
    parser.add_argument("--source-season", default="2025_26")
    parser.add_argument("--target-season", default="2026_27")
    parser.add_argument("--target-gw", type=int, default=1)
    parser.add_argument(
        "--rollover-config",
        default=str(default_config_root / "target_season_rollover_2026_27.json"),
    )
    parser.add_argument("--config-root", default=str(default_config_root))
    parser.add_argument("--source-players-csv", default=None)
    parser.add_argument("--source-teams-csv", default=None)
    parser.add_argument("--bootstrap-json", default=None)
    parser.add_argument("--fixtures-json", default=None)
    parser.add_argument("--bootstrap-url", default=BOOTSTRAP_URL)
    parser.add_argument("--fixtures-url", default=FIXTURES_URL)
    parser.add_argument("--api-timeout-seconds", type=int, default=30)
    parser.add_argument("--as-of-time", default=None)
    parser.add_argument("--artifact-root", default="/tmp/fpl-artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_root = Path(args.config_root).expanduser().resolve()
    config = load_rollover_config(args.rollover_config)

    if bool(args.source_players_csv) != bool(args.source_teams_csv):
        raise ValueError("--source-players-csv and --source-teams-csv must be supplied together.")
    if args.source_players_csv:
        source_teams, source_players = load_source_csvs(args.source_players_csv, args.source_teams_csv)
    else:
        source_teams, source_players = load_source_from_database()

    bootstrap = (
        load_json_file(args.bootstrap_json, "bootstrap snapshot")
        if args.bootstrap_json
        else fetch_json(args.bootstrap_url, args.api_timeout_seconds)
    )
    fixtures_raw = (
        load_json_file(args.fixtures_json, "fixtures snapshot")
        if args.fixtures_json
        else fetch_json(args.fixtures_url, args.api_timeout_seconds)
    )
    as_of_time = args.as_of_time or utc_now()

    report = run_checkpoint(
        config=config,
        source_teams=source_teams,
        source_players=source_players,
        bootstrap=required_mapping(bootstrap, "bootstrap snapshot"),
        fixtures_raw=required_list(fixtures_raw, "fixtures snapshot"),
        config_root=config_root,
        artifact_root=Path(args.artifact_root),
        source_season=args.source_season,
        target_season=args.target_season,
        target_gw=args.target_gw,
        as_of_time=as_of_time,
    )

    print("=== Day76C 2026/27 Target-Season Rollover Checkpoint ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("target_gw:", report["target_gw"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("ready_for_prediction_write:", report["readiness"]["ready_for_prediction_write"])
    print("stop_point_satisfied:", report["readiness"]["stop_point_satisfied"])
    print("target_player_count:", report["current_player_pool"]["player_count"])
    print("accepted_player_mapping_count:", report["player_identity_mapping"]["accepted_mapping_count"])
    print("unresolved_player_mapping_count:", report["player_identity_mapping"]["unresolved_or_blocked_mapping_count"])
    print(
        "verified_position_change_count:",
        report["player_identity_mapping"]["verified_official_position_change_count"],
    )
    print(
        "present_expected_position_change_count:",
        report["player_identity_mapping"]["present_expected_official_position_change_count"],
    )
    print(
        "target_absent_expected_position_change_count:",
        report["player_identity_mapping"]["target_absent_expected_position_change_count"],
    )
    print("fixture_count_gw1_gw5:", report["fixture_scope"]["fixture_count"])
    print("rules_versions:", report["target_season_rules"]["actual_versions"])
    print("artifact_root:", report["artifacts"]["root"])
    print("report_json_key:", report["artifacts"]["keys"]["rollover_report_json"])
    print("report_md_key:", report["artifacts"]["keys"]["rollover_report_md"])
    if report["blockers"]:
        print("blockers:")
        for blocker in report["blockers"]:
            print("-", blocker)
    else:
        print("blockers: none")
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print("-", warning)
    else:
        print("warnings: none")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
