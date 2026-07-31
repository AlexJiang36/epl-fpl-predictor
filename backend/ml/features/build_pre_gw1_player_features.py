from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal
from ml.validation.resolve_prediction_mode import resolve_prediction_mode


FEATURE_VERSION = "day71a_v0"
FEATURE_SCOPE = "pre_gw1_player_features"
PROMOTED_TEAM_SHORT_NAMES = {"BUR", "LEE", "SUN"}

OUTPUT_COLUMNS = [
    "source_season",
    "target_season",
    "target_gw",
    "prediction_mode",
    "feature_scope",
    "feature_version",
    "player_id",
    "fpl_player_id",
    "player_name",
    "web_name",
    "team_id",
    "team_name",
    "team_short_name",
    "position",
    "price",
    "status",
    "has_fixture",
    "fixture_id",
    "fpl_fixture_id",
    "opponent_team_id",
    "opponent_team_name",
    "opponent_short_name",
    "is_home",
    "blank_gw_flag",
    "has_prev_season_player_prior",
    "player_mapping_status",
    "player_mapping_confidence",
    "player_mapping_reason",
    "raw_player_id",
    "raw_player_name",
    "prior_source",
    "prior_identity_scope",
    "prev_season_minutes",
    "prev_season_total_points",
    "prev_season_points_per90",
    "prev_season_goals",
    "prev_season_assists",
    "prev_season_bonus",
    "prev_season_clean_sheets",
    "prev_season_goals_conceded",
    "prev_season_starts_proxy",
    "prev_season_appearances",
    "prev_season_points_per_match",
    "prior_reliability_score",
    "team_effective_prev_season_points_per_match",
    "opponent_effective_prev_season_points_per_match",
    "team_effective_prev_season_goals_for_per_match",
    "opponent_effective_prev_season_goals_for_per_match",
    "team_effective_prev_season_goals_against_per_match",
    "opponent_effective_prev_season_goals_against_per_match",
    "team_fallback_applied",
    "opponent_fallback_applied",
    "no_prior_flag",
    "promoted_team_player_flag",
    "opponent_promoted_team_flag",
    "uncertain_status_flag",
    "missing_price_flag",
    "missing_position_flag",
    "missing_team_context_flag",
    "missing_fixture_context_flag",
    "prediction_write_allowed",
    "production_ready",
    "requires_player_feature_manifest_before_prediction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only Pre-GW1 player feature artifact from current target-season "
            "players, safe accepted historical player mappings, previous-season player "
            "priors, GW1 fixtures, and effective team/opponent prior context."
        )
    )
    parser.add_argument("--source-season", "--source_season", required=True)
    parser.add_argument("--target-season", "--target_season", required=True)
    parser.add_argument("--target-gw", "--target_gw", type=int, required=True)
    parser.add_argument(
        "--prediction-mode",
        "--prediction_mode",
        default="auto",
        choices=["auto", "pre_gw1_prior", "early_season_blend", "normal_weekly"],
    )
    parser.add_argument(
        "--stabilization-gw",
        "--stabilization_gw",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--allow-experimental-mode",
        "--allow_experimental_mode",
        action="store_true",
    )
    parser.add_argument("--player-prior-csv", "--player_prior_csv", required=True)
    parser.add_argument("--player-mapping-csv", "--player_mapping_csv", required=True)
    parser.add_argument("--match-features-csv", "--match_features_csv", required=True)
    parser.add_argument("--out-csv", "--out_csv", required=True)
    parser.add_argument("--out-json", "--out_json", required=True)
    parser.add_argument("--out-md", "--out_md", required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_sql_dataframe(
    sql: str,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    db = SessionLocal()
    try:
        return pd.read_sql(text(sql), db.bind, params=params or {})
    finally:
        db.close()


def db_scalar(
    sql: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    db = SessionLocal()
    try:
        return db.execute(text(sql), params or {}).scalar()
    finally:
        db.close()


def first_existing_column(
    columns: Iterable[str],
    candidates: Sequence[str],
) -> Optional[str]:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def parse_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    return str(value).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


def nullable_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    value_text = str(value).strip()
    if not value_text:
        return None

    try:
        return int(float(value_text))
    except (TypeError, ValueError):
        return None


def nullable_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    value_text = str(value).strip()
    if not value_text:
        return None

    try:
        return float(value_text)
    except (TypeError, ValueError):
        return None


def normalize_raw_id(value: Any) -> Optional[str]:
    normalized = nullable_int(value)
    if normalized is None:
        return None
    return str(normalized)


def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    result = str(value).strip()
    return result or None


def ensure_input_file(path_value: str, label: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError("%s does not exist: %s" % (label, path))
    if not path.is_file():
        raise ValueError("%s is not a file: %s" % (label, path))
    return path


def filter_artifact_seasons(
    dataframe: pd.DataFrame,
    source_season: str,
    target_season: str,
    target_gw: Optional[int] = None,
) -> pd.DataFrame:
    result = dataframe.copy()

    if "source_season" in result.columns:
        result = result[
            result["source_season"].astype(str).str.strip() == source_season
        ].copy()

    if "target_season" in result.columns:
        result = result[
            result["target_season"].astype(str).str.strip() == target_season
        ].copy()

    if target_gw is not None and "target_gw" in result.columns:
        numeric_target_gw = pd.to_numeric(result["target_gw"], errors="coerce")
        result = result[numeric_target_gw == target_gw].copy()

    return result


def load_target_players() -> pd.DataFrame:
    dataframe = read_sql_dataframe(
        """
        SELECT
            p.id AS player_id,
            p.fpl_player_id,
            NULLIF(
                BTRIM(
                    CONCAT_WS(
                        ' ',
                        NULLIF(BTRIM(p.first_name), ''),
                        NULLIF(BTRIM(p.second_name), '')
                    )
                ),
                ''
            ) AS player_name,
            p.web_name,
            p.team_id,
            p.position,
            p.now_cost,
            p.status
        FROM players AS p
        ORDER BY p.id
        """
    )

    dataframe["player_id"] = dataframe["player_id"].apply(nullable_int)
    dataframe["fpl_player_id"] = dataframe["fpl_player_id"].apply(nullable_int)
    dataframe["team_id"] = dataframe["team_id"].apply(nullable_int)
    dataframe["price"] = pd.to_numeric(
        dataframe["now_cost"],
        errors="coerce",
    ) / 10.0

    return dataframe.drop(columns=["now_cost"])


def load_target_teams() -> pd.DataFrame:
    dataframe = read_sql_dataframe(
        """
        SELECT
            t.id AS team_id,
            t.fpl_team_id,
            t.name AS team_name,
            t.short_name AS team_short_name
        FROM teams AS t
        ORDER BY t.id
        """
    )

    dataframe["team_id"] = dataframe["team_id"].apply(nullable_int)
    dataframe["fpl_team_id"] = dataframe["fpl_team_id"].apply(nullable_int)
    return dataframe


def load_target_fixtures(
    target_season: str,
    target_gw: int,
) -> pd.DataFrame:
    dataframe = read_sql_dataframe(
        """
        SELECT
            f.id AS fixture_id,
            f.fpl_fixture_id,
            f.home_team_id,
            f.away_team_id,
            f.kickoff_time,
            f.finished,
            f.home_score,
            f.away_score,
            f.gw,
            f.season
        FROM fixtures AS f
        WHERE f.season = :target_season
          AND f.gw = :target_gw
        ORDER BY f.kickoff_time, f.id
        """,
        {
            "target_season": target_season,
            "target_gw": target_gw,
        },
    )

    for column in (
        "fixture_id",
        "fpl_fixture_id",
        "home_team_id",
        "away_team_id",
        "gw",
    ):
        dataframe[column] = dataframe[column].apply(nullable_int)

    return dataframe


def load_player_priors(
    path_value: str,
    source_season: str,
    target_season: str,
) -> pd.DataFrame:
    path = ensure_input_file(path_value, "player prior CSV")
    dataframe = pd.read_csv(path, low_memory=False)
    return filter_artifact_seasons(
        dataframe,
        source_season=source_season,
        target_season=target_season,
    )


def load_player_mapping_candidates(
    path_value: str,
    source_season: str,
    target_season: str,
) -> pd.DataFrame:
    path = ensure_input_file(path_value, "player mapping CSV")
    dataframe = pd.read_csv(path, low_memory=False)
    return filter_artifact_seasons(
        dataframe,
        source_season=source_season,
        target_season=target_season,
    )


def load_match_features(
    path_value: str,
    source_season: str,
    target_season: str,
    target_gw: int,
) -> pd.DataFrame:
    path = ensure_input_file(path_value, "match feature CSV")
    dataframe = pd.read_csv(path, low_memory=False)
    return filter_artifact_seasons(
        dataframe,
        source_season=source_season,
        target_season=target_season,
        target_gw=target_gw,
    )


def representative_mapping_rows(
    player_mapping: pd.DataFrame,
) -> pd.DataFrame:
    if player_mapping.empty:
        return player_mapping.copy()

    result = player_mapping.copy()
    result["_raw_player_key"] = result["raw_player_id"].apply(normalize_raw_id)
    result["_candidate_rank_numeric"] = pd.to_numeric(
        result.get("candidate_rank"),
        errors="coerce",
    )
    result["_candidate_score_numeric"] = pd.to_numeric(
        result.get("candidate_score"),
        errors="coerce",
    )

    representative_rows: List[pd.Series] = []

    for _, group in result.groupby("_raw_player_key", dropna=False, sort=False):
        rank_one = group[group["_candidate_rank_numeric"] == 1]
        if not rank_one.empty:
            chosen = rank_one.sort_values(
                by=["_candidate_score_numeric"],
                ascending=False,
                na_position="last",
            ).iloc[0]
        else:
            chosen = group.sort_values(
                by=["_candidate_score_numeric"],
                ascending=False,
                na_position="last",
            ).iloc[0]

        representative_rows.append(chosen)

    return pd.DataFrame(representative_rows).reset_index(drop=True)


def build_accepted_mapping_lookup(
    player_mapping: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    required_columns = {
        "raw_player_id",
        "raw_player_name",
        "candidate_player_id",
        "match_status",
        "is_auto_approved",
        "needs_manual_review",
        "is_ambiguous",
    }
    missing_columns = sorted(required_columns - set(player_mapping.columns))
    if missing_columns:
        raise ValueError(
            "Player mapping CSV is missing required columns: %s"
            % ", ".join(missing_columns)
        )

    working = player_mapping.copy()
    working["_raw_player_key"] = working["raw_player_id"].apply(normalize_raw_id)
    working["_candidate_player_id_int"] = working["candidate_player_id"].apply(
        nullable_int
    )
    working["_is_auto_approved_bool"] = working["is_auto_approved"].apply(
        parse_bool
    )
    working["_needs_manual_review_bool"] = working[
        "needs_manual_review"
    ].apply(parse_bool)
    working["_is_ambiguous_bool"] = working["is_ambiguous"].apply(parse_bool)

    if "candidate_rank" in working.columns:
        working["_candidate_rank_numeric"] = pd.to_numeric(
            working["candidate_rank"],
            errors="coerce",
        )
        rank_is_safe = working["_candidate_rank_numeric"] == 1
    else:
        rank_is_safe = pd.Series(True, index=working.index)

    accepted = working[
        working["_is_auto_approved_bool"]
        & ~working["_needs_manual_review_bool"]
        & ~working["_is_ambiguous_bool"]
        & working["_candidate_player_id_int"].notna()
        & rank_is_safe
    ].copy()

    accepted = accepted.sort_values(
        by=["_candidate_player_id_int", "_raw_player_key"]
    )

    duplicate_candidate_player_id_count = int(
        accepted["_candidate_player_id_int"].duplicated(keep=False).sum()
    )
    duplicate_raw_player_id_count = int(
        accepted["_raw_player_key"].duplicated(keep=False).sum()
    )

    representative = representative_mapping_rows(player_mapping)
    if representative.empty:
        ambiguous_or_manual_review_rows = 0
        unmatched_rows = 0
    else:
        representative["_needs_manual_review_bool"] = representative[
            "needs_manual_review"
        ].apply(parse_bool)
        representative["_is_ambiguous_bool"] = representative[
            "is_ambiguous"
        ].apply(parse_bool)
        representative["_candidate_player_id_int"] = representative[
            "candidate_player_id"
        ].apply(nullable_int)

        representative_match_status = representative["match_status"].fillna(
            ""
        ).astype(str).str.lower()

        ambiguous_or_manual_review_rows = int(
            (
                representative["_needs_manual_review_bool"]
                | representative["_is_ambiguous_bool"]
            ).sum()
        )
        unmatched_rows = int(
            (
                representative["_candidate_player_id_int"].isna()
                | representative_match_status.str.contains("unmatched")
            ).sum()
        )

    selected_columns = {
        "_candidate_player_id_int": "player_id",
        "_raw_player_key": "_raw_player_key",
        "raw_player_id": "raw_player_id",
        "raw_player_name": "raw_player_name",
        "match_status": "player_mapping_status",
    }

    candidate_score_column = first_existing_column(
        accepted.columns,
        ["candidate_score", "mapping_confidence", "name_score"],
    )
    match_reason_column = first_existing_column(
        accepted.columns,
        ["match_reason", "mapping_reason"],
    )

    if candidate_score_column:
        selected_columns[
            candidate_score_column
        ] = "player_mapping_confidence"
    if match_reason_column:
        selected_columns[match_reason_column] = "player_mapping_reason"

    lookup = accepted[list(selected_columns.keys())].rename(
        columns=selected_columns
    )

    if "player_mapping_confidence" not in lookup.columns:
        lookup["player_mapping_confidence"] = None
    if "player_mapping_reason" not in lookup.columns:
        lookup["player_mapping_reason"] = None

    lookup["player_id"] = lookup["player_id"].apply(nullable_int)
    lookup = lookup.drop_duplicates(subset=["player_id"], keep="first")

    summary = {
        "mapping_rows": int(len(player_mapping)),
        "top_mapping_rows": int(len(representative)),
        "accepted_mapping_rows": int(len(accepted)),
        "ambiguous_or_manual_review_rows": ambiguous_or_manual_review_rows,
        "unmatched_rows": unmatched_rows,
        "duplicate_accepted_candidate_player_id_count": (
            duplicate_candidate_player_id_count
        ),
        "duplicate_accepted_raw_player_id_count": duplicate_raw_player_id_count,
    }

    return lookup, summary


def build_prior_lookup(
    player_priors: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "raw_player_id",
        "raw_player_name",
        "prior_identity_scope",
        "prior_source",
        "prev_season_minutes",
        "prev_season_appearances",
        "prev_season_starts_proxy",
        "prev_season_total_points",
        "prev_season_points_per90",
        "prev_season_goals",
        "prev_season_assists",
        "prev_season_clean_sheets",
        "prev_season_bonus",
    }
    missing_columns = sorted(required_columns - set(player_priors.columns))
    if missing_columns:
        raise ValueError(
            "Player prior CSV is missing required columns: %s"
            % ", ".join(missing_columns)
        )

    dataframe = player_priors.copy()
    dataframe["_raw_player_key"] = dataframe["raw_player_id"].apply(
        normalize_raw_id
    )

    points_per_match_column = first_existing_column(
        dataframe.columns,
        [
            "prev_season_points_per_match",
            "prev_season_points_per_appearance",
        ],
    )

    selected = pd.DataFrame(
        {
            "_raw_player_key": dataframe["_raw_player_key"],
            "prior_raw_player_name": dataframe["raw_player_name"],
            "prior_source": dataframe["prior_source"],
            "prior_identity_scope": dataframe["prior_identity_scope"],
            "prev_season_minutes": dataframe["prev_season_minutes"],
            "prev_season_total_points": dataframe[
                "prev_season_total_points"
            ],
            "prev_season_points_per90": dataframe[
                "prev_season_points_per90"
            ],
            "prev_season_goals": dataframe["prev_season_goals"],
            "prev_season_assists": dataframe["prev_season_assists"],
            "prev_season_bonus": dataframe["prev_season_bonus"],
            "prev_season_clean_sheets": dataframe[
                "prev_season_clean_sheets"
            ],
            "prev_season_starts_proxy": dataframe[
                "prev_season_starts_proxy"
            ],
            "prev_season_appearances": dataframe[
                "prev_season_appearances"
            ],
        }
    )

    if points_per_match_column:
        selected["prev_season_points_per_match"] = dataframe[
            points_per_match_column
        ]
    else:
        selected["prev_season_points_per_match"] = None

    goals_conceded_column = first_existing_column(
        dataframe.columns,
        [
            "prev_season_goals_conceded",
            "goals_conceded",
        ],
    )
    if goals_conceded_column:
        selected["prev_season_goals_conceded"] = dataframe[
            goals_conceded_column
        ]
    else:
        selected["prev_season_goals_conceded"] = None

    selected = selected.drop_duplicates(
        subset=["_raw_player_key"],
        keep="first",
    )
    return selected


def build_team_fixture_context(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    team_name_lookup = teams.set_index("team_id")[
        ["team_name", "team_short_name"]
    ].to_dict(orient="index")

    rows: List[Dict[str, Any]] = []

    for _, fixture in fixtures.iterrows():
        fixture_id = nullable_int(fixture.get("fixture_id"))
        fpl_fixture_id = nullable_int(fixture.get("fpl_fixture_id"))
        home_team_id = nullable_int(fixture.get("home_team_id"))
        away_team_id = nullable_int(fixture.get("away_team_id"))

        home_team = team_name_lookup.get(home_team_id, {})
        away_team = team_name_lookup.get(away_team_id, {})

        rows.append(
            {
                "team_id": home_team_id,
                "fixture_id": fixture_id,
                "fpl_fixture_id": fpl_fixture_id,
                "opponent_team_id": away_team_id,
                "opponent_team_name": away_team.get("team_name"),
                "opponent_short_name": away_team.get("team_short_name"),
                "is_home": True,
                "has_fixture": True,
            }
        )
        rows.append(
            {
                "team_id": away_team_id,
                "fixture_id": fixture_id,
                "fpl_fixture_id": fpl_fixture_id,
                "opponent_team_id": home_team_id,
                "opponent_team_name": home_team.get("team_name"),
                "opponent_short_name": home_team.get("team_short_name"),
                "is_home": False,
                "has_fixture": True,
            }
        )

    context = pd.DataFrame(rows)
    if context.empty:
        context = pd.DataFrame(
            columns=[
                "team_id",
                "fixture_id",
                "fpl_fixture_id",
                "opponent_team_id",
                "opponent_team_name",
                "opponent_short_name",
                "is_home",
                "has_fixture",
            ]
        )

    duplicate_team_fixture_context_rows = int(
        context["team_id"].duplicated(keep=False).sum()
    )

    summary = {
        "fixture_rows": int(len(fixtures)),
        "team_fixture_context_rows": int(len(context)),
        "duplicate_team_fixture_context_rows": (
            duplicate_team_fixture_context_rows
        ),
    }

    return context, summary


def build_effective_team_context(
    match_features: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "fixture_id",
        "home_team_id",
        "away_team_id",
        "home_team_short_name",
        "away_team_short_name",
        "home_effective_prev_season_points_per_match",
        "away_effective_prev_season_points_per_match",
        "home_effective_prev_season_goals_for_per_match",
        "away_effective_prev_season_goals_for_per_match",
        "home_effective_prev_season_goals_against_per_match",
        "away_effective_prev_season_goals_against_per_match",
        "home_team_fallback_applied",
        "away_team_fallback_applied",
    }
    missing_columns = sorted(required_columns - set(match_features.columns))
    if missing_columns:
        raise ValueError(
            "Match feature CSV is missing required columns: %s"
            % ", ".join(missing_columns)
        )

    rows: List[Dict[str, Any]] = []

    for _, fixture in match_features.iterrows():
        fixture_id = nullable_int(fixture.get("fixture_id"))
        home_team_id = nullable_int(fixture.get("home_team_id"))
        away_team_id = nullable_int(fixture.get("away_team_id"))

        rows.append(
            {
                "fixture_id": fixture_id,
                "team_id": home_team_id,
                "team_effective_prev_season_points_per_match": (
                    nullable_float(
                        fixture.get(
                            "home_effective_prev_season_points_per_match"
                        )
                    )
                ),
                "opponent_effective_prev_season_points_per_match": (
                    nullable_float(
                        fixture.get(
                            "away_effective_prev_season_points_per_match"
                        )
                    )
                ),
                "team_effective_prev_season_goals_for_per_match": (
                    nullable_float(
                        fixture.get(
                            "home_effective_prev_season_goals_for_per_match"
                        )
                    )
                ),
                "opponent_effective_prev_season_goals_for_per_match": (
                    nullable_float(
                        fixture.get(
                            "away_effective_prev_season_goals_for_per_match"
                        )
                    )
                ),
                "team_effective_prev_season_goals_against_per_match": (
                    nullable_float(
                        fixture.get(
                            "home_effective_prev_season_goals_against_per_match"
                        )
                    )
                ),
                "opponent_effective_prev_season_goals_against_per_match": (
                    nullable_float(
                        fixture.get(
                            "away_effective_prev_season_goals_against_per_match"
                        )
                    )
                ),
                "team_fallback_applied": parse_bool(
                    fixture.get("home_team_fallback_applied")
                ),
                "opponent_fallback_applied": parse_bool(
                    fixture.get("away_team_fallback_applied")
                ),
            }
        )

        rows.append(
            {
                "fixture_id": fixture_id,
                "team_id": away_team_id,
                "team_effective_prev_season_points_per_match": (
                    nullable_float(
                        fixture.get(
                            "away_effective_prev_season_points_per_match"
                        )
                    )
                ),
                "opponent_effective_prev_season_points_per_match": (
                    nullable_float(
                        fixture.get(
                            "home_effective_prev_season_points_per_match"
                        )
                    )
                ),
                "team_effective_prev_season_goals_for_per_match": (
                    nullable_float(
                        fixture.get(
                            "away_effective_prev_season_goals_for_per_match"
                        )
                    )
                ),
                "opponent_effective_prev_season_goals_for_per_match": (
                    nullable_float(
                        fixture.get(
                            "home_effective_prev_season_goals_for_per_match"
                        )
                    )
                ),
                "team_effective_prev_season_goals_against_per_match": (
                    nullable_float(
                        fixture.get(
                            "away_effective_prev_season_goals_against_per_match"
                        )
                    )
                ),
                "opponent_effective_prev_season_goals_against_per_match": (
                    nullable_float(
                        fixture.get(
                            "home_effective_prev_season_goals_against_per_match"
                        )
                    )
                ),
                "team_fallback_applied": parse_bool(
                    fixture.get("away_team_fallback_applied")
                ),
                "opponent_fallback_applied": parse_bool(
                    fixture.get("home_team_fallback_applied")
                ),
            }
        )

    return pd.DataFrame(rows).drop_duplicates(
        subset=["fixture_id", "team_id"],
        keep="first",
    )


def build_player_features(
    source_season: str,
    target_season: str,
    target_gw: int,
    resolved_prediction_mode: str,
    target_players: pd.DataFrame,
    target_teams: pd.DataFrame,
    target_fixtures: pd.DataFrame,
    player_priors: pd.DataFrame,
    player_mapping: pd.DataFrame,
    match_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    mapping_lookup, mapping_summary = build_accepted_mapping_lookup(
        player_mapping
    )
    prior_lookup = build_prior_lookup(player_priors)
    fixture_context, fixture_build_summary = build_team_fixture_context(
        target_fixtures,
        target_teams,
    )
    effective_team_context = build_effective_team_context(match_features)

    features = target_players.merge(
        target_teams[
            [
                "team_id",
                "team_name",
                "team_short_name",
            ]
        ],
        on="team_id",
        how="left",
        validate="many_to_one",
    )

    features = features.merge(
        fixture_context,
        on="team_id",
        how="left",
        validate="many_to_one",
    )

    features["has_fixture"] = features["has_fixture"].fillna(False).apply(
        parse_bool
    )
    features["blank_gw_flag"] = ~features["has_fixture"]

    features = features.merge(
        mapping_lookup,
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    features["player_mapping_status"] = features[
        "player_mapping_status"
    ].fillna("no_safe_accepted_mapping")

    features = features.merge(
        prior_lookup,
        on="_raw_player_key",
        how="left",
        validate="many_to_one",
    )

    mapped_raw_name = features["raw_player_name"]
    prior_raw_name = features["prior_raw_player_name"]
    features["raw_player_name"] = mapped_raw_name.where(
        mapped_raw_name.notna(),
        prior_raw_name,
    )
    features = features.drop(columns=["prior_raw_player_name"])

    features["has_prev_season_player_prior"] = (
        features["_raw_player_key"].notna()
        & features["prior_source"].notna()
        & features["prev_season_minutes"].notna()
    )
    features["no_prior_flag"] = ~features[
        "has_prev_season_player_prior"
    ]

    # Day71A uses binary identity/join reliability only. Minutes/role
    # reliability is deliberately deferred to the Day71B contract.
    features["prior_reliability_score"] = features[
        "has_prev_season_player_prior"
    ].astype(float)

    features = features.merge(
        effective_team_context,
        on=["fixture_id", "team_id"],
        how="left",
        validate="many_to_one",
    )

    features["promoted_team_player_flag"] = (
        features["team_short_name"].fillna("").isin(
            PROMOTED_TEAM_SHORT_NAMES
        )
    )
    features["opponent_promoted_team_flag"] = (
        features["opponent_short_name"].fillna("").isin(
            PROMOTED_TEAM_SHORT_NAMES
        )
    )

    normalized_status = features["status"].fillna("").astype(str).str.lower()
    features["uncertain_status_flag"] = normalized_status != "a"
    features["missing_price_flag"] = (
        pd.to_numeric(features["price"], errors="coerce").isna()
        | (pd.to_numeric(features["price"], errors="coerce") <= 0)
    )
    features["missing_position_flag"] = (
        features["position"].isna()
        | (features["position"].astype(str).str.strip() == "")
    )

    team_context_columns = [
        "team_effective_prev_season_points_per_match",
        "opponent_effective_prev_season_points_per_match",
        "team_effective_prev_season_goals_for_per_match",
        "opponent_effective_prev_season_goals_for_per_match",
        "team_effective_prev_season_goals_against_per_match",
        "opponent_effective_prev_season_goals_against_per_match",
    ]
    features["missing_team_context_flag"] = features[
        team_context_columns
    ].isna().any(axis=1)

    features["missing_fixture_context_flag"] = (
        ~features["has_fixture"]
        | features["fixture_id"].isna()
        | features["opponent_team_id"].isna()
    )

    features["team_fallback_applied"] = features[
        "team_fallback_applied"
    ].fillna(False).apply(parse_bool)
    features["opponent_fallback_applied"] = features[
        "opponent_fallback_applied"
    ].fillna(False).apply(parse_bool)

    features["source_season"] = source_season
    features["target_season"] = target_season
    features["target_gw"] = target_gw
    features["prediction_mode"] = resolved_prediction_mode
    features["feature_scope"] = FEATURE_SCOPE
    features["feature_version"] = FEATURE_VERSION

    features["prediction_write_allowed"] = False
    features["production_ready"] = False
    features[
        "requires_player_feature_manifest_before_prediction"
    ] = True

    for column in OUTPUT_COLUMNS:
        if column not in features.columns:
            features[column] = None

    features = features[OUTPUT_COLUMNS].sort_values(
        by=["team_short_name", "position", "player_name", "player_id"],
        na_position="last",
    ).reset_index(drop=True)

    build_summary = {
        "mapping_summary": mapping_summary,
        "fixture_build_summary": fixture_build_summary,
    }

    return features, build_summary


def build_report(
    args: argparse.Namespace,
    mode_result: Dict[str, Any],
    features: pd.DataFrame,
    build_summary: Dict[str, Any],
    player_priors: pd.DataFrame,
    player_mapping: pd.DataFrame,
    match_features: pd.DataFrame,
) -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []

    resolved_prediction_mode = mode_result.get(
        "resolved_prediction_mode"
    )
    mode_errors = list(mode_result.get("errors") or [])
    mode_warnings = list(mode_result.get("warnings") or [])

    blockers.extend(mode_errors)
    warnings.extend(mode_warnings)

    if resolved_prediction_mode != "pre_gw1_prior":
        blockers.append(
            "Day71A requires resolved_prediction_mode=pre_gw1_prior; got %s."
            % resolved_prediction_mode
        )

    target_player_rows = int(db_scalar("SELECT COUNT(*) FROM players") or 0)
    target_team_rows = int(db_scalar("SELECT COUNT(*) FROM teams") or 0)
    target_fixture_rows = int(
        db_scalar(
            """
            SELECT COUNT(*)
            FROM fixtures
            WHERE season = :target_season
              AND gw = :target_gw
            """,
            {
                "target_season": args.target_season,
                "target_gw": args.target_gw,
            },
        )
        or 0
    )

    feature_rows = int(len(features))
    duplicate_player_id_count = int(
        features["player_id"].duplicated(keep=False).sum()
    )
    players_with_prior = int(
        features["has_prev_season_player_prior"].sum()
    )
    players_without_prior = int(features["no_prior_flag"].sum())
    players_with_fixture = int(features["has_fixture"].sum())
    players_without_fixture = int(features["blank_gw_flag"].sum())
    players_on_promoted_teams = int(
        features["promoted_team_player_flag"].sum()
    )

    missing_team_context_count = int(
        features["missing_team_context_flag"].sum()
    )
    missing_fixture_context_count = int(
        features["missing_fixture_context_flag"].sum()
    )

    prediction_write_allowed_true_count = int(
        features["prediction_write_allowed"].sum()
    )
    production_ready_true_count = int(
        features["production_ready"].sum()
    )
    requires_manifest_false_count = int(
        (
            ~features[
                "requires_player_feature_manifest_before_prediction"
            ]
        ).sum()
    )

    prior_without_safe_mapping_count = int(
        (
            features["has_prev_season_player_prior"]
            & (
                features["player_mapping_status"]
                == "no_safe_accepted_mapping"
            )
        ).sum()
    )

    mapping_summary = build_summary["mapping_summary"]
    fixture_build_summary = build_summary["fixture_build_summary"]

    if target_player_rows <= 0:
        blockers.append("Target players table is empty.")
    if feature_rows != target_player_rows:
        blockers.append(
            "Feature rows (%s) do not equal target player rows (%s)."
            % (feature_rows, target_player_rows)
        )
    if duplicate_player_id_count != 0:
        blockers.append(
            "duplicate_player_id_count must be 0; got %s."
            % duplicate_player_id_count
        )
    if target_team_rows <= 0:
        blockers.append("Target teams table is empty.")
    if target_fixture_rows <= 0:
        blockers.append(
            "No target fixtures found for season=%s gw=%s."
            % (args.target_season, args.target_gw)
        )
    if mapping_summary["accepted_mapping_rows"] <= 0:
        blockers.append("No safe accepted player mappings were loaded.")
    if (
        mapping_summary[
            "duplicate_accepted_candidate_player_id_count"
        ]
        != 0
    ):
        blockers.append(
            "Accepted player mappings contain duplicate target player IDs."
        )
    if mapping_summary["duplicate_accepted_raw_player_id_count"] != 0:
        blockers.append(
            "Accepted player mappings contain duplicate raw player IDs."
        )
    if players_with_prior <= 0:
        blockers.append("No target players received a safe prior.")
    if players_without_prior <= 0:
        blockers.append(
            "All target players received priors unexpectedly; no-prior rows "
            "must remain visible."
        )
    if prior_without_safe_mapping_count != 0:
        blockers.append(
            "At least one player received a prior without a safe accepted mapping."
        )
    if (
        fixture_build_summary[
            "duplicate_team_fixture_context_rows"
        ]
        != 0
    ):
        blockers.append(
            "A target team has multiple GW fixture-context rows; Day71A requires "
            "one fixture context per player row."
        )
    if prediction_write_allowed_true_count != 0:
        blockers.append(
            "prediction_write_allowed must be False for every row."
        )
    if production_ready_true_count != 0:
        blockers.append("production_ready must be False for every row.")
    if requires_manifest_false_count != 0:
        blockers.append(
            "requires_player_feature_manifest_before_prediction must be True "
            "for every row."
        )

    if "prev_season_goals_conceded" not in player_priors.columns:
        warnings.append(
            "Player prior CSV has no prev_season_goals_conceded column; "
            "Day71A emits null values for that field."
        )
    if players_without_fixture > 0:
        warnings.append(
            "%s players have no GW%s fixture context."
            % (players_without_fixture, args.target_gw)
        )
    if missing_team_context_count > 0:
        warnings.append(
            "%s player rows have incomplete effective team/opponent context."
            % missing_team_context_count
        )
    if missing_fixture_context_count > 0:
        warnings.append(
            "%s player rows have incomplete fixture context."
            % missing_fixture_context_count
        )
    if players_with_prior != mapping_summary["accepted_mapping_rows"]:
        warnings.append(
            "players_with_prior (%s) differs from accepted_mapping_rows (%s). "
            "This can be valid only if an accepted mapping lacks a prior row or "
            "its target player is absent."
            % (
                players_with_prior,
                mapping_summary["accepted_mapping_rows"],
            )
        )

    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    passed = len(blockers) == 0

    report = {
        "created_at": utc_now(),
        "source_season": args.source_season,
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": resolved_prediction_mode,
        "feature_version": FEATURE_VERSION,
        "feature_scope": FEATURE_SCOPE,
        "audit_only": True,
        "writes_database": False,
        "uses_current_season_actual_player_gw_stats": False,
        "passed": passed,
        "ready_for_pre_gw1_player_features": passed,
        "ready_for_pre_gw1_player_prediction_preview": False,
        "ready_for_prediction_write": False,
        "inputs": {
            "player_prior_csv": str(Path(args.player_prior_csv)),
            "player_mapping_csv": str(Path(args.player_mapping_csv)),
            "match_features_csv": str(Path(args.match_features_csv)),
            "player_prior_rows": int(len(player_priors)),
            "player_mapping_rows": int(len(player_mapping)),
            "match_feature_rows": int(len(match_features)),
        },
        "outputs": {
            "out_csv": str(Path(args.out_csv)),
            "out_json": str(Path(args.out_json)),
            "out_md": str(Path(args.out_md)),
        },
        "row_counts": {
            "target_player_rows": target_player_rows,
            "target_team_rows": target_team_rows,
            "target_fixture_rows": target_fixture_rows,
            "feature_rows": feature_rows,
            "duplicate_player_id_count": duplicate_player_id_count,
            "players_with_prior": players_with_prior,
            "players_without_prior": players_without_prior,
            "players_on_promoted_teams": players_on_promoted_teams,
        },
        "mapping_summary": mapping_summary,
        "fixture_summary": {
            **fixture_build_summary,
            "players_with_fixture": players_with_fixture,
            "players_without_fixture": players_without_fixture,
        },
        "team_context_summary": {
            "missing_team_context_count": missing_team_context_count,
            "team_fallback_player_rows": int(
                features["team_fallback_applied"].sum()
            ),
            "opponent_fallback_player_rows": int(
                features["opponent_fallback_applied"].sum()
            ),
        },
        "safety_summary": {
            "prior_without_safe_mapping_count": (
                prior_without_safe_mapping_count
            ),
            "prediction_write_allowed_true_count": (
                prediction_write_allowed_true_count
            ),
            "production_ready_true_count": (
                production_ready_true_count
            ),
            "requires_manifest_false_count": (
                requires_manifest_false_count
            ),
            "uses_current_season_actual_player_gw_stats": False,
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            (
                "Target players and teams are current canonical tables without "
                "season columns; fixtures are filtered by target season and GW."
            ),
            (
                "Only safe Day66B auto-approved, non-ambiguous, non-manual-review "
                "mapping rows can join a historical player prior."
            ),
            (
                "Players without safe priors remain in the artifact with "
                "no_prior_flag=True."
            ),
            (
                "prior_reliability_score is a Day71A binary safe-identity/join "
                "indicator. Role/minutes reliability is deferred to Day71B."
            ),
            (
                "This script is read-only and does not query current-season "
                "player_gw_stats or write any database table."
            ),
        ],
    }

    return report


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    row_counts = report["row_counts"]
    mapping_summary = report["mapping_summary"]
    fixture_summary = report["fixture_summary"]
    team_context_summary = report["team_context_summary"]
    safety_summary = report["safety_summary"]

    lines = [
        "# Day71A — Pre-GW1 Player Feature Scaffolding",
        "",
        "- Created at: `%s`" % report["created_at"],
        "- Source season: `%s`" % report["source_season"],
        "- Target season: `%s`" % report["target_season"],
        "- Target GW: `%s`" % report["target_gw"],
        "- Requested prediction mode: `%s`"
        % report["requested_prediction_mode"],
        "- Resolved prediction mode: `%s`"
        % report["resolved_prediction_mode"],
        "- Feature version: `%s`" % report["feature_version"],
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `%s`" % report["audit_only"],
        "- Writes database: `%s`" % report["writes_database"],
        "- Ready for Pre-GW1 player features: `%s`"
        % report["ready_for_pre_gw1_player_features"],
        "- Ready for player prediction preview: `%s`"
        % report["ready_for_pre_gw1_player_prediction_preview"],
        "- Ready for prediction write: `%s`"
        % report["ready_for_prediction_write"],
        "",
        "## Row counts",
        "",
        "- Target players: `%s`" % row_counts["target_player_rows"],
        "- Target teams: `%s`" % row_counts["target_team_rows"],
        "- Target fixtures: `%s`" % row_counts["target_fixture_rows"],
        "- Feature rows: `%s`" % row_counts["feature_rows"],
        "- Duplicate player IDs: `%s`"
        % row_counts["duplicate_player_id_count"],
        "- Players with prior: `%s`"
        % row_counts["players_with_prior"],
        "- Players without prior: `%s`"
        % row_counts["players_without_prior"],
        "- Players on promoted teams: `%s`"
        % row_counts["players_on_promoted_teams"],
        "",
        "## Mapping summary",
        "",
        "- Mapping rows: `%s`" % mapping_summary["mapping_rows"],
        "- Top mapping rows: `%s`"
        % mapping_summary["top_mapping_rows"],
        "- Accepted mapping rows: `%s`"
        % mapping_summary["accepted_mapping_rows"],
        "- Ambiguous/manual-review rows: `%s`"
        % mapping_summary["ambiguous_or_manual_review_rows"],
        "- Unmatched rows: `%s`" % mapping_summary["unmatched_rows"],
        "- Duplicate accepted target player IDs: `%s`"
        % mapping_summary[
            "duplicate_accepted_candidate_player_id_count"
        ],
        "- Duplicate accepted raw player IDs: `%s`"
        % mapping_summary["duplicate_accepted_raw_player_id_count"],
        "",
        "## Fixture summary",
        "",
        "- Fixture rows: `%s`" % fixture_summary["fixture_rows"],
        "- Players with fixture: `%s`"
        % fixture_summary["players_with_fixture"],
        "- Players without fixture: `%s`"
        % fixture_summary["players_without_fixture"],
        "- Duplicate team fixture-context rows: `%s`"
        % fixture_summary["duplicate_team_fixture_context_rows"],
        "",
        "## Team context summary",
        "",
        "- Missing team-context rows: `%s`"
        % team_context_summary["missing_team_context_count"],
        "- Team fallback player rows: `%s`"
        % team_context_summary["team_fallback_player_rows"],
        "- Opponent fallback player rows: `%s`"
        % team_context_summary["opponent_fallback_player_rows"],
        "",
        "## Safety summary",
        "",
        "- Used current-season actual player_gw_stats: `%s`"
        % safety_summary[
            "uses_current_season_actual_player_gw_stats"
        ],
        "- Prior without safe mapping count: `%s`"
        % safety_summary["prior_without_safe_mapping_count"],
        "- prediction_write_allowed=True count: `%s`"
        % safety_summary[
            "prediction_write_allowed_true_count"
        ],
        "- production_ready=True count: `%s`"
        % safety_summary["production_ready_true_count"],
        "- requires manifest=False count: `%s`"
        % safety_summary["requires_manifest_false_count"],
        "",
        "## Blockers",
        "",
    ]

    if report["blockers"]:
        lines.extend("- %s" % item for item in report["blockers"])
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend("- %s" % item for item in report["warnings"])
    else:
        lines.append("- None")

    lines.extend(["", "## Notes", ""])
    lines.extend("- %s" % item for item in report["notes"])

    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(report: Dict[str, Any]) -> None:
    row_counts = report["row_counts"]
    mapping_summary = report["mapping_summary"]
    fixture_summary = report["fixture_summary"]
    safety_summary = report["safety_summary"]

    print("=== Day71A Pre-GW1 Player Feature Scaffolding ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("target_gw:", report["target_gw"])
    print(
        "resolved_prediction_mode:",
        report["resolved_prediction_mode"],
    )
    print("feature_version:", report["feature_version"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print(
        "ready_for_pre_gw1_player_features:",
        report["ready_for_pre_gw1_player_features"],
    )
    print(
        "ready_for_pre_gw1_player_prediction_preview:",
        report["ready_for_pre_gw1_player_prediction_preview"],
    )
    print(
        "ready_for_prediction_write:",
        report["ready_for_prediction_write"],
    )
    print("saved_csv:", report["outputs"]["out_csv"])
    print("saved_json:", report["outputs"]["out_json"])
    print("saved_md:", report["outputs"]["out_md"])

    print("\nRow counts:")
    print("- target_player_rows:", row_counts["target_player_rows"])
    print("- feature_rows:", row_counts["feature_rows"])
    print("- players_with_prior:", row_counts["players_with_prior"])
    print(
        "- players_without_prior:",
        row_counts["players_without_prior"],
    )

    print("\nMapping summary:")
    print(
        "- accepted_mapping_rows:",
        mapping_summary["accepted_mapping_rows"],
    )
    print(
        "- ambiguous_or_manual_review_rows:",
        mapping_summary["ambiguous_or_manual_review_rows"],
    )
    print("- unmatched_rows:", mapping_summary["unmatched_rows"])

    print("\nFixture summary:")
    print(
        "- players_with_fixture:",
        fixture_summary["players_with_fixture"],
    )
    print(
        "- players_without_fixture:",
        fixture_summary["players_without_fixture"],
    )

    print("\nSafety summary:")
    print(
        "- prediction_write_allowed_true_count:",
        safety_summary[
            "prediction_write_allowed_true_count"
        ],
    )
    print(
        "- production_ready_true_count:",
        safety_summary["production_ready_true_count"],
    )
    print(
        "- requires_manifest_false_count:",
        safety_summary["requires_manifest_false_count"],
    )

    print("\nBlockers:")
    if report["blockers"]:
        for blocker in report["blockers"]:
            print("-", blocker)
    else:
        print("- none")

    print("\nWarnings:")
    if report["warnings"]:
        for warning in report["warnings"]:
            print("-", warning)
    else:
        print("- none")


def main() -> None:
    args = parse_args()

    mode_result = resolve_prediction_mode(
        season=args.target_season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.source_season,
        stabilization_gw=args.stabilization_gw,
        allow_experimental_mode=args.allow_experimental_mode,
    )

    target_players = load_target_players()
    target_teams = load_target_teams()
    target_fixtures = load_target_fixtures(
        target_season=args.target_season,
        target_gw=args.target_gw,
    )
    player_priors = load_player_priors(
        path_value=args.player_prior_csv,
        source_season=args.source_season,
        target_season=args.target_season,
    )
    player_mapping = load_player_mapping_candidates(
        path_value=args.player_mapping_csv,
        source_season=args.source_season,
        target_season=args.target_season,
    )
    match_features = load_match_features(
        path_value=args.match_features_csv,
        source_season=args.source_season,
        target_season=args.target_season,
        target_gw=args.target_gw,
    )

    resolved_prediction_mode = mode_result.get(
        "resolved_prediction_mode"
    )

    features, build_summary = build_player_features(
        source_season=args.source_season,
        target_season=args.target_season,
        target_gw=args.target_gw,
        resolved_prediction_mode=str(resolved_prediction_mode),
        target_players=target_players,
        target_teams=target_teams,
        target_fixtures=target_fixtures,
        player_priors=player_priors,
        player_mapping=player_mapping,
        match_features=match_features,
    )

    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv_path, index=False)

    report = build_report(
        args=args,
        mode_result=mode_result,
        features=features,
        build_summary=build_summary,
        player_priors=player_priors,
        player_mapping=player_mapping,
        match_features=match_features,
    )

    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report)

    if report["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
