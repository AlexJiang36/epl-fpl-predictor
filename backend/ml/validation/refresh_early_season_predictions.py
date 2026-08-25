from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx
import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal
from ml.validation.resolve_prediction_mode import resolve_prediction_mode
from ml.features.build_pre_gw1_match_prediction_preview import (
    compute_signal,
    probability_from_signal,
)
from ml.features.build_pre_gw1_scoreline_preview import (
    estimate_expected_goals,
    scoreline_grid,
)


PIPELINE_VERSION = "early_season_prediction_pipeline_v0_1"
PLAYER_MODEL_NAME = "early_season_blend_player_v0"
MATCH_MODEL_NAME = "early_season_blend_match_v0"
SCORELINE_MODEL_NAME = "early_season_blend_scoreline_v0"

VALID_POSITIONS = {"GKP", "DEF", "MID", "FWD"}
FIXTURE_COMPONENT_WEIGHTS = {
    "GKP": {"attack": 0.00, "defence": 0.35},
    "DEF": {"attack": 0.15, "defence": 0.35},
    "MID": {"attack": 0.40, "defence": 0.05},
    "FWD": {"attack": 0.50, "defence": 0.00},
}

DEFAULT_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


# ---------- generic helpers ----------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def nullable_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nullable_int(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise RuntimeError("%s does not exist: %s" % (label, resolved))
    if not resolved.is_file():
        raise RuntimeError("%s is not a file: %s" % (label, resolved))
    return resolved


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only reusable early-season prediction preview for GW2-GW5. "
            "The immutable GW1 frozen prior artifacts remain the prior anchor; current-season "
            "actuals through target_gw-1 supply the current component. No prediction DB writes occur."
        )
    )
    parser.add_argument("--season", required=True)
    parser.add_argument("--prior-season", required=True)
    parser.add_argument("--target-gw", type=int, required=True)
    parser.add_argument("--stabilization-gw", type=int, default=6)
    parser.add_argument("--prediction-mode", default="auto")
    parser.add_argument(
        "--prior-artifact-dir",
        default="",
        help=(
            "Directory containing frozen GW1 prediction_preview_csv.csv and "
            "effective_match_features_csv.csv. Defaults to the durable GW1 match-model freeze."
        ),
    )
    parser.add_argument(
        "--bootstrap-json",
        default="",
        help="Optional local bootstrap-static JSON. If omitted, the official FPL API is fetched once.",
    )
    parser.add_argument("--bootstrap-url", default=DEFAULT_BOOTSTRAP_URL)
    parser.add_argument("--fixture-signal-min", type=float, default=0.75)
    parser.add_argument("--fixture-signal-max", type=float, default=1.25)
    parser.add_argument("--fixture-multiplier-min", type=float, default=0.85)
    parser.add_argument("--fixture-multiplier-max", type=float, default=1.15)
    parser.add_argument("--prediction-points-min", type=float, default=0.0)
    parser.add_argument("--prediction-points-max", type=float, default=15.0)
    parser.add_argument("--scoreline-max-goals", type=int, default=10)
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory. Defaults outside the repo under private-planning/gw-pre.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    # /repo/backend/ml/validation/file.py -> /repo
    return Path(__file__).resolve().parents[3]


def default_prior_artifact_dir(season: str) -> Path:
    return (
        repo_root().parent
        / "private-planning"
        / "frozen-snapshots"
        / season
        / "gw01"
        / "match-model"
        / "day76d-final-run"
    )


def default_output_dir(season: str, target_gw: int, run_id: str) -> Path:
    return (
        repo_root().parent
        / "private-planning"
        / "gw-pre"
        / season
        / ("gw%02d" % target_gw)
        / "early-season"
        / run_id
    )


# ---------- mode / readiness ----------

def resolve_mode(args: argparse.Namespace) -> Dict[str, Any]:
    result = resolve_prediction_mode(
        season=args.season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.prior_season,
        stabilization_gw=args.stabilization_gw,
        allow_experimental_mode=False,
    )
    if not result.get("valid"):
        raise RuntimeError("Prediction mode resolver failed: %s" % result.get("errors"))
    if result.get("resolved_prediction_mode") != "early_season_blend":
        raise RuntimeError(
            "This pipeline only supports early_season_blend; resolved=%s."
            % result.get("resolved_prediction_mode")
        )
    if not (2 <= int(args.target_gw) < int(args.stabilization_gw)):
        raise RuntimeError("target_gw must be inside the early-season window.")
    return result


def load_prior_files(args: argparse.Namespace) -> Dict[str, Any]:
    prior_dir = (
        Path(args.prior_artifact_dir)
        if args.prior_artifact_dir
        else default_prior_artifact_dir(args.season)
    ).expanduser().resolve()

    player_path = ensure_file(
        prior_dir / "prediction_preview_csv.csv",
        "frozen GW1 player prediction preview",
    )
    team_path = ensure_file(
        prior_dir / "effective_match_features_csv.csv",
        "frozen GW1 effective match features",
    )

    players = pd.read_csv(player_path, low_memory=False)
    teams = pd.read_csv(team_path, low_memory=False)

    if players.empty:
        raise RuntimeError("Frozen GW1 player prior preview is empty.")
    if teams.empty:
        raise RuntimeError("Frozen GW1 effective match features are empty.")

    player_target_gws = set(pd.to_numeric(players.get("target_gw"), errors="coerce").dropna().astype(int).tolist())
    if player_target_gws != {1}:
        raise RuntimeError("Frozen player prior must be GW1-only; found target_gw=%s." % sorted(player_target_gws))
    if "target_season" in players.columns:
        player_target_seasons = set(players["target_season"].dropna().astype(str).tolist())
        if player_target_seasons != {args.season}:
            raise RuntimeError("Frozen player prior target season mismatch: %s." % sorted(player_target_seasons))
    if "source_seasons" in players.columns:
        player_source_seasons = set(players["source_seasons"].dropna().astype(str).tolist())
        if args.prior_season not in player_source_seasons:
            raise RuntimeError("Frozen player prior does not reference prior_season=%s." % args.prior_season)
    if "prediction_mode" in players.columns:
        modes = set(players["prediction_mode"].dropna().astype(str).tolist())
        if modes != {"pre_gw1_prior"}:
            raise RuntimeError("Frozen player prior has unexpected prediction_mode=%s." % sorted(modes))

    team_target_gws = set(pd.to_numeric(teams.get("target_gw"), errors="coerce").dropna().astype(int).tolist())
    if team_target_gws != {1}:
        raise RuntimeError("Frozen team prior must be GW1-only; found target_gw=%s." % sorted(team_target_gws))
    if "target_season" in teams.columns:
        team_target_seasons = set(teams["target_season"].dropna().astype(str).tolist())
        if team_target_seasons != {args.season}:
            raise RuntimeError("Frozen team prior target season mismatch: %s." % sorted(team_target_seasons))
    if "source_season" in teams.columns:
        team_source_seasons = set(teams["source_season"].dropna().astype(str).tolist())
        if team_source_seasons != {args.prior_season}:
            raise RuntimeError("Frozen team prior source season mismatch: %s." % sorted(team_source_seasons))

    required_player = {
        "fpl_player_id",
        "position",
        "price",
        "blended_points_per90",
        "expected_minutes",
        "appearance_probability",
        "start_probability",
        "has_safe_prior",
    }
    missing_player = sorted(required_player - set(players.columns))
    if missing_player:
        raise RuntimeError("Frozen player prior missing columns: %s" % missing_player)

    required_team = {
        "home_fpl_team_id",
        "away_fpl_team_id",
        "home_effective_prev_season_points_per_match",
        "away_effective_prev_season_points_per_match",
        "home_effective_prev_season_goals_for_per_match",
        "away_effective_prev_season_goals_for_per_match",
        "home_effective_prev_season_goals_against_per_match",
        "away_effective_prev_season_goals_against_per_match",
        "home_effective_prev_season_clean_sheet_rate",
        "away_effective_prev_season_clean_sheet_rate",
        "home_effective_prev_season_goal_difference",
        "away_effective_prev_season_goal_difference",
    }
    missing_team = sorted(required_team - set(teams.columns))
    if missing_team:
        raise RuntimeError("Frozen team prior missing columns: %s" % missing_team)

    return {
        "dir": prior_dir,
        "player_path": player_path,
        "team_path": team_path,
        "players": players,
        "teams": teams,
        "player_sha256": sha256_file(player_path),
        "team_sha256": sha256_file(team_path),
    }


# ---------- current target-season data ----------

def query_dataframe(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    db = SessionLocal()
    try:
        return pd.read_sql(text(sql), db.bind, params=params or {})
    finally:
        db.close()


def load_current_teams(season: str) -> pd.DataFrame:
    df = query_dataframe(
        """
        SELECT id AS player_team_id,
               fpl_team_id,
               name AS team_name,
               short_name AS team_short_name
        FROM teams
        WHERE season = :season
        ORDER BY fpl_team_id
        """,
        {"season": season},
    )
    if df.empty:
        raise RuntimeError("No current teams for season=%s." % season)
    if int(df["fpl_team_id"].duplicated().sum()) > 0:
        raise RuntimeError("Duplicate fpl_team_id values in current team pool.")
    return df


def load_current_players(season: str) -> pd.DataFrame:
    df = query_dataframe(
        """
        SELECT p.id AS player_id,
               p.fpl_player_id,
               p.first_name,
               p.second_name,
               p.web_name,
               p.team_id,
               t.fpl_team_id,
               t.name AS team_name,
               t.short_name AS team_short_name,
               p.position,
               p.now_cost,
               p.status
        FROM players p
        JOIN teams t ON t.id = p.team_id
        WHERE p.season = :season
          AND t.season = :season
        ORDER BY p.fpl_player_id
        """,
        {"season": season},
    )
    if df.empty:
        raise RuntimeError("No current players for season=%s." % season)
    if int(df["fpl_player_id"].duplicated().sum()) > 0:
        raise RuntimeError("Duplicate fpl_player_id values in current player pool.")
    return df


def load_target_fixtures(season: str, target_gw: int) -> pd.DataFrame:
    df = query_dataframe(
        """
        SELECT f.id AS fixture_id,
               f.fpl_fixture_id,
               f.gw,
               f.kickoff_time,
               f.home_team_id,
               ht.fpl_team_id AS home_fpl_team_id,
               ht.name AS home_team_name,
               ht.short_name AS home_team_short_name,
               f.away_team_id,
               at.fpl_team_id AS away_fpl_team_id,
               at.name AS away_team_name,
               at.short_name AS away_team_short_name
        FROM fixtures f
        JOIN teams ht ON ht.id = f.home_team_id
        JOIN teams at ON at.id = f.away_team_id
        WHERE f.season = :season
          AND ht.season = :season
          AND at.season = :season
          AND f.gw = :target_gw
        ORDER BY f.kickoff_time NULLS LAST, f.id
        """,
        {"season": season, "target_gw": target_gw},
    )
    if df.empty:
        raise RuntimeError("No target fixtures for season=%s gw=%s." % (season, target_gw))
    return df


def load_past_fixtures(season: str, target_gw: int) -> pd.DataFrame:
    return query_dataframe(
        """
        SELECT f.id AS fixture_id,
               f.fpl_fixture_id,
               f.gw,
               f.finished,
               f.home_score,
               f.away_score,
               ht.fpl_team_id AS home_fpl_team_id,
               at.fpl_team_id AS away_fpl_team_id
        FROM fixtures f
        JOIN teams ht ON ht.id = f.home_team_id
        JOIN teams at ON at.id = f.away_team_id
        WHERE f.season = :season
          AND ht.season = :season
          AND at.season = :season
          AND f.gw < :target_gw
          AND f.home_score IS NOT NULL
          AND f.away_score IS NOT NULL
        ORDER BY f.gw, f.id
        """,
        {"season": season, "target_gw": target_gw},
    )


def load_player_actuals(season: str, target_gw: int) -> pd.DataFrame:
    return query_dataframe(
        """
        SELECT s.player_id,
               p.fpl_player_id,
               s.gw,
               s.minutes,
               s.goals_scored,
               s.assists,
               s.clean_sheets,
               s.total_points
        FROM player_gw_stats s
        JOIN players p ON p.id = s.player_id
        WHERE s.season = :season
          AND p.season = :season
          AND s.gw < :target_gw
        ORDER BY s.gw, p.fpl_player_id
        """,
        {"season": season, "target_gw": target_gw},
    )


def load_bootstrap(args: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    if args.bootstrap_json:
        path = ensure_file(Path(args.bootstrap_json), "bootstrap JSON")
        data = json.loads(path.read_text(encoding="utf-8"))
        return data, "file:%s" % path

    response = httpx.get(args.bootstrap_url, timeout=30)
    response.raise_for_status()
    return response.json(), args.bootstrap_url


def build_bootstrap_player_lookup(bootstrap: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for raw in bootstrap.get("elements", []):
        fpl_id = nullable_int(raw.get("id"))
        if fpl_id is None:
            continue
        lookup[fpl_id] = {
            "chance_of_playing_next_round": raw.get("chance_of_playing_next_round"),
            "news": raw.get("news"),
            "news_added": raw.get("news_added"),
            "status": raw.get("status"),
            "now_cost": raw.get("now_cost"),
            "web_name": raw.get("web_name"),
        }
    return lookup


# ---------- prior extraction ----------

def build_team_prior_lookup(frozen_match_features: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    metric_suffixes = {
        "points_per_match": "effective_prev_season_points_per_match",
        "goals_for_per_match": "effective_prev_season_goals_for_per_match",
        "goals_against_per_match": "effective_prev_season_goals_against_per_match",
        "clean_sheet_rate": "effective_prev_season_clean_sheet_rate",
        "goal_difference": "effective_prev_season_goal_difference",
    }
    lookup: Dict[int, Dict[str, Any]] = {}

    for _, row in frozen_match_features.iterrows():
        for side in ("home", "away"):
            fpl_team_id = nullable_int(row.get("%s_fpl_team_id" % side))
            if fpl_team_id is None:
                continue
            record = lookup.setdefault(
                fpl_team_id,
                {
                    "fpl_team_id": fpl_team_id,
                    "prior_fallback_applied": bool_value(
                        row.get("%s_team_fallback_applied" % side)
                    ),
                    "prior_feature_status": row.get(
                        "%s_effective_team_feature_status" % side
                    ),
                },
            )
            for out_name, suffix in metric_suffixes.items():
                value = nullable_float(row.get("%s_%s" % (side, suffix)))
                if value is not None:
                    if out_name in record and abs(float(record[out_name]) - value) > 1e-6:
                        raise RuntimeError(
                            "Frozen prior metric is inconsistent for fpl_team_id=%s metric=%s."
                            % (fpl_team_id, out_name)
                        )
                    record[out_name] = value

    required = set(metric_suffixes.keys())
    for team_id, record in lookup.items():
        missing = sorted(required - set(record.keys()))
        if missing:
            raise RuntimeError(
                "Frozen team prior incomplete for fpl_team_id=%s: %s" % (team_id, missing)
            )
    return lookup


def build_player_prior_lookup(frozen_player_preview: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for _, row in frozen_player_preview.iterrows():
        fpl_id = nullable_int(row.get("fpl_player_id"))
        if fpl_id is None:
            continue
        if fpl_id in lookup:
            raise RuntimeError("Duplicate frozen player prior fpl_player_id=%s." % fpl_id)
        lookup[fpl_id] = {
            "prior_points_per90": nullable_float(row.get("blended_points_per90")),
            "prior_expected_minutes": nullable_float(row.get("expected_minutes")),
            "prior_appearance_probability": nullable_float(row.get("appearance_probability")),
            "prior_start_probability": nullable_float(row.get("start_probability")),
            "prior_has_safe_prior": bool_value(row.get("has_safe_prior")),
            "prior_fallback_policy": row.get("fallback_policy_used"),
            "prior_prediction_confidence": row.get("prediction_confidence"),
            "prior_position": row.get("position"),
            "prior_price": nullable_float(row.get("price")),
        }
    return lookup


def build_position_prior_fallbacks(player_prior_lookup: Mapping[int, Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    rows = []
    for record in player_prior_lookup.values():
        position = str(record.get("prior_position") or "")
        if position not in VALID_POSITIONS:
            continue
        rows.append(
            {
                "position": position,
                "prior_points_per90": record.get("prior_points_per90"),
                "prior_expected_minutes": record.get("prior_expected_minutes"),
                "prior_appearance_probability": record.get("prior_appearance_probability"),
                "prior_start_probability": record.get("prior_start_probability"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Cannot build position prior fallbacks from frozen player prior.")
    result: Dict[str, Dict[str, float]] = {}
    for position, group in df.groupby("position"):
        result[str(position)] = {
            column: float(pd.to_numeric(group[column], errors="coerce").median())
            for column in [
                "prior_points_per90",
                "prior_expected_minutes",
                "prior_appearance_probability",
                "prior_start_probability",
            ]
        }
    return result


# ---------- current-season components ----------

def validate_actual_gw_coverage(actuals: pd.DataFrame, target_gw: int) -> Dict[str, Any]:
    expected = list(range(1, target_gw))
    observed = sorted(set(pd.to_numeric(actuals.get("gw"), errors="coerce").dropna().astype(int).tolist()))
    missing = [gw for gw in expected if gw not in observed]
    if missing:
        raise RuntimeError(
            "Current-season player actuals are missing required GWs before target_gw=%s: %s"
            % (target_gw, missing)
        )
    return {"expected_actual_gws": expected, "observed_actual_gws": observed}


def build_current_player_stats(actuals: pd.DataFrame, target_gw: int) -> Dict[int, Dict[str, Any]]:
    actual_gw_count = target_gw - 1
    if actual_gw_count <= 0:
        raise RuntimeError("Current player component requires at least one completed GW.")

    result: Dict[int, Dict[str, Any]] = {}
    for fpl_id, group in actuals.groupby("fpl_player_id"):
        fpl_player_id = int(fpl_id)
        minutes = pd.to_numeric(group["minutes"], errors="coerce").fillna(0.0)
        points = pd.to_numeric(group["total_points"], errors="coerce").fillna(0.0)
        total_minutes = float(minutes.sum())
        total_points = float(points.sum())
        appearances = int((minutes > 0).sum())
        current_expected_minutes = clamp(total_minutes / float(actual_gw_count), 0.0, 90.0)
        current_appearance_probability = clamp(appearances / float(actual_gw_count), 0.0, 1.0)
        current_points_per90 = (
            total_points * 90.0 / total_minutes if total_minutes > 0 else 0.0
        )
        observed_gws = sorted(set(pd.to_numeric(group["gw"], errors="coerce").dropna().astype(int).tolist()))
        result[fpl_player_id] = {
            "current_required_gws": actual_gw_count,
            "current_actual_rows": int(len(group)),
            "current_actual_gws_observed": ",".join(str(gw) for gw in observed_gws),
            "current_history_complete": observed_gws == list(range(1, target_gw)),
            "current_total_minutes": total_minutes,
            "current_total_points": total_points,
            "current_appearances": appearances,
            "current_expected_minutes": current_expected_minutes,
            "current_appearance_probability": current_appearance_probability,
            # v0 start proxy: a 60+ minute appearance. Official starts can be added later
            # without changing the pipeline interface.
            "current_start_probability_proxy": clamp(
                float((minutes >= 60).sum()) / float(actual_gw_count), 0.0, 1.0
            ),
            "current_points_per90": current_points_per90,
            "current_points_per_gw": total_points / float(actual_gw_count),
        }
    return result


def build_current_team_stats(past_fixtures: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    accum: Dict[int, Dict[str, float]] = {}

    def ensure(team_id: int) -> Dict[str, float]:
        return accum.setdefault(
            team_id,
            {
                "matches": 0.0,
                "points": 0.0,
                "goals_for": 0.0,
                "goals_against": 0.0,
                "clean_sheets": 0.0,
                "goal_difference": 0.0,
            },
        )

    for _, row in past_fixtures.iterrows():
        home_id = nullable_int(row.get("home_fpl_team_id"))
        away_id = nullable_int(row.get("away_fpl_team_id"))
        home_score = nullable_int(row.get("home_score"))
        away_score = nullable_int(row.get("away_score"))
        if None in {home_id, away_id, home_score, away_score}:
            continue

        home = ensure(int(home_id))
        away = ensure(int(away_id))
        home["matches"] += 1.0
        away["matches"] += 1.0
        home["goals_for"] += float(home_score)
        home["goals_against"] += float(away_score)
        away["goals_for"] += float(away_score)
        away["goals_against"] += float(home_score)
        home["goal_difference"] += float(home_score - away_score)
        away["goal_difference"] += float(away_score - home_score)
        if home_score == 0:
            away["clean_sheets"] += 1.0
        if away_score == 0:
            home["clean_sheets"] += 1.0

        if home_score > away_score:
            home["points"] += 3.0
        elif home_score < away_score:
            away["points"] += 3.0
        else:
            home["points"] += 1.0
            away["points"] += 1.0

    result: Dict[int, Dict[str, Any]] = {}
    for team_id, values in accum.items():
        matches = float(values["matches"])
        if matches <= 0:
            continue
        result[team_id] = {
            "current_matches": int(matches),
            "current_points_per_match": values["points"] / matches,
            "current_goals_for_per_match": values["goals_for"] / matches,
            "current_goals_against_per_match": values["goals_against"] / matches,
            "current_clean_sheet_rate": values["clean_sheets"] / matches,
            "current_goal_difference": values["goal_difference"],
            "current_goal_difference_per_match": values["goal_difference"] / matches,
        }
    return result


# ---------- blended match context ----------

def blended_metric(prior: Optional[float], current: Optional[float], prior_weight: float, current_weight: float) -> Tuple[float, float, float]:
    if current is None:
        if prior is None:
            raise RuntimeError("Both prior and current metric values are missing.")
        return float(prior), 1.0, 0.0
    if prior is None:
        return float(current), 0.0, 1.0
    return (
        float(prior_weight * prior + current_weight * current),
        float(prior_weight),
        float(current_weight),
    )


def blended_team_record(
    fpl_team_id: int,
    prior_lookup: Mapping[int, Mapping[str, Any]],
    current_lookup: Mapping[int, Mapping[str, Any]],
    prior_weight: float,
    current_weight: float,
) -> Dict[str, Any]:
    prior = prior_lookup.get(fpl_team_id)
    current = current_lookup.get(fpl_team_id)
    if prior is None and current is None:
        raise RuntimeError("No prior/current team data for fpl_team_id=%s." % fpl_team_id)

    def metric(name: str, current_name: Optional[str] = None) -> Tuple[float, float, float]:
        current_key = current_name or ("current_" + name)
        return blended_metric(
            nullable_float(prior.get(name)) if prior else None,
            nullable_float(current.get(current_key)) if current else None,
            prior_weight,
            current_weight,
        )

    ppm, eff_pw, eff_cw = metric("points_per_match")
    gfpm, _, _ = metric("goals_for_per_match")
    gapm, _, _ = metric("goals_against_per_match")
    cs, _, _ = metric("clean_sheet_rate")

    # Existing Day70A/70C math expects a season-total goal-difference value.
    # Convert the blended per-match signal to a 38-match equivalent so the
    # existing 0.01 signal coefficient remains on the same scale.
    prior_gd_per_match = None
    if prior and nullable_float(prior.get("goal_difference")) is not None:
        prior_gd_per_match = float(prior["goal_difference"]) / 38.0
    current_gd_per_match = (
        nullable_float(current.get("current_goal_difference_per_match")) if current else None
    )
    gdpm, _, _ = blended_metric(
        prior_gd_per_match,
        current_gd_per_match,
        prior_weight,
        current_weight,
    )

    return {
        "fpl_team_id": fpl_team_id,
        "points_per_match": ppm,
        "goals_for_per_match": gfpm,
        "goals_against_per_match": gapm,
        "clean_sheet_rate": cs,
        "goal_difference_equivalent_38": gdpm * 38.0,
        "effective_prior_weight": eff_pw,
        "effective_current_weight": eff_cw,
        "prior_available": prior is not None,
        "current_available": current is not None,
        "prior_fallback_applied": bool(prior.get("prior_fallback_applied")) if prior else False,
        "current_matches": int(current.get("current_matches", 0)) if current else 0,
    }


def build_match_preview(
    fixtures: pd.DataFrame,
    team_prior_lookup: Mapping[int, Mapping[str, Any]],
    current_team_lookup: Mapping[int, Mapping[str, Any]],
    prior_weight: float,
    current_weight: float,
    target_season: str,
    target_gw: int,
    scoreline_max_goals: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]]]:
    match_rows: List[Dict[str, Any]] = []
    scoreline_rows: List[Dict[str, Any]] = []
    team_context_cache: Dict[int, Dict[str, Any]] = {}

    for _, fixture in fixtures.iterrows():
        home_fpl = int(fixture["home_fpl_team_id"])
        away_fpl = int(fixture["away_fpl_team_id"])

        home = team_context_cache.setdefault(
            home_fpl,
            blended_team_record(
                home_fpl,
                team_prior_lookup,
                current_team_lookup,
                prior_weight,
                current_weight,
            ),
        )
        away = team_context_cache.setdefault(
            away_fpl,
            blended_team_record(
                away_fpl,
                team_prior_lookup,
                current_team_lookup,
                prior_weight,
                current_weight,
            ),
        )

        any_prior_fallback = bool(home["prior_fallback_applied"] or away["prior_fallback_applied"])

        # Feed the generic Day70A/70C math via its historical column contract.
        feature_row = pd.Series(
            {
                "home_effective_prev_season_points_per_match": home["points_per_match"],
                "away_effective_prev_season_points_per_match": away["points_per_match"],
                "home_effective_prev_season_goals_for_per_match": home["goals_for_per_match"],
                "away_effective_prev_season_goals_for_per_match": away["goals_for_per_match"],
                "home_effective_prev_season_goals_against_per_match": home["goals_against_per_match"],
                "away_effective_prev_season_goals_against_per_match": away["goals_against_per_match"],
                "home_effective_prev_season_clean_sheet_rate": home["clean_sheet_rate"],
                "away_effective_prev_season_clean_sheet_rate": away["clean_sheet_rate"],
                "home_effective_prev_season_goal_difference": home["goal_difference_equivalent_38"],
                "away_effective_prev_season_goal_difference": away["goal_difference_equivalent_38"],
            }
        )

        signal = compute_signal(feature_row)
        if not signal.get("valid"):
            raise RuntimeError(
                "Invalid blended match signal for fixture_id=%s: %s"
                % (fixture["fixture_id"], signal.get("missing_inputs"))
            )
        probs = probability_from_signal(
            float(signal["combined_signal"]),
            any_prior_fallback,
        )
        goal_est = estimate_expected_goals(feature_row)
        if not goal_est.get("valid"):
            raise RuntimeError(
                "Invalid blended expected goals for fixture_id=%s: %s"
                % (fixture["fixture_id"], goal_est.get("missing_inputs"))
            )
        grid = scoreline_grid(
            float(goal_est["expected_home_goals"]),
            float(goal_est["expected_away_goals"]),
            scoreline_max_goals,
        )

        base = {
            "target_season": target_season,
            "target_gw": target_gw,
            "prediction_mode": "early_season_blend",
            "fixture_id": int(fixture["fixture_id"]),
            "fpl_fixture_id": int(fixture["fpl_fixture_id"]),
            "kickoff_time": fixture.get("kickoff_time"),
            "home_team_id": int(fixture["home_team_id"]),
            "away_team_id": int(fixture["away_team_id"]),
            "home_fpl_team_id": home_fpl,
            "away_fpl_team_id": away_fpl,
            "home_team_name": fixture["home_team_name"],
            "away_team_name": fixture["away_team_name"],
            "home_team_short_name": fixture["home_team_short_name"],
            "away_team_short_name": fixture["away_team_short_name"],
            "prior_weight": prior_weight,
            "current_weight": current_weight,
            "home_current_matches": home["current_matches"],
            "away_current_matches": away["current_matches"],
            "home_prior_fallback_applied": home["prior_fallback_applied"],
            "away_prior_fallback_applied": away["prior_fallback_applied"],
            "any_prior_fallback_applied": any_prior_fallback,
            "home_blended_points_per_match": home["points_per_match"],
            "away_blended_points_per_match": away["points_per_match"],
            "home_blended_goals_for_per_match": home["goals_for_per_match"],
            "away_blended_goals_for_per_match": away["goals_for_per_match"],
            "home_blended_goals_against_per_match": home["goals_against_per_match"],
            "away_blended_goals_against_per_match": away["goals_against_per_match"],
            "home_blended_clean_sheet_rate": home["clean_sheet_rate"],
            "away_blended_clean_sheet_rate": away["clean_sheet_rate"],
            "combined_signal_home_minus_away": signal["combined_signal"],
            "expected_home_goals": goal_est["expected_home_goals"],
            "expected_away_goals": goal_est["expected_away_goals"],
            "expected_total_goals": round(
                float(goal_est["expected_home_goals"]) + float(goal_est["expected_away_goals"]),
                4,
            ),
        }

        match_row = dict(base)
        match_row.update(
            {
                "model_name": MATCH_MODEL_NAME,
                "home_win_probability": probs["home_win_probability"],
                "draw_probability": probs["draw_probability"],
                "away_win_probability": probs["away_win_probability"],
                "predicted_result_label": probs["predicted_result_label"],
                "confidence_score": probs["confidence_score"],
            }
        )
        match_rows.append(match_row)

        score_row = dict(base)
        score_row.update(
            {
                "model_name": SCORELINE_MODEL_NAME,
                "scoreline_home_win_probability": grid["scoreline_home_win_probability"],
                "scoreline_draw_probability": grid["scoreline_draw_probability"],
                "scoreline_away_win_probability": grid["scoreline_away_win_probability"],
                "scoreline_result_label": grid["scoreline_result_label"],
                "one_x_two_home_win_probability": probs["home_win_probability"],
                "one_x_two_draw_probability": probs["draw_probability"],
                "one_x_two_away_win_probability": probs["away_win_probability"],
                "one_x_two_predicted_result_label": probs["predicted_result_label"],
                "scoreline_label_matches_one_x_two_label": (
                    grid["scoreline_result_label"] == probs["predicted_result_label"]
                ),
                "score_grid_probability_sum": grid["score_grid_probability_sum"],
                "score_grid_tail_probability": grid["score_grid_tail_probability"],
            }
        )
        top = grid["top_scorelines"]
        for index in range(5):
            prefix = "top_%s_" % (index + 1)
            if index < len(top):
                score_row[prefix + "scoreline"] = top[index]["scoreline"]
                score_row[prefix + "scoreline_probability"] = top[index]["probability"]
                score_row[prefix + "home_goals"] = top[index]["home_goals"]
                score_row[prefix + "away_goals"] = top[index]["away_goals"]
            else:
                score_row[prefix + "scoreline"] = None
                score_row[prefix + "scoreline_probability"] = None
                score_row[prefix + "home_goals"] = None
                score_row[prefix + "away_goals"] = None
        scoreline_rows.append(score_row)

    return pd.DataFrame(match_rows), pd.DataFrame(scoreline_rows), team_context_cache


def scoreline_alignment_diagnostics(
    matches: pd.DataFrame,
    scorelines: pd.DataFrame,
) -> Dict[str, Any]:
    if matches.empty or scorelines.empty:
        return {
            "rows_compared": 0,
            "label_mismatch_rows": 0,
            "max_abs_probability_gap": None,
        }

    left = matches[
        [
            "fpl_fixture_id",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "predicted_result_label",
        ]
    ].copy()
    right = scorelines[
        [
            "fpl_fixture_id",
            "scoreline_home_win_probability",
            "scoreline_draw_probability",
            "scoreline_away_win_probability",
            "scoreline_result_label",
        ]
    ].copy()

    merged = left.merge(
        right,
        on="fpl_fixture_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(matches) or len(merged) != len(scorelines):
        raise RuntimeError(
            "Match/scoreline fixture coverage mismatch: matches=%s scorelines=%s merged=%s."
            % (len(matches), len(scorelines), len(merged))
        )

    gaps = pd.concat(
        [
            (
                pd.to_numeric(merged["home_win_probability"], errors="coerce")
                - pd.to_numeric(merged["scoreline_home_win_probability"], errors="coerce")
            ).abs(),
            (
                pd.to_numeric(merged["draw_probability"], errors="coerce")
                - pd.to_numeric(merged["scoreline_draw_probability"], errors="coerce")
            ).abs(),
            (
                pd.to_numeric(merged["away_win_probability"], errors="coerce")
                - pd.to_numeric(merged["scoreline_away_win_probability"], errors="coerce")
            ).abs(),
        ],
        ignore_index=True,
    )

    mismatch_rows = int(
        (
            merged["predicted_result_label"].astype(str)
            != merged["scoreline_result_label"].astype(str)
        ).sum()
    )

    return {
        "rows_compared": int(len(merged)),
        "label_mismatch_rows": mismatch_rows,
        "max_abs_probability_gap": float(gaps.max()) if len(gaps) else 0.0,
    }


# ---------- player target fixture adjustment ----------

def build_team_fixture_context(
    fixtures: pd.DataFrame,
    scorelines: pd.DataFrame,
) -> Dict[int, List[Dict[str, Any]]]:
    score_lookup = {
        int(row.fpl_fixture_id): row
        for row in scorelines.itertuples(index=False)
    }
    context: Dict[int, List[Dict[str, Any]]] = {}

    for fixture in fixtures.itertuples(index=False):
        score = score_lookup[int(fixture.fpl_fixture_id)]
        home = {
            "fixture_id": int(fixture.fixture_id),
            "fpl_fixture_id": int(fixture.fpl_fixture_id),
            "opponent_fpl_team_id": int(fixture.away_fpl_team_id),
            "opponent_team_short_name": fixture.away_team_short_name,
            "is_home": True,
            "expected_team_goals": float(score.expected_home_goals),
            "expected_opponent_goals": float(score.expected_away_goals),
        }
        away = {
            "fixture_id": int(fixture.fixture_id),
            "fpl_fixture_id": int(fixture.fpl_fixture_id),
            "opponent_fpl_team_id": int(fixture.home_fpl_team_id),
            "opponent_team_short_name": fixture.home_team_short_name,
            "is_home": False,
            "expected_team_goals": float(score.expected_away_goals),
            "expected_opponent_goals": float(score.expected_home_goals),
        }
        context.setdefault(int(fixture.home_fpl_team_id), []).append(home)
        context.setdefault(int(fixture.away_fpl_team_id), []).append(away)
    return context


def fixture_multiplier(
    position: str,
    expected_team_goals: float,
    expected_opponent_goals: float,
    league_mean_expected_goals: float,
    args: argparse.Namespace,
) -> Dict[str, float]:
    clean_sheet_probability = math.exp(-expected_opponent_goals)
    league_mean_cs = math.exp(-league_mean_expected_goals)
    attack_signal = clamp(
        expected_team_goals / league_mean_expected_goals,
        args.fixture_signal_min,
        args.fixture_signal_max,
    )
    defence_signal = clamp(
        clean_sheet_probability / league_mean_cs,
        args.fixture_signal_min,
        args.fixture_signal_max,
    )
    weights = FIXTURE_COMPONENT_WEIGHTS[position]
    neutral = 1.0 - weights["attack"] - weights["defence"]
    multiplier = (
        neutral
        + weights["attack"] * attack_signal
        + weights["defence"] * defence_signal
    )
    multiplier = clamp(
        multiplier,
        args.fixture_multiplier_min,
        args.fixture_multiplier_max,
    )
    return {
        "attack_signal": attack_signal,
        "defence_signal": defence_signal,
        "clean_sheet_probability": clean_sheet_probability,
        "fixture_multiplier": multiplier,
    }


def apply_availability_ceiling(
    predicted_points: float,
    expected_minutes_total: float,
    appearance_probability_per_fixture: float,
    chance_of_playing_next_round: Any,
) -> Dict[str, Any]:
    chance = nullable_float(chance_of_playing_next_round)
    result = {
        "predicted_points": predicted_points,
        "expected_minutes_total": expected_minutes_total,
        "official_availability_probability": None,
        "official_availability_workload_factor": 1.0,
        "official_availability_adjustment_applied": False,
    }
    if chance is None:
        return result
    if chance < 0 or chance > 100:
        raise RuntimeError("chance_of_playing_next_round must be within 0..100 or null.")
    official = chance / 100.0
    result["official_availability_probability"] = official
    baseline_app = clamp(appearance_probability_per_fixture, 0.0, 1.0)
    if baseline_app <= 1e-12 or official >= baseline_app - 1e-12:
        return result
    factor = official / baseline_app
    result.update(
        {
            "predicted_points": predicted_points * factor,
            "expected_minutes_total": expected_minutes_total * factor,
            "official_availability_workload_factor": factor,
            "official_availability_adjustment_applied": True,
        }
    )
    return result


def build_player_preview(
    players: pd.DataFrame,
    bootstrap_lookup: Mapping[int, Mapping[str, Any]],
    prior_lookup: Mapping[int, Mapping[str, Any]],
    position_fallbacks: Mapping[str, Mapping[str, float]],
    current_lookup: Mapping[int, Mapping[str, Any]],
    fixture_context: Mapping[int, Sequence[Mapping[str, Any]]],
    scorelines: pd.DataFrame,
    prior_weight: float,
    current_weight: float,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if scorelines.empty:
        raise RuntimeError("Scoreline preview is empty.")
    league_mean_expected_goals = float(
        pd.concat(
            [
                pd.to_numeric(scorelines["expected_home_goals"], errors="coerce"),
                pd.to_numeric(scorelines["expected_away_goals"], errors="coerce"),
            ],
            ignore_index=True,
        ).mean()
    )
    if not math.isfinite(league_mean_expected_goals) or league_mean_expected_goals <= 0:
        raise RuntimeError("League mean expected goals is invalid.")

    rows: List[Dict[str, Any]] = []
    prior_match_count = 0
    position_fallback_count = 0
    no_current_minutes_count = 0
    availability_adjusted_count = 0

    for player in players.itertuples(index=False):
        fpl_id = int(player.fpl_player_id)
        position = str(player.position)
        if position not in VALID_POSITIONS:
            raise RuntimeError("Unsupported player position=%s fpl_player_id=%s." % (position, fpl_id))

        prior = prior_lookup.get(fpl_id)
        if prior is None:
            prior_values = position_fallbacks.get(position)
            if prior_values is None:
                raise RuntimeError("No prior or position fallback for fpl_player_id=%s." % fpl_id)
            prior_source = "position_median_frozen_gw1_prior"
            prior_fallback_used = True
            position_fallback_count += 1
        else:
            prior_values = prior
            prior_source = "frozen_gw1_player_prior"
            prior_fallback_used = False
            prior_match_count += 1

        current = current_lookup.get(fpl_id, {})
        if float(current.get("current_total_minutes", 0.0)) <= 0:
            no_current_minutes_count += 1

        prior_pp90 = nullable_float(prior_values.get("prior_points_per90")) or 0.0
        prior_minutes = clamp(nullable_float(prior_values.get("prior_expected_minutes")) or 0.0, 0.0, 90.0)
        prior_app = clamp(nullable_float(prior_values.get("prior_appearance_probability")) or 0.0, 0.0, 1.0)
        prior_start = clamp(nullable_float(prior_values.get("prior_start_probability")) or 0.0, 0.0, prior_app)

        current_pp90 = nullable_float(current.get("current_points_per90")) or 0.0
        current_minutes = clamp(float(current.get("current_expected_minutes", 0.0)), 0.0, 90.0)
        current_app = clamp(float(current.get("current_appearance_probability", 0.0)), 0.0, 1.0)
        current_start = clamp(
            float(current.get("current_start_probability_proxy", 0.0)),
            0.0,
            current_app,
        )

        blended_app = clamp(prior_weight * prior_app + current_weight * current_app, 0.0, 1.0)
        blended_start = clamp(
            prior_weight * prior_start + current_weight * current_start,
            0.0,
            blended_app,
        )
        blended_minutes_per_fixture = clamp(
            prior_weight * prior_minutes + current_weight * current_minutes,
            0.0,
            90.0,
        )

        team_fixtures = list(fixture_context.get(int(player.fpl_team_id), []))
        fixture_count = len(team_fixtures)
        prior_component_total = 0.0
        current_component_total = 0.0
        multipliers: List[float] = []
        fixture_ids: List[int] = []
        opponents: List[str] = []

        for fixture in team_fixtures:
            adj = fixture_multiplier(
                position=position,
                expected_team_goals=float(fixture["expected_team_goals"]),
                expected_opponent_goals=float(fixture["expected_opponent_goals"]),
                league_mean_expected_goals=league_mean_expected_goals,
                args=args,
            )
            multiplier = float(adj["fixture_multiplier"])
            multipliers.append(multiplier)
            fixture_ids.append(int(fixture["fpl_fixture_id"]))
            opponents.append(str(fixture["opponent_team_short_name"]))

            prior_base = prior_pp90 * prior_minutes / 90.0
            current_base = current_pp90 * current_minutes / 90.0
            prior_component_total += prior_base * multiplier
            current_component_total += current_base * multiplier

        raw_blended_points = (
            prior_weight * prior_component_total
            + current_weight * current_component_total
        )
        raw_blended_points = max(0.0, raw_blended_points)

        bootstrap = bootstrap_lookup.get(fpl_id, {})
        chance = bootstrap.get("chance_of_playing_next_round")
        availability = apply_availability_ceiling(
            predicted_points=raw_blended_points,
            expected_minutes_total=blended_minutes_per_fixture * fixture_count,
            appearance_probability_per_fixture=blended_app,
            chance_of_playing_next_round=chance,
        )
        if availability["official_availability_adjustment_applied"]:
            availability_adjusted_count += 1

        guarded_points = clamp(
            float(availability["predicted_points"]),
            args.prediction_points_min,
            args.prediction_points_max * max(1, fixture_count),
        )

        current_meta = bootstrap_lookup.get(fpl_id, {})
        rows.append(
            {
                "target_season": args.season,
                "target_gw": args.target_gw,
                "prediction_mode": "early_season_blend",
                "model_name": PLAYER_MODEL_NAME,
                "pipeline_version": PIPELINE_VERSION,
                "player_id": int(player.player_id),
                "fpl_player_id": fpl_id,
                "player_name": ("%s %s" % (player.first_name, player.second_name)).strip(),
                "web_name": current_meta.get("web_name") or player.web_name,
                "team_id": int(player.team_id),
                "fpl_team_id": int(player.fpl_team_id),
                "team_name": player.team_name,
                "team_short_name": player.team_short_name,
                "position": position,
                "now_cost": nullable_int(current_meta.get("now_cost")) or int(player.now_cost),
                "status": current_meta.get("status") or player.status,
                "chance_of_playing_next_round": chance,
                "news": current_meta.get("news"),
                "news_added": current_meta.get("news_added"),
                "fixture_count": fixture_count,
                "fpl_fixture_ids": ",".join(str(v) for v in fixture_ids),
                "opponents": ",".join(opponents),
                "prior_weight": prior_weight,
                "current_weight": current_weight,
                "prior_source": prior_source,
                "prior_fallback_used": prior_fallback_used,
                "prior_points_per90": prior_pp90,
                "prior_expected_minutes_per_fixture": prior_minutes,
                "prior_appearance_probability": prior_app,
                "prior_start_probability": prior_start,
                "current_required_gws": int(args.target_gw - 1),
                "current_actual_rows": int(current.get("current_actual_rows", 0)),
                "current_actual_gws_observed": current.get("current_actual_gws_observed", ""),
                "current_history_complete": bool(current.get("current_history_complete", False)),
                "current_total_minutes": float(current.get("current_total_minutes", 0.0)),
                "current_total_points": float(current.get("current_total_points", 0.0)),
                "current_appearances": int(current.get("current_appearances", 0)),
                "current_points_per90": current_pp90,
                "current_points_per_gw": float(current.get("current_points_per_gw", 0.0)),
                "current_expected_minutes_per_fixture": current_minutes,
                "current_appearance_probability": current_app,
                "current_start_probability_proxy": current_start,
                "blended_appearance_probability": blended_app,
                "blended_start_probability": blended_start,
                "blended_expected_minutes_per_fixture": blended_minutes_per_fixture,
                "expected_minutes_total": float(availability["expected_minutes_total"]),
                "mean_fixture_multiplier": float(sum(multipliers) / len(multipliers)) if multipliers else 0.0,
                "prior_component_points": prior_component_total,
                "current_component_points": current_component_total,
                "pre_availability_blended_points": raw_blended_points,
                "official_availability_probability": availability["official_availability_probability"],
                "official_availability_workload_factor": availability["official_availability_workload_factor"],
                "official_availability_adjustment_applied": availability["official_availability_adjustment_applied"],
                "predicted_points": guarded_points,
                "prediction_write_allowed": False,
                "production_ready": False,
                "preview_only": True,
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["position", "predicted_points", "web_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    diagnostics = {
        "current_player_rows": int(len(players)),
        "frozen_prior_matches": prior_match_count,
        "position_prior_fallback_rows": position_fallback_count,
        "no_current_minutes_rows": no_current_minutes_count,
        "partial_current_history_rows": int((~out["current_history_complete"]).sum()),
        "availability_adjusted_rows": availability_adjusted_count,
        "blank_player_rows": int((out["fixture_count"] == 0).sum()),
        "double_fixture_player_rows": int((out["fixture_count"] > 1).sum()),
        "prediction_null_rows": int(out["predicted_points"].isna().sum()),
    }
    return out, diagnostics


# ---------- artifact / summary ----------

def build_summary_markdown(manifest: Mapping[str, Any], players: pd.DataFrame, matches: pd.DataFrame) -> str:
    top = players.sort_values("predicted_points", ascending=False).head(20)
    lines = [
        "# Early Season Prediction Preview",
        "",
        "- Status: `%s`" % manifest["status"],
        "- Season: `%s`" % manifest["season"],
        "- Target GW: `%s`" % manifest["target_gw"],
        "- Prediction mode: `%s`" % manifest["prediction_mode"],
        "- Prior/current weights: `%.2f / %.2f`" % (
            manifest["weights"]["prior"],
            manifest["weights"]["current"],
        ),
        "- Current actual GWs: `%s`" % manifest["current_actual_gws"],
        "- Player rows: `%s`" % len(players),
        "- Match rows: `%s`" % len(matches),
        "- Scoreline/1X2 label mismatches: `%s`" % (
            manifest.get("scoreline_alignment", {}).get("label_mismatch_rows")
        ),
        "- Max scoreline/1X2 probability gap: `%.6f`" % float(
            manifest.get("scoreline_alignment", {}).get("max_abs_probability_gap") or 0.0
        ),
        "- DB prediction writes: `false`",
        "",
        "## Top 20 player preview",
        "",
        "| Rank | Player | Pos | Team | Opponent(s) | Pred | Prior | Current |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            "| %s | %s | %s | %s | %s | %.3f | %.3f | %.3f |"
            % (
                idx,
                row.web_name,
                row.position,
                row.team_short_name,
                row.opponents or "BLANK",
                row.predicted_points,
                row.prior_component_points,
                row.current_component_points,
            )
        )
    lines.extend(
        [
            "",
            "## Safety contract",
            "",
            "- The immutable GW1 frozen prior artifacts are read only.",
            "- Current-season actuals are read from canonical `player_gw_stats` and scored fixtures only.",
            "- This preview does not write to `predictions` or `match_predictions`.",
            "- The same pipeline is intended for GW2-GW5; weights come from the existing prediction-mode resolver.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    mode = resolve_mode(args)
    metadata = mode["metadata"]
    prior_weight = float(metadata["default_prior_weight"])
    current_weight = float(metadata["default_current_weight"])

    frozen = load_prior_files(args)
    current_teams = load_current_teams(args.season)
    current_players = load_current_players(args.season)
    target_fixtures = load_target_fixtures(args.season, args.target_gw)
    past_fixtures = load_past_fixtures(args.season, args.target_gw)
    player_actuals = load_player_actuals(args.season, args.target_gw)

    coverage = validate_actual_gw_coverage(player_actuals, args.target_gw)
    expected_gws = coverage["expected_actual_gws"]

    # Team fixture coverage is score-based rather than finished-flag based so
    # provisional operational results can drive PRE while FINAL evaluation remains separate.
    observed_fixture_gws = sorted(
        set(pd.to_numeric(past_fixtures.get("gw"), errors="coerce").dropna().astype(int).tolist())
    )
    missing_fixture_gws = [gw for gw in expected_gws if gw not in observed_fixture_gws]
    if missing_fixture_gws:
        raise RuntimeError(
            "Scored fixture actuals are missing required GWs: %s" % missing_fixture_gws
        )

    bootstrap, bootstrap_source = load_bootstrap(args)
    bootstrap_lookup = build_bootstrap_player_lookup(bootstrap)
    db_player_ids = set(current_players["fpl_player_id"].astype(int).tolist())
    bootstrap_player_ids = set(bootstrap_lookup.keys())
    if db_player_ids != bootstrap_player_ids:
        missing_from_db = sorted(bootstrap_player_ids - db_player_ids)
        stale_in_db = sorted(db_player_ids - bootstrap_player_ids)
        raise RuntimeError(
            "Current canonical player pool does not match bootstrap-static. "
            "Refresh the season-scoped bootstrap first. missing_from_db=%s stale_in_db=%s"
            % (missing_from_db[:20], stale_in_db[:20])
        )

    team_prior_lookup = build_team_prior_lookup(frozen["teams"])
    player_prior_lookup = build_player_prior_lookup(frozen["players"])
    position_fallbacks = build_position_prior_fallbacks(player_prior_lookup)
    current_team_lookup = build_current_team_stats(past_fixtures)
    current_player_lookup = build_current_player_stats(player_actuals, args.target_gw)

    matches, scorelines, blended_team_context = build_match_preview(
        fixtures=target_fixtures,
        team_prior_lookup=team_prior_lookup,
        current_team_lookup=current_team_lookup,
        prior_weight=prior_weight,
        current_weight=current_weight,
        target_season=args.season,
        target_gw=args.target_gw,
        scoreline_max_goals=args.scoreline_max_goals,
    )
    fixture_context = build_team_fixture_context(target_fixtures, scorelines)
    players, player_diagnostics = build_player_preview(
        players=current_players,
        bootstrap_lookup=bootstrap_lookup,
        prior_lookup=player_prior_lookup,
        position_fallbacks=position_fallbacks,
        current_lookup=current_player_lookup,
        fixture_context=fixture_context,
        scorelines=scorelines,
        prior_weight=prior_weight,
        current_weight=current_weight,
        args=args,
    )

    blockers: List[str] = []
    warnings: List[str] = []

    if len(current_teams) != 20:
        blockers.append("Expected 20 current teams; got %s." % len(current_teams))
    if len(target_fixtures) != 10:
        warnings.append("Target GW fixture count is %s, not 10." % len(target_fixtures))
    if len(players) != len(current_players):
        blockers.append("Player preview row count does not equal current player pool.")
    if player_diagnostics["prediction_null_rows"] > 0:
        blockers.append("Some player predictions are null.")
    if int(matches["fpl_fixture_id"].duplicated().sum()) > 0:
        blockers.append("Duplicate match fixture predictions found.")

    probability_error = (
        matches[["home_win_probability", "draw_probability", "away_win_probability"]]
        .sum(axis=1)
        .sub(1.0)
        .abs()
        .max()
    )
    if float(probability_error) > 1e-5:
        blockers.append("Match probability sums are invalid.")

    scoreline_alignment = scoreline_alignment_diagnostics(matches, scorelines)
    if scoreline_alignment["label_mismatch_rows"] > 0:
        warnings.append(
            "Match 1X2 and scoreline result labels disagree for %s fixture(s)."
            % scoreline_alignment["label_mismatch_rows"]
        )

    run_id = "%s_%s_gw%s_%s" % (
        PIPELINE_VERSION,
        args.season,
        args.target_gw,
        utc_stamp(),
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else default_output_dir(args.season, args.target_gw, run_id)
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    bootstrap_path = out_dir / "bootstrap_snapshot.json"
    json_dump(bootstrap_path, bootstrap)

    match_path = out_dir / "match_predictions_preview.csv"
    scoreline_path = out_dir / "scoreline_preview.csv"
    player_path = out_dir / "player_predictions_preview.csv"
    players.to_csv(player_path, index=False)
    matches.to_csv(match_path, index=False)
    scorelines.to_csv(scoreline_path, index=False)

    status = "PASS_PREVIEW" if not blockers else "BLOCKED"
    manifest: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "status": status,
        "season": args.season,
        "prior_season": args.prior_season,
        "target_gw": args.target_gw,
        "prediction_mode": mode["resolved_prediction_mode"],
        "weights": {"prior": prior_weight, "current": current_weight},
        "current_actual_gws": expected_gws,
        "current_player_actual_rows": int(len(player_actuals)),
        "past_scored_fixture_rows": int(len(past_fixtures)),
        "past_finished_fixture_rows": int(pd.Series(past_fixtures.get("finished", [])).fillna(False).astype(bool).sum()),
        "past_provisional_scored_fixture_rows": int(len(past_fixtures) - pd.Series(past_fixtures.get("finished", [])).fillna(False).astype(bool).sum()),
        "target_fixture_rows": int(len(target_fixtures)),
        "current_player_pool_rows": int(len(current_players)),
        "frozen_player_prior_rows": int(len(frozen["players"])),
        "frozen_team_fixture_rows": int(len(frozen["teams"])),
        "bootstrap_source": bootstrap_source,
        "prior_artifacts": {
            "directory": str(frozen["dir"]),
            "player_preview": str(frozen["player_path"]),
            "player_preview_sha256": frozen["player_sha256"],
            "effective_match_features": str(frozen["team_path"]),
            "effective_match_features_sha256": frozen["team_sha256"],
        },
        "outputs": {
            "player_predictions_preview": player_path.name,
            "match_predictions_preview": match_path.name,
            "scoreline_preview": scoreline_path.name,
            "bootstrap_snapshot": bootstrap_path.name,
        },
        "player_diagnostics": player_diagnostics,
        "match_probability_max_sum_error": float(probability_error),
        "scoreline_alignment": scoreline_alignment,
        "blended_team_context_rows": len(blended_team_context),
        "blockers": blockers,
        "warnings": warnings,
        "database_prediction_write": False,
        "writes_frozen_gw1_artifacts": False,
        "preview_only": True,
        "publish_gate_implemented": False,
        "notes": [
            "GW1 frozen prior artifacts remain the immutable prior anchor for GW2-GW5.",
            "Current team/player components use scored canonical actuals through target_gw-1.",
            "Current start probability is a v0 60+ minute proxy; official starts can replace it later without changing the interface.",
            "Provisional scored fixtures are operational PRE inputs; GW POST FINAL evidence remains a separate contract.",
        ],
    }

    manifest_path = out_dir / "run_manifest.json"
    json_dump(manifest_path, manifest)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(build_summary_markdown(manifest, players, matches), encoding="utf-8")

    print("=== Early Season Prediction Preview ===")
    print("status:", status)
    print("run_id:", run_id)
    print("output_dir:", out_dir)
    print("season:", args.season)
    print("target_gw:", args.target_gw)
    print("prediction_mode:", mode["resolved_prediction_mode"])
    print("prior_weight:", prior_weight)
    print("current_weight:", current_weight)
    print("current_actual_gws:", expected_gws)
    print("current_player_pool_rows:", len(current_players))
    print("frozen_prior_matches:", player_diagnostics["frozen_prior_matches"])
    print("position_prior_fallback_rows:", player_diagnostics["position_prior_fallback_rows"])
    print("player_prediction_rows:", len(players))
    print("match_prediction_rows:", len(matches))
    print("scoreline_rows:", len(scorelines))
    print("scoreline_label_mismatch_rows:", scoreline_alignment["label_mismatch_rows"])
    print(
        "scoreline_one_x_two_max_probability_gap:",
        round(float(scoreline_alignment["max_abs_probability_gap"] or 0.0), 6),
    )
    print("database_prediction_write: False")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print("-", warning)
    if blockers:
        print("blockers:")
        for blocker in blockers:
            print("-", blocker)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
