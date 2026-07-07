from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from app.core.season import get_current_season
from app.utils.model_metadata_store import list_model_metadata_artifacts


router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _metric_value(metrics: Dict[str, float], keys: List[str]) -> Optional[float]:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _artifact_value(artifact: Any, attr_name: str, default: Any = None) -> Any:
    return getattr(artifact, attr_name, default)


def _artifact_matches_season(artifact: Any, season: str) -> bool:
    artifact_season = _artifact_value(artifact, "season")
    # Keep legacy metadata artifacts visible until all metadata has an explicit season.
    return artifact_season in (None, "", season)


def _serialize_artifact(artifact: Any, season: str) -> Dict[str, Any]:
    metrics = artifact.metrics_summary or {}
    artifact_season = _artifact_value(artifact, "season")

    return {
        "season": artifact_season or season,
        "metadata_season": artifact_season,
        "training_season_start": _artifact_value(artifact, "training_season_start"),
        "training_season_end": _artifact_value(artifact, "training_season_end"),
        "evaluation_season": _artifact_value(artifact, "evaluation_season"),
        "model_name": artifact.model_name,
        "task_type": artifact.task_type,
        "feature_version": artifact.feature_version,
        "status": artifact.status,
        "is_active": artifact.is_active,
        "is_production_default": artifact.is_production_default,
        "selected_reason": artifact.selected_reason,
        "notes": artifact.notes,
        "training_window_start_gw": artifact.training_window_start_gw,
        "training_window_end_gw": artifact.training_window_end_gw,
        "evaluation_start_gw": artifact.evaluation_start_gw,
        "evaluation_end_gw": artifact.evaluation_end_gw,
        "metrics_summary": metrics,
        "updated_at": artifact.updated_at.isoformat(),
    }


def _player_sort_key(row: Dict[str, Any]):
    val_mae = _metric_value(row["metrics_summary"], ["val_mae", "overall_mae", "validation_mae"])
    return (
        0 if row["is_active"] else 1,
        0 if row["is_production_default"] else 1,
        val_mae if val_mae is not None else 9999.0,
        row["model_name"],
    )


def _match_result_sort_key(row: Dict[str, Any]):
    logloss = _metric_value(row["metrics_summary"], ["val_logloss", "logloss"])
    acc = _metric_value(row["metrics_summary"], ["val_accuracy", "accuracy"])
    return (
        0 if row["is_active"] else 1,
        0 if row["is_production_default"] else 1,
        logloss if logloss is not None else 9999.0,
        -(acc if acc is not None else -9999.0),
        row["model_name"],
    )


def _match_goals_sort_key(row: Dict[str, Any]):
    avg_goal_mae = _metric_value(row["metrics_summary"], ["avg_goal_mae", "home_goals_mae"])
    return (
        0 if row["is_active"] else 1,
        0 if row["status"] == "experimental" else 1,
        avg_goal_mae if avg_goal_mae is not None else 9999.0,
        row["model_name"],
    )


def _filter_rows(rows: List[Dict[str, Any]], active_only: bool) -> List[Dict[str, Any]]:
    if not active_only:
        return rows
    return [row for row in rows if row["is_active"]]


def _load_rows_for_task(task_type: str, season: str) -> List[Dict[str, Any]]:
    artifacts = [
        artifact
        for artifact in list_model_metadata_artifacts(task_type=task_type)
        if _artifact_matches_season(artifact, season)
    ]
    return [_serialize_artifact(artifact, season) for artifact in artifacts]


@router.get("/summary")
def get_evaluation_summary(
    active_only: bool = Query(False, description="Return only active models."),
    season: Optional[str] = Query(None, description="Season scope. Defaults to current season."),
):
    resolved_season = season or get_current_season()

    player_rows = _load_rows_for_task("player_points", resolved_season)
    match_result_rows = _load_rows_for_task("match_result", resolved_season)
    match_goals_rows = _load_rows_for_task("match_goals", resolved_season)

    player_rows = _filter_rows(player_rows, active_only)
    match_result_rows = _filter_rows(match_result_rows, active_only)
    match_goals_rows = _filter_rows(match_goals_rows, active_only)

    player_rows = sorted(player_rows, key=_player_sort_key)
    match_result_rows = sorted(match_result_rows, key=_match_result_sort_key)
    match_goals_rows = sorted(match_goals_rows, key=_match_goals_sort_key)

    player_default = next((row for row in player_rows if row["is_production_default"]), None)
    match_default = next((row for row in match_result_rows if row["is_production_default"]), None)

    production_defaults = {
        "player_default_model": player_default["model_name"] if player_default else None,
        "match_default_model": match_default["model_name"] if match_default else None,
        "player_backup_model": next(
            (
                row["model_name"]
                for row in player_rows
                if row["is_active"] and not row["is_production_default"]
            ),
            None,
        ),
        "experimental_goals_model": next(
            (row["model_name"] for row in match_goals_rows if row["status"] == "experimental"),
            None,
        ),
    }

    return {
        "season": resolved_season,
        "production_defaults": production_defaults,
        "player_models": player_rows,
        "match_result_models": match_result_rows,
        "match_goals_models": match_goals_rows,
        "meta": {
            "season": resolved_season,
            "active_only": active_only,
            "player_model_count": len(player_rows),
            "match_result_model_count": len(match_result_rows),
            "match_goals_model_count": len(match_goals_rows),
            "source": "model_metadata_artifacts",
            "metadata_season_filter": "matching_season_or_legacy_without_season",
        },
    }


@router.get("/player-models")
def get_player_model_evaluation(
    active_only: bool = Query(False, description="Return only active models."),
    season: Optional[str] = Query(None, description="Season scope. Defaults to current season."),
):
    resolved_season = season or get_current_season()
    rows = _load_rows_for_task("player_points", resolved_season)
    rows = _filter_rows(rows, active_only)
    rows = sorted(rows, key=_player_sort_key)

    return {
        "season": resolved_season,
        "models": rows,
        "meta": {
            "season": resolved_season,
            "active_only": active_only,
            "count": len(rows),
            "task_type": "player_points",
            "source": "model_metadata_artifacts",
            "metadata_season_filter": "matching_season_or_legacy_without_season",
        },
    }


@router.get("/match-models")
def get_match_model_evaluation(
    active_only: bool = Query(False, description="Return only active models."),
    season: Optional[str] = Query(None, description="Season scope. Defaults to current season."),
):
    resolved_season = season or get_current_season()

    result_rows = _load_rows_for_task("match_result", resolved_season)
    goals_rows = _load_rows_for_task("match_goals", resolved_season)

    result_rows = _filter_rows(result_rows, active_only)
    goals_rows = _filter_rows(goals_rows, active_only)

    result_rows = sorted(result_rows, key=_match_result_sort_key)
    goals_rows = sorted(goals_rows, key=_match_goals_sort_key)

    return {
        "season": resolved_season,
        "match_result_models": result_rows,
        "match_goals_models": goals_rows,
        "meta": {
            "season": resolved_season,
            "active_only": active_only,
            "match_result_count": len(result_rows),
            "match_goals_count": len(goals_rows),
            "source": "model_metadata_artifacts",
            "metadata_season_filter": "matching_season_or_legacy_without_season",
        },
    }
