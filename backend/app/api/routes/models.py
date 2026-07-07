# backend/app/api/routes/models.py
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.season import get_current_season
from app.models.prediction import Prediction
from app.utils.model_metadata_store import (
    list_model_metadata_artifacts,
    maybe_load_model_metadata_artifact,
)

router = APIRouter(prefix="/models", tags=["models"])

TASK_TYPE_DEFAULT = "player_points"


def _artifact_value(artifact: Any, attr_name: str, default: Any = None) -> Any:
    if artifact is None:
        return default
    return getattr(artifact, attr_name, default)


def _artifact_matches_season(artifact: Any, season: str) -> bool:
    artifact_season = _artifact_value(artifact, "season")
    # Keep legacy metadata artifacts visible until all metadata has an explicit season.
    return artifact_season in (None, "", season)


def _serialize_model_row(model_name: str, source: str, season: str) -> Dict[str, Any]:
    meta_artifact = maybe_load_model_metadata_artifact(model_name)
    metadata_season = _artifact_value(meta_artifact, "season")

    return {
        "season": metadata_season or season,
        "model_name": model_name,
        "label": model_name,
        "task_type": meta_artifact.task_type if meta_artifact else TASK_TYPE_DEFAULT,
        "source": source,
        "status": meta_artifact.status if meta_artifact else "experimental",
        "is_active": meta_artifact.is_active if meta_artifact else False,
        "is_production_default": meta_artifact.is_production_default if meta_artifact else False,
        "selected_reason": meta_artifact.selected_reason if meta_artifact else None,
        "notes": meta_artifact.notes if meta_artifact else None,
        "metadata": {
            "season": metadata_season or season,
            "metadata_season": metadata_season,
            "training_season_start": _artifact_value(meta_artifact, "training_season_start"),
            "training_season_end": _artifact_value(meta_artifact, "training_season_end"),
            "evaluation_season": _artifact_value(meta_artifact, "evaluation_season"),
            "feature_version": meta_artifact.feature_version if meta_artifact else None,
            "training_window_start_gw": (
                meta_artifact.training_window_start_gw if meta_artifact else None
            ),
            "training_window_end_gw": meta_artifact.training_window_end_gw if meta_artifact else None,
            "evaluation_start_gw": meta_artifact.evaluation_start_gw if meta_artifact else None,
            "evaluation_end_gw": meta_artifact.evaluation_end_gw if meta_artifact else None,
            "metrics_summary": meta_artifact.metrics_summary if meta_artifact else {},
            "status": meta_artifact.status if meta_artifact else "experimental",
            "is_active": meta_artifact.is_active if meta_artifact else False,
            "is_production_default": meta_artifact.is_production_default if meta_artifact else False,
            "selected_reason": meta_artifact.selected_reason if meta_artifact else None,
            "notes": meta_artifact.notes if meta_artifact else None,
            "updated_at": meta_artifact.updated_at.isoformat() if meta_artifact else None,
        },
    }


@router.get("")
def list_models(
    active_only: bool = Query(False),
    season: Optional[str] = Query(None, description="Season scope. Defaults to current season."),
    db: Session = Depends(get_db),
):
    resolved_season = season or get_current_season()

    rows = (
        db.query(Prediction.model_name)
        .filter(Prediction.season == resolved_season)
        .filter(Prediction.model_name.isnot(None))
        .filter(Prediction.model_name != "")
        .distinct()
        .order_by(Prediction.model_name.asc())
        .all()
    )
    names_from_predictions = {r[0] for r in rows if r and r[0]}

    metadata_models = [
        artifact
        for artifact in list_model_metadata_artifacts(task_type="player_points")
        if _artifact_matches_season(artifact, resolved_season)
    ]
    names_from_metadata = {m.model_name for m in metadata_models}

    all_names = sorted(names_from_predictions | names_from_metadata)

    models = []
    for name in all_names:
        source = "predictions_distinct" if name in names_from_predictions else "model_metadata"
        row = _serialize_model_row(name, source, resolved_season)
        if active_only and not row["is_active"]:
            continue
        models.append(row)

    return {
        "season": resolved_season,
        "models": models,
        "meta": {
            "season": resolved_season,
            "count": len(models),
            "source": "predictions_distinct_plus_model_metadata",
            "active_only": active_only,
            "metadata_season_filter": "matching_season_or_legacy_without_season",
        },
    }
