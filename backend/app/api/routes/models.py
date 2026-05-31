# backend/app/api/routes/models.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models.prediction import Prediction
from app.utils.model_metadata_store import (
    list_model_metadata_artifacts,
    maybe_load_model_metadata_artifact,
)

router = APIRouter(prefix="/models", tags=["models"])

TASK_TYPE_DEFAULT = "player_points"


def _serialize_model_row(model_name: str, source: str):
    meta_artifact = maybe_load_model_metadata_artifact(model_name)

    return {
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
            "feature_version": meta_artifact.feature_version if meta_artifact else None,
            "training_window_start_gw": meta_artifact.training_window_start_gw if meta_artifact else None,
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
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Prediction.model_name)
        .filter(Prediction.model_name.isnot(None))
        .filter(Prediction.model_name != "")
        .distinct()
        .order_by(Prediction.model_name.asc())
        .all()
    )
    names_from_predictions = {r[0] for r in rows if r and r[0]}

    metadata_models = list_model_metadata_artifacts(task_type="player_points")
    names_from_metadata = {m.model_name for m in metadata_models}

    all_names = sorted(names_from_predictions | names_from_metadata)

    models = []
    for name in all_names:
        source = "predictions_distinct" if name in names_from_predictions else "model_metadata"
        row = _serialize_model_row(name, source)
        if active_only and not row["is_active"]:
            continue
        models.append(row)

    return {
        "models": models,
        "meta": {
            "count": len(models),
            "source": "predictions_distinct_plus_model_metadata",
            "active_only": active_only,
        },
    }
