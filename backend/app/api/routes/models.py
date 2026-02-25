# backend/app/api/routes/models.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.db import get_db
from app.models.prediction import Prediction

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(
    active_only: bool = Query(False, description="Reserved for future use"),
    db: Session = Depends(get_db),
):
    """
    Day23 model registry (dynamic list, lightweight version)

    Current source of truth:
    - distinct Prediction.model_name from predictions table

    Returns a normalized list so frontend can render dropdown options.
    """
    # Distinct model names from predictions table
    rows = (
        db.query(Prediction.model_name)
        .filter(Prediction.model_name.isnot(None))
        .filter(Prediction.model_name != "")
        .distinct()
        .order_by(Prediction.model_name.asc())
        .all()
    )

    model_names = [r[0] for r in rows if r and r[0]]

    # lightweight normalized payload (future-proof shape)
    models = [
        {
            "model_name": name,
            "label": name,  # future: prettier labels
            "source": "predictions_distinct",
            "is_active": True,  # future: real registry flag
            "notes": None,
        }
        for name in model_names
    ]

    return {
        "models": models,
        "meta": {
            "count": len(models),
            "source": "predictions_distinct",
        },
    }