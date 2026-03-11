# backend/app/api/routes/models.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models.prediction import Prediction

router = APIRouter(prefix="/models", tags=["models"])

TASK_TYPE_DEFAULT = "player_points"


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

    Day30 upgrade:
    - include task_type metadata (default: player_points)
    """
    rows = (
        db.query(Prediction.model_name)
        .filter(Prediction.model_name.isnot(None))
        .filter(Prediction.model_name != "")
        .distinct()
        .order_by(Prediction.model_name.asc())
        .all()
    )

    model_names = [r[0] for r in rows if r and r[0]]

    models = [
        {
            "model_name": name,
            "label": name,  # future: prettier labels
            "task_type": TASK_TYPE_DEFAULT,  # Day30: registry metadata
            "source": "predictions_distinct",
            "is_active": True,  # future: real registry flag
            "notes": None,
        }
        for name in model_names
    ]

    # Reserved for future: active_only filtering once is_active is real
    # if active_only:
    #     models = [m for m in models if m["is_active"]]

    return {
        "models": models,
        "meta": {
            "count": len(models),
            "source": "predictions_distinct",
        },
    }