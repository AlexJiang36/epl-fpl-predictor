from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func as sa_func

from app.core.db import get_db
from app.models.prediction import Prediction
from app.models.player_gw_stat import PlayerGameweekStat
from app.models.gameweek import Gameweek

router = APIRouter(prefix="/predictions", tags=["predictions"])
MODEL_NAME = "baseline_rollavg_v0"


@router.post("/baseline/run")
def run_baseline(
    target_gw: Optional[int] = None,
    window: int = Query(default=5, ge=1, le=10),
    db: Session = Depends(get_db),
):
    """
    Baseline: predict points as the average total_points over last N finished GWs.
    Writes results into predictions table (upsert by player_id+target_gw+model_name).
    """
    # 1) decide target_gw: if not provided, use next gameweek
    if target_gw is None:
        nxt = db.execute(select(Gameweek).where(Gameweek.is_next == True)).scalars().first()
        if nxt is None:
            return {"error": "No next gameweek found. Run /gameweeks/ingest/fpl first."}
        target_gw = nxt.gw

    # 2) pick last N finished gameweeks (by gw desc)
    finished_gws = db.execute(
        select(Gameweek.gw)
        .where(Gameweek.is_finished == True)
        .order_by(Gameweek.gw.desc())
        .limit(window)
    ).scalars().all()

    if len(finished_gws) == 0:
        return {"error": "No finished gameweeks found. Ingest gameweeks first."}

    # we want ascending for readability in response
    finished_gws_sorted = sorted(finished_gws)

    # 3) aggregate avg points per player over those GWs
    rows = db.execute(
        select(
            PlayerGameweekStat.player_id,
            sa_func.avg(PlayerGameweekStat.total_points).label("avg_points"),
        )
        .where(PlayerGameweekStat.gw.in_(finished_gws))
        .group_by(PlayerGameweekStat.player_id)
    ).all()

    inserted = 0
    updated = 0

    # 4) upsert into predictions
    for player_id, avg_points in rows:
        avg_points = float(avg_points or 0.0)

        existing = db.execute(
            select(Prediction).where(
                Prediction.player_id == player_id,
                Prediction.target_gw == target_gw,
                Prediction.model_name == MODEL_NAME,
            )
        ).scalars().first()

        if existing is None:
            db.add(
                Prediction(
                    player_id=player_id,
                    target_gw=target_gw,
                    model_name=MODEL_NAME,
                    predicted_points=avg_points,
                )
            )
            inserted += 1
        else:
            # update only if changed a lot (optional); simplest: always set
            existing.predicted_points = avg_points
            updated += 1

    db.commit()

    return {
        "target_gw": target_gw,
        "window": window,
        "used_finished_gws": finished_gws_sorted,
        "model_name": MODEL_NAME,
        "inserted": inserted,
        "updated": updated,
        "total_players_predicted": len(rows),
    }


@router.get("")
def list_predictions(
    target_gw: int,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """
    List predictions for a target gameweek (paginated).
    """
    total = db.execute(
        select(sa_func.count()).select_from(Prediction).where(Prediction.target_gw == target_gw)
    ).scalar_one()

    stmt = (
        select(Prediction)
        .where(Prediction.target_gw == target_gw)
        .order_by(Prediction.predicted_points.desc(), Prediction.player_id.asc())
        .offset(offset)
        .limit(limit)
    )

    preds = db.execute(stmt).scalars().all()

    return {
        "meta": {"total": total, "limit": limit, "offset": offset},
        "rows": [
            {
                "id": p.id,
                "player_id": p.player_id,
                "target_gw": p.target_gw,
                "model_name": p.model_name,
                "predicted_points": p.predicted_points,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in preds
        ],
    }
