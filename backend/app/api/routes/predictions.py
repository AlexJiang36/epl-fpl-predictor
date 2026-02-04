from typing import Optional, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func as sa_func

from app.core.db import get_db
from app.models.prediction import Prediction
from app.models.player_gw_stat import PlayerGameweekStat
from app.models.gameweek import Gameweek
from app.models.player import Player
from app.models.team import Team


router = APIRouter(prefix="/predictions", tags=["predictions"])
MODEL_NAME = "baseline_rollavg_v0"

OrderBy = Literal["points", "cost", "value"]


def build_predictions_base_query(
    *,
    target_gw: int,
    model_name: str,
    position: Optional[str] = None,
    team_id: Optional[int] = None,
    status: Optional[str] = None,  # None means no filter
    search: Optional[str] = None,
    max_cost: Optional[int] = None,
    min_predicted_points: Optional[float] = None,
):
    base = (
        select(Prediction, Player, Team)
        .join(Player, Player.id == Prediction.player_id)
        .join(Team, Team.id == Player.team_id)
        .where(
            Prediction.target_gw == target_gw,
            Prediction.model_name == model_name,
        )
    )

    if position is not None:
        base = base.where(Player.position == position)

    if team_id is not None:
        base = base.where(Player.team_id == team_id)

    if status is not None:
        base = base.where(Player.status == status)

    if max_cost is not None:
        base = base.where(Player.now_cost <= max_cost)

    if min_predicted_points is not None:
        base = base.where(Prediction.predicted_points >= min_predicted_points)

    if search is not None:
        base = base.where(Player.web_name.ilike(f"%{search}%"))

    return base


def apply_predictions_ordering(base, order_by: OrderBy):
    if order_by == "points":
        order_clause = Prediction.predicted_points.desc()
    elif order_by == "cost":
        order_clause = Player.now_cost.asc()
    else:  # value
        order_clause = (Prediction.predicted_points / (Player.now_cost + 1)).desc()

    # Always add a stable tie-breaker
    return base.order_by(order_clause, Player.id.asc())


def serialize_prediction_row(pred: Prediction, pl: Player, tm: Team):
    return {
        "prediction_id": pred.id,
        "player_id": pred.player_id,
        "target_gw": pred.target_gw,
        "model_name": pred.model_name,
        "predicted_points": float(pred.predicted_points or 0.0),
        "created_at": pred.created_at.isoformat() if pred.created_at else None,
        "fpl_player_id": pl.fpl_player_id,
        "web_name": pl.web_name,
        "position": pl.position,
        "now_cost": pl.now_cost,
        "status": pl.status,
        "team_id": pl.team_id,
        "team_short_name": tm.short_name,
        "team_name": tm.name,
    }


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
        nxt = (
            db.execute(select(Gameweek).where(Gameweek.is_next == True))
            .scalars()
            .first()
        )
        if nxt is None:
            return {"error": "No next gameweek found. Run /gameweeks/ingest/fpl first."}
        target_gw = nxt.gw

    # 2) pick last N finished gameweeks (by gw desc)
    finished_gws = (
        db.execute(
            select(Gameweek.gw)
            .where(Gameweek.is_finished == True,
                   Gameweek.gw < target_gw)
            .order_by(Gameweek.gw.desc())
            .limit(window)
        )
        .scalars()
        .all()
    )

    if len(finished_gws) == 0:
        return {"error": "No finished gameweeks found. Ingest gameweeks first."}

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

        existing = (
            db.execute(
                select(Prediction).where(
                    Prediction.player_id == player_id,
                    Prediction.target_gw == target_gw,
                    Prediction.model_name == MODEL_NAME,
                )
            )
            .scalars()
            .first()
        )

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
    model_name: str = Query(default=MODEL_NAME),
    position: Optional[str] = None,
    team_id: Optional[int] = None,
    status: Optional[str] = None,
    search: Optional[str] = Query(default=None, min_length=1),
    max_cost: Optional[int] = Query(default=None, ge=0),
    min_predicted_points: Optional[float] = Query(default=None, ge=0),
    order_by: OrderBy = Query(default="points"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """
    List predictions for a target gameweek (paginated).
    """
    base = build_predictions_base_query(
        target_gw=target_gw,
        model_name=model_name,
        position=position,
        team_id=team_id,
        status=status,
        search=search,
        max_cost=max_cost,
        min_predicted_points=min_predicted_points,
    )

    total = db.execute(
        select(sa_func.count()).select_from(base.subquery())
    ).scalar_one()

    stmt = (
        apply_predictions_ordering(base, order_by)
        .offset(offset)
        .limit(limit)
    )

    results = db.execute(stmt).all()

    return {
        "meta": {"total": total, "limit": limit, "offset": offset},
        "rows": [serialize_prediction_row(pred, pl, tm) for (pred, pl, tm) in results],
    }


@router.get("/top")
def top_predictions(
    target_gw: int,
    model_name: str = Query(default=MODEL_NAME),
    position: Optional[str] = None,
    team_id: Optional[int] = None,
    search: Optional[str] = Query(default=None, min_length=1),
    max_cost: Optional[int] = Query(default=None, ge=0),
    min_predicted_points: Optional[float] = Query(default=None, ge=0),
    order_by: OrderBy = Query(default="value"),
    limit: int = Query(default=15, ge=1, le=200),
    # allow override availability filter
    status: str = Query(default="a"),
    db: Session = Depends(get_db),
):
    """
    Convenience shortlist endpoint for FPL-style weekly picks.

    Defaults:
    - status="a" (available)
    - order_by="value"
    - limit=15
    """
    effective_status: Optional[str] = None if status == "all" else status

    base = build_predictions_base_query(
        target_gw=target_gw,
        model_name=model_name,
        position=position,
        team_id=team_id,
        status=effective_status,
        search=search,
        max_cost=max_cost,
        min_predicted_points=min_predicted_points,
    )

    stmt = apply_predictions_ordering(base, order_by).limit(limit)
    results = db.execute(stmt).all()

    return {
        "target_gw": target_gw,
        "model_name": model_name,
        "limit": limit,
        "order_by": order_by,
        "rows": [serialize_prediction_row(pred, pl, tm) for (pred, pl, tm) in results],
    }
