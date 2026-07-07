from typing import Optional, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func as sa_func

from app.core.db import get_db
from app.core.season import get_current_season
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
    season: str,
    target_gw: int,
    model_name: str,
    position: Optional[str] = None,
    team_id: Optional[int] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    max_cost: Optional[int] = None,
    min_predicted_points: Optional[float] = None,
):
    base = (
        select(Prediction, Player, Team)
        .join(Player, Player.id == Prediction.player_id)
        .join(Team, Team.id == Player.team_id)
        .where(
            Prediction.season == season,
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
        base = base.where(Player.web_name.ilike("%%%s%%" % search))

    return base


def apply_predictions_ordering(base, order_by: OrderBy):
    if order_by == "points":
        order_clause = Prediction.predicted_points.desc()
    elif order_by == "cost":
        order_clause = Player.now_cost.asc()
    else:
        order_clause = (Prediction.predicted_points / (Player.now_cost + 1)).desc()

    return base.order_by(order_clause, Player.id.asc())


def serialize_prediction_row(pred: Prediction, pl: Player, tm: Team):
    return {
        "prediction_id": pred.id,
        "season": pred.season,
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


def run_baseline_rollavg_v0_core(
    *,
    db: Session,
    target_gw: Optional[int],
    window: int,
) -> dict:
    season = get_current_season()

    if target_gw is None:
        nxt = (
            db.execute(
                select(Gameweek).where(
                    Gameweek.season == season,
                    Gameweek.is_next == True,
                )
            )
            .scalars()
            .first()
        )
        if nxt is None:
            return {
                "error": "No next gameweek found for season=%s. Run /gameweeks/ingest/fpl first." % season
            }
        target_gw = nxt.gw

    finished_gws = (
        db.execute(
            select(Gameweek.gw)
            .where(
                Gameweek.season == season,
                Gameweek.is_finished == True,
                Gameweek.gw < target_gw,
            )
            .order_by(Gameweek.gw.desc())
            .limit(window)
        )
        .scalars()
        .all()
    )

    if len(finished_gws) == 0:
        return {
            "error": "No finished gameweeks found for season=%s. Ingest gameweeks first." % season
        }

    finished_gws_sorted = sorted(finished_gws)

    rows = db.execute(
        select(
            PlayerGameweekStat.player_id,
            sa_func.avg(PlayerGameweekStat.total_points).label("avg_points"),
        )
        .where(
            PlayerGameweekStat.season == season,
            PlayerGameweekStat.gw.in_(finished_gws),
        )
        .group_by(PlayerGameweekStat.player_id)
    ).all()

    inserted = 0
    updated = 0

    for player_id, avg_points in rows:
        avg_points = float(avg_points or 0.0)

        existing = (
            db.execute(
                select(Prediction).where(
                    Prediction.season == season,
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
                    season=season,
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
        "season": season,
        "target_gw": target_gw,
        "window": window,
        "used_finished_gws": finished_gws_sorted,
        "model_name": MODEL_NAME,
        "inserted": inserted,
        "updated": updated,
        "total_players_predicted": len(rows),
    }


@router.post("/baseline/run")
def run_baseline(
    target_gw: Optional[int] = None,
    window: int = Query(default=5, ge=1, le=10),
    db: Session = Depends(get_db),
):
    return run_baseline_rollavg_v0_core(
        db=db,
        target_gw=target_gw,
        window=window,
    )


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
    season = get_current_season()

    base = build_predictions_base_query(
        season=season,
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

    stmt = apply_predictions_ordering(base, order_by).offset(offset).limit(limit)
    results = db.execute(stmt).all()

    return {
        "meta": {
            "season": season,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
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
    status: str = Query(default="a"),
    db: Session = Depends(get_db),
):
    season = get_current_season()

    effective_status = None if status == "all" else status

    base = build_predictions_base_query(
        season=season,
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
        "season": season,
        "target_gw": target_gw,
        "model_name": model_name,
        "limit": limit,
        "order_by": order_by,
        "rows": [serialize_prediction_row(pred, pl, tm) for (pred, pl, tm) in results],
    }