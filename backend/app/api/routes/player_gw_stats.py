from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.core.db import get_db
from app.models.player_gw_stat import PlayerGameweekStat

router = APIRouter(prefix="/player-gw-stats", tags=["player-gw-stats"])


@router.get("")
def list_player_gw_stats(
    player_id: Optional[int] = None,
    gw: Optional[int] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(PlayerGameweekStat)

    if player_id is not None:
        stmt = stmt.where(PlayerGameweekStat.player_id == player_id)
    if gw is not None:
        stmt = stmt.where(PlayerGameweekStat.gw == gw)

    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()

    rows = db.execute(
        stmt.order_by(PlayerGameweekStat.id).offset(offset).limit(limit)
    ).scalars().all()

    return {
        "meta": {"total": total, "limit": limit, "offset": offset},
        "rows": [
            {
                "id": r.id,
                "player_id": r.player_id,
                "gw": r.gw,
                "minutes": r.minutes,
                "goals_scored": r.goals_scored,
                "assists": r.assists,
                "clean_sheets": r.clean_sheets,
                "total_points": r.total_points,
            }
            for r in rows
        ],
    }
