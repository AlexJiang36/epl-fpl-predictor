from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.db import get_db
from app.models.player import Player

router = APIRouter(prefix="/players", tags=["players"])

@router.get("")
def list_players(
    position: Optional[str] = None,
    team_id: Optional[int] = None,
    search: Optional[str] = Query(default=None, min_length=1),
    db: Session = Depends(get_db),
):
    """
    List players with optional filters:
    - position: GKP/DEF/MID/FWD
    - team_id: integer team id in our DB
    - search: partial match on web_name
    """
    stmt = select(Player)

    if position is not None:
        stmt = stmt.where(Player.position == position)

    if team_id is not None:
        stmt = stmt.where(Player.team_id == team_id)

    if search is not None:
        stmt = stmt.where(Player.web_name.ilike(f"%{search}%"))

    players = db.execute(stmt).scalars().all()

    return {
        "players": [
            {
                "id": p.id,
                "fpl_player_id": p.fpl_player_id,
                "first_name": p.first_name,
                "second_name": p.second_name,
                "web_name": p.web_name,
                "team_id": p.team_id,
                "position": p.position,
                "now_cost": p.now_cost,
                "status": p.status,
            }
            for p in players
        ]
    }