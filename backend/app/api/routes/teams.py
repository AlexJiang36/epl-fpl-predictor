from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.models.team import Team

router = APIRouter(prefix="/teams", tags=["teams"])

@router.get("")
def list_teams(db: Session = Depends(get_db)):
    teams = db.query(Team).all()
    return {
        "teams": [
            {"id": t.id, "fpl_team_id": t.fpl_team_id, "name": t.name, "short_name": t.short_name}
            for t in teams
        ]
    }
