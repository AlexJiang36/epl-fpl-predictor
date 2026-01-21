from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.core.db import get_db
from app.models.fixture import Fixture

router = APIRouter(prefix="/fixtures", tags=["fixtures"])

@router.get("")
def list_fixtures(
    team_id: Optional[int] = None,
    finished: Optional[bool] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(Fixture)

    if team_id is not None:
        stmt = stmt.where((Fixture.home_team_id == team_id) | (Fixture.away_team_id == team_id))

    if finished is not None:
        stmt = stmt.where(Fixture.finished == finished)

    total = db.execute(
        select(func.count()).select_from(stmt.subquery())
    ).scalar_one()


    fixtures = db.execute(
        stmt.order_by(Fixture.id).offset(offset).limit(limit)
    ).scalars().all()

    return {
        "meta": {"total": total, "limit": limit, "offset": offset},
        "fixtures": [
            {
                "id": f.id,
                "fpl_fixture_id": f.fpl_fixture_id,
                "home_team_id": f.home_team_id,
                "away_team_id": f.away_team_id,
                "kickoff_time": f.kickoff_time.isoformat() if f.kickoff_time else None,
                "finished": f.finished,
                "home_score": f.home_score,
                "away_score": f.away_score,
            }
            for f in fixtures
        ],
    }