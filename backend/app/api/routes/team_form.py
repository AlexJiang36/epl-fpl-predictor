from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, desc
from datetime import datetime
from typing import Optional

from app.core.db import get_db
from app.models.fixture import Fixture
from app.models.team import Team

router = APIRouter(prefix="/teams", tags=["match"])


def parse_before(before: Optional[str]) -> Optional[datetime]:
    if not before:
        return None
    # accepts ISO8601 like "2026-03-14T17:30:00+00:00"
    return datetime.fromisoformat(before.replace("Z", "+00:00"))


@router.get("/{team_id}/form")
def team_form(
    team_id: int,
    n: int = Query(5, ge=1, le=20),
    before: Optional[str] = Query(None, description="ISO datetime. Only use fixtures with kickoff_time < before."),
    db: Session = Depends(get_db),
):
    before_dt = parse_before(before)

    # team name
    t = db.query(Team).filter(Team.id == team_id).first()
    team_name = t.name if t else f"Unknown({team_id})"

    q = (
        db.query(Fixture)
        .filter(Fixture.finished.is_(True))
        .filter(Fixture.kickoff_time.isnot(None))
        .filter(Fixture.home_score.isnot(None))
        .filter(Fixture.away_score.isnot(None))
        .filter(or_(Fixture.home_team_id == team_id, Fixture.away_team_id == team_id))
    )
    if before_dt is not None:
        q = q.filter(Fixture.kickoff_time < before_dt)

    fixtures = q.order_by(desc(Fixture.kickoff_time)).limit(n).all()

    matches_used = len(fixtures)
    if matches_used == 0:
        return {
            "team_id": team_id,
            "team_name": team_name,
            "n": n,
            "before": before,
            "matches_used": 0,
            "points_avg": 0.0,
            "goals_for_avg": 0.0,
            "goals_against_avg": 0.0,
            "clean_sheet_rate": 0.0,
            "last_matches": [],
        }

    points = 0
    gf = 0
    ga = 0
    clean_sheets = 0
    last_matches = []

    for f in fixtures:
        hs = int(f.home_score)
        as_ = int(f.away_score)

        if f.home_team_id == team_id:
            goals_for = hs
            goals_against = as_
            opp_id = f.away_team_id
        else:
            goals_for = as_
            goals_against = hs
            opp_id = f.home_team_id

        gf += goals_for
        ga += goals_against
        if goals_against == 0:
            clean_sheets += 1

        if goals_for > goals_against:
            pts = 3
        elif goals_for == goals_against:
            pts = 1
        else:
            pts = 0
        points += pts

        last_matches.append(
            {
                "fixture_id": f.id,
                "gw": getattr(f, "gw", None),
                "kickoff_time": f.kickoff_time.isoformat() if f.kickoff_time else None,
                "opponent_team_id": opp_id,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "points": pts,
            }
        )

    return {
        "team_id": team_id,
        "team_name": team_name,
        "n": n,
        "before": before,
        "matches_used": matches_used,
        "points_avg": points / matches_used,
        "goals_for_avg": gf / matches_used,
        "goals_against_avg": ga / matches_used,
        "clean_sheet_rate": clean_sheets / matches_used,
        "last_matches": last_matches,
    }
