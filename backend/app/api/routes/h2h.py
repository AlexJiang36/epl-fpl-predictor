# app/api/routes/h2h.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc

from app.core.db import get_db
from app.models.fixture import Fixture
from app.models.team import Team

router = APIRouter(prefix="/h2h", tags=["match"])


@router.get("")
def head_to_head(
    home_team_id: int = Query(..., ge=1),
    away_team_id: int = Query(..., ge=1),
    n: int = Query(5, ge=1, le=50),
    db: Session = Depends(get_db),
):
    # Fetch last N finished matches between these two teams (either direction).
    q = (
        db.query(Fixture)
        .filter(Fixture.finished.is_(True))
        .filter(Fixture.home_score.isnot(None))
        .filter(Fixture.away_score.isnot(None))
        .filter(
            or_(
                and_(
                    Fixture.home_team_id == home_team_id,
                    Fixture.away_team_id == away_team_id,
                ),
                and_(
                    Fixture.home_team_id == away_team_id,
                    Fixture.away_team_id == home_team_id,
                ),
            )
        )
        .order_by(desc(Fixture.kickoff_time))
        .limit(n)
    )

    fixtures = q.all()

    # Team name lookup (optional but nice)
    team_rows = (
    db.query(Team.id, Team.name)
    .filter(Team.id.in_([home_team_id, away_team_id]))
    .all()
)
    team_map = {tid: name for (tid, name) in team_rows}

    home_name = team_map.get(home_team_id, f"Unknown({home_team_id})")
    away_name = team_map.get(away_team_id, f"Unknown({away_team_id})")

    # Summary stats from the perspective of "home_team_id" argument
    total = len(fixtures)
    home_wins = 0
    away_wins = 0
    draws = 0
    home_goals = 0
    away_goals = 0

    out_fixtures = []
    for f in fixtures:
        hs = int(f.home_score)
        as_ = int(f.away_score)

        # attribute goals to the query's home/away team ids (not fixture home/away)
        if f.home_team_id == home_team_id:
            h_goals = hs
            a_goals = as_
        else:
            # swapped fixture direction
            h_goals = as_
            a_goals = hs

        home_goals += h_goals
        away_goals += a_goals

        if h_goals > a_goals:
            home_wins += 1
        elif h_goals < a_goals:
            away_wins += 1
        else:
            draws += 1

        out_fixtures.append(
            {
                "fixture_id": f.id,
                "fpl_fixture_id": getattr(f, "fpl_fixture_id", None),
                "kickoff_time": f.kickoff_time.isoformat() if f.kickoff_time else None,
                "fixture_home_team_id": f.home_team_id,
                "fixture_away_team_id": f.away_team_id,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_goals": h_goals,
                "away_goals": a_goals,
                "finished": bool(f.finished),
            }
        )

    summary = {
        "home_team_id": home_team_id,
        "home_team_name": home_name,
        "away_team_id": away_team_id,
        "away_team_name": away_name,
        "matches": total,
        "home_wins": home_wins,
        "away_wins": away_wins,
        "draws": draws,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "avg_total_goals_per_match": (home_goals + away_goals) / total if total > 0 else 0.0,
    }

    return {
        "meta": {"n": n, "returned": total},
        "summary": summary,
        "fixtures": out_fixtures,
    }