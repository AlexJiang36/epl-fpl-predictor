from typing import Optional
from datetime import datetime
import httpx

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.db import get_db
from app.services.fpl_client import fetch_bootstrap
from app.models.team import Team
from app.models.player import Player
from app.models.fixture import Fixture

router = APIRouter(prefix="/ingest", tags=["ingest"])

FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

@router.post("/fpl/bootstrap")
def ingest_fpl_bootstrap(db: Session = Depends(get_db)):
    data = fetch_bootstrap()

    teams_data = data.get("teams", [])
    players_data = data.get("elements", [])

    # --- upsert teams ---
    inserted_teams = 0
    updated_teams = 0

    # We assume your Team model has at least: id (pk), fpl_team_id (unique), name
    for t in teams_data:
        fpl_team_id = int(t["id"])
        name = t["name"]
        short_name = t.get("short_name") or name  # short_name should exist; fallback to name just in case

        existing = db.execute(select(Team).where(Team.fpl_team_id == fpl_team_id)).scalar_one_or_none()
        if existing is None:
            db.add(Team(fpl_team_id=fpl_team_id, name=name, short_name=short_name))
            inserted_teams += 1
        else:
            # update if changed
            changed = False
            if existing.name != name:
                existing.name = name
                changed = True
            if existing.short_name != short_name:
                existing.short_name = short_name
                changed = True
            if changed:
                updated_teams += 1
            
    db.commit()

    # Build mapping: FPL team id -> our DB team pk id
    team_rows = db.execute(select(Team)).scalars().all()
    team_map = {t.fpl_team_id: t.id for t in team_rows}

    # --- upsert players ---
    inserted_players = 0
    updated_players = 0

    # FPL element_type mapping
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    for p in players_data:
        fpl_player_id = int(p["id"])
        first_name = p["first_name"]
        second_name = p["second_name"]
        web_name = p['web_name']

        fpl_team_id = int(p["team"])
        team_id = team_map.get(fpl_team_id)

        position = pos_map.get(int(p["element_type"]), "UNK")
        now_cost = int(p["now_cost"])
        status = str(p["status"])

        existing = db.execute(select(Player).where(Player.fpl_player_id == fpl_player_id)).scalar_one_or_none()

        if existing is None:
            db.add(
                Player(
                    fpl_player_id=fpl_player_id,
                    first_name=first_name,
                    second_name=second_name,
                    web_name=web_name,
                    team_id=team_id,
                    position=position,
                    now_cost=now_cost,
                    status=status,
                )
            )
            inserted_players += 1
        else:
            changed = False
            if existing.first_name != first_name:
                existing.first_name = first_name
                changed = True
            if existing.second_name != second_name:
                existing.second_name = second_name
                changed = True
            if existing.web_name != web_name:
                existing.web_name = web_name
                changed = True
            if existing.team_id != team_id:
                existing.team_id = team_id
                changed = True
            if existing.position != position:
                existing.position = position
                changed = True
            if existing.now_cost != now_cost:
                existing.now_cost = now_cost
                changed = True
            if existing.status != status:
                existing.status = status
                changed = True

            if changed:
                updated_players += 1
    
    db.commit()

    return {
        "teams": {
            "inserted": inserted_teams,
            "updated": updated_teams,
            "total_source": len(teams_data),
        },
        "players": {
            "inserted": inserted_players,
            "updated": updated_players,
            "total_source": len(players_data)
        },
    }

def parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # FPL often uses "...Z"
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


@router.post("/fpl/fixtures")
def ingest_fpl_fixtures(db: Session = Depends(get_db)):
    fixtures = httpx.get(FPL_FIXTURES_URL, timeout=30).json()

    # Build mapping: fpl_team_id -> our teams.id (PK)
    teams = db.execute(select(Team)).scalars().all()
    team_map = {t.fpl_team_id: t.id for t in teams}

    inserted = 0
    updated = 0

    for fx in fixtures:
        fpl_fixture_id = int(fx["id"])

        fpl_home = int(fx["team_h"])
        fpl_away = int(fx["team_a"])

        home_team_id = team_map.get(fpl_home)
        away_team_id = team_map.get(fpl_away)

        # Safety: if mapping missing, skip (should not happen if teams ingested)
        if home_team_id is None or away_team_id is None:
            continue

        kickoff_time = parse_dt(fx.get("kickoff_time"))
        finished = bool(fx.get("finished"))

        # scores can be None before match played
        home_score = fx.get("team_h_score")
        away_score = fx.get("team_a_score")

        existing = db.execute(
            select(Fixture).where(Fixture.fpl_fixture_id == fpl_fixture_id)
        ).scalars().first()

        if existing is None:
            db.add(
                Fixture(
                    fpl_fixture_id=fpl_fixture_id,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    kickoff_time=kickoff_time,
                    finished=finished,
                    home_score=home_score,
                    away_score=away_score,
                )
            )
            inserted += 1
        else:
            changed = False

            if existing.home_team_id != home_team_id:
                existing.home_team_id = home_team_id; changed = True
            if existing.away_team_id != away_team_id:
                existing.away_team_id = away_team_id; changed = True
            if existing.kickoff_time != kickoff_time:
                existing.kickoff_time = kickoff_time; changed = True
            if existing.finished != finished:
                existing.finished = finished; changed = True
            if existing.home_score != home_score:
                existing.home_score = home_score; changed = True
            if existing.away_score != away_score:
                existing.away_score = away_score; changed = True

            if changed:
                updated += 1

    db.commit()

    return {"fixtures": {"inserted": inserted, "updated": updated, "total_source": len(fixtures)}}