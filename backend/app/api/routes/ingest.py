from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.db import get_db
from app.services.fpl_client import fetch_bootstrap
from app.models.team import Team
from app.models.player import Player

router = APIRouter(prefix="/ingest", tags=["ingest"])

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