from datetime import datetime
import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.db import get_db
from app.models.player import Player
from app.models.player_gw_stat import PlayerGameweekStat
from app.models.gameweek import Gameweek

router = APIRouter(prefix='/ingest', tags=['ingest'])

def fpl_event_live_url(gw: int) -> str:
    return f"https://fantasy.premierleague.com/api/event/{gw}/live/"

def ingest_one_gw(db: Session, gw: int) -> dict:

    data = httpx.get(fpl_event_live_url(gw), timeout=30).json()
    elements = data.get("elements", [])

    # map fpl_player_id -> our player_id
    players = db.execute(select(Player.id, Player.fpl_player_id)).all()
    fpl_to_player_id = {fpl_id: pid for (pid, fpl_id) in players}

    inserted = 0
    updated = 0
    skipped = 0

    skipped_ids = []
    
    for e in elements:
        fpl_player_id = int(e["id"])
        player_id = fpl_to_player_id.get(fpl_player_id)
        if player_id is None:
            skipped += 1
            skipped_ids.append(fpl_player_id)
            continue

        s = e.get("stats", {})
        minutes = int(s.get("minutes", 0) or 0)
        goals_scored = int(s.get("goals_scored", 0) or 0)
        assists = int(s.get("assists", 0) or 0)
        clean_sheets = int(s.get("clean_sheets", 0) or 0)
        total_points = int(s.get("total_points", 0) or 0)

        existing = db.execute(
            select(PlayerGameweekStat).where(
                PlayerGameweekStat.player_id == player_id,
                PlayerGameweekStat.gw == gw,
            )
        ).scalars().first()

        if existing is None:
            db.add(
                PlayerGameweekStat(
                    player_id = player_id,
                    gw = gw,
                    minutes = minutes,
                    goals_scored = goals_scored,
                    assists = assists,
                    clean_sheets = clean_sheets,
                    total_points = total_points,
                )
            )
            inserted += 1
        else:
            changed = False
            if existing.minutes != minutes: 
                existing.minutes = minutes 
                changed = True
            if existing.goals_scored != goals_scored: 
                existing.goals_scored = goals_scored 
                changed = True
            if existing.assists != assists: 
                existing.assists = assists 
                changed = True
            if existing.clean_sheets != clean_sheets: 
                existing.clean_sheets = clean_sheets 
                changed = True
            if existing.total_points != total_points: 
                existing.total_points = total_points 
                changed = True
            if changed:
                updated += 1
    
    db.commit()
    result =  {
        "gw": gw,
        "inserted": inserted, 
        "updated": updated, 
        "skipped": skipped, 
        "total_source": len(elements),
    }
    if skipped > 0:
        result["skipped_ids"] = skipped_ids[:20]
    return result


@router.post("/fpl/gw/{gw}/live")
def ingest_fpl_gw_live(gw: int, db: Session = Depends(get_db)):
    result = ingest_one_gw(db, gw)
    return {"gw": gw, "rows": result}

@router.post("/fpl/gw/finished")
def ingest_finished_gameweeks(db: Session = Depends(get_db)):
    gws = db.execute(
        select(Gameweek.gw).where(Gameweek.is_finished == True).order_by(Gameweek.gw)
    ).scalars().all()

    per_gw = []
    totals = {"inserted": 0, "updated": 0, "skipped": 0}

    for gw in gws:
        r = ingest_one_gw(db, gw)
        per_gw.append(r)
        totals["inserted"] += r["inserted"]
        totals["updated"] += r["updated"]
        totals['skipped'] += r["skipped"]

    return {'gws': len(gws), "totals": totals, "per_gw": per_gw}
