from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
import httpx

from app.core.db import get_db
from app.models.gameweek import Gameweek

router = APIRouter(prefix="/gameweeks", tags=["gameweeks"])

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

def parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # FPL uses ISO strings like "2025-08-15T17:30:00Z"
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)

@router.post("/ingest/fpl")
def ingest_gameweeks(db: Session = Depends(get_db)):
    # fetch bootstrap
    data = httpx.get(FPL_BOOTSTRAP_URL, timeout=20).json()
    events = data.get("events", [])

    inserted = 0
    updated = 0

    for e in events:
        gw = int(e["id"])
        deadline = parse_dt(e.get("deadline_time"))
        is_current = bool(e.get("is_current"))
        is_next = bool(e.get("is_next"))
        is_finished = bool(e.get("finished"))
        name = e.get("name")

        existing = db.execute(select(Gameweek).where(Gameweek.gw == gw)).scalars().first()
        if existing is None:
            db.add(
                Gameweek(
                    gw=gw,
                    deadline_time=deadline,
                    is_current=is_current,
                    is_next=is_next,
                    is_finished=is_finished,
                    name=name,
                )
            )
            inserted += 1
        else:
            changed = False
            if existing.deadline_time != deadline:
                existing.deadline_time = deadline
                changed = True
            if existing.is_current != is_current:
                existing.is_current = is_current
                changed = True
            if existing.is_next != is_next:
                existing.is_next = is_next
                changed = True
            if existing.is_finished != is_finished:
                existing.is_finished = is_finished
                changed = True
            if existing.name != name:
                existing.name = name
                changed = True
            if changed:
                updated += 1

    db.commit()

    return {"gameweeks": {"inserted": inserted, "updated": updated, "total_source": len(events)}}


@router.get("/current")
def current_and_next(db: Session = Depends(get_db)):
    current = db.execute(select(Gameweek).where(Gameweek.is_current == True)).scalars().first()
    nxt = db.execute(select(Gameweek).where(Gameweek.is_next == True)).scalars().first()

    return {
        "current": {
            "gw": current.gw,
            "deadline_time": current.deadline_time.isoformat() if current and current.deadline_time else None,
            "is_finished": current.is_finished if current else None,
            "name": current.name if current else None,
        } if current else None,
        "next": {
            "gw": nxt.gw,
            "deadline_time": nxt.deadline_time.isoformat() if nxt and nxt.deadline_time else None,
            "is_finished": nxt.is_finished if nxt else None,
            "name": nxt.name if nxt else None,
        } if nxt else None,
    }
