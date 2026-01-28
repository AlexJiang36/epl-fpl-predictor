from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Literal, Dict, List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.db import get_db
from app.models.gameweek import Gameweek

# Reuse the same query helpers + serializer from predictions route
# (Make sure these are defined in app/api/routes/predictions.py exactly like Day11)
from app.api.routes.predictions import (
    MODEL_NAME,
    OrderBy,
    build_predictions_base_query,
    apply_predictions_ordering,
    serialize_prediction_row,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

Position = Literal["GKP", "DEF", "MID", "FWD"]


@router.get("/ping")
def ping():
    return {"ok": True}


@router.get("/top")
def top_recommendations(
    target_gw: Optional[int] = None,
    model_name: str = Query(default=MODEL_NAME),
    status: str = Query(default="a"),  # "a" = available; pass "all" to disable filter
    max_cost: Optional[int] = Query(default=None, ge=0),
    min_predicted_points: Optional[float] = Query(default=None, ge=0),
    order_by: OrderBy = Query(default="value"),
    limit_per_position: int = Query(default=5, ge=1, le=50),
    max_per_team: int = Query(default=3, ge=1, le=15),
    db: Session = Depends(get_db),
):
    """
    Return FPL-style shortlist recommendations, grouped by position.

    - If target_gw is not provided: use "next" gameweek from gameweeks table.
    - Default filters: status="a" (available only), order_by="value", limit_per_position=5
    - Disable availability filter by passing status="all"
    - Adds a soft team-diversity rule: max_per_team per position bucket
    """
    # 1) Decide target_gw (default = next GW)
    if target_gw is None:
        nxt = (
            db.execute(select(Gameweek).where(Gameweek.is_next == True))
            .scalars()
            .first()
        )
        if nxt is None:
            return {"error": "No next gameweek found. Run /gameweeks/ingest/fpl first."}
        target_gw = nxt.gw

    effective_status: Optional[str] = None if status == "all" else status

    positions: List[Position] = ["GKP", "DEF", "MID", "FWD"]
    grouped: Dict[Position, List[dict]] = {p: [] for p in positions}

    # 2) For each position, reuse the SAME query-building/ordering/serialization
    for pos in positions:
        base = build_predictions_base_query(
            target_gw=target_gw,
            model_name=model_name,
            position=pos,
            status=effective_status,
            search=None,
            team_id=None,
            max_cost=max_cost,
            min_predicted_points=min_predicted_points,
        )

        stmt = apply_predictions_ordering(base, order_by)

        # pull more than needed so we can enforce per-team cap without starving results
        fetch_limit = limit_per_position * max_per_team * 2
        rows = db.execute(stmt.limit(fetch_limit)).all()

        # 3) Enforce per-team cap inside each position bucket
        team_counts: Dict[int, int] = {}
        picked: List[dict] = []

        for (pred, pl, tm) in rows:
            tid = pl.team_id
            team_counts[tid] = team_counts.get(tid, 0)

            if team_counts[tid] >= max_per_team:
                continue

            picked.append(serialize_prediction_row(pred, pl, tm))
            team_counts[tid] += 1

            if len(picked) >= limit_per_position:
                break

        grouped[pos] = picked

    generated_at = datetime.now(timezone.utc).isoformat()
    counts = {pos: len(grouped[pos]) for pos in positions}

    return {
        "target_gw": target_gw,
        "model_name": model_name,
        "generated_at": generated_at,
        "filters": {
            "status": status,
            "max_cost": max_cost,
            "min_predicted_points": min_predicted_points,
            "order_by": order_by,
            "limit_per_position": limit_per_position,
            "max_per_team": max_per_team,
        },
        "counts": counts,
        "recommendations": grouped,
    }
