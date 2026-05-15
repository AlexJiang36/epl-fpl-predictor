from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models.prediction import Prediction
from app.models.player import Player
from app.models.team import Team
from app.schemas.free_hit import FreeHitBuildRequest, FreeHitBuildResponse


router = APIRouter(prefix="/chips", tags=["chips"])


@router.get("/ping")
def ping():
    return {"ok": True}


@router.post("/free-hit/build", response_model=FreeHitBuildResponse)
def build_free_hit(
    req: FreeHitBuildRequest,
    db: Session = Depends(get_db),
):
    stmt = (
        select(Prediction, Player, Team)
        .join(Player, Player.id == Prediction.player_id)
        .join(Team, Team.id == Player.team_id)
        .where(
            Prediction.target_gw == req.target_gw,
            Prediction.model_name == req.model_name,
            Player.status == "a",
        )
        .order_by(Prediction.predicted_points.desc(), Player.id.asc())
    )

    results = db.execute(stmt).all()

    candidate_pool_counts = {
        "GKP": 0,
        "DEF": 0,
        "MID": 0,
        "FWD": 0,
    }

    locked_position_counts = {
        "GKP": 0,
        "DEF": 0,
        "MID": 0,
        "FWD": 0,
    }

    if req.locked_player_ids:
        locked_stmt = (
            select(Player)
            .where(Player.id.in_(req.locked_player_ids))
        )
        locked_players = db.execute(locked_stmt).scalars().all()

        for pl in locked_players:
            if pl.position in locked_position_counts:
                locked_position_counts[pl.position] += 1
    else:
        locked_players = []

    for pred, pl, tm in results:
        if pl.position in candidate_pool_counts:
            candidate_pool_counts[pl.position] += 1

    return FreeHitBuildResponse(
        target_gw=req.target_gw,
        budget=req.budget,
        model_name=req.model_name,
        locked_player_ids=req.locked_player_ids,
        scoring_objective="maximize_projected_gw_points_only",
        candidate_pool_counts=candidate_pool_counts,
        locked_player_count=len(locked_players),
        locked_position_counts=locked_position_counts,
        constraint_helpers_to_apply=[
            "budget_cap",
            "max_3_players_per_club",
            "squad_size_rules",
            "position_compatibility",
            "availability_filtering",
        ],
        notes="Foundation response only: request/response contract and candidate pools are ready.",
    )