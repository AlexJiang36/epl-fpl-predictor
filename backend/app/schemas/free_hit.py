from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class FreeHitPlayer(BaseModel):
    player_id: int
    web_name: str
    team_name: str
    team_short_name: str
    position: str
    now_cost: int
    predicted_points: float
    status: str


class FreeHitRejectedCandidate(BaseModel):
    player_id: int
    web_name: str
    team_name: str
    team_short_name: str
    position: str
    now_cost: int
    predicted_points: float
    rejected_reason: str


class FreeHitBuildRequest(BaseModel):
    target_gw: int = Field(ge=1)
    budget: float = Field(ge=0)
    model_name: str
    locked_player_ids: List[int] = Field(default_factory=list)


class FreeHitBuildResponse(BaseModel):
    target_gw: int
    budget: float
    model_name: str
    locked_player_ids: List[int]

    scoring_objective: str

    starting_xi: List[FreeHitPlayer]
    bench: List[FreeHitPlayer]
    captain: Optional[FreeHitPlayer] = None
    vice_captain: Optional[FreeHitPlayer] = None
    

    spent_m: float
    remaining_m: float
    projected_points_starting_xi: float
    projected_points_total_15: float

    rejected_candidates: List[FreeHitRejectedCandidate] = Field(default_factory=list)

    notes: Optional[str] = None