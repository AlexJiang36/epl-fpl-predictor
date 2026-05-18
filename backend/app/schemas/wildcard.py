from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.free_hit import FreeHitPlayer, FreeHitRejectedCandidate


class WildcardPriorityTransfer(BaseModel):
    out_player_id: int
    out_web_name: str
    out_position: str
    out_horizon_points: float

    in_player_id: int
    in_web_name: str
    in_position: str
    in_horizon_points: float

    horizon_gain: float


class WildcardBuildRequest(BaseModel):
    target_gw: int = Field(ge=1)
    horizon: int = Field(ge=1)
    budget: float = Field(ge=0)
    model_name: str
    locked_player_ids: List[int] = Field(default_factory=list)
    current_squad_player_ids: List[int] = Field(default_factory=list)


class WildcardBuildResponse(BaseModel):
    target_gw: int
    horizon: int
    budget: float
    model_name: str
    locked_player_ids: List[int]
    current_squad_player_ids: List[int]

    scoring_objective: str

    starting_xi: List[FreeHitPlayer]
    bench: List[FreeHitPlayer]
    captain: Optional[FreeHitPlayer] = None
    vice_captain: Optional[FreeHitPlayer] = None
    captain_selection_target_gw: int

    spent_m: float
    remaining_m: float
    projected_points_starting_xi_horizon: float
    projected_points_total_15_horizon: float

    priority_transfers_from_current_squad: List[WildcardPriorityTransfer] = Field(default_factory=list)
    rejected_candidates: List[FreeHitRejectedCandidate] = Field(default_factory=list)

    notes: Optional[str] = None