from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


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

    candidate_pool_counts: dict
    locked_player_count: int
    locked_position_counts: dict
    constraint_helpers_to_apply: List[str]
    notes: Optional[str] = None