from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field, field_validator


class SquadSnapshot(BaseModel):
    squad_player_ids: List[int]
    captain_player_id: int
    vice_captain_player_id: int
    bench_order_player_ids: List[int]

    bank: int = Field(ge=0)
    target_gw: int = Field(ge=1)
    model_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("squad_player_ids")
    @classmethod
    def validate_squad_player_ids(cls, v: List[int]) -> List[int]:
        if len(v) != 15:
            raise ValueError("squad_player_ids must contain exactly 15 players")
        if len(set(v)) != 15:
            raise ValueError("squad_player_ids must not contain duplicates")
        return v

    @field_validator("bench_order_player_ids")
    @classmethod
    def validate_bench_order_player_ids(cls, v: List[int]) -> List[int]:
        if len(v) != 4:
            raise ValueError("bench_order_player_ids must contain exactly 4 players")
        if len(set(v)) != 4:
            raise ValueError("bench_order_player_ids must not contain duplicates")
        return v

    @field_validator("vice_captain_player_id")
    @classmethod
    def validate_captain_and_vice_different(cls, v: int, info) -> int:
        captain_player_id = info.data.get("captain_player_id")
        if captain_player_id is not None and v == captain_player_id:
            raise ValueError("captain_player_id and vice_captain_player_id must be different")
        return v

    @field_validator("captain_player_id", "vice_captain_player_id")
    @classmethod
    def validate_captains_in_squad(cls, v: int, info) -> int:
        squad_player_ids = info.data.get("squad_player_ids")
        if squad_player_ids is not None and v not in squad_player_ids:
            raise ValueError("captain and vice-captain must be in squad_player_ids")
        return v

    @field_validator("bench_order_player_ids")
    @classmethod
    def validate_bench_players_in_squad(cls, v: List[int], info) -> List[int]:
        squad_player_ids = info.data.get("squad_player_ids")
        if squad_player_ids is not None:
            missing = [pid for pid in v if pid not in squad_player_ids]
            if missing:
                raise ValueError(f"bench players must be in squad_player_ids: {missing}")
        return v
