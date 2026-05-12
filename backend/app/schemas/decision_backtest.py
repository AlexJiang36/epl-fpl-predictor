from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field

from app.schemas.squad_snapshot import SquadSnapshot


class DecisionBacktestResult(BaseModel):
    before_snapshot: SquadSnapshot
    after_snapshot: SquadSnapshot

    num_added: int = Field(ge=0)
    num_removed: int = Field(ge=0)

    predicted_points_before: float
    predicted_points_after: float
    predicted_gain: float

    captain_changed: bool
    vice_captain_changed: bool
    bench_order_changed: bool
    bank_delta: int

    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))