from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChipScenarioResult(BaseModel):
    scenario_type: str

    baseline_projected_points: float
    modified_projected_points: float
    delta: float

    explanation: str
    details: Dict[str, Any] = Field(default_factory=dict)

    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))