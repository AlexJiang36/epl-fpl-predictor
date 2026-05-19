from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DecisionRunArtifact(BaseModel):
    run_id: str

    endpoint: str
    scenario_type: str

    target_gw: Optional[int] = None
    model_name: Optional[str] = None

    input_summary: Dict[str, Any] = Field(default_factory=dict)
    projected_outputs: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: Optional[str] = None