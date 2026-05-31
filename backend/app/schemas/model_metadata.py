from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


ModelStatus = Literal["active", "experimental", "archived"]
ModelTaskType = Literal["player_points", "match_result", "match_goals"]


class ModelMetadataArtifact(BaseModel):
    model_name: str
    task_type: ModelTaskType
    feature_version: Optional[str] = None

    training_window_start_gw: Optional[int] = None
    training_window_end_gw: Optional[int] = None
    evaluation_start_gw: Optional[int] = None
    evaluation_end_gw: Optional[int] = None

    metrics_summary: Dict[str, float] = Field(default_factory=dict)

    status: ModelStatus = "experimental"
    is_active: bool = False
    is_production_default: bool = False
    selected_reason: Optional[str] = None

    notes: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
