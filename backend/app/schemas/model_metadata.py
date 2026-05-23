from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ModelMetadataArtifact(BaseModel):
    model_name: str
    task_type: str

    feature_version: Optional[str] = None

    training_window_start_gw: Optional[int] = None
    training_window_end_gw: Optional[int] = None

    evaluation_start_gw: Optional[int] = None
    evaluation_end_gw: Optional[int] = None

    metrics_summary: Dict[str, float] = Field(default_factory=dict)

    notes: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))