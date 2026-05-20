from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RunSnapshotArtifact(BaseModel):
    snapshot_id: str

    snapshot_type: str
    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    source_command: Optional[str] = None
    endpoint: Optional[str] = None
    scenario_type: Optional[str] = None

    target_gw: Optional[int] = None
    model_name: Optional[str] = None

    row_counts: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    notes: Optional[str] = None