from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class FeatureSnapshotArtifact(BaseModel):
    snapshot_id: str
    snapshot_type: str  # e.g. player_features / match_features
    feature_version: str

    extraction_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    gw_start: int
    gw_end: int

    model_name: Optional[str] = None
    source_tables: List[str] = Field(default_factory=list)

    output_path: str
    row_count: Optional[int] = None

    notes: Optional[str] = None