from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from app.schemas.feature_snapshot import FeatureSnapshotArtifact


BASE_DIR = Path("artifacts/feature_snapshots")


def _ensure_base_dir() -> Path:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_DIR


def _artifact_path(snapshot_id: str) -> Path:
    return _ensure_base_dir() / f"{snapshot_id}.json"


def create_feature_snapshot_artifact(
    *,
    snapshot_type: str,
    feature_version: str,
    gw_start: int,
    gw_end: int,
    output_path: str,
    source_tables: List[str],
    model_name: Optional[str] = None,
    row_count: Optional[int] = None,
    notes: Optional[str] = None,
) -> FeatureSnapshotArtifact:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_id = f"{snapshot_type}_{timestamp}_{uuid4().hex[:8]}"

    return FeatureSnapshotArtifact(
        snapshot_id=snapshot_id,
        snapshot_type=snapshot_type,
        feature_version=feature_version,
        gw_start=gw_start,
        gw_end=gw_end,
        model_name=model_name,
        source_tables=source_tables,
        output_path=output_path,
        row_count=row_count,
        notes=notes,
    )


def save_feature_snapshot_artifact(artifact: FeatureSnapshotArtifact) -> Path:
    path = _artifact_path(artifact.snapshot_id)
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path


def load_feature_snapshot_artifact(snapshot_id: str) -> FeatureSnapshotArtifact:
    path = _artifact_path(snapshot_id)
    if not path.exists():
        raise FileNotFoundError(f"Feature snapshot not found: {snapshot_id}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return FeatureSnapshotArtifact(**data)


def list_feature_snapshot_artifacts(
    *,
    snapshot_type: Optional[str] = None,
    limit: int = 20,
) -> List[FeatureSnapshotArtifact]:
    base = _ensure_base_dir()
    paths = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    results: List[FeatureSnapshotArtifact] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        artifact = FeatureSnapshotArtifact(**data)

        if snapshot_type is not None and artifact.snapshot_type != snapshot_type:
            continue

        results.append(artifact)
        if len(results) >= limit:
            break

    return results