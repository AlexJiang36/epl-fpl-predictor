from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from app.schemas.run_snapshot import RunSnapshotArtifact


BASE_DIR = Path("artifacts/run_snapshots")


def _ensure_base_dir() -> Path:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_DIR


def _snapshot_path(snapshot_id: str) -> Path:
    return _ensure_base_dir() / f"{snapshot_id}.json"


def create_run_snapshot_artifact(
    *,
    snapshot_type: str,
    source_command: Optional[str] = None,
    endpoint: Optional[str] = None,
    scenario_type: Optional[str] = None,
    target_gw: Optional[int] = None,
    model_name: Optional[str] = None,
    row_counts: Optional[dict] = None,
    metadata: Optional[dict] = None,
    notes: Optional[str] = None,
) -> RunSnapshotArtifact:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_id = f"{snapshot_type}_{timestamp}_{uuid4().hex[:8]}"

    return RunSnapshotArtifact(
        snapshot_id=snapshot_id,
        snapshot_type=snapshot_type,
        source_command=source_command,
        endpoint=endpoint,
        scenario_type=scenario_type,
        target_gw=target_gw,
        model_name=model_name,
        row_counts=row_counts or {},
        metadata=metadata or {},
        notes=notes,
    )


def save_run_snapshot_artifact(artifact: RunSnapshotArtifact) -> Path:
    path = _snapshot_path(artifact.snapshot_id)
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path


def load_run_snapshot_artifact(snapshot_id: str) -> RunSnapshotArtifact:
    path = _snapshot_path(snapshot_id)
    if not path.exists():
        raise FileNotFoundError(f"Run snapshot not found: {snapshot_id}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return RunSnapshotArtifact(**data)


def list_run_snapshot_artifacts(
    *,
    snapshot_type: Optional[str] = None,
    limit: int = 20,
) -> List[RunSnapshotArtifact]:
    base = _ensure_base_dir()
    paths = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    results: List[RunSnapshotArtifact] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        artifact = RunSnapshotArtifact(**data)

        if snapshot_type is not None and artifact.snapshot_type != snapshot_type:
            continue

        results.append(artifact)
        if len(results) >= limit:
            break

    return results