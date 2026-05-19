from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from app.schemas.decision_run import DecisionRunArtifact


BASE_DIR = Path("artifacts/decision_runs")


def _ensure_base_dir() -> Path:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_DIR


def _artifact_path(run_id: str) -> Path:
    return _ensure_base_dir() / f"{run_id}.json"


def create_decision_run_artifact(
    *,
    endpoint: str,
    scenario_type: str,
    target_gw: Optional[int],
    model_name: Optional[str],
    input_summary: dict,
    projected_outputs: dict,
    notes: Optional[str] = None,
) -> DecisionRunArtifact:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{scenario_type}_{timestamp}_{uuid4().hex[:8]}"

    return DecisionRunArtifact(
        run_id=run_id,
        endpoint=endpoint,
        scenario_type=scenario_type,
        target_gw=target_gw,
        model_name=model_name,
        input_summary=input_summary,
        projected_outputs=projected_outputs,
        notes=notes,
    )


def save_decision_run_artifact(artifact: DecisionRunArtifact) -> Path:
    path = _artifact_path(artifact.run_id)
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path


def load_decision_run_artifact(run_id: str) -> DecisionRunArtifact:
    path = _artifact_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Decision run not found: {run_id}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return DecisionRunArtifact(**data)


def list_decision_run_artifacts(
    *,
    scenario_type: Optional[str] = None,
    limit: int = 20,
) -> List[DecisionRunArtifact]:
    base = _ensure_base_dir()
    paths = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    results: List[DecisionRunArtifact] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        artifact = DecisionRunArtifact(**data)

        if scenario_type is not None and artifact.scenario_type != scenario_type:
            continue

        results.append(artifact)
        if len(results) >= limit:
            break

    return results