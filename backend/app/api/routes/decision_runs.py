from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from app.schemas.decision_run import DecisionRunArtifact
from app.utils.decision_run_store import (
    load_decision_run_artifact,
    list_decision_run_artifacts,
)

router = APIRouter(prefix="/decision-runs", tags=["decision-runs"])


@router.get("", response_model=list[DecisionRunArtifact])
def list_decision_runs(
    scenario_type: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
):
    return list_decision_run_artifacts(
        scenario_type=scenario_type,
        limit=limit,
    )


@router.get("/{run_id}", response_model=DecisionRunArtifact)
def get_decision_run(run_id: str):
    try:
        return load_decision_run_artifact(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Decision run not found: {run_id}")