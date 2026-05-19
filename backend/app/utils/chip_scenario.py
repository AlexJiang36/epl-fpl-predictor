from __future__ import annotations

from app.schemas.chip_scenario import ChipScenarioResult
from typing import Optional

def evaluate_chip_scenario(
    *,
    scenario_type: str,
    baseline_projected_points: float,
    modified_projected_points: float,
    explanation: str,
    details: Optional[dict] = None,
    notes: Optional[str] = None,
) -> ChipScenarioResult:
    delta = modified_projected_points - baseline_projected_points

    return ChipScenarioResult(
        scenario_type=scenario_type,
        baseline_projected_points=baseline_projected_points,
        modified_projected_points=modified_projected_points,
        delta=delta,
        explanation=explanation,
        details=details or {},
        notes=notes,
    )