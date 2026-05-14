from __future__ import annotations

from app.schemas.chip_scenario import ChipScenarioResult


def evaluate_chip_scenario(
    *,
    scenario_type: str,
    baseline_projected_points: float,
    modified_projected_points: float,
    explanation: str,
    details: dict | None = None,
    notes: str | None = None,
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