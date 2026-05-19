from __future__ import annotations

from sqlalchemy import select, func as sa_func
from sqlalchemy.orm import Session
from typing import Optional

from app.models.prediction import Prediction
from app.schemas.squad_snapshot import SquadSnapshot
from app.schemas.chip_scenario import ChipScenarioResult
from app.utils.chip_scenario import evaluate_chip_scenario


def _sum_predicted_points_for_players(
    *,
    db: Session,
    player_ids: list[int],
    target_gw: int,
    model_name: str,
) -> float:
    if not player_ids:
        return 0.0

    stmt = (
        select(sa_func.sum(Prediction.predicted_points))
        .where(
            Prediction.target_gw == target_gw,
            Prediction.model_name == model_name,
            Prediction.player_id.in_(player_ids),
        )
    )
    total = db.execute(stmt).scalar()
    return float(total or 0.0)


def run_bench_boost_scenario(
    *,
    db: Session,
    snapshot: SquadSnapshot,
    notes: Optional[str] = None,
) -> ChipScenarioResult:
    bench_ids = list(snapshot.bench_order_player_ids)
    bench_set = set(bench_ids)

    starting_xi_ids = [
        pid for pid in snapshot.squad_player_ids
        if pid not in bench_set
    ]

    baseline_projected_points = _sum_predicted_points_for_players(
        db=db,
        player_ids=starting_xi_ids,
        target_gw=snapshot.target_gw,
        model_name=snapshot.model_name,
    )

    bench_projected_points = _sum_predicted_points_for_players(
        db=db,
        player_ids=bench_ids,
        target_gw=snapshot.target_gw,
        model_name=snapshot.model_name,
    )

    modified_projected_points = baseline_projected_points + bench_projected_points

    return evaluate_chip_scenario(
        scenario_type="bench_boost",
        baseline_projected_points=baseline_projected_points,
        modified_projected_points=modified_projected_points,
        explanation="Bench Boost adds projected bench points to the baseline starting XI projection.",
        details={
            "starting_xi_player_ids": starting_xi_ids,
            "bench_player_ids": bench_ids,
            "bench_points_added": bench_projected_points,
        },
        notes=notes,
    )