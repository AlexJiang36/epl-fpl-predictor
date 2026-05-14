from __future__ import annotations

from sqlalchemy import select, func as sa_func
from sqlalchemy.orm import Session

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


def _get_single_player_predicted_points(
    *,
    db: Session,
    player_id: int,
    target_gw: int,
    model_name: str,
) -> float:
    stmt = (
        select(Prediction.predicted_points)
        .where(
            Prediction.target_gw == target_gw,
            Prediction.model_name == model_name,
            Prediction.player_id == player_id,
        )
    )
    pts = db.execute(stmt).scalar()
    return float(pts or 0.0)


def run_triple_captain_scenario(
    *,
    db: Session,
    snapshot: SquadSnapshot,
    notes: str | None = None,
) -> ChipScenarioResult:
    bench_ids = set(snapshot.bench_order_player_ids)
    starting_xi_ids = [
        pid for pid in snapshot.squad_player_ids
        if pid not in bench_ids
    ]

    baseline_projected_points = _sum_predicted_points_for_players(
        db=db,
        player_ids=starting_xi_ids,
        target_gw=snapshot.target_gw,
        model_name=snapshot.model_name,
    )

    captain_points = _get_single_player_predicted_points(
        db=db,
        player_id=snapshot.captain_player_id,
        target_gw=snapshot.target_gw,
        model_name=snapshot.model_name,
    )

    # Triple Captain adds two extra captain copies on top of the baseline starting XI projection.
    extra_captain_bonus = 2 * captain_points
    modified_projected_points = baseline_projected_points + extra_captain_bonus

    return evaluate_chip_scenario(
        scenario_type="triple_captain",
        baseline_projected_points=baseline_projected_points,
        modified_projected_points=modified_projected_points,
        explanation="Triple Captain adds one extra captain copy on top of the normal captain multiplier baseline.",
        details={
            "captain_player_id": snapshot.captain_player_id,
            "captain_points": captain_points,
            "extra_captain_bonus": extra_captain_bonus,
            "starting_xi_player_ids": starting_xi_ids,
        },
        notes=notes,
    )