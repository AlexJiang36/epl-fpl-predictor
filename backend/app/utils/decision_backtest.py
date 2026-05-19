from __future__ import annotations

from sqlalchemy import select, func as sa_func
from sqlalchemy.orm import Session
from typing import Optional

from app.models.prediction import Prediction
from app.schemas.squad_snapshot import SquadSnapshot
from app.schemas.decision_backtest import DecisionBacktestResult
from app.utils.squad_snapshot_compare import compare_squad_snapshots


def _sum_snapshot_predicted_points(
    *,
    db: Session,
    snapshot: SquadSnapshot,
) -> float:
    stmt = (
        select(sa_func.sum(Prediction.predicted_points))
        .where(
            Prediction.target_gw == snapshot.target_gw,
            Prediction.model_name == snapshot.model_name,
            Prediction.player_id.in_(snapshot.squad_player_ids),
        )
    )
    total = db.execute(stmt).scalar()
    return float(total or 0.0)


def run_decision_backtest(
    *,
    db: Session,
    before: SquadSnapshot,
    after: SquadSnapshot,
    notes: Optional[str] = None,
) -> DecisionBacktestResult:
    comparison = compare_squad_snapshots(before, after)

    predicted_points_before = _sum_snapshot_predicted_points(
        db=db,
        snapshot=before,
    )
    predicted_points_after = _sum_snapshot_predicted_points(
        db=db,
        snapshot=after,
    )
    predicted_gain = predicted_points_after - predicted_points_before

    summary = comparison["summary"]

    return DecisionBacktestResult(
        before_snapshot=before,
        after_snapshot=after,
        num_added=summary["num_added"],
        num_removed=summary["num_removed"],
        predicted_points_before=predicted_points_before,
        predicted_points_after=predicted_points_after,
        predicted_gain=predicted_gain,
        captain_changed=summary["captain_changed"],
        vice_captain_changed=summary["vice_captain_changed"],
        bench_order_changed=summary["bench_order_changed"],
        bank_delta=summary["bank_delta"],
        notes=notes,
    )