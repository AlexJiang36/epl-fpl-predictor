from __future__ import annotations

from app.core.db import SessionLocal
from app.schemas.squad_snapshot import SquadSnapshot
from app.utils.decision_backtest import run_decision_backtest


def main() -> None:
    before = SquadSnapshot(
        squad_player_ids=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        captain_player_id=1,
        vice_captain_player_id=2,
        bench_order_player_ids=[12,13,14,15],
        bank=10,
        target_gw=32,
        model_name="baseline_rollavg_v0",
    )

    after = SquadSnapshot(
        squad_player_ids=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,99],
        captain_player_id=3,
        vice_captain_player_id=2,
        bench_order_player_ids=[12,14,13,99],
        bank=5,
        target_gw=32,
        model_name="baseline_rollavg_v0",
    )

    db = SessionLocal()
    try:
        result = run_decision_backtest(
            db=db,
            before=before,
            after=after,
            notes="sample day41 example",
        )
        print(result.model_dump())
    finally:
        db.close()


if __name__ == "__main__":
    main()