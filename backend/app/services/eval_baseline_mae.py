# backend/app/services/eval_baseline_mae.py
from __future__ import annotations

import sys
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

# Project models
from app.models.prediction import Prediction
from app.models.player_gw_stat import PlayerGameweekStat

# Try to import DB session from common project patterns.
# (Keeps this script reusable even if your db module naming differs.)
SessionLocal = None
_engine = None

try:
    # common pattern 1
    from app.core.db import SessionLocal as _SessionLocal  # type: ignore
    SessionLocal = _SessionLocal
except Exception:
    pass

if SessionLocal is None:
    try:
        # common pattern 2
        from app.core.db import get_db as _get_db  # type: ignore

        def _sessionlocal_from_get_db():
            gen = _get_db()
            s = next(gen)
            return s, gen

    except Exception:
        _sessionlocal_from_get_db = None  # type: ignore
else:
    _sessionlocal_from_get_db = None  # type: ignore

if SessionLocal is None and _sessionlocal_from_get_db is None:
    try:
        # common pattern 3
        from app.core.db import engine as _eng  # type: ignore
        from sqlalchemy.orm import sessionmaker

        _engine = _eng
        SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    except Exception:
        pass


def parse_int_arg(name: str, default: Optional[int] = None) -> Optional[int]:
    """
    Parse optional CLI args like --min-gw=24 / --max-gw=27
    """
    prefix = f"--{name}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            v = arg[len(prefix) :].strip()
            if v == "":
                return default
            try:
                return int(v)
            except ValueError:
                print(f"[ERROR] Invalid integer for {name}: {v}")
                sys.exit(1)
    return default


def parse_str_arg(name: str, default: str) -> str:
    """
    Parse optional CLI args like --model=baseline_rollavg_v0
    """
    prefix = f"--{name}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            v = arg[len(prefix) :].strip()
            return v if v else default
    return default


def open_session():
    """
    Returns (session, cleanup_fn)
    """
    if SessionLocal is not None:
        s = SessionLocal()
        return s, lambda: s.close()

    if _sessionlocal_from_get_db is not None:
        s, gen = _sessionlocal_from_get_db()
        def _cleanup():
            try:
                s.close()
            except Exception:
                pass
            try:
                gen.close()
            except Exception:
                pass
        return s, _cleanup

    raise RuntimeError(
        "Could not create DB session. Please check app.core.db and expose SessionLocal (or engine/get_db)."
    )


def main() -> None:
    model_name = parse_str_arg("model", "baseline_rollavg_v0")
    min_gw = parse_int_arg("min-gw", None)
    max_gw = parse_int_arg("max-gw", None)

    session, cleanup = open_session()
    try:
        run_eval(session, model_name=model_name, min_gw=min_gw, max_gw=max_gw)
    finally:
        cleanup()


def run_eval(session: Session, model_name: str, min_gw: Optional[int], max_gw: Optional[int]) -> None:
    """
    Compute per-GW MAE and overall MAE by joining:
      predictions (player_id, target_gw, predicted_points)
      player_gw_stats (player_id, gw, total_points)
    on (player_id, target_gw == gw)
    """

    # Base filters
    filters = [Prediction.model_name == model_name]

    if min_gw is not None:
        filters.append(Prediction.target_gw >= min_gw)
    if max_gw is not None:
        filters.append(Prediction.target_gw <= max_gw)

    # ---- 1) Count matched rows
    matched_count_stmt = (
        select(func.count())
        .select_from(Prediction)
        .join(
            PlayerGameweekStat,
            (Prediction.player_id == PlayerGameweekStat.player_id)
            & (Prediction.target_gw == PlayerGameweekStat.gw),
        )
        .where(*filters)
    )
    matched_rows = session.execute(matched_count_stmt).scalar_one()

    # ---- 2) Per-GW MAE
    # MAE = avg(abs(predicted_points - total_points))
    per_gw_stmt = (
        select(
            Prediction.target_gw.label("gw"),
            func.count().label("n"),
            func.avg(func.abs(Prediction.predicted_points - PlayerGameweekStat.total_points)).label("mae"),
        )
        .select_from(Prediction)
        .join(
            PlayerGameweekStat,
            (Prediction.player_id == PlayerGameweekStat.player_id)
            & (Prediction.target_gw == PlayerGameweekStat.gw),
        )
        .where(*filters)
        .group_by(Prediction.target_gw)
        .order_by(Prediction.target_gw.asc())
    )
    per_gw_rows = session.execute(per_gw_stmt).all()

    # ---- 3) Overall MAE
    overall_stmt = (
        select(
            func.count().label("n"),
            func.avg(func.abs(Prediction.predicted_points - PlayerGameweekStat.total_points)).label("mae"),
        )
        .select_from(Prediction)
        .join(
            PlayerGameweekStat,
            (Prediction.player_id == PlayerGameweekStat.player_id)
            & (Prediction.target_gw == PlayerGameweekStat.gw),
        )
        .where(*filters)
    )
    overall_row = session.execute(overall_stmt).one()
    overall_n = int(overall_row.n or 0)
    overall_mae = float(overall_row.mae) if overall_row.mae is not None else None

    # ---- Output (plain text, easy to copy into docs)
    print("=" * 72)
    print("Baseline Evaluation (MAE)")
    print("=" * 72)
    print(f"Model: {model_name}")
    print(f"GW filter: min_gw={min_gw if min_gw is not None else '-'} "
          f"max_gw={max_gw if max_gw is not None else '-'}")
    print(f"Matched rows: {matched_rows}")
    print()

    if not per_gw_rows:
        print("No matched rows found.")
        print("Possible reasons:")
        print("- predictions table has no rows for this model/GW range")
        print("- player_gw_stats is missing actual stats for those GWs")
        print("- player_id / GW join does not overlap yet")
        return

    print("Per-GW MAE")
    print("-" * 72)
    print(f"{'GW':>6} {'N':>8} {'MAE':>12}")
    for r in per_gw_rows:
        gw = int(r.gw)
        n = int(r.n or 0)
        mae = float(r.mae) if r.mae is not None else float("nan")
        print(f"{gw:>6} {n:>8} {mae:>12.4f}")

    print("-" * 72)
    if overall_mae is None:
        print(f"Overall MAE: N/A (N={overall_n})")
    else:
        print(f"Overall MAE: {overall_mae:.4f} (N={overall_n})")
    print("=" * 72)


if __name__ == "__main__":
    main()