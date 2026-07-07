# backend/ml/eval/eval_predictions.py
import argparse
import os
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_absolute_error

from app.core.season import get_current_season


def eval_model(model_name: str, start_gw: int, end_gw: int, season: Optional[str] = None) -> None:
    resolved_season = season or get_current_season()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            'DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"'
        )

    engine = create_engine(db_url)

    q = text(
        """
        SELECT
          p.season,
          p.target_gw AS gw,
          p.player_id,
          p.model_name,
          p.predicted_points,
          s.total_points
        FROM predictions p
        JOIN player_gw_stats s
          ON s.season = p.season
         AND s.player_id = p.player_id
         AND s.gw = p.target_gw
        WHERE p.season = :season
          AND s.season = :season
          AND p.model_name = :model_name
          AND p.target_gw BETWEEN :start_gw AND :end_gw
        ORDER BY p.target_gw, p.player_id
        """
    )

    with engine.begin() as conn:
        df = pd.read_sql(
            q,
            conn,
            params={
                "season": resolved_season,
                "model_name": model_name,
                "start_gw": start_gw,
                "end_gw": end_gw,
            },
        )

    if df.empty:
        raise RuntimeError(
            f"No joined rows found for season={resolved_season}, model={model_name} "
            f"in gw range [{start_gw}, {end_gw}]. "
            "Check that predictions exist for those GWs and that player_gw_stats has actuals."
        )

    overall = mean_absolute_error(df["total_points"], df["predicted_points"])

    per_gw_rows = []
    for gw, group in df.groupby("gw"):
        per_gw_rows.append(
            {
                "gw": int(gw),
                "rows": int(len(group)),
                "mae": float(mean_absolute_error(group["total_points"], group["predicted_points"])),
            }
        )
    per_gw = pd.DataFrame(per_gw_rows).sort_values("gw")

    print(
        f"season={resolved_season} model={model_name} "
        f"gw=[{start_gw},{end_gw}] joined_rows={len(df)} overall_MAE={overall:.4f}"
    )
    print(per_gw.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--start_gw", type=int, required=True)
    ap.add_argument("--end_gw", type=int, required=True)
    ap.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season key, for example 2025_26. Defaults to app.core.season.get_current_season().",
    )
    args = ap.parse_args()

    eval_model(
        model_name=args.model_name,
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        season=args.season,
    )


if __name__ == "__main__":
    main()
