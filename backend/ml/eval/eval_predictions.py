# backend/ml/eval/eval_predictions.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_absolute_error


def eval_model(model_name: str, start_gw: int, end_gw: int) -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError('DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"')

    engine = create_engine(db_url)

    q = text("""
        SELECT
          p.target_gw AS gw,
          p.player_id,
          p.predicted_points,
          s.total_points
        FROM predictions p
        JOIN player_gw_stats s
          ON s.player_id = p.player_id
         AND s.gw = p.target_gw
        WHERE p.model_name = :model_name
          AND p.target_gw BETWEEN :start_gw AND :end_gw
        ORDER BY p.target_gw, p.player_id
    """)

    with engine.begin() as conn:
        df = pd.read_sql(
            q,
            conn,
            params={"model_name": model_name, "start_gw": start_gw, "end_gw": end_gw},
        )

    if df.empty:
        raise RuntimeError(
            f"No joined rows found for model={model_name} in gw range [{start_gw}, {end_gw}]. "
            "Check that predictions exist for those GWs and that player_gw_stats has actuals."
        )

    overall = mean_absolute_error(df["total_points"], df["predicted_points"])

    per_gw = (
    df.groupby("gw", as_index=False)
      .agg(mae=("predicted_points", lambda x: 0.0))  # placeholder
)
    # compute MAE per group without groupby.apply
    per_gw["mae"] = (
        df.groupby("gw")
        .apply(lambda g: mean_absolute_error(g["total_points"], g["predicted_points"]), include_groups=False)
        .to_numpy()
    )
    per_gw = per_gw.sort_values("gw")

    print(f"model={model_name} gw=[{start_gw},{end_gw}] joined_rows={len(df)} overall_MAE={overall:.4f}")
    print(per_gw.to_string(index=False))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--start_gw", type=int, required=True)
    ap.add_argument("--end_gw", type=int, required=True)
    args = ap.parse_args()

    eval_model(args.model_name, args.start_gw, args.end_gw)


if __name__ == "__main__":
    main()