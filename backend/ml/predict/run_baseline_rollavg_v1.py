import os
import pandas as pd
from sqlalchemy import create_engine, text

MODEL_NAME = "baseline_rollavg_v1"

def main(target_gw: int, window: int = 5):
    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)

    with engine.begin() as conn:
        stats = pd.read_sql(
            text("""
                SELECT player_id, gw, minutes, total_points
                FROM player_gw_stats
                WHERE gw < :target_gw
                ORDER BY player_id, gw
            """),
            conn,
            params={"target_gw": target_gw},
        )

        if stats.empty:
            raise RuntimeError("No historical rows found in player_gw_stats before target_gw")

        def wavg(group: pd.DataFrame) -> float:
            g = group.tail(window)
            w = g["minutes"].astype(float).clip(lower=0.0)
            y = g["total_points"].astype(float)
            denom = w.sum()
            if denom <= 0:
                return float(y.tail(1).iloc[0]) if len(y) else 0.0
            return float((w * y).sum() / denom)

        preds = (
            stats.sort_values(["player_id", "gw"])
                .groupby("player_id", as_index=False)
                .agg(
                    predicted_points=("total_points", "last")  # placeholder, overwritten below
                )
)

        # compute weighted avg safely per player
        def wavg_df(g):
            g = g.tail(window)
            w = g["minutes"].astype(float).clip(lower=0.0)
            y = g["total_points"].astype(float)
            denom = w.sum()
            return float((w * y).sum() / denom) if denom > 0 else float(y.iloc[-1]) if len(y) else 0.0

        preds["predicted_points"] = (
            stats.sort_values(["player_id", "gw"])
                .groupby("player_id")
                .apply(wavg_df, include_groups=False)
                .to_numpy()
        )

        preds["target_gw"] = target_gw
        preds["model_name"] = MODEL_NAME

        conn.execute(
            text("""
                DELETE FROM predictions
                WHERE target_gw = :target_gw AND model_name = :model_name
            """),
            {"target_gw": target_gw, "model_name": MODEL_NAME},
        )

        preds[["player_id", "target_gw", "model_name", "predicted_points"]].to_sql(
            "predictions", conn, if_exists="append", index=False, method="multi"
        )

    print(f"OK: wrote {len(preds)} rows for gw={target_gw} model={MODEL_NAME}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gw", type=int, required=True)
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()
    main(args.target_gw, args.window)