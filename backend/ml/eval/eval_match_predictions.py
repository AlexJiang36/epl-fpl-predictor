# backend/ml/eval/eval_match_predictions.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

LABELS = ["H", "D", "A"]


def eval_match_model(model_name: str, start_gw: int, end_gw: int, use_logloss: bool = True) -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError('DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"')
    engine = create_engine(db_url)

    q = text("""
        SELECT
          f.gw,
          f.id AS fixture_id,
          f.home_score,
          f.away_score,
          mp.pred_home_win,
          mp.pred_draw,
          mp.pred_away_win,
          mp.pred_result
        FROM fixtures f
        JOIN match_predictions mp
          ON mp.fixture_id = f.id
        WHERE f.gw BETWEEN :start_gw AND :end_gw
          AND f.finished = TRUE
          AND f.home_score IS NOT NULL
          AND f.away_score IS NOT NULL
          AND mp.model_name = :model_name
        ORDER BY f.gw, f.id
    """)

    with engine.begin() as conn:
        df = pd.read_sql(q, conn, params={"start_gw": start_gw, "end_gw": end_gw, "model_name": model_name})

    if df.empty:
        raise RuntimeError(f"No joined rows found for model={model_name} in gw [{start_gw}, {end_gw}]")

    # true label
    def y(row):
        hs = int(row["home_score"])
        a = int(row["away_score"])
        if hs > a:
            return "H"
        if hs < a:
            return "A"
        return "D"

    df["y_true"] = df.apply(y, axis=1)

    acc = accuracy_score(df["y_true"], df["pred_result"])
    print(f"model={model_name} gw=[{start_gw},{end_gw}] rows={len(df)} accuracy={acc:.4f}")

    # confusion matrix
    cm = confusion_matrix(df["y_true"], df["pred_result"], labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
    print("\nConfusion matrix:")
    print(cm_df.to_string())

    if use_logloss:
        # map probs to label order H,D,A
        probs = df[["pred_home_win", "pred_draw", "pred_away_win"]].astype(float).to_numpy()
        # convert y_true to indices in LABELS
        y_idx = df["y_true"].apply(lambda v: LABELS.index(v)).to_numpy()
        ll = log_loss(y_idx, probs, labels=[0, 1, 2])
        print(f"\nlogloss={ll:.4f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--start_gw", type=int, required=True)
    ap.add_argument("--end_gw", type=int, required=True)
    ap.add_argument("--no_logloss", action="store_true")
    args = ap.parse_args()

    eval_match_model(args.model_name, args.start_gw, args.end_gw, use_logloss=not args.no_logloss)


if __name__ == "__main__":
    main()
