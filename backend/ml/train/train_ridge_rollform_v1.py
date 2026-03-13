# backend/ml/train/train_ridge_rollform_v1.py
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

MODEL_NAME = "ridge_rollform_v1"


def build_raw_df(engine, gw_max: int) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT
                  s.player_id,
                  s.gw,
                  s.minutes,
                  s.goals_scored,
                  s.assists,
                  s.clean_sheets,
                  s.total_points,
                  p.position,
                  p.now_cost,
                  p.status,
                  p.team_id
                FROM player_gw_stats s
                JOIN players p ON p.id = s.player_id
                WHERE s.gw <= :gw_max
                ORDER BY s.player_id, s.gw
                """
            ),
            conn,
            params={"gw_max": gw_max},
        )
    if df.empty:
        raise RuntimeError(f"No rows found up to gw_max={gw_max}")
    return df


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    g = df.groupby("player_id", group_keys=False)

    df["pts_last1"] = g["total_points"].shift(1)
    df["mins_last1"] = g["minutes"].shift(1)

    for w in (3, 5, 8):
        df[f"pts_roll{w}_mean"] = g["total_points"].shift(1).rolling(w, min_periods=1).mean()
    for w in (3, 5):
        df[f"mins_roll{w}_mean"] = g["minutes"].shift(1).rolling(w, min_periods=1).mean()

    df["mins_roll5_nonzero_rate"] = (
        g["minutes"]
        .shift(1)
        .rolling(5, min_periods=1)
        .apply(lambda x: (pd.Series(x) > 0).mean())
    )

    df["g_roll5_sum"] = g["goals_scored"].shift(1).rolling(5, min_periods=1).sum()
    df["a_roll5_sum"] = g["assists"].shift(1).rolling(5, min_periods=1).sum()
    df["cs_roll5_sum"] = g["clean_sheets"].shift(1).rolling(5, min_periods=1).sum()

    df["now_cost_m"] = df["now_cost"] / 10.0

    # Drop first record per player (no lag)
    df = df.dropna(subset=["pts_last1", "mins_last1"]).reset_index(drop=True)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=["position", "status"], drop_first=False)

    # Define feature columns: exclude identifiers + raw target + raw match columns
    drop_cols = {
        "player_id",
        "gw",
        "team_id",
        "now_cost",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "total_points",  # target
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]

    return df, feature_cols


def write_predictions(engine, out_df: pd.DataFrame, target_gw: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM predictions
                WHERE target_gw = :target_gw AND model_name = :model_name
                """
            ),
            {"target_gw": target_gw, "model_name": MODEL_NAME},
        )

        out_df[["player_id", "target_gw", "model_name", "predicted_points"]].to_sql(
            "predictions", conn, if_exists="append", index=False, method="multi"
        )


def main(target_gw: int, split_gw: int, alpha: float = 1.0) -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError('DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"')
    engine = create_engine(db_url)

    # Need data up to target_gw to build features for rows at target_gw
    raw = build_raw_df(engine, gw_max=target_gw)
    feat, feature_cols = make_features(raw)

    # Train on <= split_gw, validate on (split_gw, target_gw)
    train = feat[feat["gw"] <= split_gw].copy()
    val = feat[(feat["gw"] > split_gw) & (feat["gw"] < target_gw)].copy()

    if train.empty:
        raise RuntimeError("Train set is empty. Lower split_gw.")
    X_train = train[feature_cols].to_numpy()
    y_train = train["total_points"].to_numpy()

    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X_train, y_train)

    if not val.empty:
        pred_val = model.predict(val[feature_cols].to_numpy())
        mae = mean_absolute_error(val["total_points"].to_numpy(), pred_val)
        print(f"val: model={MODEL_NAME} gw=({split_gw+1}..{target_gw-1}) rows={len(val)} MAE={mae:.4f}")
    else:
        print(f"val: empty (no rows in gw=({split_gw+1}..{target_gw-1}))")

    # Predict for target_gw
    target = feat[feat["gw"] == target_gw].copy()
    if target.empty:
        raise RuntimeError(f"No feature rows for target_gw={target_gw}. Ensure player_gw_stats has gw={target_gw} rows.")
    yhat = model.predict(target[feature_cols].to_numpy())

    out = pd.DataFrame(
        {
            "player_id": target["player_id"].astype(int),
            "target_gw": int(target_gw),
            "model_name": MODEL_NAME,
            "predicted_points": yhat.astype(float),
        }
    )

    write_predictions(engine, out, target_gw)
    print(f"OK: wrote {len(out)} rows for gw={target_gw} model={MODEL_NAME}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--target_gw", type=int, required=True)
    ap.add_argument("--split_gw", type=int, required=True, help="train on <= split_gw, validate after")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    main(args.target_gw, args.split_gw, args.alpha)