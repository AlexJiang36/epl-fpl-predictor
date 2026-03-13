# backend/ml/predict/predict_next_gw_ridge_rollform_v1.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.linear_model import Ridge

MODEL_NAME = "ridge_rollform_v1"

def build_raw_df(engine, gw_max: int) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(
            text("""
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
            """),
            conn,
            params={"gw_max": gw_max},
        )
    if df.empty:
        raise RuntimeError(f"No rows found up to gw_max={gw_max}")
    return df

def make_features(df: pd.DataFrame):
    df = df.copy()
    g = df.groupby("player_id", group_keys=False)

    df["pts_last1"] = g["total_points"].shift(1)
    df["mins_last1"] = g["minutes"].shift(1)

    for w in (3, 5, 8):
        df[f"pts_roll{w}_mean"] = g["total_points"].shift(1).rolling(w, min_periods=1).mean()
    for w in (3, 5):
        df[f"mins_roll{w}_mean"] = g["minutes"].shift(1).rolling(w, min_periods=1).mean()

    df["mins_roll5_nonzero_rate"] = (
        g["minutes"].shift(1).rolling(5, min_periods=1)
         .apply(lambda x: (pd.Series(x) > 0).mean())
    )

    df["g_roll5_sum"] = g["goals_scored"].shift(1).rolling(5, min_periods=1).sum()
    df["a_roll5_sum"] = g["assists"].shift(1).rolling(5, min_periods=1).sum()
    df["cs_roll5_sum"] = g["clean_sheets"].shift(1).rolling(5, min_periods=1).sum()

    df["now_cost_m"] = df["now_cost"] / 10.0

    df = df.dropna(subset=["pts_last1", "mins_last1"]).reset_index(drop=True)
    df = pd.get_dummies(df, columns=["position", "status"], drop_first=False)

    drop_cols = {
        "player_id", "gw", "team_id", "now_cost",
        "minutes", "goals_scored", "assists", "clean_sheets",
        "total_points",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df, feature_cols

def write_predictions(engine, out_df: pd.DataFrame, target_gw: int):
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM predictions WHERE target_gw=:target_gw AND model_name=:model_name"),
            {"target_gw": target_gw, "model_name": MODEL_NAME},
        )
        out_df[["player_id", "target_gw", "model_name", "predicted_points"]].to_sql(
            "predictions", conn, if_exists="append", index=False, method="multi"
        )

def main(last_gw: int, split_gw: int, alpha: float = 1.0):
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError('DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"')
    engine = create_engine(db_url)

    raw = build_raw_df(engine, gw_max=last_gw)
    feat, feature_cols = make_features(raw)

    train = feat[feat["gw"] <= split_gw].copy()
    if train.empty:
        raise RuntimeError("Train set is empty. Lower split_gw.")

    X_train = train[feature_cols].to_numpy()
    y_train = train["total_points"].to_numpy()

    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X_train, y_train)

    # Forecast next GW using feature rows at last_gw
    target_gw = last_gw + 1
    X_forecast = feat[feat["gw"] == last_gw].copy()
    if X_forecast.empty:
        raise RuntimeError(f"No feature rows for last_gw={last_gw}.")
    yhat = model.predict(X_forecast[feature_cols].to_numpy())

    out = pd.DataFrame({
        "player_id": X_forecast["player_id"].astype(int),
        "target_gw": int(target_gw),
        "model_name": MODEL_NAME,
        "predicted_points": yhat.astype(float),
    })

    write_predictions(engine, out, target_gw)
    print(f"OK: wrote {len(out)} rows for target_gw={target_gw} model={MODEL_NAME} (forecast from last_gw={last_gw})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--last_gw", type=int, required=True, help="latest GW with actuals in player_gw_stats")
    ap.add_argument("--split_gw", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    main(args.last_gw, args.split_gw, args.alpha)