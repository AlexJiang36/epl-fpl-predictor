import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


MODEL_NAME = "ridge_player_v2_1"


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Dataset is empty: {csv_path}")
    if "gw" not in df.columns:
        raise RuntimeError("Expected column 'gw' in dataset.")
    if "total_points" not in df.columns:
        raise RuntimeError("Expected target column 'total_points' in dataset.")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = {
        "player_id",
        "gw",
        "team_id",
        "now_cost",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "total_points",
    }
    return [c for c in df.columns if c not in drop_cols]


def main(csv_path: str, split_gw: int, alpha: float) -> None:
    df = load_dataset(csv_path)
    feature_cols = get_feature_columns(df)

    if not feature_cols:
        raise RuntimeError("No feature columns found after dropping identifiers/targets.")

    train = df[df["gw"] <= split_gw].copy()
    val = df[df["gw"] > split_gw].copy()

    if train.empty:
        raise RuntimeError("Train set is empty. Lower split_gw or check dataset.")
    if val.empty:
        raise RuntimeError("Validation set is empty. Raise end_gw or lower split_gw.")

    X_train = train[feature_cols].fillna(0.0).to_numpy()
    y_train = train["total_points"].to_numpy()

    X_val = val[feature_cols].fillna(0.0).to_numpy()
    y_val = val["total_points"].to_numpy()

    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)

    print(f"model_name: {MODEL_NAME}")
    print(f"csv: {csv_path}")
    print("feature_version: v2_1")
    print(f"split_gw: {split_gw}")
    print(f"alpha: {alpha}")
    print(f"train_rows: {len(train)}")
    print(f"val_rows: {len(val)}")
    print(f"n_features: {len(feature_cols)}")
    print(f"val_gw_range: {int(val['gw'].min())}..{int(val['gw'].max())}")
    print(f"val_mae: {mae:.4f}")

    # Top feature weights by absolute magnitude (simple sanity check)
    coef = pd.Series(model.coef_, index=feature_cols)
    top = coef.abs().sort_values(ascending=False).head(15).index.tolist()

    print("\ntop_feature_weights:")
    for name in top:
        print(f"  {name}: {coef[name]:.6f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to player_features_v2_1 CSV")
    ap.add_argument("--split-gw", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    main(args.csv, args.split_gw, args.alpha)
