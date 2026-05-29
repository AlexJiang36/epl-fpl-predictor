import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


MODEL_NAME = "gbr_player_v2_1"


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


def main(
    csv_path: str,
    split_gw: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
) -> None:
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

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
    )
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)

    print(f"model_name: {MODEL_NAME}")
    print(f"csv: {csv_path}")
    print("feature_version: v2_1")
    print(f"split_gw: {split_gw}")
    print(f"n_estimators: {n_estimators}")
    print(f"learning_rate: {learning_rate}")
    print(f"max_depth: {max_depth}")
    print(f"min_samples_leaf: {min_samples_leaf}")
    print(f"train_rows: {len(train)}")
    print(f"val_rows: {len(val)}")
    print(f"n_features: {len(feature_cols)}")
    print(f"val_gw_range: {int(val['gw'].min())}..{int(val['gw'].max())}")
    print(f"val_mae: {mae:.4f}")

    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.sort_values(ascending=False).head(15)

    print("\ntop_feature_importances:")
    for name, value in top.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to player_features_v2_1 CSV")
    ap.add_argument("--split-gw", type=int, required=True)
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--min-samples-leaf", type=int, default=20)
    args = ap.parse_args()

    main(
        csv_path=args.csv,
        split_gw=args.split_gw,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
