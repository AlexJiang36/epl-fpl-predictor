import pandas as pd
from typing import List, Optional
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


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


def main(csv_path: str, split_gw: int, alphas: List[float], out_csv: Optional[str] = None) -> None:
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

    rows = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=0)
        model.fit(X_train, y_train)

        pred_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred_val)

        rows.append(
            {
                "model_name": "ridge_player_v2_1",
                "feature_version": "v2_1",
                "split_gw": split_gw,
                "alpha": float(alpha),
                "train_rows": int(len(train)),
                "val_rows": int(len(val)),
                "n_features": int(len(feature_cols)),
                "val_gw_start": int(val["gw"].min()),
                "val_gw_end": int(val["gw"].max()),
                "val_mae": float(mae),
            }
        )

    result = pd.DataFrame(rows).sort_values(["val_mae", "alpha"], ascending=[True, True]).reset_index(drop=True)

    print("ridge_player_v2_1 alpha sweep")
    print(f"csv: {csv_path}")
    print("feature_version: v2_1")
    print(f"split_gw: {split_gw}")
    print(f"train_rows: {len(train)}")
    print(f"val_rows: {len(val)}")
    print(f"n_features: {len(feature_cols)}")
    print(f"val_gw_range: {int(val['gw'].min())}..{int(val['gw'].max())}")
    print()
    print(result.to_string(index=False))

    if out_csv:
        result.to_csv(out_csv, index=False)
        print()
        print(f"saved_results: {out_csv}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to player_features_v2_1 CSV")
    ap.add_argument("--split-gw", type=int, required=True)
    ap.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        required=True,
        help="List of alpha values, e.g. --alphas 0.1 0.3 1 3 10",
    )
    ap.add_argument("--out-csv", default="", help="Optional path to save sweep results CSV")
    args = ap.parse_args()

    main(
        csv_path=args.csv,
        split_gw=args.split_gw,
        alphas=args.alphas,
        out_csv=args.out_csv or None,
    )
