import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main(
    csv_path: str,
    split_gw: int,
    n_estimators_list: List[int],
    learning_rate_list: List[float],
    max_depth_list: List[int],
    min_samples_leaf_list: List[int],
    out_csv: Optional[str] = None,
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

    rows = []
    for n_estimators in n_estimators_list:
        for learning_rate in learning_rate_list:
            for max_depth in max_depth_list:
                for min_samples_leaf in min_samples_leaf_list:
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

                    rows.append(
                        {
                            "model_name": "gbr_player_v2_1",
                            "feature_version": "v2_1",
                            "split_gw": split_gw,
                            "n_estimators": int(n_estimators),
                            "learning_rate": float(learning_rate),
                            "max_depth": int(max_depth),
                            "min_samples_leaf": int(min_samples_leaf),
                            "train_rows": int(len(train)),
                            "val_rows": int(len(val)),
                            "n_features": int(len(feature_cols)),
                            "val_gw_start": int(val["gw"].min()),
                            "val_gw_end": int(val["gw"].max()),
                            "val_mae": float(mae),
                        }
                    )

    result = pd.DataFrame(rows).sort_values(
        ["val_mae", "n_estimators", "learning_rate", "max_depth", "min_samples_leaf"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    print("gbr_player_v2_1 hyperparameter sweep")
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
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print()
        print(f"saved_results: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to player_features_v2_1 CSV")
    ap.add_argument("--split-gw", type=int, required=True)
    ap.add_argument("--n-estimators-list", default="100,150,250")
    ap.add_argument("--learning-rate-list", default="0.03,0.05,0.1")
    ap.add_argument("--max-depth-list", default="2,3")
    ap.add_argument("--min-samples-leaf-list", default="10,20,40")
    ap.add_argument("--out-csv", default="", help="Optional path to save sweep results CSV")
    args = ap.parse_args()

    main(
        csv_path=args.csv,
        split_gw=args.split_gw,
        n_estimators_list=parse_int_list(args.n_estimators_list),
        learning_rate_list=parse_float_list(args.learning_rate_list),
        max_depth_list=parse_int_list(args.max_depth_list),
        min_samples_leaf_list=parse_int_list(args.min_samples_leaf_list),
        out_csv=args.out_csv or None,
    )
