import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


MODEL_NAME = "match_goals_gbr_v2"


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Dataset is empty: {csv_path}")
    required = {"gw", "home_goals", "away_goals"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = {
        "fixture_id",
        "gw",
        "kickoff_time",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "home_goals",
        "away_goals",
        "result_label",
    }
    return [c for c in df.columns if c not in drop_cols]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fit_one_target(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    subsample: float,
):
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=0,
    )
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    return model, pred_val


def print_top_importances(title: str, model, feature_cols: list[str]) -> None:
    importances = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(12)
    )
    print(f"\n{title}:")
    for name, value in importances.items():
        print(f"  {name}: {value:.6f}")


def main(
    csv_path: str,
    split_gw: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    subsample: float,
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

    X_train = train[feature_cols].fillna(0.0)
    X_val = val[feature_cols].fillna(0.0)

    home_model, home_pred = fit_one_target(
        X_train=X_train,
        y_train=train["home_goals"],
        X_val=X_val,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
    )

    away_model, away_pred = fit_one_target(
        X_train=X_train,
        y_train=train["away_goals"],
        X_val=X_val,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
    )

    home_mae = mean_absolute_error(val["home_goals"], home_pred)
    away_mae = mean_absolute_error(val["away_goals"], away_pred)
    home_rmse = rmse(val["home_goals"], home_pred)
    away_rmse = rmse(val["away_goals"], away_pred)

    avg_mae = float((home_mae + away_mae) / 2.0)
    avg_rmse = float((home_rmse + away_rmse) / 2.0)

    print(f"model_name: {MODEL_NAME}")
    print(f"csv: {csv_path}")
    print("feature_version: v2")
    print(f"split_gw: {split_gw}")
    print(f"n_estimators: {n_estimators}")
    print(f"learning_rate: {learning_rate}")
    print(f"max_depth: {max_depth}")
    print(f"min_samples_leaf: {min_samples_leaf}")
    print(f"subsample: {subsample}")
    print(f"train_rows: {len(train)}")
    print(f"val_rows: {len(val)}")
    print(f"n_features: {len(feature_cols)}")
    print(f"val_gw_range: {int(val['gw'].min())}..{int(val['gw'].max())}")
    print(f"home_goals_mae: {home_mae:.4f}")
    print(f"away_goals_mae: {away_mae:.4f}")
    print(f"home_goals_rmse: {home_rmse:.4f}")
    print(f"away_goals_rmse: {away_rmse:.4f}")
    print(f"avg_goal_mae: {avg_mae:.4f}")
    print(f"avg_goal_rmse: {avg_rmse:.4f}")

    print_top_importances("top_home_goal_feature_importances", home_model, feature_cols)
    print_top_importances("top_away_goal_feature_importances", away_model, feature_cols)

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = pd.DataFrame(
            [
                {
                    "model_name": MODEL_NAME,
                    "feature_version": "v2",
                    "split_gw": split_gw,
                    "n_estimators": int(n_estimators),
                    "learning_rate": float(learning_rate),
                    "max_depth": int(max_depth),
                    "min_samples_leaf": int(min_samples_leaf),
                    "subsample": float(subsample),
                    "train_rows": int(len(train)),
                    "val_rows": int(len(val)),
                    "n_features": int(len(feature_cols)),
                    "val_gw_start": int(val["gw"].min()),
                    "val_gw_end": int(val["gw"].max()),
                    "home_goals_mae": float(home_mae),
                    "away_goals_mae": float(away_mae),
                    "home_goals_rmse": float(home_rmse),
                    "away_goals_rmse": float(away_rmse),
                    "avg_goal_mae": float(avg_mae),
                    "avg_goal_rmse": float(avg_rmse),
                }
            ]
        )
        result.to_csv(out_path, index=False)
        print()
        print(f"saved_results: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to match_features_v2 CSV")
    ap.add_argument("--split-gw", type=int, required=True)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--min-samples-leaf", type=int, default=3)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--out-csv", default="", help="Optional path to save summary CSV")
    args = ap.parse_args()

    main(
        csv_path=args.csv,
        split_gw=args.split_gw,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        out_csv=args.out_csv or None,
    )
