import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


MODEL_NAME = "match_gbc_v2"
LABELS = ["H", "D", "A"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Dataset is empty: {csv_path}")
    required = {"gw", "result_label"}
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


def format_confusion_matrix(cm) -> str:
    header = "        pred_H  pred_D  pred_A"
    rows = [
        f"true_H {cm[0,0]:7d} {cm[0,1]:7d} {cm[0,2]:7d}",
        f"true_D {cm[1,0]:7d} {cm[1,1]:7d} {cm[1,2]:7d}",
        f"true_A {cm[2,0]:7d} {cm[2,1]:7d} {cm[2,2]:7d}",
    ]
    return "\n".join([header] + rows)


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
    y_train = train["result_label"].astype(str)

    X_val = val[feature_cols].fillna(0.0)
    y_val = val["result_label"].astype(str)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=0,
    )
    clf.fit(X_train, y_train)

    pred_val = clf.predict(X_val)
    proba_val = clf.predict_proba(X_val)

    acc = accuracy_score(y_val, pred_val)
    ll = log_loss(y_val, proba_val, labels=list(clf.classes_))
    cm = confusion_matrix(y_val, pred_val, labels=LABELS)

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
    print(f"val_accuracy: {acc:.4f}")
    print(f"val_logloss: {ll:.4f}")
    print("\nconfusion_matrix:")
    print(format_confusion_matrix(cm))

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
    print("\ntop_feature_importances:")
    for name, value in importances.items():
        print(f"  {name}: {value:.6f}")

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
                    "val_accuracy": float(acc),
                    "val_logloss": float(ll),
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
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--min-samples-leaf", type=int, default=5)
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
