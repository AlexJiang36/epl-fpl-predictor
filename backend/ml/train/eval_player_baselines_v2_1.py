import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
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


def evaluate_baselines(
    csv_path: str,
    split_gw: int,
    baseline_cols: List[str],
    out_csv: Optional[str] = None,
) -> None:
    df = load_dataset(csv_path)

    val = df[df["gw"] > split_gw].copy()
    if val.empty:
        raise RuntimeError("Validation set is empty. Raise end_gw or lower split_gw.")

    rows = []
    missing = []

    for col in baseline_cols:
        if col not in val.columns:
            missing.append(col)
            continue

        tmp = val[["gw", "total_points", col]].copy()
        tmp = tmp.dropna(subset=[col])

        if tmp.empty:
            rows.append(
                {
                    "baseline_name": col,
                    "split_gw": split_gw,
                    "val_rows": 0,
                    "val_gw_start": int(val["gw"].min()),
                    "val_gw_end": int(val["gw"].max()),
                    "val_mae": None,
                }
            )
            continue

        mae = mean_absolute_error(tmp["total_points"].to_numpy(), tmp[col].to_numpy())

        rows.append(
            {
                "baseline_name": col,
                "split_gw": split_gw,
                "val_rows": int(len(tmp)),
                "val_gw_start": int(tmp["gw"].min()),
                "val_gw_end": int(tmp["gw"].max()),
                "val_mae": float(mae),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["val_mae", "baseline_name"], ascending=[True, True], na_position="last").reset_index(drop=True)

    print("player baseline evaluation")
    print(f"csv: {csv_path}")
    print("feature_version: derived from supplied dataset")
    print(f"split_gw: {split_gw}")
    print(f"full_val_rows: {len(val)}")
    print(f"val_gw_range: {int(val['gw'].min())}..{int(val['gw'].max())}")
    print()

    if missing:
        print("missing_baseline_columns:")
        for col in missing:
            print(f"  {col}")
        print()

    if result.empty:
        print("No valid baseline results.")
    else:
        print(result.to_string(index=False))

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print()
        print(f"saved_results: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to player features CSV")
    ap.add_argument("--split-gw", type=int, required=True)
    ap.add_argument(
        "--baseline-cols",
        nargs="+",
        default=["pts_last1", "pts_roll3_mean", "pts_roll5_mean", "pts_roll8_mean"],
        help="Baseline prediction columns to evaluate",
    )
    ap.add_argument("--out-csv", default="", help="Optional path to save evaluation results CSV")
    args = ap.parse_args()

    evaluate_baselines(
        csv_path=args.csv,
        split_gw=args.split_gw,
        baseline_cols=args.baseline_cols,
        out_csv=args.out_csv or None,
    )
