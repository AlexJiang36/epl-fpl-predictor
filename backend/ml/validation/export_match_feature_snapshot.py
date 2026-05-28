from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from app.utils.feature_snapshot_store import (
    create_feature_snapshot_artifact,
    save_feature_snapshot_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export match feature snapshot with metadata artifact."
    )
    parser.add_argument("--gw-start", type=int, required=True)
    parser.add_argument("--gw-end", type=int, required=True)
    parser.add_argument("--feature-version", type=str, default="v0")
    parser.add_argument("--model-name", type=str, default="match_baseline_v0")
    parser.add_argument("--n-form", type=int, default=5)
    parser.add_argument("--n-h2h", type=int, default=3)
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV path, e.g. artifacts/offline_datasets/match_features_gw1_27_v2.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.feature_version == "v0":
        module_name = "ml.features.export_match_dataset_v0"
        source_tables = ["fixtures"]
        notes = "Offline match feature snapshot export (v0)."
        export_cmd = [
            "python",
            "-m",
            module_name,
            "--start_gw",
            str(args.gw_start),
            "--end_gw",
            str(args.gw_end),
            "--n_form",
            str(args.n_form),
            "--out",
            str(out_csv),
        ]
    elif args.feature_version == "v2":
        module_name = "ml.features.export_match_dataset_v2"
        source_tables = ["fixtures"]
        notes = "Offline match feature snapshot export (v2)."
        export_cmd = [
            "python",
            "-m",
            module_name,
            "--start_gw",
            str(args.gw_start),
            "--end_gw",
            str(args.gw_end),
            "--n_form",
            str(args.n_form),
            "--n_h2h",
            str(args.n_h2h),
            "--out",
            str(out_csv),
        ]
    else:
        raise RuntimeError(f"Unsupported feature version: {args.feature_version}")

    child_env = os.environ.copy()
    subprocess.run(export_cmd, check=True, env=child_env)

    row_count = 0
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8") as f:
            row_count = max(sum(1 for _ in f) - 1, 0)

    artifact = create_feature_snapshot_artifact(
        snapshot_type="match_features",
        feature_version=args.feature_version,
        gw_start=args.gw_start,
        gw_end=args.gw_end,
        output_path=str(out_csv),
        source_tables=source_tables,
        model_name=args.model_name,
        row_count=row_count,
        notes=notes,
    )

    meta_path = save_feature_snapshot_artifact(artifact)

    print("exported_csv:", out_csv)
    print("row_count:", row_count)
    print("saved_metadata:", meta_path)
    print("snapshot_id:", artifact.snapshot_id)


if __name__ == "__main__":
    main()
