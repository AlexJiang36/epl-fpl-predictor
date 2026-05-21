from __future__ import annotations

import argparse
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
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV path, e.g. artifacts/offline_datasets/match_features_gw1_27_v0.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    export_cmd = [
        "python",
        "-m",
        "ml.features.export_match_dataset_v0",
        "--start_gw",
        str(args.gw_start),
        "--end_gw",
        str(args.gw_end),
        "--out",
        str(out_csv),
    ]

    subprocess.run(export_cmd, check=True)

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
        source_tables=["fixtures", "teams"],
        model_name=None,
        row_count=row_count,
        notes="Offline match feature snapshot export",
    )

    meta_path = save_feature_snapshot_artifact(artifact)

    print("exported_csv:", out_csv)
    print("row_count:", row_count)
    print("saved_metadata:", meta_path)
    print("snapshot_id:", artifact.snapshot_id)


if __name__ == "__main__":
    main()