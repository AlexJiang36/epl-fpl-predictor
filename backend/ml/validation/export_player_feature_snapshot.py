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
        description="Export player feature snapshot with metadata artifact."
    )
    parser.add_argument("--gw-start", type=int, required=True)
    parser.add_argument("--gw-end", type=int, required=True)
    parser.add_argument("--feature-version", type=str, default="v0")
    parser.add_argument("--model-name", type=str, default="baseline_rollavg_v0")
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV path, e.g. artifacts/offline_datasets/player_features_gw1_27_v2.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.feature_version == "v0":
        module_name = "ml.features.export_features_v0"
        source_tables = ["player_gw_stats", "players"]
        notes = "Offline player feature snapshot export (v0)."
    elif args.feature_version == "v2":
        module_name = "ml.features.export_features_v2"
        source_tables = ["player_gw_stats", "players", "fixtures"]
        notes = "Offline player feature snapshot export (v2)."
    elif args.feature_version == "v2_1":
        module_name = "ml.features.export_features_v2_1"
        source_tables = ["player_gw_stats", "players", "fixtures"]
        notes = "Offline player feature snapshot export (v2_1) with intrinsic / availability proxies."
    else:
        raise RuntimeError(f"Unsupported feature version: {args.feature_version}")

    export_cmd = [
        "python",
        "-m",
        module_name,
        "--start_gw",
        str(args.gw_start),
        "--end_gw",
        str(args.gw_end),
        "--out",
        str(out_csv),
    ]

    child_env = os.environ.copy()
    subprocess.run(export_cmd, check=True, env=child_env)

    row_count = 0
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8") as f:
            row_count = max(sum(1 for _ in f) - 1, 0)

    artifact = create_feature_snapshot_artifact(
        snapshot_type="player_features",
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
