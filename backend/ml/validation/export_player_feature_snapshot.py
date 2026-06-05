
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.core.season import get_current_season


EXPORT_MODULE_BY_FEATURE_VERSION = {
    "v0": "ml.features.export_features_v0",
    "v2": "ml.features.export_features_v2",
    "v2_1": "ml.features.export_features_v2_1",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def default_player_csv_path(season: str, gw_start: int, gw_end: int, feature_version: str) -> str:
    return f"artifacts/offline_datasets/player_features_{season}_gw{gw_start}_{gw_end}_{feature_version}.csv"


def default_snapshot_json_path(snapshot_id: str) -> str:
    return f"artifacts/feature_snapshots/{snapshot_id}.json"


def resolve_export_module(feature_version: str) -> str:
    module_name = EXPORT_MODULE_BY_FEATURE_VERSION.get(feature_version)
    if not module_name:
        supported = ", ".join(sorted(EXPORT_MODULE_BY_FEATURE_VERSION))
        raise RuntimeError(
            f"Unsupported feature_version={feature_version!r}. Supported values: {supported}"
        )
    return module_name


def count_rows(csv_path: str) -> int:
    df = pd.read_csv(csv_path)
    return int(len(df))


def build_metadata(
    *,
    snapshot_id: str,
    season: str,
    gw_start: int,
    gw_end: int,
    feature_version: str,
    model_name: str,
    out_csv: str,
    row_count: int,
    export_module: str,
) -> dict:
    return {
        "snapshot_id": snapshot_id,
        "artifact_type": "player_feature_snapshot",
        "season": season,
        "training_season_start": season,
        "training_season_end": season,
        "evaluation_season": season,
        "gw_start": gw_start,
        "gw_end": gw_end,
        "feature_version": feature_version,
        "model_name": model_name,
        "export_module": export_module,
        "exported_csv": out_csv,
        "row_count": row_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw-start", type=int, required=True)
    ap.add_argument("--gw-end", type=int, required=True)
    ap.add_argument("--feature-version", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    season = get_current_season()
    out_csv = args.out_csv or default_player_csv_path(
        season=season,
        gw_start=args.gw_start,
        gw_end=args.gw_end,
        feature_version=args.feature_version,
    )

    export_module = resolve_export_module(args.feature_version)

    ensure_parent_dir(out_csv)

    export_cmd = [
        "python",
        "-m",
        export_module,
        "--start_gw",
        str(args.gw_start),
        "--end_gw",
        str(args.gw_end),
        "--out",
        out_csv,
    ]
    subprocess.run(export_cmd, check=True)

    row_count = count_rows(out_csv)

    short_id = uuid4().hex[:8]
    snapshot_id = f"player_features_{utc_stamp()}_{short_id}"
    metadata = build_metadata(
        snapshot_id=snapshot_id,
        season=season,
        gw_start=args.gw_start,
        gw_end=args.gw_end,
        feature_version=args.feature_version,
        model_name=args.model_name,
        out_csv=out_csv,
        row_count=row_count,
        export_module=export_module,
    )

    metadata_path = default_snapshot_json_path(snapshot_id)
    ensure_parent_dir(metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"exported_csv: {out_csv}")
    print(f"row_count: {row_count}")
    print(f"saved_metadata: {metadata_path}")
    print(f"snapshot_id: {snapshot_id}")


if __name__ == "__main__":
    main()
