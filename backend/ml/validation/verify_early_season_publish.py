from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pandas as pd
from sqlalchemy import text


VERIFIER_VERSION = "early_season_post_publish_verify_v0_1"
EXPECTED_RECEIPT_CONTRACT = "early_season_prediction_publish_receipt_v1"
EXPECTED_SNAPSHOT_CONTRACT = "immutable_pre_deadline_model_publish_snapshot_v1"
EXPECTED_PLAYER_MODEL = "early_season_blend_player_v0"
EXPECTED_MATCH_MODEL = "early_season_blend_match_v0"
FLOAT_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an Early Season Prediction publish by reconciling the source preview, "
            "immutable published PRE snapshot, publish receipt, and canonical DB rows."
        )
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--receipt", type=str, required=True)
    parser.add_argument("--season", type=str, required=True)
    parser.add_argument("--target-gw", type=int, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonical_match_result_label(value: Any) -> str:
    raw = str(value).strip().lower()
    mapping = {
        "home_win": "H",
        "draw": "D",
        "away_win": "A",
        "h": "H",
        "d": "D",
        "a": "A",
    }
    if raw not in mapping:
        raise RuntimeError("Unsupported match result label: %r" % value)
    return mapping[raw]


def compare_float(a: Any, b: Any, tol: float = FLOAT_TOLERANCE) -> bool:
    try:
        av = float(a)
        bv = float(b)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(av) and math.isfinite(bv)):
        return False
    return abs(av - bv) <= tol


def validate_receipt_and_snapshot(
    run_dir: Path,
    receipt_path: Path,
    season: str,
    target_gw: int,
) -> Dict[str, Any]:
    blockers: List[str] = []
    checks: Dict[str, Any] = {}

    source_manifest_path = run_dir / "run_manifest.json"
    source_player_path = run_dir / "player_predictions_preview.csv"
    source_match_path = run_dir / "match_predictions_preview.csv"
    source_scoreline_path = run_dir / "scoreline_preview.csv"

    for path in [
        source_manifest_path,
        source_player_path,
        source_match_path,
        source_scoreline_path,
        receipt_path,
    ]:
        if not path.exists():
            blockers.append("Required file does not exist: %s" % path)

    if blockers:
        return {"blockers": blockers, "checks": checks}

    source_manifest = load_json(source_manifest_path)
    receipt = load_json(receipt_path)

    if receipt.get("receipt_contract") != EXPECTED_RECEIPT_CONTRACT:
        blockers.append("Unexpected receipt_contract.")
    if str(receipt.get("season")) != season:
        blockers.append("Receipt season mismatch.")
    if int(receipt.get("target_gw") or -1) != int(target_gw):
        blockers.append("Receipt target_gw mismatch.")
    if receipt.get("database_prediction_write") is not True:
        blockers.append("Receipt must record database_prediction_write=true.")
    if receipt.get("final_deadline_freeze") is not False:
        blockers.append("This verifier expects a PRE publish, not a final deadline freeze.")

    source_run_id = str(source_manifest.get("run_id") or "")
    if str(receipt.get("source_run_id") or "") != source_run_id:
        blockers.append("Receipt source_run_id does not match source preview.")
    if str(source_manifest.get("season")) != season:
        blockers.append("Source manifest season mismatch.")
    if int(source_manifest.get("target_gw") or -1) != int(target_gw):
        blockers.append("Source manifest target_gw mismatch.")
    if source_manifest.get("status") != "PASS_PREVIEW":
        blockers.append("Source manifest status is not PASS_PREVIEW.")

    snapshot_dir_raw = receipt.get("snapshot_dir")
    if not snapshot_dir_raw:
        blockers.append("Receipt is missing snapshot_dir.")
        return {
            "blockers": blockers,
            "checks": checks,
            "receipt": receipt,
            "source_manifest": source_manifest,
        }

    snapshot_dir = Path(str(snapshot_dir_raw)).expanduser().resolve()
    snapshot_manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not snapshot_dir.exists():
        blockers.append("Snapshot directory does not exist: %s" % snapshot_dir)
    elif not snapshot_manifest_path.exists():
        blockers.append("Snapshot manifest does not exist: %s" % snapshot_manifest_path)
    else:
        actual_snapshot_manifest_sha = sha256_file(snapshot_manifest_path)
        expected_snapshot_manifest_sha = str(receipt.get("snapshot_manifest_sha256") or "")
        checks["snapshot_manifest_sha256"] = {
            "expected": expected_snapshot_manifest_sha,
            "actual": actual_snapshot_manifest_sha,
            "matched": actual_snapshot_manifest_sha == expected_snapshot_manifest_sha,
        }
        if actual_snapshot_manifest_sha != expected_snapshot_manifest_sha:
            blockers.append("Snapshot manifest SHA256 does not match publish receipt.")

        snapshot_manifest = load_json(snapshot_manifest_path)
        if snapshot_manifest.get("snapshot_contract") != EXPECTED_SNAPSHOT_CONTRACT:
            blockers.append("Unexpected snapshot_contract.")
        if str(snapshot_manifest.get("source_run_id") or "") != source_run_id:
            blockers.append("Snapshot source_run_id mismatch.")
        if str(snapshot_manifest.get("season")) != season:
            blockers.append("Snapshot season mismatch.")
        if int(snapshot_manifest.get("target_gw") or -1) != int(target_gw):
            blockers.append("Snapshot target_gw mismatch.")
        if snapshot_manifest.get("overwrite_allowed") is not False:
            blockers.append("Published PRE snapshot must be immutable (overwrite_allowed=false).")
        if snapshot_manifest.get("final_deadline_freeze") is not False:
            blockers.append("Published PRE snapshot must not claim final deadline freeze.")

        artifact_hashes = dict(snapshot_manifest.get("artifact_sha256") or {})
        for name in [
            "run_manifest.json",
            "player_predictions_preview.csv",
            "match_predictions_preview.csv",
            "scoreline_preview.csv",
            "bootstrap_snapshot.json",
            "summary.md",
        ]:
            source_path = run_dir / name
            snapshot_path = snapshot_dir / name
            if not source_path.exists() or not snapshot_path.exists():
                blockers.append("Source/snapshot artifact missing: %s" % name)
                continue
            source_sha = sha256_file(source_path)
            snapshot_sha = sha256_file(snapshot_path)
            declared_sha = str(artifact_hashes.get(name) or "")
            matched = source_sha == snapshot_sha == declared_sha
            checks.setdefault("artifact_hashes", {})[name] = {
                "source": source_sha,
                "snapshot": snapshot_sha,
                "declared": declared_sha,
                "matched": matched,
            }
            if not matched:
                blockers.append("Source/snapshot artifact hash mismatch: %s" % name)

        prior_checks = dict(snapshot_manifest.get("prior_hash_checks") or {})
        for key, payload in prior_checks.items():
            if not isinstance(payload, dict):
                blockers.append("Malformed prior_hash_check entry: %s" % key)
                continue
            path_raw = payload.get("path")
            expected_sha = str(payload.get("expected_sha256") or "")
            if not path_raw:
                blockers.append("Prior lineage path missing for %s." % key)
                continue
            prior_path = Path(str(path_raw)).expanduser().resolve()
            if not prior_path.exists():
                blockers.append("Prior lineage artifact missing: %s" % prior_path)
                continue
            actual_sha = sha256_file(prior_path)
            matched = actual_sha == expected_sha
            checks.setdefault("prior_lineage", {})[key] = {
                "path": str(prior_path),
                "expected": expected_sha,
                "actual": actual_sha,
                "matched": matched,
            }
            if not matched:
                blockers.append("Frozen prior lineage SHA256 changed for %s." % key)

    return {
        "blockers": blockers,
        "checks": checks,
        "receipt": receipt,
        "source_manifest": source_manifest,
        "snapshot_dir": snapshot_dir if snapshot_dir_raw else None,
    }


def compare_player_rows(preview: pd.DataFrame, db_rows: pd.DataFrame) -> Dict[str, Any]:
    blockers: List[str] = []

    p = preview[["fpl_player_id", "predicted_points"]].copy()
    p["fpl_player_id"] = pd.to_numeric(p["fpl_player_id"], errors="coerce").astype("Int64")
    p["predicted_points"] = pd.to_numeric(p["predicted_points"], errors="coerce")

    d = db_rows[["fpl_player_id", "predicted_points"]].copy()
    d["fpl_player_id"] = pd.to_numeric(d["fpl_player_id"], errors="coerce").astype("Int64")
    d["predicted_points"] = pd.to_numeric(d["predicted_points"], errors="coerce")

    if p["fpl_player_id"].isna().any() or d["fpl_player_id"].isna().any():
        blockers.append("Null/non-numeric fpl_player_id found.")
    if p["fpl_player_id"].duplicated().any():
        blockers.append("Duplicate preview fpl_player_id found.")
    if d["fpl_player_id"].duplicated().any():
        blockers.append("Duplicate DB fpl_player_id found.")

    merged = p.merge(d, on="fpl_player_id", how="outer", suffixes=("_preview", "_db"), indicator=True)
    missing_db = merged.loc[merged["_merge"] == "left_only", "fpl_player_id"].dropna().astype(int).tolist()
    extra_db = merged.loc[merged["_merge"] == "right_only", "fpl_player_id"].dropna().astype(int).tolist()

    both = merged[merged["_merge"] == "both"].copy()
    if not both.empty:
        both["abs_diff"] = (
            both["predicted_points_preview"] - both["predicted_points_db"]
        ).abs()
        mismatches = both[both["abs_diff"] > FLOAT_TOLERANCE]
        max_diff = float(both["abs_diff"].max())
    else:
        mismatches = both
        max_diff = None

    if missing_db:
        blockers.append("Preview players missing from DB: %s" % missing_db[:20])
    if extra_db:
        blockers.append("Unexpected DB players: %s" % extra_db[:20])
    if len(mismatches):
        blockers.append("Player predicted_points mismatch for %s row(s)." % len(mismatches))

    return {
        "blockers": blockers,
        "preview_rows": int(len(p)),
        "db_rows": int(len(d)),
        "matched_keys": int(len(both)),
        "value_mismatch_rows": int(len(mismatches)),
        "max_abs_predicted_points_diff": max_diff,
        "missing_db_ids": missing_db[:20],
        "extra_db_ids": extra_db[:20],
    }


def compare_match_rows(preview: pd.DataFrame, db_rows: pd.DataFrame) -> Dict[str, Any]:
    blockers: List[str] = []

    p = preview[
        [
            "fpl_fixture_id",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "predicted_result_label",
        ]
    ].copy()
    p["fpl_fixture_id"] = pd.to_numeric(p["fpl_fixture_id"], errors="coerce").astype("Int64")
    for col in ["home_win_probability", "draw_probability", "away_win_probability"]:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p["pred_result"] = p["predicted_result_label"].map(canonical_match_result_label)

    d = db_rows[
        [
            "fpl_fixture_id",
            "pred_home_win",
            "pred_draw",
            "pred_away_win",
            "pred_result",
        ]
    ].copy()
    d["fpl_fixture_id"] = pd.to_numeric(d["fpl_fixture_id"], errors="coerce").astype("Int64")
    for col in ["pred_home_win", "pred_draw", "pred_away_win"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["pred_result"] = d["pred_result"].astype(str)

    if p["fpl_fixture_id"].duplicated().any():
        blockers.append("Duplicate preview fpl_fixture_id found.")
    if d["fpl_fixture_id"].duplicated().any():
        blockers.append("Duplicate DB fpl_fixture_id found.")

    merged = p.merge(d, on="fpl_fixture_id", how="outer", indicator=True)
    missing_db = merged.loc[merged["_merge"] == "left_only", "fpl_fixture_id"].dropna().astype(int).tolist()
    extra_db = merged.loc[merged["_merge"] == "right_only", "fpl_fixture_id"].dropna().astype(int).tolist()
    both = merged[merged["_merge"] == "both"].copy()

    probability_mismatch_rows = 0
    label_mismatch_rows = 0
    max_diff = 0.0
    for row in both.itertuples(index=False):
        diffs = [
            abs(float(row.home_win_probability) - float(row.pred_home_win)),
            abs(float(row.draw_probability) - float(row.pred_draw)),
            abs(float(row.away_win_probability) - float(row.pred_away_win)),
        ]
        row_max = max(diffs)
        max_diff = max(max_diff, row_max)
        if row_max > FLOAT_TOLERANCE:
            probability_mismatch_rows += 1
    if not both.empty:
        # Pandas renames duplicate pred_result columns when merging.
        preview_label_col = "pred_result_x"
        db_label_col = "pred_result_y"
        if preview_label_col in both.columns and db_label_col in both.columns:
            label_mismatch_rows = int((both[preview_label_col] != both[db_label_col]).sum())

    if missing_db:
        blockers.append("Preview fixtures missing from DB: %s" % missing_db[:20])
    if extra_db:
        blockers.append("Unexpected DB fixtures: %s" % extra_db[:20])
    if probability_mismatch_rows:
        blockers.append("Match probability mismatch for %s row(s)." % probability_mismatch_rows)
    if label_mismatch_rows:
        blockers.append("Match result label mismatch for %s row(s)." % label_mismatch_rows)

    return {
        "blockers": blockers,
        "preview_rows": int(len(p)),
        "db_rows": int(len(d)),
        "matched_keys": int(len(both)),
        "probability_mismatch_rows": int(probability_mismatch_rows),
        "label_mismatch_rows": int(label_mismatch_rows),
        "max_abs_probability_diff": float(max_diff),
        "missing_db_ids": missing_db[:20],
        "extra_db_ids": extra_db[:20],
    }


def load_published_db_rows(
    db: Any,
    season: str,
    target_gw: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    player_rows = db.execute(
        text(
            """
            SELECT
              p.fpl_player_id,
              pr.predicted_points
            FROM predictions pr
            JOIN players p
              ON p.id = pr.player_id
             AND p.season = pr.season
            WHERE pr.season = :season
              AND pr.target_gw = :target_gw
              AND pr.model_name = :player_model
            ORDER BY p.fpl_player_id
            """
        ),
        {
            "season": season,
            "target_gw": target_gw,
            "player_model": EXPECTED_PLAYER_MODEL,
        },
    ).mappings().all()

    match_rows = db.execute(
        text(
            """
            SELECT
              f.fpl_fixture_id,
              mp.pred_home_win,
              mp.pred_draw,
              mp.pred_away_win,
              mp.pred_result
            FROM match_predictions mp
            JOIN fixtures f
              ON f.id = mp.fixture_id
             AND f.season = mp.season
            WHERE mp.season = :season
              AND f.gw = :target_gw
              AND mp.model_name = :match_model
            ORDER BY f.fpl_fixture_id
            """
        ),
        {
            "season": season,
            "target_gw": target_gw,
            "match_model": EXPECTED_MATCH_MODEL,
        },
    ).mappings().all()

    scope_counts = {
        "early_player_rows_other_seasons": int(
            db.execute(
                text(
                    "SELECT COUNT(*) FROM predictions "
                    "WHERE season <> :season AND model_name = :model"
                ),
                {"season": season, "model": EXPECTED_PLAYER_MODEL},
            ).scalar()
            or 0
        ),
        "early_match_rows_other_seasons": int(
            db.execute(
                text(
                    "SELECT COUNT(*) FROM match_predictions "
                    "WHERE season <> :season AND model_name = :model"
                ),
                {"season": season, "model": EXPECTED_MATCH_MODEL},
            ).scalar()
            or 0
        ),
        "early_player_rows_gw1": int(
            db.execute(
                text(
                    "SELECT COUNT(*) FROM predictions "
                    "WHERE season = :season AND target_gw = 1 AND model_name = :model"
                ),
                {"season": season, "model": EXPECTED_PLAYER_MODEL},
            ).scalar()
            or 0
        ),
        "early_match_rows_gw1": int(
            db.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM match_predictions mp
                    JOIN fixtures f
                      ON f.id = mp.fixture_id
                     AND f.season = mp.season
                    WHERE mp.season = :season
                      AND f.gw = 1
                      AND mp.model_name = :model
                    """
                ),
                {"season": season, "model": EXPECTED_MATCH_MODEL},
            ).scalar()
            or 0
        ),
    }

    return pd.DataFrame(player_rows), pd.DataFrame(match_rows), scope_counts


def main() -> None:
    from app.core.db import SessionLocal
    from app.core.season import get_current_season

    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    receipt_path = Path(args.receipt).expanduser().resolve()

    if get_current_season() != args.season:
        raise RuntimeError(
            "Runtime FPL_SEASON mismatch: get_current_season()=%s requested=%s"
            % (get_current_season(), args.season)
        )

    evidence = validate_receipt_and_snapshot(
        run_dir=run_dir,
        receipt_path=receipt_path,
        season=args.season,
        target_gw=args.target_gw,
    )
    blockers: List[str] = list(evidence["blockers"])

    player_preview_path = run_dir / "player_predictions_preview.csv"
    match_preview_path = run_dir / "match_predictions_preview.csv"
    players = pd.read_csv(player_preview_path, low_memory=False)
    matches = pd.read_csv(match_preview_path, low_memory=False)

    if not players.empty:
        player_models = set(players["model_name"].dropna().astype(str))
        if player_models != {EXPECTED_PLAYER_MODEL}:
            blockers.append("Unexpected player model in preview: %s" % sorted(player_models))
    if not matches.empty:
        match_models = set(matches["model_name"].dropna().astype(str))
        if match_models != {EXPECTED_MATCH_MODEL}:
            blockers.append("Unexpected match model in preview: %s" % sorted(match_models))

    db = SessionLocal()
    try:
        db_players, db_matches, scope_counts = load_published_db_rows(
            db,
            season=args.season,
            target_gw=args.target_gw,
        )
    finally:
        db.close()

    player_compare = compare_player_rows(players, db_players)
    match_compare = compare_match_rows(matches, db_matches)
    blockers.extend(player_compare["blockers"])
    blockers.extend(match_compare["blockers"])

    receipt = evidence.get("receipt") or {}
    publish_result = dict(receipt.get("publish_result") or {})
    if int(publish_result.get("player_rows_published") or -1) != len(players):
        blockers.append("Receipt player_rows_published does not match source preview.")
    if int(publish_result.get("match_rows_published") or -1) != len(matches):
        blockers.append("Receipt match_rows_published does not match source preview.")
    if int(publish_result.get("player_rows_verified_after") or -1) != len(db_players):
        blockers.append("Receipt player_rows_verified_after does not match current DB.")
    if int(publish_result.get("match_rows_verified_after") or -1) != len(db_matches):
        blockers.append("Receipt match_rows_verified_after does not match current DB.")

    if scope_counts["early_player_rows_other_seasons"] != 0:
        blockers.append("Early-season player model rows unexpectedly exist in another season.")
    if scope_counts["early_match_rows_other_seasons"] != 0:
        blockers.append("Early-season match model rows unexpectedly exist in another season.")
    if scope_counts["early_player_rows_gw1"] != 0:
        blockers.append("Early-season player model unexpectedly wrote GW1 rows.")
    if scope_counts["early_match_rows_gw1"] != 0:
        blockers.append("Early-season match model unexpectedly wrote GW1 rows.")

    status = "PASS_POST_PUBLISH" if not blockers else "BLOCKED"

    print("=== Early Season Post-Publish Verification ===")
    print("status:", status)
    print("verifier_version:", VERIFIER_VERSION)
    print("season:", args.season)
    print("target_gw:", args.target_gw)
    print("source_run_id:", (evidence.get("source_manifest") or {}).get("run_id"))
    print("player_preview_rows:", len(players))
    print("player_db_rows:", len(db_players))
    print("player_value_mismatch_rows:", player_compare["value_mismatch_rows"])
    print("player_max_abs_diff:", player_compare["max_abs_predicted_points_diff"])
    print("match_preview_rows:", len(matches))
    print("match_db_rows:", len(db_matches))
    print("match_probability_mismatch_rows:", match_compare["probability_mismatch_rows"])
    print("match_label_mismatch_rows:", match_compare["label_mismatch_rows"])
    print("match_max_abs_probability_diff:", match_compare["max_abs_probability_diff"])
    print("early_player_rows_other_seasons:", scope_counts["early_player_rows_other_seasons"])
    print("early_match_rows_other_seasons:", scope_counts["early_match_rows_other_seasons"])
    print("early_player_rows_gw1:", scope_counts["early_player_rows_gw1"])
    print("early_match_rows_gw1:", scope_counts["early_match_rows_gw1"])
    print("snapshot_hash_verified:", not any(
        "snapshot" in b.lower() and "sha" in b.lower() for b in blockers
    ))
    print("prior_lineage_hash_verified:", not any(
        "prior" in b.lower() and "sha" in b.lower() for b in blockers
    ))
    print("database_prediction_write_verified:", status == "PASS_POST_PUBLISH")
    print("final_deadline_freeze:", False)

    if blockers:
        print("blockers:")
        for blocker in blockers:
            print("-", blocker)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
