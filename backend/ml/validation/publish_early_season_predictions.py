from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import text

GATE_VERSION = "early_season_publish_gate_v0_1"
EXPECTED_PIPELINE_VERSION = "early_season_prediction_pipeline_v0_1"
EXPECTED_PLAYER_MODEL = "early_season_blend_player_v0"
EXPECTED_MATCH_MODEL = "early_season_blend_match_v0"
EXPECTED_SCORELINE_MODEL = "early_season_blend_scoreline_v0"



def canonical_match_result_label(value: Any) -> str:
    text_value = str(value).strip().lower()
    mapping = {
        "home_win": "H",
        "draw": "D",
        "away_win": "A",
        "h": "H",
        "d": "D",
        "a": "A",
    }
    if text_value not in mapping:
        raise RuntimeError("Unsupported predicted_result_label=%r." % value)
    return mapping[text_value]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_snapshot_root(season: str, target_gw: int) -> Path:
    return (
        repo_root().parent
        / "private-planning"
        / "gw-pre"
        / season
        / ("gw%02d" % target_gw)
        / "published"
    )


def default_receipt_root(season: str, target_gw: int) -> Path:
    return (
        repo_root().parent
        / "private-planning"
        / "gw-pre"
        / season
        / ("gw%02d" % target_gw)
        / "publish-receipts"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and optionally publish one PASS_PREVIEW early-season run. "
            "Default is dry-run. --publish first creates an immutable published snapshot, "
            "then writes player and match predictions transactionally."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--target-gw", type=int, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Explicitly allow replacing existing predictions for the same "
            "season/target_gw/model names. Default refuses any overwrite."
        ),
    )
    parser.add_argument(
        "--snapshot-root",
        default="",
        help="Optional published snapshot root. Defaults outside the repo under private-planning/gw-pre.",
    )
    parser.add_argument(
        "--receipt-root",
        default="",
        help="Optional publish receipt root. Defaults outside the repo under private-planning/gw-pre.",
    )
    return parser.parse_args()


def required_file(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    if not path.exists() or not path.is_file():
        raise RuntimeError("Required preview artifact is missing: %s" % path)
    return path


def single_value(df: pd.DataFrame, column: str, label: str) -> str:
    if column not in df.columns:
        raise RuntimeError("%s is missing required column %s." % (label, column))
    values = set(df[column].dropna().astype(str).tolist())
    if len(values) != 1:
        raise RuntimeError("%s column %s must have exactly one value; got %s." % (label, column, sorted(values)))
    return next(iter(values))


def single_int(df: pd.DataFrame, column: str, label: str) -> int:
    if column not in df.columns:
        raise RuntimeError("%s is missing required column %s." % (label, column))
    values = set(pd.to_numeric(df[column], errors="coerce").dropna().astype(int).tolist())
    if len(values) != 1:
        raise RuntimeError("%s column %s must have exactly one integer value; got %s." % (label, column, sorted(values)))
    return next(iter(values))


def validate_preview_artifacts(
    run_dir: Path,
    season: str,
    target_gw: int,
) -> Dict[str, Any]:
    manifest_path = required_file(run_dir, "run_manifest.json")
    player_path = required_file(run_dir, "player_predictions_preview.csv")
    match_path = required_file(run_dir, "match_predictions_preview.csv")
    scoreline_path = required_file(run_dir, "scoreline_preview.csv")
    bootstrap_path = required_file(run_dir, "bootstrap_snapshot.json")
    summary_path = required_file(run_dir, "summary.md")

    manifest = load_json(manifest_path)
    blockers: List[str] = []
    warnings: List[str] = []

    if manifest.get("pipeline_version") != EXPECTED_PIPELINE_VERSION:
        blockers.append(
            "Expected pipeline_version=%s; got %s."
            % (EXPECTED_PIPELINE_VERSION, manifest.get("pipeline_version"))
        )
    if manifest.get("status") != "PASS_PREVIEW":
        blockers.append("Preview status must be PASS_PREVIEW.")
    if str(manifest.get("season")) != season:
        blockers.append("Preview season does not match requested season.")
    if int(manifest.get("target_gw") or -1) != int(target_gw):
        blockers.append("Preview target_gw does not match requested target_gw.")
    if manifest.get("prediction_mode") != "early_season_blend":
        blockers.append("Preview prediction_mode must be early_season_blend.")
    if manifest.get("database_prediction_write") is not False:
        blockers.append("Source preview must record database_prediction_write=false.")
    if manifest.get("preview_only") is not True:
        blockers.append("Source preview must record preview_only=true.")
    if manifest.get("blockers"):
        blockers.append("Source preview manifest contains blockers.")

    scoreline_alignment = dict(manifest.get("scoreline_alignment") or {})
    if "label_mismatch_rows" not in scoreline_alignment:
        blockers.append("v0.1 scoreline alignment diagnostics are missing from manifest.")
    elif int(scoreline_alignment.get("label_mismatch_rows") or 0) > 0:
        warnings.append(
            "1X2 and scoreline result labels disagree for %s fixture(s)."
            % int(scoreline_alignment.get("label_mismatch_rows") or 0)
        )

    players = pd.read_csv(player_path, low_memory=False)
    matches = pd.read_csv(match_path, low_memory=False)
    scorelines = pd.read_csv(scoreline_path, low_memory=False)

    if players.empty:
        blockers.append("Player preview is empty.")
    if matches.empty:
        blockers.append("Match preview is empty.")
    if scorelines.empty:
        blockers.append("Scoreline preview is empty.")

    if not players.empty:
        if single_value(players, "target_season", "player preview") != season:
            blockers.append("Player preview target_season mismatch.")
        if single_int(players, "target_gw", "player preview") != target_gw:
            blockers.append("Player preview target_gw mismatch.")
        if single_value(players, "model_name", "player preview") != EXPECTED_PLAYER_MODEL:
            blockers.append("Unexpected player model name.")
        if "fpl_player_id" not in players.columns or players["fpl_player_id"].isna().any():
            blockers.append("Player preview has null fpl_player_id values.")
        elif int(players["fpl_player_id"].duplicated().sum()) > 0:
            blockers.append("Player preview has duplicate fpl_player_id rows.")
        if "predicted_points" not in players.columns:
            blockers.append("Player preview is missing predicted_points.")
        else:
            numeric_points = pd.to_numeric(players["predicted_points"], errors="coerce")
            if numeric_points.isna().any():
                blockers.append("Player preview has null/non-numeric predicted_points.")
            elif not numeric_points.map(math.isfinite).all():
                blockers.append("Player preview has non-finite predicted_points.")

    if not matches.empty:
        if single_value(matches, "target_season", "match preview") != season:
            blockers.append("Match preview target_season mismatch.")
        if single_int(matches, "target_gw", "match preview") != target_gw:
            blockers.append("Match preview target_gw mismatch.")
        if single_value(matches, "model_name", "match preview") != EXPECTED_MATCH_MODEL:
            blockers.append("Unexpected match model name.")
        if "fpl_fixture_id" not in matches.columns or matches["fpl_fixture_id"].isna().any():
            blockers.append("Match preview has null fpl_fixture_id values.")
        elif int(matches["fpl_fixture_id"].duplicated().sum()) > 0:
            blockers.append("Match preview has duplicate fpl_fixture_id rows.")
        probability_cols = [
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ]
        missing_prob = [c for c in probability_cols if c not in matches.columns]
        if missing_prob:
            blockers.append("Match preview is missing probability columns: %s" % missing_prob)
        else:
            probs = matches[probability_cols].apply(pd.to_numeric, errors="coerce")
            if probs.isna().any().any():
                blockers.append("Match preview has null/non-numeric probabilities.")
            elif float(probs.sum(axis=1).sub(1.0).abs().max()) > 1e-5:
                blockers.append("Match probability sums are invalid.")

    if not scorelines.empty:
        if single_value(scorelines, "target_season", "scoreline preview") != season:
            blockers.append("Scoreline preview target_season mismatch.")
        if single_int(scorelines, "target_gw", "scoreline preview") != target_gw:
            blockers.append("Scoreline preview target_gw mismatch.")
        if single_value(scorelines, "model_name", "scoreline preview") != EXPECTED_SCORELINE_MODEL:
            blockers.append("Unexpected scoreline model name.")
        if len(scorelines) != len(matches):
            blockers.append("Scoreline row count does not equal match row count.")

    expected_player_rows = int(manifest.get("current_player_pool_rows") or -1)
    if expected_player_rows >= 0 and len(players) != expected_player_rows:
        blockers.append(
            "Player preview row count %s does not equal manifest current_player_pool_rows=%s."
            % (len(players), expected_player_rows)
        )
    expected_match_rows = int(manifest.get("target_fixture_rows") or -1)
    if expected_match_rows >= 0 and len(matches) != expected_match_rows:
        blockers.append(
            "Match preview row count %s does not equal manifest target_fixture_rows=%s."
            % (len(matches), expected_match_rows)
        )

    prior_artifacts = dict(manifest.get("prior_artifacts") or {})
    prior_hash_checks: Dict[str, Any] = {}
    for path_key, hash_key in [
        ("player_preview", "player_preview_sha256"),
        ("effective_match_features", "effective_match_features_sha256"),
    ]:
        raw_path = prior_artifacts.get(path_key)
        expected_hash = prior_artifacts.get(hash_key)
        if not raw_path or not expected_hash:
            blockers.append("Prior artifact lineage is missing %s/%s." % (path_key, hash_key))
            continue
        prior_path = Path(str(raw_path)).expanduser().resolve()
        if not prior_path.exists():
            blockers.append("Prior artifact no longer exists: %s" % prior_path)
            continue
        actual_hash = sha256_file(prior_path)
        prior_hash_checks[path_key] = {
            "path": str(prior_path),
            "expected_sha256": str(expected_hash),
            "actual_sha256": actual_hash,
            "matched": actual_hash == str(expected_hash),
        }
        if actual_hash != str(expected_hash):
            blockers.append("Prior artifact SHA256 changed: %s" % prior_path)

    hashes = {
        "run_manifest.json": sha256_file(manifest_path),
        "player_predictions_preview.csv": sha256_file(player_path),
        "match_predictions_preview.csv": sha256_file(match_path),
        "scoreline_preview.csv": sha256_file(scoreline_path),
        "bootstrap_snapshot.json": sha256_file(bootstrap_path),
        "summary.md": sha256_file(summary_path),
    }

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "players": players,
        "matches": matches,
        "scorelines": scorelines,
        "artifact_paths": {
            "run_manifest.json": manifest_path,
            "player_predictions_preview.csv": player_path,
            "match_predictions_preview.csv": match_path,
            "scoreline_preview.csv": scoreline_path,
            "bootstrap_snapshot.json": bootstrap_path,
            "summary.md": summary_path,
        },
        "artifact_sha256": hashes,
        "prior_hash_checks": prior_hash_checks,
        "blockers": blockers,
        "warnings": warnings,
    }


def validate_database_mapping(
    db: Any,
    season: str,
    target_gw: int,
    players: pd.DataFrame,
    matches: pd.DataFrame,
) -> Dict[str, Any]:
    blockers: List[str] = []

    canonical_players = db.execute(
        text(
            "SELECT id, fpl_player_id FROM players "
            "WHERE season = :season ORDER BY fpl_player_id"
        ),
        {"season": season},
    ).all()
    player_map = {int(row.fpl_player_id): int(row.id) for row in canonical_players}
    preview_player_ids = set(pd.to_numeric(players["fpl_player_id"], errors="coerce").dropna().astype(int).tolist())
    canonical_player_ids = set(player_map.keys())
    if preview_player_ids != canonical_player_ids:
        blockers.append(
            "Preview player universe no longer matches canonical season player pool. "
            "missing_from_preview=%s extra_in_preview=%s"
            % (
                sorted(canonical_player_ids - preview_player_ids)[:20],
                sorted(preview_player_ids - canonical_player_ids)[:20],
            )
        )

    player_internal_mismatches = 0
    if "player_id" in players.columns:
        for row in players[["fpl_player_id", "player_id"]].itertuples(index=False):
            fpl_id = int(row.fpl_player_id)
            if fpl_id in player_map and int(row.player_id) != player_map[fpl_id]:
                player_internal_mismatches += 1
        if player_internal_mismatches:
            blockers.append("Preview player_id values are stale for %s rows." % player_internal_mismatches)

    canonical_fixtures = db.execute(
        text(
            "SELECT id, fpl_fixture_id FROM fixtures "
            "WHERE season = :season AND gw = :target_gw ORDER BY fpl_fixture_id"
        ),
        {"season": season, "target_gw": target_gw},
    ).all()
    fixture_map = {int(row.fpl_fixture_id): int(row.id) for row in canonical_fixtures}
    preview_fixture_ids = set(pd.to_numeric(matches["fpl_fixture_id"], errors="coerce").dropna().astype(int).tolist())
    canonical_fixture_ids = set(fixture_map.keys())
    if preview_fixture_ids != canonical_fixture_ids:
        blockers.append(
            "Preview fixture universe no longer matches canonical target-GW fixtures. "
            "missing_from_preview=%s extra_in_preview=%s"
            % (
                sorted(canonical_fixture_ids - preview_fixture_ids)[:20],
                sorted(preview_fixture_ids - canonical_fixture_ids)[:20],
            )
        )

    fixture_internal_mismatches = 0
    if "fixture_id" in matches.columns:
        for row in matches[["fpl_fixture_id", "fixture_id"]].itertuples(index=False):
            fpl_id = int(row.fpl_fixture_id)
            if fpl_id in fixture_map and int(row.fixture_id) != fixture_map[fpl_id]:
                fixture_internal_mismatches += 1
        if fixture_internal_mismatches:
            blockers.append("Preview fixture_id values are stale for %s rows." % fixture_internal_mismatches)

    player_model = single_value(players, "model_name", "player preview")
    match_model = single_value(matches, "model_name", "match preview")

    existing_player_rows = int(
        db.execute(
            text(
                "SELECT COUNT(*) FROM predictions "
                "WHERE season=:season AND target_gw=:target_gw AND model_name=:model"
            ),
            {"season": season, "target_gw": target_gw, "model": player_model},
        ).scalar()
        or 0
    )
    existing_match_rows = int(
        db.execute(
            text(
                "SELECT COUNT(*) FROM match_predictions mp "
                "JOIN fixtures f ON f.id = mp.fixture_id "
                "WHERE mp.season=:season AND f.season=:season "
                "AND f.gw=:target_gw AND mp.model_name=:model"
            ),
            {"season": season, "target_gw": target_gw, "model": match_model},
        ).scalar()
        or 0
    )

    return {
        "player_map": player_map,
        "fixture_map": fixture_map,
        "canonical_player_rows": len(player_map),
        "canonical_target_fixture_rows": len(fixture_map),
        "player_internal_id_mismatches": player_internal_mismatches,
        "fixture_internal_id_mismatches": fixture_internal_mismatches,
        "existing_player_prediction_rows": existing_player_rows,
        "existing_match_prediction_rows": existing_match_rows,
        "blockers": blockers,
    }


def create_immutable_snapshot(
    validated: Mapping[str, Any],
    snapshot_root: Path,
    season: str,
    target_gw: int,
) -> Tuple[Path, Dict[str, Any]]:
    run_id = str(validated["manifest"]["run_id"])
    snapshot_dir = snapshot_root / run_id
    if snapshot_dir.exists():
        raise RuntimeError(
            "Published snapshot already exists and will not be overwritten: %s" % snapshot_dir
        )
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    copied_hashes: Dict[str, str] = {}
    for name, source in validated["artifact_paths"].items():
        destination = snapshot_dir / name
        shutil.copy2(str(source), str(destination))
        copied_hashes[name] = sha256_file(destination)
        if copied_hashes[name] != validated["artifact_sha256"][name]:
            raise RuntimeError("Snapshot copy hash mismatch for %s." % name)

    manifest: Dict[str, Any] = {
        "snapshot_contract": "immutable_pre_deadline_model_publish_snapshot_v1",
        "gate_version": GATE_VERSION,
        "created_at": utc_now(),
        "season": season,
        "target_gw": target_gw,
        "source_run_id": run_id,
        "source_run_dir": str(validated["run_dir"]),
        "source_pipeline_version": validated["manifest"].get("pipeline_version"),
        "prediction_mode": validated["manifest"].get("prediction_mode"),
        "weights": validated["manifest"].get("weights"),
        "artifact_sha256": copied_hashes,
        "prior_hash_checks": validated["prior_hash_checks"],
        "player_rows": int(len(validated["players"])),
        "match_rows": int(len(validated["matches"])),
        "scoreline_rows": int(len(validated["scorelines"])),
        "player_model_name": EXPECTED_PLAYER_MODEL,
        "match_model_name": EXPECTED_MATCH_MODEL,
        "scoreline_model_name": EXPECTED_SCORELINE_MODEL,
        "database_write_occurred_during_snapshot_creation": False,
        "overwrite_allowed": False,
        "final_deadline_freeze": False,
        "notes": [
            "This snapshot is created before canonical prediction DB publication.",
            "It is a published PRE model snapshot, not the final deadline freeze.",
            "Scoreline predictions remain artifact-only because no canonical scoreline table exists.",
        ],
    }
    snapshot_manifest_path = snapshot_dir / "snapshot_manifest.json"
    write_json(snapshot_manifest_path, manifest)
    manifest["snapshot_manifest_sha256"] = sha256_file(snapshot_manifest_path)
    return snapshot_dir, manifest


def publish_transaction(
    db: Any,
    season: str,
    target_gw: int,
    validated: Mapping[str, Any],
    mapping: Mapping[str, Any],
    replace_existing: bool,
) -> Dict[str, Any]:
    players: pd.DataFrame = validated["players"]
    matches: pd.DataFrame = validated["matches"]
    player_map: Dict[int, int] = dict(mapping["player_map"])
    fixture_map: Dict[int, int] = dict(mapping["fixture_map"])

    player_model = single_value(players, "model_name", "player preview")
    match_model = single_value(matches, "model_name", "match preview")

    existing_player_rows = int(mapping["existing_player_prediction_rows"])
    existing_match_rows = int(mapping["existing_match_prediction_rows"])
    if (existing_player_rows or existing_match_rows) and not replace_existing:
        raise RuntimeError(
            "Canonical target/model rows already exist. "
            "Use --replace-existing only for an explicit republish. "
            "existing_player_rows=%s existing_match_rows=%s"
            % (existing_player_rows, existing_match_rows)
        )

    try:
        if replace_existing:
            db.execute(
                text(
                    "DELETE FROM predictions "
                    "WHERE season=:season AND target_gw=:target_gw AND model_name=:model"
                ),
                {"season": season, "target_gw": target_gw, "model": player_model},
            )
            db.execute(
                text(
                    "DELETE FROM match_predictions mp USING fixtures f "
                    "WHERE mp.fixture_id=f.id AND mp.season=:season AND f.season=:season "
                    "AND f.gw=:target_gw AND mp.model_name=:model"
                ),
                {"season": season, "target_gw": target_gw, "model": match_model},
            )

        player_payload = []
        for row in players.itertuples(index=False):
            fpl_player_id = int(row.fpl_player_id)
            player_payload.append(
                {
                    "season": season,
                    "player_id": player_map[fpl_player_id],
                    "target_gw": target_gw,
                    "model_name": player_model,
                    "predicted_points": float(row.predicted_points),
                }
            )
        if player_payload:
            db.execute(
                text(
                    "INSERT INTO predictions "
                    "(season, player_id, target_gw, model_name, predicted_points) "
                    "VALUES (:season, :player_id, :target_gw, :model_name, :predicted_points)"
                ),
                player_payload,
            )

        match_payload = []
        for row in matches.itertuples(index=False):
            fpl_fixture_id = int(row.fpl_fixture_id)
            match_payload.append(
                {
                    "season": season,
                    "fixture_id": fixture_map[fpl_fixture_id],
                    "model_name": match_model,
                    "pred_home_win": float(row.home_win_probability),
                    "pred_draw": float(row.draw_probability),
                    "pred_away_win": float(row.away_win_probability),
                    "pred_result": canonical_match_result_label(row.predicted_result_label),
                }
            )
        if match_payload:
            db.execute(
                text(
                    "INSERT INTO match_predictions "
                    "(season, fixture_id, model_name, pred_home_win, pred_draw, pred_away_win, pred_result) "
                    "VALUES (:season, :fixture_id, :model_name, :pred_home_win, :pred_draw, :pred_away_win, :pred_result)"
                ),
                match_payload,
            )

        player_count = int(
            db.execute(
                text(
                    "SELECT COUNT(*) FROM predictions "
                    "WHERE season=:season AND target_gw=:target_gw AND model_name=:model"
                ),
                {"season": season, "target_gw": target_gw, "model": player_model},
            ).scalar()
            or 0
        )
        match_count = int(
            db.execute(
                text(
                    "SELECT COUNT(*) FROM match_predictions mp "
                    "JOIN fixtures f ON f.id=mp.fixture_id "
                    "WHERE mp.season=:season AND f.season=:season "
                    "AND f.gw=:target_gw AND mp.model_name=:model"
                ),
                {"season": season, "target_gw": target_gw, "model": match_model},
            ).scalar()
            or 0
        )

        if player_count != len(players):
            raise RuntimeError(
                "Post-publish player row count mismatch: expected=%s actual=%s"
                % (len(players), player_count)
            )
        if match_count != len(matches):
            raise RuntimeError(
                "Post-publish match row count mismatch: expected=%s actual=%s"
                % (len(matches), match_count)
            )

        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "player_model_name": player_model,
        "match_model_name": match_model,
        "existing_player_rows_before": existing_player_rows,
        "existing_match_rows_before": existing_match_rows,
        "player_rows_published": len(players),
        "match_rows_published": len(matches),
        "player_rows_verified_after": player_count,
        "match_rows_verified_after": match_count,
        "replace_existing": bool(replace_existing),
    }


def main() -> None:
    from app.core.db import SessionLocal
    from app.core.season import get_current_season

    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise RuntimeError("run_dir does not exist: %s" % run_dir)

    if get_current_season() != args.season:
        raise RuntimeError(
            "Runtime FPL_SEASON mismatch: get_current_season()=%s requested=%s"
            % (get_current_season(), args.season)
        )

    validated = validate_preview_artifacts(run_dir, args.season, args.target_gw)
    blockers = list(validated["blockers"])
    warnings = list(validated["warnings"])

    db = SessionLocal()
    try:
        mapping = validate_database_mapping(
            db,
            args.season,
            args.target_gw,
            validated["players"],
            validated["matches"],
        )
        blockers.extend(mapping["blockers"])

        if (
            (mapping["existing_player_prediction_rows"] > 0 or mapping["existing_match_prediction_rows"] > 0)
            and not args.replace_existing
        ):
            blockers.append(
                "Canonical rows already exist for the target/model names; explicit --replace-existing is required."
            )

        status = "READY_TO_PUBLISH" if not blockers else "BLOCKED"
        print("=== Early Season Publish Gate ===")
        print("status:", status)
        print("gate_version:", GATE_VERSION)
        print("source_run_id:", validated["manifest"].get("run_id"))
        print("season:", args.season)
        print("target_gw:", args.target_gw)
        print("player_rows:", len(validated["players"]))
        print("match_rows:", len(validated["matches"]))
        print("scoreline_rows:", len(validated["scorelines"]))
        print("canonical_player_rows:", mapping["canonical_player_rows"])
        print("canonical_target_fixture_rows:", mapping["canonical_target_fixture_rows"])
        print("existing_player_prediction_rows:", mapping["existing_player_prediction_rows"])
        print("existing_match_prediction_rows:", mapping["existing_match_prediction_rows"])
        print("publish_requested:", bool(args.publish))
        print("replace_existing:", bool(args.replace_existing))
        if warnings:
            print("warnings:")
            for warning in warnings:
                print("-", warning)
        if blockers:
            print("blockers:")
            for blocker in blockers:
                print("-", blocker)
            raise SystemExit(1)

        if not args.publish:
            print("database_prediction_write: False")
            print("snapshot_created: False")
            return

        snapshot_root = (
            Path(args.snapshot_root).expanduser().resolve()
            if args.snapshot_root
            else default_snapshot_root(args.season, args.target_gw)
        )
        snapshot_root.mkdir(parents=True, exist_ok=True)
        snapshot_dir, snapshot_manifest = create_immutable_snapshot(
            validated,
            snapshot_root,
            args.season,
            args.target_gw,
        )

        publish_result = publish_transaction(
            db,
            args.season,
            args.target_gw,
            validated,
            mapping,
            replace_existing=bool(args.replace_existing),
        )

        receipt_root = (
            Path(args.receipt_root).expanduser().resolve()
            if args.receipt_root
            else default_receipt_root(args.season, args.target_gw)
        )
        receipt_root.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_root / (
            "%s_%s.json" % (validated["manifest"]["run_id"], utc_stamp())
        )
        receipt = {
            "receipt_contract": "early_season_prediction_publish_receipt_v1",
            "gate_version": GATE_VERSION,
            "published_at": utc_now(),
            "season": args.season,
            "target_gw": args.target_gw,
            "source_run_id": validated["manifest"]["run_id"],
            "source_run_dir": str(run_dir),
            "snapshot_dir": str(snapshot_dir),
            "snapshot_manifest_sha256": sha256_file(snapshot_dir / "snapshot_manifest.json"),
            "publish_result": publish_result,
            "scoreline_artifact_only": True,
            "database_prediction_write": True,
            "final_deadline_freeze": False,
        }
        write_json(receipt_path, receipt)

        print("snapshot_created:", snapshot_dir)
        print("player_rows_published:", publish_result["player_rows_published"])
        print("match_rows_published:", publish_result["match_rows_published"])
        print("publish_receipt:", receipt_path)
        print("database_prediction_write: True")
        print("final_deadline_freeze: False")
    finally:
        db.close()


if __name__ == "__main__":
    main()
