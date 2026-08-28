from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FREEZER_VERSION = "weekly_deadline_freeze_v0_1"
LEGAL_FORMATIONS = {"3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-2-3", "5-3-2", "5-4-1"}


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


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("Expected JSON object: %s" % path)
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )


def json_contains_string(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, Mapping):
        return any(json_contains_string(v, needle) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(json_contains_string(v, needle) for v in value)
    return False


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        raise RuntimeError("Refusing to overwrite existing freeze content: %s" % dst)
    shutil.copytree(src, dst)


def find_squad(value: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(value, list):
        if len(value) == 15 and all(isinstance(item, dict) for item in value):
            keys = set()
            for item in value:
                keys.update(item.keys())
            if (
                {"position"} <= keys
                and (
                    "player_id" in keys
                    or "fpl_player_id" in keys
                    or "fantasy_player_id" in keys
                )
            ):
                return [dict(item) for item in value]
        for item in value:
            found = find_squad(item)
            if found is not None:
                return found
    elif isinstance(value, Mapping):
        preferred = value.get("squad")
        if isinstance(preferred, list):
            found = find_squad(preferred)
            if found is not None:
                return found
        for item in value.values():
            found = find_squad(item)
            if found is not None:
                return found
    return None


def get_player_id(player: Mapping[str, Any]) -> Any:
    for key in ("player_id", "fpl_player_id", "fantasy_player_id"):
        if key in player and player[key] is not None:
            return player[key]
    return None


def get_team_key(player: Mapping[str, Any]) -> Any:
    for key in ("team_id", "club_id", "fpl_team_id", "team_name", "team_short_name"):
        if key in player and player[key] is not None:
            return player[key]
    return None


def validate_squad(squad: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(squad) != 15:
        raise RuntimeError("Frozen Model Team must contain exactly 15 players.")

    ids = [get_player_id(player) for player in squad]
    if any(player_id is None for player_id in ids):
        raise RuntimeError("At least one squad row has no player identifier.")
    if len(set(str(v) for v in ids)) != 15:
        raise RuntimeError("Frozen Model Team contains duplicate player IDs.")

    counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for player in squad:
        pos = str(player.get("position") or "")
        if pos not in counts:
            raise RuntimeError("Invalid squad position: %s" % pos)
        counts[pos] += 1

    expected = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    if counts != expected:
        raise RuntimeError("Squad position quotas invalid: %s" % counts)

    club_counts: Dict[str, int] = {}
    for player in squad:
        club = get_team_key(player)
        if club is None:
            raise RuntimeError("At least one squad row has no team/club identifier.")
        key = str(club)
        club_counts[key] = club_counts.get(key, 0) + 1
    max_club = max(club_counts.values()) if club_counts else 0
    if max_club > 3:
        raise RuntimeError("Squad violates max-3-per-club rule.")

    return {
        "player_count": 15,
        "unique_player_ids": 15,
        "position_counts": counts,
        "max_players_same_club": max_club,
    }


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise RuntimeError("%s missing: %s" % (label, path))
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise RuntimeError("%s missing: %s" % (label, path))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an explicit immutable weekly FPL deadline freeze from an already-verified PRE candidate."
    )
    parser.add_argument("--planning-root", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--target-gw", required=True, type=int)
    parser.add_argument("--prediction-run", required=True)
    parser.add_argument("--transfer-run", required=True)
    parser.add_argument("--candidate-package", required=True)
    parser.add_argument("--publish-receipt", required=True)
    parser.add_argument("--confirm-final-freeze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_final_freeze:
        raise RuntimeError("Refusing to create immutable freeze without --confirm-final-freeze.")

    planning_root = Path(args.planning_root).expanduser().resolve()
    prediction_run = require_dir(Path(args.prediction_run).expanduser().resolve(), "prediction run")
    transfer_run = require_dir(Path(args.transfer_run).expanduser().resolve(), "transfer run")
    candidate_package = require_dir(Path(args.candidate_package).expanduser().resolve(), "candidate package")
    receipt_path = require_file(Path(args.publish_receipt).expanduser().resolve(), "publish receipt")

    prediction_manifest_path = require_file(prediction_run / "run_manifest.json", "prediction manifest")
    player_csv = require_file(prediction_run / "player_predictions_preview.csv", "player predictions")
    match_csv = require_file(prediction_run / "match_predictions_preview.csv", "match predictions")
    scoreline_csv = require_file(prediction_run / "scoreline_preview.csv", "scoreline predictions")

    transfer_manifest_path = require_file(transfer_run / "run_manifest.json", "transfer manifest")
    transfer_decision_path = require_file(
        transfer_run / "transfer_decision_preview.json", "transfer decision"
    )
    transfer_state_path = require_file(
        transfer_run / "next_gameweek_transfer_state_preview.json", "next-GW transfer state"
    )

    candidate_manifest_path = require_file(candidate_package / "run_manifest.json", "candidate manifest")
    lineup_path = require_file(candidate_package / "weekly_lineup_preview.json", "lineup preview")
    require_file(candidate_package / "summary.md", "candidate summary")

    prediction_manifest = read_json(prediction_manifest_path)
    transfer_manifest = read_json(transfer_manifest_path)
    transfer_decision = read_json(transfer_decision_path)
    transfer_state = read_json(transfer_state_path)
    candidate_manifest = read_json(candidate_manifest_path)
    lineup = read_json(lineup_path)
    receipt = read_json(receipt_path)

    if str(prediction_manifest.get("season")) != str(args.season):
        raise RuntimeError("Prediction manifest season mismatch.")
    if int(prediction_manifest.get("target_gw") or -1) != int(args.target_gw):
        raise RuntimeError("Prediction manifest target_gw mismatch.")
    if str(prediction_manifest.get("status")) != "PASS_PREVIEW":
        raise RuntimeError("Prediction manifest is not PASS_PREVIEW.")

    player_rows = csv_row_count(player_csv)
    match_rows = csv_row_count(match_csv)
    scoreline_rows = csv_row_count(scoreline_csv)
    if player_rows <= 0:
        raise RuntimeError("No player predictions to freeze.")
    if match_rows != 10 or scoreline_rows != 10:
        raise RuntimeError(
            "Expected 10 match rows and 10 scoreline rows; got %s and %s."
            % (match_rows, scoreline_rows)
        )

    run_id = prediction_run.name
    if not json_contains_string(receipt, run_id):
        raise RuntimeError("Publish receipt does not reference the selected prediction run.")
    if not json_contains_string(transfer_manifest, run_id):
        raise RuntimeError("Transfer run does not reference the selected prediction run.")

    candidate_inputs = candidate_manifest.get("inputs") or {}
    if str(Path(str(candidate_inputs.get("prediction_run") or "")).name) != run_id:
        raise RuntimeError("Candidate package prediction lineage mismatch.")
    if str(Path(str(candidate_inputs.get("transfer_run") or "")).name) != transfer_run.name:
        raise RuntimeError("Candidate package transfer lineage mismatch.")
    if bool(candidate_manifest.get("final_deadline_freeze")):
        raise RuntimeError("Source candidate is already marked as a final freeze.")

    # weekly_lineup_preview.json does NOT persist the terminal-only
    # "status: PASS_PREVIEW" line. Validate the real artifact contract instead.
    if str(lineup.get("artifact_type") or "") != "weekly_lineup_preview":
        raise RuntimeError("Lineup source artifact_type is not weekly_lineup_preview.")
    if lineup.get("legal") is not True:
        raise RuntimeError("Lineup source is not legal=true.")
    if lineup.get("preview_only") is not True:
        raise RuntimeError("Lineup source is not preview_only=true.")
    if lineup.get("writes_database") is not False:
        raise RuntimeError("Lineup source must record writes_database=false.")
    if lineup.get("final_deadline_freeze") is not False:
        raise RuntimeError("Lineup source must still be an unfrozen PRE artifact.")
    formation = str(lineup.get("formation") or "")
    if formation not in LEGAL_FORMATIONS:
        raise RuntimeError("Lineup formation is not recognized as legal: %s" % formation)
    starters = list(lineup.get("starting_player_ids") or [])
    bench = list(lineup.get("bench_order") or [])
    if len(starters) != 11:
        raise RuntimeError("Lineup does not contain 11 starters.")
    if len(bench) != 4:
        raise RuntimeError("Lineup does not contain 4 bench players.")
    if len(set(str(v) for v in starters + bench)) != 15:
        raise RuntimeError("Lineup starter/bench assignments are not a unique 15-player partition.")

    squad = find_squad(transfer_state)
    if squad is None:
        raise RuntimeError("Could not extract a canonical 15-player squad from transfer state.")
    squad_validation = validate_squad(squad)

    rec = transfer_decision.get("recommended_action_by_target_gw_objective") or {}
    if not isinstance(rec, dict):
        rec = {}
    recommended_action = "ROLL"
    if rec.get("action") == "TRANSFER":
        recommended_action = "%s -> %s" % (rec.get("out_name"), rec.get("in_name"))

    stamp = utc_stamp()
    freeze_id = "gw%02d_final_freeze_%s" % (args.target_gw, stamp)
    freeze_dir = (
        planning_root
        / "frozen-snapshots"
        / str(args.season)
        / ("gw%02d" % args.target_gw)
        / "model-team"
        / freeze_id
    )
    if freeze_dir.exists():
        raise RuntimeError("Freeze destination already exists: %s" % freeze_dir)

    freeze_dir.mkdir(parents=True)

    copy_tree(prediction_run, freeze_dir / "prediction")
    copy_tree(transfer_run, freeze_dir / "transfer")
    copy_tree(candidate_package, freeze_dir / "candidate")

    published_source = (
        planning_root
        / "gw-pre"
        / str(args.season)
        / ("gw%02d" % args.target_gw)
        / "published"
        / prediction_run.name
    )
    published_snapshot_copied = False
    if published_source.is_dir():
        copy_tree(published_source, freeze_dir / "published")
        published_snapshot_copied = True

    shutil.copy2(receipt_path, freeze_dir / "publish_receipt.json")

    frozen_state = {
        "artifact_type": "fpl_model_team_frozen_state",
        "artifact_version": FREEZER_VERSION,
        "season": str(args.season),
        "gw": int(args.target_gw),
        "snapshot_kind": "final_pre_deadline",
        "final_pre_deadline_snapshot_frozen": True,
        "final_deadline_freeze": True,
        "freeze_id": freeze_id,
        "frozen_at_utc": utc_now(),
        "squad": squad,
        "lineup": {
            "formation": formation,
            "starting_player_ids": starters,
            "bench_order": bench,
            "captain": lineup.get("captain"),
            "vice_captain": lineup.get("vice_captain"),
            "starting_xi_predicted_points": lineup.get("starting_xi_predicted_points"),
            "objective_points": lineup.get("objective_points"),
        },
        "transfer_decision": {
            "recommended_action": recommended_action,
            "net_gain_vs_roll": rec.get("net_gain_vs_roll"),
            "bank_after_units": rec.get("bank_after_units"),
            "free_transfers_next_gameweek": rec.get("free_transfers_next_gameweek"),
            "final_weekly_transfer_decision": True,
        },
        "source_prediction_run": prediction_run.name,
        "source_transfer_run": transfer_run.name,
        "source_candidate_package": candidate_package.name,
        "source_publish_receipt": receipt_path.name,
        "writes_database": False,
        "writes_prediction_tables": False,
        "writes_live_squad_state": False,
    }
    frozen_state_path = freeze_dir / "model_team_state.json"
    write_json(frozen_state_path, frozen_state)

    fingerprints: Dict[str, Dict[str, Any]] = {}
    for path in sorted(p for p in freeze_dir.rglob("*") if p.is_file()):
        if path.name == "freeze_manifest.json":
            continue
        rel = str(path.relative_to(freeze_dir))
        fingerprints[rel] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    manifest = {
        "artifact_type": "fpl_weekly_final_deadline_freeze",
        "artifact_version": FREEZER_VERSION,
        "freeze_id": freeze_id,
        "created_at_utc": utc_now(),
        "season": str(args.season),
        "target_gw": int(args.target_gw),
        "snapshot_kind": "final_pre_deadline",
        "final_pre_deadline_snapshot_frozen": True,
        "final_deadline_freeze": True,
        "source_lineage": {
            "prediction_run": str(prediction_run),
            "transfer_run": str(transfer_run),
            "candidate_package": str(candidate_package),
            "publish_receipt": str(receipt_path),
            "published_snapshot_copied": published_snapshot_copied,
        },
        "validation": {
            "prediction_status": prediction_manifest.get("status"),
            "player_prediction_rows": player_rows,
            "match_prediction_rows": match_rows,
            "scoreline_rows": scoreline_rows,
            "publish_receipt_matches_prediction_run": True,
            "transfer_matches_prediction_run": True,
            "candidate_matches_prediction_and_transfer": True,
            "lineup_artifact_type": lineup.get("artifact_type"),
            "lineup_legal": lineup.get("legal"),
            "lineup_preview_only": lineup.get("preview_only"),
            "formation": formation,
            "starter_count": len(starters),
            "bench_count": len(bench),
            "squad": squad_validation,
        },
        "model_team_decision": {
            "recommended_action": recommended_action,
            "net_gain_vs_roll": rec.get("net_gain_vs_roll"),
            "bank_after_units": rec.get("bank_after_units"),
            "free_transfers_next_gameweek": rec.get("free_transfers_next_gameweek"),
            "captain": (lineup.get("captain") or {}).get("web_name")
                if isinstance(lineup.get("captain"), dict)
                else lineup.get("captain"),
            "vice_captain": (lineup.get("vice_captain") or {}).get("web_name")
                if isinstance(lineup.get("vice_captain"), dict)
                else lineup.get("vice_captain"),
            "starting_xi_predicted_points": lineup.get("starting_xi_predicted_points"),
            "objective_points": lineup.get("objective_points"),
        },
        "file_fingerprints": fingerprints,
        "immutable_contract": {
            "do_not_overwrite": True,
            "later_actuals_must_be_joined_separately": True,
            "source_pre_artifacts_remain_historical": True,
        },
    }
    manifest_path = freeze_dir / "freeze_manifest.json"
    write_json(manifest_path, manifest)

    summary = """# GW{gw} FINAL Deadline Freeze

- Freeze ID: `{freeze_id}`
- Season: `{season}`
- Snapshot kind: `final_pre_deadline`
- Frozen: **True**
- Player predictions: **{player_rows}**
- Match predictions: **{match_rows}**
- Scorelines: **{scoreline_rows}**
- Transfer: **{transfer}**
- Gain vs ROLL: **{gain}**
- Formation: **{formation}**
- Captain: **{captain}**
- Vice-captain: **{vice}**
- XI predicted points: **{xi}**
- Objective points: **{objective}**

This directory is the immutable pre-deadline Model Team evidence for GW{gw}.
Later actuals must be ingested separately and must never overwrite or reconstruct this freeze.
""".format(
        gw=args.target_gw,
        freeze_id=freeze_id,
        season=args.season,
        player_rows=player_rows,
        match_rows=match_rows,
        scoreline_rows=scoreline_rows,
        transfer=recommended_action,
        gain=rec.get("net_gain_vs_roll"),
        formation=formation,
        captain=manifest["model_team_decision"]["captain"],
        vice=manifest["model_team_decision"]["vice_captain"],
        xi=lineup.get("starting_xi_predicted_points"),
        objective=lineup.get("objective_points"),
    )
    (freeze_dir / "summary.md").write_text(summary, encoding="utf-8")

    print("=== FPL Weekly FINAL Deadline Freeze ===")
    print("status: PASS_FINAL_FREEZE")
    print("freeze_id:", freeze_id)
    print("freeze_dir:", freeze_dir)
    print("snapshot_kind: final_pre_deadline")
    print("final_pre_deadline_snapshot_frozen: True")
    print("player_prediction_rows:", player_rows)
    print("match_prediction_rows:", match_rows)
    print("scoreline_rows:", scoreline_rows)
    print("recommended_action:", recommended_action)
    print("net_gain_vs_roll:", rec.get("net_gain_vs_roll"))
    print("formation:", formation)
    print("captain:", manifest["model_team_decision"]["captain"])
    print("vice_captain:", manifest["model_team_decision"]["vice_captain"])
    print("starting_xi_predicted_points:", lineup.get("starting_xi_predicted_points"))
    print("objective_points:", lineup.get("objective_points"))
    print("manifest:", manifest_path)
    print("frozen_state:", frozen_state_path)


if __name__ == "__main__":
    main()
