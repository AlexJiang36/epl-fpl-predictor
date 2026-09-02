#!/usr/bin/env python3
"""Generic leakage-safe FPL Gameweek POST evaluation.

Day129A contract:
- official match/player actuals are captured only after target-GW finality;
- database ingest remains outside this module and is orchestrated by run_fpl_refresh;
- evaluation consumes only immutable FINAL pre-deadline evidence;
- predictions are never regenerated from post-result state;
- FINAL actual/evaluation artifacts are append-only.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

from ml.eval.evaluate_gameweek_post import (
    captain_bonus,
    classification_accuracy,
    evaluate_team,
    json_safe,
    mean,
    ndcg_at_k,
    point_metrics,
    result_label,
    spearman,
    top_k_metrics,
    write_csv,
)

POST_EVALUATION_VERSION = "fpl_gameweek_post_evaluation_v2"
ACTUALS_CONTRACT_VERSION = "fpl_gameweek_final_actuals_v1"
POST_MANIFEST_VERSION = "fpl_gameweek_post_manifest_v2"
POSITIONS = ("GKP", "DEF", "MID", "FWD")
FPL_BASE = "https://fantasy.premierleague.com/api"

# Narrow grandfathered compatibility for the real GW2 Team Alex Free Hit
# evidence captured before the deadline.  This is intentionally pinned to the
# exact archive fingerprint; it is not a generic post-result import path.
LEGACY_TEAM_ALEX_ARCHIVE_CONTRACTS: Dict[Tuple[str, int], Dict[str, str]] = {
    ("2026_27", 2): {
        "relative_path": "team-alex/TEAM_ALEX_GW2_FREE_HIT_SNAPSHOT.zip",
        "sha256": "9d3ab2a7c39cf14c217adad2fde385a150991653fe6d498167982d707d859b27",
        "member": "team_alex_gw2_free_hit_snapshot.json",
        "artifact_type": "team_alex_free_hit_snapshot",
    }
}


class GameweekEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenBaseline:
    kind: str
    root: Path
    manifest_path: Path
    player_predictions: Path
    match_predictions: Path
    scoreline_predictions: Path
    model_team_state: Path
    model_team_decision: Optional[Path]
    team_alex_state: Path
    as_of_utc: str
    deadline_utc: str
    fingerprint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture FINAL FPL actuals and evaluate immutable frozen PRE evidence."
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--planning-root", default=None)
    parser.add_argument("--season", required=True)
    parser.add_argument("--gw", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("full", "capture", "evaluate"),
        default="full",
    )
    parser.add_argument("--actual-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-final-actuals-only",
        action="store_true",
        help="Do not fetch FPL actual evidence; require an existing FINAL actual manifest.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def detect_repo_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / ".git").exists():
            raise GameweekEvaluationError("repo-root does not contain .git: %s" % root)
        return root
    script = Path(__file__).resolve()
    for parent in (script.parent,) + tuple(script.parents):
        if (parent / ".git").exists():
            return parent
    raise GameweekEvaluationError("Could not auto-detect repository root.")


def detect_planning_root(repo_root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return repo_root.parent / "private-planning"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json_new(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise GameweekEvaluationError("Refusing overwrite: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def write_bytes_new(path: Path, payload: bytes) -> Path:
    if path.exists():
        raise GameweekEvaluationError("Refusing overwrite: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def as_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    return float(value)


def first_value(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row.get(name) not in (None, ""):
            return row.get(name)
    return None


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def fetch_json_bytes(url: str, timeout: float = 30.0) -> Tuple[bytes, Any]:
    request = Request(url, headers={"User-Agent": "fpl-predictor-post-evaluation/1.0"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return raw, json.loads(raw.decode("utf-8"))


def actuals_root(planning_root: Path, season: str, gw: int) -> Path:
    return planning_root / "gw-post" / season / ("gw%02d" % gw) / "actuals"


def evaluation_root(planning_root: Path, season: str, gw: int) -> Path:
    return planning_root / "gw-post" / season / ("gw%02d" % gw) / "evaluation"


def _actual_manifest_candidates(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    candidates: List[Path] = []
    for path in root.rglob("*.json"):
        name = path.name.lower()
        if "actual" not in name or "manifest" not in name:
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        if payload.get("event_live_file") and payload.get("fixtures_file"):
            candidates.append(path)
    return candidates


def validate_final_actual_manifest(path: Path, season: str, gw: int) -> Dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise GameweekEvaluationError("Actual manifest root is not an object: %s" % path)
    if payload.get("season") not in (None, season):
        raise GameweekEvaluationError("Actual manifest season mismatch: %s" % path)
    if payload.get("gw") not in (None, gw, str(gw)):
        raise GameweekEvaluationError("Actual manifest GW mismatch: %s" % path)
    if not bool(payload.get("all_fixtures_finished")):
        raise GameweekEvaluationError("Actual manifest is not FINAL: %s" % path)
    if payload.get("event_finished") is False:
        raise GameweekEvaluationError("FPL event is not finished in actual manifest.")
    if payload.get("event_data_checked") is False:
        raise GameweekEvaluationError("FPL event data_checked is false in actual manifest.")

    parent = path.parent
    checks = (
        ("event_live_file", "event_live_sha256"),
        ("fixtures_file", "fixtures_sha256"),
    )
    for file_key, hash_key in checks:
        source = parent / str(payload.get(file_key) or "")
        if not source.is_file():
            raise GameweekEvaluationError("Actual evidence file missing: %s" % source)
        expected = str(payload.get(hash_key) or "")
        actual = sha256_file(source)
        if not expected or expected != actual:
            raise GameweekEvaluationError("Actual evidence hash mismatch: %s" % file_key)
    return dict(payload)


def discover_final_actual_manifest(
    planning_root: Path,
    season: str,
    gw: int,
) -> Optional[Path]:
    valid: List[Path] = []
    for path in _actual_manifest_candidates(actuals_root(planning_root, season, gw)):
        try:
            validate_final_actual_manifest(path, season, gw)
        except Exception:
            continue
        valid.append(path)
    if not valid:
        return None
    return max(valid, key=lambda p: p.stat().st_mtime)


def capture_or_reuse_final_actuals(
    *,
    planning_root: Path,
    season: str,
    gw: int,
    reuse_existing: bool = False,
    reuse_only: bool = False,
) -> Path:
    existing = discover_final_actual_manifest(planning_root, season, gw)
    # FINAL official evidence is immutable and idempotent: once a validated capture
    # exists, every later POST run reuses it instead of creating a duplicate capture.
    if existing is not None:
        validate_final_actual_manifest(existing, season, gw)
        return existing
    if reuse_only:
        raise GameweekEvaluationError(
            "No existing FINAL actual manifest is available for offline/reuse-only POST."
        )

    live_raw, live = fetch_json_bytes("%s/event/%s/live/" % (FPL_BASE, gw))
    fixtures_raw, all_fixtures = fetch_json_bytes("%s/fixtures/?event=%s" % (FPL_BASE, gw))
    bootstrap_raw, bootstrap = fetch_json_bytes("%s/bootstrap-static/" % FPL_BASE)

    if not isinstance(live, Mapping) or not isinstance(live.get("elements"), list):
        raise GameweekEvaluationError("FPL event-live response is missing elements.")
    if not isinstance(all_fixtures, list) or not all_fixtures:
        raise GameweekEvaluationError("FPL fixture response contains no target-GW fixtures.")
    events = bootstrap.get("events") if isinstance(bootstrap, Mapping) else None
    if not isinstance(events, list):
        raise GameweekEvaluationError("FPL bootstrap response is missing events.")
    event = next((row for row in events if as_int(row.get("id"), -1) == int(gw)), None)
    if not isinstance(event, Mapping):
        raise GameweekEvaluationError("FPL bootstrap does not contain target GW=%s." % gw)

    fixture_rows = [row for row in all_fixtures if as_int(row.get("event"), -1) == int(gw)]
    if not fixture_rows:
        raise GameweekEvaluationError("No target-GW fixture rows were returned by FPL.")
    finished_count = sum(1 for row in fixture_rows if bool(row.get("finished")))
    score_complete = sum(
        1
        for row in fixture_rows
        if row.get("team_h_score") is not None and row.get("team_a_score") is not None
    )
    event_finished = bool(event.get("finished"))
    event_data_checked = bool(event.get("data_checked"))
    all_finished = finished_count == len(fixture_rows) and score_complete == len(fixture_rows)

    if not event_finished or not event_data_checked or not all_finished:
        raise GameweekEvaluationError(
            "Target GW actuals are incomplete: event_finished=%s data_checked=%s "
            "fixtures_finished=%s/%s scores_complete=%s/%s"
            % (
                event_finished,
                event_data_checked,
                finished_count,
                len(fixture_rows),
                score_complete,
                len(fixture_rows),
            )
        )

    elements = live.get("elements") or []
    if not elements:
        raise GameweekEvaluationError("Target GW event-live contains zero player rows.")

    stamp = utc_stamp()
    capture_dir = actuals_root(planning_root, season, gw) / ("final_%s" % stamp)
    if capture_dir.exists():
        raise GameweekEvaluationError("Actual capture already exists: %s" % capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=False)

    live_name = "fpl_event_%s_live_%s_FINAL.json" % (gw, stamp)
    fixtures_name = "fpl_fixtures_gw%s_%s_FINAL.json" % (gw, stamp)
    bootstrap_name = "fpl_bootstrap_gw%s_%s_FINAL.json" % (gw, stamp)
    write_bytes_new(capture_dir / live_name, live_raw)
    write_bytes_new(capture_dir / fixtures_name, fixtures_raw)
    write_bytes_new(capture_dir / bootstrap_name, bootstrap_raw)

    stats_rows = [row.get("stats") or {} for row in elements if isinstance(row, Mapping)]
    manifest = {
        "contract_version": ACTUALS_CONTRACT_VERSION,
        "status": "FINAL",
        "season": season,
        "gw": int(gw),
        "captured_at_utc": stamp,
        "source": "official_fpl_api",
        "immutable": True,
        "event_live_file": live_name,
        "event_live_sha256": sha256_bytes(live_raw),
        "fixtures_file": fixtures_name,
        "fixtures_sha256": sha256_bytes(fixtures_raw),
        "bootstrap_file": bootstrap_name,
        "bootstrap_sha256": sha256_bytes(bootstrap_raw),
        "event_finished": event_finished,
        "event_data_checked": event_data_checked,
        "all_fixtures_finished": all_finished,
        "fixture_count": len(fixture_rows),
        "fixtures_finished": finished_count,
        "fixture_scores_complete": score_complete,
        "live_player_rows": len(elements),
        "minutes_gt_0": sum(1 for row in stats_rows if as_int(row.get("minutes"), 0) > 0),
        "points_nonzero": sum(1 for row in stats_rows if as_int(row.get("total_points"), 0) != 0),
        "total_event_points": sum(as_int(row.get("total_points"), 0) for row in stats_rows),
    }
    manifest_name = "gw%s_actuals_manifest_%s_FINAL.json" % (gw, stamp)
    manifest_path = write_json_new(capture_dir / manifest_name, manifest)
    validate_final_actual_manifest(manifest_path, season, gw)
    return manifest_path


def _parse_utc(value: Any, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise GameweekEvaluationError("Missing %s." % label)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise GameweekEvaluationError("%s must be timezone-aware." % label)
    return parsed.astimezone(timezone.utc)


def _verify_checksum_file(root: Path, checksum_file: Path) -> None:
    if not checksum_file.is_file():
        raise GameweekEvaluationError("Frozen baseline is missing checksum file: %s" % checksum_file)
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise GameweekEvaluationError("Malformed frozen checksum row: %s" % line)
        expected, rel = parts[0], parts[1].strip()
        path = root / rel
        if not path.is_file() or sha256_file(path) != expected:
            raise GameweekEvaluationError("Frozen baseline checksum mismatch: %s" % rel)


def _find_csv_with_columns(root: Path, required: Sequence[str], preferred: Sequence[str]) -> Path:
    for name in preferred:
        candidate = root / name
        if candidate.is_file():
            return candidate
    matches: List[Path] = []
    for path in root.rglob("*.csv") if root.is_dir() else []:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
        except Exception:
            continue
        if all(column in header for column in required):
            matches.append(path)
    if len(matches) != 1:
        raise GameweekEvaluationError(
            "Expected exactly one CSV under %s with columns=%s; found %s"
            % (root, list(required), matches)
        )
    return matches[0]


def _reference_source(root: Path, track: str) -> Path:
    track_dir = root / "tracks" / track
    reference_path = track_dir / "reference.json"
    reference = read_json(reference_path)
    if not isinstance(reference, Mapping):
        raise GameweekEvaluationError("Invalid %s reference.json" % track)
    rel = reference.get("snapshot_copy_path")
    if not rel:
        display = "Team Alex" if track == "team_alex" else track.replace("_", " ").title()
        raise GameweekEvaluationError("%s reference has no immutable snapshot copy." % display)
    source = root / str(rel)
    if not source.exists():
        raise GameweekEvaluationError("Frozen %s source copy is missing: %s" % (track, source))
    expected = str(reference.get("snapshot_copy_sha256") or "")
    if expected and sha256_path(source) != expected:
        raise GameweekEvaluationError("Frozen %s source-copy hash mismatch." % track)
    return source


def _resolve_team_alex_source_from_generic(root: Path) -> Path:
    source = _reference_source(root, "team_alex")
    if source.is_file():
        return source
    jsons = [p for p in source.rglob("*.json") if p.is_file()]
    if len(jsons) == 1:
        return jsons[0]
    for path in jsons:
        name = path.name.lower()
        if "team" in name and "alex" in name:
            return path
    raise GameweekEvaluationError("Frozen Team Alex source is not a uniquely scoreable JSON artifact.")


def _discover_generic_final(gw_root: Path, season: str, gw: int) -> Optional[FrozenBaseline]:
    candidates: List[Tuple[Path, Mapping[str, Any]]] = []
    for manifest_path in gw_root.rglob("snapshot_manifest.json") if gw_root.is_dir() else []:
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if not isinstance(manifest, Mapping):
            continue
        if not bool(manifest.get("final_pre_deadline_snapshot_frozen")):
            continue
        if str(manifest.get("snapshot_kind")) != "final_pre_deadline":
            continue
        if str(manifest.get("season")) != season or as_int(manifest.get("target_gw"), -1) != gw:
            continue
        candidates.append((manifest_path.parent, manifest))
    if not candidates:
        return None
    root, manifest = max(candidates, key=lambda item: item[0].stat().st_mtime)

    if not bool(manifest.get("as_of_before_deadline")):
        raise GameweekEvaluationError("Frozen PRE manifest does not assert as_of_before_deadline=true.")
    safety = manifest.get("safety") or {}
    if bool(safety.get("target_gw_actuals_consumed")):
        raise GameweekEvaluationError("Frozen PRE manifest consumed target-GW actuals.")
    if safety.get("post_deadline_reconstruction_allowed") not in (False, None):
        raise GameweekEvaluationError("Frozen PRE manifest allows post-deadline reconstruction.")
    as_of = _parse_utc(manifest.get("as_of_utc"), "frozen as_of_utc")
    deadline = _parse_utc(manifest.get("fpl_deadline_utc"), "frozen fpl_deadline_utc")
    if as_of >= deadline:
        raise GameweekEvaluationError("Frozen PRE as-of is not before deadline.")
    _verify_checksum_file(root, root / "SHA256SUMS.txt")

    player_source = _reference_source(root, "player_model")
    match_source = _reference_source(root, "match_model")
    player_csv = _find_csv_with_columns(
        player_source if player_source.is_dir() else player_source.parent,
        ("fpl_player_id", "predicted_points"),
        (player_source.name,) if player_source.is_file() else ("player_predictions_preview.csv", "global_player_prediction_snapshot.csv"),
    )
    match_csv = _find_csv_with_columns(
        match_source if match_source.is_dir() else match_source.parent,
        ("fpl_fixture_id",),
        ("match_predictions_preview.csv", "match_preview_csv.csv"),
    )
    scoreline_csv = _find_csv_with_columns(
        match_source if match_source.is_dir() else match_source.parent,
        ("fpl_fixture_id",),
        ("scoreline_preview.csv", "scoreline_csv.csv"),
    )
    model_state = root / "tracks" / "model_team" / "model_team_state.json"
    chosen_plan = root / "tracks" / "model_team" / "chosen_plan.json"
    team_alex = _resolve_team_alex_source_from_generic(root)
    for required in (model_state, chosen_plan, team_alex):
        if not required.is_file():
            raise GameweekEvaluationError("Frozen baseline missing required file: %s" % required)

    return FrozenBaseline(
        kind="day127b_final_snapshot",
        root=root,
        manifest_path=root / "snapshot_manifest.json",
        player_predictions=player_csv,
        match_predictions=match_csv,
        scoreline_predictions=scoreline_csv,
        model_team_state=model_state,
        model_team_decision=chosen_plan,
        team_alex_state=team_alex,
        as_of_utc=str(manifest.get("as_of_utc")),
        deadline_utc=str(manifest.get("fpl_deadline_utc")),
        fingerprint=str(manifest.get("package_fingerprint") or sha256_path(root)),
    )


def _verify_legacy_freeze(root: Path, manifest: Mapping[str, Any]) -> None:
    fingerprints = manifest.get("file_fingerprints") or {}
    if not isinstance(fingerprints, Mapping) or not fingerprints:
        raise GameweekEvaluationError("Legacy FINAL freeze has no file fingerprints.")
    for rel, meta in fingerprints.items():
        path = root / str(rel)
        expected = str((meta or {}).get("sha256") or "") if isinstance(meta, Mapping) else ""
        if not path.is_file() or not expected or sha256_file(path) != expected:
            raise GameweekEvaluationError("Legacy FINAL freeze fingerprint mismatch: %s" % rel)


def _deadline_from_bootstrap(bootstrap_path: Path, gw: int) -> str:
    payload = read_json(bootstrap_path)
    events = payload.get("events") if isinstance(payload, Mapping) else None
    if not isinstance(events, list):
        raise GameweekEvaluationError("Frozen bootstrap is missing events.")
    event = next((row for row in events if as_int(row.get("id"), -1) == gw), None)
    if not isinstance(event, Mapping) or not event.get("deadline_time"):
        raise GameweekEvaluationError("Frozen bootstrap is missing target-GW deadline.")
    return str(event["deadline_time"])


def _validate_legacy_team_alex_archive(
    path: Path,
    deadline_utc: str,
    season: str,
    gw: int,
) -> Mapping[str, Any]:
    contract = LEGACY_TEAM_ALEX_ARCHIVE_CONTRACTS.get((season, int(gw)))
    if not contract:
        raise GameweekEvaluationError(
            "No approved legacy Team Alex archive contract for season=%s gw=%s." % (season, gw)
        )
    expected_name = Path(contract["relative_path"]).name
    if path.name != expected_name:
        raise GameweekEvaluationError("Unexpected legacy Team Alex archive name: %s" % path.name)
    expected_sha = contract["sha256"]
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise GameweekEvaluationError(
            "Legacy Team Alex archive fingerprint mismatch: expected=%s actual=%s"
            % (expected_sha, actual_sha)
        )

    deadline = _parse_utc(deadline_utc, "deadline_utc")
    member = contract["member"]
    try:
        with zipfile.ZipFile(str(path), "r") as archive:
            names = archive.namelist()
            if member not in names:
                raise GameweekEvaluationError(
                    "Legacy Team Alex archive missing scoreable member: %s" % member
                )
            info = archive.getinfo(member)
            # ZIP stores a timezone-naive member timestamp.  This archive is a
            # one-off grandfathered artifact whose exact bytes are SHA-pinned;
            # interpreting its embedded timestamp as UTC is deliberately
            # conservative for the historical acceptance check.
            member_time = datetime(*info.date_time, tzinfo=timezone.utc)
            if member_time >= deadline:
                raise GameweekEvaluationError(
                    "Legacy Team Alex archive member timestamp is not pre-deadline."
                )
            raw = archive.read(member)
    except zipfile.BadZipFile as exc:
        raise GameweekEvaluationError("Invalid legacy Team Alex ZIP artifact: %s" % path) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise GameweekEvaluationError("Legacy Team Alex JSON member is invalid.") from exc
    if not isinstance(payload, Mapping):
        raise GameweekEvaluationError("Legacy Team Alex JSON member is not an object.")
    if str(payload.get("artifact_type")) != contract["artifact_type"]:
        raise GameweekEvaluationError("Legacy Team Alex artifact_type mismatch.")
    if str(payload.get("season")) != season or as_int(payload.get("gameweek"), -1) != int(gw):
        raise GameweekEvaluationError("Legacy Team Alex season/gameweek mismatch.")
    if str(payload.get("team")) != "Team Alex":
        raise GameweekEvaluationError("Legacy Team Alex team marker mismatch.")

    chip = payload.get("chip") or {}
    if not isinstance(chip, Mapping):
        raise GameweekEvaluationError("Legacy Team Alex chip block is invalid.")
    if str(chip.get("name")) != "free_hit" or not bool(chip.get("active")):
        raise GameweekEvaluationError("Legacy Team Alex artifact is not the active GW2 Free Hit snapshot.")
    if chip.get("persistent_squad_overwrite") is not False:
        raise GameweekEvaluationError(
            "Legacy Team Alex Free Hit artifact must preserve persistent_squad_overwrite=false."
        )

    starters = payload.get("starting_xi")
    bench = payload.get("bench")
    if not isinstance(starters, list) or len(starters) != 11:
        raise GameweekEvaluationError("Legacy Team Alex snapshot must contain 11 starters.")
    if not isinstance(bench, list) or len(bench) != 4:
        raise GameweekEvaluationError("Legacy Team Alex snapshot must contain four bench players.")
    names: List[str] = []
    for row in starters + bench:
        if not isinstance(row, Mapping) or not row.get("name"):
            raise GameweekEvaluationError("Legacy Team Alex selection row is missing a player name.")
        names.append(normalize_name(row.get("name")))
    if len(set(names)) != 15:
        raise GameweekEvaluationError("Legacy Team Alex snapshot must contain 15 unique players.")
    if not payload.get("captain") or not payload.get("vice_captain"):
        raise GameweekEvaluationError("Legacy Team Alex snapshot is missing captain/vice-captain.")
    return payload


def _discover_frozen_team_alex(
    gw_root: Path,
    deadline_utc: str,
    season: str,
    gw: int,
) -> Path:
    deadline = _parse_utc(deadline_utc, "deadline_utc")
    candidates: List[Path] = []
    for path in gw_root.rglob("*.json") if gw_root.is_dir() else []:
        lowered = path.as_posix().lower()
        if "team-alex" not in lowered and "team_alex" not in lowered and "gliding" not in lowered:
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        marker = bool(payload.get("final_pre_deadline_snapshot_frozen")) or str(
            payload.get("snapshot_kind") or ""
        ) == "final_pre_deadline" or bool(payload.get("final_deadline_freeze"))
        if not marker:
            continue
        as_of_value = first_value(
            payload,
            ("as_of_utc", "frozen_at_utc", "created_at_utc", "submitted_at_utc"),
        )
        if as_of_value:
            try:
                if _parse_utc(as_of_value, "Team Alex frozen timestamp") >= deadline:
                    continue
            except Exception:
                continue
        candidates.append(path)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # Day129A grandfathered compatibility: consume only the exact SHA-pinned
    # historical GW2 Free Hit archive.  Never scan arbitrary ZIPs and never
    # reconstruct a squad from notes or post-result state.
    contract = LEGACY_TEAM_ALEX_ARCHIVE_CONTRACTS.get((season, int(gw)))
    if contract:
        archive_path = gw_root / contract["relative_path"]
        if archive_path.is_file():
            _validate_legacy_team_alex_archive(archive_path, deadline_utc, season, gw)
            return archive_path

    raise GameweekEvaluationError(
        "Missing immutable frozen Team Alex baseline under %s. "
        "POST may not reconstruct Team Alex from notes or post-result state." % gw_root
    )


def _discover_legacy_weekly_final(gw_root: Path, season: str, gw: int) -> Optional[FrozenBaseline]:
    candidates: List[Tuple[Path, Mapping[str, Any]]] = []
    for manifest_path in gw_root.rglob("freeze_manifest.json") if gw_root.is_dir() else []:
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if not isinstance(manifest, Mapping):
            continue
        if not bool(manifest.get("final_pre_deadline_snapshot_frozen")):
            continue
        if str(manifest.get("snapshot_kind")) != "final_pre_deadline":
            continue
        if str(manifest.get("season")) != season or as_int(manifest.get("target_gw"), -1) != gw:
            continue
        candidates.append((manifest_path.parent, manifest))
    if not candidates:
        return None
    root, manifest = max(candidates, key=lambda item: item[0].stat().st_mtime)
    _verify_legacy_freeze(root, manifest)

    prediction = root / "prediction"
    player_csv = prediction / "player_predictions_preview.csv"
    match_csv = prediction / "match_predictions_preview.csv"
    scoreline_csv = prediction / "scoreline_preview.csv"
    bootstrap = prediction / "bootstrap_snapshot.json"
    prediction_manifest_path = prediction / "run_manifest.json"
    model_state = root / "model_team_state.json"
    for required in (
        player_csv,
        match_csv,
        scoreline_csv,
        bootstrap,
        prediction_manifest_path,
        model_state,
    ):
        if not required.is_file():
            raise GameweekEvaluationError("Legacy FINAL freeze missing: %s" % required)

    prediction_manifest = read_json(prediction_manifest_path)
    as_of = str(prediction_manifest.get("created_at") or prediction_manifest.get("created_at_utc") or "")
    deadline = _deadline_from_bootstrap(bootstrap, gw)
    if _parse_utc(as_of, "frozen prediction created_at") >= _parse_utc(deadline, "FPL deadline"):
        raise GameweekEvaluationError("Frozen legacy prediction was not created before deadline.")
    freeze_created = str(manifest.get("created_at_utc") or "")
    if freeze_created and _parse_utc(freeze_created, "freeze created_at_utc") >= _parse_utc(deadline, "FPL deadline"):
        raise GameweekEvaluationError("Legacy FINAL freeze timestamp is not before deadline.")

    transfer_decision: Optional[Path] = None
    for name in ("transfer_decision_preview.json", "transfer_decision.json"):
        found = list((root / "transfer").rglob(name)) if (root / "transfer").is_dir() else []
        if found:
            transfer_decision = max(found, key=lambda p: p.stat().st_mtime)
            break
    team_alex = _discover_frozen_team_alex(gw_root, deadline, season, gw)

    return FrozenBaseline(
        kind="legacy_weekly_final_freeze",
        root=root,
        manifest_path=root / "freeze_manifest.json",
        player_predictions=player_csv,
        match_predictions=match_csv,
        scoreline_predictions=scoreline_csv,
        model_team_state=model_state,
        model_team_decision=transfer_decision,
        team_alex_state=team_alex,
        as_of_utc=as_of,
        deadline_utc=deadline,
        fingerprint=hashlib.sha256(
            (sha256_path(root) + "|" + sha256_file(team_alex)).encode("utf-8")
        ).hexdigest(),
    )


def discover_frozen_baseline(planning_root: Path, season: str, gw: int) -> FrozenBaseline:
    gw_root = planning_root / "frozen-snapshots" / season / ("gw%02d" % gw)
    if not gw_root.is_dir():
        raise GameweekEvaluationError("Missing frozen pre-deadline root: %s" % gw_root)
    generic = _discover_generic_final(gw_root, season, gw)
    if generic is not None:
        return generic
    legacy = _discover_legacy_weekly_final(gw_root, season, gw)
    if legacy is not None:
        return legacy
    raise GameweekEvaluationError(
        "No immutable FINAL pre-deadline baseline found for season=%s gw=%s. "
        "POST refuses post-result reconstruction." % (season, gw)
    )


def _player_id(row: Mapping[str, Any]) -> int:
    value = first_value(row, ("fpl_player_id", "player_id", "element", "id"))
    if value is None:
        raise GameweekEvaluationError("Player row is missing FPL identity: %s" % row)
    return as_int(value)


def _position(row: Mapping[str, Any]) -> str:
    value = str(first_value(row, ("position", "position_name", "element_type_name")) or "").upper()
    if value in POSITIONS:
        return value
    numeric = first_value(row, ("element_type", "position_id"))
    mapping = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    if numeric is not None and as_int(numeric, -1) in mapping:
        return mapping[as_int(numeric)]
    raise GameweekEvaluationError("Player row has unknown position: %s" % row)


def _selection_eligible(row: Mapping[str, Any]) -> bool:
    if "selection_eligible" in row and row.get("selection_eligible") not in (None, ""):
        return boolish(row.get("selection_eligible"))
    status = str(row.get("status") or "").strip().lower()
    chance = first_value(row, ("chance_of_playing_next_round", "chance_of_playing_this_round"))
    fixture_count = first_value(row, ("fixture_count",))
    if fixture_count is not None and as_int(fixture_count, 0) <= 0:
        return False
    if status in ("s", "u", "n"):
        return False
    if chance is not None and as_float(chance, 0.0) <= 0.0:
        return False
    return True


def build_player_evaluation(
    player_csv: Path,
    event_live_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    frozen = read_csv(player_csv)
    if not frozen:
        raise GameweekEvaluationError("Frozen Player Model artifact contains no rows.")
    live = read_json(event_live_path)
    elements = live.get("elements") if isinstance(live, Mapping) else None
    if not isinstance(elements, list) or not elements:
        raise GameweekEvaluationError("Official event-live contains no elements.")
    actual_by_id = {
        as_int(row.get("id")): (row.get("stats") or {})
        for row in elements
        if isinstance(row, Mapping) and row.get("id") is not None
    }

    rows: List[Dict[str, Any]] = []
    frozen_ids: set = set()
    missing: List[int] = []
    for raw in frozen:
        pid = _player_id(raw)
        frozen_ids.add(pid)
        stats = actual_by_id.get(pid)
        if stats is None:
            missing.append(pid)
            continue
        predicted_raw = first_value(raw, ("predicted_points", "gw_predicted_points", "next_gw_predicted_points"))
        if predicted_raw is None:
            raise GameweekEvaluationError("Frozen player row missing predicted_points for id=%s." % pid)
        expected_minutes = first_value(raw, ("expected_minutes", "expected_minutes_total"))
        appearance = first_value(raw, ("appearance_probability", "blended_appearance_probability"))
        start = first_value(raw, ("start_probability", "blended_start_probability"))
        row = {
            "fpl_player_id": pid,
            "player_name": str(first_value(raw, ("player_name", "web_name", "name")) or pid),
            "web_name": str(first_value(raw, ("web_name", "player_name", "name")) or pid),
            "position": _position(raw),
            "team_short_name": str(first_value(raw, ("team_short_name", "team_name", "team")) or ""),
            "selection_eligible": _selection_eligible(raw),
            "prediction_available": True,
            "predicted_points": as_float(predicted_raw),
            "expected_minutes": as_float(expected_minutes, 0.0),
            "appearance_probability": as_float(appearance, 0.0),
            "start_probability": as_float(start, 0.0),
            "actual_points": as_int(stats.get("total_points"), 0),
            "actual_minutes": as_int(stats.get("minutes"), 0),
            "actual_starts": as_int(stats.get("starts"), 0),
            "actual_appearance": as_int(stats.get("minutes"), 0) > 0,
            "actual_start": as_int(stats.get("starts"), 0) > 0,
        }
        rows.append(row)
    if missing:
        raise GameweekEvaluationError("Frozen prediction rows missing official actuals: %s" % missing)

    eligible = [row for row in rows if row["selection_eligible"] and row["prediction_available"]]
    if not eligible:
        raise GameweekEvaluationError("Frozen Player Model has zero eligible evaluation rows.")
    played = [row for row in eligible if row["actual_minutes"] > 0]
    sixty = [row for row in eligible if row["actual_minutes"] >= 60]
    position_level: Dict[str, Any] = {}
    for position in POSITIONS:
        subset = [row for row in eligible if row["position"] == position]
        if subset:
            metrics = point_metrics(subset)
            top5 = top_k_metrics(subset, 5)
            position_level[position] = {
                "n": len(subset),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["mean_error_bias_pred_minus_actual"],
                "top_5_hits": top5["hits"],
                "top_5_precision": top5["precision_at_k"],
            }
        else:
            position_level[position] = {"n": 0}

    actual_only = sorted(set(actual_by_id) - frozen_ids)
    result = {
        "frozen_prediction_source": str(first_value(frozen[0], ("prediction_source", "model_name", "model_version")) or ""),
        "coverage": {
            "frozen_rows": len(frozen),
            "matched_frozen_rows": len(rows),
            "official_actual_rows": len(elements),
            "actual_only_rows": len(actual_only),
            "actual_only_ids": actual_only,
            "primary_eligible_rows": len(eligible),
        },
        "cohorts": {
            "all_eligible_players": point_metrics(eligible),
            "actually_played_players": point_metrics(played),
            "60_plus_minutes_players": point_metrics(sixty),
        },
        "top_k": {str(k): top_k_metrics(eligible, k) for k in (10, 20, 50)},
        "ranking_quality": {
            "spearman_rank_correlation": spearman(
                [row["predicted_points"] for row in eligible],
                [row["actual_points"] for row in eligible],
            ),
            "ndcg_at_20": ndcg_at_k(eligible, 20),
        },
        "position_level": position_level,
        "availability_and_minutes": {
            "appearance_accuracy_threshold_0_5": classification_accuracy(
                eligible, "appearance_probability", "actual_appearance"
            ),
            "start_accuracy_threshold_0_5": classification_accuracy(
                eligible, "start_probability", "actual_start"
            ),
            "minutes_mae": mean(
                [abs(row["expected_minutes"] - row["actual_minutes"]) for row in eligible]
            ),
        },
    }
    by_id = {row["fpl_player_id"]: row for row in rows}
    return result, rows, by_id


def _probability(row: Mapping[str, Any], kind: str) -> float:
    names = {
        "home": ("home_win_probability", "pred_home_win", "p_home_win", "home_probability"),
        "draw": ("draw_probability", "pred_draw", "p_draw"),
        "away": ("away_win_probability", "pred_away_win", "p_away_win", "away_probability"),
    }[kind]
    value = first_value(row, names)
    if value is None:
        raise GameweekEvaluationError("Frozen Match Model missing %s probability." % kind)
    return as_float(value)


def _prediction_label(row: Mapping[str, Any]) -> str:
    raw = str(first_value(row, ("predicted_result_label", "pred_result", "predicted_result", "prediction")) or "").strip().lower()
    mapping = {
        "h": "home_win",
        "home": "home_win",
        "home_win": "home_win",
        "d": "draw",
        "draw": "draw",
        "a": "away_win",
        "away": "away_win",
        "away_win": "away_win",
    }
    if raw in mapping:
        return mapping[raw]
    probs = {"home_win": _probability(row, "home"), "draw": _probability(row, "draw"), "away_win": _probability(row, "away")}
    return max(probs, key=probs.get)


def _scoreline_value(row: Mapping[str, Any], index: int) -> Optional[str]:
    value = first_value(
        row,
        (
            "top_%d_scoreline" % index,
            "top%d_scoreline" % index,
            "scoreline_top_%d" % index,
        ),
    )
    return str(value) if value not in (None, "") else None


def build_match_evaluation(
    match_csv: Path,
    scoreline_csv: Path,
    fixtures_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    match_rows = read_csv(match_csv)
    score_rows = read_csv(scoreline_csv)
    if not match_rows or not score_rows:
        raise GameweekEvaluationError("Frozen Match Model/scoreline artifact is empty.")
    score_by_fixture = {_player_or_fixture_id(row, fixture=True): row for row in score_rows}
    fixtures = read_json(fixtures_path)
    if not isinstance(fixtures, list):
        raise GameweekEvaluationError("Official fixtures evidence must be a list.")
    actual = {as_int(row.get("id")): row for row in fixtures if isinstance(row, Mapping)}

    per_match: List[Dict[str, Any]] = []
    log_losses: List[float] = []
    briers: List[float] = []
    home_errors: List[float] = []
    away_errors: List[float] = []
    total_errors: List[float] = []
    correct = top1 = top3 = top5 = 0
    for row in match_rows:
        fixture_id = _player_or_fixture_id(row, fixture=True)
        actual_row = actual.get(fixture_id)
        if actual_row is None or not bool(actual_row.get("finished")):
            raise GameweekEvaluationError("Frozen match fixture lacks FINAL actual: %s" % fixture_id)
        home_goals = as_int(actual_row.get("team_h_score"))
        away_goals = as_int(actual_row.get("team_a_score"))
        label = result_label(home_goals, away_goals)
        probs = {
            "home_win": _probability(row, "home"),
            "draw": _probability(row, "draw"),
            "away_win": _probability(row, "away"),
        }
        probability_actual = max(1e-15, min(1.0, probs[label]))
        log_losses.append(-math.log(probability_actual))
        target = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        target[label] = 1.0
        briers.append(sum((probs[key] - target[key]) ** 2 for key in target))
        predicted_label = _prediction_label(row)
        correct += int(predicted_label == label)

        score = score_by_fixture.get(fixture_id)
        if score is None:
            raise GameweekEvaluationError("Frozen scoreline missing fixture=%s." % fixture_id)
        # Day128B can carry expected-goal fields on the 1X2 match row while
        # scoreline ranks live in a separate scoreline CSV. Prefer scoreline xG
        # when present, but fall back to the frozen match row; never reconstruct.
        expected_home = first_value(score, ("expected_home_goals", "home_expected_goals", "home_xg"))
        expected_away = first_value(score, ("expected_away_goals", "away_expected_goals", "away_xg"))
        if expected_home is None:
            expected_home = first_value(row, ("expected_home_goals", "home_expected_goals", "home_xg"))
        if expected_away is None:
            expected_away = first_value(row, ("expected_away_goals", "away_expected_goals", "away_xg"))
        if expected_home is None or expected_away is None:
            raise GameweekEvaluationError(
                "Frozen Match/scoreline evidence missing expected goals for fixture=%s." % fixture_id
            )
        eh, ea = as_float(expected_home), as_float(expected_away)
        home_errors.append(abs(eh - home_goals))
        away_errors.append(abs(ea - away_goals))
        total_errors.append(abs((eh + ea) - (home_goals + away_goals)))
        actual_score = "%d-%d" % (home_goals, away_goals)
        tops = [_scoreline_value(score, index) for index in range(1, 6)]
        available_top_count = sum(1 for value in tops if value is not None)
        if tops[0] is None:
            raise GameweekEvaluationError(
                "Frozen scoreline evidence missing Top-1 scoreline for fixture=%s." % fixture_id
            )
        top1 += int(actual_score == tops[0])
        if available_top_count >= 3:
            top3 += int(actual_score in [value for value in tops[:3] if value is not None])
        if available_top_count >= 5:
            top5 += int(actual_score in [value for value in tops[:5] if value is not None])
        per_match.append(
            {
                "fixture_id": fixture_id,
                "home_team": str(first_value(row, ("home_team_short_name", "home_team_name", "home_team")) or ""),
                "away_team": str(first_value(row, ("away_team_short_name", "away_team_name", "away_team")) or ""),
                "actual_home_goals": home_goals,
                "actual_away_goals": away_goals,
                "actual_result": label,
                "predicted_result": predicted_label,
                "home_win_probability": probs["home_win"],
                "draw_probability": probs["draw"],
                "away_win_probability": probs["away_win"],
                "expected_home_goals": eh,
                "expected_away_goals": ea,
                "actual_scoreline": actual_score,
                "top_1_scoreline": tops[0],
                "top_2_scoreline": tops[1],
                "top_3_scoreline": tops[2],
                "top_4_scoreline": tops[3],
                "top_5_scoreline": tops[4],
                "scoreline_rank_count_available": available_top_count,
                "top1_hit": actual_score == tops[0],
                "top3_hit": (actual_score in [value for value in tops[:3] if value is not None]) if available_top_count >= 3 else None,
                "top5_hit": (actual_score in [value for value in tops[:5] if value is not None]) if available_top_count >= 5 else None,
            }
        )
    n = len(per_match)
    result = {
        "frozen_model_name": str(first_value(match_rows[0], ("model_name", "model_version", "prediction_source")) or ""),
        "n": n,
        "one_x_two_accuracy": correct / n,
        "log_loss": mean(log_losses),
        "multiclass_brier": mean(briers),
        "home_goals_mae": mean(home_errors),
        "away_goals_mae": mean(away_errors),
        "total_goals_mae": mean(total_errors),
        "exact_score_top1_accuracy": top1 / n,
        "exact_score_top3_accuracy": (
            top3 / n
            if all(int(row.get("scoreline_rank_count_available") or 0) >= 3 for row in per_match)
            else None
        ),
        "exact_score_top5_accuracy": (
            top5 / n
            if all(int(row.get("scoreline_rank_count_available") or 0) >= 5 for row in per_match)
            else None
        ),
        "probability_heads_reconciled": False,
        "scoreline_head_reported_separately": True,
    }
    return result, per_match


def _player_or_fixture_id(row: Mapping[str, Any], fixture: bool = False) -> int:
    names = ("fpl_fixture_id", "fixture_id", "id") if fixture else ("fpl_player_id", "player_id", "id")
    value = first_value(row, names)
    if value is None:
        raise GameweekEvaluationError("Row missing %s identity: %s" % ("fixture" if fixture else "player", row))
    return as_int(value)


def _extract_player_list(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for key in ("players", "squad", "owned_players"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    primary = payload.get("primary")
    if isinstance(primary, Mapping) and isinstance(primary.get("players"), list):
        return [row for row in primary["players"] if isinstance(row, Mapping)]
    raise GameweekEvaluationError("Frozen squad state has no player list.")


def _resolve_id(value: Any, by_id: Mapping[int, Mapping[str, Any]], by_name: Mapping[str, int]) -> int:
    if isinstance(value, Mapping):
        raw = first_value(value, ("fpl_player_id", "player_id", "element", "id"))
        if raw is not None:
            return as_int(raw)
        name = first_value(value, ("web_name", "player_name", "name"))
        if name is not None and normalize_name(name) in by_name:
            return by_name[normalize_name(name)]
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        return as_int(value)
    key = normalize_name(value)
    if key and key in by_name:
        return by_name[key]
    raise GameweekEvaluationError("Could not resolve frozen player identity: %r" % value)


def _selection_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("selection", "lineup"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def _extract_starting_ids(selection: Mapping[str, Any], by_id: Mapping[int, Mapping[str, Any]], by_name: Mapping[str, int]) -> List[int]:
    for key in ("starting_xi_player_ids", "starting_player_ids", "starting_xi", "starters"):
        value = selection.get(key)
        if isinstance(value, list):
            return [_resolve_id(item, by_id, by_name) for item in value]
        if isinstance(value, Mapping):
            flattened: List[Any] = []
            for position in POSITIONS:
                part = value.get(position)
                if isinstance(part, list):
                    flattened.extend(part)
            if flattened:
                return [_resolve_id(item, by_id, by_name) for item in flattened]
    raise GameweekEvaluationError("Frozen selection has no starting XI.")


def _extract_bench_ids(selection: Mapping[str, Any], by_id: Mapping[int, Mapping[str, Any]], by_name: Mapping[str, int]) -> List[int]:
    for key in ("bench_order_player_ids", "bench_order", "bench"):
        value = selection.get(key)
        if isinstance(value, list):
            return [_resolve_id(item, by_id, by_name) for item in value]
        if isinstance(value, Mapping):
            ordered: List[Any] = []
            gk = value.get("GKP") or value.get("gkp") or value.get("goalkeeper")
            if gk is not None:
                ordered.append(gk)
            outfield = value.get("outfield")
            if isinstance(outfield, list):
                ordered.extend(outfield)
            if ordered:
                return [_resolve_id(item, by_id, by_name) for item in ordered]
    raise GameweekEvaluationError("Frozen selection has no bench order.")


def _extract_captain_id(selection: Mapping[str, Any], key_names: Sequence[str], by_id: Mapping[int, Mapping[str, Any]], by_name: Mapping[str, int]) -> int:
    value = first_value(selection, key_names)
    if value is None:
        raise GameweekEvaluationError("Frozen selection missing %s." % key_names[0])
    return _resolve_id(value, by_id, by_name)


def _team_name_index(player_by_id: Mapping[int, Mapping[str, Any]]) -> Dict[str, int]:
    candidates: Dict[str, set] = {}
    for pid, row in player_by_id.items():
        aliases: List[str] = []
        for raw_name in (row.get("web_name"), row.get("player_name")):
            if not raw_name:
                continue
            aliases.append(normalize_name(raw_name))
            text = str(raw_name).strip()
            # Support historical screenshot forms such as "M.Sangaré" while
            # remaining fail-closed if the shorter alias is not unique.
            if re.match(r"^[A-Za-z]\s*[.·-]\s*.+", text):
                aliases.append(normalize_name(re.sub(r"^[A-Za-z]\s*[.·-]\s*", "", text)))
        for alias in aliases:
            if alias:
                candidates.setdefault(alias, set()).add(int(pid))
    return {alias: next(iter(ids)) for alias, ids in candidates.items() if len(ids) == 1}


def _read_team_state_payload(path: Path) -> Tuple[Mapping[str, Any], str]:
    if path.suffix.lower() != ".zip":
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            raise GameweekEvaluationError("Frozen team state is not an object: %s" % path)
        return payload, str(path)

    contract = next(
        (value for value in LEGACY_TEAM_ALEX_ARCHIVE_CONTRACTS.values() if Path(value["relative_path"]).name == path.name),
        None,
    )
    if not contract:
        raise GameweekEvaluationError("Unapproved Team Alex ZIP artifact: %s" % path)
    member = contract["member"]
    try:
        with zipfile.ZipFile(str(path), "r") as archive:
            raw = archive.read(member)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise GameweekEvaluationError("Could not read Team Alex ZIP member: %s" % member) from exc
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise GameweekEvaluationError("Frozen Team Alex ZIP member is not an object.")
    return payload, "%s::%s" % (path, member)


def _team_squad_ids(
    payload: Mapping[str, Any],
    player_by_id: Mapping[int, Mapping[str, Any]],
    by_name: Mapping[str, int],
) -> List[int]:
    try:
        players = _extract_player_list(payload)
    except GameweekEvaluationError:
        starters = payload.get("starting_xi")
        bench = payload.get("bench")
        if not isinstance(starters, list) or not isinstance(bench, list):
            raise
        players = [row for row in starters + bench if isinstance(row, Mapping)]
    ids = [_resolve_id(row, player_by_id, by_name) for row in players]
    return ids


def build_team_evaluation(
    label: str,
    state_path: Path,
    player_by_id: Mapping[int, Dict[str, Any]],
    autosub_mode: str,
) -> Dict[str, Any]:
    payload, source_state = _read_team_state_payload(state_path)
    by_name = _team_name_index(player_by_id)
    ids = _team_squad_ids(payload, player_by_id, by_name)
    if len(ids) != 15 or len(set(ids)) != 15:
        raise GameweekEvaluationError("%s frozen squad must contain 15 unique players." % label)
    missing = [pid for pid in ids if pid not in player_by_id]
    if missing:
        raise GameweekEvaluationError("%s squad players missing Player Model/actual rows: %s" % (label, missing))
    selection = _selection_payload(payload)
    starters = _extract_starting_ids(selection, player_by_id, by_name)
    bench = _extract_bench_ids(selection, player_by_id, by_name)
    captain = _extract_captain_id(
        selection,
        ("captain_player_id", "captain", "captain_id"),
        player_by_id,
        by_name,
    )
    vice = _extract_captain_id(
        selection,
        ("vice_captain_player_id", "vice_captain", "vice", "vice_id"),
        player_by_id,
        by_name,
    )
    if len(starters) != 11 or len(bench) != 4 or set(starters + bench) != set(ids):
        raise GameweekEvaluationError("%s frozen XI/bench is not a 15-player partition." % label)
    if captain not in starters or vice not in starters:
        raise GameweekEvaluationError("%s captain/vice must be in frozen XI." % label)
    squad_rows = [dict(player_by_id[pid]) for pid in ids]
    starter_rows = [dict(player_by_id[pid]) for pid in starters]
    bench_rows = [dict(player_by_id[pid]) for pid in bench]
    result = evaluate_team(label, squad_rows, starter_rows, bench_rows, captain, vice, autosub_mode)
    result["source_state"] = source_state
    if state_path.suffix.lower() == ".zip":
        result["source_archive_sha256"] = sha256_file(state_path)
        result["source_archive_immutable"] = True
    return result


def _load_team_alex_scoreable_state(path: Path) -> Path:
    if path.suffix.lower() == ".zip":
        return path
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise GameweekEvaluationError("Frozen Team Alex artifact is not an object.")
    # If this is a wrapper/reference, follow only an immutable path inside the same frozen tree.
    for key in ("state_path", "snapshot_path", "team_state_path"):
        raw = payload.get(key)
        if raw:
            candidate = (path.parent / str(raw)).resolve()
            if candidate.is_file() and str(candidate).startswith(str(path.parent.resolve())):
                return candidate
    return path


def _lineup_from_plan(plan: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    winner = plan.get("winner")
    if isinstance(winner, Mapping) and isinstance(winner.get("lineup"), Mapping):
        return winner.get("lineup")  # type: ignore
    if isinstance(plan.get("lineup"), Mapping):
        return plan.get("lineup")  # type: ignore
    return None


def _score_frozen_lineup(
    lineup: Mapping[str, Any],
    player_by_id: Mapping[int, Dict[str, Any]],
) -> Dict[str, Any]:
    by_name: Dict[str, int] = {}
    for pid, row in player_by_id.items():
        for name in (row.get("web_name"), row.get("player_name")):
            if name:
                by_name[normalize_name(name)] = int(pid)
    starters = _extract_starting_ids(lineup, player_by_id, by_name)
    if len(starters) != 11 or len(set(starters)) != 11:
        raise GameweekEvaluationError("Frozen transfer counterfactual XI must contain 11 unique players.")
    captain = _extract_captain_id(
        lineup,
        ("captain_player_id", "captain", "captain_id"),
        player_by_id,
        by_name,
    )
    vice = _extract_captain_id(
        lineup,
        ("vice_captain_player_id", "vice_captain", "vice", "vice_id"),
        player_by_id,
        by_name,
    )
    rows = [dict(player_by_id[pid]) for pid in starters]
    cap = captain_bonus(rows, captain, vice)
    raw = sum(row["actual_points"] for row in rows)
    return {
        "starting_player_ids": starters,
        "captain_player_id": captain,
        "vice_captain_player_id": vice,
        "actual_xi_points_raw": raw,
        "captain_result": cap,
        "actual_total_with_captain": raw + cap["bonus_points"],
    }


def _find_frozen_no_transfer_plan(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    for key in ("no_transfer_plan", "roll_option"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    for key in ("plans", "ranked_plans", "evaluated_plans", "options"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for row in value:
            if not isinstance(row, Mapping):
                continue
            action = str(row.get("action") or "").strip().upper().replace(" ", "_")
            if action in {"NO_TRANSFER", "ROLL"}:
                return row
    return None


def build_decision_evaluation(
    decision_path: Optional[Path],
    model_team_result: Mapping[str, Any],
    player_by_id: Mapping[int, Dict[str, Any]],
) -> Dict[str, Any]:
    captain_result = model_team_result.get("primary_captain_result") or {}
    frozen_ids = list(model_team_result.get("frozen_starter_ids") or [])
    best_actual = None
    if frozen_ids:
        best_row = max(
            (player_by_id[as_int(pid)] for pid in frozen_ids),
            key=lambda row: (row["actual_points"], -row["fpl_player_id"]),
        )
        best_actual = {
            "fpl_player_id": best_row["fpl_player_id"],
            "web_name": best_row["web_name"],
            "actual_points": best_row["actual_points"],
        }
    captaincy = {
        "status": "evaluated_from_frozen_xi",
        "captain": model_team_result.get("captain"),
        "vice_captain": model_team_result.get("vice_captain"),
        "effective_captain": captain_result.get("effective_captain"),
        "captain_bonus_actual_points": captain_result.get("bonus_points"),
        "vice_triggered": captain_result.get("vice_triggered"),
        "best_actual_scorer_in_frozen_xi": best_actual,
        "hindsight_captain_gap": (
            None
            if best_actual is None
            else best_actual["actual_points"] - as_int(captain_result.get("bonus_points"), 0)
        ),
    }

    transfer: Dict[str, Any] = {
        "status": "not_available_frozen_decision_artifact_missing",
        "chosen_action": None,
        "predicted_net_gain_vs_no_transfer": None,
        "actual_chosen_total_after_hits": None,
        "actual_no_transfer_total": None,
        "actual_counterfactual_vs_no_transfer": None,
        "counterfactual_status": "not_available",
    }
    if decision_path is not None and decision_path.is_file():
        payload = read_json(decision_path)
        if isinstance(payload, Mapping):
            winner = payload.get("winner") if isinstance(payload.get("winner"), Mapping) else payload
            legacy_rec = (
                payload.get("recommended_action_by_target_gw_objective")
                if isinstance(payload.get("recommended_action_by_target_gw_objective"), Mapping)
                else None
            )
            action_source = legacy_rec if legacy_rec is not None else winner
            action = str(action_source.get("action") or "").strip() if isinstance(action_source, Mapping) else ""
            normalized_action = action.upper().replace(" ", "_")

            transfer_count = as_int(
                first_value(winner, ("transfer_count",)) if isinstance(winner, Mapping) else None,
                0,
            )
            transfers = list(winner.get("transfers") or []) if isinstance(winner, Mapping) else []
            hit_points = as_int(
                first_value(winner, ("transfer_hit_points", "hit_points"))
                if isinstance(winner, Mapping)
                else None,
                0,
            )
            predicted_gain = (
                first_value(winner, ("net_gain_vs_no_transfer", "net_gain_vs_roll"))
                if isinstance(winner, Mapping)
                else None
            )

            if legacy_rec is not None:
                if normalized_action == "TRANSFER" and transfer_count == 0:
                    transfer_count = as_int(legacy_rec.get("transfer_count"), 1)
                if normalized_action == "TRANSFER" and not transfers:
                    outgoing = first_value(legacy_rec, ("out_name", "out_player_name", "out_player_id"))
                    incoming = first_value(legacy_rec, ("in_name", "in_player_name", "in_player_id"))
                    if outgoing is not None or incoming is not None:
                        transfers = [{"out": outgoing, "in": incoming}]
                if predicted_gain is None:
                    predicted_gain = first_value(legacy_rec, ("net_gain_vs_no_transfer", "net_gain_vs_roll"))
                hit_points = as_int(
                    first_value(legacy_rec, ("transfer_hit_points", "hit_points")),
                    hit_points,
                )

            chosen_actual = float(model_team_result.get("primary_frozen_xi_actual_total") or 0.0) - float(hit_points)
            transfer.update(
                {
                    "status": "evaluated_from_frozen_decision",
                    "chosen_action": action or None,
                    "transfer_count": transfer_count,
                    "transfers": transfers,
                    "transfer_hit_points": hit_points,
                    "predicted_net_gain_vs_no_transfer": predicted_gain,
                    "actual_chosen_total_after_hits": chosen_actual,
                }
            )

            if normalized_action in {"NO_TRANSFER", "ROLL"}:
                transfer["counterfactual_status"] = "chosen_action_is_no_transfer"
                transfer["actual_no_transfer_total"] = chosen_actual
                transfer["actual_counterfactual_vs_no_transfer"] = 0.0
            else:
                no_transfer = _find_frozen_no_transfer_plan(payload)
                no_transfer_lineup = _lineup_from_plan(no_transfer) if isinstance(no_transfer, Mapping) else None
                if no_transfer_lineup is not None:
                    counterfactual = _score_frozen_lineup(no_transfer_lineup, player_by_id)
                    no_transfer_total = float(counterfactual["actual_total_with_captain"])
                    transfer["counterfactual_status"] = "evaluated_from_frozen_no_transfer_lineup"
                    transfer["actual_no_transfer_total"] = no_transfer_total
                    transfer["actual_counterfactual_vs_no_transfer"] = chosen_actual - no_transfer_total
                    transfer["no_transfer_frozen_lineup"] = counterfactual
                else:
                    transfer["counterfactual_status"] = "not_available_frozen_counterfactual_lineup_missing"
    return {"transfer_no_transfer": transfer, "captaincy": captaincy}


def build_markdown(result: Mapping[str, Any]) -> str:
    p = result["player_model"]
    m = result["match_model"]
    mt = result["model_team"]
    ta = result["team_alex"]
    decision = result["decision_evaluation"]
    return "\n".join(
        [
            "# FPL GW%s POST Evaluation — FINAL" % result["gw"],
            "",
            "- Season: `%s`" % result["season"],
            "- Leakage-safe: **True**",
            "- Frozen baseline kind: `%s`" % result["frozen_inputs"]["kind"],
            "- Prediction regeneration: **False**",
            "- Post-result PRE reconstruction: **False**",
            "",
            "## Player Model",
            "",
            "- Eligible rows: **%s**" % p["coverage"]["primary_eligible_rows"],
            "- MAE / RMSE / bias: **%.4f / %.4f / %.4f**"
            % (
                p["cohorts"]["all_eligible_players"]["mae"],
                p["cohorts"]["all_eligible_players"]["rmse"],
                p["cohorts"]["all_eligible_players"]["mean_error_bias_pred_minus_actual"],
            ),
            "- Spearman / NDCG@20: **%.4f / %.4f**"
            % (
                p["ranking_quality"]["spearman_rank_correlation"],
                p["ranking_quality"]["ndcg_at_20"],
            ),
            "",
            "## Match Model",
            "",
            "- Fixtures: **%s**" % m["n"],
            "- 1X2 accuracy / log loss / Brier: **%.3f / %.4f / %.4f**"
            % (m["one_x_two_accuracy"], m["log_loss"], m["multiclass_brier"]),
            "- 1X2 and scoreline/xG remain separate heads; this evaluation does not claim probability reconciliation.",
            "",
            "## Model Team",
            "",
            "- Frozen XI + captain actual: **%.1f**" % mt["primary_frozen_xi_actual_total"],
            "- Predicted XI + captain: **%.3f**" % mt["submitted_predicted_total_with_captain"],
            "- Captain: **%s**; vice: **%s**" % (mt["captain"], mt["vice_captain"]),
            "",
            "## Team Alex",
            "",
            "- Frozen XI + captain actual: **%.1f**" % ta["primary_frozen_xi_actual_total"],
            "- Predicted XI + captain: **%.3f**" % ta["submitted_predicted_total_with_captain"],
            "- Captain: **%s**; vice: **%s**" % (ta["captain"], ta["vice_captain"]),
            "",
            "## Decision Evaluation",
            "",
            "- Transfer/no-transfer: `%s`" % decision["transfer_no_transfer"]["status"],
            "- Captaincy: `%s`" % decision["captaincy"]["status"],
            "",
            "## Safety",
            "",
            "- Actuals are captured separately from frozen PRE evidence.",
            "- Frozen PRE hashes/timestamps are validated before scoring.",
            "- A missing FINAL baseline is a hard failure.",
            "- Team Alex is never reconstructed from post-result notes.",
            "",
        ]
    )


def discover_final_evaluation(planning_root: Path, season: str, gw: int) -> Optional[Path]:
    root = evaluation_root(planning_root, season, gw)
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("final_")
        and (path / "evaluation_summary.json").is_file()
        and (path / "evaluation_manifest.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_gameweek_evaluation(
    *,
    planning_root: Path,
    season: str,
    gw: int,
    actual_manifest_path: Path,
    output_dir: Optional[Path] = None,
    resume: bool = False,
) -> Path:
    actual_manifest = validate_final_actual_manifest(actual_manifest_path, season, gw)
    existing = discover_final_evaluation(planning_root, season, gw)
    if resume and existing is not None:
        return existing

    # GW1 is already sealed historical evidence. Never create a second GW1 FINAL
    # evaluation package even when the caller omitted --resume.
    if gw == 1 and existing is not None:
        return existing

    # Preserve the already-sealed GW1 evaluator exactly as historical evidence.
    if gw == 1:
        from ml.eval import evaluate_gameweek_post as legacy

        out_dir = output_dir or (
            evaluation_root(planning_root, season, gw) / ("final_%s" % utc_stamp())
        )
        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "evaluate_gameweek_post",
                "--planning-root",
                str(planning_root),
                "--season",
                season,
                "--gw",
                str(gw),
                "--actual-manifest",
                str(actual_manifest_path),
                "--output-dir",
                str(out_dir),
                "--require-final",
            ]
            rc = legacy.main()
        finally:
            sys.argv = old_argv
        if rc != 0:
            raise GameweekEvaluationError("Legacy GW1 evaluator failed with rc=%s." % rc)
        return Path(out_dir)

    baseline = discover_frozen_baseline(planning_root, season, gw)
    actual_dir = actual_manifest_path.parent
    event_live_path = actual_dir / str(actual_manifest["event_live_file"])
    fixtures_path = actual_dir / str(actual_manifest["fixtures_file"])

    player_result, player_rows, player_by_id = build_player_evaluation(
        baseline.player_predictions,
        event_live_path,
    )
    match_result, match_rows = build_match_evaluation(
        baseline.match_predictions,
        baseline.scoreline_predictions,
        fixtures_path,
    )
    model_team_result = build_team_evaluation(
        "Model Team",
        baseline.model_team_state,
        player_by_id,
        "model_team",
    )
    team_alex_path = _load_team_alex_scoreable_state(baseline.team_alex_state)
    team_alex_result = build_team_evaluation(
        "Team Alex / Gliding Tiger",
        team_alex_path,
        player_by_id,
        "team_alex",
    )
    decision_result = build_decision_evaluation(
        baseline.model_team_decision,
        model_team_result,
        player_by_id,
    )

    out_dir = output_dir or (
        evaluation_root(planning_root, season, gw) / ("final_%s" % utc_stamp())
    )
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists():
        raise GameweekEvaluationError("Evaluation destination already exists: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    result: Dict[str, Any] = {
        "contract_version": POST_EVALUATION_VERSION,
        "season": season,
        "gw": int(gw),
        "evaluation_status": "FINAL",
        "generated_at_utc": utc_now(),
        "leakage_safe": True,
        "database_access": False,
        "prediction_regeneration": False,
        "post_result_pre_reconstruction": False,
        "writes_frozen_pre_artifacts": False,
        "actuals": {
            **dict(actual_manifest),
            "manifest_path": str(actual_manifest_path),
        },
        "frozen_inputs": {
            "kind": baseline.kind,
            "root": str(baseline.root),
            "manifest_path": str(baseline.manifest_path),
            "fingerprint": baseline.fingerprint,
            "as_of_utc": baseline.as_of_utc,
            "deadline_utc": baseline.deadline_utc,
            "as_of_before_deadline": True,
            "player_predictions": str(baseline.player_predictions),
            "match_predictions": str(baseline.match_predictions),
            "scoreline_predictions": str(baseline.scoreline_predictions),
            "model_team_state": str(baseline.model_team_state),
            "team_alex_state": str(baseline.team_alex_state),
            "team_alex_state_sha256": sha256_file(baseline.team_alex_state),
            "model_team_decision": str(baseline.model_team_decision) if baseline.model_team_decision else None,
        },
        "player_model": player_result,
        "match_model": match_result,
        "model_team": model_team_result,
        "team_alex": team_alex_result,
        "decision_evaluation": decision_result,
    }
    safe_result = json_safe(result)
    summary_json = write_json_new(out_dir / "evaluation_summary.json", safe_result)
    (out_dir / "evaluation_summary.md").write_text(build_markdown(result), encoding="utf-8")
    write_csv(
        out_dir / "player_evaluation_rows.csv",
        player_rows,
        (
            "fpl_player_id",
            "player_name",
            "web_name",
            "position",
            "team_short_name",
            "selection_eligible",
            "prediction_available",
            "predicted_points",
            "actual_points",
            "expected_minutes",
            "actual_minutes",
            "appearance_probability",
            "actual_appearance",
            "start_probability",
            "actual_start",
            "actual_starts",
        ),
    )
    write_csv(
        out_dir / "match_evaluation_rows.csv",
        match_rows,
        (
            "fixture_id",
            "home_team",
            "away_team",
            "actual_home_goals",
            "actual_away_goals",
            "actual_result",
            "predicted_result",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "expected_home_goals",
            "expected_away_goals",
            "actual_scoreline",
            "top_1_scoreline",
            "top_2_scoreline",
            "top_3_scoreline",
            "top_4_scoreline",
            "top_5_scoreline",
            "scoreline_rank_count_available",
            "top1_hit",
            "top3_hit",
            "top5_hit",
        ),
    )
    write_json_new(out_dir / "model_team_evaluation.json", json_safe(model_team_result))
    write_json_new(out_dir / "team_alex_evaluation.json", json_safe(team_alex_result))
    write_json_new(out_dir / "decision_evaluation.json", json_safe(decision_result))

    artifacts = []
    for path in sorted(p for p in out_dir.iterdir() if p.is_file()):
        if path.name == "evaluation_manifest.json":
            continue
        artifacts.append(
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "contract_version": POST_MANIFEST_VERSION,
        "season": season,
        "gw": int(gw),
        "evaluation_status": "FINAL",
        "output_dir": str(out_dir),
        "actual_manifest": str(actual_manifest_path),
        "actual_manifest_sha256": sha256_file(actual_manifest_path),
        "frozen_manifest": str(baseline.manifest_path),
        "frozen_baseline_fingerprint": baseline.fingerprint,
        "leakage_safe": True,
        "prediction_regeneration": False,
        "post_result_pre_reconstruction": False,
        "immutable": True,
        "artifacts": artifacts,
    }
    write_json_new(out_dir / "evaluation_manifest.json", manifest)

    print("=== GW%d POST Evaluation ===" % gw)
    print("status: FINAL")
    print("output_dir:", out_dir)
    print("frozen_baseline_kind:", baseline.kind)
    print("player_mae:", round(player_result["cohorts"]["all_eligible_players"]["mae"], 6))
    print("match_1x2_accuracy:", round(match_result["one_x_two_accuracy"], 6))
    print("model_team_primary_frozen_xi_total:", model_team_result["primary_frozen_xi_actual_total"])
    print("team_alex_primary_frozen_xi_total:", team_alex_result["primary_frozen_xi_actual_total"])
    print("leakage_safe: True")
    print("prediction_regeneration: False")
    print("post_result_pre_reconstruction: False")
    _ = summary_json
    return out_dir


def main() -> int:
    args = parse_args()
    repo_root = detect_repo_root(args.repo_root)
    planning_root = detect_planning_root(repo_root, args.planning_root)

    actual_manifest_path: Optional[Path] = None
    if args.actual_manifest:
        actual_manifest_path = Path(args.actual_manifest).expanduser().resolve()
        validate_final_actual_manifest(actual_manifest_path, args.season, args.gw)
    elif args.mode in ("full", "capture"):
        actual_manifest_path = capture_or_reuse_final_actuals(
            planning_root=planning_root,
            season=args.season,
            gw=args.gw,
            reuse_existing=bool(args.resume),
            reuse_only=bool(args.reuse_final_actuals_only),
        )
    else:
        actual_manifest_path = discover_final_actual_manifest(planning_root, args.season, args.gw)
        if actual_manifest_path is None:
            raise GameweekEvaluationError("No FINAL actual manifest available for evaluate mode.")

    print("actual_manifest:", actual_manifest_path)
    if args.mode == "capture":
        return 0

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    result = run_gameweek_evaluation(
        planning_root=planning_root,
        season=args.season,
        gw=args.gw,
        actual_manifest_path=actual_manifest_path,
        output_dir=output_dir,
        resume=bool(args.resume),
    )
    print("evaluation_dir:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
