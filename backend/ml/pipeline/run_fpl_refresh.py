from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


RUNNER_VERSION = "fpl_weekly_runner_v0_1"
POSITIONS = ("GKP", "DEF", "MID", "FWD")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Canonical one-command FPL weekly runner. It orchestrates existing "
            "ingest/prediction/evaluation/decision stages and writes one candidate package."
        )
    )
    p.add_argument("--phase", required=True, choices=["pre", "post", "auto", "status", "freeze"])
    p.add_argument("--season", required=True)
    p.add_argument("--target-gw", type=int, required=True)
    p.add_argument("--prior-season", default=None)
    p.add_argument("--stabilization-gw", type=int, default=6)
    p.add_argument("--planning-root", default=None)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--publish-predictions",
        action="store_true",
        help="Explicitly allow the existing early-season publisher to write prediction tables.",
    )
    p.add_argument(
        "--skip-live-refresh",
        action="store_true",
        help="Do not call FPL ingest endpoints. Useful for an offline/resume-only package.",
    )
    p.add_argument(
        "--base-url",
        default="",
        help="Optional already-running backend URL. If omitted, a private local uvicorn is started automatically.",
    )
    p.add_argument("--api-port", type=int, default=8765)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--position-top-n", type=int, default=10)
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def detect_repo_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / ".git").exists():
            raise RuntimeError("repo-root does not contain .git: %s" % root)
        return root

    here = Path(__file__).resolve()
    for parent in (here.parent,) + tuple(here.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not auto-detect repository root.")


def detect_planning_root(repo_root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return repo_root.parent / "private-planning"


def load_dotenv_value(path: Path, key: str) -> Optional[str]:
    if not path.is_file():
        return None
    prefix = key + "="
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def child_env(repo_root: Path, season: str) -> Dict[str, str]:
    env = os.environ.copy()
    if not env.get("DATABASE_URL"):
        value = load_dotenv_value(repo_root / "backend" / ".env", "DATABASE_URL")
        if value:
            env["DATABASE_URL"] = value
    env["FPL_SEASON"] = season
    return env


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("JSON root must be an object: %s" % path)
    return payload


def json_contains_string(payload: Any, needle: str) -> bool:
    if isinstance(payload, dict):
        return any(json_contains_string(v, needle) for v in payload.values())
    if isinstance(payload, list):
        return any(json_contains_string(v, needle) for v in payload)
    return needle in str(payload)


def latest_path(paths: Iterable[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


class StageRecorder:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(
        self,
        name: str,
        status: str,
        started: float,
        outputs: Optional[Sequence[str]] = None,
        note: str = "",
    ) -> None:
        self.rows.append(
            {
                "stage": name,
                "status": status,
                "elapsed_seconds": round(time.time() - started, 3),
                "outputs": list(outputs or []),
                "note": note,
            }
        )


def run_command(
    name: str,
    parts: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    recorder: StageRecorder,
    dry_run: bool,
) -> str:
    started = time.time()
    printable = " ".join(parts)
    print("\n[%s] %s" % (name, printable))
    if dry_run:
        recorder.add(name, "DRY_RUN", started, note=printable)
        return ""

    proc = subprocess.run(
        list(parts),
        cwd=str(cwd),
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    if proc.returncode != 0:
        recorder.add(name, "FAILED", started, note="exit=%s" % proc.returncode)
        raise RuntimeError(
            "%s failed with exit code %s\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (name, proc.returncode, proc.stdout, proc.stderr)
        )
    recorder.add(name, "PASS", started)
    return proc.stdout


def http_json(method: str, url: str, timeout: float = 90.0) -> Dict[str, Any]:
    req = Request(url=url, method=method)
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    if not raw:
        return {}
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {"payload": payload}


def wait_for_api(base_url: str, season: str, timeout_seconds: float = 25.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            payload = http_json("GET", base_url + "/gameweeks/current", timeout=2.0)
            returned = payload.get("season")
            if returned and str(returned) != str(season):
                raise RuntimeError(
                    "Backend season mismatch: expected %s, got %s." % (season, returned)
                )
            return True
        except RuntimeError:
            raise
        except Exception:
            time.sleep(0.5)
    return False


def start_private_api(
    backend_dir: Path,
    python_exe: str,
    env: Mapping[str, str],
    port: int,
    season: str,
) -> Tuple[subprocess.Popen, str]:
    base_url = "http://127.0.0.1:%s" % int(port)
    proc = subprocess.Popen(
        [
            python_exe,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(backend_dir),
        env=dict(env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if not wait_for_api(base_url, season, timeout_seconds=25.0):
        proc.terminate()
        raise RuntimeError("Private backend did not become ready at %s." % base_url)
    return proc, base_url


def refresh_live_data(
    base_url: str,
    recorder: StageRecorder,
    dry_run: bool,
) -> None:
    endpoints = [
        ("ingest_gameweeks", "/gameweeks/ingest/fpl"),
        ("ingest_bootstrap", "/ingest/fpl/bootstrap"),
        ("ingest_fixtures", "/ingest/fpl/fixtures"),
        ("ingest_finished_gw_stats", "/ingest/fpl/gw/finished"),
    ]
    for name, suffix in endpoints:
        started = time.time()
        url = base_url + suffix
        print("\n[%s] POST %s" % (name, url))
        if dry_run:
            recorder.add(name, "DRY_RUN", started, note=url)
            continue
        try:
            payload = http_json("POST", url)
        except Exception as exc:
            recorder.add(name, "FAILED", started, note=str(exc))
            raise
        recorder.add(name, "PASS", started, note=json.dumps(payload, default=str)[:500])


def prediction_root(planning_root: Path, season: str, gw: int) -> Path:
    return planning_root / "gw-pre" / season / ("gw%02d" % gw) / "early-season"


def discover_prediction_run(
    planning_root: Path,
    season: str,
    gw: int,
) -> Optional[Path]:
    base = prediction_root(planning_root, season, gw)
    if not base.is_dir():
        return None
    candidates = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if (p / "player_predictions_preview.csv").is_file() and (p / "run_manifest.json").is_file():
            candidates.append(p)
    return latest_path(candidates)


def run_prediction_stage(
    repo_root: Path,
    planning_root: Path,
    season: str,
    prior_season: str,
    gw: int,
    stabilization_gw: int,
    python_exe: str,
    env: Mapping[str, str],
    recorder: StageRecorder,
    resume: bool,
    dry_run: bool,
) -> Path:
    if not (2 <= gw < stabilization_gw):
        raise RuntimeError(
            "Runner v0.1 currently wires the safe early-season PRE path only "
            "(2 <= target_gw < %s). Refusing to fall back to legacy baseline models."
            % stabilization_gw
        )

    existing = discover_prediction_run(planning_root, season, gw)
    if resume and existing is not None:
        started = time.time()
        recorder.add(
            "early_season_predictions",
            "REUSED",
            started,
            outputs=[str(existing)],
            note="--resume reused latest complete preview artifact.",
        )
        print("\n[early_season_predictions] REUSED %s" % existing)
        return existing

    before = set()
    base = prediction_root(planning_root, season, gw)
    if base.is_dir():
        before = {p.resolve() for p in base.iterdir() if p.is_dir()}

    run_command(
        "early_season_predictions",
        [
            python_exe,
            "-m",
            "ml.validation.refresh_early_season_predictions",
            "--season",
            season,
            "--prior-season",
            prior_season,
            "--target-gw",
            str(gw),
            "--stabilization-gw",
            str(stabilization_gw),
        ],
        repo_root / "backend",
        env,
        recorder,
        dry_run,
    )

    if dry_run:
        return base / "<new_prediction_run>"

    after = {p.resolve() for p in base.iterdir() if p.is_dir()}
    new_dirs = [
        p for p in (after - before)
        if (p / "player_predictions_preview.csv").is_file()
    ]
    result = latest_path(new_dirs) or discover_prediction_run(planning_root, season, gw)
    if result is None:
        raise RuntimeError("Prediction command passed but no prediction run directory was found.")
    return result


def module_help(
    python_exe: str,
    module: str,
    backend_dir: Path,
    env: Mapping[str, str],
) -> str:
    proc = subprocess.run(
        [python_exe, "-m", module, "--help"],
        cwd=str(backend_dir),
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def supported_flags(help_text: str) -> List[str]:
    return sorted(set(re.findall(r"(?<!\w)(--[a-zA-Z0-9][a-zA-Z0-9_-]*)", help_text)))


def first_supported(flags: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    available = set(flags)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def latest_publish_receipt(
    planning_root: Path,
    season: str,
    gw: int,
    source_run_id: str,
) -> Optional[Path]:
    base = planning_root / "gw-pre" / season / ("gw%02d" % gw) / "publish-receipts"
    if not base.is_dir():
        return None
    matches: List[Path] = []
    for path in base.glob("*.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if json_contains_string(payload, source_run_id):
            matches.append(path)
    return latest_path(matches)


def publish_predictions_if_requested(
    repo_root: Path,
    planning_root: Path,
    season: str,
    gw: int,
    prediction_run: Path,
    python_exe: str,
    env: Mapping[str, str],
    recorder: StageRecorder,
    resume: bool,
    dry_run: bool,
) -> Optional[Path]:
    run_id = prediction_run.name
    existing = latest_publish_receipt(planning_root, season, gw, run_id)
    if resume and existing is not None:
        started = time.time()
        recorder.add(
            "publish_predictions",
            "REUSED",
            started,
            outputs=[str(existing)],
            note="Existing publish receipt matches source run.",
        )
        print("\n[publish_predictions] REUSED %s" % existing)
        return existing

    module = "ml.validation.publish_early_season_predictions"
    help_text = module_help(python_exe, module, repo_root / "backend", env)
    flags = supported_flags(help_text)
    source_flag = first_supported(
        flags,
        [
            "--source-run-dir",
            "--run-dir",
            "--preview-run-dir",
            "--source-dir",
            "--prediction-run-dir",
        ],
    )
    if source_flag is None:
        raise RuntimeError(
            "Publisher exists but runner could not resolve its source-run option. "
            "Supported flags were: %s" % flags
        )

    cmd = [python_exe, "-m", module, source_flag, str(prediction_run)]
    for flag, value in (
        ("--season", season),
        ("--target-gw", str(gw)),
    ):
        if flag in flags:
            cmd.extend([flag, value])
    if "--publish" in flags:
        cmd.append("--publish")
    if "--replace-existing" in flags:
        cmd.append("--replace-existing")

    run_command(
        "publish_predictions",
        cmd,
        repo_root / "backend",
        env,
        recorder,
        dry_run,
    )
    if dry_run:
        return None

    receipt = latest_publish_receipt(planning_root, season, gw, run_id)
    if receipt is None:
        raise RuntimeError("Publisher passed but no matching publish receipt was found.")
    return receipt


def verify_published_predictions(
    repo_root: Path,
    season: str,
    gw: int,
    prediction_run: Path,
    receipt: Path,
    python_exe: str,
    env: Mapping[str, str],
    recorder: StageRecorder,
    dry_run: bool,
) -> None:
    module = "ml.validation.verify_early_season_publish"
    help_text = module_help(python_exe, module, repo_root / "backend", env)
    flags = supported_flags(help_text)
    source_flag = first_supported(
        flags,
        [
            "--source-run-dir",
            "--run-dir",
            "--preview-run-dir",
            "--source-dir",
            "--prediction-run-dir",
        ],
    )
    if source_flag is None:
        raise RuntimeError(
            "Post-publish verifier exists but runner could not resolve its source-run option. "
            "Supported flags were: %s" % flags
        )
    cmd = [python_exe, "-m", module, source_flag, str(prediction_run)]
    if "--receipt" not in flags:
        raise RuntimeError(
            "Post-publish verifier does not expose required --receipt option. "
            "Supported flags were: %s" % flags
        )
    cmd.extend(["--receipt", str(receipt)])
    for flag, value in (
        ("--season", season),
        ("--target-gw", str(gw)),
    ):
        if flag in flags:
            cmd.extend([flag, value])
    run_command(
        "verify_published_predictions",
        cmd,
        repo_root / "backend",
        env,
        recorder,
        dry_run,
    )


def build_market_csv(prediction_csv: Path, out_csv: Path) -> Path:
    df = pd.read_csv(prediction_csv)
    required = [
        "player_id",
        "fpl_player_id",
        "web_name",
        "position",
        "team_id",
        "team_name",
        "team_short_name",
        "now_cost",
        "status",
        "predicted_points",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Prediction CSV cannot form market; missing %s." % missing)
    market = df[required].copy()
    if market["fpl_player_id"].duplicated().any():
        raise RuntimeError("Duplicate fpl_player_id rows in market source.")
    if market["predicted_points"].isna().any():
        raise RuntimeError("Market source has missing predicted_points.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    market.to_csv(out_csv, index=False)
    return out_csv


def build_top10_by_position(
    prediction_csv: Path,
    out_dir: Path,
    top_n: int,
) -> Tuple[Path, Path, Path, Dict[str, List[Dict[str, Any]]]]:
    df = pd.read_csv(prediction_csv)
    required = {"position", "predicted_points", "fpl_player_id", "web_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("Prediction CSV missing Top-N fields: %s" % missing)

    rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for pos in POSITIONS:
        part = (
            df[df["position"].astype(str).str.upper() == pos]
            .sort_values(["predicted_points", "fpl_player_id"], ascending=[False, True])
            .head(int(top_n))
        )
        grouped[pos] = []
        for rank, raw in enumerate(part.to_dict(orient="records"), start=1):
            item = {
                "rank": rank,
                "position": pos,
                "fpl_player_id": int(raw["fpl_player_id"]),
                "web_name": str(raw["web_name"]),
                "team_name": str(raw.get("team_name") or ""),
                "team_short_name": str(raw.get("team_short_name") or ""),
                "now_cost": int(raw.get("now_cost") or 0),
                "status": str(raw.get("status") or ""),
                "chance_of_playing_next_round": raw.get("chance_of_playing_next_round"),
                "predicted_points": float(raw["predicted_points"]),
                "expected_minutes_total": raw.get("expected_minutes_total"),
                "blended_start_probability": raw.get("blended_start_probability"),
                "blended_appearance_probability": raw.get("blended_appearance_probability"),
                "news": str(raw.get("news") or ""),
            }
            grouped[pos].append(item)
            rows.append(item)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "top10_by_position.csv"
    json_path = out_dir / "top10_by_position.json"
    md_path = out_dir / "top10_by_position.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(grouped, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = ["# Player Prediction Top %s by Position" % top_n, ""]
    for pos in POSITIONS:
        lines.extend(
            [
                "## %s" % pos,
                "",
                "| Rank | Player | Team | Price | Status | Pred |",
                "|---:|---|---|---:|---|---:|",
            ]
        )
        for item in grouped[pos]:
            lines.append(
                "| %s | %s | %s | %.1f | %s | %.3f |"
                % (
                    item["rank"],
                    item["web_name"],
                    item["team_short_name"],
                    item["now_cost"] / 10.0,
                    item["status"],
                    item["predicted_points"],
                )
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path, grouped


def find_previous_squad_json(
    planning_root: Path,
    season: str,
    target_gw: int,
) -> Path:
    previous_gw = target_gw - 1
    if previous_gw < 1:
        raise RuntimeError("No previous squad exists before GW1.")

    frozen = planning_root / "frozen-snapshots" / season / ("gw%02d" % previous_gw)
    if frozen.is_dir():
        exact = list(frozen.rglob("gw1_opening_squad.json"))
        if exact:
            preferred = [p for p in exact if "model-team" in str(p)]
            return latest_path(preferred or exact)  # type: ignore

        candidates = []
        for path in frozen.rglob("*.json"):
            try:
                payload = read_json(path)
            except Exception:
                continue
            primary = payload.get("primary")
            if isinstance(primary, dict) and isinstance(primary.get("players"), list):
                if len(primary["players"]) == 15:
                    candidates.append(path)
            if isinstance(payload.get("squad"), list) and len(payload["squad"]) == 15:
                candidates.append(path)
        chosen = latest_path(candidates)
        if chosen is not None:
            return chosen

    # Current GW2 compatibility fallback: the previously validated working copy.
    if target_gw == 2:
        temp = Path(
            "/private/tmp/gw2_transfer_state_inputs/gw01_frozen/"
            "model-team/day101c-final-run/gw1_opening_squad.json"
        )
        if temp.is_file():
            return temp

    raise RuntimeError(
        "Could not auto-discover previous Model Team squad for season=%s target_gw=%s."
        % (season, target_gw)
    )


def latest_previous_transfer_state(
    planning_root: Path,
    season: str,
    target_gw: int,
) -> Optional[Path]:
    previous_gw = target_gw - 1
    if previous_gw < 2:
        return None
    base = (
        planning_root
        / "gw-pre"
        / season
        / ("gw%02d" % previous_gw)
        / "transfer-decision"
    )
    if not base.is_dir():
        return None
    return latest_path(base.rglob("next_gameweek_transfer_state_preview.json"))


def transfer_run_matches_prediction(run_dir: Path, prediction_run: Path) -> bool:
    manifest = run_dir / "run_manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = read_json(manifest)
    except Exception:
        return False
    return (
        json_contains_string(payload, prediction_run.name)
        or json_contains_string(payload, str(prediction_run / "player_predictions_preview.csv"))
    )


def discover_transfer_run(
    planning_root: Path,
    season: str,
    gw: int,
    prediction_run: Path,
) -> Optional[Path]:
    base = planning_root / "gw-pre" / season / ("gw%02d" % gw) / "transfer-decision"
    if not base.is_dir():
        return None
    candidates = [
        p for p in base.iterdir()
        if p.is_dir()
        and (p / "transfer_decision_preview.json").is_file()
        and (p / "next_gameweek_transfer_state_preview.json").is_file()
        and transfer_run_matches_prediction(p, prediction_run)
    ]
    return latest_path(candidates)


def run_transfer_stage(
    repo_root: Path,
    planning_root: Path,
    season: str,
    gw: int,
    prediction_run: Path,
    market_csv: Path,
    python_exe: str,
    env: Mapping[str, str],
    recorder: StageRecorder,
    resume: bool,
    dry_run: bool,
    top_n: int,
) -> Path:
    existing = discover_transfer_run(planning_root, season, gw, prediction_run)
    if resume and existing is not None:
        started = time.time()
        recorder.add(
            "weekly_transfer_optimizer",
            "REUSED",
            started,
            outputs=[str(existing)],
            note="Existing transfer run matches prediction lineage.",
        )
        print("\n[weekly_transfer_optimizer] REUSED %s" % existing)
        return existing

    previous_squad = find_previous_squad_json(planning_root, season, gw)
    previous_state = latest_previous_transfer_state(planning_root, season, gw)
    prediction_csv = prediction_run / "player_predictions_preview.csv"
    prediction_manifest = prediction_run / "run_manifest.json"

    cmd = [
        python_exe,
        "-m",
        "ml.decision.optimize_weekly_transfers",
        "--season",
        season,
        "--target-gw",
        str(gw),
        "--previous-squad-json",
        str(previous_squad),
        "--market-csv",
        str(market_csv),
        "--prediction-csv",
        str(prediction_csv),
        "--prediction-manifest",
        str(prediction_manifest),
        "--top-n",
        str(top_n),
    ]
    if previous_state is not None:
        cmd.extend(["--previous-transfer-state-json", str(previous_state)])

    before: set = set()
    base = planning_root / "gw-pre" / season / ("gw%02d" % gw) / "transfer-decision"
    if base.is_dir():
        before = {p.resolve() for p in base.iterdir() if p.is_dir()}

    run_command(
        "weekly_transfer_optimizer",
        cmd,
        repo_root / "backend",
        env,
        recorder,
        dry_run,
    )
    if dry_run:
        return base / "<new_transfer_run>"

    after = {p.resolve() for p in base.iterdir() if p.is_dir()}
    new_dirs = [
        p for p in (after - before)
        if (p / "transfer_decision_preview.json").is_file()
    ]
    chosen = latest_path(new_dirs) or discover_transfer_run(
        planning_root, season, gw, prediction_run
    )
    if chosen is None:
        raise RuntimeError("Transfer optimizer passed but no output run was discovered.")
    return chosen


def run_lineup_stage(
    repo_root: Path,
    season: str,
    gw: int,
    prediction_run: Path,
    transfer_run: Path,
    out_dir: Path,
    python_exe: str,
    env: Mapping[str, str],
    recorder: StageRecorder,
    resume: bool,
    dry_run: bool,
) -> Optional[Path]:
    target = out_dir / "weekly_lineup_preview.json"
    if resume and target.is_file():
        started = time.time()
        recorder.add(
            "weekly_lineup_preview",
            "REUSED",
            started,
            outputs=[str(target)],
        )
        return target

    state_json = transfer_run / "next_gameweek_transfer_state_preview.json"
    if not state_json.is_file() and not dry_run:
        raise RuntimeError("Transfer run is missing next_gameweek_transfer_state_preview.json.")

    run_command(
        "weekly_lineup_preview",
        [
            python_exe,
            "-m",
            "ml.pipeline.weekly_lineup_preview",
            "--season",
            season,
            "--target-gw",
            str(gw),
            "--transfer-state-json",
            str(state_json),
            "--prediction-csv",
            str(prediction_run / "player_predictions_preview.csv"),
            "--out-dir",
            str(out_dir),
        ],
        repo_root / "backend",
        env,
        recorder,
        dry_run,
    )
    return None if dry_run else target


def discover_final_actual_manifest(
    planning_root: Path,
    season: str,
    gw: int,
) -> Optional[Path]:
    base = planning_root / "gw-post" / season / ("gw%02d" % gw) / "actuals"
    if not base.is_dir():
        return None
    finals = list(base.glob("*FINAL*.json"))
    return latest_path(finals)


def discover_final_evaluation(
    planning_root: Path,
    season: str,
    gw: int,
) -> Optional[Path]:
    base = planning_root / "gw-post" / season / ("gw%02d" % gw) / "evaluation"
    if not base.is_dir():
        return None
    dirs = [
        p for p in base.iterdir()
        if p.is_dir() and p.name.startswith("final_") and (p / "evaluation_summary.json").is_file()
    ]
    return latest_path(dirs)


def run_post_evaluation(
    repo_root: Path,
    planning_root: Path,
    season: str,
    gw: int,
    python_exe: str,
    env: Mapping[str, str],
    recorder: StageRecorder,
    resume: bool,
    dry_run: bool,
) -> Optional[Path]:
    existing = discover_final_evaluation(planning_root, season, gw)
    if resume and existing is not None:
        started = time.time()
        recorder.add(
            "post_evaluation",
            "REUSED",
            started,
            outputs=[str(existing)],
            note="Existing FINAL evaluation reused.",
        )
        print("\n[post_evaluation] REUSED %s" % existing)
        return existing

    actual_manifest = discover_final_actual_manifest(planning_root, season, gw)
    if actual_manifest is None:
        print("\n[post_evaluation] SKIPPED: no FINAL actual manifest for GW%s." % gw)
        recorder.add(
            "post_evaluation",
            "SKIPPED",
            time.time(),
            note="No FINAL actual manifest available.",
        )
        return None

    output_dir = (
        planning_root
        / "gw-post"
        / season
        / ("gw%02d" % gw)
        / "evaluation"
        / ("final_%s" % utc_stamp())
    )
    run_command(
        "post_evaluation",
        [
            python_exe,
            "-m",
            "ml.eval.evaluate_gameweek_post",
            "--repo-root",
            str(repo_root),
            "--planning-root",
            str(planning_root),
            "--season",
            season,
            "--gw",
            str(gw),
            "--actual-manifest",
            str(actual_manifest),
            "--output-dir",
            str(output_dir),
            "--require-final",
        ],
        repo_root / "backend",
        env,
        recorder,
        dry_run,
    )
    return None if dry_run else output_dir


def extract_transfer_summary(transfer_run: Path) -> Dict[str, Any]:
    path = transfer_run / "transfer_decision_preview.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    rec = payload.get("recommended_action_by_target_gw_objective") or {}
    roll = payload.get("roll_option") or {}
    return {
        "recommended_action": (
            "%s -> %s" % (rec.get("out_name"), rec.get("in_name"))
            if rec.get("action") == "TRANSFER"
            else "ROLL"
        ),
        "net_gain_vs_roll": rec.get("net_gain_vs_roll"),
        "bank_after_units": rec.get("bank_after_units"),
        "free_transfers_next_if_recommendation": rec.get("free_transfers_next_gameweek"),
        "free_transfers_next_if_roll": roll.get("free_transfers_next_gameweek"),
        "formation_objective_only": rec.get("formation"),
        "captain_objective_only": rec.get("captain_name"),
        "future_ft_option_value_monetized": payload.get("future_ft_option_value_monetized"),
        "final_weekly_transfer_decision": payload.get("final_weekly_transfer_decision"),
    }


def extract_evaluation_summary(eval_dir: Optional[Path]) -> Dict[str, Any]:
    if eval_dir is None:
        return {}
    path = eval_dir / "evaluation_summary.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    return payload


def candidate_package_dir(
    planning_root: Path,
    season: str,
    gw: int,
) -> Path:
    return (
        planning_root
        / "gw-pre"
        / season
        / ("gw%02d" % gw)
        / "candidate-package"
        / ("%s_%s_gw%s_%s" % (RUNNER_VERSION, season, gw, utc_stamp()))
    )


def find_reusable_candidate_package(
    planning_root: Path,
    season: str,
    gw: int,
) -> Optional[Path]:
    base = planning_root / "gw-pre" / season / ("gw%02d" % gw) / "candidate-package"
    if not base.is_dir():
        return None
    dirs = [
        p for p in base.iterdir()
        if p.is_dir() and (p / "run_manifest.json").is_file()
    ]
    return latest_path(dirs)


def write_candidate_package(
    out_dir: Path,
    season: str,
    target_gw: int,
    prior_season: str,
    prediction_run: Path,
    transfer_run: Path,
    lineup_json: Optional[Path],
    top_paths: Sequence[Path],
    grouped_top: Mapping[str, Sequence[Mapping[str, Any]]],
    previous_eval: Optional[Path],
    recorder: StageRecorder,
    publish_requested: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    transfer_summary = extract_transfer_summary(transfer_run)
    lineup = read_json(lineup_json) if lineup_json is not None and lineup_json.is_file() else {}
    evaluation = extract_evaluation_summary(previous_eval)

    inputs = {
        "prediction_run": str(prediction_run),
        "prediction_manifest": str(prediction_run / "run_manifest.json"),
        "player_prediction_csv": str(prediction_run / "player_predictions_preview.csv"),
        "transfer_run": str(transfer_run),
        "lineup_preview": str(lineup_json) if lineup_json else None,
        "previous_final_evaluation": str(previous_eval) if previous_eval else None,
    }
    fingerprints = {}
    for key, raw in inputs.items():
        if raw and Path(raw).is_file():
            fingerprints[key] = sha256_file(Path(raw))

    manifest = {
        "artifact_type": "fpl_weekly_candidate_package",
        "artifact_version": RUNNER_VERSION,
        "created_at_utc": utc_now(),
        "season": season,
        "target_gw": int(target_gw),
        "prior_season": prior_season,
        "phase": "pre",
        "publish_predictions_requested": bool(publish_requested),
        "inputs": inputs,
        "input_fingerprints": fingerprints,
        "stage_results": recorder.rows,
        "transfer_summary": transfer_summary,
        "lineup_summary": {
            "formation": lineup.get("formation"),
            "captain": (lineup.get("captain") or {}).get("web_name"),
            "vice_captain": (lineup.get("vice_captain") or {}).get("web_name"),
            "bench": [
                item.get("web_name") for item in (lineup.get("bench") or [])
            ],
            "starting_xi_predicted_points": lineup.get("starting_xi_predicted_points"),
            "objective_points": lineup.get("objective_points"),
        },
        "top10_outputs": [str(p) for p in top_paths],
        "previous_evaluation_status": evaluation.get("status"),
        "preview_only": True,
        "final_deadline_freeze": False,
        "writes_squad_state": False,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )

    lines = [
        "# FPL Weekly Candidate Package",
        "",
        "- Runner: `%s`" % RUNNER_VERSION,
        "- Season: `%s`" % season,
        "- Target GW: **%s**" % target_gw,
        "- Prediction run: `%s`" % prediction_run.name,
        "- Status: **PREVIEW ONLY**",
        "- Final deadline freeze: **False**",
        "",
        "## Previous GW evaluation",
        "",
    ]

    if previous_eval is not None:
        lines.append("- Reused/final evaluation: `%s`" % previous_eval)
        if evaluation:
            # Keep the raw authoritative summary path as the source of detail.
            lines.append("- Evaluation summary artifact: `%s`" % (previous_eval / "evaluation_summary.json"))
    else:
        lines.append("- No FINAL previous-GW evaluation was available.")

    lines.extend(
        [
            "",
            "## Model Team transfer decision",
            "",
            "- Recommendation: **%s**" % transfer_summary.get("recommended_action"),
            "- Net gain vs ROLL: **%s**" % transfer_summary.get("net_gain_vs_roll"),
            "- Bank after: **%s units**" % transfer_summary.get("bank_after_units"),
            "- Next-GW FT if recommendation: **%s**"
            % transfer_summary.get("free_transfers_next_if_recommendation"),
            "- Next-GW FT if ROLL: **%s**"
            % transfer_summary.get("free_transfers_next_if_roll"),
            "- Future FT option value monetized: **%s**"
            % transfer_summary.get("future_ft_option_value_monetized"),
            "",
            "## Model Team lineup",
            "",
        ]
    )
    if lineup:
        lines.extend(
            [
                "- Formation: **%s**" % lineup.get("formation"),
                "- Captain: **%s**" % ((lineup.get("captain") or {}).get("web_name")),
                "- Vice-captain: **%s**" % ((lineup.get("vice_captain") or {}).get("web_name")),
                "- Bench: **%s**"
                % " → ".join(
                    str(item.get("web_name"))
                    for item in (lineup.get("bench") or [])
                ),
                "- XI predicted points: **%.3f**"
                % float(lineup.get("starting_xi_predicted_points") or 0.0),
                "- XI + captain objective: **%.3f**"
                % float(lineup.get("objective_points") or 0.0),
            ]
        )
    else:
        lines.append("- Lineup stage unavailable.")

    lines.extend(["", "## Top 10 player predictions by position", ""])
    for pos in POSITIONS:
        lines.extend(
            [
                "### %s" % pos,
                "",
                "| # | Player | Team | Price | Status | Pred |",
                "|---:|---|---|---:|---|---:|",
            ]
        )
        for item in grouped_top.get(pos, []):
            lines.append(
                "| %s | %s | %s | %.1f | %s | %.3f |"
                % (
                    item["rank"],
                    item["web_name"],
                    item["team_short_name"],
                    float(item["now_cost"]) / 10.0,
                    item["status"],
                    float(item["predicted_points"]),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Safety",
            "",
            "- `auto` and `pre` never create a FINAL deadline freeze.",
            "- Prediction DB publication happens only with `--publish-predictions`.",
            "- Existing successful artifacts are reused with `--resume`.",
            "- Missing stages fail closed rather than silently switching to legacy baseline models.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def print_status(
    planning_root: Path,
    season: str,
    gw: int,
) -> None:
    prediction = discover_prediction_run(planning_root, season, gw)
    previous_eval = discover_final_evaluation(planning_root, season, gw - 1) if gw > 1 else None
    print("=== FPL Weekly Pipeline Status ===")
    print("season:", season)
    print("target_gw:", gw)
    print("latest_prediction_run:", prediction)
    print("previous_final_evaluation:", previous_eval)
    if prediction is not None:
        transfer = discover_transfer_run(planning_root, season, gw, prediction)
        print("matching_transfer_run:", transfer)
    package = find_reusable_candidate_package(planning_root, season, gw)
    print("latest_candidate_package:", package)


def run_pre(
    args: argparse.Namespace,
    repo_root: Path,
    planning_root: Path,
    python_exe: str,
    env: Mapping[str, str],
    recorder: StageRecorder,
) -> Path:
    prior_season = args.prior_season
    if not prior_season:
        year = int(args.season.split("_")[0])
        prior_season = "%s_%02d" % (year - 1, year % 100)

    api_proc: Optional[subprocess.Popen] = None
    try:
        if not args.skip_live_refresh:
            if args.base_url:
                base_url = args.base_url.rstrip("/")
                if not wait_for_api(base_url, args.season, timeout_seconds=5.0):
                    raise RuntimeError("Configured backend is not reachable: %s" % base_url)
            else:
                if args.dry_run:
                    base_url = "http://127.0.0.1:%s" % args.api_port
                else:
                    api_proc, base_url = start_private_api(
                        repo_root / "backend",
                        python_exe,
                        env,
                        args.api_port,
                        args.season,
                    )
            refresh_live_data(base_url, recorder, args.dry_run)

        prediction_run = run_prediction_stage(
            repo_root=repo_root,
            planning_root=planning_root,
            season=args.season,
            prior_season=prior_season,
            gw=args.target_gw,
            stabilization_gw=args.stabilization_gw,
            python_exe=python_exe,
            env=env,
            recorder=recorder,
            resume=args.resume,
            dry_run=args.dry_run,
        )

        if args.publish_predictions:
            publish_receipt = publish_predictions_if_requested(
                repo_root=repo_root,
                planning_root=planning_root,
                season=args.season,
                gw=args.target_gw,
                prediction_run=prediction_run,
                python_exe=python_exe,
                env=env,
                recorder=recorder,
                resume=args.resume,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                if publish_receipt is None:
                    raise RuntimeError(
                        "Prediction publish completed but no publish receipt was resolved for verification."
                    )
                verify_published_predictions(
                    repo_root=repo_root,
                    season=args.season,
                    gw=args.target_gw,
                    prediction_run=prediction_run,
                    receipt=publish_receipt,
                    python_exe=python_exe,
                    env=env,
                    recorder=recorder,
                    dry_run=args.dry_run,
                )

        package_dir = candidate_package_dir(
            planning_root,
            args.season,
            args.target_gw,
        )
        if args.dry_run:
            print("\n[candidate_package] would write:", package_dir)
            return package_dir
        package_dir.mkdir(parents=True, exist_ok=True)

        prediction_csv = prediction_run / "player_predictions_preview.csv"

        started = time.time()
        market_csv = build_market_csv(
            prediction_csv,
            package_dir / "current_market.csv",
        )
        recorder.add(
            "build_current_market",
            "PASS",
            started,
            outputs=[str(market_csv)],
        )

        started = time.time()
        top_csv, top_json, top_md, grouped_top = build_top10_by_position(
            prediction_csv,
            package_dir,
            args.position_top_n,
        )
        recorder.add(
            "top10_by_position",
            "PASS",
            started,
            outputs=[str(top_csv), str(top_json), str(top_md)],
        )

        transfer_run = run_transfer_stage(
            repo_root=repo_root,
            planning_root=planning_root,
            season=args.season,
            gw=args.target_gw,
            prediction_run=prediction_run,
            market_csv=market_csv,
            python_exe=python_exe,
            env=env,
            recorder=recorder,
            resume=args.resume,
            dry_run=args.dry_run,
            top_n=args.top_n,
        )

        lineup_json = run_lineup_stage(
            repo_root=repo_root,
            season=args.season,
            gw=args.target_gw,
            prediction_run=prediction_run,
            transfer_run=transfer_run,
            out_dir=package_dir,
            python_exe=python_exe,
            env=env,
            recorder=recorder,
            resume=args.resume,
            dry_run=args.dry_run,
        )

        previous_eval = (
            discover_final_evaluation(
                planning_root,
                args.season,
                args.target_gw - 1,
            )
            if args.target_gw > 1
            else None
        )

        write_candidate_package(
            out_dir=package_dir,
            season=args.season,
            target_gw=args.target_gw,
            prior_season=prior_season,
            prediction_run=prediction_run,
            transfer_run=transfer_run,
            lineup_json=lineup_json,
            top_paths=[top_csv, top_json, top_md],
            grouped_top=grouped_top,
            previous_eval=previous_eval,
            recorder=recorder,
            publish_requested=args.publish_predictions,
        )

        print("\n=== FPL Weekly PRE Complete ===")
        print("candidate_package:", package_dir)
        print("summary:", package_dir / "summary.md")
        print("manifest:", package_dir / "run_manifest.json")
        print("top10:", package_dir / "top10_by_position.csv")
        print("final_deadline_freeze: False")
        return package_dir
    finally:
        if api_proc is not None:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=5)
            except Exception:
                api_proc.kill()


def main() -> None:
    args = parse_args()
    repo_root = detect_repo_root(args.repo_root)
    planning_root = detect_planning_root(repo_root, args.planning_root)
    backend_dir = repo_root / "backend"
    python_exe = str(backend_dir / ".venv" / "bin" / "python")
    if not Path(python_exe).is_file():
        raise RuntimeError("Backend virtualenv Python not found: %s" % python_exe)

    env = child_env(repo_root, args.season)
    recorder = StageRecorder()

    if args.phase == "status":
        print_status(planning_root, args.season, args.target_gw)
        return

    if args.phase == "freeze":
        raise RuntimeError(
            "Generic FINAL FREEZE is intentionally fail-closed in runner v0.1. "
            "A normal PRE/auto run must never silently create an immutable deadline freeze."
        )

    if args.phase == "post":
        result = run_post_evaluation(
            repo_root=repo_root,
            planning_root=planning_root,
            season=args.season,
            gw=args.target_gw,
            python_exe=python_exe,
            env=env,
            recorder=recorder,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        print("\n=== FPL Weekly POST Complete ===")
        print("evaluation_dir:", result)
        return

    if args.phase == "auto" and args.target_gw > 1:
        # Safe, non-destructive POST catch-up for the previous GW.
        run_post_evaluation(
            repo_root=repo_root,
            planning_root=planning_root,
            season=args.season,
            gw=args.target_gw - 1,
            python_exe=python_exe,
            env=env,
            recorder=recorder,
            resume=args.resume,
            dry_run=args.dry_run,
        )

    run_pre(
        args=args,
        repo_root=repo_root,
        planning_root=planning_root,
        python_exe=python_exe,
        env=env,
        recorder=recorder,
    )


if __name__ == "__main__":
    main()
