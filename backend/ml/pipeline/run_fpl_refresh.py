from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from app.rules.chips import derive_chip_inventory, load_chip_rules
from ml.contracts.squad_state import (
    ChipInventoryEntry,
    ChipInventoryState,
    FreeTransferState,
    SquadPlayerState,
    SquadSelectionState,
    SquadState,
    calculate_selling_price_units,
    load_squad_state_json,
)
from ml.decision.free_transfer_ledger import build_ledger_state
from ml.decision.generate_transfer_candidates import (
    CandidatePruningPolicy,
    generate_transfer_candidates,
)
from ml.decision.optimize_single_gw_transfers import optimize_single_gw_transfers
from ml.validation.export_gameweek_pre_deadline_snapshot import (
    SNAPSHOT_KIND_CANDIDATE,
    export_gameweek_pre_deadline_snapshot,
)
from ml.validation.resolve_prediction_mode import resolve_prediction_mode


RUNNER_VERSION = "fpl_weekly_runner_v0_2"
PRE_INTEGRATION_VERSION = "fpl_pre_pipeline_integration_v1"
ELIGIBILITY_ADAPTER_VERSION = "fpl_live_selection_eligibility_v1"
LEGACY_GW2_STATE_ADAPTER_VERSION = "legacy_gw2_model_team_to_squad_state_v1"
DAG_VERSION = "fpl_gameweek_dag_v1"
POSITIONS = ("GKP", "DEF", "MID", "FWD")
VALID_PHASES = ("pre", "freeze", "post", "auto", "status")
FAIL_FAST = True


class GameweekDAGError(RuntimeError):
    """Raised when a requested Gameweek DAG is unsafe or internally invalid."""


@dataclass(frozen=True)
class StageAdapter:
    """Orchestration-only description of one pipeline stage.

    The runner owns dependencies and execution order. Business/model logic stays
    in the referenced implementation module and is wired by later integration
    milestones rather than reimplemented here.
    """

    name: str
    phase: str
    depends_on: Tuple[str, ...]
    planned_inputs: Tuple[str, ...]
    planned_outputs: Tuple[str, ...]
    implementation: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "depends_on": list(self.depends_on),
            "planned_inputs": list(self.planned_inputs),
            "planned_outputs": list(self.planned_outputs),
            "implementation": self.implementation,
            "note": self.note,
        }


def stage_adapter(
    name: str,
    phase: str,
    *,
    depends_on: Sequence[str] = (),
    planned_inputs: Sequence[str] = (),
    planned_outputs: Sequence[str] = (),
    implementation: str,
    note: str = "",
) -> StageAdapter:
    return StageAdapter(
        name=name,
        phase=phase,
        depends_on=tuple(depends_on),
        planned_inputs=tuple(planned_inputs),
        planned_outputs=tuple(planned_outputs),
        implementation=implementation,
        note=note,
    )


def pre_stage_adapters(
    *,
    root_dependencies: Sequence[str] = (),
    publish_predictions: bool = False,
    skip_live_refresh: bool = False,
) -> List[StageAdapter]:
    """Return the canonical PRE graph without executing any stage."""

    live_note = (
        "--skip-live-refresh requested; later execution adapter may reuse live state."
        if skip_live_refresh
        else "Refresh live FPL inputs through the existing ingest adapter."
    )
    stages = [
        stage_adapter(
            "live_ingest",
            "pre",
            depends_on=root_dependencies,
            planned_inputs=("FPL API", "season", "target_gw"),
            planned_outputs=("live_players", "live_teams", "live_fixtures", "completed_actuals"),
            implementation="existing runner/API ingest adapter",
            note=live_note,
        ),
        stage_adapter(
            "prediction_mode",
            "pre",
            depends_on=("live_ingest",),
            planned_inputs=("season", "target_gw", "completed_actuals"),
            planned_outputs=("resolved_prediction_mode",),
            implementation="existing prediction-mode resolver",
        ),
        stage_adapter(
            "player_model",
            "pre",
            depends_on=("prediction_mode",),
            planned_inputs=("resolved_prediction_mode", "live_players", "prior_evidence"),
            planned_outputs=("target_gw_player_model_artifact",),
            implementation="existing prediction producer wired by Day128B",
        ),
        stage_adapter(
            "match_model",
            "pre",
            depends_on=("prediction_mode",),
            planned_inputs=("resolved_prediction_mode", "live_fixtures", "prior_evidence"),
            planned_outputs=("target_gw_match_model_artifact", "scoreline_artifact"),
            implementation="existing prediction producer wired by Day128B",
        ),
        stage_adapter(
            "prediction_horizon",
            "pre",
            depends_on=("player_model", "match_model"),
            planned_inputs=("target_gw_player_model_artifact", "target_gw_match_model_artifact"),
            planned_outputs=("player_prediction_horizon", "fixture_prediction_horizon"),
            implementation="Day128B target-GW horizon adapter",
        ),
        stage_adapter(
            "owned_squad_state",
            "pre",
            depends_on=("live_ingest",),
            planned_inputs=("previous_frozen_squad_state", "current_prices", "current_player_metadata"),
            planned_outputs=("current_owned_squad_state",),
            implementation="ml.contracts.squad_state",
            note="GW2+ continuity only; never rebuild a normal-week squad from scratch.",
        ),
        stage_adapter(
            "free_transfer_ledger",
            "pre",
            depends_on=("owned_squad_state",),
            planned_inputs=("current_owned_squad_state", "target_season_transfer_policy"),
            planned_outputs=("free_transfer_ledger_state",),
            implementation="ml.decision.free_transfer_ledger",
        ),
        stage_adapter(
            "transfer_candidates",
            "pre",
            depends_on=(
                "player_model",
                "prediction_horizon",
                "owned_squad_state",
                "free_transfer_ledger",
            ),
            planned_inputs=(
                "current_owned_squad_state",
                "player_prediction_horizon",
                "free_transfer_ledger_state",
            ),
            planned_outputs=("legal_transfer_candidate_set",),
            implementation="ml.decision.generate_transfer_candidates",
        ),
        stage_adapter(
            "transfer_decision",
            "pre",
            depends_on=("transfer_candidates", "free_transfer_ledger"),
            planned_inputs=(
                "previous_frozen_squad_state",
                "legal_transfer_candidate_set",
                "target_gw_player_predictions",
                "free_transfer_ledger_state",
            ),
            planned_outputs=("chosen_transfer_or_no_transfer_plan",),
            implementation="ml.decision.optimize_single_gw_transfers",
            note="NO TRANSFER is first-class; opening-squad optimizer is prohibited.",
        ),
        stage_adapter(
            "lineup_selection",
            "pre",
            depends_on=("transfer_decision", "player_model"),
            planned_inputs=("chosen_transfer_or_no_transfer_plan",),
            planned_outputs=("reoptimized_xi", "captain", "vice_captain", "ordered_bench"),
            implementation="ml.decision.optimize_single_gw_transfers",
            note=(
                "Reuses the XI/C/VC already re-optimized inside Day127A; "
                "this stage is orchestration/selection only, not a third lineup optimizer."
            ),
        ),
    ]

    if publish_predictions:
        stages.append(
            stage_adapter(
                "prediction_publish",
                "pre",
                depends_on=("player_model", "match_model"),
                planned_inputs=("target_gw_player_model_artifact", "target_gw_match_model_artifact"),
                planned_outputs=("prediction_publish_receipt",),
                implementation="existing prediction publisher/verifier",
                note="Optional explicit write path; never implied by PRE.",
            )
        )

    candidate_deps = [
        "player_model",
        "match_model",
        "owned_squad_state",
        "free_transfer_ledger",
        "transfer_decision",
        "lineup_selection",
    ]
    if publish_predictions:
        candidate_deps.append("prediction_publish")

    stages.append(
        stage_adapter(
            "pre_deadline_candidate",
            "pre",
            depends_on=candidate_deps,
            planned_inputs=(
                "target_gw_player_model_artifact",
                "target_gw_match_model_artifact",
                "current_owned_squad_state",
                "chosen_transfer_or_no_transfer_plan",
                "reoptimized_xi",
                "captain",
                "vice_captain",
                "free_transfer_ledger_state",
                "optional_team_alex_reference",
            ),
            planned_outputs=("immutable_pre_deadline_candidate_snapshot",),
            implementation="ml.validation.export_gameweek_pre_deadline_snapshot",
            note="Candidate only. PRE must never silently create FINAL.",
        )
    )
    return stages


def freeze_stage_adapters(
    *,
    require_pre_candidate_dependency: bool,
) -> List[StageAdapter]:
    first_deps: Sequence[str] = (
        ("pre_deadline_candidate",) if require_pre_candidate_dependency else ()
    )
    return [
        stage_adapter(
            "freeze_window_validation",
            "freeze",
            depends_on=first_deps,
            planned_inputs=("pre_deadline_candidate", "as_of_time", "fpl_deadline"),
            planned_outputs=("freeze_window_authorized",),
            implementation="ml.contracts.gameweek_cycle.validate_freeze_window",
            note="FINAL freeze is explicit and must still be before the deadline.",
        ),
        stage_adapter(
            "final_freeze_export",
            "freeze",
            depends_on=("freeze_window_validation",),
            planned_inputs=(
                "player_model_artifact",
                "match_model_artifact",
                "model_team_state",
                "chosen_transfer_or_no_transfer_plan",
                "reoptimized_xi_c_vc",
                "free_transfer_ledger_state",
                "optional_team_alex_reference",
            ),
            planned_outputs=("immutable_final_pre_deadline_snapshot",),
            implementation="ml.validation.export_gameweek_pre_deadline_snapshot",
            note="Reuses Day127B; never create another freezer implementation.",
        ),
    ]


def post_stage_adapters(
    *,
    root_dependencies: Sequence[str] = (),
    stage_prefix: str = "",
) -> List[StageAdapter]:
    finality_name = stage_prefix + "actuals_finality"
    evaluation_name = stage_prefix + "post_evaluation"
    return [
        stage_adapter(
            finality_name,
            "post",
            depends_on=root_dependencies,
            planned_inputs=("official_target_gw_actuals", "fixture_finality"),
            planned_outputs=("final_actuals_manifest",),
            implementation="existing actual-ingest/finality adapter; Day129A wiring pending",
        ),
        stage_adapter(
            evaluation_name,
            "post",
            depends_on=(finality_name,),
            planned_inputs=("frozen_pre_deadline_snapshot", "final_actuals_manifest"),
            planned_outputs=("immutable_post_gameweek_evaluation",),
            implementation="ml.eval.evaluate_gameweek_post",
            note="Evaluation must point to frozen PRE evidence; never reconstruct predictions.",
        ),
    ]


def validate_phase_request(phase: str, *, final_freeze: bool) -> None:
    normalized = str(phase).strip().lower()
    if normalized not in VALID_PHASES:
        raise GameweekDAGError("Unsupported phase: %s" % phase)
    if final_freeze and normalized not in ("freeze", "auto"):
        raise GameweekDAGError(
            "--final-freeze is legal only with --phase freeze or --phase auto."
        )
    if normalized == "freeze" and not final_freeze:
        raise GameweekDAGError(
            "--phase freeze requires the explicit --final-freeze flag."
        )


def validate_stage_adapters(stages: Sequence[StageAdapter]) -> Tuple[StageAdapter, ...]:
    """Validate unique names, dependency existence, and acyclic ordering."""

    by_name: Dict[str, StageAdapter] = {}
    insertion_order: List[str] = []
    for stage in stages:
        if stage.name in by_name:
            raise GameweekDAGError("Duplicate stage name: %s" % stage.name)
        by_name[stage.name] = stage
        insertion_order.append(stage.name)

    for stage in stages:
        missing = [dep for dep in stage.depends_on if dep not in by_name]
        if missing:
            raise GameweekDAGError(
                "Stage %s has missing dependencies: %s" % (stage.name, missing)
            )

    resolved: List[str] = []
    unresolved = list(insertion_order)
    while unresolved:
        progressed = False
        for name in list(unresolved):
            stage = by_name[name]
            if all(dep in resolved for dep in stage.depends_on):
                resolved.append(name)
                unresolved.remove(name)
                progressed = True
        if not progressed:
            raise GameweekDAGError(
                "Stage dependency cycle detected among: %s" % unresolved
            )

    return tuple(by_name[name] for name in resolved)


def build_dag_plan(
    *,
    phase: str,
    season: str,
    target_gw: int,
    final_freeze: bool = False,
    resume: bool = False,
    publish_predictions: bool = False,
    skip_live_refresh: bool = False,
) -> Dict[str, Any]:
    """Build and validate the intended Gameweek DAG without running model logic."""

    normalized = str(phase).strip().lower()
    validate_phase_request(normalized, final_freeze=final_freeze)

    if normalized == "pre":
        stages = pre_stage_adapters(
            publish_predictions=publish_predictions,
            skip_live_refresh=skip_live_refresh,
        )
    elif normalized == "freeze":
        stages = freeze_stage_adapters(require_pre_candidate_dependency=False)
    elif normalized == "post":
        stages = post_stage_adapters()
    elif normalized == "auto":
        auto_post = stage_adapter(
            "previous_gw_post",
            "auto",
            planned_inputs=("previous_gw_final_actuals_if_available",),
            planned_outputs=("previous_gw_post_evaluation_or_reuse",),
            implementation="existing POST adapter",
            note="Safe previous-GW catch-up before target-GW PRE.",
        )
        stages = [auto_post]
        stages.extend(
            pre_stage_adapters(
                root_dependencies=("previous_gw_post",),
                publish_predictions=publish_predictions,
                skip_live_refresh=skip_live_refresh,
            )
        )
        if final_freeze:
            stages.extend(
                freeze_stage_adapters(require_pre_candidate_dependency=True)
            )
    else:  # status
        stages = [
            stage_adapter(
                "status_discovery",
                "status",
                planned_inputs=("planning_root", "season", "target_gw"),
                planned_outputs=("discovered_pipeline_status",),
                implementation="existing runner artifact discovery",
                note="Read-only.",
            )
        ]

    ordered = validate_stage_adapters(stages)
    return {
        "dag_version": DAG_VERSION,
        "runner_version": RUNNER_VERSION,
        "phase": normalized,
        "season": str(season),
        "target_gw": int(target_gw),
        "resume": bool(resume),
        "publish_predictions": bool(publish_predictions),
        "skip_live_refresh": bool(skip_live_refresh),
        "final_freeze_requested": bool(final_freeze),
        "fail_fast": FAIL_FAST,
        "model_logic_executed": False,
        "valid": True,
        "stages": [stage.to_dict() for stage in ordered],
    }


def render_dag_plan(plan: Mapping[str, Any]) -> str:
    lines = [
        "=== FPL Unified Gameweek DAG ===",
        "dag_version: %s" % plan["dag_version"],
        "runner_version: %s" % plan["runner_version"],
        "phase: %s" % plan["phase"],
        "season: %s" % plan["season"],
        "target_gw: %s" % plan["target_gw"],
        "resume: %s" % plan["resume"],
        "fail_fast: %s" % plan["fail_fast"],
        "final_freeze_requested: %s" % plan["final_freeze_requested"],
        "",
    ]
    for index, stage in enumerate(plan["stages"], start=1):
        lines.extend(
            [
                "%02d. %s [%s]" % (index, stage["name"], stage["phase"]),
                "    depends_on: %s"
                % (", ".join(stage["depends_on"]) if stage["depends_on"] else "-"),
                "    implementation: %s" % stage["implementation"],
                "    planned_inputs: %s"
                % (", ".join(stage["planned_inputs"]) if stage["planned_inputs"] else "-"),
                "    planned_outputs: %s"
                % (", ".join(stage["planned_outputs"]) if stage["planned_outputs"] else "-"),
            ]
        )
        if stage.get("note"):
            lines.append("    note: %s" % stage["note"])
    lines.extend(
        [
            "",
            "DAG_VALID: %s" % bool(plan["valid"]),
            "MODEL_LOGIC_EXECUTED: %s" % bool(plan["model_logic_executed"]),
        ]
    )
    return "\n".join(lines)


def execute_stage_adapters(
    stages: Sequence[StageAdapter],
    executor: Callable[[StageAdapter], Optional[Sequence[str]]],
    recorder: "StageRecorder",
) -> None:
    """Generic fail-fast adapter executor used by future runner wiring.

    Day128A tests this orchestration contract with dummy executors only.
    """

    ordered = validate_stage_adapters(stages)
    for stage in ordered:
        started = time.time()
        try:
            outputs = executor(stage)
        except Exception as exc:
            recorder.add(
                stage.name,
                "FAILED",
                started,
                dependencies=stage.depends_on,
                planned_inputs=stage.planned_inputs,
                planned_outputs=stage.planned_outputs,
                note=str(exc),
            )
            if FAIL_FAST:
                raise
            continue

        recorder.add(
            stage.name,
            "PASS",
            started,
            outputs=outputs,
            dependencies=stage.depends_on,
            planned_inputs=stage.planned_inputs,
            planned_outputs=stage.planned_outputs,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Canonical one-command FPL weekly runner. It orchestrates existing "
            "ingest/prediction/evaluation/decision stages and writes one candidate package."
        )
    )
    p.add_argument("--phase", required=True, choices=list(VALID_PHASES))
    p.add_argument("--season", required=True)
    p.add_argument("--target-gw", type=int, required=True)
    p.add_argument("--prior-season", default=None)
    p.add_argument("--stabilization-gw", type=int, default=6)
    p.add_argument("--planning-root", default=None)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--final-freeze",
        action="store_true",
        help=(
            "Explicit authorization for a FINAL freeze. Day128A dry-run can plan "
            "this stage; live runner wiring remains fail-closed until integration."
        ),
    )
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
    p.add_argument(
        "--max-transfers",
        type=int,
        default=2,
        help=(
            "Transparent Day127A plan-search cap for one PRE run. "
            "This is not an FPL legality rule; hit pricing remains policy-driven."
        ),
    )
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
        dependencies: Optional[Sequence[str]] = None,
        planned_inputs: Optional[Sequence[str]] = None,
        planned_outputs: Optional[Sequence[str]] = None,
    ) -> None:
        ended = time.time()
        started_at = datetime.fromtimestamp(started, timezone.utc)
        ended_at = datetime.fromtimestamp(ended, timezone.utc)
        duration = round(ended - started, 3)
        self.rows.append(
            {
                "stage": name,
                "status": status,
                "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
                "ended_at_utc": ended_at.isoformat().replace("+00:00", "Z"),
                "duration_seconds": duration,
                # Compatibility field retained for existing candidate-package consumers.
                "elapsed_seconds": duration,
                "dependencies": list(dependencies or []),
                "planned_inputs": list(planned_inputs or []),
                "planned_outputs": list(planned_outputs or []),
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
    if "--receipt" in flags:
        cmd.extend(["--receipt", str(receipt)])
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



def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    method = getattr(value, "to_dict", None)
    if callable(method):
        return _jsonable(method())
    return value


def write_json_new(path: Path, payload: Any) -> Path:
    target = Path(path)
    if target.exists():
        raise RuntimeError("Refusing to overwrite existing artifact: %s" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return target


def _formal_stage(
    recorder: StageRecorder,
    name: str,
    *,
    dependencies: Sequence[str],
    planned_inputs: Sequence[str],
    planned_outputs: Sequence[str],
    function: Callable[[], Any],
    output_paths: Optional[Callable[[Any], Sequence[str]]] = None,
) -> Any:
    started = time.time()
    try:
        result = function()
    except Exception as exc:
        recorder.add(
            name,
            "FAILED",
            started,
            dependencies=dependencies,
            planned_inputs=planned_inputs,
            planned_outputs=planned_outputs,
            note=str(exc),
        )
        raise

    outputs: Sequence[str] = ()
    if output_paths is not None:
        outputs = output_paths(result)
    recorder.add(
        name,
        "PASS",
        started,
        outputs=outputs,
        dependencies=dependencies,
        planned_inputs=planned_inputs,
        planned_outputs=planned_outputs,
    )
    return result


def resolve_runner_prediction_mode(
    *,
    season: str,
    target_gw: int,
    prior_season: Optional[str],
    stabilization_gw: int,
) -> Dict[str, Any]:
    """Resolve the existing prediction mode and fail closed on unwired producers."""

    result = resolve_prediction_mode(
        season=season,
        target_gw=target_gw,
        requested_prediction_mode="auto",
        prior_season=prior_season,
        stabilization_gw=stabilization_gw,
        allow_experimental_mode=False,
    )
    if not bool(result.get("valid")):
        raise RuntimeError(
            "Prediction-mode resolution failed: %s" % list(result.get("errors") or [])
        )

    mode = str(result.get("resolved_prediction_mode") or "")
    if mode == "early_season_blend":
        return dict(result)
    if mode == "normal_weekly":
        raise RuntimeError(
            "prediction_mode=normal_weekly resolved for target_gw=%s, but the "
            "approved GW6+ normal-weekly producer is still outstanding. Refusing "
            "to fall back to legacy baseline_rollavg_*." % target_gw
        )
    raise RuntimeError(
        "Day128B rolling PRE is a GW2+ stateful path; resolved prediction_mode=%s "
        "is not wired here." % mode
    )


def validate_prediction_run(
    prediction_run: Path,
    *,
    season: str,
    target_gw: int,
    resolved_mode: str,
) -> Dict[str, Any]:
    run_dir = Path(prediction_run).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)

    if str(manifest.get("status")) != "PASS_PREVIEW":
        raise RuntimeError(
            "Prediction run is not PASS_PREVIEW: %s" % manifest.get("status")
        )
    if str(manifest.get("season")) != str(season):
        raise RuntimeError("Prediction run season does not match PRE request.")
    if int(manifest.get("target_gw") or -1) != int(target_gw):
        raise RuntimeError("Prediction run target_gw does not match PRE request.")
    if str(manifest.get("prediction_mode")) != str(resolved_mode):
        raise RuntimeError(
            "Prediction run mode=%s does not match resolved mode=%s."
            % (manifest.get("prediction_mode"), resolved_mode)
        )
    if manifest.get("preview_only") is not True:
        raise RuntimeError("Prediction run must remain preview_only before freeze.")
    if manifest.get("database_prediction_write") not in (None, False):
        raise RuntimeError(
            "Prediction preview may not claim database_prediction_write=true."
        )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("Prediction run manifest is missing outputs mapping.")

    required = {
        "player_predictions_preview": "player_predictions_preview.csv",
        "match_predictions_preview": "match_predictions_preview.csv",
        "scoreline_preview": "scoreline_preview.csv",
        "bootstrap_snapshot": "bootstrap_snapshot.json",
    }
    paths: Dict[str, Path] = {}
    for key, fallback in required.items():
        raw = outputs.get(key) or fallback
        path = run_dir / str(raw)
        if not path.is_file():
            raise RuntimeError(
                "Prediction run is missing required %s artifact: %s" % (key, path)
            )
        paths[key] = path

    created_at = str(manifest.get("created_at") or "").strip()
    if not created_at:
        raise RuntimeError("Prediction run manifest is missing created_at.")

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "created_at": created_at,
        "player_csv": paths["player_predictions_preview"],
        "match_csv": paths["match_predictions_preview"],
        "scoreline_csv": paths["scoreline_preview"],
        "bootstrap_json": paths["bootstrap_snapshot"],
    }


def target_gw_deadline_from_bootstrap(path: Path, target_gw: int) -> str:
    payload = read_json(Path(path))
    events = payload.get("events")
    if not isinstance(events, list):
        raise RuntimeError("Bootstrap snapshot is missing events.")
    for raw in events:
        if not isinstance(raw, Mapping):
            continue
        try:
            event_id = int(raw.get("id"))
        except (TypeError, ValueError):
            continue
        if event_id != int(target_gw):
            continue
        deadline = str(raw.get("deadline_time") or "").strip()
        if not deadline:
            raise RuntimeError(
                "Bootstrap target Gameweek is missing deadline_time."
            )
        return deadline
    raise RuntimeError(
        "Bootstrap snapshot does not contain target Gameweek=%s." % target_gw
    )


def _nullable_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _nullable_int(value: Any) -> Optional[int]:
    number = _nullable_float(value)
    if number is None:
        return None
    return int(number)


def selection_eligibility_from_live_prediction(
    row: Mapping[str, Any],
) -> Tuple[bool, str, List[str]]:
    """Translate official live status into the explicit Day126B eligibility field.

    This is intentionally a hard-status adapter, not a football-opinion model.
    Doubtful players remain selectable with a visible risk flag. Missing/unknown
    status fails closed for transfer-IN eligibility.
    """

    status = str(row.get("status") or "").strip().lower()
    fixture_count = _nullable_int(row.get("fixture_count")) or 0
    chance = _nullable_float(row.get("chance_of_playing_next_round"))

    risk_flags: List[str] = []
    if bool(row.get("official_availability_adjustment_applied")):
        risk_flags.append("official_availability_adjustment")
    if status in ("d", "i"):
        risk_flags.append("uncertain_status")

    if fixture_count <= 0:
        return False, "no_target_gw_fixture", sorted(set(risk_flags))
    if status in ("s", "u", "n"):
        return (
            False,
            "hard_unavailable_status_%s" % status,
            sorted(set(risk_flags + ["hard_unavailable_status"])),
        )
    if status == "i" and (chance is None or chance <= 0.0):
        return (
            False,
            "injured_zero_or_unknown_availability",
            sorted(set(risk_flags + ["hard_unavailable_status"])),
        )
    if chance is not None and chance <= 0.0:
        return (
            False,
            "official_zero_availability",
            sorted(set(risk_flags + ["official_zero_availability"])),
        )
    if status not in ("a", "d", "i"):
        return (
            False,
            "unrecognized_live_status",
            sorted(set(risk_flags + ["status_review_required"])),
        )
    if status == "d":
        return True, "eligible_doubtful_with_visible_risk", sorted(set(risk_flags))
    if status == "i":
        return True, "eligible_injury_with_positive_official_chance", sorted(set(risk_flags))
    return True, "eligible_live_status", sorted(set(risk_flags))


def build_target_gw_horizon_artifacts(
    *,
    player_csv: Path,
    match_csv: Path,
    scoreline_csv: Path,
    target_gw: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Normalize the current prediction producer into a one-GW live horizon.

    The Master Plan requires the next GW and makes later GWs optional when
    reliable. Day128B therefore uses a target-GW-only effective horizon rather
    than inventing missing future predictions.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    player_df = pd.read_csv(player_csv)
    required_player = {
        "fpl_player_id",
        "position",
        "team_id",
        "now_cost",
        "status",
        "fixture_count",
        "predicted_points",
    }
    missing_player = sorted(required_player - set(player_df.columns))
    if missing_player:
        raise RuntimeError(
            "Player Model artifact cannot form PRE horizon; missing %s."
            % missing_player
        )
    if player_df["fpl_player_id"].duplicated().any():
        raise RuntimeError("Player Model contains duplicate fpl_player_id rows.")
    if player_df["predicted_points"].isna().any():
        raise RuntimeError("Player Model contains missing predicted_points.")

    market_rows: List[Dict[str, Any]] = []
    horizon_rows: List[Dict[str, Any]] = []
    for raw in player_df.to_dict(orient="records"):
        eligible, reason, risk_flags = selection_eligibility_from_live_prediction(raw)
        fpl_id = int(raw["fpl_player_id"])
        predicted = float(raw["predicted_points"])
        market_rows.append(
            {
                "fpl_player_id": fpl_id,
                "player_name": raw.get("web_name") or raw.get("player_name"),
                "web_name": raw.get("web_name") or raw.get("player_name"),
                "position": str(raw["position"]).upper(),
                "club_id": int(raw["team_id"]),
                "team_id": int(raw["team_id"]),
                "team_name": raw.get("team_name"),
                "team_short_name": raw.get("team_short_name"),
                "now_cost": int(raw["now_cost"]),
                "status": raw.get("status"),
                "selection_eligible": bool(eligible),
                "eligibility_reason": reason,
            }
        )
        horizon_rows.append(
            {
                "fpl_player_id": fpl_id,
                "player_name": raw.get("web_name") or raw.get("player_name"),
                "target_gw": int(target_gw),
                "predicted_points": predicted,
                "horizon_predicted_points": predicted,
                "selection_eligible": bool(eligible),
                "eligibility_reason": reason,
                "risk_flags": risk_flags,
                "expected_minutes": _nullable_float(
                    raw.get("expected_minutes_total")
                ),
                "appearance_probability": _nullable_float(
                    raw.get("blended_appearance_probability")
                ),
                "start_probability": _nullable_float(
                    raw.get("blended_start_probability")
                ),
                "fallback_used": bool(raw.get("prior_fallback_used", False)),
                "role_proxy": "target_gw_live_preview",
            }
        )

    market_csv = out_dir / "current_market.csv"
    horizon_csv = out_dir / "player_prediction_horizon.csv"
    horizon_json = out_dir / "player_prediction_horizon.json"
    pd.DataFrame(market_rows).to_csv(market_csv, index=False)
    pd.DataFrame(horizon_rows).to_csv(horizon_csv, index=False)
    write_json_new(
        horizon_json,
        {
            "artifact_type": "fpl_live_player_prediction_horizon",
            "artifact_version": PRE_INTEGRATION_VERSION,
            "eligibility_adapter_version": ELIGIBILITY_ADAPTER_VERSION,
            "target_gw": int(target_gw),
            "effective_horizon_gameweeks": [int(target_gw)],
            "multi_gw_horizon_used": False,
            "missing_future_predictions_zero_filled": False,
            "rows": horizon_rows,
        },
    )

    match_df = pd.read_csv(match_csv)
    score_df = pd.read_csv(scoreline_csv)
    if "fpl_fixture_id" not in match_df.columns or "fpl_fixture_id" not in score_df.columns:
        raise RuntimeError("Match/scoreline artifacts require fpl_fixture_id.")
    if match_df["fpl_fixture_id"].duplicated().any():
        raise RuntimeError("Match Model contains duplicate fpl_fixture_id rows.")
    if score_df["fpl_fixture_id"].duplicated().any():
        raise RuntimeError("Scoreline Model contains duplicate fpl_fixture_id rows.")

    score_lookup = {
        int(row["fpl_fixture_id"]): row
        for row in score_df.to_dict(orient="records")
    }
    fixture_rows: List[Dict[str, Any]] = []
    for match_row in match_df.to_dict(orient="records"):
        fixture_id = int(match_row["fpl_fixture_id"])
        score_row = score_lookup.get(fixture_id)
        if score_row is None:
            raise RuntimeError(
                "Scoreline artifact is missing fpl_fixture_id=%s." % fixture_id
            )
        combined = dict(match_row)
        for key, value in score_row.items():
            if key not in combined or key.startswith("top_") or key.startswith("scoreline_") or key.startswith("score_grid_"):
                combined[key] = value
        combined["target_gw"] = int(target_gw)
        combined["horizon_index"] = 1
        fixture_rows.append(combined)

    if set(score_lookup) != {
        int(row["fpl_fixture_id"]) for row in match_df.to_dict(orient="records")
    }:
        raise RuntimeError("Match and scoreline fixture identities do not match exactly.")

    fixture_csv = out_dir / "fixture_prediction_horizon.csv"
    fixture_json = out_dir / "fixture_prediction_horizon.json"
    pd.DataFrame(fixture_rows).to_csv(fixture_csv, index=False)
    write_json_new(
        fixture_json,
        {
            "artifact_type": "fpl_live_fixture_prediction_horizon",
            "artifact_version": PRE_INTEGRATION_VERSION,
            "target_gw": int(target_gw),
            "effective_horizon_gameweeks": [int(target_gw)],
            "multi_gw_horizon_used": False,
            "rows": fixture_rows,
        },
    )

    return {
        "market_rows": market_rows,
        "horizon_rows": horizon_rows,
        "fixture_rows": fixture_rows,
        "market_csv": market_csv,
        "player_horizon_csv": horizon_csv,
        "player_horizon_json": horizon_json,
        "fixture_horizon_csv": fixture_csv,
        "fixture_horizon_json": fixture_json,
    }


def _extract_player_id(value: Any, label: str) -> int:
    if isinstance(value, Mapping):
        for key in ("fpl_player_id", "player_id", "id"):
            if value.get(key) not in (None, ""):
                return int(value[key])
        raise RuntimeError("%s is missing player identity." % label)
    return int(value)


def _legacy_gw2_chip_inventory(season: str, gameweek: int) -> ChipInventoryState:
    if int(gameweek) != 2:
        raise RuntimeError(
            "Legacy no-chip compatibility is intentionally restricted to the verified GW2 Model Team freeze."
        )
    rules = load_chip_rules(season)
    raw = derive_chip_inventory(
        rules,
        usage_history=[],
        as_of_gameweek=int(gameweek),
        validate_history=True,
    )
    entries: List[ChipInventoryEntry] = []
    for chip_id, item in sorted(raw.items()):
        current = dict(item["current_window"])
        entries.append(
            ChipInventoryEntry(
                chip_id=str(chip_id),
                remaining=int(current["remaining"]),
                available_now=int(current["available_now"]),
                window_id=str(item["current_window_id"]),
            )
        )
    return ChipInventoryState(
        as_of_gameweek=int(gameweek),
        entries=tuple(entries),
    )


def canonicalize_legacy_gw2_model_team_state(
    payload: Mapping[str, Any],
    *,
    expected_season: str,
    expected_gameweek: int,
) -> SquadState:
    """Adapt only the verified old GW2 FINAL wrapper into the Day125B contract."""

    if int(expected_gameweek) != 2:
        raise RuntimeError(
            "Legacy Model Team compatibility is restricted to previous GW2."
        )
    if str(payload.get("artifact_type")) != "fpl_model_team_frozen_state":
        raise RuntimeError("Not a supported legacy Model Team frozen wrapper.")
    if str(payload.get("season")) != str(expected_season):
        raise RuntimeError("Legacy frozen wrapper season mismatch.")
    if int(payload.get("gw") or -1) != 2:
        raise RuntimeError("Legacy frozen wrapper is not GW2.")
    if payload.get("final_pre_deadline_snapshot_frozen") is not True:
        raise RuntimeError("Legacy GW2 wrapper is not FINAL frozen.")
    if payload.get("final_deadline_freeze") is not True:
        raise RuntimeError("Legacy GW2 wrapper lacks final_deadline_freeze=true.")

    raw_squad = payload.get("squad")
    if not isinstance(raw_squad, list) or len(raw_squad) != 15:
        raise RuntimeError("Legacy GW2 wrapper must contain exactly 15 squad rows.")

    players: List[SquadPlayerState] = []
    for index, raw in enumerate(raw_squad):
        if not isinstance(raw, Mapping):
            raise RuntimeError("Legacy squad row %s is not a mapping." % index)
        purchase = int(raw["purchase_price_units"])
        current = int(raw["current_price_units"])
        players.append(
            SquadPlayerState(
                fpl_player_id=int(raw["fpl_player_id"]),
                player_name=raw.get("web_name") or raw.get("player_name"),
                position=str(raw["position"]).upper(),
                club_id=int(raw.get("team_id") or raw.get("club_id")),
                purchase_price_units=purchase,
                current_price_units=current,
                selling_price_units=calculate_selling_price_units(
                    purchase,
                    current,
                ),
            )
        )

    lineup = payload.get("lineup")
    if not isinstance(lineup, Mapping):
        raise RuntimeError("Legacy GW2 wrapper is missing lineup.")

    starting = tuple(
        _extract_player_id(item, "legacy starting XI")
        for item in lineup.get("starting_player_ids", [])
    )
    bench = tuple(
        _extract_player_id(item, "legacy bench")
        for item in lineup.get("bench_order", [])
    )
    captain = _extract_player_id(lineup.get("captain"), "legacy captain")
    vice = _extract_player_id(lineup.get("vice_captain"), "legacy vice captain")

    transfer = payload.get("transfer_decision")
    if not isinstance(transfer, Mapping):
        raise RuntimeError("Legacy GW2 wrapper is missing transfer_decision.")
    ft_next = int(transfer["free_transfers_next_gameweek"])
    bank_after = int(transfer["bank_after_units"])
    frozen_at = str(payload.get("frozen_at_utc") or "").strip()
    freeze_id = str(payload.get("freeze_id") or "").strip()
    if not frozen_at or not freeze_id:
        raise RuntimeError("Legacy GW2 wrapper is missing freeze identity/time.")

    return SquadState(
        season=str(expected_season),
        gameweek=2,
        as_of_utc=frozen_at,
        state_version=LEGACY_GW2_STATE_ADAPTER_VERSION,
        state_kind="model_team",
        state_status="frozen",
        source_phase_id="GW02-FREEZE",
        source_run_id=freeze_id,
        players=tuple(players),
        selection=SquadSelectionState(
            starting_xi_player_ids=starting,
            bench_order_player_ids=bench,
            captain_player_id=captain,
            vice_captain_player_id=vice,
        ),
        bank_units=bank_after,
        chip_inventory=_legacy_gw2_chip_inventory(
            expected_season,
            2,
        ),
        free_transfers=FreeTransferState(
            available_for_gameweek=3,
            count=ft_next,
        ),
        predecessor=None,
        shadow_optimal=None,
    )


def discover_previous_model_team_state(
    planning_root: Path,
    *,
    season: str,
    target_gw: int,
) -> Tuple[SquadState, Path, str]:
    previous_gw = int(target_gw) - 1
    if previous_gw < 1:
        raise RuntimeError("Stateful PRE requires target_gw >= 2.")

    roots = [
        planning_root
        / "frozen-snapshots"
        / season
        / ("gw%02d" % previous_gw),
        planning_root
        / "gw-pre"
        / season
        / ("gw%02d" % previous_gw),
    ]
    candidates: List[Path] = []
    for root in roots:
        if root.is_dir():
            candidates.extend(root.rglob("model_team_state.json"))
    candidates = sorted(
        {path.resolve() for path in candidates},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    legacy_errors: List[str] = []
    for path in candidates:
        try:
            state = load_squad_state_json(path)
        except Exception as canonical_exc:
            try:
                payload = read_json(path)
                state = canonicalize_legacy_gw2_model_team_state(
                    payload,
                    expected_season=season,
                    expected_gameweek=previous_gw,
                )
            except Exception as legacy_exc:
                legacy_errors.append(
                    "%s: canonical=%s; legacy=%s"
                    % (path, canonical_exc, legacy_exc)
                )
                continue
            source_kind = "legacy_gw2_compat"
        else:
            source_kind = "canonical"

        if (
            state.season == season
            and int(state.gameweek) == previous_gw
            and state.state_kind == "model_team"
            and state.state_status == "frozen"
        ):
            return state, path, source_kind

    raise RuntimeError(
        "Could not discover a frozen Model Team state for season=%s previous_gw=%s. "
        "Checked model_team_state.json candidates. Details: %s"
        % (season, previous_gw, legacy_errors[-3:])
    )


def build_current_owned_state_for_candidates(
    previous: SquadState,
    market_rows: Sequence[Mapping[str, Any]],
    *,
    target_gw: int,
) -> Dict[str, Any]:
    """Overlay current official prices/club metadata on immutable ownership lineage."""

    market_by_id = {
        int(row["fpl_player_id"]): dict(row)
        for row in market_rows
    }
    players: List[Dict[str, Any]] = []
    changes: List[Dict[str, Any]] = []

    for player in previous.players:
        pid = int(player.fpl_player_id)
        current = market_by_id.get(pid)
        if current is None:
            raise RuntimeError(
                "Current Player Model/market is missing owned fpl_player_id=%s." % pid
            )
        current_position = str(current["position"]).upper()
        current_club = int(current["club_id"])
        current_price = int(current["now_cost"])
        current_selling = calculate_selling_price_units(
            int(player.purchase_price_units),
            current_price,
        )

        changed_fields: Dict[str, Any] = {}
        if current_position != str(player.position):
            changed_fields["position"] = {
                "previous": str(player.position),
                "current": current_position,
            }
        if current_club != int(player.club_id):
            changed_fields["club_id"] = {
                "previous": int(player.club_id),
                "current": current_club,
            }
        if current_price != int(player.current_price_units):
            changed_fields["current_price_units"] = {
                "previous": int(player.current_price_units),
                "current": current_price,
            }
        if current_selling != int(player.selling_price_units):
            changed_fields["selling_price_units"] = {
                "previous": int(player.selling_price_units),
                "current": current_selling,
            }
        if changed_fields:
            changes.append(
                {
                    "fpl_player_id": pid,
                    "changes": changed_fields,
                }
            )

        players.append(
            {
                "fpl_player_id": pid,
                "player_name": player.player_name,
                "position": current_position,
                "club_id": current_club,
                "purchase_price_units": int(player.purchase_price_units),
                "current_price_units": current_price,
                "selling_price_units": current_selling,
            }
        )

    return {
        "season": previous.season,
        "gameweek": int(target_gw),
        "state_kind": previous.state_kind,
        "bank_units": int(previous.bank_units),
        "players": players,
        "source_frozen_state_id": previous.state_id,
        "source_owned_state_fingerprint": previous.owned_state_fingerprint,
        "current_valuation_overlay": True,
        "price_as_of_policy": "current_market_with_fpl_selling_price_rule_v1",
        "metadata_or_price_changes": changes,
        "change_count": len(changes),
    }


def build_match_model_source_bundle(
    *,
    out_dir: Path,
    match_csv: Path,
    scoreline_csv: Path,
    fixture_horizon_json: Path,
    prediction_run_id: str,
) -> Path:
    bundle = Path(out_dir)
    bundle.mkdir(parents=True, exist_ok=False)
    for source in (match_csv, scoreline_csv, fixture_horizon_json):
        shutil.copy2(str(source), str(bundle / source.name))
    write_json_new(
        bundle / "source_reference.json",
        {
            "artifact_type": "fpl_match_model_plus_scoreline_bundle",
            "artifact_version": PRE_INTEGRATION_VERSION,
            "source_prediction_run_id": prediction_run_id,
            "match_predictions": match_csv.name,
            "scoreline_predictions": scoreline_csv.name,
            "fixture_horizon": fixture_horizon_json.name,
        },
    )
    return bundle


def write_formal_pre_manifest(
    *,
    package_dir: Path,
    season: str,
    target_gw: int,
    prior_season: str,
    prediction_mode_result: Mapping[str, Any],
    prediction_info: Mapping[str, Any],
    previous_state: SquadState,
    previous_state_path: Path,
    previous_state_source_kind: str,
    current_owned_state_path: Path,
    horizon: Mapping[str, Any],
    ledger_state: Any,
    candidates_path: Path,
    decision_path: Path,
    snapshot_result: Mapping[str, Any],
    recorder: StageRecorder,
    publish_requested: bool,
    max_transfers: int,
    status: str = "PASS_PRE_CANDIDATE",
    blocker: str = "",
) -> Path:
    inputs = {
        "prediction_run": str(prediction_info["run_dir"]),
        "prediction_manifest": str(prediction_info["manifest_path"]),
        "player_model": str(prediction_info["player_csv"]),
        "match_model": str(prediction_info["match_csv"]),
        "scoreline_model": str(prediction_info["scoreline_csv"]),
        "previous_frozen_model_team": str(previous_state_path),
        "current_owned_state": str(current_owned_state_path),
    }
    fingerprints: Dict[str, str] = {}
    for key, raw in inputs.items():
        path = Path(raw)
        if path.is_file():
            fingerprints[key] = sha256_file(path)

    payload = {
        "artifact_type": "fpl_unified_pre_candidate_package",
        "artifact_version": PRE_INTEGRATION_VERSION,
        "runner_version": RUNNER_VERSION,
        "dag_version": DAG_VERSION,
        "created_at_utc": utc_now(),
        "status": status,
        "season": season,
        "target_gw": int(target_gw),
        "prior_season": prior_season,
        "phase": "pre",
        "resolved_prediction_mode": prediction_mode_result.get(
            "resolved_prediction_mode"
        ),
        "publish_predictions_requested": bool(publish_requested),
        "configured_max_transfers": int(max_transfers),
        "inputs": inputs,
        "input_fingerprints": fingerprints,
        "previous_squad_state": {
            "state_id": previous_state.state_id,
            "owned_state_fingerprint": previous_state.owned_state_fingerprint,
            "source_kind": previous_state_source_kind,
            "gameweek": int(previous_state.gameweek),
            "bank_units": int(previous_state.bank_units),
            "free_transfers_available_for_gameweek": int(
                previous_state.free_transfers.available_for_gameweek
            ),
            "free_transfers": int(previous_state.free_transfers.count),
        },
        "horizon": {
            "effective_gameweeks": [int(target_gw)],
            "multi_gw_horizon_used": False,
            "missing_future_predictions_zero_filled": False,
            "player_horizon_csv": str(horizon["player_horizon_csv"]),
            "fixture_horizon_csv": str(horizon["fixture_horizon_csv"]),
        },
        "free_transfer_ledger": _jsonable(ledger_state),
        "transfer_candidates": str(candidates_path),
        "transfer_decision": str(decision_path),
        "pre_deadline_snapshot": dict(snapshot_result),
        "stage_results": list(recorder.rows),
        "blocker": blocker or None,
        "preview_only": True,
        "final_deadline_freeze": False,
        "final_pre_deadline_snapshot_frozen": False,
        "safety": {
            "opening_squad_optimizer_used": False,
            "legacy_weekly_transfer_optimizer_used": False,
            "legacy_weekly_lineup_preview_used": False,
            "team_alex_consumed_by_model_team": False,
            "target_gw_actuals_consumed": False,
            "auto_final_freeze": False,
        },
    }
    return write_json_new(package_dir / "run_manifest.json", payload)



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


def _run_pre_legacy_compat(
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
                    raise RuntimeError("Prediction publish completed without a receipt path.")
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



def run_pre(
    args: argparse.Namespace,
    repo_root: Path,
    planning_root: Path,
    python_exe: str,
    env: Mapping[str, str],
    recorder: StageRecorder,
) -> Path:
    """Run the Day128B formal PRE path through an immutable candidate snapshot."""

    if int(args.target_gw) < 2:
        raise RuntimeError(
            "Day128B is the rolling GW2+ PRE path; GW1 remains the opening-squad lifecycle."
        )
    if int(args.max_transfers) < 0:
        raise RuntimeError("--max-transfers must be >= 0.")

    prior_season = args.prior_season
    if not prior_season:
        year = int(args.season.split("_")[0])
        prior_season = "%s_%02d" % (year - 1, year % 100)

    api_proc: Optional[subprocess.Popen] = None
    try:
        # Stage 1/2: reuse the existing ingest adapter exactly; do not duplicate API logic.
        if args.skip_live_refresh:
            started = time.time()
            recorder.add(
                "live_ingest",
                "REUSED",
                started,
                dependencies=(),
                planned_inputs=("canonical current DB state",),
                planned_outputs=("live_players", "live_teams", "live_fixtures", "completed_actuals"),
                note="--skip-live-refresh explicitly requested.",
            )
        else:
            if args.base_url:
                base_url = args.base_url.rstrip("/")
                if not wait_for_api(base_url, args.season, timeout_seconds=5.0):
                    raise RuntimeError(
                        "Configured backend is not reachable: %s" % base_url
                    )
            else:
                api_proc, base_url = start_private_api(
                    repo_root / "backend",
                    python_exe,
                    env,
                    args.api_port,
                    args.season,
                )
            started = time.time()
            try:
                refresh_live_data(base_url, recorder, False)
            except Exception as exc:
                recorder.add(
                    "live_ingest",
                    "FAILED",
                    started,
                    dependencies=(),
                    planned_inputs=("FPL API", "season", "target_gw"),
                    planned_outputs=("live_players", "live_teams", "live_fixtures", "completed_actuals"),
                    note=str(exc),
                )
                raise
            recorder.add(
                "live_ingest",
                "PASS",
                started,
                dependencies=(),
                planned_inputs=("FPL API", "season", "target_gw"),
                planned_outputs=("live_players", "live_teams", "live_fixtures", "completed_actuals"),
            )

        # Stage 3: resolve prediction mode through the existing resolver.
        mode_result = _formal_stage(
            recorder,
            "prediction_mode",
            dependencies=("live_ingest",),
            planned_inputs=("season", "target_gw", "prior_season"),
            planned_outputs=("resolved_prediction_mode",),
            function=lambda: resolve_runner_prediction_mode(
                season=args.season,
                target_gw=args.target_gw,
                prior_season=prior_season,
                stabilization_gw=args.stabilization_gw,
            ),
        )
        resolved_mode = str(mode_result["resolved_prediction_mode"])

        # Stages 4/5: current safe producer emits Player + Match + scoreline together.
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
            dry_run=False,
        )
        prediction_info = validate_prediction_run(
            prediction_run,
            season=args.season,
            target_gw=args.target_gw,
            resolved_mode=resolved_mode,
        )

        started = time.time()
        recorder.add(
            "player_model",
            "PASS",
            started,
            outputs=[str(prediction_info["player_csv"])],
            dependencies=("prediction_mode",),
            planned_inputs=("resolved_prediction_mode", "live_players", "prior_evidence"),
            planned_outputs=("target_gw_player_model_artifact",),
            note="Produced by existing early-season prediction pipeline.",
        )
        started = time.time()
        recorder.add(
            "match_model",
            "PASS",
            started,
            outputs=[
                str(prediction_info["match_csv"]),
                str(prediction_info["scoreline_csv"]),
            ],
            dependencies=("prediction_mode",),
            planned_inputs=("resolved_prediction_mode", "live_fixtures", "prior_evidence"),
            planned_outputs=("target_gw_match_model_artifact", "scoreline_artifact"),
            note="Produced by existing early-season prediction pipeline.",
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
                dry_run=False,
            )
            if publish_receipt is None:
                raise RuntimeError(
                    "Prediction publish completed without a receipt path."
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
                dry_run=False,
            )

        package_dir = candidate_package_dir(
            planning_root,
            args.season,
            args.target_gw,
        )
        package_dir.mkdir(parents=True, exist_ok=False)

        write_json_new(package_dir / "prediction_mode.json", mode_result)

        # Stage 5: target-GW horizon is mandatory; later GWs remain optional.
        horizon = _formal_stage(
            recorder,
            "prediction_horizon",
            dependencies=("player_model", "match_model"),
            planned_inputs=(
                str(prediction_info["player_csv"]),
                str(prediction_info["match_csv"]),
                str(prediction_info["scoreline_csv"]),
            ),
            planned_outputs=(
                "current_market.csv",
                "player_prediction_horizon.csv",
                "fixture_prediction_horizon.csv",
            ),
            function=lambda: build_target_gw_horizon_artifacts(
                player_csv=prediction_info["player_csv"],
                match_csv=prediction_info["match_csv"],
                scoreline_csv=prediction_info["scoreline_csv"],
                target_gw=args.target_gw,
                out_dir=package_dir,
            ),
            output_paths=lambda result: [
                str(result["market_csv"]),
                str(result["player_horizon_csv"]),
                str(result["fixture_horizon_csv"]),
            ],
        )

        # Stage 6a: ownership always starts from a previous FINAL frozen state.
        previous_state, previous_state_path, previous_source_kind = _formal_stage(
            recorder,
            "owned_squad_state",
            dependencies=("live_ingest",),
            planned_inputs=("previous_frozen_model_team_state", "current_market.csv"),
            planned_outputs=("current_owned_squad_state",),
            function=lambda: discover_previous_model_team_state(
                planning_root,
                season=args.season,
                target_gw=args.target_gw,
            ),
            output_paths=lambda result: [str(result[1])],
        )
        current_owned = build_current_owned_state_for_candidates(
            previous_state,
            horizon["market_rows"],
            target_gw=args.target_gw,
        )
        current_owned_path = write_json_new(
            package_dir / "current_owned_squad_state.json",
            current_owned,
        )

        # Stage 6b: policy-driven FT ledger.
        ledger_state = _formal_stage(
            recorder,
            "free_transfer_ledger",
            dependencies=("owned_squad_state",),
            planned_inputs=("previous_frozen_model_team_state", "target_season_transfer_policy"),
            planned_outputs=("free_transfer_ledger_state",),
            function=lambda: build_ledger_state(
                season=args.season,
                state_kind="model_team",
                gameweek=args.target_gw,
                available_free_transfers=int(previous_state.free_transfers.count),
            ),
        )
        ledger_path = write_json_new(
            package_dir / "free_transfer_ledger_state.json",
            ledger_state,
        )

        # Stage 6c: Day126B candidates from owned state/current prices/horizon.
        candidate_report = _formal_stage(
            recorder,
            "transfer_candidates",
            dependencies=(
                "player_model",
                "prediction_horizon",
                "owned_squad_state",
                "free_transfer_ledger",
            ),
            planned_inputs=(
                "current_owned_squad_state",
                "current_market",
                "player_prediction_horizon",
            ),
            planned_outputs=("legal_transfer_candidate_set",),
            function=lambda: generate_transfer_candidates(
                current_owned,
                horizon["market_rows"],
                horizon["horizon_rows"],
                pruning_policy=CandidatePruningPolicy(
                    max_pair_candidates_per_out=int(args.top_n),
                ),
            ),
        )
        candidates_path = write_json_new(
            package_dir / "transfer_candidates.json",
            candidate_report,
        )

        # Stage 6d + 7: Day127A chooses transfers and already re-optimizes XI/C/VC.
        decision = _formal_stage(
            recorder,
            "transfer_decision",
            dependencies=("transfer_candidates", "free_transfer_ledger"),
            planned_inputs=(
                "previous_frozen_model_team_state",
                "legal_transfer_candidate_set",
                "target_gw_player_predictions",
                "free_transfer_ledger_state",
            ),
            planned_outputs=("chosen_transfer_or_no_transfer_plan",),
            function=lambda: optimize_single_gw_transfers(
                previous_state,
                candidate_report,
                horizon["horizon_rows"],
                ledger_state,
                max_transfers=int(args.max_transfers),
                current_owned_state=current_owned,
            ),
        )
        decision_path = write_json_new(
            package_dir / "transfer_decision.json",
            decision,
        )

        winner = decision.get("winner")
        if not isinstance(winner, Mapping) or not isinstance(
            winner.get("lineup"), Mapping
        ):
            raise RuntimeError(
                "Day127A output is missing winner.lineup; cannot form PRE candidate."
            )
        started = time.time()
        recorder.add(
            "lineup_selection",
            "PASS",
            started,
            outputs=[str(decision_path)],
            dependencies=("transfer_decision", "player_model"),
            planned_inputs=("chosen_transfer_or_no_transfer_plan",),
            planned_outputs=("reoptimized_xi", "captain", "vice_captain", "ordered_bench"),
            note="Reused Day127A winner.lineup; legacy weekly_lineup_preview was not called.",
        )

        # Stage 8: immutable Day127B candidate only, never FINAL.
        match_bundle = build_match_model_source_bundle(
            out_dir=package_dir / "match_model_source",
            match_csv=prediction_info["match_csv"],
            scoreline_csv=prediction_info["scoreline_csv"],
            fixture_horizon_json=horizon["fixture_horizon_json"],
            prediction_run_id=prediction_run.name,
        )
        deadline = target_gw_deadline_from_bootstrap(
            prediction_info["bootstrap_json"],
            args.target_gw,
        )
        as_of_time = str(prediction_info["created_at"])

        player_spec = {
            "run_id": prediction_run.name,
            "artifact_kind": "target_gw_player_model",
            "path": str(prediction_info["player_csv"]),
            "season": args.season,
            "target_gw": int(args.target_gw),
            "as_of_utc": as_of_time,
        }
        match_spec = {
            "run_id": prediction_run.name,
            "artifact_kind": "target_gw_match_model_plus_scoreline",
            "path": str(match_bundle),
            "season": args.season,
            "target_gw": int(args.target_gw),
            "as_of_utc": as_of_time,
        }

        snapshot_result = _formal_stage(
            recorder,
            "pre_deadline_candidate",
            dependencies=(
                "player_model",
                "match_model",
                "owned_squad_state",
                "free_transfer_ledger",
                "transfer_decision",
                "lineup_selection",
            ),
            planned_inputs=(
                "player_model_artifact",
                "match_model_artifact",
                "previous_frozen_model_team_state",
                "chosen_transfer_or_no_transfer_plan",
                "free_transfer_ledger_state",
            ),
            planned_outputs=("immutable_pre_deadline_candidate_snapshot",),
            function=lambda: export_gameweek_pre_deadline_snapshot(
                artifact_root=package_dir / "pre-deadline-snapshot",
                season=args.season,
                target_gw=args.target_gw,
                as_of_time=as_of_time,
                fpl_deadline_time=deadline,
                player_model_artifact=player_spec,
                match_model_artifact=match_spec,
                previous_model_team_state=previous_state,
                chosen_plan=decision,
                transfer_ledger_state=ledger_state,
                current_model_team_state=current_owned,
                team_alex_reference=None,
                final_freeze=False,
                run_id=package_dir.name + "_snapshot",
            ),
            output_paths=lambda result: [
                str(result["snapshot_dir"]),
                str(result["manifest_path"]),
            ],
        )
        if str(snapshot_result.get("snapshot_kind")) != SNAPSHOT_KIND_CANDIDATE:
            raise RuntimeError("PRE exporter did not return candidate snapshot kind.")
        if snapshot_result.get("final_pre_deadline_snapshot_frozen") is not False:
            raise RuntimeError("PRE must never create a FINAL freeze.")

        # Stage 9: machine-readable unified stage manifest.
        manifest_path = write_formal_pre_manifest(
            package_dir=package_dir,
            season=args.season,
            target_gw=args.target_gw,
            prior_season=prior_season,
            prediction_mode_result=mode_result,
            prediction_info=prediction_info,
            previous_state=previous_state,
            previous_state_path=previous_state_path,
            previous_state_source_kind=previous_source_kind,
            current_owned_state_path=current_owned_path,
            horizon=horizon,
            ledger_state=ledger_state,
            candidates_path=candidates_path,
            decision_path=decision_path,
            snapshot_result=snapshot_result,
            recorder=recorder,
            publish_requested=args.publish_predictions,
            max_transfers=args.max_transfers,
        )

        # Small human-readable pointer without duplicating decision/model logic.
        summary_lines = [
            "# FPL Unified PRE Candidate",
            "",
            "- Runner: `%s`" % RUNNER_VERSION,
            "- Season: `%s`" % args.season,
            "- Target GW: **%s**" % args.target_gw,
            "- Prediction mode: `%s`" % resolved_mode,
            "- Previous frozen state: `%s`" % previous_state.state_id,
            "- Previous state source: `%s`" % previous_source_kind,
            "- Decision: **%s**" % winner.get("action"),
            "- Transfer count: **%s**" % winner.get("transfer_count"),
            "- Net gain vs NO TRANSFER: **%s**" % winner.get("net_gain_vs_no_transfer"),
            "- Candidate snapshot: `%s`" % snapshot_result.get("snapshot_dir"),
            "- FINAL freeze: **False**",
            "",
            "This package is PRE only. It does not silently promote itself to FINAL.",
            "",
        ]
        (package_dir / "summary.md").write_text(
            "\n".join(summary_lines),
            encoding="utf-8",
        )

        print("\n=== FPL Unified PRE Complete ===")
        print("candidate_package:", package_dir)
        print("manifest:", manifest_path)
        print("decision:", decision_path)
        print("snapshot:", snapshot_result["snapshot_dir"])
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

    validate_phase_request(args.phase, final_freeze=bool(args.final_freeze))

    if args.dry_run:
        plan = build_dag_plan(
            phase=args.phase,
            season=args.season,
            target_gw=args.target_gw,
            final_freeze=bool(args.final_freeze),
            resume=bool(args.resume),
            publish_predictions=bool(args.publish_predictions),
            skip_live_refresh=bool(args.skip_live_refresh),
        )
        print(render_dag_plan(plan))
        return

    if args.phase == "status":
        print_status(planning_root, args.season, args.target_gw)
        return

    if args.phase == "freeze":
        raise RuntimeError(
            "Day128A defines and validates the FINAL-freeze DAG adapter but does not "
            "wire live freeze execution yet. The explicit Day127B exporter remains the "
            "formal freeze producer until a later runner-integration milestone."
        )

    if args.phase == "auto" and args.final_freeze:
        raise RuntimeError(
            "Day128A can plan --phase auto --final-freeze with --dry-run, but live "
            "FINAL-freeze execution is intentionally fail-closed until runner integration."
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
