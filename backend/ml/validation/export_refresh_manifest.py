from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


REFRESH_MANIFEST_VERSION = "fpl_unified_refresh_manifest_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def stable_fingerprint(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def refresh_run_dir(
    planning_root: Path,
    *,
    season: str,
    target_gw: int,
    run_id: str,
) -> Path:
    return (
        Path(planning_root)
        / "refresh-runs"
        / str(season)
        / ("gw%02d" % int(target_gw))
        / str(run_id)
    )


def build_initial_manifest(
    *,
    run_id: str,
    runner_version: str,
    dag_version: str,
    season: str,
    target_gw: int,
    requested_phase: str,
    resume_requested: bool,
    final_freeze_requested: bool,
    discovered_state: Optional[Mapping[str, Any]] = None,
    candidate_run_id: Optional[str] = None,
    active_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    created = utc_now()
    return {
        "manifest_version": REFRESH_MANIFEST_VERSION,
        "run_id": str(run_id),
        "runner_version": str(runner_version),
        "dag_version": str(dag_version),
        "season": str(season),
        "target_gw": int(target_gw),
        "requested_phase": str(requested_phase),
        "resolved_action": None,
        "status": "RUNNING",
        "resume_requested": bool(resume_requested),
        "final_freeze_requested": bool(final_freeze_requested),
        "created_at_utc": created,
        "updated_at_utc": created,
        "warnings": [],
        "blockers": [],
        "candidate_run_id": candidate_run_id,
        "active_run_id": active_run_id,
        "discovered_state": dict(discovered_state or {}),
        "stage_results": [],
        "outputs": {},
        "failure": None,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.%s" % os.getpid())
    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


class RefreshManifestWriter:
    """Mutable run journal; it points to immutable PRE/FINAL/POST artifacts.

    The journal itself may be atomically updated while one invocation is running.
    It never mutates the historical artifacts it references.
    """

    def __init__(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.payload: Dict[str, Any] = dict(payload)
        _atomic_write_json(self.path, self.payload)

    @classmethod
    def create(
        cls,
        *,
        planning_root: Path,
        runner_version: str,
        dag_version: str,
        season: str,
        target_gw: int,
        requested_phase: str,
        resume_requested: bool,
        final_freeze_requested: bool,
        discovered_state: Optional[Mapping[str, Any]] = None,
        candidate_run_id: Optional[str] = None,
        active_run_id: Optional[str] = None,
    ) -> "RefreshManifestWriter":
        run_id = "%s_%s_gw%s_%s" % (
            str(runner_version),
            str(season),
            int(target_gw),
            utc_stamp(),
        )
        root = refresh_run_dir(
            planning_root,
            season=season,
            target_gw=target_gw,
            run_id=run_id,
        )
        payload = build_initial_manifest(
            run_id=run_id,
            runner_version=runner_version,
            dag_version=dag_version,
            season=season,
            target_gw=target_gw,
            requested_phase=requested_phase,
            resume_requested=resume_requested,
            final_freeze_requested=final_freeze_requested,
            discovered_state=discovered_state,
            candidate_run_id=candidate_run_id,
            active_run_id=active_run_id,
        )
        return cls(root / "refresh_manifest.json", payload)

    def add_warning(self, message: str) -> None:
        warnings = list(self.payload.get("warnings") or [])
        if message and message not in warnings:
            warnings.append(str(message))
        self.payload["warnings"] = warnings
        self.payload["updated_at_utc"] = utc_now()
        _atomic_write_json(self.path, self.payload)

    def add_blocker(self, message: str) -> None:
        blockers = list(self.payload.get("blockers") or [])
        if message and message not in blockers:
            blockers.append(str(message))
        self.payload["blockers"] = blockers
        self.payload["updated_at_utc"] = utc_now()
        _atomic_write_json(self.path, self.payload)

    def update(self, **fields: Any) -> None:
        for key, value in fields.items():
            self.payload[str(key)] = value
        self.payload["updated_at_utc"] = utc_now()
        _atomic_write_json(self.path, self.payload)


def manifest_stage_summary(stage_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(row) for row in stage_results]
    latest_success = None
    failed = None
    reused = 0
    skipped = 0
    duration = 0.0
    for row in rows:
        duration += float(row.get("duration_seconds") or 0.0)
        status = str(row.get("status") or "")
        if status in ("PASS", "REUSED"):
            latest_success = row.get("stage")
        if status == "FAILED":
            failed = row.get("stage")
        if status == "REUSED":
            reused += 1
        if status == "SKIPPED":
            skipped += 1
    return {
        "stage_count": len(rows),
        "latest_successful_stage": latest_success,
        "failed_stage": failed,
        "reused_stage_count": reused,
        "skipped_stage_count": skipped,
        "total_stage_duration_seconds": round(duration, 3),
    }
