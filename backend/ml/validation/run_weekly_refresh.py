from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import text

from app.core.db import SessionLocal
from app.utils.run_snapshot_store import (
    create_run_snapshot_artifact,
    save_run_snapshot_artifact,
)
from ml.validation.run_refresh_checks import run_checks


DEFAULT_DATABASE_URL = "postgresql://app:app@localhost:5432/epl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the main weekly refresh path from data update to predictions."
    )
    parser.add_argument("--target-gw", type=int, required=True)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument(
        "--database-url",
        type=str,
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        help="Database URL passed to child prediction commands via env.",
    )

    parser.add_argument("--validation-model-name", type=str, default="baseline_rollavg_v0")

    parser.add_argument(
        "--player-refresh-command",
        action="append",
        default=[],
        help=(
            "Repeatable shell-style command template for player prediction refresh. "
            "Supports {target_gw} and {last_actual_gw}. "
            "If omitted, baseline_rollavg_v0 and baseline_rollavg_v1 defaults are used."
        ),
    )
    parser.add_argument(
        "--match-refresh-command",
        action="append",
        default=[],
        help=(
            "Repeatable shell-style command template for match prediction refresh. "
            "Supports {target_gw}. "
            "If omitted, a default /match/predictions/run_gw call is used."
        ),
    )

    parser.add_argument("--include-ridge-next-gw", action="store_true")
    parser.add_argument("--ridge-split-gw", type=int, default=22)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)

    parser.add_argument("--match-model-name", type=str, default="match_baseline_v0")
    parser.add_argument("--match-n", type=int, default=5)
    parser.add_argument("--match-threshold", type=float, default=0.30)

    parser.add_argument(
        "--live-gw",
        type=int,
        default=None,
        help="Optional specific GW live ingest step after finished-gw ingest.",
    )

    parser.add_argument("--save-refresh-snapshot", action="store_true")
    parser.add_argument("--save-decision-snapshot", action="store_true")
    parser.add_argument("--decision-endpoint", type=str, default=None)
    parser.add_argument("--decision-scenario-type", type=str, default=None)

    return parser.parse_args()


def print_run(step: str, detail: str) -> None:
    print(f"[RUN] {step}")
    print(f"  {detail}")
    sys.stdout.flush()


def print_pass(step: str) -> None:
    print(f"[PASS] {step}")
    sys.stdout.flush()


def run_http_post(name: str, url: str) -> Dict[str, Any]:
    print_run(name, f"url: {url}")
    resp = httpx.post(url, timeout=90.0)
    if resp.status_code >= 400:
        raise RuntimeError(f"{name} failed with status {resp.status_code}: {resp.text}")
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw_text": resp.text}
    print_pass(name)
    return {
        "step": name,
        "ok": True,
        "kind": "http_post",
        "url": url,
        "payload": payload,
    }


def run_shell_command(
    name: str,
    command_template: str,
    fmt: Dict[str, Any],
    child_env: Dict[str, str],
) -> Dict[str, Any]:
    command = command_template.format(**fmt)
    parts = shlex.split(command)

    print_run(name, f"command: {command}")

    proc = subprocess.run(
        parts,
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)

    print_pass(name)
    return {
        "step": name,
        "ok": True,
        "kind": "command",
        "command": command,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def get_last_actual_gw() -> Optional[int]:
    db = SessionLocal()
    try:
        value = db.execute(text("SELECT MAX(gw) FROM player_gw_stats")).scalar()
        if value is None:
            return None
        return int(value)
    finally:
        db.close()


def build_default_player_commands(
    *,
    target_gw: int,
    last_actual_gw: Optional[int],
    include_ridge_next_gw: bool,
    ridge_split_gw: int,
    ridge_alpha: float,
) -> List[str]:
    commands = [
        "python -m ml.predict.run_baseline_rollavg_v0 --target-gw {target_gw} --window 5",
        "python -m ml.predict.run_baseline_rollavg_v1 --target-gw {target_gw}",
    ]

    if include_ridge_next_gw:
        if last_actual_gw is None:
            raise RuntimeError("Cannot run ridge next-GW forecast: last_actual_gw is unknown.")
        expected_next_gw = last_actual_gw + 1
        if target_gw != expected_next_gw:
            raise RuntimeError(
                "Cannot run ridge next-GW forecast in default mode: "
                f"target_gw={target_gw}, but last_actual_gw={last_actual_gw} so expected next gw is {expected_next_gw}."
            )
        commands.append(
            "python -m ml.predict.predict_next_gw_ridge_rollform_v1 "
            f"--last_gw {{last_actual_gw}} --split_gw {ridge_split_gw} --alpha {ridge_alpha}"
        )

    return commands


def build_default_match_commands(
    *,
    base_url: str,
    match_model_name: str,
    match_n: int,
    match_threshold: float,
) -> List[str]:
    return [
        (
            "curl -s -X POST "
            f"'{base_url}/match/predictions/run_gw?"
            "gw={target_gw}"
            f"&n={match_n}"
            f"&model_name={match_model_name}"
            f"&threshold={match_threshold:.2f}'"
        )
    ]


def run_validation_step(
    *,
    step_name: str,
    target_gw: int,
    model_name: str,
    require_prediction_count_check: bool,
) -> Dict[str, Any]:
    mode = "post-refresh validation" if require_prediction_count_check else "pre-refresh validation"
    print_run(step_name, mode)
    report = run_checks(
        target_gw=target_gw,
        model_name=model_name,
        require_prediction_count_check=require_prediction_count_check,
    )
    if not report["overall_passed"]:
        raise RuntimeError(f"{step_name} failed: {report}")
    print_pass(step_name)
    return {
        "step": step_name,
        "ok": True,
        "kind": "validation",
        "report": report,
    }


def extract_validation_row_counts(report: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for check in report.get("checks", []):
        name = check.get("name")
        details = check.get("details", {})

        if name == "fixtures_gw_coverage_for_kickoff_rows":
            out["fixtures_with_kickoff_time"] = int(details.get("fixtures_with_kickoff_time", 0))
            out["fixtures_missing_gw_with_kickoff_time"] = int(
                details.get("fixtures_missing_gw_with_kickoff_time", 0)
            )
        elif name == "prediction_counts_present_for_target_gw_model":
            pred_count = details.get("prediction_row_count")
            out["prediction_row_count"] = int(pred_count or 0)

    return out


def maybe_save_refresh_snapshot(
    *,
    target_gw: int,
    model_name: str,
    validation_report: Dict[str, Any],
) -> Optional[str]:
    row_counts = extract_validation_row_counts(validation_report)
    artifact = create_run_snapshot_artifact(
        snapshot_type="refresh_validation",
        source_command=f"python -m ml.validation.run_weekly_refresh --target-gw {target_gw}",
        target_gw=target_gw,
        model_name=model_name,
        row_counts=row_counts,
        metadata={
            "overall_passed": validation_report.get("overall_passed", False),
            "checks": validation_report.get("checks", []),
        },
        notes="Weekly refresh orchestrator validation snapshot.",
    )
    path = save_run_snapshot_artifact(artifact)
    return str(path)


def maybe_save_decision_snapshot(
    *,
    target_gw: int,
    model_name: str,
    decision_endpoint: Optional[str],
    decision_scenario_type: Optional[str],
) -> Optional[str]:
    if not decision_endpoint or not decision_scenario_type:
        return None

    artifact = create_run_snapshot_artifact(
        snapshot_type="decision_support",
        source_command="weekly_refresh_optional_decision_snapshot",
        endpoint=decision_endpoint,
        scenario_type=decision_scenario_type,
        target_gw=target_gw,
        model_name=model_name,
        row_counts={},
        metadata={},
        notes="Optional decision-support snapshot recorded by weekly orchestrator.",
    )
    path = save_run_snapshot_artifact(artifact)
    return str(path)


def print_summary(results: List[Dict[str, Any]], last_actual_gw: Optional[int]) -> None:
    print("\n=== Weekly Refresh Summary ===")
    if last_actual_gw is not None:
        print(f"last_actual_gw_after_ingest: {last_actual_gw}")
    for r in results:
        print(f"[PASS] {r['step']}")
        if r["kind"] == "http_post":
            print(f"  url: {r['url']}")
        elif r["kind"] == "command":
            print(f"  command: {r['command']}")
        elif r["kind"] == "validation":
            print(f"  overall_passed: {r['report'].get('overall_passed')}")
        elif r["kind"] == "snapshot":
            print(f"  path: {r['path']}")
    print()


def main() -> None:
    args = parse_args()

    results: List[Dict[str, Any]] = []
    last_actual_gw: Optional[int] = None
    child_env = os.environ.copy()
    child_env["DATABASE_URL"] = args.database_url

    try:
        # 1) refresh gameweeks metadata
        results.append(
            run_http_post(
                "ingest_gameweeks",
                f"{args.base_url}/gameweeks/ingest/fpl",
            )
        )

        # 2) refresh bootstrap (teams / players metadata)
        results.append(
            run_http_post(
                "ingest_bootstrap",
                f"{args.base_url}/ingest/fpl/bootstrap",
            )
        )

        # 3) refresh fixtures
        results.append(
            run_http_post(
                "ingest_fixtures",
                f"{args.base_url}/ingest/fpl/fixtures",
            )
        )

        # 4) ingest finished GW stats (player actuals)
        results.append(
            run_http_post(
                "ingest_finished_gw_stats",
                f"{args.base_url}/ingest/fpl/gw/finished",
            )
        )

        # 5) optional live gw ingest
        if args.live_gw is not None:
            results.append(
                run_http_post(
                    "ingest_live_gw_stats",
                    f"{args.base_url}/ingest/fpl/gw/{args.live_gw}/live",
                )
            )

        # discover latest actual GW after data ingest
        last_actual_gw = get_last_actual_gw()
        print(f"[INFO] last_actual_gw_after_ingest: {last_actual_gw}")
        sys.stdout.flush()

        # 6) pre-refresh validation
        pre_validation_result = run_validation_step(
            step_name="pre_validation",
            target_gw=args.target_gw,
            model_name=args.validation_model_name,
            require_prediction_count_check=False,
        )
        results.append(pre_validation_result)

        fmt: Dict[str, Any] = {
            "target_gw": args.target_gw,
            "last_actual_gw": last_actual_gw if last_actual_gw is not None else "",
        }

        # 7) player prediction refresh
        player_commands = args.player_refresh_command or build_default_player_commands(
            target_gw=args.target_gw,
            last_actual_gw=last_actual_gw,
            include_ridge_next_gw=args.include_ridge_next_gw,
            ridge_split_gw=args.ridge_split_gw,
            ridge_alpha=args.ridge_alpha,
        )

        for idx, command_template in enumerate(player_commands, start=1):
            results.append(
                run_shell_command(
                    f"player_prediction_refresh_{idx}",
                    command_template,
                    fmt,
                    child_env,
                )
            )

        # 8) match prediction refresh
        match_commands = args.match_refresh_command or build_default_match_commands(
            base_url=args.base_url,
            match_model_name=args.match_model_name,
            match_n=args.match_n,
            match_threshold=args.match_threshold,
        )

        for idx, command_template in enumerate(match_commands, start=1):
            results.append(
                run_shell_command(
                    f"match_prediction_refresh_{idx}",
                    command_template,
                    fmt,
                    child_env,
                )
            )

        # 9) post-refresh validation
        post_validation_result = run_validation_step(
            step_name="post_validation",
            target_gw=args.target_gw,
            model_name=args.validation_model_name,
            require_prediction_count_check=True,
        )
        results.append(post_validation_result)

        # 10) optional snapshots
        if args.save_refresh_snapshot:
            print_run("refresh_snapshot", "saving refresh validation snapshot")
            refresh_path = maybe_save_refresh_snapshot(
                target_gw=args.target_gw,
                model_name=args.validation_model_name,
                validation_report=post_validation_result["report"],
            )
            if refresh_path:
                results.append(
                    {
                        "step": "refresh_snapshot",
                        "ok": True,
                        "kind": "snapshot",
                        "path": refresh_path,
                    }
                )
                print_pass("refresh_snapshot")

        if args.save_decision_snapshot:
            print_run("decision_snapshot", "saving optional decision-support snapshot")
            decision_path = maybe_save_decision_snapshot(
                target_gw=args.target_gw,
                model_name=args.validation_model_name,
                decision_endpoint=args.decision_endpoint,
                decision_scenario_type=args.decision_scenario_type,
            )
            if decision_path:
                results.append(
                    {
                        "step": "decision_snapshot",
                        "ok": True,
                        "kind": "snapshot",
                        "path": decision_path,
                    }
                )
                print_pass("decision_snapshot")

        print_summary(results, last_actual_gw)

    except Exception as e:
        print("\n=== Weekly Refresh FAILED ===")
        if last_actual_gw is not None:
            print(f"last_actual_gw_after_ingest: {last_actual_gw}")
        for r in results:
            print(f"[PASS] {r['step']}")
        print(f"[FAIL] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
