from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import text

from app.core.db import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight refresh-pipeline data quality checks."
    )
    parser.add_argument("--target-gw", type=int, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional path to save JSON report, e.g. docs/examples/day48_refresh_check_report.json",
    )
    return parser.parse_args()


def make_check(name: str, passed: bool, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "details": details,
    }


def run_scalar_int(db, sql: str, params: Dict[str, Any] = None) -> int:
    value = db.execute(text(sql), params or {}).scalar()
    return int(value or 0)


def run_checks(target_gw: int, model_name: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        checks: List[Dict[str, Any]] = []

        # 1) no null critical ids after ingest
        players_null_id = run_scalar_int(
            db,
            "SELECT COUNT(*) FROM players WHERE id IS NULL"
        )
        teams_null_id = run_scalar_int(
            db,
            "SELECT COUNT(*) FROM teams WHERE id IS NULL"
        )
        fixtures_null_id = run_scalar_int(
            db,
            "SELECT COUNT(*) FROM fixtures WHERE id IS NULL"
        )
        predictions_null_key = run_scalar_int(
            db,
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE player_id IS NULL
               OR target_gw IS NULL
               OR model_name IS NULL
            """
        )

        null_total = players_null_id + teams_null_id + fixtures_null_id + predictions_null_key
        checks.append(
            make_check(
                name="no_null_critical_ids",
                passed=(null_total == 0),
                details={
                    "players_null_id": players_null_id,
                    "teams_null_id": teams_null_id,
                    "fixtures_null_id": fixtures_null_id,
                    "predictions_null_key_fields": predictions_null_key,
                },
            )
        )

        # 2) no duplicate prediction keys where uniqueness is expected
        duplicate_prediction_keys = run_scalar_int(
            db,
            """
            SELECT COUNT(*)
            FROM (
                SELECT player_id, target_gw, model_name, COUNT(*) AS n
                FROM predictions
                GROUP BY player_id, target_gw, model_name
                HAVING COUNT(*) > 1
            ) dup
            """
        )
        checks.append(
            make_check(
                name="no_duplicate_prediction_keys",
                passed=(duplicate_prediction_keys == 0),
                details={
                    "duplicate_prediction_key_groups": duplicate_prediction_keys,
                    "expected_unique_key": ["player_id", "target_gw", "model_name"],
                },
            )
        )

        # 3) fixtures.gw coverage for rows with kickoff_time
        fixtures_missing_gw_with_kickoff = run_scalar_int(
            db,
            """
            SELECT COUNT(*)
            FROM fixtures
            WHERE kickoff_time IS NOT NULL
              AND gw IS NULL
            """
        )
        fixtures_with_kickoff = run_scalar_int(
            db,
            """
            SELECT COUNT(*)
            FROM fixtures
            WHERE kickoff_time IS NOT NULL
            """
        )
        checks.append(
            make_check(
                name="fixtures_gw_coverage_for_kickoff_rows",
                passed=(fixtures_missing_gw_with_kickoff == 0),
                details={
                    "fixtures_with_kickoff_time": fixtures_with_kickoff,
                    "fixtures_missing_gw_with_kickoff_time": fixtures_missing_gw_with_kickoff,
                },
            )
        )

        # 4) prediction counts present for target GW / model
        prediction_count_for_target = run_scalar_int(
            db,
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE target_gw = :target_gw
              AND model_name = :model_name
            """,
            {"target_gw": target_gw, "model_name": model_name},
        )
        checks.append(
            make_check(
                name="prediction_counts_present_for_target_gw_model",
                passed=(prediction_count_for_target > 0),
                details={
                    "target_gw": target_gw,
                    "model_name": model_name,
                    "prediction_row_count": prediction_count_for_target,
                },
            )
        )

        overall_passed = all(c["passed"] for c in checks)

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_gw": target_gw,
            "model_name": model_name,
            "overall_passed": overall_passed,
            "checks": checks,
        }
        return report
    finally:
        db.close()


def print_summary(report: Dict[str, Any]) -> None:
    print("=== Refresh Validation Summary ===")
    print("target_gw:", report["target_gw"])
    print("model_name:", report["model_name"])
    print("overall_passed:", report["overall_passed"])
    print()

    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}")
        for k, v in check["details"].items():
            print(f"  - {k}: {v}")
        print()


def maybe_save_report(report: Dict[str, Any], out_path: str) -> None:
    if not out_path:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved_report:", path)


def main() -> None:
    args = parse_args()
    report = run_checks(
        target_gw=args.target_gw,
        model_name=args.model_name,
    )
    print_summary(report)
    maybe_save_report(report, args.out)

    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()