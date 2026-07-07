from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.core.db import SessionLocal
from app.core.season import get_current_season


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
    parser.add_argument(
        "--skip-prediction-count-check",
        action="store_true",
        help="Skip the target_gw/model prediction count check.",
    )
    return parser.parse_args()


def make_check(name: str, passed: bool, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "details": details,
    }


def run_scalar_int(db, sql: str, params: Optional[Dict[str, Any]] = None) -> int:
    value = db.execute(text(sql), params or {}).scalar()
    return int(value or 0)


def run_checks(
    target_gw: int,
    model_name: str,
    require_prediction_count_check: bool = True,
) -> Dict[str, Any]:
    season = get_current_season()
    db = SessionLocal()
    try:
        checks: List[Dict[str, Any]] = []

        players_null_id = run_scalar_int(
            db,
            "SELECT COUNT(*) FROM players WHERE id IS NULL",
        )
        teams_null_id = run_scalar_int(
            db,
            "SELECT COUNT(*) FROM teams WHERE id IS NULL",
        )
        fixtures_null_id = run_scalar_int(
            db,
            """
            SELECT COUNT(*)
            FROM fixtures
            WHERE season = :season
              AND id IS NULL
            """,
            {"season": season},
        )
        predictions_null_key = run_scalar_int(
            db,
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE season = :season
              AND (
                   player_id IS NULL
                OR target_gw IS NULL
                OR model_name IS NULL
                OR season IS NULL
              )
            """,
            {"season": season},
        )

        null_total = players_null_id + teams_null_id + fixtures_null_id + predictions_null_key
        checks.append(
            make_check(
                name="no_null_critical_ids",
                passed=(null_total == 0),
                details={
                    "season": season,
                    "players_null_id": players_null_id,
                    "teams_null_id": teams_null_id,
                    "fixtures_null_id": fixtures_null_id,
                    "predictions_null_key_fields": predictions_null_key,
                },
            )
        )

        duplicate_prediction_keys = run_scalar_int(
            db,
            """
            SELECT COUNT(*)
            FROM (
                SELECT season, player_id, target_gw, model_name, COUNT(*) AS n
                FROM predictions
                WHERE season = :season
                GROUP BY season, player_id, target_gw, model_name
                HAVING COUNT(*) > 1
            ) dup
            """,
            {"season": season},
        )
        checks.append(
            make_check(
                name="no_duplicate_prediction_keys",
                passed=(duplicate_prediction_keys == 0),
                details={
                    "season": season,
                    "duplicate_prediction_key_groups": duplicate_prediction_keys,
                    "expected_unique_key": ["season", "player_id", "target_gw", "model_name"],
                },
            )
        )

        fixtures_missing_gw_with_kickoff = run_scalar_int(
            db,
            """
            SELECT COUNT(*)
            FROM fixtures
            WHERE season = :season
              AND kickoff_time IS NOT NULL
              AND gw IS NULL
            """,
            {"season": season},
        )
        fixtures_with_kickoff = run_scalar_int(
            db,
            """
            SELECT COUNT(*)
            FROM fixtures
            WHERE season = :season
              AND kickoff_time IS NOT NULL
            """,
            {"season": season},
        )
        checks.append(
            make_check(
                name="fixtures_gw_coverage_for_kickoff_rows",
                passed=(fixtures_missing_gw_with_kickoff == 0),
                details={
                    "season": season,
                    "fixtures_with_kickoff_time": fixtures_with_kickoff,
                    "fixtures_missing_gw_with_kickoff_time": fixtures_missing_gw_with_kickoff,
                },
            )
        )

        if require_prediction_count_check:
            prediction_count_for_target = run_scalar_int(
                db,
                """
                SELECT COUNT(*)
                FROM predictions
                WHERE season = :season
                  AND target_gw = :target_gw
                  AND model_name = :model_name
                """,
                {
                    "season": season,
                    "target_gw": target_gw,
                    "model_name": model_name,
                },
            )
            checks.append(
                make_check(
                    name="prediction_counts_present_for_target_gw_model",
                    passed=(prediction_count_for_target > 0),
                    details={
                        "season": season,
                        "target_gw": target_gw,
                        "model_name": model_name,
                        "prediction_row_count": prediction_count_for_target,
                    },
                )
            )
        else:
            checks.append(
                make_check(
                    name="prediction_counts_present_for_target_gw_model",
                    passed=True,
                    details={
                        "season": season,
                        "target_gw": target_gw,
                        "model_name": model_name,
                        "prediction_row_count": None,
                        "skipped": True,
                        "reason": "pre-refresh validation phase",
                    },
                )
            )

        overall_passed = all(c["passed"] for c in checks)

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "season": season,
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
    print("season:", report["season"])
    print("target_gw:", report["target_gw"])
    print("model_name:", report["model_name"])
    print("overall_passed:", report["overall_passed"])
    print()

    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print("[%s] %s" % (status, check["name"]))
        for k, v in check["details"].items():
            print("  - %s: %s" % (k, v))
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
        require_prediction_count_check=not args.skip_prediction_count_check,
    )
    print_summary(report)
    maybe_save_report(report, args.out)

    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()