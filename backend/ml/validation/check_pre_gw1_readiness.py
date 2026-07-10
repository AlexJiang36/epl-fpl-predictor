from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal
from ml.validation.resolve_prediction_mode import resolve_prediction_mode


POSITIONAL_TABLES = {"players", "teams", "gameweeks", "fixtures", "player_gw_stats"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Pre-GW1 readiness for prior-driven player/match prediction scaffolding. "
            "This is read-only and does not write to the database."
        )
    )
    parser.add_argument("--source-season", required=True, help="Prior/source season, for example 2024_25.")
    parser.add_argument("--target-season", required=True, help="Target season, for example 2025_26.")
    parser.add_argument("--target-gw", type=int, default=1, help="Target GW. Pre-GW1 readiness expects GW1.")
    parser.add_argument(
        "--prediction-mode",
        default="auto",
        choices=["auto", "pre_gw1_prior", "early_season_blend", "normal_weekly"],
        help="Requested prediction mode.",
    )
    parser.add_argument(
        "--prior-csv",
        required=True,
        help="Day65 player prior CSV.",
    )
    parser.add_argument(
        "--mapping-csv",
        required=True,
        help="Day66B identity mapping candidates CSV.",
    )
    parser.add_argument(
        "--day66c-json",
        default="/tmp/day66c_previous_season_prior_join_audit.json",
        help="Day66C prior join dry-run JSON.",
    )
    parser.add_argument(
        "--day66d-json",
        default="/tmp/day66d_multi_season_training_contract.json",
        help="Day66D multi-season training contract JSON.",
    )
    parser.add_argument(
        "--team-prior-csv",
        default="",
        help="Optional team prior artifact. Day67B records this but does not require it yet.",
    )
    parser.add_argument(
        "--min-accepted-mappings",
        type=int,
        default=1,
        help="Minimum accepted auto-approved player mappings required for player prior readiness.",
    )
    parser.add_argument(
        "--min-player-prior-coverage-rate",
        type=float,
        default=0.50,
        help="Minimum Day66C player prior coverage rate required for player prior readiness.",
    )
    parser.add_argument("--out-json", required=True, help="Output JSON report path.")
    parser.add_argument("--out-md", required=True, help="Output Markdown report path.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_file(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        return {
            "exists": False,
            "loaded": False,
            "path": str(path),
            "data": None,
            "error": "file_not_found",
        }
    try:
        return {
            "exists": True,
            "loaded": True,
            "path": str(path),
            "data": json.loads(path.read_text(encoding="utf-8")),
            "error": None,
        }
    except Exception as exc:
        return {
            "exists": True,
            "loaded": False,
            "path": str(path),
            "data": None,
            "error": str(exc),
        }


def nested_get(data: Optional[Dict[str, Any]], keys: Sequence[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def read_sql_dataframe(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    db = SessionLocal()
    try:
        return pd.read_sql(text(sql), db.bind, params=params or {})
    finally:
        db.close()


def db_scalar(sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
    db = SessionLocal()
    try:
        return db.execute(text(sql), params or {}).scalar()
    finally:
        db.close()


def table_exists(table_name: str) -> bool:
    value = db_scalar(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = :table_name
        """,
        {"table_name": table_name},
    )
    return int(value or 0) > 0


def table_columns(table_name: str) -> List[str]:
    if not table_exists(table_name):
        return []
    df = read_sql_dataframe(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = :table_name
        ORDER BY ordinal_position
        """,
        {"table_name": table_name},
    )
    return [str(value) for value in df["column_name"].tolist()]


def first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def count_table_rows(
    table_name: str,
    season: Optional[str] = None,
    gw: Optional[int] = None,
) -> Dict[str, Any]:
    if not table_exists(table_name):
        return {
            "table": table_name,
            "exists": False,
            "count": None,
            "season_filter_applied": False,
            "gw_filter_applied": False,
            "columns": [],
            "error": None,
        }

    columns = table_columns(table_name)
    filters: List[str] = []
    params: Dict[str, Any] = {}

    season_col = first_existing_column(columns, ["season"])
    gw_col = first_existing_column(columns, ["gw", "gameweek", "event"])

    if season is not None and season_col is not None:
        filters.append("%s = :season" % season_col)
        params["season"] = season

    if gw is not None and gw_col is not None:
        filters.append("%s = :gw" % gw_col)
        params["gw"] = gw

    where_sql = ""
    if filters:
        where_sql = " WHERE " + " AND ".join(filters)

    value = db_scalar("SELECT COUNT(*) FROM %s%s" % (table_name, where_sql), params)

    return {
        "table": table_name,
        "exists": True,
        "count": int(value or 0),
        "season_filter_applied": bool(season is not None and season_col is not None),
        "gw_filter_applied": bool(gw is not None and gw_col is not None),
        "season_column": season_col,
        "gw_column": gw_col,
        "columns": columns,
        "error": None,
    }


def load_prior_csv(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "loaded": False,
            "row_count": 0,
            "raw_player_id_unique_count": 0,
            "duplicate_raw_player_id_count": None,
            "columns": [],
            "data": None,
            "error": "file_not_found",
        }

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {
            "exists": True,
            "path": str(path),
            "loaded": False,
            "row_count": 0,
            "raw_player_id_unique_count": 0,
            "duplicate_raw_player_id_count": None,
            "columns": [],
            "data": None,
            "error": str(exc),
        }

    if "raw_player_id" in df.columns:
        raw_ids = df["raw_player_id"].astype(str)
        duplicate_count = int(raw_ids.duplicated().sum())
        unique_count = int(raw_ids.nunique(dropna=True))
    else:
        duplicate_count = None
        unique_count = 0

    return {
        "exists": True,
        "path": str(path),
        "loaded": True,
        "row_count": int(len(df)),
        "raw_player_id_unique_count": unique_count,
        "duplicate_raw_player_id_count": duplicate_count,
        "columns": [str(col) for col in df.columns],
        "data": df,
        "error": None,
    }


def load_mapping_csv(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "loaded": False,
            "row_count": 0,
            "top_row_count": 0,
            "accepted_mapping_rows": 0,
            "duplicate_accepted_candidate_player_id_count": None,
            "duplicate_accepted_raw_player_id_count": None,
            "unsafe_accepted_mapping_count": None,
            "status_counts": {},
            "data": None,
            "error": "file_not_found",
        }

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {
            "exists": True,
            "path": str(path),
            "loaded": False,
            "row_count": 0,
            "top_row_count": 0,
            "accepted_mapping_rows": 0,
            "duplicate_accepted_candidate_player_id_count": None,
            "duplicate_accepted_raw_player_id_count": None,
            "unsafe_accepted_mapping_count": None,
            "status_counts": {},
            "data": None,
            "error": str(exc),
        }

    top = df[(df["candidate_rank"].isna()) | (df["candidate_rank"] == 1)].copy()
    accepted = top[top["match_status"] == "auto_approved_candidate"].copy()

    duplicate_candidate = int(accepted["candidate_player_id"].dropna().duplicated().sum())
    duplicate_raw = int(accepted["raw_player_id"].dropna().astype(str).duplicated().sum())

    unsafe_count = None
    if "safe_name_match_for_auto_approval" in accepted.columns:
        unsafe_count = int((accepted["safe_name_match_for_auto_approval"] != True).sum())

    return {
        "exists": True,
        "path": str(path),
        "loaded": True,
        "row_count": int(len(df)),
        "top_row_count": int(len(top)),
        "accepted_mapping_rows": int(len(accepted)),
        "duplicate_accepted_candidate_player_id_count": duplicate_candidate,
        "duplicate_accepted_raw_player_id_count": duplicate_raw,
        "unsafe_accepted_mapping_count": unsafe_count,
        "status_counts": top["match_status"].value_counts(dropna=False).to_dict(),
        "accepted_raw_player_ids": set(accepted["raw_player_id"].dropna().astype(str).tolist()),
        "data": df,
        "error": None,
    }


def check_required_columns(columns: Sequence[str], required: Sequence[str]) -> Dict[str, Any]:
    column_set = set(columns)
    missing = [column for column in required if column not in column_set]
    return {
        "required": list(required),
        "missing": missing,
        "passed": len(missing) == 0,
    }


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    mode_result = resolve_prediction_mode(
        season=args.target_season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.source_season,
        stabilization_gw=6,
        allow_experimental_mode=False,
    )

    prior = load_prior_csv(args.prior_csv)
    mapping = load_mapping_csv(args.mapping_csv)
    day66c = read_json_file(args.day66c_json)
    day66d = read_json_file(args.day66d_json)

    db_counts = {
        "teams": count_table_rows("teams", season=args.target_season),
        "players": count_table_rows("players", season=args.target_season),
        "gameweeks_target_gw": count_table_rows("gameweeks", season=args.target_season, gw=args.target_gw),
        "fixtures_target_gw": count_table_rows("fixtures", season=args.target_season, gw=args.target_gw),
        "player_gw_stats_target_gw": count_table_rows("player_gw_stats", season=args.target_season, gw=args.target_gw),
    }

    day66c_data = day66c["data"] if day66c.get("loaded") else None
    day66d_data = day66d["data"] if day66d.get("loaded") else None

    day66c_join = nested_get(day66c_data, ["join_summary"], {}) or {}
    day66c_mapping = nested_get(day66c_data, ["mapping_summary"], {}) or {}
    day66d_overall = nested_get(day66d_data, ["overall"], {}) or {}

    blockers: List[str] = []
    warnings: List[str] = []
    full_pipeline_blockers: List[str] = []

    if not mode_result["valid"]:
        blockers.append("prediction mode resolver returned invalid: %s" % mode_result["errors"])

    if mode_result["resolved_prediction_mode"] != "pre_gw1_prior":
        blockers.append(
            "Pre-GW1 readiness requires resolved_prediction_mode=pre_gw1_prior, got %s."
            % mode_result["resolved_prediction_mode"]
        )

    if args.target_gw != 1:
        blockers.append("Pre-GW1 readiness expects target_gw=1, got %s." % args.target_gw)

    if not prior["loaded"]:
        blockers.append("Prior CSV could not be loaded: %s" % prior["error"])
    else:
        required_prior_cols = [
            "raw_player_id",
            "raw_player_name",
            "prev_season_minutes",
            "prev_season_total_points",
            "prev_season_points_per90",
        ]
        prior_col_check = check_required_columns(prior["columns"], required_prior_cols)
        if not prior_col_check["passed"]:
            blockers.append("Prior CSV missing required columns: %s" % prior_col_check["missing"])
        if prior["duplicate_raw_player_id_count"] not in (0, None):
            blockers.append("Prior CSV has duplicate raw_player_id rows.")
    if not mapping["loaded"]:
        blockers.append("Mapping CSV could not be loaded: %s" % mapping["error"])
    else:
        if mapping["accepted_mapping_rows"] < args.min_accepted_mappings:
            blockers.append(
                "Accepted mapping count below minimum: %s < %s."
                % (mapping["accepted_mapping_rows"], args.min_accepted_mappings)
            )
        if mapping["duplicate_accepted_candidate_player_id_count"] not in (0, None):
            blockers.append("Accepted mappings contain duplicate candidate_player_id.")
        if mapping["duplicate_accepted_raw_player_id_count"] not in (0, None):
            blockers.append("Accepted mappings contain duplicate raw_player_id.")
        if mapping["unsafe_accepted_mapping_count"] not in (0, None):
            blockers.append("Accepted mappings contain unsafe name matches.")

    if prior["loaded"] and mapping["loaded"]:
        missing_prior_ids = sorted(mapping["accepted_raw_player_ids"] - set(prior["data"]["raw_player_id"].dropna().astype(str).tolist()))
        if missing_prior_ids:
            blockers.append("Accepted mapping raw_player_id values missing from prior CSV: %s" % missing_prior_ids[:20])

    if not day66c["loaded"]:
        blockers.append("Day66C report could not be loaded: %s" % day66c["error"])
    else:
        if nested_get(day66c_data, ["passed"], None) is not True:
            blockers.append("Day66C report did not pass.")
        if nested_get(day66c_data, ["ready_for_prior_feature_export"], None) is not True:
            blockers.append("Day66C report is not ready_for_prior_feature_export.")
        if nested_get(day66c_data, ["blockers"], []) not in ([], None):
            blockers.append("Day66C report has blockers: %s" % nested_get(day66c_data, ["blockers"], []))
        if day66c_join.get("row_count_preserved") is not True:
            blockers.append("Day66C prior join did not preserve row count.")
        if day66c_join.get("joined_feature_key_duplicate_count") != 0:
            blockers.append("Day66C prior join has duplicate joined feature keys.")
        coverage = day66c_join.get("prior_coverage_rate_players")
        if coverage is None or float(coverage) < args.min_player_prior_coverage_rate:
            blockers.append(
                "Day66C player prior coverage below threshold: %s < %s."
                % (coverage, args.min_player_prior_coverage_rate)
            )

    if not day66d["loaded"]:
        blockers.append("Day66D contract could not be loaded: %s" % day66d["error"])
    else:
        if nested_get(day66d_data, ["audit_only"], None) is not True:
            blockers.append("Day66D contract is not audit_only.")
        if nested_get(day66d_data, ["writes_database"], None) is not False:
            blockers.append("Day66D contract writes_database is not False.")
        if day66d_overall.get("ready_for_prior_feature_export") is not True:
            blockers.append("Day66D contract is not ready_for_prior_feature_export.")
        if day66d_overall.get("ready_for_pre_gw1_implementation") is not True:
            blockers.append("Day66D contract is not ready_for_pre_gw1_implementation.")

    if db_counts["teams"]["count"] is None or int(db_counts["teams"]["count"]) <= 0:
        blockers.append("No teams found in target-season DB context.")

    if db_counts["players"]["count"] is None or int(db_counts["players"]["count"]) <= 0:
        blockers.append("No target players found in DB.")

    if db_counts["gameweeks_target_gw"]["count"] is None or int(db_counts["gameweeks_target_gw"]["count"]) <= 0:
        blockers.append("No target GW row found in gameweeks table.")

    if db_counts["fixtures_target_gw"]["count"] is None or int(db_counts["fixtures_target_gw"]["count"]) <= 0:
        blockers.append("No target GW fixtures found.")

    if db_counts["player_gw_stats_target_gw"]["count"] and int(db_counts["player_gw_stats_target_gw"]["count"]) > 0:
        warnings.append(
            "player_gw_stats rows already exist for target GW. This is acceptable for historical dry-run, "
            "but actual Pre-GW1 production must not use current-season actuals."
        )

    if args.team_prior_csv:
        team_prior_path = Path(args.team_prior_csv)
        if not team_prior_path.exists():
            full_pipeline_blockers.append("team_prior_csv was supplied but does not exist.")
        else:
            warnings.append("team_prior_csv exists, but Day67B does not validate team-prior schema yet.")
    else:
        full_pipeline_blockers.append("team prior artifact is not supplied yet; expected in Day68.")
        full_pipeline_blockers.append("match/team prior readiness is not checked yet; expected in Day68-Day69.")

    player_prior_ready = len(blockers) == 0
    full_pre_gw1_pipeline_ready = player_prior_ready and len(full_pipeline_blockers) == 0

    report: Dict[str, Any] = {
        "created_at": utc_now(),
        "source_season": args.source_season,
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": mode_result["resolved_prediction_mode"],
        "audit_only": True,
        "writes_database": False,
        "passed": True,
        "ready_for_pre_gw1_player_prior_mode": player_prior_ready,
        "ready_for_full_pre_gw1_pipeline": full_pre_gw1_pipeline_ready,
        "ready_for_prediction_write": False,
        "reason_ready_for_prediction_write_false": (
            "Day67B is a readiness checker only. Prediction writes should wait for mode-resolved "
            "refresh orchestration, team priors, and explicit artifact manifests."
        ),
        "mode_resolution": mode_result,
        "inputs": {
            "prior_csv": prior["path"],
            "mapping_csv": mapping["path"],
            "day66c_json": day66c["path"],
            "day66d_json": day66d["path"],
            "team_prior_csv": args.team_prior_csv or None,
        },
        "prior_summary": {
            "exists": prior["exists"],
            "loaded": prior["loaded"],
            "row_count": prior["row_count"],
            "raw_player_id_unique_count": prior["raw_player_id_unique_count"],
            "duplicate_raw_player_id_count": prior["duplicate_raw_player_id_count"],
        },
        "mapping_summary": {
            "exists": mapping["exists"],
            "loaded": mapping["loaded"],
            "row_count": mapping["row_count"],
            "top_row_count": mapping["top_row_count"],
            "accepted_mapping_rows": mapping["accepted_mapping_rows"],
            "duplicate_accepted_candidate_player_id_count": mapping["duplicate_accepted_candidate_player_id_count"],
            "duplicate_accepted_raw_player_id_count": mapping["duplicate_accepted_raw_player_id_count"],
            "unsafe_accepted_mapping_count": mapping["unsafe_accepted_mapping_count"],
            "status_counts": mapping["status_counts"],
        },
        "day66c_summary": {
            "exists": day66c["exists"],
            "loaded": day66c["loaded"],
            "passed": nested_get(day66c_data, ["passed"], None),
            "ready_for_prior_feature_export": nested_get(day66c_data, ["ready_for_prior_feature_export"], None),
            "blockers": nested_get(day66c_data, ["blockers"], None),
            "warnings": nested_get(day66c_data, ["warnings"], None),
            "join_summary": day66c_join,
            "mapping_summary": day66c_mapping,
        },
        "day66d_summary": {
            "exists": day66d["exists"],
            "loaded": day66d["loaded"],
            "contract_version": nested_get(day66d_data, ["contract_version"], None),
            "ready_for_prior_feature_export": day66d_overall.get("ready_for_prior_feature_export"),
            "ready_for_pre_gw1_implementation": day66d_overall.get("ready_for_pre_gw1_implementation"),
            "ready_for_full_multi_season_training": day66d_overall.get("ready_for_full_multi_season_training"),
        },
        "db_counts": db_counts,
        "blockers": blockers,
        "warnings": warnings,
        "full_pipeline_blockers": full_pipeline_blockers,
        "notes": [
            "This report is read-only and does not write predictions.",
            "Day67B checks player prior readiness for Pre-GW1 scaffolding.",
            "Full Pre-GW1 pipeline readiness is expected to remain false until team priors and refresh manifests are added.",
            "Current-season actuals are not required for pre_gw1_prior mode.",
        ],
    }

    return report


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    lines: List[str] = []
    lines.append("# Day67B — Pre-GW1 Readiness Check")
    lines.append("")
    lines.append("Created at: `%s`" % report["created_at"])
    lines.append("")
    lines.append("Source season: `%s`" % report["source_season"])
    lines.append("Target season: `%s`" % report["target_season"])
    lines.append("Target GW: `%s`" % report["target_gw"])
    lines.append("Requested mode: `%s`" % report["requested_prediction_mode"])
    lines.append("Resolved mode: `%s`" % report["resolved_prediction_mode"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Audit only: `%s`" % report["audit_only"])
    lines.append("- Writes database: `%s`" % report["writes_database"])
    lines.append("- Ready for Pre-GW1 player prior mode: `%s`" % report["ready_for_pre_gw1_player_prior_mode"])
    lines.append("- Ready for full Pre-GW1 pipeline: `%s`" % report["ready_for_full_pre_gw1_pipeline"])
    lines.append("- Ready for prediction write: `%s`" % report["ready_for_prediction_write"])
    lines.append("")
    lines.append("## Mode Resolution")
    lines.append("")
    mode = report["mode_resolution"]
    lines.append("- Valid: `%s`" % mode["valid"])
    lines.append("- Resolved prediction mode: `%s`" % mode["resolved_prediction_mode"])
    lines.append("- Requires prior season: `%s`" % mode["metadata"].get("requires_prior_season"))
    lines.append("- Requires current-season actuals: `%s`" % mode["metadata"].get("requires_current_season_actuals"))
    lines.append("- Default prior weight: `%s`" % mode["metadata"].get("default_prior_weight"))
    lines.append("- Default current weight: `%s`" % mode["metadata"].get("default_current_weight"))
    lines.append("")
    lines.append("## Player Prior Inputs")
    lines.append("")
    lines.append("- Prior rows: `%s`" % report["prior_summary"]["row_count"])
    lines.append("- Prior unique raw_player_id count: `%s`" % report["prior_summary"]["raw_player_id_unique_count"])
    lines.append("- Accepted mapping rows: `%s`" % report["mapping_summary"]["accepted_mapping_rows"])
    lines.append("- Duplicate accepted candidate_player_id count: `%s`" % report["mapping_summary"]["duplicate_accepted_candidate_player_id_count"])
    lines.append("- Unsafe accepted mapping count: `%s`" % report["mapping_summary"]["unsafe_accepted_mapping_count"])
    lines.append("")
    lines.append("## Day66C Prior Join Evidence")
    lines.append("")
    join = report["day66c_summary"]["join_summary"]
    lines.append("- Day66C passed: `%s`" % report["day66c_summary"]["passed"])
    lines.append("- Day66C ready_for_prior_feature_export: `%s`" % report["day66c_summary"]["ready_for_prior_feature_export"])
    lines.append("- Feature unique players: `%s`" % join.get("feature_unique_players"))
    lines.append("- Feature players with prior: `%s`" % join.get("feature_players_with_prior"))
    lines.append("- Feature players without prior: `%s`" % join.get("feature_players_without_prior"))
    lines.append("- Prior coverage rate players: `%s`" % join.get("prior_coverage_rate_players"))
    lines.append("- Feature rows with prior: `%s`" % join.get("feature_rows_with_prior"))
    lines.append("- Row count preserved: `%s`" % join.get("row_count_preserved"))
    lines.append("- Joined key duplicate count: `%s`" % join.get("joined_feature_key_duplicate_count"))
    lines.append("")
    lines.append("## Target DB Readiness")
    lines.append("")
    for key, value in report["db_counts"].items():
        lines.append("- %s: count=`%s`, exists=`%s`, season_filter=`%s`, gw_filter=`%s`" % (
            key,
            value.get("count"),
            value.get("exists"),
            value.get("season_filter_applied"),
            value.get("gw_filter_applied"),
        ))
    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append("- %s" % blocker)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append("- %s" % warning)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Full Pipeline Blockers")
    lines.append("")
    if report["full_pipeline_blockers"]:
        for blocker in report["full_pipeline_blockers"]:
            lines.append("- %s" % blocker)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append("- %s" % note)
    lines.append("")

    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: Dict[str, Any], out_json: str, out_md: str) -> None:
    print("=== Day67B Pre-GW1 Readiness Check ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("target_gw:", report["target_gw"])
    print("requested_prediction_mode:", report["requested_prediction_mode"])
    print("resolved_prediction_mode:", report["resolved_prediction_mode"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("ready_for_pre_gw1_player_prior_mode:", report["ready_for_pre_gw1_player_prior_mode"])
    print("ready_for_full_pre_gw1_pipeline:", report["ready_for_full_pre_gw1_pipeline"])
    print("ready_for_prediction_write:", report["ready_for_prediction_write"])
    print("saved_json:", out_json)
    print("saved_md:", out_md)
    print()
    print("Mode:")
    print("- valid:", report["mode_resolution"]["valid"])
    print("- errors:", report["mode_resolution"]["errors"])
    print("- default_prior_weight:", report["mode_resolution"]["metadata"].get("default_prior_weight"))
    print("- default_current_weight:", report["mode_resolution"]["metadata"].get("default_current_weight"))
    print()
    print("Prior summary:", report["prior_summary"])
    print("Mapping summary:", report["mapping_summary"])
    print()
    print("Day66C join summary:", report["day66c_summary"]["join_summary"])
    print()
    print("DB counts:")
    for key, value in report["db_counts"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Blockers:", report["blockers"] or "none")
    print("Warnings:", report["warnings"] or "none")
    print("Full pipeline blockers:", report["full_pipeline_blockers"] or "none")


def main() -> None:
    args = parse_args()
    report = build_report(args)

    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report, args.out_json, args.out_md)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
