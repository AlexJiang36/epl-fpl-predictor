from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal
from ml.validation.resolve_prediction_mode import resolve_prediction_mode


TEAM_PRIOR_STAT_COLUMNS = [
    "prev_season_clean_sheets",
    "prev_season_draws",
    "prev_season_goal_difference",
    "prev_season_goals_against",
    "prev_season_goals_for",
    "prev_season_losses",
    "prev_season_matches",
    "prev_season_total_points",
    "prev_season_wins",
    "prev_season_points_per_match",
    "prev_season_goals_for_per_match",
    "prev_season_goals_against_per_match",
    "prev_season_scheduled_fixtures",
    "prev_season_home_matches",
    "prev_season_away_matches",
    "prev_season_clean_sheet_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only Pre-GW1 match feature preview using target GW fixtures and "
            "safe historical team-prior mappings. No predictions or DB writes are performed."
        )
    )
    parser.add_argument("--source-season", required=True, help="Prior/source season, for example 2024_25.")
    parser.add_argument("--target-season", required=True, help="Target season, for example 2025_26.")
    parser.add_argument("--target-gw", type=int, default=1, help="Target GW. Day69A expects GW1.")
    parser.add_argument("--prediction-mode", default="auto", choices=["auto", "pre_gw1_prior", "early_season_blend", "normal_weekly"])
    parser.add_argument("--team-prior-csv", required=True, help="Day68A team prior CSV.")
    parser.add_argument("--team-mapping-csv", required=True, help="Day68B team mapping candidate CSV.")
    parser.add_argument("--out-csv", required=True, help="Output Pre-GW1 match feature preview CSV.")
    parser.add_argument("--out-json", required=True, help="Output JSON audit report.")
    parser.add_argument("--out-md", required=True, help="Output Markdown audit report.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def normalize_raw_id(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    if text_value.endswith(".0"):
        text_value = text_value[:-2]
    return text_value


def nullable_int(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def nullable_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_target_teams() -> pd.DataFrame:
    if not table_exists("teams"):
        raise RuntimeError("teams table does not exist.")

    columns = table_columns("teams")
    df = read_sql_dataframe("SELECT * FROM teams", {})

    if df.empty:
        raise RuntimeError("teams table is empty.")

    team_id_col = first_existing_column(columns, ["id", "team_id"])
    fpl_team_id_col = first_existing_column(columns, ["fpl_team_id"])
    team_name_col = first_existing_column(columns, ["name", "team_name"])
    team_short_col = first_existing_column(columns, ["short_name", "code"])

    if team_id_col is None:
        raise RuntimeError("teams table must contain id or team_id.")
    if team_short_col is None:
        raise RuntimeError("teams table must contain short_name or code.")

    out = pd.DataFrame()
    out["team_id"] = df[team_id_col].apply(nullable_int)
    out["fpl_team_id"] = df[fpl_team_id_col].apply(nullable_int) if fpl_team_id_col else None
    out["team_name"] = df[team_name_col].astype(str) if team_name_col else ""
    out["team_short_name"] = df[team_short_col].astype(str)
    return out


def load_target_fixtures(target_season: str, target_gw: int) -> pd.DataFrame:
    if not table_exists("fixtures"):
        raise RuntimeError("fixtures table does not exist.")

    columns = table_columns("fixtures")
    required = ["id", "home_team_id", "away_team_id"]
    missing = [col for col in required if col not in columns]
    if missing:
        raise RuntimeError("fixtures table missing required columns: %s" % missing)

    season_col = first_existing_column(columns, ["season"])
    gw_col = first_existing_column(columns, ["gw", "gameweek", "event"])

    if season_col is None:
        raise RuntimeError("fixtures table must contain season for Day69A.")
    if gw_col is None:
        raise RuntimeError("fixtures table must contain gw/gameweek/event for Day69A.")

    select_cols = [
        "id",
        "fpl_fixture_id" if "fpl_fixture_id" in columns else "NULL AS fpl_fixture_id",
        season_col + " AS season",
        gw_col + " AS gw",
        "home_team_id",
        "away_team_id",
        "kickoff_time" if "kickoff_time" in columns else "NULL AS kickoff_time",
        "finished" if "finished" in columns else "NULL AS finished",
        "home_score" if "home_score" in columns else "NULL AS home_score",
        "away_score" if "away_score" in columns else "NULL AS away_score",
    ]

    sql = """
        SELECT %s
        FROM fixtures
        WHERE %s = :season
          AND %s = :gw
        ORDER BY kickoff_time NULLS LAST, id
    """ % (", ".join(select_cols), season_col, gw_col)

    df = read_sql_dataframe(sql, {"season": target_season, "gw": target_gw})
    return df


def load_team_priors(path_value: str, source_season: str, target_season: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("team prior CSV does not exist: %s" % path)
    df = pd.read_csv(path)
    required = [
        "source_season",
        "target_season",
        "raw_team_id",
        "raw_team_name",
        "raw_team_short_name",
        "prior_identity_scope",
        "prior_source",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError("team prior CSV missing required columns: %s" % missing)

    df = df[
        (df["source_season"].astype(str) == str(source_season))
        & (df["target_season"].astype(str) == str(target_season))
    ].copy()

    if df.empty:
        raise RuntimeError("No team prior rows found for source=%s target=%s." % (source_season, target_season))

    df["raw_team_id"] = df["raw_team_id"].apply(normalize_raw_id)
    return df


def load_team_mapping_candidates(path_value: str, source_season: str, target_season: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("team mapping CSV does not exist: %s" % path)
    df = pd.read_csv(path)
    required = [
        "source_season",
        "target_season",
        "raw_team_id",
        "raw_team_name",
        "raw_team_short_name",
        "candidate_rank",
        "candidate_team_id",
        "candidate_team_name",
        "candidate_team_short_name",
        "match_status",
        "is_auto_approved",
        "needs_manual_review",
        "safe_team_match_for_auto_approval",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError("team mapping CSV missing required columns: %s" % missing)

    df = df[
        (df["source_season"].astype(str) == str(source_season))
        & (df["target_season"].astype(str) == str(target_season))
    ].copy()

    if df.empty:
        raise RuntimeError("No team mapping rows found for source=%s target=%s." % (source_season, target_season))

    df["raw_team_id"] = df["raw_team_id"].apply(normalize_raw_id)
    df["candidate_team_id"] = df["candidate_team_id"].apply(nullable_int)
    return df


def build_target_team_prior_lookup(team_priors: pd.DataFrame, team_mapping: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    top_rows = team_mapping[(team_mapping["candidate_rank"].isna()) | (team_mapping["candidate_rank"] == 1)].copy()
    accepted = top_rows[top_rows["match_status"] == "auto_approved_team_candidate"].copy()

    diagnostics: Dict[str, Any] = {
        "top_mapping_rows": int(len(top_rows)),
        "accepted_mapping_rows": int(len(accepted)),
        "manual_review_rows": int(top_rows["needs_manual_review"].fillna(True).sum()),
        "unmatched_rows": int((top_rows["match_status"] == "historical_only_unmatched").sum()),
        "duplicate_accepted_candidate_team_id_count": int(accepted["candidate_team_id"].dropna().duplicated().sum()),
        "duplicate_accepted_raw_team_id_count": int(accepted["raw_team_id"].dropna().astype(str).duplicated().sum()),
        "unsafe_accepted_team_match_count": int((accepted["safe_team_match_for_auto_approval"] != True).sum()) if len(accepted) else 0,
        "historical_only_short_names": sorted(
            top_rows.loc[top_rows["match_status"] == "historical_only_unmatched", "raw_team_short_name"]
            .dropna()
            .astype(str)
            .tolist()
        ),
    }

    if accepted.empty:
        return pd.DataFrame(), diagnostics

    prior_cols = [
        "raw_team_id",
        "raw_team_name",
        "raw_team_short_name",
        "prior_identity_scope",
        "prior_source",
        "has_prev_season_data",
        "is_prev_season_active",
    ] + [col for col in TEAM_PRIOR_STAT_COLUMNS if col in team_priors.columns]

    priors_subset = team_priors[prior_cols].copy()
    accepted = accepted[
        [
            "raw_team_id",
            "candidate_team_id",
            "candidate_team_name",
            "candidate_team_short_name",
            "match_status",
            "candidate_score",
            "match_reason",
        ]
    ].copy()

    lookup = accepted.merge(priors_subset, on="raw_team_id", how="left", validate="one_to_one")
    diagnostics["accepted_mappings_missing_prior_count"] = int(lookup["prior_source"].isna().sum())
    return lookup, diagnostics


def side_feature_prefix_row(
    side: str,
    team_id: Optional[int],
    target_team_lookup: Dict[int, Dict[str, Any]],
    prior_lookup: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    prefix = side + "_"
    row: Dict[str, Any] = {}

    target = target_team_lookup.get(team_id or -1)
    prior = prior_lookup.get(team_id or -1)

    row[prefix + "team_id"] = team_id

    if target:
        row[prefix + "team_name"] = target.get("team_name")
        row[prefix + "team_short_name"] = target.get("team_short_name")
        row[prefix + "fpl_team_id"] = target.get("fpl_team_id")
        row[prefix + "team_row_found"] = True
    else:
        row[prefix + "team_name"] = None
        row[prefix + "team_short_name"] = None
        row[prefix + "fpl_team_id"] = None
        row[prefix + "team_row_found"] = False

    if prior:
        row[prefix + "has_prev_season_team_prior"] = True
        row[prefix + "team_prior_status"] = "reliable_prior"
        row[prefix + "team_prior_source"] = prior.get("prior_source")
        row[prefix + "prior_identity_scope"] = prior.get("prior_identity_scope")
        row[prefix + "raw_team_id"] = prior.get("raw_team_id")
        row[prefix + "raw_team_name"] = prior.get("raw_team_name")
        row[prefix + "raw_team_short_name"] = prior.get("raw_team_short_name")
        row[prefix + "team_mapping_status"] = prior.get("match_status")
        row[prefix + "team_mapping_score"] = prior.get("candidate_score")
        row[prefix + "team_mapping_reason"] = prior.get("match_reason")
        for stat_col in TEAM_PRIOR_STAT_COLUMNS:
            row[prefix + stat_col] = prior.get(stat_col)
    else:
        row[prefix + "has_prev_season_team_prior"] = False
        row[prefix + "team_prior_status"] = "target_only_or_unmapped_prior_missing"
        row[prefix + "team_prior_source"] = "fallback_required"
        row[prefix + "prior_identity_scope"] = None
        row[prefix + "raw_team_id"] = None
        row[prefix + "raw_team_name"] = None
        row[prefix + "raw_team_short_name"] = None
        row[prefix + "team_mapping_status"] = None
        row[prefix + "team_mapping_score"] = None
        row[prefix + "team_mapping_reason"] = None
        for stat_col in TEAM_PRIOR_STAT_COLUMNS:
            row[prefix + stat_col] = None

    return row


def build_match_features(
    fixtures: pd.DataFrame,
    target_teams: pd.DataFrame,
    team_prior_lookup: pd.DataFrame,
    source_season: str,
    target_season: str,
    target_gw: int,
    resolved_prediction_mode: str,
) -> pd.DataFrame:
    target_team_lookup: Dict[int, Dict[str, Any]] = {}
    for _, row in target_teams.iterrows():
        team_id = nullable_int(row.get("team_id"))
        if team_id is not None:
            target_team_lookup[team_id] = row.to_dict()

    prior_lookup: Dict[int, Dict[str, Any]] = {}
    if not team_prior_lookup.empty:
        for _, row in team_prior_lookup.iterrows():
            team_id = nullable_int(row.get("candidate_team_id"))
            if team_id is not None:
                prior_lookup[team_id] = row.to_dict()

    rows: List[Dict[str, Any]] = []

    for _, fixture in fixtures.iterrows():
        fixture_id = nullable_int(fixture.get("id"))
        home_team_id = nullable_int(fixture.get("home_team_id"))
        away_team_id = nullable_int(fixture.get("away_team_id"))

        out: Dict[str, Any] = {
            "source_season": source_season,
            "target_season": target_season,
            "target_gw": target_gw,
            "prediction_mode": resolved_prediction_mode,
            "fixture_id": fixture_id,
            "fpl_fixture_id": nullable_int(fixture.get("fpl_fixture_id")),
            "kickoff_time": fixture.get("kickoff_time"),
            "fixture_finished": fixture.get("finished"),
            "home_score": fixture.get("home_score"),
            "away_score": fixture.get("away_score"),
        }

        out.update(side_feature_prefix_row("home", home_team_id, target_team_lookup, prior_lookup))
        out.update(side_feature_prefix_row("away", away_team_id, target_team_lookup, prior_lookup))

        out["both_team_rows_found"] = bool(out["home_team_row_found"] and out["away_team_row_found"])
        out["both_teams_have_reliable_prior"] = bool(
            out["home_has_prev_season_team_prior"] and out["away_has_prev_season_team_prior"]
        )
        out["any_team_missing_prior"] = not out["both_teams_have_reliable_prior"]

        home_ppm = nullable_float(out.get("home_prev_season_points_per_match"))
        away_ppm = nullable_float(out.get("away_prev_season_points_per_match"))
        home_gfpm = nullable_float(out.get("home_prev_season_goals_for_per_match"))
        away_gfpm = nullable_float(out.get("away_prev_season_goals_for_per_match"))
        home_gapm = nullable_float(out.get("home_prev_season_goals_against_per_match"))
        away_gapm = nullable_float(out.get("away_prev_season_goals_against_per_match"))

        out["prior_points_per_match_diff_home_minus_away"] = (
            None if home_ppm is None or away_ppm is None else round(home_ppm - away_ppm, 4)
        )
        out["prior_goals_for_per_match_diff_home_minus_away"] = (
            None if home_gfpm is None or away_gfpm is None else round(home_gfpm - away_gfpm, 4)
        )
        out["prior_goals_against_per_match_diff_home_minus_away"] = (
            None if home_gapm is None or away_gapm is None else round(home_gapm - away_gapm, 4)
        )

        rows.append(out)

    return pd.DataFrame(rows)


def build_report(
    features: pd.DataFrame,
    fixtures: pd.DataFrame,
    target_teams: pd.DataFrame,
    team_prior_lookup: pd.DataFrame,
    mapping_diagnostics: Dict[str, Any],
    mode_result: Dict[str, Any],
    args: argparse.Namespace,
    blockers: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    target_team_ids = set(target_teams["team_id"].dropna().astype(int).tolist())
    prior_target_team_ids = set(team_prior_lookup["candidate_team_id"].dropna().astype(int).tolist()) if not team_prior_lookup.empty else set()
    target_team_ids_without_prior = sorted(target_team_ids - prior_target_team_ids)

    target_without_prior = target_teams[target_teams["team_id"].isin(target_team_ids_without_prior)].copy()
    target_only_short_names = sorted(target_without_prior["team_short_name"].dropna().astype(str).tolist())

    gw_team_ids = set()
    if not fixtures.empty:
        gw_team_ids = set(fixtures["home_team_id"].dropna().astype(int).tolist()) | set(fixtures["away_team_id"].dropna().astype(int).tolist())
    gw_team_ids_without_prior = sorted(gw_team_ids - prior_target_team_ids)
    gw_without_prior = target_teams[target_teams["team_id"].isin(gw_team_ids_without_prior)].copy()

    fixture_count = int(len(fixtures))
    feature_rows = int(len(features))

    if feature_rows != fixture_count:
        blockers.append("Feature row count does not equal fixture row count.")
    if feature_rows > 0 and int(features["fixture_id"].dropna().duplicated().sum()) > 0:
        blockers.append("Duplicate fixture_id values found in output features.")

    if mapping_diagnostics.get("duplicate_accepted_candidate_team_id_count") != 0:
        blockers.append("Accepted team mappings contain duplicate candidate_team_id.")
    if mapping_diagnostics.get("duplicate_accepted_raw_team_id_count") != 0:
        blockers.append("Accepted team mappings contain duplicate raw_team_id.")
    if mapping_diagnostics.get("unsafe_accepted_team_match_count") != 0:
        blockers.append("Accepted team mappings contain unsafe team matches.")
    if mapping_diagnostics.get("accepted_mappings_missing_prior_count") not in (0, None):
        blockers.append("Accepted team mappings are missing source prior rows.")

    if target_team_ids_without_prior:
        warnings.append(
            "Target teams without accepted previous-season EPL priors require fallback handling: %s"
            % target_only_short_names
        )

    fixture_summary = {
        "fixture_count": fixture_count,
        "feature_rows": feature_rows,
        "target_gw_team_count": int(len(gw_team_ids)),
        "fixture_duplicate_count": int(features["fixture_id"].dropna().duplicated().sum()) if feature_rows else 0,
        "fixtures_with_both_team_rows_found": int(features["both_team_rows_found"].sum()) if feature_rows else 0,
        "fixtures_with_both_reliable_priors": int(features["both_teams_have_reliable_prior"].sum()) if feature_rows else 0,
        "fixtures_with_any_missing_prior": int(features["any_team_missing_prior"].sum()) if feature_rows else 0,
        "target_gw_team_ids_without_prior": gw_team_ids_without_prior,
        "target_gw_team_short_names_without_prior": sorted(gw_without_prior["team_short_name"].dropna().astype(str).tolist()),
    }

    report = {
        "created_at": utc_now(),
        "source_season": args.source_season,
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": mode_result["resolved_prediction_mode"],
        "audit_only": True,
        "writes_database": False,
        "passed": len(blockers) == 0,
        "ready_for_pre_gw1_match_feature_scaffolding": len(blockers) == 0,
        "ready_for_full_pre_gw1_match_prediction": False,
        "ready_for_prediction_write": False,
        "reason_prediction_not_ready": (
            "Day69A builds match feature scaffolding only. Full prediction requires promoted-team fallback "
            "policy, model scoring, calibration/guardrail policy, and an explicit prediction manifest."
        ),
        "inputs": {
            "team_prior_csv": args.team_prior_csv,
            "team_mapping_csv": args.team_mapping_csv,
            "out_csv": args.out_csv,
            "out_json": args.out_json,
            "out_md": args.out_md,
        },
        "mode_resolution": mode_result,
        "row_counts": {
            "target_team_rows": int(len(target_teams)),
            "team_prior_lookup_rows": int(len(team_prior_lookup)),
            "fixture_rows": fixture_count,
            "feature_rows": feature_rows,
        },
        "mapping_summary": {
            "accepted_mapping_rows": mapping_diagnostics.get("accepted_mapping_rows"),
            "manual_review_rows": mapping_diagnostics.get("manual_review_rows"),
            "unmatched_rows": mapping_diagnostics.get("unmatched_rows"),
            "duplicate_accepted_candidate_team_id_count": mapping_diagnostics.get("duplicate_accepted_candidate_team_id_count"),
            "duplicate_accepted_raw_team_id_count": mapping_diagnostics.get("duplicate_accepted_raw_team_id_count"),
            "unsafe_accepted_team_match_count": mapping_diagnostics.get("unsafe_accepted_team_match_count"),
            "accepted_mappings_missing_prior_count": mapping_diagnostics.get("accepted_mappings_missing_prior_count"),
            "historical_only_short_names": mapping_diagnostics.get("historical_only_short_names"),
            "target_team_short_names_without_prior": target_only_short_names,
        },
        "fixture_summary": fixture_summary,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "This is a feature preview and does not generate match probabilities.",
            "Accepted team-prior mappings are used only when match_status=auto_approved_team_candidate.",
            "Target-season teams without accepted priors should receive an explicit fallback in a later step.",
            "Historical-only teams do not appear in target-season fixtures and are excluded from target feature rows.",
        ],
    }
    return report


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    lines: List[str] = []
    lines.append("# Day69A — Pre-GW1 Match Feature Scaffolding")
    lines.append("")
    lines.append("Created at: `%s`" % report["created_at"])
    lines.append("")
    lines.append("Source season: `%s`" % report["source_season"])
    lines.append("Target season: `%s`" % report["target_season"])
    lines.append("Target GW: `%s`" % report["target_gw"])
    lines.append("Resolved prediction mode: `%s`" % report["resolved_prediction_mode"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Passed: `%s`" % report["passed"])
    lines.append("- Audit only: `%s`" % report["audit_only"])
    lines.append("- Writes database: `%s`" % report["writes_database"])
    lines.append("- Ready for Pre-GW1 match feature scaffolding: `%s`" % report["ready_for_pre_gw1_match_feature_scaffolding"])
    lines.append("- Ready for full Pre-GW1 match prediction: `%s`" % report["ready_for_full_pre_gw1_match_prediction"])
    lines.append("- Ready for prediction write: `%s`" % report["ready_for_prediction_write"])
    lines.append("")
    lines.append("## Row Counts")
    lines.append("")
    for key, value in report["row_counts"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Mapping Summary")
    lines.append("")
    for key, value in report["mapping_summary"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Fixture Summary")
    lines.append("")
    for key, value in report["fixture_summary"].items():
        lines.append("- %s: `%s`" % (key, value))
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
    lines.append("## Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append("- %s" % note)
    lines.append("")

    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: Dict[str, Any]) -> None:
    print("=== Day69A Pre-GW1 Match Feature Scaffolding ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("target_gw:", report["target_gw"])
    print("resolved_prediction_mode:", report["resolved_prediction_mode"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("ready_for_pre_gw1_match_feature_scaffolding:", report["ready_for_pre_gw1_match_feature_scaffolding"])
    print("ready_for_full_pre_gw1_match_prediction:", report["ready_for_full_pre_gw1_match_prediction"])
    print("ready_for_prediction_write:", report["ready_for_prediction_write"])
    print("saved_csv:", report["inputs"]["out_csv"])
    print("saved_json:", report["inputs"]["out_json"])
    print("saved_md:", report["inputs"]["out_md"])
    print()
    print("Row counts:")
    for key, value in report["row_counts"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Mapping summary:")
    for key, value in report["mapping_summary"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Fixture summary:")
    for key, value in report["fixture_summary"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Blockers:", report["blockers"] or "none")
    print("Warnings:", report["warnings"] or "none")


def main() -> None:
    args = parse_args()

    blockers: List[str] = []
    warnings: List[str] = []

    mode_result = resolve_prediction_mode(
        season=args.target_season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.source_season,
        stabilization_gw=6,
        allow_experimental_mode=False,
    )

    if not mode_result["valid"]:
        blockers.append("Prediction mode resolver returned invalid: %s" % mode_result["errors"])
    if mode_result["resolved_prediction_mode"] != "pre_gw1_prior":
        blockers.append(
            "Day69A expects resolved_prediction_mode=pre_gw1_prior, got %s."
            % mode_result["resolved_prediction_mode"]
        )
    if args.target_gw != 1:
        blockers.append("Day69A expects target_gw=1, got %s." % args.target_gw)

    target_teams = load_target_teams()
    fixtures = load_target_fixtures(args.target_season, args.target_gw)
    team_priors = load_team_priors(args.team_prior_csv, args.source_season, args.target_season)
    team_mapping = load_team_mapping_candidates(args.team_mapping_csv, args.source_season, args.target_season)

    team_prior_lookup, mapping_diagnostics = build_target_team_prior_lookup(team_priors, team_mapping)

    features = build_match_features(
        fixtures=fixtures,
        target_teams=target_teams,
        team_prior_lookup=team_prior_lookup,
        source_season=args.source_season,
        target_season=args.target_season,
        target_gw=args.target_gw,
        resolved_prediction_mode=mode_result["resolved_prediction_mode"],
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv, index=False)

    report = build_report(
        features=features,
        fixtures=fixtures,
        target_teams=target_teams,
        team_prior_lookup=team_prior_lookup,
        mapping_diagnostics=mapping_diagnostics,
        mode_result=mode_result,
        args=args,
        blockers=blockers,
        warnings=warnings,
    )

    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
