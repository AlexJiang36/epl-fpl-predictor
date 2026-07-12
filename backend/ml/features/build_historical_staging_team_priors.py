from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


IDENTITY_COLUMN_CANDIDATES = {
    "raw_team_id": ["raw_team_id", "team_id", "id", "fpl_team_id"],
    "raw_team_name": ["raw_team_name", "team_name", "name"],
    "raw_team_short_name": ["raw_team_short_name", "team_short_name", "short_name", "code"],
    "season": ["season", "source_season"],
}

STAT_COLUMN_ALIASES = {
    "prev_season_matches": ["matches", "matches_played", "match_count", "fixtures_played", "games", "played", "total_matches", "matches_counted"],
    "prev_season_wins": ["wins", "won"],
    "prev_season_draws": ["draws", "drawn"],
    "prev_season_losses": ["losses", "lost"],
    "prev_season_goals_for": ["goals_for", "gf", "team_goals_for"],
    "prev_season_goals_against": ["goals_against", "ga", "team_goals_against"],
    "prev_season_goal_difference": ["goal_difference", "gd"],
    "prev_season_clean_sheets": ["clean_sheets", "cs"],
    "prev_season_total_points": ["total_points", "points", "team_points", "league_points"],
    "prev_season_home_points": ["home_points"],
    "prev_season_away_points": ["away_points"],
    "prev_season_home_goals_for": ["home_goals_for", "home_gf"],
    "prev_season_away_goals_for": ["away_goals_for", "away_gf"],
    "prev_season_home_goals_against": ["home_goals_against", "home_ga"],
    "prev_season_away_goals_against": ["away_goals_against", "away_ga"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build previous-season team priors from historical staging team season summaries. Read-only."
    )
    parser.add_argument("--source-season", required=True)
    parser.add_argument("--target-season", required=True)
    parser.add_argument("--team-summary-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_raw_id(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    if text_value.endswith(".0"):
        text_value = text_value[:-2]
    return text_value


def normalize_text(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text_value = str(value).strip()
    return text_value or None


def nullable_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def find_identity_columns(columns: Sequence[str]) -> Dict[str, Optional[str]]:
    return {key: first_existing_column(columns, candidates) for key, candidates in IDENTITY_COLUMN_CANDIDATES.items()}


def find_stat_source_column(columns: Sequence[str], output_column: str) -> Optional[str]:
    return first_existing_column(columns, STAT_COLUMN_ALIASES.get(output_column, []))


def copy_known_stats(row: pd.Series, columns: Sequence[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for output_column in sorted(STAT_COLUMN_ALIASES.keys()):
        source_column = find_stat_source_column(columns, output_column)
        if source_column is not None:
            values[output_column] = row.get(source_column)

    matches = nullable_float(values.get("prev_season_matches"))
    total_points = nullable_float(values.get("prev_season_total_points"))
    goals_for = nullable_float(values.get("prev_season_goals_for"))
    goals_against = nullable_float(values.get("prev_season_goals_against"))

    if values.get("prev_season_goal_difference") is None and goals_for is not None and goals_against is not None:
        values["prev_season_goal_difference"] = goals_for - goals_against

    values["prev_season_points_per_match"] = safe_divide(total_points, matches)
    values["prev_season_goals_for_per_match"] = safe_divide(goals_for, matches)
    values["prev_season_goals_against_per_match"] = safe_divide(goals_against, matches)
    return values


def copy_extra_stats(row: pd.Series, columns: Sequence[str], identity_columns: Sequence[Optional[str]]) -> Dict[str, Any]:
    identity_set = set([col for col in identity_columns if col])
    known_source_columns = set()
    for candidates in STAT_COLUMN_ALIASES.values():
        for candidate in candidates:
            if candidate in columns:
                known_source_columns.add(candidate)

    metadata_columns_to_skip = {
        "canonical_team_id",
        "canonical_team_name",
        "mapping_status",
        "mapping_confidence",
        "notes",
    }

    extra: Dict[str, Any] = {}
    for column in columns:
        if column in identity_set or column in known_source_columns or column.startswith("Unnamed:"):
            continue
        if column in metadata_columns_to_skip:
            continue
        extra["prev_season_%s" % column] = row.get(column)
    return extra


def load_team_summary(path_value: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("team summary CSV does not exist: %s" % path)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("team summary CSV is empty: %s" % path)
    return df


def build_team_priors(source_season: str, target_season: str, team_summary: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    columns = [str(col) for col in team_summary.columns]
    identity = find_identity_columns(columns)
    errors: List[str] = []
    warnings: List[str] = []

    if identity["raw_team_id"] is None:
        errors.append("Team summary CSV must contain one of: %s" % IDENTITY_COLUMN_CANDIDATES["raw_team_id"])
    if identity["raw_team_name"] is None:
        warnings.append("Team summary CSV does not contain a recognized raw team name column.")
    if identity["raw_team_short_name"] is None:
        warnings.append("Team summary CSV does not contain a recognized raw team short-name column.")
    if errors:
        return pd.DataFrame(), {"passed": False, "errors": errors, "warnings": warnings, "identity_columns": identity}

    rows: List[Dict[str, Any]] = []
    for _, row in team_summary.iterrows():
        prior_row: Dict[str, Any] = {
            "source_season": source_season,
            "target_season": target_season,
            "raw_team_id": normalize_raw_id(row.get(identity["raw_team_id"])),
            "raw_team_name": normalize_text(row.get(identity["raw_team_name"])) if identity["raw_team_name"] else None,
            "raw_team_short_name": normalize_text(row.get(identity["raw_team_short_name"])) if identity["raw_team_short_name"] else None,
            "canonical_team_id": None,
            "canonical_team_name": None,
            "mapping_status": "unmapped",
            "mapping_confidence": None,
            "prior_identity_scope": "historical_raw_team_id",
            "prior_source": "historical_staging",
            "has_prev_season_data": True,
            "is_prev_season_active": True,
            "notes": None,
        }
        prior_row.update(copy_known_stats(row, columns))
        for key, value in copy_extra_stats(row, columns, list(identity.values())).items():
            if key not in prior_row:
                prior_row[key] = value

        matches = nullable_float(prior_row.get("prev_season_matches"))
        if matches is not None:
            prior_row["is_prev_season_active"] = matches > 0
        rows.append(prior_row)

    priors = pd.DataFrame(rows)
    duplicate_raw_team_ids = int(priors["raw_team_id"].dropna().duplicated().sum())
    missing_raw_team_ids = int(priors["raw_team_id"].isna().sum())

    if duplicate_raw_team_ids > 0:
        errors.append("Duplicate raw_team_id rows found: %s" % duplicate_raw_team_ids)
    if missing_raw_team_ids > 0:
        errors.append("Missing raw_team_id rows found: %s" % missing_raw_team_ids)
    if len(priors) != 20:
        warnings.append("Expected 20 Premier League teams, found %s rows." % len(priors))

    total_points = None
    if "prev_season_total_points" in priors.columns:
        total_points = nullable_float(priors["prev_season_total_points"].sum())

    total_matches = None
    if "prev_season_matches" in priors.columns:
        total_matches = nullable_float(priors["prev_season_matches"].sum())

    top_team = None
    bottom_team = None
    if "prev_season_total_points" in priors.columns and priors["prev_season_total_points"].notna().any():
        sorted_by_points = priors.sort_values("prev_season_total_points", ascending=False)
        top = sorted_by_points.iloc[0]
        bottom = sorted_by_points.iloc[-1]
        top_team = {
            "raw_team_id": top.get("raw_team_id"),
            "raw_team_name": top.get("raw_team_name"),
            "raw_team_short_name": top.get("raw_team_short_name"),
            "prev_season_total_points": nullable_float(top.get("prev_season_total_points")),
        }
        bottom_team = {
            "raw_team_id": bottom.get("raw_team_id"),
            "raw_team_name": bottom.get("raw_team_name"),
            "raw_team_short_name": bottom.get("raw_team_short_name"),
            "prev_season_total_points": nullable_float(bottom.get("prev_season_total_points")),
        }

    audit = {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "identity_columns": identity,
        "row_count": int(len(priors)),
        "active_team_count": int(priors["is_prev_season_active"].fillna(False).sum()),
        "unmapped_team_count": int((priors["mapping_status"] == "unmapped").sum()),
        "duplicate_raw_team_id_count": duplicate_raw_team_ids,
        "missing_raw_team_id_count": missing_raw_team_ids,
        "total_prev_season_points": total_points,
        "total_prev_season_matches": total_matches,
        "top_team_by_points": top_team,
        "bottom_team_by_points": bottom_team,
        "output_columns": [str(col) for col in priors.columns],
    }
    return priors, audit


def build_report(source_season: str, target_season: str, team_summary_csv: str, out_csv: str, priors: pd.DataFrame, audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": utc_now(),
        "source_season": source_season,
        "target_season": target_season,
        "team_summary_csv": team_summary_csv,
        "out_csv": out_csv,
        "passed": bool(audit["passed"]),
        "audit_only": True,
        "writes_database": False,
        "ready_for_team_mapping_audit": bool(audit["passed"]),
        "ready_for_canonical_team_join": False,
        "reason_canonical_team_join_not_ready": "Day68A keeps historical raw team identity only. Canonical team mapping should be audited separately in Day68B.",
        "row_counts": {
            "team_summary_rows": int(len(priors)),
            "team_prior_rows": int(len(priors)),
            "active_team_count": audit.get("active_team_count"),
            "unmapped_team_count": audit.get("unmapped_team_count"),
        },
        "quality": {
            "duplicate_raw_team_id_count": audit.get("duplicate_raw_team_id_count"),
            "missing_raw_team_id_count": audit.get("missing_raw_team_id_count"),
            "total_prev_season_points": audit.get("total_prev_season_points"),
            "total_prev_season_matches": audit.get("total_prev_season_matches"),
            "top_team_by_points": audit.get("top_team_by_points"),
            "bottom_team_by_points": audit.get("bottom_team_by_points"),
        },
        "identity_columns": audit.get("identity_columns"),
        "output_columns": audit.get("output_columns"),
        "notes": [
            "This artifact is read-only and does not write to the database.",
            "Team priors use historical raw team identity only.",
            "Promoted/relegated teams are not resolved in Day68A.",
            "Day68B should audit historical-to-target team identity mapping before any canonical team join.",
        ],
        "warnings": audit.get("warnings", []),
        "errors": audit.get("errors", []),
    }


def write_json(report: Dict[str, Any], out_json: str) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    if not out_md:
        return
    lines: List[str] = []
    lines.extend([
        "# Day68A — Historical Staging Team Priors",
        "",
        "Created at: `%s`" % report["created_at"],
        "",
        "Source season: `%s`" % report["source_season"],
        "Target season: `%s`" % report["target_season"],
        "",
        "## Summary",
        "",
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `%s`" % report["audit_only"],
        "- Writes database: `%s`" % report["writes_database"],
        "- Ready for team mapping audit: `%s`" % report["ready_for_team_mapping_audit"],
        "- Ready for canonical team join: `%s`" % report["ready_for_canonical_team_join"],
        "",
        "## Row Counts",
        "",
    ])
    for key, value in report["row_counts"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.extend(["", "## Quality", ""])
    for key, value in report["quality"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.extend(["", "## Identity Columns", ""])
    for key, value in report["identity_columns"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append("- %s" % note)
    lines.extend(["", "## Warnings", ""])
    lines.extend(["- %s" % warning for warning in report["warnings"]] or ["- none"])
    lines.extend(["", "## Errors", ""])
    lines.extend(["- %s" % error for error in report["errors"]] or ["- none"])
    lines.append("")
    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: Dict[str, Any], out_json: str, out_md: str) -> None:
    print("=== Day68A Historical Staging Team Priors ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("ready_for_team_mapping_audit:", report["ready_for_team_mapping_audit"])
    print("ready_for_canonical_team_join:", report["ready_for_canonical_team_join"])
    print("saved_csv:", report["out_csv"])
    if out_json:
        print("saved_json:", out_json)
    if out_md:
        print("saved_md:", out_md)
    print()
    print("Row counts:")
    for key, value in report["row_counts"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Quality:")
    for key, value in report["quality"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Warnings:", report["warnings"] or "none")
    print("Errors:", report["errors"] or "none")


def main() -> None:
    args = parse_args()
    team_summary = load_team_summary(args.team_summary_csv)
    priors, audit = build_team_priors(args.source_season, args.target_season, team_summary)

    if not priors.empty:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        priors.to_csv(out_csv, index=False)

    report = build_report(args.source_season, args.target_season, args.team_summary_csv, args.out_csv, priors, audit)
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report, args.out_json, args.out_md)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
