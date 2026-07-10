from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_SAFE_PRIOR_COLUMNS = [
    "prev_season_minutes",
    "prev_season_appearances",
    "prev_season_starts_proxy",
    "prev_season_starts_proxy_rate",
    "prev_season_total_points",
    "prev_season_points_per_appearance",
    "prev_season_points_per90",
    "prev_season_minutes_per_appearance",
    "prev_season_goals",
    "prev_season_assists",
    "prev_season_clean_sheets",
    "prev_season_bonus",
    "prev_season_latest_value",
    "prev_season_max_value",
    "prev_season_negative_points_gws",
    "prev_season_zero_minute_rows",
    "is_prev_season_active",
    "has_prev_season_data",
]

DEFAULT_ACCEPTED_STATUSES = ["auto_approved_candidate"]


class AuditError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run audit for joining previous-season staging priors into target-season "
            "player feature rows. This script is read-only and does not write to DB."
        )
    )
    parser.add_argument("--source-season", required=True, help="Prior/source season, for example 2024_25.")
    parser.add_argument("--target-season", required=True, help="Feature/target season, for example 2025_26.")
    parser.add_argument("--prior-csv", required=True, help="Day65 player priors CSV path.")
    parser.add_argument("--mapping-csv", required=True, help="Day66B player identity mapping candidates CSV path.")
    parser.add_argument("--player-features-csv", required=True, help="Target-season player feature CSV path.")
    parser.add_argument("--out-json", required=True, help="Output JSON report path.")
    parser.add_argument("--out-md", default="", help="Optional output Markdown report path.")
    parser.add_argument(
        "--out-joined-preview-csv",
        default="",
        help="Optional joined feature preview CSV path. This is an artifact only, not a DB write.",
    )
    parser.add_argument(
        "--accepted-statuses",
        default=",".join(DEFAULT_ACCEPTED_STATUSES),
        help="Comma-separated mapping statuses accepted for join. Default: auto_approved_candidate.",
    )
    parser.add_argument(
        "--safe-prior-columns",
        default=",".join(DEFAULT_SAFE_PRIOR_COLUMNS),
        help="Comma-separated prior columns to attempt to join. Missing columns are ignored and reported.",
    )
    parser.add_argument(
        "--max-example-rows",
        type=int,
        default=20,
        help="Maximum examples to include in report lists. Default: 20.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_raw_id(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def nullable_int(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def require_file(path_value: str, label: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise AuditError("%s does not exist: %s" % (label, path))
    return path


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise AuditError("%s is missing required columns: %s" % (label, missing))


def load_prior_csv(path: Path, source_season: str, target_season: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["raw_player_id"], "prior CSV")
    df["raw_player_id"] = df["raw_player_id"].apply(normalize_raw_id)

    if "source_season" in df.columns:
        bad_source = sorted(set(df.loc[df["source_season"] != source_season, "source_season"].dropna().astype(str)))
        if bad_source:
            raise AuditError("prior CSV contains unexpected source_season values: %s" % bad_source)

    if "target_season" in df.columns:
        bad_target = sorted(set(df.loc[df["target_season"] != target_season, "target_season"].dropna().astype(str)))
        if bad_target:
            raise AuditError("prior CSV contains unexpected target_season values: %s" % bad_target)

    return df


def load_mapping_csv(path: Path, source_season: str, target_season: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(
        df,
        [
            "source_season",
            "target_season",
            "raw_player_id",
            "candidate_player_id",
            "candidate_rank",
            "match_status",
            "needs_manual_review",
        ],
        "mapping CSV",
    )
    df["raw_player_id"] = df["raw_player_id"].apply(normalize_raw_id)
    df["candidate_player_id"] = df["candidate_player_id"].apply(nullable_int)

    bad_source = sorted(set(df.loc[df["source_season"] != source_season, "source_season"].dropna().astype(str)))
    bad_target = sorted(set(df.loc[df["target_season"] != target_season, "target_season"].dropna().astype(str)))
    if bad_source:
        raise AuditError("mapping CSV contains unexpected source_season values: %s" % bad_source)
    if bad_target:
        raise AuditError("mapping CSV contains unexpected target_season values: %s" % bad_target)

    return df


def load_features_csv(path: Path, target_season: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["player_id", "gw", "season"], "player features CSV")
    df["player_id"] = df["player_id"].apply(nullable_int)
    df["gw"] = df["gw"].apply(nullable_int)

    bad_seasons = sorted(set(df.loc[df["season"] != target_season, "season"].dropna().astype(str)))
    if bad_seasons:
        raise AuditError("player features CSV contains unexpected season values: %s" % bad_seasons)

    return df


def top_mapping_rows(mapping: pd.DataFrame) -> pd.DataFrame:
    rank = pd.to_numeric(mapping["candidate_rank"], errors="coerce")
    return mapping[(rank.isna()) | (rank == 1)].copy()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def accepted_mapping_rows(mapping: pd.DataFrame, accepted_statuses: Sequence[str]) -> pd.DataFrame:
    top = top_mapping_rows(mapping)
    accepted = top[top["match_status"].isin(accepted_statuses)].copy()

    if "is_auto_approved" in accepted.columns:
        accepted = accepted[bool_series(accepted["is_auto_approved"])]

    accepted = accepted[~bool_series(accepted["needs_manual_review"])]

    if "safe_name_match_for_auto_approval" in accepted.columns:
        accepted = accepted[bool_series(accepted["safe_name_match_for_auto_approval"])]

    accepted = accepted[accepted["candidate_player_id"].notna()].copy()
    accepted["candidate_player_id"] = accepted["candidate_player_id"].apply(nullable_int)
    return accepted


def build_prior_by_player(
    priors: pd.DataFrame,
    accepted: pd.DataFrame,
    safe_prior_columns: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    available_prior_columns = [col for col in safe_prior_columns if col in priors.columns]
    missing_prior_columns = [col for col in safe_prior_columns if col not in priors.columns]

    prior_meta_cols = [
        col
        for col in [
            "source_season",
            "target_season",
            "raw_player_id",
            "raw_player_name",
            "raw_team_id",
            "raw_position",
            "prior_identity_scope",
            "prior_source",
            "mapping_status",
            "mapping_confidence",
            "notes",
        ]
        if col in priors.columns
    ]

    mapping_cols = [
        col
        for col in [
            "raw_player_id",
            "candidate_player_id",
            "candidate_web_name",
            "candidate_full_name",
            "candidate_team_id",
            "candidate_position",
            "candidate_score",
            "match_status",
            "team_match",
            "exact_web_name_match",
            "exact_full_name_match",
            "exact_initial_surname_match",
            "safe_name_match_for_auto_approval",
        ]
        if col in accepted.columns
    ]

    joined = accepted[mapping_cols].merge(
        priors[prior_meta_cols + available_prior_columns].drop_duplicates("raw_player_id"),
        on="raw_player_id",
        how="left",
        validate="one_to_one",
    )

    rename_map = {
        "candidate_player_id": "player_id",
        "candidate_web_name": "prior_mapped_candidate_web_name",
        "candidate_full_name": "prior_mapped_candidate_full_name",
        "candidate_team_id": "prior_mapped_candidate_team_id",
        "candidate_position": "prior_mapped_candidate_position",
        "candidate_score": "prior_mapping_candidate_score",
        "match_status": "prior_mapping_status",
        "team_match": "prior_mapping_team_match",
        "exact_web_name_match": "prior_mapping_exact_web_name_match",
        "exact_full_name_match": "prior_mapping_exact_full_name_match",
        "exact_initial_surname_match": "prior_mapping_exact_initial_surname_match",
        "safe_name_match_for_auto_approval": "prior_mapping_safe_name_match",
        "raw_player_id": "prior_raw_player_id",
        "raw_player_name": "prior_raw_player_name",
        "raw_team_id": "prior_raw_team_id",
        "raw_position": "prior_raw_position",
        "source_season": "prior_source_season",
        "target_season": "prior_target_season",
        "mapping_status": "prior_original_mapping_status",
        "mapping_confidence": "prior_original_mapping_confidence",
        "notes": "prior_notes",
    }
    joined = joined.rename(columns=rename_map)
    joined["player_id"] = joined["player_id"].apply(nullable_int)
    return joined, available_prior_columns, missing_prior_columns


def feature_key_duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=["season", "player_id", "gw"]).sum())


def report_examples(df: pd.DataFrame, columns: Sequence[str], max_rows: int) -> List[Dict[str, Any]]:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return []
    return df[existing].head(max_rows).to_dict(orient="records")


def safe_json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (pd.isna(value) or value == float("inf") or value == float("-inf")):
            return None
        return value
    if pd.isna(value):
        return None
    return str(value)


def clean_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    return safe_json_value(obj)


def build_markdown(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Day66C Previous-Season Prior Join Dry-Run")
    lines.append("")
    lines.append("Created at: `%s`" % report["created_at"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in [
        "source_season",
        "target_season",
        "passed",
        "ready_for_prior_feature_export",
        "audit_only",
        "writes_database",
    ]:
        lines.append("- %s: `%s`" % (key, report.get(key)))
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
    lines.append("## Join Summary")
    lines.append("")
    for key, value in report["join_summary"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Safe Prior Columns")
    lines.append("")
    for col in report["safe_prior_columns"]["available"]:
        lines.append("- `%s`" % col)
    if report["safe_prior_columns"]["missing"]:
        lines.append("")
        lines.append("Missing requested columns:")
        for col in report["safe_prior_columns"]["missing"]:
            lines.append("- `%s`" % col)
    lines.append("")
    if report["blockers"]:
        lines.append("## Blockers")
        lines.append("")
        for item in report["blockers"]:
            lines.append("- %s" % item)
        lines.append("")
    if report["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for item in report["warnings"]:
            lines.append("- %s" % item)
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    for item in report["notes"]:
        lines.append("- %s" % item)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    accepted_statuses = split_csv_arg(args.accepted_statuses)
    safe_prior_columns = split_csv_arg(args.safe_prior_columns)

    prior_path = require_file(args.prior_csv, "prior CSV")
    mapping_path = require_file(args.mapping_csv, "mapping CSV")
    features_path = require_file(args.player_features_csv, "player features CSV")

    priors = load_prior_csv(prior_path, args.source_season, args.target_season)
    mapping = load_mapping_csv(mapping_path, args.source_season, args.target_season)
    features = load_features_csv(features_path, args.target_season)

    top = top_mapping_rows(mapping)
    accepted = accepted_mapping_rows(mapping, accepted_statuses)
    prior_by_player, available_prior_columns, missing_prior_columns = build_prior_by_player(
        priors=priors,
        accepted=accepted,
        safe_prior_columns=safe_prior_columns,
    )

    blockers: List[str] = []
    warnings: List[str] = []

    if len(top) != priors["raw_player_id"].nunique():
        warnings.append(
            "Top mapping rows (%s) do not equal unique prior raw players (%s). This can be okay if priors contain extra rows, but review."
            % (len(top), priors["raw_player_id"].nunique())
        )

    duplicate_accepted_raw = int(accepted["raw_player_id"].duplicated().sum())
    duplicate_accepted_player_id = int(accepted["candidate_player_id"].dropna().duplicated().sum())
    if duplicate_accepted_raw:
        blockers.append("Accepted mappings contain duplicate raw_player_id values: %s" % duplicate_accepted_raw)
    if duplicate_accepted_player_id:
        blockers.append("Accepted mappings contain duplicate candidate_player_id values: %s" % duplicate_accepted_player_id)

    if len(accepted) == 0:
        blockers.append("No accepted mappings available for prior join.")

    existing_prior_cols_in_features = [col for col in available_prior_columns if col in features.columns]
    if existing_prior_cols_in_features:
        blockers.append(
            "Player features already contain prior columns that would collide: %s" % existing_prior_cols_in_features
        )

    original_row_count = int(len(features))
    original_key_dupes = feature_key_duplicate_count(features)
    if original_key_dupes:
        blockers.append("Original player features contain duplicate season/player_id/gw rows: %s" % original_key_dupes)

    joined = features.merge(prior_by_player, on="player_id", how="left", validate="many_to_one")
    joined_row_count = int(len(joined))
    joined_key_dupes = feature_key_duplicate_count(joined)

    if joined_row_count != original_row_count:
        blockers.append("Joined row count changed from %s to %s." % (original_row_count, joined_row_count))
    if joined_key_dupes:
        blockers.append("Joined features contain duplicate season/player_id/gw rows: %s" % joined_key_dupes)

    feature_players = set([int(x) for x in features["player_id"].dropna().unique()])
    accepted_players = set([int(x) for x in accepted["candidate_player_id"].dropna().unique()])
    accepted_players_in_features = sorted(feature_players.intersection(accepted_players))

    prior_flag_col = "prior_raw_player_id" if "prior_raw_player_id" in joined.columns else None
    feature_rows_with_prior = int(joined[prior_flag_col].notna().sum()) if prior_flag_col else 0
    feature_players_with_prior = int(joined.loc[joined[prior_flag_col].notna(), "player_id"].nunique()) if prior_flag_col else 0
    feature_players_without_prior = int(joined.loc[joined[prior_flag_col].isna(), "player_id"].nunique()) if prior_flag_col else len(feature_players)

    if feature_rows_with_prior == 0:
        blockers.append("Prior join produced zero feature rows with prior data.")

    prior_coverage_rate_players = (
        round(float(feature_players_with_prior) / float(len(feature_players)), 4) if feature_players else None
    )
    prior_coverage_rate_rows = round(float(feature_rows_with_prior) / float(len(joined)), 4) if len(joined) else None

    top_status_counts = top["match_status"].value_counts(dropna=False).to_dict()
    accepted_status_counts = accepted["match_status"].value_counts(dropna=False).to_dict()

    manual_review_count = int(top["needs_manual_review"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    ambiguous_count = int((top["match_status"] == "ambiguous_top_candidate").sum())
    duplicate_manual_count = int((top["match_status"] == "duplicate_auto_approved_manual_review").sum())
    unmatched_count = int((top["match_status"] == "unmatched").sum())

    if manual_review_count:
        warnings.append(
            "%s top mapping rows require manual review and are excluded from this dry-run join." % manual_review_count
        )
    if ambiguous_count:
        warnings.append("%s ambiguous top candidates are excluded from accepted mappings." % ambiguous_count)
    if duplicate_manual_count:
        warnings.append("%s duplicate auto-approved candidates were demoted and excluded." % duplicate_manual_count)
    if unmatched_count:
        warnings.append("%s historical players are unmatched and excluded." % unmatched_count)

    preview_path = None
    if args.out_joined_preview_csv:
        preview_path = Path(args.out_joined_preview_csv)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(preview_path, index=False)

    report: Dict[str, Any] = {
        "created_at": utc_now(),
        "source_season": args.source_season,
        "target_season": args.target_season,
        "passed": len(blockers) == 0,
        "ready_for_prior_feature_export": len(blockers) == 0,
        "audit_only": True,
        "writes_database": False,
        "inputs": {
            "prior_csv": str(prior_path),
            "mapping_csv": str(mapping_path),
            "player_features_csv": str(features_path),
        },
        "outputs": {
            "out_json": args.out_json,
            "out_md": args.out_md or None,
            "out_joined_preview_csv": str(preview_path) if preview_path else None,
        },
        "accepted_statuses": accepted_statuses,
        "row_counts": {
            "prior_rows": int(len(priors)),
            "mapping_rows": int(len(mapping)),
            "top_mapping_rows": int(len(top)),
            "accepted_mapping_rows": int(len(accepted)),
            "prior_by_player_rows": int(len(prior_by_player)),
            "feature_rows_before_join": original_row_count,
            "feature_rows_after_join": joined_row_count,
        },
        "mapping_summary": {
            "top_status_counts": top_status_counts,
            "accepted_status_counts": accepted_status_counts,
            "manual_review_count": manual_review_count,
            "ambiguous_count": ambiguous_count,
            "duplicate_auto_approved_manual_review_count": duplicate_manual_count,
            "unmatched_count": unmatched_count,
            "duplicate_accepted_raw_player_id_count": duplicate_accepted_raw,
            "duplicate_accepted_candidate_player_id_count": duplicate_accepted_player_id,
        },
        "join_summary": {
            "feature_unique_players": int(len(feature_players)),
            "accepted_canonical_players": int(len(accepted_players)),
            "accepted_canonical_players_present_in_features": int(len(accepted_players_in_features)),
            "feature_players_with_prior": feature_players_with_prior,
            "feature_players_without_prior": feature_players_without_prior,
            "feature_rows_with_prior": feature_rows_with_prior,
            "feature_rows_without_prior": int(len(joined) - feature_rows_with_prior),
            "prior_coverage_rate_players": prior_coverage_rate_players,
            "prior_coverage_rate_rows": prior_coverage_rate_rows,
            "original_feature_key_duplicate_count": original_key_dupes,
            "joined_feature_key_duplicate_count": joined_key_dupes,
            "row_count_preserved": joined_row_count == original_row_count,
        },
        "safe_prior_columns": {
            "requested": safe_prior_columns,
            "available": available_prior_columns,
            "missing": missing_prior_columns,
        },
        "blockers": blockers,
        "warnings": warnings,
        "examples": {
            "accepted_mapping_examples": report_examples(
                accepted,
                [
                    "raw_player_id",
                    "raw_player_name",
                    "candidate_player_id",
                    "candidate_web_name",
                    "candidate_full_name",
                    "candidate_score",
                    "match_status",
                    "team_match",
                ],
                args.max_example_rows,
            ),
            "manual_review_examples": report_examples(
                top[top["needs_manual_review"].astype(str).str.lower().isin(["true", "1", "yes"])],
                [
                    "raw_player_id",
                    "raw_player_name",
                    "candidate_player_id",
                    "candidate_web_name",
                    "candidate_full_name",
                    "candidate_score",
                    "match_status",
                    "match_reason",
                ],
                args.max_example_rows,
            ),
            "feature_players_without_prior_examples": report_examples(
                joined[joined[prior_flag_col].isna()].drop_duplicates("player_id") if prior_flag_col else joined.drop_duplicates("player_id"),
                ["player_id", "season", "gw", "team_id", "position_DEF", "position_MID", "position_FWD", "position_GKP"],
                args.max_example_rows,
            ),
        },
        "notes": [
            "This is a dry-run audit. It does not modify feature exporters, models, predictions, or database tables.",
            "Only accepted mapping statuses are joined. Manual-review, ambiguous, duplicate-demoted, and unmatched mappings are excluded.",
            "The join is valid only if row count and season/player_id/gw grain are preserved.",
            "Day66C does not decide that priors improve model quality; it only decides whether a safe feature join is possible.",
        ],
    }

    report = clean_for_json(report)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_markdown(report), encoding="utf-8")

    print("=== Day66C Previous-Season Prior Join Dry-Run ===")
    print("source_season:", args.source_season)
    print("target_season:", args.target_season)
    print("passed:", report["passed"])
    print("ready_for_prior_feature_export:", report["ready_for_prior_feature_export"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("saved_json:", out_json)
    if args.out_md:
        print("saved_md:", args.out_md)
    if preview_path:
        print("saved_joined_preview_csv:", preview_path)
    print()
    print("Row counts:")
    for key, value in report["row_counts"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Join summary:")
    for key, value in report["join_summary"].items():
        print("- %s: %s" % (key, value))
    print()
    if blockers:
        print("Blockers:")
        for item in blockers:
            print("-", item)
    else:
        print("Blockers: none")
    if warnings:
        print()
        print("Warnings:")
        for item in warnings:
            print("-", item)

    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
