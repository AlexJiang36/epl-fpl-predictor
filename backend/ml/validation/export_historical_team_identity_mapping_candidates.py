from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export candidate mappings from historical raw teams to target-season canonical teams. "
            "This is read-only and does not update the database."
        )
    )
    parser.add_argument("--source-season", required=True, help="Historical source season, for example 2024_25.")
    parser.add_argument("--target-season", required=True, help="Target season, for example 2025_26.")
    parser.add_argument("--team-prior-csv", required=True, help="Day68A team prior CSV.")
    parser.add_argument("--out-csv", required=True, help="Output candidate mapping CSV path.")
    parser.add_argument("--out-json", default="", help="Optional output JSON report path.")
    parser.add_argument("--out-md", default="", help="Optional output Markdown report path.")
    parser.add_argument("--max-candidates-per-team", type=int, default=5)
    parser.add_argument("--min-candidate-score", type=float, default=0.55)
    parser.add_argument("--auto-approve-threshold", type=float, default=0.93)
    parser.add_argument("--ambiguous-gap", type=float, default=0.03)
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


def simple_ascii(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text_value = str(value).strip().lower()
    replacements = {
        "á": "a", "à": "a", "ä": "a", "â": "a", "ã": "a", "å": "a",
        "ç": "c", "é": "e", "è": "e", "ë": "e", "ê": "e",
        "í": "i", "ì": "i", "ï": "i", "î": "i", "ñ": "n",
        "ó": "o", "ò": "o", "ö": "o", "ô": "o", "õ": "o", "ø": "o",
        "ú": "u", "ù": "u", "ü": "u", "û": "u", "ý": "y", "ÿ": "y",
        "š": "s", "ć": "c", "č": "c", "ž": "z", "ğ": "g", "ı": "i",
        "ł": "l", "ß": "ss", "’": "'", "‘": "'", "ʼ": "'",
    }
    for src, dst in replacements.items():
        text_value = text_value.replace(src, dst)
    return text_value


def normalize_name(value: Any) -> str:
    text_value = simple_ascii(value)
    text_value = text_value.replace("&", " and ")
    text_value = re.sub(r"[^a-z0-9]+", " ", text_value)
    text_value = re.sub(r"\s+", " ", text_value).strip()
    return text_value


def compact_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", simple_ascii(value))


def normalize_short_name(value: Any) -> str:
    return compact_name(value).upper()


def name_variants(value: Any) -> List[str]:
    normalized = normalize_name(value)
    compact = compact_name(value)
    variants = [normalized, compact]
    if normalized.startswith("man "):
        variants.append(normalized.replace("man ", "manchester ", 1))
    if normalized.startswith("nottm "):
        variants.append(normalized.replace("nottm ", "nottingham ", 1))
    if normalized.startswith("spurs"):
        variants.append("tottenham")
        variants.append("tottenham hotspur")
    return sorted(set([variant for variant in variants if variant]))


def best_name_similarity(left_variants: Sequence[str], right_variants: Sequence[str]) -> Tuple[float, str, str, str]:
    best_score = 0.0
    best_left = ""
    best_right = ""
    best_match_type = "none"

    for left in left_variants:
        for right in right_variants:
            if not left or not right:
                continue
            if left == right:
                score = 1.0
                match_type = "exact_name_variant"
            elif left in right or right in left:
                shorter = min(len(left), len(right))
                longer = max(len(left), len(right))
                score = 0.78 + 0.12 * (float(shorter) / float(longer))
                match_type = "contained_name_variant"
            else:
                score = SequenceMatcher(None, left, right).ratio()
                match_type = "fuzzy_name_variant"

            if score > best_score:
                best_score = score
                best_left = left
                best_right = right
                best_match_type = match_type

    return best_score, best_left, best_right, best_match_type


def load_team_priors(team_prior_csv: str, source_season: str, target_season: str) -> pd.DataFrame:
    path = Path(team_prior_csv)
    if not path.exists():
        raise RuntimeError("team prior CSV does not exist: %s" % path)

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("team prior CSV is empty: %s" % path)

    required = ["source_season", "target_season", "raw_team_id", "raw_team_name", "raw_team_short_name"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError("team prior CSV missing required columns: %s" % missing)

    df = df[
        (df["source_season"].astype(str) == str(source_season))
        & (df["target_season"].astype(str) == str(target_season))
    ].copy()
    if df.empty:
        raise RuntimeError(
            "No team prior rows found for source_season=%s target_season=%s."
            % (source_season, target_season)
        )

    df["raw_team_id"] = df["raw_team_id"].apply(normalize_raw_id)
    df["raw_team_short_name_normalized"] = df["raw_team_short_name"].apply(normalize_short_name)
    df["historical_name_variants"] = df["raw_team_name"].apply(name_variants)
    return df


def load_target_teams() -> pd.DataFrame:
    if not table_exists("teams"):
        raise RuntimeError("teams table does not exist.")

    columns = table_columns("teams")
    df = read_sql_dataframe("SELECT * FROM teams", {})
    if df.empty:
        raise RuntimeError("No target teams found in teams table.")

    team_id_col = first_existing_column(columns, ["id", "team_id"])
    fpl_team_id_col = first_existing_column(columns, ["fpl_team_id"])
    team_name_col = first_existing_column(columns, ["name", "team_name"])
    team_short_col = first_existing_column(columns, ["short_name", "code"])

    if not team_id_col:
        raise RuntimeError("teams table must contain id or team_id.")
    if not team_short_col:
        raise RuntimeError("teams table must contain short_name or code.")

    out = pd.DataFrame()
    out["candidate_team_id"] = df[team_id_col].apply(nullable_int)
    out["candidate_fpl_team_id"] = df[fpl_team_id_col].apply(nullable_int) if fpl_team_id_col else None
    out["candidate_team_name"] = df[team_name_col] if team_name_col else ""
    out["candidate_team_short_name"] = df[team_short_col]
    out["candidate_team_short_name_normalized"] = out["candidate_team_short_name"].apply(normalize_short_name)
    out["target_name_variants"] = out["candidate_team_name"].apply(name_variants)
    return out


def score_candidate(hrow: pd.Series, trow: pd.Series) -> Dict[str, Any]:
    hist_short = normalize_short_name(hrow.get("raw_team_short_name"))
    target_short = normalize_short_name(trow.get("candidate_team_short_name"))
    exact_short_name_match = bool(hist_short and target_short and hist_short == target_short)

    name_score, best_hist_name, best_target_name, name_match_type = best_name_similarity(
        hrow["historical_name_variants"],
        trow["target_name_variants"],
    )
    exact_name_match = name_match_type == "exact_name_variant"

    score = name_score * 0.35
    if exact_short_name_match:
        score += 0.65

    safe_team_match_for_auto_approval = bool(exact_short_name_match or exact_name_match)

    if exact_short_name_match and exact_name_match:
        score = max(score, 1.0)
        match_type = "exact_short_name_and_exact_name"
    elif exact_short_name_match:
        score = max(score, 0.96)
        match_type = "exact_short_name"
    elif exact_name_match:
        score = max(score, 0.94)
        match_type = "exact_name"
    else:
        match_type = name_match_type

    score = max(0.0, min(1.0, score))

    reasons = [
        "team_match_type=%s" % match_type,
        "name_score=%.4f" % name_score,
    ]
    if exact_short_name_match:
        reasons.append("exact_short_name_match")
    else:
        reasons.append("short_name_mismatch")
    if exact_name_match:
        reasons.append("exact_name_match")
    if safe_team_match_for_auto_approval:
        reasons.append("safe_auto_team_match")
    else:
        reasons.append("not_safe_auto_team_match")

    return {
        "candidate_score": round(float(score), 4),
        "name_score": round(float(name_score), 4),
        "best_historical_name_variant": best_hist_name,
        "best_target_name_variant": best_target_name,
        "team_match_type": match_type,
        "exact_short_name_match": exact_short_name_match,
        "exact_name_match": exact_name_match,
        "safe_team_match_for_auto_approval": safe_team_match_for_auto_approval,
        "match_reason": "; ".join(reasons),
    }


def base_candidate_row(hrow: pd.Series, trow: Optional[pd.Series], score: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "source_season": hrow.get("source_season"),
        "target_season": hrow.get("target_season"),
        "raw_team_id": hrow.get("raw_team_id"),
        "raw_team_name": hrow.get("raw_team_name"),
        "raw_team_short_name": hrow.get("raw_team_short_name"),
        "historical_existing_canonical_team_id": hrow.get("canonical_team_id"),
        "historical_existing_mapping_status": hrow.get("mapping_status"),
    }

    if trow is not None:
        row.update(
            {
                "candidate_team_id": trow.get("candidate_team_id"),
                "candidate_fpl_team_id": trow.get("candidate_fpl_team_id"),
                "candidate_team_name": trow.get("candidate_team_name"),
                "candidate_team_short_name": trow.get("candidate_team_short_name"),
            }
        )
    else:
        row.update(
            {
                "candidate_team_id": None,
                "candidate_fpl_team_id": None,
                "candidate_team_name": None,
                "candidate_team_short_name": None,
            }
        )

    if score:
        row.update(score)

    return row


def build_candidates(
    team_priors: pd.DataFrame,
    target_teams: pd.DataFrame,
    max_candidates_per_team: int,
    min_candidate_score: float,
    auto_approve_threshold: float,
    ambiguous_gap: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, hrow in team_priors.iterrows():
        candidate_rows: List[Dict[str, Any]] = []

        for _, trow in target_teams.iterrows():
            score = score_candidate(hrow, trow)
            row = base_candidate_row(hrow, trow, score)
            if score["candidate_score"] >= min_candidate_score:
                candidate_rows.append(row)

        candidate_rows = sorted(
            candidate_rows,
            key=lambda row: (
                row["candidate_score"],
                row["name_score"],
                bool(row["exact_short_name_match"]),
                bool(row["exact_name_match"]),
            ),
            reverse=True,
        )

        if not candidate_rows:
            unmatched = base_candidate_row(hrow, None, None)
            unmatched.update(
                {
                    "candidate_rank": None,
                    "candidate_count": 0,
                    "candidate_score": 0.0,
                    "name_score": 0.0,
                    "score_gap_to_next": None,
                    "best_historical_name_variant": None,
                    "best_target_name_variant": None,
                    "team_match_type": "unmatched",
                    "exact_short_name_match": False,
                    "exact_name_match": False,
                    "safe_team_match_for_auto_approval": False,
                    "match_status": "historical_only_unmatched",
                    "is_auto_approved": False,
                    "needs_manual_review": True,
                    "is_ambiguous": False,
                    "match_reason": "no candidate above threshold; likely promoted/relegated or renamed team",
                }
            )
            rows.append(unmatched)
            continue

        candidate_rows = candidate_rows[:max_candidates_per_team]
        candidate_count = len(candidate_rows)
        top_score = float(candidate_rows[0]["candidate_score"])
        second_score = float(candidate_rows[1]["candidate_score"]) if len(candidate_rows) > 1 else None
        score_gap = None if second_score is None else round(top_score - second_score, 4)
        is_ambiguous = second_score is not None and (top_score - second_score) <= ambiguous_gap

        for rank, candidate in enumerate(candidate_rows, start=1):
            candidate["candidate_rank"] = rank
            candidate["candidate_count"] = candidate_count
            candidate["score_gap_to_next"] = score_gap if rank == 1 else None

            safe_team_match = bool(candidate.get("safe_team_match_for_auto_approval"))

            if rank == 1 and top_score >= auto_approve_threshold and not is_ambiguous and safe_team_match:
                candidate["match_status"] = "auto_approved_team_candidate"
                candidate["is_auto_approved"] = True
                candidate["needs_manual_review"] = False
                candidate["is_ambiguous"] = False
            elif rank == 1 and top_score >= auto_approve_threshold and not is_ambiguous:
                candidate["match_status"] = "high_score_manual_review_team_candidate"
                candidate["is_auto_approved"] = False
                candidate["needs_manual_review"] = True
                candidate["is_ambiguous"] = False
            elif rank == 1 and is_ambiguous:
                candidate["match_status"] = "ambiguous_team_candidate"
                candidate["is_auto_approved"] = False
                candidate["needs_manual_review"] = True
                candidate["is_ambiguous"] = True
            elif rank == 1:
                candidate["match_status"] = "low_confidence_team_candidate"
                candidate["is_auto_approved"] = False
                candidate["needs_manual_review"] = True
                candidate["is_ambiguous"] = False
            else:
                candidate["match_status"] = "alternative_team_candidate"
                candidate["is_auto_approved"] = False
                candidate["needs_manual_review"] = True
                candidate["is_ambiguous"] = is_ambiguous

            rows.append(candidate)

    candidates = pd.DataFrame(rows)
    return demote_duplicate_auto_approved_candidates(candidates)


def demote_duplicate_auto_approved_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    required = {"candidate_rank", "match_status", "candidate_team_id"}
    if not required.issubset(set(candidates.columns)):
        return candidates

    result = candidates.copy()
    result["duplicate_auto_approved_candidate_team_id"] = False

    top_auto_mask = (
        (result["candidate_rank"] == 1)
        & (result["match_status"] == "auto_approved_team_candidate")
        & result["candidate_team_id"].notna()
    )

    auto_top = result[top_auto_mask].copy()
    duplicate_candidate_ids = auto_top.loc[
        auto_top["candidate_team_id"].duplicated(keep=False),
        "candidate_team_id",
    ].dropna().unique()

    if len(duplicate_candidate_ids) == 0:
        return result

    duplicate_mask = top_auto_mask & result["candidate_team_id"].isin(duplicate_candidate_ids)
    result.loc[duplicate_mask, "duplicate_auto_approved_candidate_team_id"] = True
    result.loc[duplicate_mask, "match_status"] = "duplicate_auto_approved_team_manual_review"
    result.loc[duplicate_mask, "is_auto_approved"] = False
    result.loc[duplicate_mask, "needs_manual_review"] = True
    result.loc[duplicate_mask, "is_ambiguous"] = True
    result.loc[duplicate_mask, "match_reason"] = (
        result.loc[duplicate_mask, "match_reason"].astype(str)
        + "; duplicate_auto_approved_candidate_team_id"
    )
    return result


def build_report(
    candidates: pd.DataFrame,
    team_priors: pd.DataFrame,
    target_teams: pd.DataFrame,
    source_season: str,
    target_season: str,
    team_prior_csv: str,
    out_csv: str,
    auto_approve_threshold: float,
    ambiguous_gap: float,
) -> Dict[str, Any]:
    top_rows = candidates[(candidates["candidate_rank"].isna()) | (candidates["candidate_rank"] == 1)].copy()
    auto_rows = top_rows[top_rows["match_status"] == "auto_approved_team_candidate"].copy()

    auto_count = int((top_rows["match_status"] == "auto_approved_team_candidate").sum())
    ambiguous_count = int((top_rows["match_status"] == "ambiguous_team_candidate").sum())
    low_conf_count = int((top_rows["match_status"] == "low_confidence_team_candidate").sum())
    high_score_manual_count = int((top_rows["match_status"] == "high_score_manual_review_team_candidate").sum())
    duplicate_auto_manual_count = int((top_rows["match_status"] == "duplicate_auto_approved_team_manual_review").sum())
    unmatched_count = int((top_rows["match_status"] == "historical_only_unmatched").sum())
    manual_review_count = int(top_rows["needs_manual_review"].fillna(True).sum())

    duplicate_accepted_candidate_team_id_count = int(auto_rows["candidate_team_id"].dropna().duplicated().sum())
    duplicate_accepted_raw_team_id_count = int(auto_rows["raw_team_id"].dropna().astype(str).duplicated().sum())
    unsafe_accepted_team_match_count = int((auto_rows["safe_team_match_for_auto_approval"] != True).sum()) if len(auto_rows) else 0

    source_short_names = set(team_priors["raw_team_short_name"].dropna().astype(str).tolist())
    target_short_names = set(target_teams["candidate_team_short_name"].dropna().astype(str).tolist())
    historical_only_short_names = sorted(source_short_names - target_short_names)
    target_only_short_names = sorted(target_short_names - source_short_names)

    errors: List[str] = []
    if len(top_rows) != len(team_priors):
        errors.append(
            "Expected exactly one top/unmatched row per historical team, got %s vs %s."
            % (len(top_rows), len(team_priors))
        )
    if duplicate_accepted_candidate_team_id_count > 0:
        errors.append("Accepted team mappings contain duplicate candidate_team_id.")
    if duplicate_accepted_raw_team_id_count > 0:
        errors.append("Accepted team mappings contain duplicate raw_team_id.")
    if unsafe_accepted_team_match_count > 0:
        errors.append("Accepted team mappings contain unsafe team matches.")

    return {
        "created_at": utc_now(),
        "source_season": source_season,
        "target_season": target_season,
        "team_prior_csv": team_prior_csv,
        "out_csv": out_csv,
        "passed": len(errors) == 0,
        "audit_only": True,
        "writes_database": False,
        "ready_for_team_prior_join": len(errors) == 0 and auto_count > 0,
        "ready_for_full_pre_gw1_match_pipeline": False,
        "reason_full_pre_gw1_match_pipeline_not_ready": (
            "Day68B audits identity mapping only. Team-prior feature joins and match prediction scaffolding "
            "should be implemented separately."
        ),
        "thresholds": {
            "auto_approve_threshold": auto_approve_threshold,
            "ambiguous_gap": ambiguous_gap,
        },
        "row_counts": {
            "team_prior_rows": int(len(team_priors)),
            "target_team_rows": int(len(target_teams)),
            "candidate_rows": int(len(candidates)),
            "top_candidate_or_unmatched_rows": int(len(top_rows)),
        },
        "mapping_summary": {
            "auto_approved_count": auto_count,
            "ambiguous_count": ambiguous_count,
            "low_confidence_count": low_conf_count,
            "high_score_manual_review_count": high_score_manual_count,
            "duplicate_auto_approved_manual_review_count": duplicate_auto_manual_count,
            "unmatched_count": unmatched_count,
            "manual_review_count": manual_review_count,
            "auto_approved_rate": round(auto_count / float(len(team_priors)), 4) if len(team_priors) else None,
            "duplicate_accepted_candidate_team_id_count": duplicate_accepted_candidate_team_id_count,
            "duplicate_accepted_raw_team_id_count": duplicate_accepted_raw_team_id_count,
            "unsafe_accepted_team_match_count": unsafe_accepted_team_match_count,
        },
        "season_transition_summary": {
            "historical_only_short_names": historical_only_short_names,
            "target_only_short_names": target_only_short_names,
            "expected_relegated_or_missing_source_teams": historical_only_short_names,
            "expected_promoted_or_new_target_teams": target_only_short_names,
        },
        "status_counts": top_rows["match_status"].value_counts(dropna=False).to_dict(),
        "team_match_type_counts": candidates["team_match_type"].value_counts(dropna=False).to_dict(),
        "candidate_score_summary": {
            "min": float(candidates["candidate_score"].min()) if len(candidates) else None,
            "max": float(candidates["candidate_score"].max()) if len(candidates) else None,
            "mean": float(candidates["candidate_score"].mean()) if len(candidates) else None,
        },
        "top_auto_approved_examples": auto_rows.head(10).to_dict(orient="records"),
        "top_manual_review_examples": top_rows[top_rows["needs_manual_review"] == True].head(10).to_dict(orient="records"),
        "notes": [
            "This report is audit-only and does not update team priors or teams.",
            "Only auto_approved_team_candidate rows should be consumed automatically by future team prior joins.",
            "Historical-only teams are expected when a source-season club was relegated or is absent in the target season.",
            "Target-only teams are expected when a target-season club was promoted or newly present.",
        ],
        "errors": errors,
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
    lines.append("# Day68B — Historical Team Identity Mapping Audit")
    lines.append("")
    lines.append("Created at: `%s`" % report["created_at"])
    lines.append("")
    lines.append("Source season: `%s`" % report["source_season"])
    lines.append("Target season: `%s`" % report["target_season"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Passed: `%s`" % report["passed"])
    lines.append("- Audit only: `%s`" % report["audit_only"])
    lines.append("- Writes database: `%s`" % report["writes_database"])
    lines.append("- Ready for team prior join: `%s`" % report["ready_for_team_prior_join"])
    lines.append("- Ready for full Pre-GW1 match pipeline: `%s`" % report["ready_for_full_pre_gw1_match_pipeline"])
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
    lines.append("## Season Transition Summary")
    lines.append("")
    for key, value in report["season_transition_summary"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Status Counts")
    lines.append("")
    for key, value in report["status_counts"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append("- %s" % note)
    lines.append("")
    lines.append("## Errors")
    lines.append("")
    if report["errors"]:
        for error in report["errors"]:
            lines.append("- %s" % error)
    else:
        lines.append("- none")
    lines.append("")

    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: Dict[str, Any], out_json: str, out_md: str) -> None:
    print("=== Day68B Historical Team Identity Mapping Audit ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("ready_for_team_prior_join:", report["ready_for_team_prior_join"])
    print("ready_for_full_pre_gw1_match_pipeline:", report["ready_for_full_pre_gw1_match_pipeline"])
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
    print("Mapping summary:")
    for key, value in report["mapping_summary"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Season transition summary:")
    for key, value in report["season_transition_summary"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Status counts:")
    for key, value in report["status_counts"].items():
        print("- %s: %s" % (key, value))
    if report["errors"]:
        print()
        print("Errors:")
        for error in report["errors"]:
            print("-", error)


def main() -> None:
    args = parse_args()

    team_priors = load_team_priors(args.team_prior_csv, args.source_season, args.target_season)
    target_teams = load_target_teams()

    candidates = build_candidates(
        team_priors=team_priors,
        target_teams=target_teams,
        max_candidates_per_team=args.max_candidates_per_team,
        min_candidate_score=args.min_candidate_score,
        auto_approve_threshold=args.auto_approve_threshold,
        ambiguous_gap=args.ambiguous_gap,
    )

    ordered_first = [
        "source_season",
        "target_season",
        "raw_team_id",
        "raw_team_name",
        "raw_team_short_name",
        "candidate_rank",
        "candidate_count",
        "candidate_team_id",
        "candidate_fpl_team_id",
        "candidate_team_name",
        "candidate_team_short_name",
        "candidate_score",
        "name_score",
        "score_gap_to_next",
        "match_status",
        "is_auto_approved",
        "needs_manual_review",
        "is_ambiguous",
        "duplicate_auto_approved_candidate_team_id",
        "exact_short_name_match",
        "exact_name_match",
        "safe_team_match_for_auto_approval",
        "team_match_type",
        "best_historical_name_variant",
        "best_target_name_variant",
        "match_reason",
    ]
    other_cols = [col for col in candidates.columns if col not in ordered_first]
    candidates = candidates[[col for col in ordered_first if col in candidates.columns] + other_cols]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_csv, index=False)

    report = build_report(
        candidates=candidates,
        team_priors=team_priors,
        target_teams=target_teams,
        source_season=args.source_season,
        target_season=args.target_season,
        team_prior_csv=args.team_prior_csv,
        out_csv=str(out_csv),
        auto_approve_threshold=args.auto_approve_threshold,
        ambiguous_gap=args.ambiguous_gap,
    )

    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report, args.out_json, args.out_md)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
