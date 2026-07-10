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


POSITION_NAME_BY_ID = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export candidate mappings from historical raw players to target-season canonical players. "
            "This is read-only and does not update the database."
        )
    )
    parser.add_argument("--source-season", required=True, help="Historical source season, for example 2024_25.")
    parser.add_argument("--target-season", required=True, help="Target season, for example 2025_26.")
    parser.add_argument("--prior-csv", default="", help="Optional Day65 player prior CSV for context.")
    parser.add_argument("--out-csv", required=True, help="Output candidate mapping CSV path.")
    parser.add_argument("--out-json", default="", help="Optional output JSON report path.")
    parser.add_argument("--max-candidates-per-player", type=int, default=5)
    parser.add_argument("--min-candidate-score", type=float, default=0.55)
    parser.add_argument("--auto-approve-threshold", type=float, default=0.93)
    parser.add_argument("--ambiguous-gap", type=float, default=0.03)
    parser.add_argument(
        "--include-low-confidence",
        action="store_true",
        help="Keep one best low-confidence candidate below --min-candidate-score instead of only unmatched row.",
    )
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


def normalize_position(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None

    text_value = str(value).strip()
    if not text_value:
        return None

    upper_value = text_value.upper()

    position_lookup = {
        "1": 1,
        "GKP": 1,
        "GK": 1,
        "GOALKEEPER": 1,
        "2": 2,
        "DEF": 2,
        "DEFENDER": 2,
        "3": 3,
        "MID": 3,
        "MIDFIELDER": 3,
        "4": 4,
        "FWD": 4,
        "FOR": 4,
        "FORWARD": 4,
        "STRIKER": 4,
    }

    if upper_value in position_lookup:
        return position_lookup[upper_value]

    return nullable_int(value)


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
        "š": "s", "ć": "c", "č": "c", "ž": "z", "ğ": "g", "ı": "i", "ł": "l", "ß": "ss",
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


def surname_from_name(value: Any) -> str:
    normalized = normalize_name(value)
    if not normalized:
        return ""
    return normalized.split()[-1]


def initial_surname_variant(first_name: Any, second_name: Any) -> str:
    first = normalize_name(first_name)
    second = normalize_name(second_name)
    if not first or not second:
        return ""
    return "%s%s" % (first[0], compact_name(second))


def historical_initial_surname(value: Any) -> str:
    text_value = simple_ascii(value).strip()
    match = re.match(r"^([a-z])[\.\s_-]*([a-z].*)$", text_value)
    if not match:
        return ""
    return "%s%s" % (match.group(1), compact_name(match.group(2)))



def safe_name_match_details(hrow: pd.Series, trow: pd.Series) -> Dict[str, bool]:
    raw_name = hrow.get("raw_player_name")

    raw_norm = normalize_name(raw_name)
    raw_compact = compact_name(raw_name)

    web_norm = normalize_name(trow.get("candidate_web_name"))
    web_compact = compact_name(trow.get("candidate_web_name"))

    full_norm = normalize_name(trow.get("candidate_full_name"))
    full_compact = compact_name(trow.get("candidate_full_name"))

    hist_initial = historical_initial_surname(raw_name)
    target_initial = initial_surname_variant(
        trow.get("candidate_first_name"),
        trow.get("candidate_second_name"),
    )

    exact_web_name_match = bool(
        raw_norm
        and web_norm
        and (
            raw_norm == web_norm
            or raw_compact == web_compact
        )
    )

    exact_full_name_match = bool(
        raw_norm
        and full_norm
        and (
            raw_norm == full_norm
            or raw_compact == full_compact
        )
    )

    exact_initial_surname_match = bool(
        hist_initial
        and target_initial
        and hist_initial == target_initial
    )

    return {
        "exact_web_name_match": exact_web_name_match,
        "exact_full_name_match": exact_full_name_match,
        "exact_initial_surname_match": exact_initial_surname_match,
        "safe_name_match_for_auto_approval": bool(
            exact_web_name_match
            or exact_full_name_match
            or exact_initial_surname_match
        ),
    }


def name_variants_for_historical(raw_name: Any) -> List[str]:
    variants = [
        normalize_name(raw_name),
        compact_name(raw_name),
        surname_from_name(raw_name),
        historical_initial_surname(raw_name),
    ]
    return sorted(set([variant for variant in variants if variant]))


def name_variants_for_target(row: pd.Series) -> List[str]:
    values: List[Any] = []
    for column in ["full_name", "display_name", "player_name", "name", "web_name", "first_name", "second_name"]:
        if column in row.index:
            values.append(row.get(column))

    first = row.get("first_name") if "first_name" in row.index else None
    second = row.get("second_name") if "second_name" in row.index else None
    if first is not None and second is not None and not pd.isna(first) and not pd.isna(second):
        values.append("%s %s" % (first, second))
        values.append(initial_surname_variant(first, second))

    variants: List[str] = []
    for value in values:
        variants.extend([normalize_name(value), compact_name(value), surname_from_name(value)])
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
                score = 1.0 if len(left) >= 4 else 0.85
                match_type = "exact_variant" if len(left) >= 4 else "short_exact_variant"
            elif left in right or right in left:
                shorter = min(len(left), len(right))
                longer = max(len(left), len(right))
                score = 0.78 + 0.12 * (float(shorter) / float(longer))
                match_type = "contained_variant"
            else:
                score = SequenceMatcher(None, left, right).ratio()
                match_type = "fuzzy_variant"

            if score > best_score:
                best_score = score
                best_left = left
                best_right = right
                best_match_type = match_type

    return best_score, best_left, best_right, best_match_type


def load_historical_players(source_season: str) -> pd.DataFrame:
    if not table_exists("historical_players"):
        raise RuntimeError("historical_players table does not exist.")

    df = read_sql_dataframe(
        """
        SELECT
            season,
            raw_player_id,
            raw_player_name,
            raw_team_id,
            raw_position,
            canonical_player_id,
            canonical_player_name,
            mapping_status,
            mapping_confidence,
            notes
        FROM historical_players
        WHERE season = :source_season
        ORDER BY raw_player_id
        """,
        {"source_season": source_season},
    )
    if df.empty:
        raise RuntimeError("No historical_players rows found for source_season=%s." % source_season)

    df["raw_player_id"] = df["raw_player_id"].apply(normalize_raw_id)
    df["raw_team_id"] = df["raw_team_id"].apply(normalize_raw_id)
    df["raw_position"] = df["raw_position"].apply(normalize_position)
    df["historical_name_variants"] = df["raw_player_name"].apply(name_variants_for_historical)
    return df


def load_historical_team_mapping(source_season: str) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=[
            "raw_team_id", "raw_team_name", "raw_team_short_name", "matched_team_id", "matched_team_name", "matched_team_short_name"
        ]
    )
    if not table_exists("historical_teams") or not table_exists("teams"):
        return empty

    historical_teams = read_sql_dataframe(
        """
        SELECT raw_team_id, raw_team_name, raw_team_short_name
        FROM historical_teams
        WHERE season = :source_season
        """,
        {"source_season": source_season},
    )
    teams = read_sql_dataframe("SELECT * FROM teams", {})
    if historical_teams.empty or teams.empty:
        return empty

    team_cols = list(teams.columns)
    team_id_col = first_existing_column(team_cols, ["id", "team_id"])
    team_name_col = first_existing_column(team_cols, ["name", "team_name"])
    team_short_col = first_existing_column(team_cols, ["short_name", "code"])
    if not team_id_col or not team_short_col:
        return empty

    historical_teams["raw_team_id"] = historical_teams["raw_team_id"].apply(normalize_raw_id)
    historical_teams["join_short_name"] = historical_teams["raw_team_short_name"].apply(compact_name)

    current = pd.DataFrame(
        {
            "matched_team_id": teams[team_id_col],
            "matched_team_name": teams[team_name_col] if team_name_col else None,
            "matched_team_short_name": teams[team_short_col],
        }
    )
    current["join_short_name"] = current["matched_team_short_name"].apply(compact_name)

    merged = historical_teams.merge(current, on="join_short_name", how="left")
    return merged[
        ["raw_team_id", "raw_team_name", "raw_team_short_name", "matched_team_id", "matched_team_name", "matched_team_short_name"]
    ]


def load_target_players(target_season: str) -> pd.DataFrame:
    if not table_exists("players"):
        raise RuntimeError("players table does not exist.")

    columns = table_columns("players")
    if "season" in columns:
        df = read_sql_dataframe("SELECT * FROM players WHERE season = :target_season", {"target_season": target_season})
    else:
        df = read_sql_dataframe("SELECT * FROM players", {})

    if df.empty:
        raise RuntimeError("No target players found for target_season=%s." % target_season)

    player_id_col = first_existing_column(columns, ["id", "player_id"])
    if not player_id_col:
        raise RuntimeError("players table must contain id or player_id.")

    web_name_col = first_existing_column(columns, ["web_name", "name", "player_name"])
    first_name_col = first_existing_column(columns, ["first_name"])
    second_name_col = first_existing_column(columns, ["second_name"])
    team_id_col = first_existing_column(columns, ["team_id"])
    position_col = first_existing_column(columns, ["element_type", "position", "position_id"])
    status_col = first_existing_column(columns, ["status"])
    cost_col = first_existing_column(columns, ["now_cost", "cost", "price"])

    out = pd.DataFrame()
    out["candidate_player_id"] = df[player_id_col].apply(nullable_int)
    out["candidate_web_name"] = df[web_name_col] if web_name_col else ""
    out["candidate_first_name"] = df[first_name_col] if first_name_col else ""
    out["candidate_second_name"] = df[second_name_col] if second_name_col else ""

    if first_name_col and second_name_col:
        out["candidate_full_name"] = (
            df[first_name_col].fillna("").astype(str).str.strip()
            + " "
            + df[second_name_col].fillna("").astype(str).str.strip()
        ).str.strip()
    elif web_name_col:
        out["candidate_full_name"] = df[web_name_col]
    else:
        out["candidate_full_name"] = ""

    out["candidate_team_id"] = df[team_id_col].apply(nullable_int) if team_id_col else None
    out["candidate_position"] = df[position_col].apply(normalize_position) if position_col else None
    out["candidate_status"] = df[status_col] if status_col else None
    out["candidate_now_cost"] = df[cost_col] if cost_col else None

    # Temporary normalized columns used by name_variants_for_target.
    out["web_name"] = out["candidate_web_name"]
    out["first_name"] = out["candidate_first_name"]
    out["second_name"] = out["candidate_second_name"]
    out["full_name"] = out["candidate_full_name"]
    out["target_name_variants"] = out.apply(name_variants_for_target, axis=1)
    return out


def load_prior_context(prior_csv: str) -> pd.DataFrame:
    if not prior_csv:
        return pd.DataFrame()
    path = Path(prior_csv)
    if not path.exists():
        raise RuntimeError("prior CSV does not exist: %s" % path)

    df = pd.read_csv(path)
    if "raw_player_id" not in df.columns:
        raise RuntimeError("prior CSV is missing raw_player_id column: %s" % path)

    df["raw_player_id"] = df["raw_player_id"].apply(normalize_raw_id)
    keep_cols = [
        col
        for col in [
            "raw_player_id",
            "prev_season_minutes",
            "prev_season_appearances",
            "prev_season_starts_proxy",
            "prev_season_total_points",
            "prev_season_points_per90",
            "prev_season_goals",
            "prev_season_assists",
            "prev_season_clean_sheets",
            "prev_season_bonus",
            "is_prev_season_active",
        ]
        if col in df.columns
    ]
    return df[keep_cols].drop_duplicates("raw_player_id")


def score_candidate(hrow: pd.Series, trow: pd.Series, raw_to_canonical_team: Dict[str, Optional[int]]) -> Dict[str, Any]:
    name_score, best_hist_name, best_target_name, name_match_type = best_name_similarity(
        hrow["historical_name_variants"],
        trow["target_name_variants"],
    )

    hist_position = normalize_position(hrow.get("raw_position"))
    target_position = normalize_position(trow.get("candidate_position"))
    position_match = hist_position is not None and target_position is not None and hist_position == target_position

    raw_team_id = normalize_raw_id(hrow.get("raw_team_id"))
    mapped_team_id = raw_to_canonical_team.get(raw_team_id)
    target_team_id = nullable_int(trow.get("candidate_team_id"))
    team_match = mapped_team_id is not None and target_team_id is not None and int(mapped_team_id) == int(target_team_id)

    score = name_score * 0.82
    if position_match:
        score += 0.12
    elif hist_position is not None and target_position is not None:
        score -= 0.10
    if team_match:
        score += 0.06

    hist_initial = historical_initial_surname(hrow.get("raw_player_name"))
    target_initial = initial_surname_variant(trow.get("candidate_first_name"), trow.get("candidate_second_name"))
    exact_initial_surname_match = bool(hist_initial and target_initial and hist_initial == target_initial)

    exact_variant_match = bool(best_hist_name and best_target_name and best_hist_name == best_target_name and len(best_hist_name) >= 4)

    if exact_initial_surname_match:
        score = max(score, 0.98 if position_match else 0.92)
        name_match_type = "exact_initial_surname"
    if exact_variant_match and position_match:
        score = max(score, 0.94)
        if name_match_type == "exact_variant":
            name_match_type = "exact_name_or_web_with_position"

    name_safety = safe_name_match_details(hrow, trow)

    score = max(0.0, min(1.0, score))

    reasons = ["name_match_type=%s" % name_match_type, "name_score=%.4f" % name_score]
    if name_safety["safe_name_match_for_auto_approval"]:
        reasons.append("safe_auto_name_match")
    else:
        reasons.append("not_safe_auto_name_match")
    if position_match:
        reasons.append("position_match")
    elif hist_position is not None and target_position is not None:
        reasons.append("position_mismatch")
    else:
        reasons.append("position_unknown")

    if team_match:
        reasons.append("team_short_name_match")
    elif mapped_team_id is not None and target_team_id is not None:
        reasons.append("mapped_team_mismatch_or_transfer")
    else:
        reasons.append("team_mapping_unknown")

    return {
        "candidate_score": round(float(score), 4),
        "name_score": round(float(name_score), 4),
        "best_historical_name_variant": best_hist_name,
        "best_target_name_variant": best_target_name,
        "name_match_type": name_match_type,
        "position_match": bool(position_match),
        "team_match": bool(team_match),
        "mapped_team_id_from_historical_team": mapped_team_id,
        "match_reason": "; ".join(reasons),
        "exact_web_name_match": name_safety["exact_web_name_match"],
        "exact_full_name_match": name_safety["exact_full_name_match"],
        "exact_initial_surname_match": name_safety["exact_initial_surname_match"],
        "safe_name_match_for_auto_approval": name_safety["safe_name_match_for_auto_approval"],
    }


def base_candidate_row(hrow: pd.Series, trow: Optional[pd.Series], score: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "source_season": hrow.get("season"),
        "raw_player_id": hrow.get("raw_player_id"),
        "raw_player_name": hrow.get("raw_player_name"),
        "raw_team_id": hrow.get("raw_team_id"),
        "raw_position": hrow.get("raw_position"),
        "raw_position_name": POSITION_NAME_BY_ID.get(normalize_position(hrow.get("raw_position"))),
        "historical_existing_canonical_player_id": hrow.get("canonical_player_id"),
        "historical_existing_mapping_status": hrow.get("mapping_status"),
    }
    if trow is not None:
        row.update(
            {
                "candidate_player_id": trow.get("candidate_player_id"),
                "candidate_web_name": trow.get("candidate_web_name"),
                "candidate_full_name": trow.get("candidate_full_name"),
                "candidate_team_id": trow.get("candidate_team_id"),
                "candidate_position": trow.get("candidate_position"),
                "candidate_position_name": POSITION_NAME_BY_ID.get(normalize_position(trow.get("candidate_position"))),
                "candidate_status": trow.get("candidate_status"),
                "candidate_now_cost": trow.get("candidate_now_cost"),
            }
        )
    else:
        row.update(
            {
                "candidate_player_id": None,
                "candidate_web_name": None,
                "candidate_full_name": None,
                "candidate_team_id": None,
                "candidate_position": None,
                "candidate_position_name": None,
                "candidate_status": None,
                "candidate_now_cost": None,
            }
        )
    if score:
        row.update(score)
    return row


def build_candidates(
    historical_players: pd.DataFrame,
    target_players: pd.DataFrame,
    raw_to_canonical_team: Dict[str, Optional[int]],
    max_candidates_per_player: int,
    min_candidate_score: float,
    auto_approve_threshold: float,
    ambiguous_gap: float,
    include_low_confidence: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, hrow in historical_players.iterrows():
        candidate_rows: List[Dict[str, Any]] = []
        low_rows: List[Dict[str, Any]] = []

        for _, trow in target_players.iterrows():
            score = score_candidate(hrow, trow, raw_to_canonical_team)
            row = base_candidate_row(hrow, trow, score)
            if score["candidate_score"] >= min_candidate_score:
                candidate_rows.append(row)
            else:
                low_rows.append(row)

        candidate_rows = sorted(
            candidate_rows,
            key=lambda row: (row["candidate_score"], row["name_score"], bool(row["position_match"]), bool(row["team_match"])),
            reverse=True,
        )

        if not candidate_rows and include_low_confidence:
            candidate_rows = sorted(low_rows, key=lambda row: row["candidate_score"], reverse=True)[:1]

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
                    "name_match_type": "unmatched",
                    "position_match": False,
                    "team_match": False,
                    "mapped_team_id_from_historical_team": raw_to_canonical_team.get(normalize_raw_id(hrow.get("raw_team_id"))),
                    "match_status": "unmatched",
                    "is_auto_approved": False,
                    "needs_manual_review": True,
                    "is_ambiguous": False,
                    "match_reason": "no candidate above threshold",
                }
            )
            rows.append(unmatched)
            continue

        candidate_rows = candidate_rows[:max_candidates_per_player]
        candidate_count = len(candidate_rows)
        top_score = float(candidate_rows[0]["candidate_score"])
        second_score = float(candidate_rows[1]["candidate_score"]) if len(candidate_rows) > 1 else None
        score_gap = None if second_score is None else round(top_score - second_score, 4)
        is_ambiguous = second_score is not None and (top_score - second_score) <= ambiguous_gap

        for rank, candidate in enumerate(candidate_rows, start=1):
            candidate["candidate_rank"] = rank
            candidate["candidate_count"] = candidate_count
            candidate["score_gap_to_next"] = score_gap if rank == 1 else None

            safe_name_match = bool(candidate.get("safe_name_match_for_auto_approval"))

            if rank == 1 and top_score >= auto_approve_threshold and not is_ambiguous and safe_name_match:
                candidate["match_status"] = "auto_approved_candidate"
                candidate["is_auto_approved"] = True
                candidate["needs_manual_review"] = False
                candidate["is_ambiguous"] = False
            elif rank == 1 and top_score >= auto_approve_threshold and not is_ambiguous:
                candidate["match_status"] = "high_score_manual_review_candidate"
                candidate["is_auto_approved"] = False
                candidate["needs_manual_review"] = True
                candidate["is_ambiguous"] = False
            elif rank == 1 and is_ambiguous:
                candidate["match_status"] = "ambiguous_top_candidate"
                candidate["is_auto_approved"] = False
                candidate["needs_manual_review"] = True
                candidate["is_ambiguous"] = True
            elif rank == 1:
                candidate["match_status"] = "low_confidence_top_candidate"
                candidate["is_auto_approved"] = False
                candidate["needs_manual_review"] = True
                candidate["is_ambiguous"] = False
            else:
                candidate["match_status"] = "alternative_candidate"
                candidate["is_auto_approved"] = False
                candidate["needs_manual_review"] = True
                candidate["is_ambiguous"] = is_ambiguous

            rows.append(candidate)

    return pd.DataFrame(rows)


def attach_prior_context(candidates: pd.DataFrame, prior_context: pd.DataFrame) -> pd.DataFrame:
    if prior_context.empty:
        return candidates
    return candidates.merge(prior_context, on="raw_player_id", how="left")



def demote_duplicate_auto_approved_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    required_columns = {"candidate_rank", "match_status", "candidate_player_id"}
    if not required_columns.issubset(set(candidates.columns)):
        return candidates

    result = candidates.copy()

    top_auto_mask = (
        (result["candidate_rank"] == 1)
        & (result["match_status"] == "auto_approved_candidate")
        & result["candidate_player_id"].notna()
    )

    auto_top = result[top_auto_mask].copy()
    duplicate_candidate_ids = auto_top.loc[
        auto_top["candidate_player_id"].duplicated(keep=False),
        "candidate_player_id",
    ].dropna().unique()

    if len(duplicate_candidate_ids) == 0:
        result["duplicate_auto_approved_candidate_id"] = False
        return result

    duplicate_mask = top_auto_mask & result["candidate_player_id"].isin(duplicate_candidate_ids)

    result["duplicate_auto_approved_candidate_id"] = False
    result.loc[duplicate_mask, "duplicate_auto_approved_candidate_id"] = True
    result.loc[duplicate_mask, "match_status"] = "duplicate_auto_approved_manual_review"
    result.loc[duplicate_mask, "is_auto_approved"] = False
    result.loc[duplicate_mask, "needs_manual_review"] = True
    result.loc[duplicate_mask, "is_ambiguous"] = True

    result.loc[duplicate_mask, "match_reason"] = (
        result.loc[duplicate_mask, "match_reason"].astype(str)
        + "; duplicate_auto_approved_candidate_id"
    )

    return result


def build_report(
    candidates: pd.DataFrame,
    historical_players: pd.DataFrame,
    target_players: pd.DataFrame,
    team_mapping: pd.DataFrame,
    source_season: str,
    target_season: str,
    out_csv: str,
    prior_csv: str,
    auto_approve_threshold: float,
    ambiguous_gap: float,
) -> Dict[str, Any]:
    top_rows = candidates[(candidates["candidate_rank"].isna()) | (candidates["candidate_rank"] == 1)].copy()

    auto_count = int((top_rows["match_status"] == "auto_approved_candidate").sum())
    ambiguous_count = int((top_rows["match_status"] == "ambiguous_top_candidate").sum())
    low_conf_count = int((top_rows["match_status"] == "low_confidence_top_candidate").sum())
    high_score_manual_count = int((top_rows["match_status"] == "high_score_manual_review_candidate").sum())
    duplicate_auto_manual_count = int((top_rows["match_status"] == "duplicate_auto_approved_manual_review").sum())
    unmatched_count = int((top_rows["match_status"] == "unmatched").sum())
    manual_review_count = int(top_rows["needs_manual_review"].fillna(True).sum())

    mapped_team_count = 0
    if not team_mapping.empty and "matched_team_id" in team_mapping.columns:
        mapped_team_count = int(team_mapping["matched_team_id"].notna().sum())

    report: Dict[str, Any] = {
        "created_at": utc_now(),
        "source_season": source_season,
        "target_season": target_season,
        "out_csv": out_csv,
        "prior_csv": prior_csv or None,
        "passed": True,
        "audit_only": True,
        "writes_database": False,
        "thresholds": {
            "auto_approve_threshold": auto_approve_threshold,
            "ambiguous_gap": ambiguous_gap,
        },
        "row_counts": {
            "historical_players": int(len(historical_players)),
            "target_players": int(len(target_players)),
            "candidate_rows": int(len(candidates)),
            "top_candidate_or_unmatched_rows": int(len(top_rows)),
            "historical_team_rows": int(len(team_mapping)) if not team_mapping.empty else 0,
            "historical_teams_mapped_by_short_name": mapped_team_count,
        },
        "mapping_summary": {
            "auto_approved_count": auto_count,
            "ambiguous_count": ambiguous_count,
            "low_confidence_count": low_conf_count,
            "high_score_manual_review_count": high_score_manual_count,
            "duplicate_auto_approved_manual_review_count": duplicate_auto_manual_count,
            "unmatched_count": unmatched_count,
            "manual_review_count": manual_review_count,
            "auto_approved_rate": round(auto_count / float(len(historical_players)), 4) if len(historical_players) else None,
        },
        "status_counts": top_rows["match_status"].value_counts(dropna=False).to_dict(),
        "name_match_type_counts": candidates["name_match_type"].value_counts(dropna=False).to_dict(),
        "candidate_score_summary": {
            "min": float(candidates["candidate_score"].min()) if len(candidates) else None,
            "max": float(candidates["candidate_score"].max()) if len(candidates) else None,
            "mean": float(candidates["candidate_score"].mean()) if len(candidates) else None,
        },
        "top_auto_approved_examples": top_rows[top_rows["match_status"] == "auto_approved_candidate"].head(10).to_dict(orient="records"),
        "top_manual_review_examples": top_rows[top_rows["needs_manual_review"] == True].head(10).to_dict(orient="records"),
        "notes": [
            "This report is audit-only and does not update historical_players.",
            "Auto-approved candidates are still candidates; Day66C should consume only accepted mapping statuses.",
            "Team matching is based on historical/current short_name where available and is used as a confidence signal, not as a hard requirement.",
            "Transfers can be valid mappings even when team_match is false.",
            "Unmatched and ambiguous players require manual review before model integration.",
        ],
        "errors": [],
    }

    if len(top_rows) != len(historical_players):
        report["passed"] = False
        report["errors"].append(
            "Expected exactly one top/unmatched row per historical player, got %s vs %s."
            % (len(top_rows), len(historical_players))
        )

    return report


def write_json(report: Dict[str, Any], out_json: str) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("saved_report:", path)


def print_report(report: Dict[str, Any]) -> None:
    print("=== Historical Player Identity Mapping Candidates ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("saved_csv:", report["out_csv"])
    print()
    print("Row counts:")
    for key, value in report["row_counts"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Mapping summary:")
    for key, value in report["mapping_summary"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Status counts:")
    for key, value in report["status_counts"].items():
        print("- %s: %s" % (key, value))
    if report.get("errors"):
        print()
        print("Errors:")
        for error in report["errors"]:
            print("-", error)


def main() -> None:
    args = parse_args()

    historical_players = load_historical_players(args.source_season)
    target_players = load_target_players(args.target_season)
    team_mapping = load_historical_team_mapping(args.source_season)

    raw_to_canonical_team: Dict[str, Optional[int]] = {}
    if not team_mapping.empty:
        for _, row in team_mapping.iterrows():
            raw_to_canonical_team[normalize_raw_id(row.get("raw_team_id"))] = nullable_int(row.get("matched_team_id"))

    prior_context = load_prior_context(args.prior_csv)

    candidates = build_candidates(
        historical_players=historical_players,
        target_players=target_players,
        raw_to_canonical_team=raw_to_canonical_team,
        max_candidates_per_player=args.max_candidates_per_player,
        min_candidate_score=args.min_candidate_score,
        auto_approve_threshold=args.auto_approve_threshold,
        ambiguous_gap=args.ambiguous_gap,
        include_low_confidence=args.include_low_confidence,
    )

    candidates = demote_duplicate_auto_approved_candidates(candidates)
    candidates = attach_prior_context(candidates, prior_context)
    candidates.insert(1, "target_season", args.target_season)

    ordered_first = [
        "source_season", "target_season", "raw_player_id", "raw_player_name", "raw_team_id", "raw_position", "raw_position_name",
        "candidate_rank", "candidate_count", "candidate_player_id", "candidate_web_name", "candidate_full_name",
        "candidate_team_id", "candidate_position", "candidate_position_name", "candidate_status", "candidate_now_cost",
        "candidate_score", "name_score", "score_gap_to_next", "match_status", "is_auto_approved",
        "needs_manual_review", "is_ambiguous", "position_match", "team_match", "mapped_team_id_from_historical_team",
        "name_match_type", "best_historical_name_variant", "best_target_name_variant", "match_reason",
    ]
    other_cols = [col for col in candidates.columns if col not in ordered_first]
    candidates = candidates[[col for col in ordered_first if col in candidates.columns] + other_cols]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_csv, index=False)

    report = build_report(
        candidates=candidates,
        historical_players=historical_players,
        target_players=target_players,
        team_mapping=team_mapping,
        source_season=args.source_season,
        target_season=args.target_season,
        out_csv=str(out_csv),
        prior_csv=args.prior_csv,
        auto_approve_threshold=args.auto_approve_threshold,
        ambiguous_gap=args.ambiguous_gap,
    )

    write_json(report, args.out_json)
    print_report(report)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
