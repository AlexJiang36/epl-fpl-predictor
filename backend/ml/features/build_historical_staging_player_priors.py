from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal


REQUIRED_TABLES = ["historical_players", "historical_player_gw_stats"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build previous-season player prior artifacts from historical staging tables. "
            "This is read-only and does not modify the database."
        )
    )
    parser.add_argument("--source-season", required=True, help="Historical source season, for example 2024_25.")
    parser.add_argument("--target-season", required=True, help="Target season consuming the priors, for example 2025_26.")
    parser.add_argument("--out-csv", required=True, help="Output prior artifact CSV path.")
    parser.add_argument("--out-json", default=None, help="Optional metadata/report JSON path.")
    parser.add_argument(
        "--player-summary-csv",
        default=None,
        help="Optional Day64B player_season_summary.csv for validation.",
    )
    parser.add_argument(
        "--starts-minutes-threshold",
        type=int,
        default=60,
        help="Minutes threshold used for starts_proxy. Default: 60.",
    )
    parser.add_argument("--sample-limit", type=int, default=10)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def table_exists(table_name: str) -> bool:
    db = SessionLocal()
    try:
        value = db.execute(
            text(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = :table_name
                """
            ),
            {"table_name": table_name},
        ).scalar()
        return int(value or 0) > 0
    finally:
        db.close()


def read_sql_dataframe(sql: str, params: Dict[str, Any]) -> pd.DataFrame:
    db = SessionLocal()
    try:
        return pd.read_sql(text(sql), db.bind, params=params)
    finally:
        db.close()


def normalize_raw_id(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    if text_value.endswith(".0"):
        text_value = text_value[:-2]
    return text_value


def safe_divide(numerator: Any, denominator: Any) -> Optional[float]:
    if denominator is None or pd.isna(denominator) or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def require_tables() -> None:
    missing = [table_name for table_name in REQUIRED_TABLES if not table_exists(table_name)]
    if missing:
        raise RuntimeError("Missing required historical staging tables: %s" % missing)


def load_players(source_season: str) -> pd.DataFrame:
    players = read_sql_dataframe(
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
        """,
        {"source_season": source_season},
    )
    if players.empty:
        raise RuntimeError("No historical_players rows found for source_season=%s." % source_season)

    players["raw_player_id"] = players["raw_player_id"].apply(normalize_raw_id)
    players["raw_team_id"] = players["raw_team_id"].apply(normalize_raw_id)
    return players


def load_stats(source_season: str) -> pd.DataFrame:
    stats = read_sql_dataframe(
        """
        SELECT
            season,
            raw_player_id,
            gw,
            minutes,
            goals_scored,
            assists,
            clean_sheets,
            total_points,
            bonus,
            value
        FROM historical_player_gw_stats
        WHERE season = :source_season
        """,
        {"source_season": source_season},
    )
    if stats.empty:
        raise RuntimeError("No historical_player_gw_stats rows found for source_season=%s." % source_season)

    stats["raw_player_id"] = stats["raw_player_id"].apply(normalize_raw_id)
    for column in ["gw", "minutes", "goals_scored", "assists", "clean_sheets", "total_points", "bonus", "value"]:
        stats[column] = pd.to_numeric(stats[column], errors="coerce")
    return stats


def build_priors(source_season: str, target_season: str, starts_minutes_threshold: int) -> pd.DataFrame:
    require_tables()
    players = load_players(source_season)
    stats = load_stats(source_season)

    stats["appearance_flag"] = (stats["minutes"].fillna(0) > 0).astype(int)
    stats["starts_proxy_flag"] = (stats["minutes"].fillna(0) >= starts_minutes_threshold).astype(int)
    stats["negative_points_flag"] = (stats["total_points"].fillna(0) < 0).astype(int)
    stats["zero_minute_flag"] = (stats["minutes"].fillna(0) <= 0).astype(int)

    grouped = (
        stats.groupby(["season", "raw_player_id"], dropna=False)
        .agg(
            prev_season_games_in_dataset=("gw", "count"),
            prev_season_first_gw=("gw", "min"),
            prev_season_last_gw=("gw", "max"),
            prev_season_minutes=("minutes", "sum"),
            prev_season_appearances=("appearance_flag", "sum"),
            prev_season_starts_proxy=("starts_proxy_flag", "sum"),
            prev_season_total_points=("total_points", "sum"),
            prev_season_goals=("goals_scored", "sum"),
            prev_season_assists=("assists", "sum"),
            prev_season_clean_sheets=("clean_sheets", "sum"),
            prev_season_bonus=("bonus", "sum"),
            prev_season_latest_value=("value", "last"),
            prev_season_max_value=("value", "max"),
            prev_season_negative_points_gws=("negative_points_flag", "sum"),
            prev_season_zero_minute_rows=("zero_minute_flag", "sum"),
        )
        .reset_index()
    )

    priors = players.merge(grouped, on=["season", "raw_player_id"], how="left", validate="one_to_one")
    priors = priors.rename(columns={"season": "source_season"})
    priors.insert(1, "target_season", target_season)

    zero_fill_columns = [
        "prev_season_games_in_dataset",
        "prev_season_minutes",
        "prev_season_appearances",
        "prev_season_starts_proxy",
        "prev_season_total_points",
        "prev_season_goals",
        "prev_season_assists",
        "prev_season_clean_sheets",
        "prev_season_bonus",
        "prev_season_negative_points_gws",
        "prev_season_zero_minute_rows",
    ]
    for column in zero_fill_columns:
        priors[column] = priors[column].fillna(0).astype(int)

    priors["prev_season_points_per_appearance"] = priors.apply(
        lambda row: safe_divide(row["prev_season_total_points"], row["prev_season_appearances"]),
        axis=1,
    )
    priors["prev_season_points_per90"] = priors.apply(
        lambda row: safe_divide(row["prev_season_total_points"] * 90.0, row["prev_season_minutes"]),
        axis=1,
    )
    priors["prev_season_minutes_per_appearance"] = priors.apply(
        lambda row: safe_divide(row["prev_season_minutes"], row["prev_season_appearances"]),
        axis=1,
    )
    priors["prev_season_starts_proxy_rate"] = priors.apply(
        lambda row: safe_divide(row["prev_season_starts_proxy"], row["prev_season_appearances"]),
        axis=1,
    )

    for column in [
        "prev_season_points_per_appearance",
        "prev_season_points_per90",
        "prev_season_minutes_per_appearance",
        "prev_season_starts_proxy_rate",
    ]:
        priors[column] = priors[column].round(4)

    priors["has_prev_season_data"] = priors["prev_season_games_in_dataset"] > 0
    priors["is_prev_season_active"] = priors["prev_season_minutes"] > 0
    priors["prior_identity_scope"] = "historical_raw_player_id"
    priors["prior_source"] = "historical_staging"

    ordered_columns = [
        "source_season",
        "target_season",
        "raw_player_id",
        "raw_player_name",
        "raw_team_id",
        "raw_position",
        "canonical_player_id",
        "canonical_player_name",
        "mapping_status",
        "mapping_confidence",
        "prior_identity_scope",
        "prior_source",
        "has_prev_season_data",
        "is_prev_season_active",
        "prev_season_games_in_dataset",
        "prev_season_first_gw",
        "prev_season_last_gw",
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
        "notes",
    ]

    return priors[ordered_columns].sort_values(
        by=["prev_season_total_points", "prev_season_minutes", "raw_player_name"],
        ascending=[False, False, True],
    )


def validate_against_player_summary(priors: pd.DataFrame, player_summary_csv: Optional[str]) -> Dict[str, Any]:
    if not player_summary_csv:
        return {"enabled": False, "notes": ["No --player-summary-csv supplied; validation skipped."]}

    path = Path(player_summary_csv)
    result: Dict[str, Any] = {"enabled": True, "path": str(path), "exists": path.exists(), "errors": []}
    if not path.exists():
        result["errors"].append("player summary CSV does not exist.")
        return result

    summary = pd.read_csv(path)
    result.update(
        {
            "summary_row_count": int(len(summary)),
            "prior_row_count": int(len(priors)),
            "row_count_matches": int(len(summary)) == int(len(priors)),
            "summary_total_minutes": int(summary["minutes"].sum()) if "minutes" in summary.columns else None,
            "prior_total_minutes": int(priors["prev_season_minutes"].sum()),
            "summary_total_points": int(summary["total_points"].sum()) if "total_points" in summary.columns else None,
            "prior_total_points": int(priors["prev_season_total_points"].sum()),
            "top_player_summary": str(summary.iloc[0]["raw_player_name"]) if len(summary) else None,
            "top_player_prior": str(priors.iloc[0]["raw_player_name"]) if len(priors) else None,
        }
    )
    result["total_minutes_matches"] = result["summary_total_minutes"] == result["prior_total_minutes"]
    result["total_points_matches"] = result["summary_total_points"] == result["prior_total_points"]
    result["top_player_matches"] = result["top_player_summary"] == result["top_player_prior"]

    for key in ["row_count_matches", "total_minutes_matches", "total_points_matches", "top_player_matches"]:
        if result.get(key) is False:
            result["errors"].append("Validation failed: %s" % key)

    return result


def build_report(
    priors: pd.DataFrame,
    source_season: str,
    target_season: str,
    out_csv: str,
    starts_minutes_threshold: int,
    validation: Dict[str, Any],
    sample_limit: int,
) -> Dict[str, Any]:
    validation_errors = validation.get("errors", []) if validation else []
    report = {
        "created_at": utc_now(),
        "source_season": source_season,
        "target_season": target_season,
        "out_csv": out_csv,
        "passed": len(validation_errors) == 0,
        "errors": validation_errors,
        "row_count": int(len(priors)),
        "active_player_count": int((priors["prev_season_minutes"] > 0).sum()),
        "players_without_prev_season_data": int((priors["has_prev_season_data"] == False).sum()),  # noqa: E712
        "unmapped_player_count": int(priors["canonical_player_id"].isna().sum()),
        "starts_proxy_definition": "minutes >= %s" % starts_minutes_threshold,
        "total_prev_season_minutes": int(priors["prev_season_minutes"].sum()),
        "total_prev_season_points": int(priors["prev_season_total_points"].sum()),
        "total_prev_season_goals": int(priors["prev_season_goals"].sum()),
        "total_prev_season_assists": int(priors["prev_season_assists"].sum()),
        "negative_points_gw_count": int(priors["prev_season_negative_points_gws"].sum()),
        "player_summary_validation": validation,
        "top_priors_by_total_points": priors.head(sample_limit)[
            [
                "raw_player_id",
                "raw_player_name",
                "raw_team_id",
                "raw_position",
                "prev_season_minutes",
                "prev_season_total_points",
                "prev_season_points_per90",
            ]
        ].to_dict(orient="records"),
        "notes": [
            "This artifact is built from historical staging tables and is read-only.",
            "The primary identity is raw_player_id, not canonical player_id.",
            "Do not join this artifact directly to canonical players until identity mapping is resolved.",
            "This is suitable as a staging prior artifact for later model integration.",
        ],
    }
    return report


def write_json(report: Dict[str, Any], out_json: Optional[str]) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("saved_report:", path)


def print_report(report: Dict[str, Any]) -> None:
    print("=== Historical Staging Player Priors ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("passed:", report["passed"])
    print("row_count:", report["row_count"])
    print("active_player_count:", report["active_player_count"])
    print("players_without_prev_season_data:", report["players_without_prev_season_data"])
    print("unmapped_player_count:", report["unmapped_player_count"])
    print("starts_proxy_definition:", report["starts_proxy_definition"])
    print("total_prev_season_minutes:", report["total_prev_season_minutes"])
    print("total_prev_season_points:", report["total_prev_season_points"])
    print("negative_points_gw_count:", report["negative_points_gw_count"])
    print("saved_csv:", report["out_csv"])

    validation = report.get("player_summary_validation", {})
    if validation.get("enabled"):
        print()
        print("Player summary validation:")
        for key in [
            "exists",
            "row_count_matches",
            "total_minutes_matches",
            "total_points_matches",
            "top_player_matches",
        ]:
            if key in validation:
                print("- %s: %s" % (key, validation[key]))

    if report.get("errors"):
        print()
        print("Errors:")
        for error in report["errors"]:
            print("-", error)


def main() -> None:
    args = parse_args()

    priors = build_priors(
        source_season=args.source_season,
        target_season=args.target_season,
        starts_minutes_threshold=args.starts_minutes_threshold,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    priors.to_csv(out_csv, index=False)

    validation = validate_against_player_summary(priors, args.player_summary_csv)
    report = build_report(
        priors=priors,
        source_season=args.source_season,
        target_season=args.target_season,
        out_csv=str(out_csv),
        starts_minutes_threshold=args.starts_minutes_threshold,
        validation=validation,
        sample_limit=args.sample_limit,
    )

    write_json(report, args.out_json)
    print_report(report)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
