from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import text

from app.core.db import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a historical player season summary from historical staging tables. "
            "This is read-only and does not modify the database."
        )
    )
    parser.add_argument("--season", required=True, help="Historical season key, for example 2024_25.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--out-json", default=None, help="Optional metadata/report JSON path.")
    parser.add_argument(
        "--starts-minutes-threshold",
        type=int,
        default=60,
        help="Minutes threshold used for starts_proxy. Default: 60.",
    )
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


def safe_divide(numerator: Any, denominator: Any) -> Optional[float]:
    if denominator is None or pd.isna(denominator) or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def build_player_summary(season: str, starts_minutes_threshold: int) -> pd.DataFrame:
    required_tables = ["historical_players", "historical_player_gw_stats"]
    missing_tables = [table_name for table_name in required_tables if not table_exists(table_name)]
    if missing_tables:
        raise RuntimeError("Missing required historical staging tables: %s" % missing_tables)

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
        WHERE season = :season
        """,
        {"season": season},
    )

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
        WHERE season = :season
        """,
        {"season": season},
    )

    if players.empty:
        raise RuntimeError("No historical_players rows found for season=%s." % season)
    if stats.empty:
        raise RuntimeError("No historical_player_gw_stats rows found for season=%s." % season)

    numeric_columns = [
        "gw",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "total_points",
        "bonus",
        "value",
    ]
    for column in numeric_columns:
        if column in stats.columns:
            stats[column] = pd.to_numeric(stats[column], errors="coerce")

    stats["appearance_flag"] = (stats["minutes"].fillna(0) > 0).astype(int)
    stats["starts_proxy_flag"] = (stats["minutes"].fillna(0) >= starts_minutes_threshold).astype(int)

    grouped = (
        stats.groupby(["season", "raw_player_id"], dropna=False)
        .agg(
            games_in_dataset=("gw", "count"),
            first_gw=("gw", "min"),
            last_gw=("gw", "max"),
            minutes=("minutes", "sum"),
            appearances=("appearance_flag", "sum"),
            starts_proxy=("starts_proxy_flag", "sum"),
            total_points=("total_points", "sum"),
            goals=("goals_scored", "sum"),
            assists=("assists", "sum"),
            clean_sheets=("clean_sheets", "sum"),
            bonus=("bonus", "sum"),
            latest_value=("value", "last"),
            max_value=("value", "max"),
        )
        .reset_index()
    )

    summary = players.merge(grouped, on=["season", "raw_player_id"], how="left", validate="one_to_one")

    fill_zero_columns = [
        "games_in_dataset",
        "minutes",
        "appearances",
        "starts_proxy",
        "total_points",
        "goals",
        "assists",
        "clean_sheets",
        "bonus",
    ]
    for column in fill_zero_columns:
        summary[column] = summary[column].fillna(0)

    int_columns = [
        "games_in_dataset",
        "minutes",
        "appearances",
        "starts_proxy",
        "total_points",
        "goals",
        "assists",
        "clean_sheets",
        "bonus",
    ]
    for column in int_columns:
        summary[column] = summary[column].astype(int)

    summary["points_per_appearance"] = summary.apply(
        lambda row: safe_divide(row["total_points"], row["appearances"]), axis=1
    )
    summary["points_per90"] = summary.apply(
        lambda row: safe_divide(row["total_points"] * 90.0, row["minutes"]), axis=1
    )
    summary["minutes_per_appearance"] = summary.apply(
        lambda row: safe_divide(row["minutes"], row["appearances"]), axis=1
    )

    for column in ["points_per_appearance", "points_per90", "minutes_per_appearance"]:
        summary[column] = summary[column].round(4)

    ordered_columns = [
        "season",
        "raw_player_id",
        "raw_player_name",
        "raw_team_id",
        "raw_position",
        "canonical_player_id",
        "canonical_player_name",
        "mapping_status",
        "mapping_confidence",
        "games_in_dataset",
        "first_gw",
        "last_gw",
        "minutes",
        "appearances",
        "starts_proxy",
        "total_points",
        "points_per_appearance",
        "points_per90",
        "minutes_per_appearance",
        "goals",
        "assists",
        "clean_sheets",
        "bonus",
        "latest_value",
        "max_value",
        "notes",
    ]
    existing_ordered_columns = [column for column in ordered_columns if column in summary.columns]
    summary = summary[existing_ordered_columns].sort_values(
        by=["total_points", "minutes", "raw_player_name"], ascending=[False, False, True]
    )

    return summary


def build_report(summary: pd.DataFrame, season: str, starts_minutes_threshold: int, out_csv: str) -> Dict[str, Any]:
    unmapped_count = int(summary["canonical_player_id"].isna().sum()) if "canonical_player_id" in summary.columns else None
    active_player_count = int((summary["minutes"] > 0).sum()) if "minutes" in summary.columns else None

    report = {
        "created_at": utc_now(),
        "season": season,
        "out_csv": out_csv,
        "row_count": int(len(summary)),
        "active_player_count": active_player_count,
        "unmapped_player_count": unmapped_count,
        "starts_proxy_definition": "minutes >= %s" % starts_minutes_threshold,
        "total_minutes": int(summary["minutes"].sum()) if "minutes" in summary.columns else None,
        "total_points": int(summary["total_points"].sum()) if "total_points" in summary.columns else None,
        "top_players_by_total_points": summary.head(10)[
            ["raw_player_id", "raw_player_name", "raw_team_id", "raw_position", "minutes", "total_points", "points_per90"]
        ].to_dict(orient="records"),
        "notes": [
            "This exporter is read-only.",
            "Player identity remains historical/raw unless canonical_player_id has been mapped.",
            "starts_proxy is not official starts; it is derived from minutes threshold.",
        ],
    }
    return report


def save_json(report: Dict[str, Any], out_json: Optional[str]) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("saved_report:", path)


def main() -> None:
    args = parse_args()
    summary = build_player_summary(args.season, args.starts_minutes_threshold)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    report = build_report(summary, args.season, args.starts_minutes_threshold, str(out_csv))
    save_json(report, args.out_json)

    print("=== Historical Player Season Summary ===")
    print("season:", args.season)
    print("row_count:", report["row_count"])
    print("active_player_count:", report["active_player_count"])
    print("unmapped_player_count:", report["unmapped_player_count"])
    print("total_minutes:", report["total_minutes"])
    print("total_points:", report["total_points"])
    print("starts_proxy_definition:", report["starts_proxy_definition"])
    print("saved_csv:", out_csv)


if __name__ == "__main__":
    main()
