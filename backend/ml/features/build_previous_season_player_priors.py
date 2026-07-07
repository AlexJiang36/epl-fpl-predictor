from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import text

from app.core.db import SessionLocal
from app.core.season import get_current_season


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build previous-season player prior features. "
            "The script fails clearly if the requested source season is not loaded."
        )
    )
    parser.add_argument(
        "--source-season",
        type=str,
        required=True,
        help="Season used to compute priors, for example 2024_25.",
    )
    parser.add_argument(
        "--target-season",
        type=str,
        default=None,
        help="Season that will consume these priors. Defaults to current configured season.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "artifacts/offline_datasets/player_previous_season_priors_{source}_to_{target}.csv"
        ),
    )
    return parser.parse_args()


def _default_out_path(source_season: str, target_season: str) -> Path:
    return Path(
        "artifacts/offline_datasets/"
        f"player_previous_season_priors_{source_season}_to_{target_season}.csv"
    )


def _get_table_columns(table_name: str) -> List[str]:
    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = :table_name
                ORDER BY ordinal_position
                """
            ),
            {"table_name": table_name},
        ).mappings().all()
        return [str(row["column_name"]) for row in rows]
    finally:
        db.close()


def _count_source_rows(source_season: str) -> int:
    db = SessionLocal()
    try:
        value = db.execute(
            text(
                """
                SELECT COUNT(*) AS n
                FROM player_gw_stats
                WHERE season = :source_season
                """
            ),
            {"source_season": source_season},
        ).scalar()
        return int(value or 0)
    finally:
        db.close()


def _require_columns(table_name: str, existing: List[str], required: List[str]) -> None:
    missing = [col for col in required if col not in existing]
    if missing:
        raise RuntimeError(
            f"Cannot build previous-season priors: table={table_name} is missing "
            f"required columns {missing}. Existing columns: {existing}"
        )


def _optional_sum_expr(
    *,
    existing_columns: List[str],
    column_name: str,
    alias: str,
) -> str:
    if column_name in existing_columns:
        return f"SUM(COALESCE(s.{column_name}, 0)) AS {alias}"
    return f"NULL AS {alias}"


def _build_prior_rows(source_season: str, target_season: str) -> List[Dict[str, Any]]:
    stat_columns = _get_table_columns("player_gw_stats")
    player_columns = _get_table_columns("players")

    _require_columns(
        "player_gw_stats",
        stat_columns,
        ["season", "player_id", "minutes", "total_points"],
    )
    _require_columns(
        "players",
        player_columns,
        ["id", "fpl_player_id", "web_name", "team_id", "position"],
    )

    optional_goals = "goals_scored" in stat_columns
    optional_assists = "assists" in stat_columns

    goals_expr = _optional_sum_expr(
        existing_columns=stat_columns,
        column_name="goals_scored",
        alias="prev_season_goals",
    )
    assists_expr = _optional_sum_expr(
        existing_columns=stat_columns,
        column_name="assists",
        alias="prev_season_assists",
    )
    clean_sheets_expr = _optional_sum_expr(
        existing_columns=stat_columns,
        column_name="clean_sheets",
        alias="prev_season_clean_sheets",
    )
    bonus_expr = _optional_sum_expr(
        existing_columns=stat_columns,
        column_name="bonus",
        alias="prev_season_bonus",
    )

    if optional_goals and optional_assists:
        goal_involvement_expr = (
            "SUM(COALESCE(s.goals_scored, 0) + COALESCE(s.assists, 0)) "
            "AS prev_season_goal_involvement_proxy"
        )
    else:
        goal_involvement_expr = "NULL AS prev_season_goal_involvement_proxy"

    sql = f"""
        WITH player_prior AS (
            SELECT
                :source_season AS source_season,
                :target_season AS target_season,
                p.id AS player_id,
                p.fpl_player_id AS fpl_player_id,
                p.web_name AS web_name,
                p.team_id AS team_id,
                p.position AS position,
                SUM(COALESCE(s.minutes, 0)) AS prev_season_minutes,
                SUM(CASE WHEN COALESCE(s.minutes, 0) > 0 THEN 1 ELSE 0 END)
                    AS prev_season_appearances,
                SUM(CASE WHEN COALESCE(s.minutes, 0) >= 60 THEN 1 ELSE 0 END)
                    AS prev_season_starts_proxy,
                SUM(COALESCE(s.total_points, 0)) AS prev_season_total_points,
                {goals_expr},
                {assists_expr},
                {clean_sheets_expr},
                {bonus_expr},
                {goal_involvement_expr}
            FROM player_gw_stats s
            JOIN players p
              ON p.id = s.player_id
            WHERE s.season = :source_season
            GROUP BY
                p.id,
                p.fpl_player_id,
                p.web_name,
                p.team_id,
                p.position
        )
        SELECT
            source_season,
            target_season,
            player_id,
            fpl_player_id,
            web_name,
            team_id,
            position,
            prev_season_minutes,
            prev_season_appearances,
            prev_season_starts_proxy,
            prev_season_total_points,
            CASE
                WHEN prev_season_appearances > 0
                THEN prev_season_total_points::float / prev_season_appearances
                ELSE NULL
            END AS prev_season_points_per_appearance,
            CASE
                WHEN prev_season_minutes > 0
                THEN prev_season_total_points::float * 90.0 / prev_season_minutes
                ELSE NULL
            END AS prev_season_points_per90,
            CASE
                WHEN prev_season_appearances > 0
                THEN prev_season_minutes::float / prev_season_appearances
                ELSE NULL
            END AS prev_season_minutes_per_appearance,
            prev_season_goal_involvement_proxy,
            prev_season_goals,
            prev_season_assists,
            prev_season_clean_sheets,
            prev_season_bonus
        FROM player_prior
        ORDER BY
            prev_season_total_points DESC,
            prev_season_minutes DESC,
            player_id ASC
    """

    db = SessionLocal()
    try:
        rows = db.execute(
            text(sql),
            {
                "source_season": source_season,
                "target_season": target_season,
            },
        ).mappings().all()
        return [dict(row) for row in rows]
    finally:
        db.close()


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "source_season",
        "target_season",
        "player_id",
        "fpl_player_id",
        "web_name",
        "team_id",
        "position",
        "prev_season_minutes",
        "prev_season_appearances",
        "prev_season_starts_proxy",
        "prev_season_total_points",
        "prev_season_points_per_appearance",
        "prev_season_points_per90",
        "prev_season_minutes_per_appearance",
        "prev_season_goal_involvement_proxy",
        "prev_season_goals",
        "prev_season_assists",
        "prev_season_clean_sheets",
        "prev_season_bonus",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_metadata(
    *,
    csv_path: Path,
    source_season: str,
    target_season: str,
    row_count: int,
) -> Path:
    meta_path = csv_path.with_suffix(".metadata.json")
    payload = {
        "artifact_type": "player_previous_season_priors",
        "source_season": source_season,
        "target_season": target_season,
        "csv_path": str(csv_path),
        "row_count": row_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Previous-season player priors built from player_gw_stats. "
            "Use as supplementary early-season stabilizers, not replacements "
            "for current-season rolling form."
        ),
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path


def main() -> None:
    args = parse_args()
    source_season = args.source_season
    target_season = args.target_season or get_current_season()
    out_path = Path(args.out) if args.out else _default_out_path(source_season, target_season)

    source_row_count = _count_source_rows(source_season)
    if source_row_count == 0:
        raise SystemExit(
            "No player_gw_stats found for "
            f"source_season={source_season}. "
            "Cannot build previous-season priors. "
            "Load or import previous-season data before running this builder."
        )

    rows = _build_prior_rows(source_season=source_season, target_season=target_season)
    if not rows:
        raise SystemExit(
            "No prior rows were produced for "
            f"source_season={source_season}, target_season={target_season}. "
            "Check player_gw_stats and players joins."
        )

    _write_csv(out_path, rows)
    metadata_path = _write_metadata(
        csv_path=out_path,
        source_season=source_season,
        target_season=target_season,
        row_count=len(rows),
    )

    print("Previous-season player priors artifact written.")
    print(f"source_season: {source_season}")
    print(f"target_season: {target_season}")
    print(f"source_player_gw_stats_rows: {source_row_count}")
    print(f"prior_rows: {len(rows)}")
    print(f"csv_path: {out_path}")
    print(f"metadata_path: {metadata_path}")


if __name__ == "__main__":
    main()
