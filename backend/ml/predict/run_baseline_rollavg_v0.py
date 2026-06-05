import json
import os

import pandas as pd
from sqlalchemy import create_engine, text

from app.core.season import get_current_season


MODEL_NAME = "baseline_rollavg_v0"


def load_raw_df(conn, season: str, gw_max: int) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            """
            SELECT
              s.player_id,
              s.gw,
              s.minutes,
              s.goals_scored,
              s.assists,
              s.clean_sheets,
              s.total_points,
              p.position,
              p.now_cost,
              p.status,
              p.team_id
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            WHERE s.season = :season
              AND s.gw <= :gw_max
            ORDER BY s.player_id, s.gw
            """
        ),
        conn,
        params={"season": season, "gw_max": gw_max},
    )
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("player_id", group_keys=False)

    df["pts_last1"] = g["total_points"].shift(1)
    df["mins_last1"] = g["minutes"].shift(1)
    df["pts_roll5_mean"] = g["total_points"].shift(1).rolling(5, min_periods=1).mean()

    df = df.dropna(subset=["pts_last1", "mins_last1"]).reset_index(drop=True)
    return df


def write_predictions(conn, season: str, out_df: pd.DataFrame, target_gw: int) -> tuple[int, int]:
    existing_ids = {
        row[0]
        for row in conn.execute(
            text(
                """
                SELECT player_id
                FROM predictions
                WHERE season = :season
                  AND target_gw = :target_gw
                  AND model_name = :model_name
                """
            ),
            {"season": season, "target_gw": target_gw, "model_name": MODEL_NAME},
        ).fetchall()
    }

    conn.execute(
        text(
            """
            DELETE FROM predictions
            WHERE season = :season
              AND target_gw = :target_gw
              AND model_name = :model_name
            """
        ),
        {"season": season, "target_gw": target_gw, "model_name": MODEL_NAME},
    )

    payload = [
        {
            "season": season,
            "player_id": int(r.player_id),
            "target_gw": int(target_gw),
            "model_name": MODEL_NAME,
            "predicted_points": float(r.predicted_points),
        }
        for r in out_df.itertuples(index=False)
    ]

    if payload:
        conn.execute(
            text(
                """
                INSERT INTO predictions
                  (season, player_id, target_gw, model_name, predicted_points)
                VALUES
                  (:season, :player_id, :target_gw, :model_name, :predicted_points)
                """
            ),
            payload,
        )

    inserted = sum(1 for r in out_df.itertuples(index=False) if int(r.player_id) not in existing_ids)
    updated = len(out_df) - inserted
    return inserted, updated


def main(target_gw: int, window: int = 5) -> None:
    db_url = os.environ["DATABASE_URL"]
    season = get_current_season()

    engine = create_engine(db_url)

    with engine.begin() as conn:
        raw = load_raw_df(conn, season=season, gw_max=target_gw - 1)
        if raw.empty:
            raise RuntimeError(f"No rows found for season={season} up to gw={target_gw - 1}")

        feature_df = make_features(raw)
        if feature_df.empty:
            raise RuntimeError(
                f"No rows left after feature construction for season={season}, target_gw={target_gw}"
            )

        last_finished_gws = sorted(feature_df["gw"].dropna().astype(int).unique().tolist())
        used_finished_gws = last_finished_gws[-window:] if len(last_finished_gws) >= window else last_finished_gws

        filtered = feature_df[feature_df["gw"].isin(used_finished_gws)].copy()
        pred_df = (
            filtered.groupby("player_id", as_index=False)["total_points"]
            .mean()
            .rename(columns={"total_points": "predicted_points"})
        )

        inserted, updated = write_predictions(conn, season=season, out_df=pred_df, target_gw=target_gw)

    print(
        json.dumps(
            {
                "season": season,
                "target_gw": target_gw,
                "window": window,
                "used_finished_gws": used_finished_gws,
                "model_name": MODEL_NAME,
                "inserted": inserted,
                "updated": updated,
                "total_players_predicted": int(len(pred_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gw", type=int, required=True)
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()

    main(target_gw=args.target_gw, window=args.window)
