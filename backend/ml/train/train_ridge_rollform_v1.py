# backend/ml/train/train_ridge_rollform_v1.py
import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from app.core.season import get_current_season

MODEL_NAME = "ridge_rollform_v1"


def parse_seasons(value: Optional[str], default_season: str) -> List[str]:
    if value is None or value.strip() == "":
        return [default_season]

    seasons = []
    for item in value.split(","):
        season = item.strip()
        if season and season not in seasons:
            seasons.append(season)

    if not seasons:
        raise RuntimeError("No valid seasons were provided.")
    return seasons


def build_raw_df(engine, seasons: Sequence[str], gw_max: int) -> pd.DataFrame:
    if not seasons:
        raise RuntimeError("build_raw_df requires at least one season.")

    q = (
        text(
            """
            SELECT
              s.season,
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
            JOIN players p
              ON p.id = s.player_id
            WHERE s.season IN :seasons
              AND s.gw <= :gw_max
            ORDER BY s.season, s.player_id, s.gw
            """
        )
        .bindparams(bindparam("seasons", expanding=True))
    )

    with engine.begin() as conn:
        df = pd.read_sql(
            q,
            conn,
            params={"seasons": list(seasons), "gw_max": gw_max},
        )

    if df.empty:
        raise RuntimeError(
            f"No rows found for seasons={list(seasons)} up to gw_max={gw_max}."
        )
    return df


def _lag(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.shift(periods)


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).sum()


def _rolling_nonzero_rate(series: pd.Series, window: int) -> pd.Series:
    return (
        series.shift(1)
        .rolling(window, min_periods=1)
        .apply(lambda x: float((pd.Series(x) > 0).mean()))
    )


def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df = df.sort_values(["season", "player_id", "gw"]).reset_index(drop=True)

    group_cols = ["season", "player_id"]
    g = df.groupby(group_cols, group_keys=False)

    df["pts_last1"] = g["total_points"].transform(_lag)
    df["mins_last1"] = g["minutes"].transform(_lag)

    for w in (3, 5, 8):
        df[f"pts_roll{w}_mean"] = g["total_points"].transform(
            lambda s, window=w: _rolling_mean(s, window)
        )
    for w in (3, 5):
        df[f"mins_roll{w}_mean"] = g["minutes"].transform(
            lambda s, window=w: _rolling_mean(s, window)
        )

    df["mins_roll5_nonzero_rate"] = g["minutes"].transform(
        lambda s: _rolling_nonzero_rate(s, 5)
    )

    df["g_roll5_sum"] = g["goals_scored"].transform(lambda s: _rolling_sum(s, 5))
    df["a_roll5_sum"] = g["assists"].transform(lambda s: _rolling_sum(s, 5))
    df["cs_roll5_sum"] = g["clean_sheets"].transform(lambda s: _rolling_sum(s, 5))

    df["now_cost_m"] = df["now_cost"] / 10.0

    # Drop first record per player per season because lag features are unavailable.
    df = df.dropna(subset=["pts_last1", "mins_last1"]).reset_index(drop=True)

    # One-hot encode categoricals after combining train and target rows so feature columns align.
    df = pd.get_dummies(df, columns=["position", "status"], drop_first=False)

    drop_cols = {
        "season",
        "player_id",
        "gw",
        "team_id",
        "now_cost",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "total_points",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]

    return df, feature_cols


def write_predictions(
    engine,
    season: str,
    out_df: pd.DataFrame,
    target_gw: int,
    model_name: str = MODEL_NAME,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM predictions
                WHERE season = :season
                  AND target_gw = :target_gw
                  AND model_name = :model_name
                """
            ),
            {"season": season, "target_gw": target_gw, "model_name": model_name},
        )

        out_df[
            ["season", "player_id", "target_gw", "model_name", "predicted_points"]
        ].to_sql("predictions", conn, if_exists="append", index=False, method="multi")


def main(
    target_gw: int,
    split_gw: int,
    alpha: float = 1.0,
    target_season: Optional[str] = None,
    train_seasons_arg: Optional[str] = None,
) -> None:
    resolved_target_season = target_season or get_current_season()
    train_seasons = parse_seasons(train_seasons_arg, default_season=resolved_target_season)

    seasons_to_load = list(train_seasons)
    if resolved_target_season not in seasons_to_load:
        seasons_to_load.append(resolved_target_season)

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            'DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"'
        )
    engine = create_engine(db_url)

    raw = build_raw_df(engine, seasons=seasons_to_load, gw_max=target_gw)
    feat, feature_cols = make_features(raw)

    train = feat[
        (feat["season"].isin(train_seasons))
        & (feat["gw"] <= split_gw)
    ].copy()
    val = feat[
        (feat["season"] == resolved_target_season)
        & (feat["gw"] > split_gw)
        & (feat["gw"] < target_gw)
    ].copy()

    if train.empty:
        raise RuntimeError(
            f"Train set is empty for train_seasons={train_seasons}, split_gw={split_gw}."
        )

    X_train = train[feature_cols].to_numpy()
    y_train = train["total_points"].to_numpy()

    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X_train, y_train)

    if not val.empty:
        pred_val = model.predict(val[feature_cols].to_numpy())
        mae = mean_absolute_error(val["total_points"].to_numpy(), pred_val)
        print(
            f"val: season={resolved_target_season} model={MODEL_NAME} "
            f"gw=({split_gw + 1}..{target_gw - 1}) rows={len(val)} MAE={mae:.4f}"
        )
    else:
        print(
            f"val: empty for season={resolved_target_season} "
            f"gw=({split_gw + 1}..{target_gw - 1})"
        )

    target = feat[
        (feat["season"] == resolved_target_season)
        & (feat["gw"] == target_gw)
    ].copy()
    if target.empty:
        raise RuntimeError(
            f"No feature rows for season={resolved_target_season}, target_gw={target_gw}. "
            "This script requires target-season player_gw_stats rows for the target GW. "
            "For a true pre-GW1 forecast, use the future pre-GW1 prediction mode instead."
        )

    yhat = model.predict(target[feature_cols].to_numpy())

    out = pd.DataFrame(
        {
            "season": resolved_target_season,
            "player_id": target["player_id"].astype(int),
            "target_gw": int(target_gw),
            "model_name": MODEL_NAME,
            "predicted_points": yhat.astype(float),
        }
    )

    write_predictions(
        engine,
        season=resolved_target_season,
        out_df=out,
        target_gw=target_gw,
        model_name=MODEL_NAME,
    )

    print(
        f"OK: season={resolved_target_season} train_seasons={train_seasons} "
        f"wrote {len(out)} rows for gw={target_gw} model={MODEL_NAME}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_gw", "--target-gw", dest="target_gw", type=int, required=True)
    ap.add_argument(
        "--split_gw",
        "--split-gw",
        dest="split_gw",
        type=int,
        required=True,
        help="Train on rows with gw <= split_gw.",
    )
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument(
        "--target-season",
        type=str,
        default=None,
        help="Season to write predictions for. Defaults to app.core.season.get_current_season().",
    )
    ap.add_argument(
        "--train-seasons",
        type=str,
        default=None,
        help=(
            "Comma-separated seasons used for training, for example "
            "2024_25,2025_26. Defaults to target season."
        ),
    )
    args = ap.parse_args()

    main(
        target_gw=args.target_gw,
        split_gw=args.split_gw,
        alpha=args.alpha,
        target_season=args.target_season,
        train_seasons_arg=args.train_seasons,
    )
