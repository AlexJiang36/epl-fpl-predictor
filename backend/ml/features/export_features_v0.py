# backend/ml/features/export_features_v0.py
import os
import pandas as pd
from sqlalchemy import create_engine, text


def export_features_v0(start_gw: int, end_gw: int, out_csv: str) -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            'DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"'
        )

    engine = create_engine(db_url)

    with engine.begin() as conn:
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
                WHERE s.gw BETWEEN :start_gw AND :end_gw
                ORDER BY s.player_id, s.gw
                """
            ),
            conn,
            params={"start_gw": start_gw, "end_gw": end_gw},
        )

    if df.empty:
        raise RuntimeError(f"No rows found in player_gw_stats for gw range [{start_gw}, {end_gw}]")

    # group by player to create leakage-safe lag/rolling features
    g = df.groupby("player_id", group_keys=False)

    # Lag (previous GW)
    df["pts_last1"] = g["total_points"].shift(1)
    df["mins_last1"] = g["minutes"].shift(1)

    # Rolling means (shifted by 1)
    for w in (3, 5, 8):
        df[f"pts_roll{w}_mean"] = g["total_points"].shift(1).rolling(w, min_periods=1).mean()
    for w in (3, 5):
        df[f"mins_roll{w}_mean"] = g["minutes"].shift(1).rolling(w, min_periods=1).mean()

    # Nonzero minutes rate (last 5, shifted)
    df["mins_roll5_nonzero_rate"] = (
        g["minutes"]
        .shift(1)
        .rolling(5, min_periods=1)
        .apply(lambda x: (pd.Series(x) > 0).mean())
    )

    # Event rolling sums (shifted)
    df["g_roll5_sum"] = g["goals_scored"].shift(1).rolling(5, min_periods=1).sum()
    df["a_roll5_sum"] = g["assists"].shift(1).rolling(5, min_periods=1).sum()
    df["cs_roll5_sum"] = g["clean_sheets"].shift(1).rolling(5, min_periods=1).sum()

    # Static transforms
    df["now_cost_m"] = df["now_cost"] / 10.0

    # Drop rows that cannot have lag features (first GW per player within range)
    df = df.dropna(subset=["pts_last1", "mins_last1"]).reset_index(drop=True)

    # Keep target as total_points (label for training)
    # Keep raw ids for joining back later
    # One-hot encode categoricals (ridge-friendly)
    df = pd.get_dummies(df, columns=["position", "status"], drop_first=False)

    df.to_csv(out_csv, index=False)
    print(f"OK: wrote {len(df)} rows -> {out_csv}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start_gw", type=int, required=True)
    ap.add_argument("--end_gw", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    export_features_v0(args.start_gw, args.end_gw, args.out)


if __name__ == "__main__":
    main()