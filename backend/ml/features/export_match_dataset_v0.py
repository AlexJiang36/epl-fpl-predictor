# backend/ml/features/export_match_dataset_v0.py
import os
import pandas as pd
from sqlalchemy import create_engine, text


def export_match_dataset_v0(start_gw: int, end_gw: int, n_form: int, out_csv: str) -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError('DATABASE_URL is not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"')

    engine = create_engine(db_url)

    # Pull finished fixtures in GW range with scores
    with engine.begin() as conn:
        fixtures = pd.read_sql(
            text(
                """
                SELECT
                  id AS fixture_id,
                  gw,
                  kickoff_time,
                  home_team_id,
                  away_team_id,
                  home_score,
                  away_score
                FROM fixtures
                WHERE gw BETWEEN :start_gw AND :end_gw
                  AND kickoff_time IS NOT NULL
                  AND finished = TRUE
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                ORDER BY kickoff_time ASC
                """
            ),
            conn,
            params={"start_gw": start_gw, "end_gw": end_gw},
        )

        if fixtures.empty:
            raise RuntimeError(f"No finished fixtures found in gw range [{start_gw}, {end_gw}]")

        # Build team form features using only matches BEFORE each fixture kickoff_time.
        # We compute recent points average for each (team_id, kickoff_time) pair.
        # Approach: for each fixture, query last N matches for each team (can be optimized later).
        # For Day35, correctness > speed. Dataset size is small (~10 fixtures per GW).

        def team_recent_points_avg(team_id: int, before_ts) -> float:
            q = text(
                """
                SELECT
                  f.id,
                  f.home_team_id,
                  f.away_team_id,
                  f.home_score,
                  f.away_score
                FROM fixtures f
                WHERE f.finished = TRUE
                  AND f.kickoff_time IS NOT NULL
                  AND f.kickoff_time < :before_ts
                  AND f.home_score IS NOT NULL
                  AND f.away_score IS NOT NULL
                  AND (f.home_team_id = :team_id OR f.away_team_id = :team_id)
                ORDER BY f.kickoff_time DESC
                LIMIT :n_form
                """
            )
            rows = conn.execute(q, {"before_ts": before_ts, "team_id": team_id, "n_form": n_form}).fetchall()
            if not rows:
                return 1.0  # neutral prior

            pts = 0
            for r in rows:
                hs = int(r.home_score)
                a_s = int(r.away_score)
                if r.home_team_id == team_id:
                    gf, ga = hs, a_s
                else:
                    gf, ga = a_s, hs

                if gf > ga:
                    pts += 3
                elif gf == ga:
                    pts += 1
                else:
                    pts += 0

            return pts / len(rows)

        home_form = []
        away_form = []
        y_result = []

        for _, row in fixtures.iterrows():
            before_ts = row["kickoff_time"]
            h_id = int(row["home_team_id"])
            a_id = int(row["away_team_id"])

            hf = team_recent_points_avg(h_id, before_ts)
            af = team_recent_points_avg(a_id, before_ts)

            home_form.append(hf)
            away_form.append(af)

            # label: H/D/A from actual score
            hs = int(row["home_score"])
            a_s = int(row["away_score"])
            if hs > a_s:
                y = "H"
            elif hs < a_s:
                y = "A"
            else:
                y = "D"
            y_result.append(y)

    fixtures["home_form_points_avg"] = home_form
    fixtures["away_form_points_avg"] = away_form
    fixtures["home_adv"] = 1.0
    fixtures["y_result"] = y_result

    # keep a clean column order
    cols = [
        "fixture_id", "gw", "kickoff_time",
        "home_team_id", "away_team_id",
        "home_score", "away_score",
        "home_form_points_avg", "away_form_points_avg",
        "home_adv",
        "y_result",
    ]
    out = fixtures[cols].copy()
    out.to_csv(out_csv, index=False)
    print(f"OK: wrote {len(out)} rows -> {out_csv}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_gw", type=int, required=True)
    ap.add_argument("--end_gw", type=int, required=True)
    ap.add_argument("--n_form", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    export_match_dataset_v0(args.start_gw, args.end_gw, args.n_form, args.out)


if __name__ == "__main__":
    main()
