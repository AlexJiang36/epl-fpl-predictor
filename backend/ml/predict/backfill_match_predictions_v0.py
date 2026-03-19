# backend/ml/predict/backfill_match_predictions_v0.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression

BASELINE_MODEL = "match_baseline_v0"
LOGREG_MODEL = "match_logreg_v0"

def recent_points_avg(conn, team_id: int, before_ts, n_form: int) -> float:
    q = text("""
        SELECT home_team_id, away_team_id, home_score, away_score
        FROM fixtures
        WHERE finished = TRUE
          AND kickoff_time IS NOT NULL
          AND kickoff_time < :before_ts
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND (home_team_id = :team_id OR away_team_id = :team_id)
        ORDER BY kickoff_time DESC
        LIMIT :n_form
    """)
    rows = conn.execute(q, {"before_ts": before_ts, "team_id": team_id, "n_form": n_form}).fetchall()
    if not rows:
        return 1.0
    pts = 0
    for r in rows:
        hs = int(r.home_score); a = int(r.away_score)
        if r.home_team_id == team_id:
            gf, ga = hs, a
        else:
            gf, ga = a, hs
        pts += 3 if gf > ga else 1 if gf == ga else 0
    return pts / len(rows)

def softmax3(a, b, c):
    import math
    m = max(a,b,c)
    ea, eb, ec = math.exp(a-m), math.exp(b-m), math.exp(c-m)
    s = ea+eb+ec
    return ea/s, eb/s, ec/s

def baseline_predict(hf, af, threshold=0.30):
    diff = hf - af
    if diff > threshold:
        pred = "H"
    elif diff < -threshold:
        pred = "A"
    else:
        pred = "D"
    ph, pd_, pa = softmax3(diff, 0.0, -diff)
    return ph, pd_, pa, pred

def y_result(hs, a):
    if hs > a: return "H"
    if hs < a: return "A"
    return "D"

def delete_existing(conn, fixture_ids, model_name):
    conn.execute(
        text("DELETE FROM match_predictions WHERE model_name=:m AND fixture_id = ANY(:ids)"),
        {"m": model_name, "ids": fixture_ids},
    )

def insert_rows(conn, rows, model_name):
    conn.execute(
        text("""
            INSERT INTO match_predictions
              (fixture_id, model_name, pred_home_win, pred_draw, pred_away_win, pred_result)
            VALUES (:fixture_id, :model_name, :ph, :pd, :pa, :pr)
        """),
        [
            {
                "fixture_id": int(r["fixture_id"]),
                "model_name": model_name,
                "ph": float(r["ph"]),
                "pd": float(r["pd"]),
                "pa": float(r["pa"]),
                "pr": r["pr"],
            }
            for r in rows
        ],
    )

def main(start_gw: int, end_gw: int, n_form: int, do_baseline: bool, do_logreg: bool, C: float = 1.0):
    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)

    with engine.begin() as conn:
        fx = pd.read_sql(
            text("""
                SELECT id AS fixture_id, gw, kickoff_time, home_team_id, away_team_id, home_score, away_score
                FROM fixtures
                WHERE gw BETWEEN :start_gw AND :end_gw
                  AND finished = TRUE
                  AND kickoff_time IS NOT NULL
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                ORDER BY kickoff_time ASC
            """),
            conn,
            params={"start_gw": start_gw, "end_gw": end_gw},
        )
        if fx.empty:
            raise RuntimeError("No finished fixtures found in range.")

        # Build features for every fixture
        feats = []
        for _, r in fx.iterrows():
            before_ts = r["kickoff_time"]
            h_id = int(r["home_team_id"])
            a_id = int(r["away_team_id"])
            hf = recent_points_avg(conn, h_id, before_ts, n_form)
            af = recent_points_avg(conn, a_id, before_ts, n_form)
            feats.append((hf, af))
        fx["home_form_points_avg"] = [x[0] for x in feats]
        fx["away_form_points_avg"] = [x[1] for x in feats]
        fx["home_adv"] = 1.0
        fx["y_true"] = fx.apply(lambda r: y_result(int(r["home_score"]), int(r["away_score"])), axis=1)

        fixture_ids = fx["fixture_id"].astype(int).tolist()

        # Baseline backfill
        if do_baseline:
            rows = []
            for _, r in fx.iterrows():
                ph, pd_, pa, pr = baseline_predict(float(r["home_form_points_avg"]), float(r["away_form_points_avg"]))
                rows.append({"fixture_id": int(r["fixture_id"]), "ph": ph, "pd": pd_, "pa": pa, "pr": pr})
            delete_existing(conn, fixture_ids, BASELINE_MODEL)
            insert_rows(conn, rows, BASELINE_MODEL)
            print(f"OK: backfilled {len(rows)} rows for {BASELINE_MODEL} gw=[{start_gw},{end_gw}]")

        # Logreg backfill
        if do_logreg:
            X = fx[["home_form_points_avg", "away_form_points_avg", "home_adv"]].to_numpy()
            y = fx["y_true"].to_numpy()

            clf = LogisticRegression(
                multi_class="multinomial", solver="lbfgs", C=C, max_iter=1000, random_state=0
            )
            clf.fit(X, y)
            proba = clf.predict_proba(X)
            classes = list(clf.classes_)

            def p(label, rowp):
                return float(rowp[classes.index(label)]) if label in classes else 0.0

            rows = []
            for idx, rowp in enumerate(proba):
                fid = int(fx.iloc[idx]["fixture_id"])
                ph = p("H", rowp); pd_ = p("D", rowp); pa = p("A", rowp)
                pr = "H" if ph >= pd_ and ph >= pa else ("D" if pd_ >= pa else "A")
                rows.append({"fixture_id": fid, "ph": ph, "pd": pd_, "pa": pa, "pr": pr})

            delete_existing(conn, fixture_ids, LOGREG_MODEL)
            insert_rows(conn, rows, LOGREG_MODEL)
            print(f"OK: backfilled {len(rows)} rows for {LOGREG_MODEL} gw=[{start_gw},{end_gw}]")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_gw", type=int, required=True)
    ap.add_argument("--end_gw", type=int, required=True)
    ap.add_argument("--n_form", type=int, default=5)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--logreg", action="store_true")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    if not args.baseline and not args.logreg:
        raise SystemExit("Pass at least one of --baseline or --logreg")

    main(args.start_gw, args.end_gw, args.n_form, args.baseline, args.logreg, args.C)
