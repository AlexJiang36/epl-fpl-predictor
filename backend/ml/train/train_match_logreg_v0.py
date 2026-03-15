# backend/ml/train/train_match_logreg_v0.py
import os
import math
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL_NAME = "match_logreg_v0"
LABELS = ["H", "D", "A"]  # home win, draw, away win


def compute_team_recent_points_avg(conn, team_id: int, before_ts, n_form: int) -> float:
    q = text(
        """
        SELECT
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


def load_finished_fixtures(conn, start_gw: int, end_gw: int) -> pd.DataFrame:
    df = pd.read_sql(
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
    return df


def load_target_fixtures(conn, target_gw: int) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            """
            SELECT
              id AS fixture_id,
              gw,
              kickoff_time,
              home_team_id,
              away_team_id,
              finished,
              home_score,
              away_score
            FROM fixtures
            WHERE gw = :target_gw
              AND kickoff_time IS NOT NULL
            ORDER BY kickoff_time ASC
            """
        ),
        conn,
        params={"target_gw": target_gw},
    )
    return df


def add_features(conn, fixtures: pd.DataFrame, n_form: int) -> pd.DataFrame:
    home_form = []
    away_form = []

    for _, r in fixtures.iterrows():
        before_ts = r["kickoff_time"]
        h_id = int(r["home_team_id"])
        a_id = int(r["away_team_id"])
        home_form.append(compute_team_recent_points_avg(conn, h_id, before_ts, n_form))
        away_form.append(compute_team_recent_points_avg(conn, a_id, before_ts, n_form))

    out = fixtures.copy()
    out["home_form_points_avg"] = home_form
    out["away_form_points_avg"] = away_form
    out["home_adv"] = 1.0
    return out


def make_label(row) -> str:
    hs = int(row["home_score"])
    a_s = int(row["away_score"])
    if hs > a_s:
        return "H"
    if hs < a_s:
        return "A"
    return "D"


def write_match_predictions(conn, pred_df: pd.DataFrame, model_name: str):
    # idempotent per (fixture_id, model_name)
    conn.execute(
        text("DELETE FROM match_predictions WHERE model_name=:m AND fixture_id = ANY(:ids)"),
        {"m": model_name, "ids": list(pred_df["fixture_id"].astype(int).tolist())},
    )

    # insert
    conn.execute(
        text(
            """
            INSERT INTO match_predictions
              (fixture_id, model_name, pred_home_win, pred_draw, pred_away_win, pred_result)
            VALUES
              (:fixture_id, :model_name, :ph, :pd, :pa, :pr)
            """
        ),
        [
            {
                "fixture_id": int(r.fixture_id),
                "model_name": model_name,
                "ph": float(r.pred_home_win),
                "pd": float(r.pred_draw),
                "pa": float(r.pred_away_win),
                "pr": str(r.pred_result),
            }
            for r in pred_df.itertuples(index=False)
        ],
    )


def main(train_start_gw: int, train_end_gw: int, target_gw: int, n_form: int, C: float):
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError('DATABASE_URL not set. Example: export DATABASE_URL="postgresql://app:app@localhost:5432/epl"')

    engine = create_engine(db_url)

    with engine.begin() as conn:
        # 1) training data
        train_fx = load_finished_fixtures(conn, train_start_gw, train_end_gw)
        if train_fx.empty:
            raise RuntimeError(f"No finished fixtures in gw range [{train_start_gw}, {train_end_gw}]")

        train_fx = add_features(conn, train_fx, n_form=n_form)
        train_fx["y_result"] = train_fx.apply(make_label, axis=1)

        X = train_fx[["home_form_points_avg", "away_form_points_avg", "home_adv"]].to_numpy()
        y = train_fx["y_result"].to_numpy()

        # 2) train multinomial logreg
        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=C,
            max_iter=1000,
            random_state=0,
        )
        clf.fit(X, y)

        # quick in-sample sanity (not a real test split yet)
        yhat = clf.predict(X)
        acc = accuracy_score(y, yhat)
        print(f"train: rows={len(train_fx)} acc={acc:.3f} (sanity) labels={pd.Series(y).value_counts().to_dict()}")

        # 3) target fixtures (future GW)
        tgt_fx = load_target_fixtures(conn, target_gw)
        if tgt_fx.empty:
            raise RuntimeError(f"No fixtures found for target_gw={target_gw}")
        tgt_fx = add_features(conn, tgt_fx, n_form=n_form)

        Xt = tgt_fx[["home_form_points_avg", "away_form_points_avg", "home_adv"]].to_numpy()
        proba = clf.predict_proba(Xt)
        classes = list(clf.classes_)  # should be some ordering of H/D/A

        # map probabilities to H/D/A columns
        def p_for(label: str, rowp):
            if label in classes:
                return float(rowp[classes.index(label)])
            return 0.0

        preds = []
        for i, r in tgt_fx.iterrows():
            rowp = proba[list(range(len(tgt_fx)))][i - tgt_fx.index[0]] if False else None

        # safer loop with enumerate
        rows = []
        for idx, rowp in enumerate(proba):
            fx = tgt_fx.iloc[idx]
            ph = p_for("H", rowp)
            pd_ = p_for("D", rowp)
            pa = p_for("A", rowp)
            # choose argmax
            pred_result = "H" if ph >= pd_ and ph >= pa else ("D" if pd_ >= pa else "A")
            rows.append(
                {
                    "fixture_id": int(fx["fixture_id"]),
                    "pred_home_win": ph,
                    "pred_draw": pd_,
                    "pred_away_win": pa,
                    "pred_result": pred_result,
                }
            )

        pred_df = pd.DataFrame(rows)
        pred_df["model_name"] = MODEL_NAME

        # 4) write to DB
        write_match_predictions(conn, pred_df, MODEL_NAME)

        print(f"OK: wrote {len(pred_df)} rows for gw={target_gw} model={MODEL_NAME}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start_gw", type=int, required=True)
    ap.add_argument("--train_end_gw", type=int, required=True)
    ap.add_argument("--target_gw", type=int, required=True)
    ap.add_argument("--n_form", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    main(args.train_start_gw, args.train_end_gw, args.target_gw, args.n_form, args.C)
