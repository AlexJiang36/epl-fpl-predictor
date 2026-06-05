from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, desc
import math

from app.core.db import get_db
from app.core.season import get_current_season
from app.models.fixture import Fixture
from app.models.team import Team
from app.models.match_prediction import MatchPrediction
from app.utils.model_metadata_store import (
    list_model_metadata_artifacts,
    maybe_load_model_metadata_artifact,
)

router = APIRouter(prefix="/match/predictions", tags=["match"])

DEFAULT_MODEL_NAME = "match_baseline_v0"


def softmax3(a: float, b: float, c: float):
    m = max(a, b, c)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    ec = math.exp(c - m)
    s = ea + eb + ec
    return ea / s, eb / s, ec / s


def team_points_for_fixture(team_id: int, f: Fixture) -> int:
    hs = int(f.home_score)
    as_ = int(f.away_score)

    if f.home_team_id == team_id:
        gf, ga = hs, as_
    else:
        gf, ga = as_, hs

    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def recent_form_avg_points(db: Session, team_id: int, before_time, n: int, season: str) -> float:
    q = (
        db.query(Fixture)
        .filter(Fixture.season == season)
        .filter(Fixture.finished.is_(True))
        .filter(Fixture.home_score.isnot(None))
        .filter(Fixture.away_score.isnot(None))
        .filter(Fixture.kickoff_time.isnot(None))
        .filter(Fixture.kickoff_time < before_time)
        .filter(or_(Fixture.home_team_id == team_id, Fixture.away_team_id == team_id))
        .order_by(desc(Fixture.kickoff_time))
        .limit(n)
    )
    rows = q.all()
    if not rows:
        return 1.0

    pts = [team_points_for_fixture(team_id, f) for f in rows]
    return sum(pts) / len(pts)


@router.get("")
def get_match_prediction(
    fixture_id: int = Query(..., ge=1),
    model_name: str = Query(DEFAULT_MODEL_NAME),
    db: Session = Depends(get_db),
):
    season = get_current_season()

    row = (
        db.query(MatchPrediction)
        .filter(MatchPrediction.season == season)
        .filter(MatchPrediction.fixture_id == fixture_id)
        .filter(MatchPrediction.model_name == model_name)
        .first()
    )
    if not row:
        return {
            "found": False,
            "season": season,
            "fixture_id": fixture_id,
            "model_name": model_name,
        }

    return {
        "found": True,
        "season": season,
        "fixture_id": row.fixture_id,
        "model_name": row.model_name,
        "pred_home_win": row.pred_home_win,
        "pred_draw": row.pred_draw,
        "pred_away_win": row.pred_away_win,
        "pred_result": row.pred_result,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.post("/run")
def run_match_baseline_and_store(
    fixture_id: int = Query(..., ge=1),
    n: int = Query(5, ge=1, le=20),
    model_name: str = Query(DEFAULT_MODEL_NAME),
    threshold: float = Query(0.30, ge=0.0, le=2.0),
    db: Session = Depends(get_db),
):
    season = get_current_season()

    fixture = (
        db.query(Fixture)
        .filter(Fixture.season == season)
        .filter(Fixture.id == fixture_id)
        .first()
    )
    if not fixture:
        return {"ok": False, "error": f"fixture_id={fixture_id} not found for season={season}"}

    if fixture.kickoff_time is None:
        return {"ok": False, "error": "fixture has no kickoff_time"}

    home_id = fixture.home_team_id
    away_id = fixture.away_team_id

    team_rows = db.query(Team.id, Team.name).filter(Team.id.in_([home_id, away_id])).all()
    team_map = {tid: name for tid, name in team_rows}
    home_name = team_map.get(home_id, f"Unknown({home_id})")
    away_name = team_map.get(away_id, f"Unknown({away_id})")

    home_avg = recent_form_avg_points(db, home_id, fixture.kickoff_time, n, season)
    away_avg = recent_form_avg_points(db, away_id, fixture.kickoff_time, n, season)

    diff = home_avg - away_avg

    if diff > threshold:
        pred_result = "H"
    elif diff < -threshold:
        pred_result = "A"
    else:
        pred_result = "D"

    scale = 1.0
    pH, pD, pA = softmax3(scale * diff, 0.0, -scale * diff)

    db.query(MatchPrediction).filter(
        MatchPrediction.season == season,
        MatchPrediction.fixture_id == fixture_id,
        MatchPrediction.model_name == model_name,
    ).delete()

    row = MatchPrediction(
        season=season,
        fixture_id=fixture_id,
        model_name=model_name,
        pred_home_win=float(pH),
        pred_draw=float(pD),
        pred_away_win=float(pA),
        pred_result=pred_result,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "ok": True,
        "season": season,
        "fixture": {
            "id": fixture.id,
            "kickoff_time": fixture.kickoff_time.isoformat(),
            "home_team_id": home_id,
            "home_team_name": home_name,
            "away_team_id": away_id,
            "away_team_name": away_name,
        },
        "model": {
            "model_name": model_name,
            "n_recent_matches": n,
            "threshold": threshold,
        },
        "features": {
            "home_recent_points_avg": home_avg,
            "away_recent_points_avg": away_avg,
            "diff": diff,
        },
        "prediction": {
            "pred_home_win": float(pH),
            "pred_draw": float(pD),
            "pred_away_win": float(pA),
            "pred_result": pred_result,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        },
    }


@router.get("/list")
def list_match_predictions(
    gw: int = Query(..., ge=1),
    model_name: str = Query(DEFAULT_MODEL_NAME),
    db: Session = Depends(get_db),
):
    season = get_current_season()

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.season == season)
        .filter(Fixture.gw == gw)
        .filter(Fixture.kickoff_time.isnot(None))
        .order_by(Fixture.kickoff_time.asc())
        .all()
    )

    fixture_ids = [f.id for f in fixtures]
    if not fixture_ids:
        return {
            "meta": {
                "season": season,
                "gw": gw,
                "model_name": model_name,
                "count": 0,
            },
            "rows": [],
        }

    preds = (
        db.query(MatchPrediction)
        .filter(MatchPrediction.season == season)
        .filter(MatchPrediction.model_name == model_name)
        .filter(MatchPrediction.fixture_id.in_(fixture_ids))
        .all()
    )
    pred_map = {p.fixture_id: p for p in preds}

    team_ids = set()
    for f in fixtures:
        team_ids.add(f.home_team_id)
        team_ids.add(f.away_team_id)

    teams = db.query(Team).filter(Team.id.in_(list(team_ids))).all()
    team_name = {t.id: t.name for t in teams}

    rows = []
    for f in fixtures:
        p = pred_map.get(f.id)
        rows.append(
            {
                "fixture_id": f.id,
                "season": season,
                "gw": f.gw,
                "kickoff_time": f.kickoff_time.isoformat() if f.kickoff_time else None,
                "home_team_id": f.home_team_id,
                "home_team_name": team_name.get(f.home_team_id, f"Unknown({f.home_team_id})"),
                "away_team_id": f.away_team_id,
                "away_team_name": team_name.get(f.away_team_id, f"Unknown({f.away_team_id})"),
                "finished": bool(f.finished),
                "home_score": f.home_score,
                "away_score": f.away_score,
                "model_name": model_name,
                "found": p is not None,
                "pred_home_win": float(p.pred_home_win) if p else None,
                "pred_draw": float(p.pred_draw) if p else None,
                "pred_away_win": float(p.pred_away_win) if p else None,
                "pred_result": p.pred_result if p else None,
                "created_at": p.created_at.isoformat() if (p and p.created_at) else None,
            }
        )

    return {
        "meta": {
            "season": season,
            "gw": gw,
            "model_name": model_name,
            "count": len(rows),
        },
        "rows": rows,
    }


@router.get("/models")
def list_match_models(
    active_only: bool = Query(False),
    db: Session = Depends(get_db),
):
    season = get_current_season()

    rows = (
        db.query(MatchPrediction.model_name)
        .filter(MatchPrediction.season == season)
        .filter(MatchPrediction.model_name.isnot(None))
        .filter(MatchPrediction.model_name != "")
        .distinct()
        .order_by(MatchPrediction.model_name.asc())
        .all()
    )

    names_from_predictions = {r[0] for r in rows if r and r[0]}
    metadata_models = list_model_metadata_artifacts(task_type="match_result")
    names_from_metadata = {m.model_name for m in metadata_models}

    all_names = sorted(names_from_predictions | names_from_metadata)

    models = []
    for name in all_names:
        meta_artifact = maybe_load_model_metadata_artifact(name)
        source = "match_predictions_distinct" if name in names_from_predictions else "model_metadata"

        row = {
            "model_name": name,
            "label": name,
            "task_type": meta_artifact.task_type if meta_artifact else "match_result",
            "source": source,
            "status": meta_artifact.status if meta_artifact else "experimental",
            "is_active": meta_artifact.is_active if meta_artifact else False,
            "is_production_default": meta_artifact.is_production_default if meta_artifact else False,
            "selected_reason": meta_artifact.selected_reason if meta_artifact else None,
            "notes": meta_artifact.notes if meta_artifact else None,
            "metadata": {
                "season": season,
                "feature_version": meta_artifact.feature_version if meta_artifact else None,
                "training_window_start_gw": meta_artifact.training_window_start_gw if meta_artifact else None,
                "training_window_end_gw": meta_artifact.training_window_end_gw if meta_artifact else None,
                "evaluation_start_gw": meta_artifact.evaluation_start_gw if meta_artifact else None,
                "evaluation_end_gw": meta_artifact.evaluation_end_gw if meta_artifact else None,
                "metrics_summary": meta_artifact.metrics_summary if meta_artifact else {},
                "status": meta_artifact.status if meta_artifact else "experimental",
                "is_active": meta_artifact.is_active if meta_artifact else False,
                "is_production_default": meta_artifact.is_production_default if meta_artifact else False,
                "selected_reason": meta_artifact.selected_reason if meta_artifact else None,
                "notes": meta_artifact.notes if meta_artifact else None,
                "updated_at": meta_artifact.updated_at.isoformat() if meta_artifact else None,
            },
        }

        if active_only and not row["is_active"]:
            continue
        models.append(row)

    return {
        "models": models,
        "meta": {
            "season": season,
            "count": len(models),
            "source": "match_predictions_distinct_plus_model_metadata",
            "active_only": active_only,
        },
    }


@router.post("/run_gw")
def run_match_baseline_for_gw(
    gw: int = Query(..., ge=1),
    n: int = Query(5, ge=1, le=20),
    model_name: str = Query(DEFAULT_MODEL_NAME),
    threshold: float = Query(0.30, ge=0.0, le=2.0),
    db: Session = Depends(get_db),
):
    season = get_current_season()

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.season == season)
        .filter(Fixture.gw == gw)
        .filter(Fixture.kickoff_time.isnot(None))
        .order_by(Fixture.kickoff_time.asc())
        .all()
    )

    if not fixtures:
        return {"ok": False, "error": f"No fixtures found for season={season}, gw={gw} (kickoff_time is required)."}

    results = []
    success = 0
    failed = 0

    for f in fixtures:
        try:
            home_id = f.home_team_id
            away_id = f.away_team_id

            team_rows = db.query(Team.id, Team.name).filter(Team.id.in_([home_id, away_id])).all()
            team_map = {tid: name for tid, name in team_rows}
            home_name = team_map.get(home_id, f"Unknown({home_id})")
            away_name = team_map.get(away_id, f"Unknown({away_id})")

            home_avg = recent_form_avg_points(db, home_id, f.kickoff_time, n, season)
            away_avg = recent_form_avg_points(db, away_id, f.kickoff_time, n, season)
            diff = home_avg - away_avg

            if diff > threshold:
                pred_result = "H"
            elif diff < -threshold:
                pred_result = "A"
            else:
                pred_result = "D"

            scale = 1.0
            pH, pD, pA = softmax3(scale * diff, 0.0, -scale * diff)

            db.query(MatchPrediction).filter(
                MatchPrediction.season == season,
                MatchPrediction.fixture_id == f.id,
                MatchPrediction.model_name == model_name,
            ).delete()

            row = MatchPrediction(
                season=season,
                fixture_id=f.id,
                model_name=model_name,
                pred_home_win=float(pH),
                pred_draw=float(pD),
                pred_away_win=float(pA),
                pred_result=pred_result,
            )
            db.add(row)
            db.commit()
            db.refresh(row)

            success += 1
            results.append(
                {
                    "fixture_id": f.id,
                    "season": season,
                    "gw": gw,
                    "kickoff_time": f.kickoff_time.isoformat() if f.kickoff_time else None,
                    "home_team_id": home_id,
                    "home_team_name": home_name,
                    "away_team_id": away_id,
                    "away_team_name": away_name,
                    "pred_home_win": float(pH),
                    "pred_draw": float(pD),
                    "pred_away_win": float(pA),
                    "pred_result": pred_result,
                }
            )
        except Exception as e:
            db.rollback()
            failed += 1
            results.append(
                {
                    "fixture_id": getattr(f, "id", None),
                    "season": season,
                    "gw": gw,
                    "error": str(e),
                }
            )

    return {
        "ok": True,
        "meta": {
            "season": season,
            "gw": gw,
            "model_name": model_name,
            "n_form": n,
            "threshold": threshold,
            "fixtures_total": len(fixtures),
            "success": success,
            "failed": failed,
        },
        "results": results,
    }
