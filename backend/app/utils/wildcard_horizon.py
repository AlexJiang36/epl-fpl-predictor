from __future__ import annotations

from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session
from typing import Dict, List, Optional

from app.models.prediction import Prediction
from app.models.player import Player
from app.models.team import Team
from app.models.player_gw_stat import PlayerGameweekStat


def load_horizon_prediction_rows(
    *,
    db: Session,
    start_gw: int,
    horizon: int,
    model_name: str,
):
    end_gw = start_gw + horizon - 1

    stmt = (
        select(Prediction, Player, Team)
        .join(Player, Player.id == Prediction.player_id)
        .join(Team, Team.id == Player.team_id)
        .where(
            Prediction.target_gw >= start_gw,
            Prediction.target_gw <= end_gw,
            Prediction.model_name == model_name,
            Player.status == "a",
        )
        .order_by(Prediction.target_gw.asc(), Player.id.asc())
    )

    return db.execute(stmt).all()

def load_recent_minutes_by_player(
    *,
    db: Session,
    player_ids: List[int],
    start_gw: int,
    window: int = 5,
) -> Dict[int, List[int]]:
    if not player_ids:
        return {}

    lower_gw = max(1, start_gw - window)

    stmt = (
        select(PlayerGameweekStat)
        .where(
            PlayerGameweekStat.player_id.in_(player_ids),
            PlayerGameweekStat.gw >= lower_gw,
            PlayerGameweekStat.gw < start_gw,
        )
        .order_by(PlayerGameweekStat.player_id.asc(), PlayerGameweekStat.gw.asc())
    )

    rows = db.execute(stmt).scalars().all()

    out: Dict[int, List[int]] = {}
    for r in rows:
        out.setdefault(r.player_id, []).append(int(r.minutes or 0))
    return out

def aggregate_player_horizon_features(
    rows,
    recent_minutes_map: Optional[Dict[int, List[int]]] = None,
) -> List[Dict]:
    by_player: Dict[int, Dict] = {}

    for pred, pl, tm in rows:
        if pl.id not in by_player:
            by_player[pl.id] = {
                "player_id": pl.id,
                "web_name": pl.web_name,
                "team_id": tm.id,
                "team_name": tm.name,
                "team_short_name": tm.short_name,
                "position": pl.position,
                "now_cost": int(pl.now_cost),
                "status": pl.status,
                "horizon_predicted_points": 0.0,
                "gw_predictions": [],
                "fixture_count": 0,
            }

        pts = float(pred.predicted_points or 0.0)
        by_player[pl.id]["horizon_predicted_points"] += pts
        by_player[pl.id]["gw_predictions"].append(
            {
                "target_gw": pred.target_gw,
                "predicted_points": pts,
            }
        )
        by_player[pl.id]["fixture_count"] += 1

    results = []
    for row in by_player.values():
        preds = [x["predicted_points"] for x in row["gw_predictions"]]
        avg_pts = sum(preds) / len(preds) if preds else 0.0

        if preds:
            spread = max(preds) - min(preds)
        else:
            spread = 0.0

        if spread <= 1.5:
            prediction_consistency = "high"
        elif spread <= 3.0:
            prediction_consistency = "medium"
        else:
            prediction_consistency = "low"

        minutes = []
        if recent_minutes_map is not None:
            minutes = recent_minutes_map.get(row["player_id"], [])

        if minutes:
            avg_minutes = sum(minutes) / len(minutes)
            mins_60_plus_count = sum(1 for m in minutes if m >= 60)

            if avg_minutes >= 75 and mins_60_plus_count >= max(3, len(minutes) - 1):
                minutes_stability = "high"
            elif avg_minutes >= 45:
                minutes_stability = "medium"
            else:
                minutes_stability = "low"
        else:
            avg_minutes = None
            mins_60_plus_count = 0
            minutes_stability = "unknown"

        row["avg_predicted_points_per_gw"] = round(avg_pts, 4)
        row["horizon_predicted_points"] = round(row["horizon_predicted_points"], 4)
        row["prediction_consistency"] = prediction_consistency
        row["minutes_stability"] = minutes_stability
        row["recent_avg_minutes"] = round(avg_minutes, 2) if avg_minutes is not None else None
        row["recent_mins_60_plus_count"] = mins_60_plus_count

        results.append(row)

    results = sorted(
        results,
        key=lambda r: (
            r["horizon_predicted_points"],
            r["avg_predicted_points_per_gw"],
            -r["now_cost"],
            -r["player_id"],
        ),
        reverse=True,
    )

    return results

def build_wildcard_horizon_snapshot(
    *,
    db: Session,
    start_gw: int,
    horizon: int,
    model_name: str,
) -> Dict:
    rows = load_horizon_prediction_rows(
        db=db,
        start_gw=start_gw,
        horizon=horizon,
        model_name=model_name,
    )

    player_ids = list({pl.id for _, pl, _ in rows})
    recent_minutes_map = load_recent_minutes_by_player(
        db=db,
        player_ids=player_ids,
        start_gw=start_gw,
        window=5,
    )

    player_features = aggregate_player_horizon_features(
        rows,
        recent_minutes_map=recent_minutes_map,
    )

    return {
        "metadata": {
            "start_gw": start_gw,
            "horizon": horizon,
            "end_gw": start_gw + horizon - 1,
            "model_name": model_name,
            "num_prediction_rows": len(rows),
            "num_players": len(player_features),
        },
        "assumptions": {
            "future_vs_historical_planning": "Uses currently stored future prediction rows from the selected model for the requested GW horizon.",
            "minutes_stability_source": "Computed from historical player_gw_stats before start_gw.",
            "prediction_consistency_source": "Computed from spread of predicted points inside the selected future horizon.",
            "availability_rule": "Only players with status='a' are included in the horizon feature snapshot.",
        },
        "player_features": player_features,
    }