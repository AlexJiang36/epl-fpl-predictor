from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Literal, Tuple

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.season import get_current_season
from app.models.prediction import Prediction
from app.models.player import Player
from app.models.team import Team
from app.models.gameweek import Gameweek
from app.models.player_gw_stat import PlayerGameweekStat

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

MODEL_NAME = "baseline_rollavg_v0"

Position = Literal["GKP", "DEF", "MID", "FWD"]
OrderBy = Literal["points", "value"]
ViewMode = Literal["compact", "full"]

SQUAD_RULES = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_FORMATION = {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3}
POSITION_CYCLE = ["FWD", "MID", "DEF", "GKP"]


@router.get("/ping")
def ping():
    return {"ok": True}


def _decide_target_gw(db: Session, target_gw: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
    season = get_current_season()

    if target_gw is not None:
        return target_gw, None

    nxt = db.execute(
        select(Gameweek).where(
            Gameweek.season == season,
            Gameweek.is_next == True,
        )
    ).scalars().first()

    if nxt is None:
        return None, "No next gameweek found for season=%s. Run /gameweeks/ingest/fpl first." % season

    return int(nxt.gw), None


def _base_candidates_query(
    *,
    season: str,
    target_gw: int,
    model_name: str,
    status: Optional[str],
    max_cost: Optional[int],
    min_predicted_points: Optional[float],
):
    q = (
        select(Prediction, Player, Team)
        .join(Player, Player.id == Prediction.player_id)
        .join(Team, Team.id == Player.team_id)
        .where(
            Prediction.season == season,
            Prediction.target_gw == target_gw,
            Prediction.model_name == model_name,
        )
    )

    if status is not None:
        q = q.where(Player.status == status)

    if max_cost is not None:
        q = q.where(Player.now_cost <= max_cost)

    if min_predicted_points is not None:
        q = q.where(Prediction.predicted_points >= min_predicted_points)

    return q


def _calc_cost_m(now_cost: int) -> float:
    return float(now_cost) / 10.0


def _calc_value(predicted_points: float, cost_m: float) -> float:
    denom = cost_m if cost_m > 0 else 0.1
    return predicted_points / denom


def _serialize_compact(pred: Prediction, pl: Player, tm: Team) -> dict:
    cost_m = _calc_cost_m(int(pl.now_cost))
    pts = float(pred.predicted_points)
    val = _calc_value(pts, cost_m)
    return {
        "season": pred.season,
        "name": pl.web_name,
        "position": pl.position,
        "team": tm.short_name,
        "cost_m": round(cost_m, 1),
        "predicted_points": round(pts, 2),
        "value": round(val, 2),
        "player_id": pl.id,
        "fpl_player_id": pl.fpl_player_id,
        "team_id": tm.id,
    }


def _serialize_full(pred: Prediction, pl: Player, tm: Team) -> dict:
    cost_m = _calc_cost_m(int(pl.now_cost))
    pts = float(pred.predicted_points)
    val = _calc_value(pts, cost_m)
    return {
        "prediction_id": pred.id,
        "season": pred.season,
        "player_id": pl.id,
        "target_gw": pred.target_gw,
        "model_name": pred.model_name,
        "predicted_points": pts,
        "value": val,
        "created_at": pred.created_at.isoformat() if pred.created_at else None,
        "fpl_player_id": pl.fpl_player_id,
        "web_name": pl.web_name,
        "position": pl.position,
        "now_cost": pl.now_cost,
        "status": pl.status,
        "team_id": tm.id,
        "team_short_name": tm.short_name,
        "team_name": tm.name,
    }


def _build_candidate_buckets(
    rows: List[Tuple[Prediction, Player, Team]],
) -> Dict[Position, List[Tuple[Prediction, Player, Team]]]:
    buckets = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for pred, pl, tm in rows:
        pos = pl.position
        if pos in buckets:
            buckets[pos].append((pred, pl, tm))
    return buckets


def _sort_bucket(bucket: List[Tuple[Prediction, Player, Team]], order_by: OrderBy) -> List[Tuple[Prediction, Player, Team]]:
    if order_by == "points":
        return sorted(
            bucket,
            key=lambda r: (float(r[0].predicted_points), -int(r[1].now_cost), -int(r[2].id), -int(r[1].id)),
            reverse=True,
        )
    return sorted(
        bucket,
        key=lambda r: (
            _calc_value(float(r[0].predicted_points), _calc_cost_m(int(r[1].now_cost))),
            float(r[0].predicted_points),
        ),
        reverse=True,
    )


def _remaining_needed(required: Dict[Position, int], have: Dict[Position, int]) -> Dict[Position, int]:
    return dict((p, max(0, required[p] - have.get(p, 0))) for p in required)


def _sum_cheapest_cost_m(
    *,
    pos: Position,
    k: int,
    buckets: Dict[Position, List[Tuple[Prediction, Player, Team]]],
    selected_player_ids: set,
    team_counts: Dict[int, int],
    max_per_team: int,
) -> Optional[float]:
    if k <= 0:
        return 0.0

    costs = []
    for pred, pl, tm in buckets[pos]:
        if pl.id in selected_player_ids:
            continue
        if team_counts.get(tm.id, 0) >= max_per_team:
            continue
        costs.append(_calc_cost_m(int(pl.now_cost)))

    if len(costs) < k:
        return None

    costs.sort()
    return float(sum(costs[:k]))


def _can_complete_squad(
    *,
    remaining_budget_m: float,
    remaining_needed_total: Dict[Position, int],
    buckets: Dict[Position, List[Tuple[Prediction, Player, Team]]],
    selected_player_ids: set,
    team_counts: Dict[int, int],
    max_per_team: int,
) -> bool:
    for pos, need in remaining_needed_total.items():
        if need <= 0:
            continue
        available = 0
        for _, pl, tm in buckets[pos]:
            if pl.id in selected_player_ids:
                continue
            if team_counts.get(tm.id, 0) >= max_per_team:
                continue
            available += 1
        if available < need:
            return False

    min_possible = 0.0
    for pos, need in remaining_needed_total.items():
        if need <= 0:
            continue
        s = _sum_cheapest_cost_m(
            pos=pos,
            k=need,
            buckets=buckets,
            selected_player_ids=selected_player_ids,
            team_counts=team_counts,
            max_per_team=max_per_team,
        )
        if s is None:
            return False
        min_possible += s

    return min_possible <= remaining_budget_m + 1e-9


class TransferRecommendationsRequest(BaseModel):
    target_gw: int
    model_name: str
    squad_player_ids: List[int]
    bank: int = Field(default=0, ge=0)
    free_transfers: int = Field(default=1, ge=0)
    limit: int = Field(default=3, ge=1, le=10)


def _build_squad_team_counts(rows: List[dict]) -> Dict[int, int]:
    counts = {}
    for row in rows:
        team_id = int(row["team_id"])
        counts[team_id] = counts.get(team_id, 0) + 1
    return counts


def _build_squad_position_counts(rows: List[dict]) -> Dict[str, int]:
    counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for row in rows:
        pos = str(row["position"])
        if pos in counts:
            counts[pos] += 1
    return counts


def _passes_budget_cap(*, outgoing_now_cost: int, incoming_now_cost: int, bank: int) -> bool:
    return incoming_now_cost <= outgoing_now_cost + bank


def _passes_position_compatibility(*, outgoing_position: str, incoming_position: str) -> bool:
    return outgoing_position == incoming_position


def _passes_availability_filter(*, incoming_status: str, require_available: bool = True) -> bool:
    if not require_available:
        return True
    return incoming_status == "a"


def _passes_team_cap(*, squad_team_counts: Dict[int, int], outgoing_team_id: int, incoming_team_id: int, max_per_team: int = 3) -> bool:
    current_in_team_count = squad_team_counts.get(incoming_team_id, 0)
    if incoming_team_id == outgoing_team_id:
        resulting_team_count = current_in_team_count
    else:
        resulting_team_count = current_in_team_count + 1
    return resulting_team_count <= max_per_team


def _passes_squad_size_rules(*, squad_position_counts: Dict[str, int], outgoing_position: str, incoming_position: str) -> bool:
    new_counts = dict(squad_position_counts)
    if outgoing_position not in new_counts or incoming_position not in new_counts:
        return False
    new_counts[outgoing_position] -= 1
    if new_counts[outgoing_position] < 0:
        return False
    new_counts[incoming_position] += 1
    return new_counts == squad_position_counts


def _validate_transfer_candidate(
    *,
    incoming_player: Player,
    incoming_team: Team,
    outgoing_position: str,
    outgoing_team_id: int,
    outgoing_now_cost: int,
    squad_player_ids: set,
    squad_team_counts: Dict[int, int],
    squad_position_counts: Dict[str, int],
    bank: int,
    require_available: bool = True,
    max_per_team: int = 3,
) -> Tuple[bool, Optional[str]]:
    if incoming_player.id in squad_player_ids:
        return False, "already_in_squad"
    if not _passes_position_compatibility(
        outgoing_position=outgoing_position,
        incoming_position=incoming_player.position,
    ):
        return False, "position_mismatch"
    if not _passes_availability_filter(
        incoming_status=incoming_player.status,
        require_available=require_available,
    ):
        return False, "incoming_not_available"
    if not _passes_budget_cap(
        outgoing_now_cost=outgoing_now_cost,
        incoming_now_cost=int(incoming_player.now_cost),
        bank=bank,
    ):
        return False, "over_budget"
    if not _passes_team_cap(
        squad_team_counts=squad_team_counts,
        outgoing_team_id=outgoing_team_id,
        incoming_team_id=int(incoming_team.id),
        max_per_team=max_per_team,
    ):
        return False, "team_cap_exceeded"
    if not _passes_squad_size_rules(
        squad_position_counts=squad_position_counts,
        outgoing_position=outgoing_position,
        incoming_position=incoming_player.position,
    ):
        return False, "invalid_squad_transition"
    return True, None


def _serialize_outgoing_candidate(pred: Prediction, pl: Player, tm: Team) -> dict:
    return {
        "season": pred.season,
        "player_id": pl.id,
        "web_name": pl.web_name,
        "position": pl.position,
        "team_id": tm.id,
        "team_name": tm.name,
        "team_short_name": tm.short_name,
        "now_cost": pl.now_cost,
        "predicted_points": float(pred.predicted_points or 0.0),
        "status": pl.status,
    }


def _rank_outgoing_candidates(rows: List[dict]) -> List[dict]:
    return sorted(
        rows,
        key=lambda r: (
            r["predicted_points"],
            -r["now_cost"],
            r["player_id"],
        ),
    )


def _run_transfer_constraint_unit_checks() -> None:
    assert _passes_budget_cap(outgoing_now_cost=60, incoming_now_cost=70, bank=10) is True
    assert _passes_budget_cap(outgoing_now_cost=60, incoming_now_cost=71, bank=10) is False
    assert _passes_team_cap(squad_team_counts={5: 2}, outgoing_team_id=3, incoming_team_id=5, max_per_team=3) is True
    assert _passes_team_cap(squad_team_counts={5: 3}, outgoing_team_id=3, incoming_team_id=5, max_per_team=3) is False
    assert _passes_team_cap(squad_team_counts={3: 3}, outgoing_team_id=3, incoming_team_id=3, max_per_team=3) is True
    assert _passes_position_compatibility(outgoing_position="MID", incoming_position="MID") is True
    assert _passes_position_compatibility(outgoing_position="MID", incoming_position="FWD") is False
    assert _passes_availability_filter(incoming_status="a", require_available=True) is True
    assert _passes_availability_filter(incoming_status="d", require_available=True) is False
    assert _passes_availability_filter(incoming_status="d", require_available=False) is True
    counts = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert _passes_squad_size_rules(squad_position_counts=counts, outgoing_position="MID", incoming_position="MID") is True
    assert _passes_squad_size_rules(squad_position_counts=counts, outgoing_position="MID", incoming_position="FWD") is False


def _try_pick_one(
    *,
    pos: Position,
    ordered_bucket: List[Tuple[Prediction, Player, Team]],
    selected_player_ids: set,
    team_counts: Dict[int, int],
    max_per_team: int,
    remaining_budget_m: float,
    total_have: Dict[Position, int],
    total_required: Dict[Position, int],
    starting_have: Dict[Position, int],
    starting_required: Optional[Dict[Position, int]],
    buckets_all: Dict[Position, List[Tuple[Prediction, Player, Team]]],
) -> Tuple[Optional[Tuple[Prediction, Player, Team]], Optional[str]]:
    need_total = total_required[pos] - total_have.get(pos, 0)
    if need_total <= 0:
        return None, "Position=%s already full for total squad." % pos

    if starting_required is not None:
        need_start = starting_required[pos] - starting_have.get(pos, 0)
        if need_start <= 0:
            return None, "Position=%s already full for starting XI." % pos

    for pred, pl, tm in ordered_bucket:
        if pl.id in selected_player_ids:
            continue
        if team_counts.get(tm.id, 0) >= max_per_team:
            continue

        cost_m = _calc_cost_m(int(pl.now_cost))
        if cost_m > remaining_budget_m + 1e-9:
            continue

        selected_player_ids.add(pl.id)
        team_counts[tm.id] = team_counts.get(tm.id, 0) + 1
        total_have[pos] = total_have.get(pos, 0) + 1
        if starting_required is not None:
            starting_have[pos] = starting_have.get(pos, 0) + 1

        remaining_budget_after = remaining_budget_m - cost_m
        remaining_needed_total = _remaining_needed(total_required, total_have)

        feasible = _can_complete_squad(
            remaining_budget_m=remaining_budget_after,
            remaining_needed_total=remaining_needed_total,
            buckets=buckets_all,
            selected_player_ids=selected_player_ids,
            team_counts=team_counts,
            max_per_team=max_per_team,
        )

        if feasible:
            return (pred, pl, tm), None

        selected_player_ids.remove(pl.id)
        team_counts[tm.id] -= 1
        if team_counts[tm.id] <= 0:
            del team_counts[tm.id]
        total_have[pos] -= 1
        if total_have[pos] <= 0:
            del total_have[pos]
        if starting_required is not None:
            starting_have[pos] -= 1
            if starting_have[pos] <= 0:
                del starting_have[pos]

    return None, "No feasible candidate for position=%s under current constraints." % pos


def _pick_starting_xi(
    *,
    buckets: Dict[Position, List[Tuple[Prediction, Player, Team]]],
    budget_m: float,
    max_per_team: int,
    total_required: Dict[Position, int],
    starting_required: Dict[Position, int],
) -> Tuple[List[Tuple[Prediction, Player, Team]], float, Dict[int, int], Dict[Position, int], List[str]]:
    selected_ids = set()
    team_counts = {}
    total_have = {}
    starting_have = {}
    picked = []
    reasons = []

    ordered_points = dict((p, _sort_bucket(buckets[p], "points")) for p in buckets)
    ordered_value = dict((p, _sort_bucket(buckets[p], "value")) for p in buckets)

    remaining_budget = budget_m

    def starting_done() -> bool:
        return all(starting_have.get(p, 0) >= starting_required[p] for p in starting_required)

    cycle = 0
    guard = 0
    while not starting_done():
        guard += 1
        if guard > 2000:
            reasons.append("Guard hit while building starting XI (unexpected loop).")
            break

        metric = "points" if cycle % 2 == 0 else "value"
        ordered = ordered_points if metric == "points" else ordered_value

        progress_this_cycle = False
        for pos in POSITION_CYCLE:
            if starting_have.get(pos, 0) >= starting_required[pos]:
                continue

            picked_row, err = _try_pick_one(
                pos=pos,
                ordered_bucket=ordered[pos],
                selected_player_ids=selected_ids,
                team_counts=team_counts,
                max_per_team=max_per_team,
                remaining_budget_m=remaining_budget,
                total_have=total_have,
                total_required=total_required,
                starting_have=starting_have,
                starting_required=starting_required,
                buckets_all=buckets,
            )
            if picked_row is not None:
                pred, pl, tm = picked_row
                picked.append(picked_row)
                remaining_budget -= _calc_cost_m(int(pl.now_cost))
                progress_this_cycle = True
            else:
                if err and len(reasons) < 6:
                    reasons.append("[starting:%s] %s" % (metric, err))

        if not progress_this_cycle:
            if len(reasons) < 6:
                reasons.append("Cannot progress while building starting XI. Try relaxing filters.")
            break

        cycle += 1

    return picked, remaining_budget, team_counts, total_have, reasons


def _pick_bench(
    *,
    buckets: Dict[Position, List[Tuple[Prediction, Player, Team]]],
    already_selected: List[Tuple[Prediction, Player, Team]],
    remaining_budget_m: float,
    team_counts: Dict[int, int],
    total_have: Dict[Position, int],
    total_required: Dict[Position, int],
    max_per_team: int,
) -> Tuple[List[Tuple[Prediction, Player, Team]], float, List[str]]:
    selected_ids = set(pl.id for _, pl, _ in already_selected)
    picked = []
    reasons = []

    ordered_value = dict((p, _sort_bucket(buckets[p], "value")) for p in buckets)

    guard = 0
    while any(total_have.get(p, 0) < total_required[p] for p in total_required):
        guard += 1
        if guard > 3000:
            reasons.append("Guard hit while building bench (unexpected loop).")
            break

        progress = False
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            need = total_required[pos] - total_have.get(pos, 0)
            if need <= 0:
                continue

            picked_row, err = _try_pick_one(
                pos=pos,
                ordered_bucket=ordered_value[pos],
                selected_player_ids=selected_ids,
                team_counts=team_counts,
                max_per_team=max_per_team,
                remaining_budget_m=remaining_budget_m,
                total_have=total_have,
                total_required=total_required,
                starting_have={},
                starting_required=None,
                buckets_all=buckets,
            )
            if picked_row is not None:
                pred, pl, tm = picked_row
                picked.append(picked_row)
                remaining_budget_m -= _calc_cost_m(int(pl.now_cost))
                progress = True
            else:
                if err and len(reasons) < 6:
                    reasons.append("[bench:value] %s" % err)

        if not progress:
            if len(reasons) < 6:
                reasons.append("Cannot progress while building bench. Try relaxing filters.")
            break

    return picked, remaining_budget_m, reasons


def _group_by_position(rows: List[Tuple[Prediction, Player, Team]]) -> Dict[Position, List[Tuple[Prediction, Player, Team]]]:
    out = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for pred, pl, tm in rows:
        if pl.position in out:
            out[pl.position].append((pred, pl, tm))
    return out


def _get_recent_player_stats(
    *,
    db: Session,
    season: str,
    player_id: int,
    target_gw: int,
    window: int = 5,
) -> List[PlayerGameweekStat]:
    stmt = (
        select(PlayerGameweekStat)
        .where(
            PlayerGameweekStat.season == season,
            PlayerGameweekStat.player_id == player_id,
            PlayerGameweekStat.gw < target_gw,
        )
        .order_by(PlayerGameweekStat.gw.desc())
        .limit(window)
    )
    return list(db.execute(stmt).scalars().all())


def _build_recent_form_summary(stats: List[PlayerGameweekStat]) -> str:
    if not stats:
        return "No recent stats available."

    avg_points = sum(float(s.total_points or 0.0) for s in stats) / len(stats)
    return "Last %s GWs: avg %.2f pts" % (len(stats), avg_points)


def _build_minutes_stability(stats: List[PlayerGameweekStat]) -> dict:
    if not stats:
        return {
            "label": "unknown",
            "avg_minutes": None,
            "mins_60_plus_count": 0,
            "sample_size": 0,
        }

    minutes = [int(s.minutes or 0) for s in stats]
    avg_minutes = sum(minutes) / len(minutes)
    mins_60_plus_count = sum(1 for m in minutes if m >= 60)

    if avg_minutes >= 75 and mins_60_plus_count >= max(3, len(minutes) - 1):
        label = "high"
    elif avg_minutes >= 45:
        label = "medium"
    else:
        label = "low"

    return {
        "label": label,
        "avg_minutes": round(avg_minutes, 1),
        "mins_60_plus_count": mins_60_plus_count,
        "sample_size": len(stats),
    }


def _captain_label_from_signals(
    predicted_points: float,
    minutes_stability: dict,
    recent_form_summary: str,
) -> str:
    minutes_label = minutes_stability.get("label")
    avg_recent_points = 0.0

    try:
        parts = recent_form_summary.split("avg ")
        if len(parts) > 1:
            avg_recent_points = float(parts[1].split(" pts")[0])
    except Exception:
        avg_recent_points = 0.0

    if predicted_points >= 7.5 and minutes_label == "high" and avg_recent_points >= 5.0:
        return "safe"

    return "upside"


@router.get("/squad")
def recommend_squad(
    target_gw: Optional[int] = None,
    model_name: str = Query(default=MODEL_NAME),
    status: str = Query(default="a"),
    max_cost: Optional[int] = Query(default=None, ge=0),
    min_predicted_points: Optional[float] = Query(default=None, ge=0),
    budget_m: float = Query(default=100.0, ge=50.0, le=200.0),
    max_per_team: int = Query(default=3, ge=1, le=3),
    view: ViewMode = Query(default="compact", regex="^(compact|full)$"),
    db: Session = Depends(get_db),
):
    season = get_current_season()

    decided_gw, err = _decide_target_gw(db, target_gw)
    if err is not None:
        return {"season": season, "error": err}
    assert decided_gw is not None
    target_gw = decided_gw

    effective_status = None if status == "all" else status

    q = _base_candidates_query(
        season=season,
        target_gw=target_gw,
        model_name=model_name,
        status=effective_status,
        max_cost=max_cost,
        min_predicted_points=min_predicted_points,
    )

    rows = db.execute(q).all()
    buckets = _build_candidate_buckets(rows)

    candidates_count = dict((p, len(buckets[p])) for p in buckets)

    missing_by_position = {}
    for pos, need in SQUAD_RULES.items():
        have = candidates_count.get(pos, 0)
        if have < need:
            missing_by_position[pos] = {"need": need, "have": have}
    if missing_by_position:
        return {
            "season": season,
            "target_gw": target_gw,
            "model_name": model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "status": status,
                "max_cost": max_cost,
                "min_predicted_points": min_predicted_points,
                "budget_m": budget_m,
                "max_per_team": max_per_team,
                "rules": SQUAD_RULES,
                "starting_formation": STARTING_FORMATION,
                "view": view,
            },
            "error": "Not enough candidates to satisfy squad rules under current filters.",
            "diagnostics": {
                "missing_by_position": missing_by_position,
                "candidates_count": candidates_count,
            },
        }

    starting_rows, remaining_budget, team_counts, total_have, reasons1 = _pick_starting_xi(
        buckets=buckets,
        budget_m=budget_m,
        max_per_team=max_per_team,
        total_required=SQUAD_RULES,
        starting_required=STARTING_FORMATION,
    )

    starting_have = dict((p, 0) for p in STARTING_FORMATION)
    for _, pl, _ in starting_rows:
        if pl.position in starting_have:
            starting_have[pl.position] += 1
    starting_done = all(starting_have[p] >= STARTING_FORMATION[p] for p in STARTING_FORMATION)
    if not starting_done:
        spent = budget_m - remaining_budget
        return {
            "season": season,
            "target_gw": target_gw,
            "model_name": model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "status": status,
                "max_cost": max_cost,
                "min_predicted_points": min_predicted_points,
                "budget_m": budget_m,
                "max_per_team": max_per_team,
                "rules": SQUAD_RULES,
                "starting_formation": STARTING_FORMATION,
                "view": view,
            },
            "error": "Failed to build a valid starting XI with current constraints.",
            "diagnostics": {
                "reasons": reasons1,
                "starting_have": starting_have,
                "spent_m": round(spent, 1),
                "remaining_m": round(remaining_budget, 1),
                "team_counts": dict((str(k), v) for k, v in team_counts.items()),
                "candidates_count": candidates_count,
            },
        }

    bench_rows, remaining_budget2, reasons2 = _pick_bench(
        buckets=buckets,
        already_selected=starting_rows,
        remaining_budget_m=remaining_budget,
        team_counts=team_counts,
        total_have=total_have,
        total_required=SQUAD_RULES,
        max_per_team=max_per_team,
    )

    final_rows = starting_rows + bench_rows
    final_have = dict((p, 0) for p in SQUAD_RULES)
    for _, pl, _ in final_rows:
        if pl.position in final_have:
            final_have[pl.position] += 1

    squad_done = all(final_have[p] >= SQUAD_RULES[p] for p in SQUAD_RULES) and (len(final_rows) == 15)
    if not squad_done:
        spent = budget_m - remaining_budget2
        return {
            "season": season,
            "target_gw": target_gw,
            "model_name": model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "status": status,
                "max_cost": max_cost,
                "min_predicted_points": min_predicted_points,
                "budget_m": budget_m,
                "max_per_team": max_per_team,
                "rules": SQUAD_RULES,
                "starting_formation": STARTING_FORMATION,
                "view": view,
            },
            "error": "Failed to build a valid squad with current constraints.",
            "diagnostics": {
                "reasons": reasons1 + reasons2,
                "have_by_position": final_have,
                "spent_m": round(spent, 1),
                "remaining_m": round(remaining_budget2, 1),
                "team_counts": dict((str(k), v) for k, v in team_counts.items()),
                "candidates_count": candidates_count,
                "hint": "Try relaxing filters (e.g., max_cost, min_predicted_points, or status=all).",
            },
        }

    serialize = _serialize_compact if view == "compact" else _serialize_full

    starting_grouped = _group_by_position(starting_rows)
    bench_grouped = _group_by_position(bench_rows)

    spent = budget_m - remaining_budget2

    starting_payload = {
        "GKP": [serialize(*r) for r in starting_grouped["GKP"]],
        "DEF": [serialize(*r) for r in starting_grouped["DEF"]],
        "MID": [serialize(*r) for r in starting_grouped["MID"]],
        "FWD": [serialize(*r) for r in starting_grouped["FWD"]],
    }

    bench_payload = {
        "GKP": [serialize(*r) for r in bench_grouped["GKP"]],
        "DEF": [serialize(*r) for r in bench_grouped["DEF"]],
        "MID": [serialize(*r) for r in bench_grouped["MID"]],
        "FWD": [serialize(*r) for r in bench_grouped["FWD"]],
    }

    def _flatten_pos_dict(pos_dict: dict) -> list:
        out = []
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            out.extend(pos_dict.get(pos, []))
        return out

    def _build_bench_list(bench_dict: dict) -> list:
        bench_list_local = []
        gk = bench_dict.get("GKP", [])
        if gk:
            bench_list_local.append(gk[0])

        outfield = []
        outfield.extend(bench_dict.get("DEF", []))
        outfield.extend(bench_dict.get("MID", []))
        outfield.extend(bench_dict.get("FWD", []))

        need = 4 - len(bench_list_local)
        bench_list_local.extend(outfield[:need])
        return bench_list_local

    bench_list = _build_bench_list(bench_payload)
    starting_flat = _flatten_pos_dict(starting_payload)

    def _tag(items: list, role: str) -> list:
        tagged = []
        for i, x in enumerate(items, start=1):
            y = dict(x)
            y["role"] = role
            y["slot"] = i
            tagged.append(y)
        return tagged

    squad_list = _tag(starting_flat, "starting") + _tag(bench_list, "bench")

    return {
        "season": season,
        "target_gw": target_gw,
        "model_name": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "status": status,
            "max_cost": max_cost,
            "min_predicted_points": min_predicted_points,
            "budget_m": budget_m,
            "max_per_team": max_per_team,
            "rules": SQUAD_RULES,
            "starting_formation": STARTING_FORMATION,
            "view": view,
        },
        "summary": {
            "spent_m": round(spent, 1),
            "remaining_m": round(remaining_budget2, 1),
            "team_counts": dict((str(k), v) for k, v in team_counts.items()),
            "squad_counts": final_have,
        },
        "starting_xi": starting_payload,
        "bench": bench_payload,
        "bench_list": bench_list,
        "squad_list": squad_list,
    }


@router.get("/transfers/self-test")
def transfer_constraint_self_test():
    _run_transfer_constraint_unit_checks()
    return {"ok": True}


@router.post("/transfers")
def recommend_transfers(
    req: TransferRecommendationsRequest,
    db: Session = Depends(get_db),
):
    season = get_current_season()

    outgoing_stmt = (
        select(Prediction, Player, Team)
        .join(Player, Player.id == Prediction.player_id)
        .join(Team, Team.id == Player.team_id)
        .where(
            Prediction.season == season,
            Prediction.target_gw == req.target_gw,
            Prediction.model_name == req.model_name,
            Prediction.player_id.in_(req.squad_player_ids),
        )
        .order_by(Prediction.predicted_points.asc(), Player.id.asc())
    )

    outgoing_results = db.execute(outgoing_stmt).all()
    outgoing_candidates = [
        _serialize_outgoing_candidate(pred, pl, tm)
        for pred, pl, tm in outgoing_results
    ]
    outgoing_candidates = _rank_outgoing_candidates(outgoing_candidates)

    if not outgoing_candidates:
        return {
            "season": season,
            "target_gw": req.target_gw,
            "model_name": req.model_name,
            "bank": req.bank,
            "free_transfers": req.free_transfers,
            "squad_player_ids": req.squad_player_ids,
            "limit": req.limit,
            "selected_outgoing": None,
            "outgoing_candidates": [],
            "rows": [],
        }

    out = outgoing_candidates[0]
    squad_team_counts = _build_squad_team_counts(outgoing_candidates)
    squad_position_counts = _build_squad_position_counts(outgoing_candidates)
    squad_player_ids_set = set(req.squad_player_ids)

    incoming_stmt = (
        select(Prediction, Player, Team)
        .join(Player, Player.id == Prediction.player_id)
        .join(Team, Team.id == Player.team_id)
        .where(
            Prediction.season == season,
            Prediction.target_gw == req.target_gw,
            Prediction.model_name == req.model_name,
        )
        .order_by(Prediction.predicted_points.desc(), Player.id.asc())
    )

    incoming_results = db.execute(incoming_stmt).all()

    rows = []
    for pred, pl, tm in incoming_results:
        allowed, _ = _validate_transfer_candidate(
            incoming_player=pl,
            incoming_team=tm,
            outgoing_position=out["position"],
            outgoing_team_id=out["team_id"],
            outgoing_now_cost=out["now_cost"],
            squad_player_ids=squad_player_ids_set,
            squad_team_counts=squad_team_counts,
            squad_position_counts=squad_position_counts,
            bank=req.bank,
            require_available=True,
            max_per_team=3,
        )
        if not allowed:
            continue

        in_pred = float(pred.predicted_points or 0.0)
        out_pred = float(out["predicted_points"])
        gain = in_pred - out_pred
        transfer_cost_points = 0 if req.free_transfers >= 1 else 4
        net_gain_after_cost = gain - transfer_cost_points

        risk_flags = []
        if pl.now_cost > out["now_cost"]:
            risk_flags.append("uses_bank")
        if transfer_cost_points == 4:
            risk_flags.append("costs_minus_4")

        rows.append({
            "season": season,
            "out_player_id": out["player_id"],
            "out_web_name": out["web_name"],
            "out_position": out["position"],
            "out_team_name": out["team_name"],
            "out_now_cost": out["now_cost"],
            "out_predicted_points": out_pred,
            "in_player_id": pl.id,
            "in_web_name": pl.web_name,
            "in_team_name": tm.name,
            "in_team_short_name": tm.short_name,
            "in_position": pl.position,
            "in_now_cost": pl.now_cost,
            "in_predicted_points": in_pred,
            "projected_gain": gain,
            "is_actionable_now": True,
            "transfer_cost_points": transfer_cost_points,
            "net_gain_after_cost": net_gain_after_cost,
            "why_recommended": "Higher predicted points within budget and same position.",
            "risk_flags": risk_flags,
        })

    rows = sorted(
        rows,
        key=lambda r: (r["net_gain_after_cost"], r["projected_gain"]),
        reverse=True,
    )[: req.limit]

    return {
        "season": season,
        "target_gw": req.target_gw,
        "model_name": req.model_name,
        "bank": req.bank,
        "free_transfers": req.free_transfers,
        "squad_player_ids": req.squad_player_ids,
        "limit": req.limit,
        "selected_outgoing": out,
        "outgoing_candidates": outgoing_candidates,
        "squad_team_counts": squad_team_counts,
        "rows": rows,
    }


@router.get("/captain")
def recommend_captain(
    target_gw: int,
    model_name: str = Query(default=MODEL_NAME),
    limit: int = Query(default=5, ge=2, le=10),
    db: Session = Depends(get_db),
):
    season = get_current_season()

    stmt = (
        select(Prediction, Player, Team)
        .join(Player, Player.id == Prediction.player_id)
        .join(Team, Team.id == Player.team_id)
        .where(
            Prediction.season == season,
            Prediction.target_gw == target_gw,
            Prediction.model_name == model_name,
            Player.status == "a",
        )
        .order_by(Prediction.predicted_points.desc(), Player.id.asc())
        .limit(limit)
    )

    results = db.execute(stmt).all()

    top_candidates = []
    for pred, pl, tm in results:
        pts = float(pred.predicted_points or 0.0)

        recent_stats = _get_recent_player_stats(
            db=db,
            season=season,
            player_id=pl.id,
            target_gw=target_gw,
            window=5,
        )
        recent_form_summary = _build_recent_form_summary(recent_stats)
        minutes_stability = _build_minutes_stability(recent_stats)
        captain_label = _captain_label_from_signals(
            pts,
            minutes_stability,
            recent_form_summary,
        )

        top_candidates.append({
            "season": season,
            "player_id": pl.id,
            "web_name": pl.web_name,
            "team_name": tm.name,
            "team_short_name": tm.short_name,
            "position": pl.position,
            "now_cost": pl.now_cost,
            "predicted_points": pts,
            "captain_label": captain_label,
            "recent_form_summary": recent_form_summary,
            "minutes_stability": minutes_stability,
            "explanation": "Predicted %.2f points; %s; minutes stability %s; profile %s." % (
                pts,
                recent_form_summary,
                minutes_stability["label"],
                captain_label,
            ),
            "future_factors": {
                "fixture_difficulty": None,
                "opponent_defense_strength": None,
                "home_away": None,
                "fixture_count": None,
                "match_model_signal": None,
            },
        })

    captain = top_candidates[0] if len(top_candidates) >= 1 else None
    vice_captain = top_candidates[1] if len(top_candidates) >= 2 else None

    return {
        "season": season,
        "target_gw": target_gw,
        "model_name": model_name,
        "captain": captain,
        "vice_captain": vice_captain,
        "top_candidates": top_candidates,
    }