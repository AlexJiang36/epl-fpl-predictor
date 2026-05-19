from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models.player import Player
from app.models.prediction import Prediction
from app.models.team import Team
from app.schemas.free_hit import (
    FreeHitBuildRequest,
    FreeHitBuildResponse,
    FreeHitPlayer,
    FreeHitRejectedCandidate,
)
from app.schemas.wildcard import WildcardBuildRequest, WildcardBuildResponse
from app.utils.wildcard_builder import build_wildcard_squad_v1

router = APIRouter(prefix="/chips", tags=["chips"])

SQUAD_RULES = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_RULES = {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3}
POSITION_ORDER = ["FWD", "MID", "DEF", "GKP"]


@router.get("/ping")
def ping():
    return {"ok": True}


def _serialize_player(pred: Prediction, pl: Player, tm: Team) -> FreeHitPlayer:
    return FreeHitPlayer(
        player_id=pl.id,
        web_name=pl.web_name,
        team_name=tm.name,
        team_short_name=tm.short_name,
        position=pl.position,
        now_cost=int(pl.now_cost),
        predicted_points=float(pred.predicted_points or 0.0),
        status=pl.status,
    )


def _serialize_rejected(
    pred: Prediction,
    pl: Player,
    tm: Team,
    reason: str,
) -> FreeHitRejectedCandidate:
    return FreeHitRejectedCandidate(
        player_id=pl.id,
        web_name=pl.web_name,
        team_name=tm.name,
        team_short_name=tm.short_name,
        position=pl.position,
        now_cost=int(pl.now_cost),
        predicted_points=float(pred.predicted_points or 0.0),
        rejected_reason=reason,
    )


def _group_by_position(
    rows: List[Tuple[Prediction, Player, Team]],
) -> Dict[str, List[Tuple[Prediction, Player, Team]]]:
    buckets = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for row in rows:
        _, pl, _ = row
        if pl.position in buckets:
            buckets[pl.position].append(row)

    for pos in buckets:
        buckets[pos] = sorted(
            buckets[pos],
            key=lambda r: (
                float(r[0].predicted_points or 0.0),
                -int(r[1].now_cost),
                -int(r[1].id),
            ),
            reverse=True,
        )
    return buckets


def _min_cost_needed_for_remaining(
    *,
    buckets: Dict[str, List[Tuple[Prediction, Player, Team]]],
    selected_ids: set[int],
    team_counts: Dict[int, int],
    need_by_pos: Dict[str, int],
    max_per_team: int = 3,
) -> Optional[int]:
    total = 0

    for pos, need in need_by_pos.items():
        if need <= 0:
            continue

        costs: List[int] = []
        for pred, pl, tm in buckets[pos]:
            if pl.id in selected_ids:
                continue
            if team_counts.get(tm.id, 0) >= max_per_team:
                continue
            costs.append(int(pl.now_cost))

        if len(costs) < need:
            return None

        costs.sort()
        total += sum(costs[:need])

    return total


def _try_add_player(
    *,
    row: Tuple[Prediction, Player, Team],
    buckets: Dict[str, List[Tuple[Prediction, Player, Team]]],
    selected_rows: List[Tuple[Prediction, Player, Team]],
    selected_ids: set[int],
    team_counts: Dict[int, int],
    pos_counts: Dict[str, int],
    budget_tenths: int,
    spent_tenths: int,
    rejected_candidates: List[FreeHitRejectedCandidate],
    max_rejected: int = 40,
) -> tuple[bool, int]:
    pred, pl, tm = row
    pos = pl.position
    cost = int(pl.now_cost)

    if pl.id in selected_ids:
        return False, spent_tenths

    if pos_counts[pos] >= SQUAD_RULES[pos]:
        if len(rejected_candidates) < max_rejected:
            rejected_candidates.append(_serialize_rejected(pred, pl, tm, "position_full"))
        return False, spent_tenths

    if team_counts.get(tm.id, 0) >= 3:
        if len(rejected_candidates) < max_rejected:
            rejected_candidates.append(_serialize_rejected(pred, pl, tm, "max_3_players_per_club"))
        return False, spent_tenths

    new_spent = spent_tenths + cost
    if new_spent > budget_tenths:
        if len(rejected_candidates) < max_rejected:
            rejected_candidates.append(_serialize_rejected(pred, pl, tm, "budget_cap"))
        return False, spent_tenths

    # Feasibility lower-bound check for remaining slots
    selected_ids.add(pl.id)
    team_counts[tm.id] = team_counts.get(tm.id, 0) + 1
    pos_counts[pos] += 1

    need_by_pos = {
        p: SQUAD_RULES[p] - pos_counts[p]
        for p in SQUAD_RULES
    }
    min_remaining = _min_cost_needed_for_remaining(
        buckets=buckets,
        selected_ids=selected_ids,
        team_counts=team_counts,
        need_by_pos=need_by_pos,
        max_per_team=3,
    )

    feasible = min_remaining is not None and (new_spent + min_remaining <= budget_tenths)

    if feasible:
        selected_rows.append(row)
        return True, new_spent

    # rollback
    selected_ids.remove(pl.id)
    team_counts[tm.id] -= 1
    if team_counts[tm.id] <= 0:
        del team_counts[tm.id]
    pos_counts[pos] -= 1

    if len(rejected_candidates) < max_rejected:
        rejected_candidates.append(_serialize_rejected(pred, pl, tm, "cannot_complete_valid_squad"))
    return False, spent_tenths


def _build_starting_and_bench(
    squad_players: List[FreeHitPlayer],
) -> tuple[List[FreeHitPlayer], List[FreeHitPlayer]]:
    by_pos: Dict[str, List[FreeHitPlayer]] = defaultdict(list)
    for p in squad_players:
        by_pos[p.position].append(p)

    for pos in by_pos:
        by_pos[pos] = sorted(
            by_pos[pos],
            key=lambda x: (x.predicted_points, -x.now_cost, -x.player_id),
            reverse=True,
        )

    starting_xi: List[FreeHitPlayer] = []
    bench: List[FreeHitPlayer] = []

    # starting XI: 1/3/4/3
    starting_xi.extend(by_pos["GKP"][:1])
    starting_xi.extend(by_pos["DEF"][:3])
    starting_xi.extend(by_pos["MID"][:4])
    starting_xi.extend(by_pos["FWD"][:3])

    starting_ids = {p.player_id for p in starting_xi}

    # bench: leftover GKP first, then remaining outfield by projected points
    leftover_gkp = [p for p in by_pos["GKP"] if p.player_id not in starting_ids]
    leftover_def = [p for p in by_pos["DEF"] if p.player_id not in starting_ids]
    leftover_mid = [p for p in by_pos["MID"] if p.player_id not in starting_ids]
    leftover_fwd = [p for p in by_pos["FWD"] if p.player_id not in starting_ids]

    bench.extend(leftover_gkp[:1])

    outfield_leftovers = leftover_def + leftover_mid + leftover_fwd
    outfield_leftovers = sorted(
        outfield_leftovers,
        key=lambda x: (x.predicted_points, -x.now_cost, -x.player_id),
        reverse=True,
    )
    bench.extend(outfield_leftovers[:3])

    return starting_xi, bench


def _choose_captains(starting_xi: List[FreeHitPlayer]) -> Tuple[Optional[FreeHitPlayer], Optional[FreeHitPlayer]]:
    ordered = sorted(
        starting_xi,
        key=lambda x: (x.predicted_points, -x.now_cost, -x.player_id),
        reverse=True,
    )
    captain = ordered[0] if len(ordered) >= 1 else None
    vice_captain = ordered[1] if len(ordered) >= 2 else None
    return captain, vice_captain

def _count_positions(players: List[FreeHitPlayer]) -> Dict[str, int]:
    counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in players:
        if p.position in counts:
            counts[p.position] += 1
    return counts


def _validate_free_hit_result(
    *,
    starting_xi: List[FreeHitPlayer],
    bench: List[FreeHitPlayer],
    captain: Optional[FreeHitPlayer],
    vice_captain: Optional[FreeHitPlayer],
    locked_player_ids: List[int],
    budget: float,
    spent_m: float,
) -> List[str]:
    errors: List[str] = []

    squad = starting_xi + bench

    squad_ids = [p.player_id for p in squad]
    starting_ids = [p.player_id for p in starting_xi]
    bench_ids = [p.player_id for p in bench]

    # sizes
    if len(starting_xi) != 11:
        errors.append(f"starting_xi must have 11 players, got {len(starting_xi)}")
    if len(bench) != 4:
        errors.append(f"bench must have 4 players, got {len(bench)}")
    if len(squad) != 15:
        errors.append(f"full squad must have 15 players, got {len(squad)}")

    # uniqueness
    if len(set(squad_ids)) != 15:
        errors.append("squad contains duplicate player_ids")
    if set(starting_ids) & set(bench_ids):
        errors.append("starting_xi and bench overlap")

    # exact position rules
    squad_pos_counts = _count_positions(squad)
    starting_pos_counts = _count_positions(starting_xi)

    if squad_pos_counts != SQUAD_RULES:
        errors.append(f"invalid squad composition: got {squad_pos_counts}, expected {SQUAD_RULES}")

    if starting_pos_counts != STARTING_RULES:
        errors.append(f"invalid starting XI formation: got {starting_pos_counts}, expected {STARTING_RULES}")

    # captain / vice
    if captain is None:
        errors.append("captain is missing")
    if vice_captain is None:
        errors.append("vice_captain is missing")

    if captain is not None and captain.player_id not in set(starting_ids):
        errors.append("captain must be in starting_xi")
    if vice_captain is not None and vice_captain.player_id not in set(starting_ids):
        errors.append("vice_captain must be in starting_xi")
    if captain is not None and vice_captain is not None and captain.player_id == vice_captain.player_id:
        errors.append("captain and vice_captain must be different")

    # team cap
    team_counts: Dict[str, int] = {}
    for p in squad:
        team_counts[p.team_short_name] = team_counts.get(p.team_short_name, 0) + 1
    too_many = {team: n for team, n in team_counts.items() if n > 3}
    if too_many:
        errors.append(f"max 3 players per club violated: {too_many}")

    # availability
    unavailable = [p.web_name for p in squad if p.status != "a"]
    if unavailable:
        errors.append(f"squad contains non-available players: {unavailable}")

    # locked players must be included
    missing_locked = [pid for pid in locked_player_ids if pid not in set(squad_ids)]
    if missing_locked:
        errors.append(f"locked players missing from squad: {missing_locked}")

    # budget
    if spent_m - budget > 1e-9:
        errors.append(f"budget exceeded: spent {spent_m}, budget {budget}")

    return errors

@router.post("/free-hit/build", response_model=FreeHitBuildResponse)
def build_free_hit(
    req: FreeHitBuildRequest,
    db: Session = Depends(get_db),
):
    budget_tenths = int(round(req.budget * 10))

    stmt = (
        select(Prediction, Player, Team)
        .join(Player, Player.id == Prediction.player_id)
        .join(Team, Team.id == Player.team_id)
        .where(
            Prediction.target_gw == req.target_gw,
            Prediction.model_name == req.model_name,
            Player.status == "a",
        )
        .order_by(Prediction.predicted_points.desc(), Player.id.asc())
    )
    all_rows = db.execute(stmt).all()

    if not all_rows:
        raise HTTPException(status_code=400, detail="No available prediction rows found for this target_gw/model_name.")

    buckets = _group_by_position(all_rows)

    # Build lookup for locked players
    row_by_player_id = {}
    for row in all_rows:
        pred, pl, tm = row
        row_by_player_id[pl.id] = row

    selected_rows: List[Tuple[Prediction, Player, Team]] = []
    selected_ids: set[int] = set()
    team_counts: Dict[int, int] = {}
    pos_counts: Dict[str, int] = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    rejected_candidates: List[FreeHitRejectedCandidate] = []
    spent_tenths = 0

    # Step 1: lock required players
    for player_id in req.locked_player_ids:
        row = row_by_player_id.get(player_id)
        if row is None:
            raise HTTPException(
                status_code=400,
                detail=f"Locked player_id={player_id} not found in available candidate pool.",
            )
        ok, spent_tenths = _try_add_player(
            row=row,
            buckets=buckets,
            selected_rows=selected_rows,
            selected_ids=selected_ids,
            team_counts=team_counts,
            pos_counts=pos_counts,
            budget_tenths=budget_tenths,
            spent_tenths=spent_tenths,
            rejected_candidates=rejected_candidates,
        )
        if not ok:
            raise HTTPException(
                status_code=400,
                detail=f"Locked player_id={player_id} makes the squad infeasible under current constraints.",
            )

    # Step 2: greedily complete squad by position
    for pos in POSITION_ORDER:
        while pos_counts[pos] < SQUAD_RULES[pos]:
            added = False
            for row in buckets[pos]:
                ok, spent_tenths = _try_add_player(
                    row=row,
                    buckets=buckets,
                    selected_rows=selected_rows,
                    selected_ids=selected_ids,
                    team_counts=team_counts,
                    pos_counts=pos_counts,
                    budget_tenths=budget_tenths,
                    spent_tenths=spent_tenths,
                    rejected_candidates=rejected_candidates,
                )
                if ok:
                    added = True
                    break

            if not added:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to complete a valid Free Hit squad for position {pos}.",
                )

    squad_list = [_serialize_player(pred, pl, tm) for pred, pl, tm in selected_rows]
    squad_list = sorted(
        squad_list,
        key=lambda x: (POSITION_ORDER.index(x.position), -x.predicted_points, -x.now_cost, -x.player_id),
    )

    starting_xi, bench = _build_starting_and_bench(squad_list)
    captain, vice_captain = _choose_captains(starting_xi)

    spent_m = spent_tenths / 10.0
    remaining_m = (budget_tenths - spent_tenths) / 10.0
    projected_points_starting_xi = sum(p.predicted_points for p in starting_xi)
    projected_points_total_15 = sum(p.predicted_points for p in squad_list)

    validation_errors = _validate_free_hit_result(
        starting_xi=starting_xi,
        bench=bench,
        captain=captain,
        vice_captain=vice_captain,
        locked_player_ids=req.locked_player_ids,
        budget=req.budget,
        spent_m=spent_m,
    )

    if validation_errors:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Built invalid Free Hit squad.",
                "errors": validation_errors,
            },
        )

    return FreeHitBuildResponse(
        target_gw=req.target_gw,
        budget=req.budget,
        model_name=req.model_name,
        locked_player_ids=req.locked_player_ids,
        scoring_objective="maximize_projected_gw_points_only",
        starting_xi=starting_xi,
        bench=bench,
        captain=captain,
        vice_captain=vice_captain,
        spent_m=round(spent_m, 1),
        remaining_m=round(remaining_m, 1),
        projected_points_starting_xi=round(projected_points_starting_xi, 2),
        projected_points_total_15=round(projected_points_total_15, 2),
        rejected_candidates=rejected_candidates,
        notes="Deterministic greedy Free Hit builder v1.",
    )

def _run_wildcard_plan(
    req: WildcardBuildRequest,
    db: Session,
) -> WildcardBuildResponse:
    return build_wildcard_squad_v1(
        db=db,
        target_gw=req.target_gw,
        horizon=req.horizon,
        budget=req.budget,
        model_name=req.model_name,
        locked_player_ids=req.locked_player_ids,
        current_squad_player_ids=req.current_squad_player_ids,
    )

@router.post("/wildcard/plan", response_model=WildcardBuildResponse)
def plan_wildcard(
    req: WildcardBuildRequest,
    db: Session = Depends(get_db),
):
    try:
        return _run_wildcard_plan(req, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/wildcard/build", response_model=WildcardBuildResponse)
def build_wildcard_legacy(
    req: WildcardBuildRequest,
    db: Session = Depends(get_db),
):
    try:
        return _run_wildcard_plan(req, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))