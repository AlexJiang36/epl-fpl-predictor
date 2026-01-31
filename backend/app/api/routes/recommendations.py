# app/api/routes/recommendations.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Literal, Tuple

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func as sa_func
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models.prediction import Prediction
from app.models.player import Player
from app.models.team import Team
from app.models.gameweek import Gameweek

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

MODEL_NAME = "baseline_rollavg_v0"

Position = Literal["GKP", "DEF", "MID", "FWD"]
OrderBy = Literal["points", "value"]
ViewMode = Literal["compact", "full"]

SQUAD_RULES: Dict[Position, int] = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
# Day13：先把 11 人首发凑齐（默认 3-4-3）
STARTING_FORMATION: Dict[Position, int] = {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3}
POSITION_CYCLE: List[Position] = ["FWD", "MID", "DEF", "GKP"]


@router.get("/ping")
def ping():
    return {"ok": True}


# -----------------------------
# Helpers: querying candidates
# -----------------------------
def _decide_target_gw(db: Session, target_gw: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
    if target_gw is not None:
        return target_gw, None
    nxt = db.execute(select(Gameweek).where(Gameweek.is_next == True)).scalars().first()
    if nxt is None:
        return None, "No next gameweek found. Run /gameweeks/ingest/fpl first."
    return int(nxt.gw), None


def _base_candidates_query(
    *,
    target_gw: int,
    model_name: str,
    status: Optional[str],  # None means no filter
    max_cost: Optional[int],
    min_predicted_points: Optional[float],
):
    q = (
        select(Prediction, Player, Team)
        .join(Player, Player.id == Prediction.player_id)
        .join(Team, Team.id == Player.team_id)
        .where(
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
    # FPL now_cost is like 80 => 8.0m
    return float(now_cost) / 10.0


def _calc_value(predicted_points: float, cost_m: float) -> float:
    # avoid div by zero; cost_m should never be 0, but be safe
    denom = cost_m if cost_m > 0 else 0.1
    return predicted_points / denom


def _serialize_compact(pred: Prediction, pl: Player, tm: Team) -> dict:
    cost_m = _calc_cost_m(int(pl.now_cost))
    pts = float(pred.predicted_points)
    val = _calc_value(pts, cost_m)
    return {
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
    buckets: Dict[Position, List[Tuple[Prediction, Player, Team]]] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
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
    # value
    return sorted(
        bucket,
        key=lambda r: (
            _calc_value(float(r[0].predicted_points), _calc_cost_m(int(r[1].now_cost))),
            float(r[0].predicted_points),
        ),
        reverse=True,
    )


# -----------------------------
# Helpers: feasibility checks
# -----------------------------
def _remaining_needed(required: Dict[Position, int], have: Dict[Position, int]) -> Dict[Position, int]:
    return {p: max(0, required[p] - have.get(p, 0)) for p in required}


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

    costs: List[float] = []
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
    # 1) Quantity check (with team cap applied)
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

    # 2) Lower-bound budget check: sum of cheapest possible players needed per position
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


# -----------------------------
# Helpers: picking logic
# -----------------------------
def _try_pick_one(
    *,
    pos: Position,
    ordered_bucket: List[Tuple[Prediction, Player, Team]],
    selected_player_ids: set,
    team_counts: Dict[int, int],
    max_per_team: int,
    remaining_budget_m: float,
    # constraints:
    total_have: Dict[Position, int],
    total_required: Dict[Position, int],
    starting_have: Dict[Position, int],
    starting_required: Optional[Dict[Position, int]],  # None means not in starting phase
    # feasibility:
    buckets_all: Dict[Position, List[Tuple[Prediction, Player, Team]]],
) -> Tuple[Optional[Tuple[Prediction, Player, Team]], Optional[str]]:
    """
    Pick the best feasible player from ordered_bucket for position=pos.
    Returns (picked_row, error_reason_if_none)
    """

    need_total = total_required[pos] - total_have.get(pos, 0)
    if need_total <= 0:
        return None, f"Position={pos} already full for total squad."

    if starting_required is not None:
        need_start = starting_required[pos] - starting_have.get(pos, 0)
        if need_start <= 0:
            return None, f"Position={pos} already full for starting XI."

    for pred, pl, tm in ordered_bucket:
        if pl.id in selected_player_ids:
            continue
        if team_counts.get(tm.id, 0) >= max_per_team:
            continue

        cost_m = _calc_cost_m(int(pl.now_cost))
        if cost_m > remaining_budget_m + 1e-9:
            continue

        # Hypothetical add
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

        # rollback
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

    return None, f"No feasible candidate for position={pos} under current constraints."


def _pick_starting_xi(
    *,
    buckets: Dict[Position, List[Tuple[Prediction, Player, Team]]],
    budget_m: float,
    max_per_team: int,
    total_required: Dict[Position, int],
    starting_required: Dict[Position, int],
) -> Tuple[List[Tuple[Prediction, Player, Team]], float, Dict[int, int], Dict[Position, int], List[str]]:
    """
    Returns (starting_rows, remaining_budget, team_counts, total_have_counts, diagnostics_reasons)
    """
    selected_ids: set = set()
    team_counts: Dict[int, int] = {}
    total_have: Dict[Position, int] = {}
    starting_have: Dict[Position, int] = {}
    picked: List[Tuple[Prediction, Player, Team]] = []
    reasons: List[str] = []

    # Prepare ordered buckets for both metrics
    ordered_points = {p: _sort_bucket(buckets[p], "points") for p in buckets}
    ordered_value = {p: _sort_bucket(buckets[p], "value") for p in buckets}

    remaining_budget = budget_m

    # We alternate cycles:
    # cycle A: points, positions in FWD->MID->DEF->GKP
    # cycle B: value,  positions in FWD->MID->DEF->GKP
    # until starting XI complete
    def starting_done() -> bool:
        return all(starting_have.get(p, 0) >= starting_required[p] for p in starting_required)

    cycle = 0
    guard = 0
    while not starting_done():
        guard += 1
        if guard > 2000:
            reasons.append("Guard hit while building starting XI (unexpected loop).")
            break

        metric: OrderBy = "points" if cycle % 2 == 0 else "value"
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
                if err:
                    # don't spam the response too much; only keep a few
                    if len(reasons) < 6:
                        reasons.append(f"[starting:{metric}] {err}")

        if not progress_this_cycle:
            # cannot progress -> fail early
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
    selected_ids = {pl.id for _, pl, _ in already_selected}
    picked: List[Tuple[Prediction, Player, Team]] = []
    reasons: List[str] = []

    ordered_value = {p: _sort_bucket(buckets[p], "value") for p in buckets}

    guard = 0
    while any(total_have.get(p, 0) < total_required[p] for p in total_required):
        guard += 1
        if guard > 3000:
            reasons.append("Guard hit while building bench (unexpected loop).")
            break

        progress = False
        # Bench: fill missing positions by value. Iterate pos order stable.
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos = pos  # type: ignore[assignment]
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
                starting_have={},            # not used in bench phase
                starting_required=None,      # bench phase
                buckets_all=buckets,
            )
            if picked_row is not None:
                pred, pl, tm = picked_row
                picked.append(picked_row)
                remaining_budget_m -= _calc_cost_m(int(pl.now_cost))
                progress = True
            else:
                if err and len(reasons) < 6:
                    reasons.append(f"[bench:value] {err}")

        if not progress:
            if len(reasons) < 6:
                reasons.append("Cannot progress while building bench. Try relaxing filters.")
            break

    return picked, remaining_budget_m, reasons


def _group_by_position(rows: List[Tuple[Prediction, Player, Team]]) -> Dict[Position, List[Tuple[Prediction, Player, Team]]]:
    out: Dict[Position, List[Tuple[Prediction, Player, Team]]] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for pred, pl, tm in rows:
        if pl.position in out:
            out[pl.position].append((pred, pl, tm))
    return out


# -----------------------------
# Endpoint: /recommendations/squad
# -----------------------------
@router.get("/squad")
def recommend_squad(
    target_gw: Optional[int] = None,
    model_name: str = Query(default=MODEL_NAME),

    # Filters (keep them light by default; let budget/team cap do most work)
    status: str = Query(default="a"),  # pass "all" to disable
    max_cost: Optional[int] = Query(default=None, ge=0),
    min_predicted_points: Optional[float] = Query(default=None, ge=0),

    # Squad constraints
    budget_m: float = Query(default=100.0, ge=50.0, le=200.0),
    max_per_team: int = Query(default=3, ge=1, le=3),

    # Output
    view: ViewMode = Query(default="compact", pattern="^(compact|full)$"),

    db: Session = Depends(get_db),
):
    """
    Build a full 15-man FPL-style squad in one shot.

    Rules:
    - Total squad: 2 GKP / 5 DEF / 5 MID / 3 FWD (15)
    - Starting XI (default): 3-4-3 (11)
    - Budget: 100.0m
    - Max 3 players per team
    - Selection strategy:
      1) Build starting XI using alternating cycles:
         - cycle points: FWD -> MID -> DEF -> GKP
         - cycle value : FWD -> MID -> DEF -> GKP
      2) Fill bench to complete 15-man squad using value
      3) Every pick runs feasibility checks to avoid dead-ends
    """
    decided_gw, err = _decide_target_gw(db, target_gw)
    if err is not None:
        return {"error": err}
    assert decided_gw is not None
    target_gw = decided_gw

    effective_status: Optional[str] = None if status == "all" else status

    # Pull candidate rows
    q = _base_candidates_query(
        target_gw=target_gw,
        model_name=model_name,
        status=effective_status,
        max_cost=max_cost,
        min_predicted_points=min_predicted_points,
    )

    rows = db.execute(q).all()
    buckets = _build_candidate_buckets(rows)

    # Diagnostics: how many candidates per position under filters
    candidates_count = {p: len(buckets[p]) for p in buckets}

    # Quick fail if impossible even by raw counts (without team cap/budget)
    missing_by_position = {}
    for pos, need in SQUAD_RULES.items():
        have = candidates_count.get(pos, 0)
        if have < need:
            missing_by_position[pos] = {"need": need, "have": have}
    if missing_by_position:
        return {
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

    # 1) Build starting XI
    starting_rows, remaining_budget, team_counts, total_have, reasons1 = _pick_starting_xi(
        buckets=buckets,
        budget_m=budget_m,
        max_per_team=max_per_team,
        total_required=SQUAD_RULES,
        starting_required=STARTING_FORMATION,
    )

    # if starting not complete, fail (with helpful diag)
    starting_have = {p: 0 for p in STARTING_FORMATION}
    for _, pl, _ in starting_rows:
        if pl.position in starting_have:
            starting_have[pl.position] += 1
    starting_done = all(starting_have[p] >= STARTING_FORMATION[p] for p in STARTING_FORMATION)
    if not starting_done:
        spent = budget_m - remaining_budget
        return {
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
                "team_counts": {str(k): v for k, v in team_counts.items()},
                "candidates_count": candidates_count,
            },
        }

    # 2) Fill bench to complete squad
    bench_rows, remaining_budget2, reasons2 = _pick_bench(
        buckets=buckets,
        already_selected=starting_rows,
        remaining_budget_m=remaining_budget,
        team_counts=team_counts,
        total_have=total_have,
        total_required=SQUAD_RULES,
        max_per_team=max_per_team,
    )

    # Verify full 15-man squad
    final_rows = starting_rows + bench_rows
    final_have = {p: 0 for p in SQUAD_RULES}
    for _, pl, _ in final_rows:
        if pl.position in final_have:
            final_have[pl.position] += 1

    squad_done = all(final_have[p] >= SQUAD_RULES[p] for p in SQUAD_RULES) and (len(final_rows) == 15)
    if not squad_done:
        spent = budget_m - remaining_budget2
        return {
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
                "team_counts": {str(k): v for k, v in team_counts.items()},
                "candidates_count": candidates_count,
                "hint": "Try relaxing filters (e.g., max_cost, min_predicted_points, or status=all).",
            },
        }

    # Output serialization
    if view == "compact":
        serialize = _serialize_compact
    else:
        serialize = _serialize_full

    starting_grouped = _group_by_position(starting_rows)
    bench_grouped = _group_by_position(bench_rows)

    spent = budget_m - remaining_budget2

    # 1) Build payload dicts first (so we can reuse them)
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

    # 2) Helpers: flatten + bench_list (fixed 4)
    def _flatten_pos_dict(pos_dict: dict) -> list:
        out = []
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            out.extend(pos_dict.get(pos, []))
        return out

    def _build_bench_list(bench_dict: dict) -> list:
        """
        Fixed 4 bench players in order:
        [bench_gkp, outfield1, outfield2, outfield3]
        Outfield priority: DEF -> MID -> FWD
        """
        bench_list_local = []

        # bench GK
        gk = bench_dict.get("GKP", [])
        if gk:
            bench_list_local.append(gk[0])

        # outfield priority
        outfield = []
        outfield.extend(bench_dict.get("DEF", []))
        outfield.extend(bench_dict.get("MID", []))
        outfield.extend(bench_dict.get("FWD", []))

        need = 4 - len(bench_list_local)
        bench_list_local.extend(outfield[:need])
        return bench_list_local

    bench_list = _build_bench_list(bench_payload)

    # 3) Optional: squad_list (15 flat)
    starting_flat = _flatten_pos_dict(starting_payload)

    def _tag(items: list, role: str) -> list:
        tagged = []
        for i, x in enumerate(items, start=1):
            y = dict(x)  # shallow copy
            y["role"] = role
            y["slot"] = i
            tagged.append(y)
        return tagged

    squad_list = _tag(starting_flat, "starting") + _tag(bench_list, "bench")

    return {
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
            "team_counts": {str(k): v for k, v in team_counts.items()},
            "squad_counts": final_have,
        },
        "starting_xi": starting_payload,
        "bench": bench_payload,

        # ✅ new
        "bench_list": bench_list,   # fixed 4
        "squad_list": squad_list,   # 15 flat
    }

