from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from app.schemas.free_hit import FreeHitPlayer, FreeHitRejectedCandidate
from app.schemas.wildcard import WildcardBuildResponse
from app.utils.wildcard_horizon import build_wildcard_horizon_snapshot
from app.schemas.wildcard import (
    WildcardBuildResponse,
    WildcardPriorityTransfer,
)
from app.utils.wildcard_transfer_summary import build_priority_transfers_from_current_squad

SQUAD_RULES = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_RULES = {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3}
POSITION_ORDER = ["FWD", "MID", "DEF", "GKP"]


def _serialize_horizon_player(row: dict) -> FreeHitPlayer:
    return FreeHitPlayer(
        player_id=row["player_id"],
        web_name=row["web_name"],
        team_name=row["team_name"],
        team_short_name=row["team_short_name"],
        position=row["position"],
        now_cost=int(row["now_cost"]),
        predicted_points=float(row["horizon_predicted_points"]),
        status=row["status"],
    )


def _serialize_rejected(row: dict, reason: str) -> FreeHitRejectedCandidate:
    return FreeHitRejectedCandidate(
        player_id=row["player_id"],
        web_name=row["web_name"],
        team_name=row["team_name"],
        team_short_name=row["team_short_name"],
        position=row["position"],
        now_cost=int(row["now_cost"]),
        predicted_points=float(row["horizon_predicted_points"]),
        rejected_reason=reason,
    )


def _group_by_position(rows: List[dict]) -> Dict[str, List[dict]]:
    buckets = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for row in rows:
        pos = row["position"]
        if pos in buckets:
            buckets[pos].append(row)

    for pos in buckets:
        buckets[pos] = sorted(
            buckets[pos],
            key=lambda r: (
                float(r["horizon_predicted_points"]),
                float(r["avg_predicted_points_per_gw"]),
                -int(r["now_cost"]),
                -int(r["player_id"]),
            ),
            reverse=True,
        )
    return buckets


def _count_positions(players: List[FreeHitPlayer]) -> Dict[str, int]:
    counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in players:
        if p.position in counts:
            counts[p.position] += 1
    return counts


def _min_cost_needed_for_remaining(
    *,
    buckets: Dict[str, List[dict]],
    selected_ids: set[int],
    team_counts: Dict[int, int],
    need_by_pos: Dict[str, int],
    max_per_team: int = 3,
) -> int | None:
    total = 0

    for pos, need in need_by_pos.items():
        if need <= 0:
            continue

        costs: List[int] = []
        for row in buckets[pos]:
            if row["player_id"] in selected_ids:
                continue
            if team_counts.get(row["team_id"], 0) >= max_per_team:
                continue
            costs.append(int(row["now_cost"]))

        if len(costs) < need:
            return None

        costs.sort()
        total += sum(costs[:need])

    return total


def _try_add_player(
    *,
    row: dict,
    buckets: Dict[str, List[dict]],
    selected_rows: List[dict],
    selected_ids: set[int],
    team_counts: Dict[int, int],
    pos_counts: Dict[str, int],
    budget_tenths: int,
    spent_tenths: int,
    rejected_candidates: List[FreeHitRejectedCandidate],
    max_rejected: int = 40,
) -> tuple[bool, int]:
    pos = row["position"]
    cost = int(row["now_cost"])
    team_id = int(row["team_id"])
    player_id = int(row["player_id"])

    if player_id in selected_ids:
        return False, spent_tenths

    if pos_counts[pos] >= SQUAD_RULES[pos]:
        if len(rejected_candidates) < max_rejected:
            rejected_candidates.append(_serialize_rejected(row, "position_full"))
        return False, spent_tenths

    if team_counts.get(team_id, 0) >= 3:
        if len(rejected_candidates) < max_rejected:
            rejected_candidates.append(_serialize_rejected(row, "max_3_players_per_club"))
        return False, spent_tenths

    new_spent = spent_tenths + cost
    if new_spent > budget_tenths:
        if len(rejected_candidates) < max_rejected:
            rejected_candidates.append(_serialize_rejected(row, "budget_cap"))
        return False, spent_tenths

    selected_ids.add(player_id)
    team_counts[team_id] = team_counts.get(team_id, 0) + 1
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

    selected_ids.remove(player_id)
    team_counts[team_id] -= 1
    if team_counts[team_id] <= 0:
        del team_counts[team_id]
    pos_counts[pos] -= 1

    if len(rejected_candidates) < max_rejected:
        rejected_candidates.append(_serialize_rejected(row, "cannot_complete_valid_squad"))
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

    starting_xi.extend(by_pos["GKP"][:1])
    starting_xi.extend(by_pos["DEF"][:3])
    starting_xi.extend(by_pos["MID"][:4])
    starting_xi.extend(by_pos["FWD"][:3])

    starting_ids = {p.player_id for p in starting_xi}

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


def _choose_captains(starting_xi: List[FreeHitPlayer]) -> tuple[FreeHitPlayer | None, FreeHitPlayer | None]:
    ordered = sorted(
        starting_xi,
        key=lambda x: (x.predicted_points, -x.now_cost, -x.player_id),
        reverse=True,
    )
    captain = ordered[0] if len(ordered) >= 1 else None
    vice_captain = ordered[1] if len(ordered) >= 2 else None
    return captain, vice_captain

def _build_first_gw_points_map(
    horizon_rows: List[dict],
    target_gw: int,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for row in horizon_rows:
        player_id = int(row["player_id"])
        first_gw_pts = 0.0
        for pred in row.get("gw_predictions", []):
            if int(pred["target_gw"]) == target_gw:
                first_gw_pts = float(pred["predicted_points"])
                break
        out[player_id] = first_gw_pts
    return out


def _choose_captains_for_first_gw(
    starting_xi: List[FreeHitPlayer],
    first_gw_points_map: Dict[int, float],
) -> tuple[FreeHitPlayer | None, FreeHitPlayer | None]:
    ordered = sorted(
        starting_xi,
        key=lambda x: (
            first_gw_points_map.get(x.player_id, 0.0),
            x.predicted_points,
            -x.now_cost,
            -x.player_id,
        ),
        reverse=True,
    )
    captain = ordered[0] if len(ordered) >= 1 else None
    vice_captain = ordered[1] if len(ordered) >= 2 else None
    return captain, vice_captain

def _validate_wildcard_result(
    *,
    starting_xi: List[FreeHitPlayer],
    bench: List[FreeHitPlayer],
    captain: FreeHitPlayer | None,
    vice_captain: FreeHitPlayer | None,
    locked_player_ids: List[int],
    budget: float,
    spent_m: float,
) -> List[str]:
    errors: List[str] = []

    squad = starting_xi + bench
    squad_ids = [p.player_id for p in squad]
    starting_ids = [p.player_id for p in starting_xi]
    bench_ids = [p.player_id for p in bench]

    if len(starting_xi) != 11:
        errors.append(f"starting_xi must have 11 players, got {len(starting_xi)}")
    if len(bench) != 4:
        errors.append(f"bench must have 4 players, got {len(bench)}")
    if len(squad) != 15:
        errors.append(f"full squad must have 15 players, got {len(squad)}")

    if len(set(squad_ids)) != 15:
        errors.append("squad contains duplicate player_ids")
    if set(starting_ids) & set(bench_ids):
        errors.append("starting_xi and bench overlap")

    squad_pos_counts = _count_positions(squad)
    starting_pos_counts = _count_positions(starting_xi)

    if squad_pos_counts != SQUAD_RULES:
        errors.append(f"invalid squad composition: got {squad_pos_counts}, expected {SQUAD_RULES}")
    if starting_pos_counts != STARTING_RULES:
        errors.append(f"invalid starting XI formation: got {starting_pos_counts}, expected {STARTING_RULES}")

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

    team_counts: Dict[str, int] = {}
    for p in squad:
        team_counts[p.team_short_name] = team_counts.get(p.team_short_name, 0) + 1
    too_many = {team: n for team, n in team_counts.items() if n > 3}
    if too_many:
        errors.append(f"max 3 players per club violated: {too_many}")

    unavailable = [p.web_name for p in squad if p.status != "a"]
    if unavailable:
        errors.append(f"squad contains non-available players: {unavailable}")

    missing_locked = [pid for pid in locked_player_ids if pid not in set(squad_ids)]
    if missing_locked:
        errors.append(f"locked players missing from squad: {missing_locked}")

    if spent_m - budget > 1e-9:
        errors.append(f"budget exceeded: spent {spent_m}, budget {budget}")

    return errors


def build_wildcard_squad_v1(
    *,
    db,
    target_gw: int,
    horizon: int,
    budget: float,
    model_name: str,
    locked_player_ids: List[int],
    current_squad_player_ids: List[int],
) -> WildcardBuildResponse:
    snapshot = build_wildcard_horizon_snapshot(
        db=db,
        start_gw=target_gw,
        horizon=horizon,
        model_name=model_name,
    )

    all_rows = snapshot["player_features"]
    if not all_rows:
        raise ValueError("No horizon player features found.")

    budget_tenths = int(round(budget * 10))
    buckets = _group_by_position(all_rows)

    row_by_player_id = {row["player_id"]: row for row in all_rows}

    selected_rows: List[dict] = []
    selected_ids: set[int] = set()
    team_counts: Dict[int, int] = {}
    pos_counts: Dict[str, int] = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    rejected_candidates: List[FreeHitRejectedCandidate] = []
    spent_tenths = 0

    # lock required players
    for player_id in locked_player_ids:
        row = row_by_player_id.get(player_id)
        if row is None:
            raise ValueError(f"Locked player_id={player_id} not found in wildcard horizon candidate pool.")
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
            raise ValueError(f"Locked player_id={player_id} makes the wildcard squad infeasible.")

    # greedily complete squad by position
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
                raise ValueError(f"Failed to complete wildcard squad for position {pos}.")

    all_players = [_serialize_horizon_player(row) for row in selected_rows]
    all_players = sorted(
        all_players,
        key=lambda x: (POSITION_ORDER.index(x.position), -x.predicted_points, -x.now_cost, -x.player_id),
    )

    starting_xi, bench = _build_starting_and_bench(all_players)
    first_gw_points_map = _build_first_gw_points_map(all_rows, target_gw)
    captain, vice_captain = _choose_captains_for_first_gw(starting_xi, first_gw_points_map)

    spent_m = spent_tenths / 10.0
    remaining_m = (budget_tenths - spent_tenths) / 10.0
    projected_points_starting_xi_horizon = sum(p.predicted_points for p in starting_xi)
    projected_points_total_15_horizon = sum(p.predicted_points for p in all_players)
    priority_transfers_from_current_squad = build_priority_transfers_from_current_squad(
    current_squad_player_ids=current_squad_player_ids,
    wildcard_players=all_players,
    horizon_rows=all_rows,
)

    validation_errors = _validate_wildcard_result(
        starting_xi=starting_xi,
        bench=bench,
        captain=captain,
        vice_captain=vice_captain,
        locked_player_ids=locked_player_ids,
        budget=budget,
        spent_m=spent_m,
    )
    if validation_errors:
        raise ValueError(f"Built invalid wildcard squad: {validation_errors}")

    return WildcardBuildResponse(
    target_gw=target_gw,
    horizon=horizon,
    budget=budget,
    model_name=model_name,
    locked_player_ids=locked_player_ids,
    current_squad_player_ids=current_squad_player_ids,
    scoring_objective="maximize_horizon_predicted_points_only",
    starting_xi=starting_xi,
    bench=bench,
    captain=captain,
    vice_captain=vice_captain,
    captain_selection_target_gw=target_gw,
    spent_m=round(spent_m, 1),
    remaining_m=round(remaining_m, 1),
    projected_points_starting_xi_horizon=round(projected_points_starting_xi_horizon, 2),
    projected_points_total_15_horizon=round(projected_points_total_15_horizon, 2),
    priority_transfers_from_current_squad=priority_transfers_from_current_squad,
    rejected_candidates=rejected_candidates,
    notes="Wildcard generator v1 using Day45 horizon features with deterministic greedy construction.",
)