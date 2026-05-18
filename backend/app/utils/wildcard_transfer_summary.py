from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from app.schemas.free_hit import FreeHitPlayer
from app.schemas.wildcard import WildcardPriorityTransfer


def build_priority_transfers_from_current_squad(
    *,
    current_squad_player_ids: List[int],
    wildcard_players: List[FreeHitPlayer],
    horizon_rows: List[dict],
) -> List[WildcardPriorityTransfer]:
    if not current_squad_player_ids:
        return []

    row_by_player_id: Dict[int, dict] = {
        int(r["player_id"]): r for r in horizon_rows
    }

    wildcard_ids = {p.player_id for p in wildcard_players}
    current_ids = set(current_squad_player_ids)

    wildcard_only_ids = wildcard_ids - current_ids
    current_only_ids = current_ids - wildcard_ids

    wildcard_only_rows = [
        row_by_player_id[pid]
        for pid in wildcard_only_ids
        if pid in row_by_player_id
    ]
    current_only_rows = [
        row_by_player_id[pid]
        for pid in current_only_ids
        if pid in row_by_player_id
    ]

    wildcard_by_pos: Dict[str, List[dict]] = defaultdict(list)
    current_by_pos: Dict[str, List[dict]] = defaultdict(list)

    for row in wildcard_only_rows:
        wildcard_by_pos[row["position"]].append(row)

    for row in current_only_rows:
        current_by_pos[row["position"]].append(row)

    for pos in wildcard_by_pos:
        wildcard_by_pos[pos] = sorted(
            wildcard_by_pos[pos],
            key=lambda r: float(r["horizon_predicted_points"]),
            reverse=True,
        )

    for pos in current_by_pos:
        current_by_pos[pos] = sorted(
            current_by_pos[pos],
            key=lambda r: float(r["horizon_predicted_points"]),
        )

    suggestions: List[WildcardPriorityTransfer] = []

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        outs = current_by_pos.get(pos, [])
        ins = wildcard_by_pos.get(pos, [])
        pair_n = min(len(outs), len(ins))

        for i in range(pair_n):
            out_row = outs[i]
            in_row = ins[i]
            out_pts = float(out_row["horizon_predicted_points"])
            in_pts = float(in_row["horizon_predicted_points"])

            suggestions.append(
                WildcardPriorityTransfer(
                    out_player_id=int(out_row["player_id"]),
                    out_web_name=str(out_row["web_name"]),
                    out_position=str(out_row["position"]),
                    out_horizon_points=out_pts,
                    in_player_id=int(in_row["player_id"]),
                    in_web_name=str(in_row["web_name"]),
                    in_position=str(in_row["position"]),
                    in_horizon_points=in_pts,
                    horizon_gain=round(in_pts - out_pts, 4),
                )
            )

    suggestions = sorted(
        suggestions,
        key=lambda x: x.horizon_gain,
        reverse=True,
    )

    return suggestions