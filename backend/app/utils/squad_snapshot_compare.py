from __future__ import annotations

from typing import Dict, List

from app.schemas.squad_snapshot import SquadSnapshot


def compare_squad_snapshots(
    before: SquadSnapshot,
    after: SquadSnapshot,
) -> Dict:
    before_set = set(before.squad_player_ids)
    after_set = set(after.squad_player_ids)

    added_player_ids = sorted(list(after_set - before_set))
    removed_player_ids = sorted(list(before_set - after_set))

    captain_changed = before.captain_player_id != after.captain_player_id
    vice_captain_changed = before.vice_captain_player_id != after.vice_captain_player_id
    bench_order_changed = before.bench_order_player_ids != after.bench_order_player_ids
    bank_delta = after.bank - before.bank
    model_changed = before.model_name != after.model_name
    target_gw_changed = before.target_gw != after.target_gw

    return {
        "summary": {
            "num_added": len(added_player_ids),
            "num_removed": len(removed_player_ids),
            "captain_changed": captain_changed,
            "vice_captain_changed": vice_captain_changed,
            "bench_order_changed": bench_order_changed,
            "bank_delta": bank_delta,
            "model_changed": model_changed,
            "target_gw_changed": target_gw_changed,
        },
        "details": {
            "added_player_ids": added_player_ids,
            "removed_player_ids": removed_player_ids,
            "captain_change": {
                "before": before.captain_player_id,
                "after": after.captain_player_id,
            },
            "vice_captain_change": {
                "before": before.vice_captain_player_id,
                "after": after.vice_captain_player_id,
            },
            "bench_order_change": {
                "before": before.bench_order_player_ids,
                "after": after.bench_order_player_ids,
            },
            "bank_change": {
                "before": before.bank,
                "after": after.bank,
                "delta": bank_delta,
            },
            "model_change": {
                "before": before.model_name,
                "after": after.model_name,
            },
            "target_gw_change": {
                "before": before.target_gw,
                "after": after.target_gw,
            },
        },
    }