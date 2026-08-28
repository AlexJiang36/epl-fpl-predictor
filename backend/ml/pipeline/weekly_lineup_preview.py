from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from app.rules.squad import load_squad_transfer_rules
from ml.contracts.opening_squad import build_default_opening_squad_objective_policy
from ml.decision.optimize_opening_squad import base_player_utility
from ml.decision.squad_rules import SquadLegalityEngine


LINEUP_VERSION = "weekly_lineup_preview_v0_1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a read-only weekly Model Team XI/C/VC/bench preview from the "
            "post-transfer 15-player state and the target-GW prediction artifact."
        )
    )
    p.add_argument("--season", required=True)
    p.add_argument("--target-gw", type=int, required=True)
    p.add_argument("--transfer-state-json", required=True)
    p.add_argument("--prediction-csv", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("JSON root must be an object: %s" % path)
    return payload


def build_compat_projection_rows(
    raw: pd.DataFrame,
    squad_fpl_ids: Sequence[int],
    target_gw: int,
) -> pd.DataFrame:
    frame = raw[raw["fpl_player_id"].isin(list(squad_fpl_ids))].copy()
    if len(frame) != 15:
        raise RuntimeError(
            "Expected exactly 15 prediction rows for the post-transfer squad; found %s."
            % len(frame)
        )
    if set(frame["fpl_player_id"].astype(int)) != set(int(v) for v in squad_fpl_ids):
        raise RuntimeError("Prediction rows do not reconcile to the post-transfer squad.")
    if set(pd.to_numeric(frame["target_gw"], errors="coerce").dropna().astype(int)) != {int(target_gw)}:
        raise RuntimeError("Prediction source is not target_gw=%s only." % target_gw)

    rows: List[Dict[str, Any]] = []
    for src in frame.to_dict(orient="records"):
        blended_app = finite_float(
            src.get("blended_appearance_probability"),
            finite_float(src.get("appearance_probability"), 1.0),
        )
        blended_start = finite_float(
            src.get("blended_start_probability"),
            finite_float(src.get("start_probability"), blended_app),
        )
        factor = finite_float(src.get("official_availability_workload_factor"), 1.0)
        factor = max(0.0, min(1.0, factor))

        appearance = max(0.0, min(1.0, blended_app * factor))
        start = max(0.0, min(appearance, blended_start * factor))

        expected_minutes = finite_float(
            src.get("expected_minutes_total"),
            finite_float(src.get("expected_minutes"), 0.0),
        )
        expected_minutes = max(0.0, min(90.0, expected_minutes))

        fallback_used = boolish(src.get("prior_fallback_used", src.get("fallback_used", False)))
        risk_flags: List[str] = []
        if fallback_used:
            risk_flags.append("early_season_prior_fallback")
        if boolish(src.get("official_availability_adjustment_applied")):
            risk_flags.append("official_availability_adjusted_flag")
        status = str(src.get("status") or "").strip().lower()
        if status and status != "a":
            risk_flags.append("official_status_%s" % status)

        rows.append(
            {
                "player_id": int(src["fpl_player_id"]),
                "target_gw": int(target_gw),
                "player_name": str(src.get("player_name") or src.get("web_name") or ""),
                "web_name": str(src.get("web_name") or src.get("player_name") or ""),
                "team_id": int(src["team_id"]),
                "team_name": str(src.get("team_name") or ""),
                "team_short_name": str(src.get("team_short_name") or ""),
                "position": str(src["position"]).upper().strip(),
                "now_cost": int(src["now_cost"]),
                "predicted_points": finite_float(src["predicted_points"]),
                "expected_minutes": expected_minutes,
                "start_probability": start,
                "appearance_probability": appearance,
                "fallback_used": fallback_used,
                "fallback_level": 1 if fallback_used else 0,
                "uncertainty_lower": float("nan"),
                "uncertainty_upper": float("nan"),
                "risk_flags": risk_flags,
                "manual_review_required": bool(risk_flags),
            }
        )

    return pd.DataFrame(rows).sort_values("player_id").reset_index(drop=True)


def choose_vice(
    starters: Sequence[int],
    captain_id: int,
    projection_by_id: Mapping[int, Mapping[str, Any]],
    utility_by_id: Mapping[int, float],
) -> int:
    candidates = [int(v) for v in starters if int(v) != int(captain_id)]
    if not candidates:
        raise RuntimeError("No vice-captain candidate exists.")
    return sorted(
        candidates,
        key=lambda pid: (
            -float(projection_by_id[pid]["appearance_probability"]),
            -float(projection_by_id[pid]["start_probability"]),
            -float(projection_by_id[pid]["expected_minutes"]),
            -float(utility_by_id[pid]),
            pid,
        ),
    )[0]


def rolling_valuation_limit_units(squad: pd.DataFrame, bank_units: int) -> int:
    """Return current squad market value + authoritative rolling bank.

    In-season FPL squad value changes with market prices, so this amount is not
    required to remain equal to the original 100.0m opening budget.
    """
    return (
        sum(int(row["now_cost"]) for row in squad.to_dict(orient="records"))
        + int(bank_units)
    )


def legality_players(squad: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "player_id": int(row["player_id"]),
            "player_name": str(row.get("player_name") or row.get("web_name") or ""),
            "position": str(row["position"]),
            "club_id": int(row["team_id"]),
            "price_units": int(row["now_cost"]),
            "selection_eligible": True,
            "eligibility_reason": "weekly_owned_squad",
        }
        for row in squad.to_dict(orient="records")
    ]


def candidate_plan(
    starters: Sequence[int],
    squad: pd.DataFrame,
    prediction_by_id: Mapping[int, Mapping[str, Any]],
    utility_by_id: Mapping[int, float],
    engine: SquadLegalityEngine,
    bank_units: int,
) -> Optional[Dict[str, Any]]:
    squad_ids = [int(v) for v in squad["player_id"].tolist()]
    positions = {
        int(row["player_id"]): str(row["position"])
        for row in squad.to_dict(orient="records")
    }

    starting_ids = [int(v) for v in starters]
    bench_ids = [pid for pid in squad_ids if pid not in set(starting_ids)]
    bench_gks = [pid for pid in bench_ids if positions[pid] == "GKP"]
    if len(bench_gks) != 1:
        return None

    bench_gk = bench_gks[0]
    outfield_bench = [pid for pid in bench_ids if pid != bench_gk]
    if len(outfield_bench) != 3:
        return None

    captain_id = max(
        starting_ids,
        key=lambda pid: (float(prediction_by_id[pid]["predicted_points"]), -pid),
    )
    vice_id = choose_vice(
        starting_ids,
        captain_id,
        prediction_by_id,
        utility_by_id,
    )
    ordered_outfield = sorted(
        outfield_bench,
        key=lambda pid: (-float(utility_by_id[pid]), pid),
    )
    bench_order = [bench_gk] + ordered_outfield

    # In-season FPL squad value can move above/below the original 100.0m budget
    # as player prices change. The rolling transfer state is authoritative for
    # bank accounting; lineup legality should validate structure/formation, not
    # incorrectly require current market value + bank == the original budget.
    rolling_valuation_limit = rolling_valuation_limit_units(squad, bank_units)
    legality = engine.validate_plan(
        legality_players(squad),
        starting_player_ids=starting_ids,
        bench_order=bench_order,
        captain_player_id=captain_id,
        vice_captain_player_id=vice_id,
        declared_bank_units=int(bank_units),
        budget_limit_units=int(rolling_valuation_limit),
    )
    if not legality["valid"]:
        return None

    xi_points = sum(
        float(prediction_by_id[pid]["predicted_points"])
        for pid in starting_ids
    )
    captain_bonus = float(prediction_by_id[captain_id]["predicted_points"])
    objective = xi_points + captain_bonus

    return {
        "starting_player_ids": starting_ids,
        "bench_order": bench_order,
        "captain_player_id": captain_id,
        "vice_captain_player_id": vice_id,
        "formation": legality["lineup"]["formation"],
        "starting_xi_predicted_points": xi_points,
        "captain_bonus_predicted_points": captain_bonus,
        "objective_points": objective,
        "legality": legality,
    }


def solve_lineup(
    squad: pd.DataFrame,
    projections: pd.DataFrame,
    season: str,
    bank_units: int,
) -> Dict[str, Any]:
    policy = build_default_opening_squad_objective_policy(
        target_season=season,
        horizon_mode="gw1_gw5",
    )
    rules = load_squad_transfer_rules(season)
    engine = SquadLegalityEngine(rules)

    prediction_by_id = {
        int(row["player_id"]): row
        for row in projections.to_dict(orient="records")
    }
    utility_by_id = {
        int(row["player_id"]): float(base_player_utility(policy, row))
        for row in projections.to_dict(orient="records")
    }

    ids = [int(v) for v in squad["player_id"].tolist()]
    plans: List[Dict[str, Any]] = []
    for starters in combinations(ids, 11):
        plan = candidate_plan(
            starters=starters,
            squad=squad,
            prediction_by_id=prediction_by_id,
            utility_by_id=utility_by_id,
            engine=engine,
            bank_units=bank_units,
        )
        if plan is not None:
            plans.append(plan)

    if not plans:
        raise RuntimeError("No legal weekly lineup exists for the 15-player squad.")

    plans.sort(
        key=lambda p: (
            -float(p["objective_points"]),
            tuple(sorted(int(v) for v in p["starting_player_ids"])),
            tuple(int(v) for v in p["bench_order"]),
            int(p["captain_player_id"]),
            int(p["vice_captain_player_id"]),
        )
    )
    best = plans[0]
    best["legal_candidate_count"] = len(plans)
    best["utility_by_id"] = utility_by_id
    return best


def write_outputs(
    out_dir: Path,
    season: str,
    target_gw: int,
    state_path: Path,
    prediction_path: Path,
    state: Mapping[str, Any],
    projections: pd.DataFrame,
    plan: Mapping[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {
        int(row["player_id"]): row
        for row in projections.to_dict(orient="records")
    }
    utility_by_id = plan["utility_by_id"]

    def player_payload(pid: int) -> Dict[str, Any]:
        row = by_id[int(pid)]
        return {
            "fpl_player_id": int(pid),
            "web_name": str(row["web_name"]),
            "team_name": str(row["team_name"]),
            "position": str(row["position"]),
            "now_cost": int(row["now_cost"]),
            "predicted_points": float(row["predicted_points"]),
            "expected_minutes": float(row["expected_minutes"]),
            "appearance_probability": float(row["appearance_probability"]),
            "start_probability": float(row["start_probability"]),
            "utility": float(utility_by_id[int(pid)]),
            "risk_flags": list(row["risk_flags"]),
        }

    payload = {
        "artifact_type": "weekly_lineup_preview",
        "artifact_version": LINEUP_VERSION,
        "season": season,
        "target_gw": int(target_gw),
        "transfer_state_json": str(state_path),
        "prediction_csv": str(prediction_path),
        "bank_units": int(state.get("bank_units", 0)),
        "formation": plan["formation"],
        "starting_player_ids": [int(v) for v in plan["starting_player_ids"]],
        "bench_order": [int(v) for v in plan["bench_order"]],
        "captain_player_id": int(plan["captain_player_id"]),
        "vice_captain_player_id": int(plan["vice_captain_player_id"]),
        "starting_xi_predicted_points": float(plan["starting_xi_predicted_points"]),
        "captain_bonus_predicted_points": float(plan["captain_bonus_predicted_points"]),
        "objective_points": float(plan["objective_points"]),
        "legal_candidate_count": int(plan["legal_candidate_count"]),
        "starting_xi": [player_payload(pid) for pid in plan["starting_player_ids"]],
        "bench": [player_payload(pid) for pid in plan["bench_order"]],
        "captain": player_payload(int(plan["captain_player_id"])),
        "vice_captain": player_payload(int(plan["vice_captain_player_id"])),
        "legal": bool(plan["legality"]["valid"]),
        "preview_only": True,
        "writes_database": False,
        "final_deadline_freeze": False,
        "notes": [
            "Weekly XI and captain maximize target-GW predicted-points objective.",
            "Vice-captain uses appearance, start, minutes, then utility ordering.",
            "Outfield bench is ordered by the existing Day100B/Day101B player utility.",
            "Official availability is consumed through the early-season workload factor when present.",
        ],
    }

    (out_dir / "weekly_lineup_preview.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    rows: List[Dict[str, Any]] = []
    for role, ids in (
        ("starter", plan["starting_player_ids"]),
        ("bench", plan["bench_order"]),
    ):
        for order, pid in enumerate(ids, start=1):
            item = player_payload(int(pid))
            item.update(
                {
                    "role": role,
                    "role_order": order,
                    "is_captain": int(pid) == int(plan["captain_player_id"]),
                    "is_vice_captain": int(pid) == int(plan["vice_captain_player_id"]),
                }
            )
            rows.append(item)

    pd.DataFrame(rows).to_csv(out_dir / "weekly_lineup_preview.csv", index=False)

    starters_by_pos: Dict[str, List[str]] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for pid in plan["starting_player_ids"]:
        row = by_id[int(pid)]
        starters_by_pos[str(row["position"])].append(str(row["web_name"]))

    bench_names = [str(by_id[int(pid)]["web_name"]) for pid in plan["bench_order"]]
    captain_name = str(by_id[int(plan["captain_player_id"])]["web_name"])
    vice_name = str(by_id[int(plan["vice_captain_player_id"])]["web_name"])

    md = [
        "# Weekly Model Team Lineup Preview",
        "",
        "- Season: `%s`" % season,
        "- Target GW: `%s`" % target_gw,
        "- Formation: **%s**" % plan["formation"],
        "- Captain: **%s**" % captain_name,
        "- Vice-captain: **%s**" % vice_name,
        "- XI predicted points: **%.3f**" % float(plan["starting_xi_predicted_points"]),
        "- XI + captain objective: **%.3f**" % float(plan["objective_points"]),
        "- Legal candidate lineups checked: **%s**" % plan["legal_candidate_count"],
        "",
        "## Starting XI",
        "",
    ]
    for pos in ("GKP", "DEF", "MID", "FWD"):
        md.append("- %s: %s" % (pos, ", ".join(starters_by_pos[pos])))
    md.extend(
        [
            "",
            "## Bench",
            "",
            "- GK: %s" % bench_names[0],
            "- 1: %s" % bench_names[1],
            "- 2: %s" % bench_names[2],
            "- 3: %s" % bench_names[3],
            "",
            "> PREVIEW ONLY. This file does not freeze a deadline decision and does not write the database.",
            "",
        ]
    )
    (out_dir / "weekly_lineup_preview.md").write_text(
        "\n".join(md),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    state_path = Path(args.transfer_state_json).expanduser().resolve()
    prediction_path = Path(args.prediction_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    state = read_json(state_path)
    raw_squad = state.get("squad")
    if not isinstance(raw_squad, list) or len(raw_squad) != 15:
        raise RuntimeError("Transfer state must contain a 15-player squad list.")

    squad_rows: List[Dict[str, Any]] = []
    for item in raw_squad:
        squad_rows.append(
            {
                "player_id": int(item["fpl_player_id"]),
                "player_name": str(item.get("web_name") or ""),
                "web_name": str(item.get("web_name") or ""),
                "team_id": int(item["team_id"]),
                "team_name": str(item.get("team_name") or ""),
                "team_short_name": str(item.get("team_short_name") or ""),
                "position": str(item["position"]).upper(),
                "now_cost": int(item["current_price_units"]),
            }
        )
    squad = pd.DataFrame(squad_rows)

    raw_predictions = pd.read_csv(prediction_path)
    projections = build_compat_projection_rows(
        raw_predictions,
        squad["player_id"].tolist(),
        args.target_gw,
    )

    # Enrich team short names from prediction source if transfer-state JSON omits them.
    pred_team_short = {
        int(row["player_id"]): str(row["team_short_name"])
        for row in projections.to_dict(orient="records")
    }
    squad["team_short_name"] = [
        pred_team_short.get(int(pid), "")
        for pid in squad["player_id"].tolist()
    ]

    plan = solve_lineup(
        squad=squad,
        projections=projections,
        season=args.season,
        bank_units=int(state.get("bank_units", 0)),
    )
    write_outputs(
        out_dir=out_dir,
        season=args.season,
        target_gw=args.target_gw,
        state_path=state_path,
        prediction_path=prediction_path,
        state=state,
        projections=projections,
        plan=plan,
    )

    by_id = {
        int(row["player_id"]): row
        for row in projections.to_dict(orient="records")
    }
    print("=== Weekly Model Team Lineup Preview ===")
    print("status: PASS_PREVIEW")
    print("formation:", plan["formation"])
    print("starting_player_ids:", plan["starting_player_ids"])
    print("bench_order:", plan["bench_order"])
    print("captain:", by_id[int(plan["captain_player_id"])]["web_name"])
    print("vice_captain:", by_id[int(plan["vice_captain_player_id"])]["web_name"])
    print("starting_xi_predicted_points:", round(float(plan["starting_xi_predicted_points"]), 6))
    print("objective_points:", round(float(plan["objective_points"]), 6))
    print("output_dir:", out_dir)
    print("final_deadline_freeze: False")


if __name__ == "__main__":
    main()
