from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from app.rules.squad import load_squad_transfer_rules
from ml.artifacts.paths import build_immutable_artifact_key
from ml.artifacts.storage import LocalArtifactStorage
from ml.contracts.opening_squad import (
    OpeningSquadObjectivePolicy,
    evaluate_opening_squad_objective,
    reconcile_objective_evaluation,
)
from ml.contracts.run_metadata import build_run_metadata, provenance_inputs_from_file_metadata
from ml.decision.optimize_opening_squad import (
    base_player_utility,
    file_metadata,
    normalize_projection_rows,
    parse_list_cell,
    projection_records,
    solve_variant,
)
from ml.decision.squad_rules import SquadLegalityEngine


OPTIMIZER_VERSION = "day101b_v1"
ARTIFACT_TYPE = "opening_lineup_optimizer"
ARTIFACT_VERSION = "fpl_opening_lineup_optimizer_v1"
RECOMMENDATION_STATUS = "preview_only"


class OpeningLineupOptimizerError(RuntimeError):
    """Raised when Day101B inputs or outputs are unsafe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise OpeningLineupOptimizerError("%s does not exist: %s" % (label, resolved))
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpeningLineupOptimizerError("Invalid JSON in %s: %s" % (resolved, exc))
    if not isinstance(payload, dict):
        raise OpeningLineupOptimizerError("%s must contain a JSON object." % label)
    return payload


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", "", "none", "nan"}:
        return False
    raise OpeningLineupOptimizerError("Cannot parse boolean value: %r" % value)


def validate_day101a_report(report: Mapping[str, Any]) -> None:
    blockers: List[str] = []
    checks = [
        (report.get("artifact_type") == "opening_squad_optimizer",
         "Day101B requires a Day101A opening_squad_optimizer report."),
        (report.get("passed") is True, "Day101A report must have passed=true."),
        (report.get("ready_for_day101b") is True, "Day101A report must be ready_for_day101b."),
        (report.get("stop_point_satisfied") is True, "Day101A stop point must be satisfied."),
        (report.get("preview_only") is True, "Day101A source must remain preview_only."),
        (report.get("production_approved") is False, "Day101A source must not be production approved."),
        (report.get("writes_database") is False, "Day101A source must not write database state."),
        (report.get("writes_predictions_table") is False, "Day101A source must not write predictions."),
        (report.get("writes_recommendations") is False, "Day101A source must not write recommendations."),
        (report.get("writes_squad_state") is False, "Day101A source must not write squad state."),
        (int(report.get("target_gw") or 0) == 1, "Day101B Fast Lane requires target_gw=1."),
    ]
    for valid, message in checks:
        if not valid:
            blockers.append(message)

    primary = (report.get("variants") or {}).get("primary")
    if not isinstance(primary, Mapping):
        blockers.append("Day101A report is missing variants.primary.")
    else:
        if len(primary.get("selected_player_ids") or []) != 15:
            blockers.append("Day101A Primary must contain exactly 15 players.")
        if ((primary.get("squad_legality") or {}).get("valid") is not True):
            blockers.append("Day101A Primary must be Day100A-legal.")
        if ((primary.get("objective_reconciliation") or {}).get("passed") is not True):
            blockers.append("Day101A Primary objective must reconcile.")
        if not isinstance(primary.get("objective_policy"), Mapping):
            blockers.append("Day101A Primary is missing the Day100B objective policy.")

    scope = report.get("scope_boundary") or {}
    if scope.get("day101b_required") is not True:
        blockers.append("Day101A must explicitly require Day101B.")
    if scope.get("final_lineup_selected") is True:
        blockers.append("Day101A must not already claim a final lineup.")

    if blockers:
        raise OpeningLineupOptimizerError("Unsafe Day101A report: %s" % " | ".join(blockers))


def resolve_day101a_artifact(report: Mapping[str, Any], key_name: str) -> Path:
    artifacts = report.get("artifacts") or {}
    root = artifacts.get("root")
    keys = artifacts.get("keys") or {}
    key = keys.get(key_name)
    if not root or not key:
        raise OpeningLineupOptimizerError("Day101A artifact metadata is missing %s." % key_name)
    path = Path(str(root)).expanduser().resolve() / Path(str(key))
    if not path.is_file():
        raise OpeningLineupOptimizerError("Day101A artifact does not exist: %s" % path)
    return path


def resolve_projection_source(report: Mapping[str, Any]) -> Path:
    metadata = (report.get("source_artifacts") or {}).get("optimizer_projection_rows") or {}
    raw_path = metadata.get("path")
    if not raw_path:
        raise OpeningLineupOptimizerError("Day101A report has no optimizer projection source path.")
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise OpeningLineupOptimizerError("Optimizer projection source does not exist: %s" % path)
    expected_sha = str(metadata.get("sha256") or "")
    if expected_sha and sha256_file(path) != expected_sha:
        raise OpeningLineupOptimizerError("Optimizer projection source fingerprint changed.")
    return path


def normalize_primary_squad(frame: pd.DataFrame, report: Mapping[str, Any]) -> pd.DataFrame:
    required = {
        "variant", "player_id", "player_name", "web_name", "team_id", "team_name",
        "team_short_name", "position", "now_cost", "variant_total_cost_units",
        "variant_bank_units",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise OpeningLineupOptimizerError("opening_squad.csv missing columns: %s" % missing)

    primary = frame[frame["variant"].astype(str) == "primary"].copy()
    if len(primary) != 15:
        raise OpeningLineupOptimizerError("Day101A Primary CSV must contain exactly 15 rows.")

    for column in ("player_id", "team_id", "now_cost"):
        primary[column] = pd.to_numeric(primary[column], errors="coerce")
    if primary[["player_id", "team_id", "now_cost"]].isna().any().any():
        raise OpeningLineupOptimizerError("Primary squad has missing identity/price values.")
    primary["player_id"] = primary["player_id"].astype(int)
    primary["team_id"] = primary["team_id"].astype(int)
    primary["now_cost"] = primary["now_cost"].astype(int)
    primary["position"] = primary["position"].astype(str).str.upper().str.strip()

    if primary["player_id"].duplicated().any():
        raise OpeningLineupOptimizerError("Primary squad has duplicate player IDs.")

    expected = set(int(v) for v in report["variants"]["primary"]["selected_player_ids"])
    actual = set(primary["player_id"].tolist())
    if expected != actual:
        raise OpeningLineupOptimizerError("Primary CSV IDs do not match Day101A report.")

    reported_cost = int(report["variants"]["primary"]["total_cost_units"])
    reported_bank = int(report["variants"]["primary"]["bank_units"])
    if int(primary["now_cost"].sum()) != reported_cost:
        raise OpeningLineupOptimizerError("Primary squad cost does not reconcile.")
    if int(primary["variant_total_cost_units"].iloc[0]) != reported_cost:
        raise OpeningLineupOptimizerError("Primary CSV reported cost does not reconcile.")
    if int(primary["variant_bank_units"].iloc[0]) != reported_bank:
        raise OpeningLineupOptimizerError("Primary CSV bank does not reconcile.")
    return primary.sort_values("player_id").reset_index(drop=True)


def select_primary_projections(frame: pd.DataFrame, squad: pd.DataFrame) -> pd.DataFrame:
    ids = set(int(v) for v in squad["player_id"].tolist())
    selected = frame[(frame["target_gw"] == 1) & frame["player_id"].isin(ids)].copy()
    if len(selected) != 15 or set(selected["player_id"].tolist()) != ids:
        raise OpeningLineupOptimizerError("Missing GW1 projection for a Primary player.")
    if not selected["selection_eligible"].all():
        raise OpeningLineupOptimizerError("Primary contains a selection-ineligible source row.")
    return selected.sort_values("player_id").reset_index(drop=True)


def metadata_for_solver(squad: pd.DataFrame) -> pd.DataFrame:
    return squad[
        ["player_id", "player_name", "web_name", "team_id", "team_name",
         "team_short_name", "position", "now_cost"]
    ].copy()


def utilities(policy: OpeningSquadObjectivePolicy, projections: pd.DataFrame) -> Dict[int, float]:
    return {
        int(row["player_id"]): float(base_player_utility(policy, row))
        for row in projections.to_dict(orient="records")
    }


def projection_lookup(projections: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    return {
        int(row["player_id"]): row
        for row in projections.to_dict(orient="records")
    }


def choose_vice(
    starters: Sequence[int],
    captain_id: int,
    projections: pd.DataFrame,
    utility_by_id: Mapping[int, float],
) -> int:
    by_id = projection_lookup(projections)
    candidates = [int(v) for v in starters if int(v) != int(captain_id)]
    if not candidates:
        raise OpeningLineupOptimizerError("No vice-captain candidate exists.")
    return sorted(
        candidates,
        key=lambda pid: (
            -float(by_id[pid]["appearance_probability"]),
            -float(by_id[pid]["start_probability"]),
            -float(by_id[pid]["expected_minutes"]),
            -float(utility_by_id[pid]),
            pid,
        ),
    )[0]


def legality_players(squad: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "player_id": int(row["player_id"]),
            "player_name": str(row.get("player_name") or row.get("web_name") or ""),
            "position": str(row["position"]),
            "club_id": int(row["team_id"]),
            "price_units": int(row["now_cost"]),
            "selection_eligible": True,
            "eligibility_reason": "eligible_day101a_primary",
        }
        for row in squad.to_dict(orient="records")
    ]


def validate_and_evaluate(
    *,
    plan: Mapping[str, Any],
    squad: pd.DataFrame,
    projections: pd.DataFrame,
    policy: OpeningSquadObjectivePolicy,
    engine: SquadLegalityEngine,
    bank_units: int,
) -> Dict[str, Any]:
    legality = engine.validate_plan(
        legality_players(squad),
        starting_player_ids=plan["starting_player_ids"],
        bench_order=plan["bench_order"],
        captain_player_id=plan["captain_player_id"],
        vice_captain_player_id=plan["vice_captain_player_id"],
        declared_bank_units=int(bank_units),
    )
    if not legality["valid"]:
        raise OpeningLineupOptimizerError("Day100A rejected plan: %s" % legality["issues"])

    evaluation = evaluate_opening_squad_objective(
        policy,
        projection_records(projections),
        starting_player_ids=plan["starting_player_ids"],
        bench_order=plan["bench_order"],
        captain_player_id=plan["captain_player_id"],
        vice_captain_player_id=plan["vice_captain_player_id"],
        bank_units=int(bank_units),
    )
    reconciliation = reconcile_objective_evaluation(evaluation)
    if not reconciliation["passed"]:
        raise OpeningLineupOptimizerError("Day100B objective reconciliation failed.")

    result = dict(plan)
    result["formation"] = legality["lineup"]["formation"]
    result["legality"] = legality
    result["objective_evaluation"] = evaluation
    result["objective_reconciliation"] = reconciliation
    return result


def primary_plan(
    *,
    squad: pd.DataFrame,
    projections: pd.DataFrame,
    policy: OpeningSquadObjectivePolicy,
    engine: SquadLegalityEngine,
    bank_units: int,
) -> Dict[str, Any]:
    utility_by_id = utilities(policy, projections)
    solved = solve_variant(
        variant="day101b_primary",
        policy=policy,
        projection_frame=projections,
        metadata=metadata_for_solver(squad),
        rules=engine.rules,
        effective_gameweeks=[1],
    )
    raw = dict(solved["objective_evaluation_plan"])
    raw["vice_captain_player_id"] = choose_vice(
        raw["starting_player_ids"],
        int(raw["captain_player_id"]),
        projections,
        utility_by_id,
    )
    raw["is_final_day101b_decision"] = True
    raw["purpose"] = "Day101B selected GW1 lineup/captaincy preview."
    return validate_and_evaluate(
        plan=raw,
        squad=squad,
        projections=projections,
        policy=policy,
        engine=engine,
        bank_units=bank_units,
    )


def best_bench_order(
    bench_gk: int,
    outfield_ids: Sequence[int],
    starters: Sequence[int],
    captain_id: int,
    projections: pd.DataFrame,
    policy: OpeningSquadObjectivePolicy,
    squad: pd.DataFrame,
    engine: SquadLegalityEngine,
    bank_units: int,
) -> Optional[Dict[str, Any]]:
    utility_by_id = utilities(policy, projections)
    vice_id = choose_vice(starters, captain_id, projections, utility_by_id)
    candidates: List[Dict[str, Any]] = []
    for order in permutations([int(v) for v in outfield_ids]):
        raw = {
            "starting_player_ids": [int(v) for v in starters],
            "bench_order": [int(bench_gk)] + list(order),
            "captain_player_id": int(captain_id),
            "vice_captain_player_id": int(vice_id),
            "is_final_day101b_decision": False,
            "purpose": "Controlled Day101B lineup alternative.",
        }
        try:
            candidates.append(
                validate_and_evaluate(
                    plan=raw,
                    squad=squad,
                    projections=projections,
                    policy=policy,
                    engine=engine,
                    bank_units=bank_units,
                )
            )
        except OpeningLineupOptimizerError:
            continue
    if not candidates:
        return None
    return sorted(candidates, key=plan_sort_key)[0]


def plan_sort_key(plan: Mapping[str, Any]) -> Tuple[Any, ...]:
    totals = plan["objective_evaluation"]["totals"]
    return (
        -float(totals["objective_value"]),
        float(totals["total_penalty"]),
        tuple(sorted(int(v) for v in plan["starting_player_ids"])),
        tuple(int(v) for v in plan["bench_order"]),
        int(plan["captain_player_id"]),
        int(plan["vice_captain_player_id"]),
    )


def lineup_alternative(
    *,
    primary: Mapping[str, Any],
    squad: pd.DataFrame,
    projections: pd.DataFrame,
    policy: OpeningSquadObjectivePolicy,
    engine: SquadLegalityEngine,
    bank_units: int,
) -> Dict[str, Any]:
    utility_by_id = utilities(policy, projections)
    squad_pos = {int(r["player_id"]): str(r["position"]) for r in squad.to_dict(orient="records")}
    primary_starters = [int(v) for v in primary["starting_player_ids"]]
    primary_bench = [int(v) for v in primary["bench_order"]]
    candidates: List[Dict[str, Any]] = []

    # Goalkeeper swap is always a controlled legal candidate if the source plan is valid.
    starting_gk = next(pid for pid in primary_starters if squad_pos[pid] == "GKP")
    bench_gk = primary_bench[0]
    gk_swap_starters = [pid for pid in primary_starters if pid != starting_gk] + [bench_gk]
    gk_plan = best_bench_order(
        starting_gk,
        primary_bench[1:],
        gk_swap_starters,
        max(gk_swap_starters, key=lambda pid: (utility_by_id[pid], -pid)),
        projections,
        policy,
        squad,
        engine,
        bank_units,
    )
    if gk_plan is not None:
        candidates.append(gk_plan)

    # One-for-one outfield starter/bench swaps preserve a close alternative.
    for starter_out in primary_starters:
        if squad_pos[starter_out] == "GKP":
            continue
        for bench_in in primary_bench[1:]:
            starters = [pid for pid in primary_starters if pid != starter_out] + [bench_in]
            outfield_bench = [pid for pid in primary_bench[1:] if pid != bench_in] + [starter_out]
            captain_id = max(starters, key=lambda pid: (utility_by_id[pid], -pid))
            candidate = best_bench_order(
                primary_bench[0],
                outfield_bench,
                starters,
                captain_id,
                projections,
                policy,
                squad,
                engine,
                bank_units,
            )
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        raise OpeningLineupOptimizerError("No legal controlled lineup alternative exists.")
    distinct = [
        plan for plan in candidates
        if set(plan["starting_player_ids"]) != set(primary_starters)
    ]
    if not distinct:
        raise OpeningLineupOptimizerError("Lineup alternative did not change the XI.")
    return sorted(distinct, key=plan_sort_key)[0]


def captaincy_alternative(
    *,
    primary: Mapping[str, Any],
    squad: pd.DataFrame,
    projections: pd.DataFrame,
    policy: OpeningSquadObjectivePolicy,
    engine: SquadLegalityEngine,
    bank_units: int,
) -> Dict[str, Any]:
    utility_by_id = utilities(policy, projections)
    candidates: List[Dict[str, Any]] = []
    for captain_id in primary["starting_player_ids"]:
        captain_id = int(captain_id)
        if captain_id == int(primary["captain_player_id"]):
            continue
        raw = {
            "starting_player_ids": [int(v) for v in primary["starting_player_ids"]],
            "bench_order": [int(v) for v in primary["bench_order"]],
            "captain_player_id": captain_id,
            "vice_captain_player_id": choose_vice(
                primary["starting_player_ids"], captain_id, projections, utility_by_id
            ),
            "is_final_day101b_decision": False,
            "purpose": "Day101B captaincy alternative with Primary XI/bench fixed.",
        }
        candidates.append(
            validate_and_evaluate(
                plan=raw,
                squad=squad,
                projections=projections,
                policy=policy,
                engine=engine,
                bank_units=bank_units,
            )
        )
    if not candidates:
        raise OpeningLineupOptimizerError("No captaincy alternative exists.")
    return sorted(candidates, key=plan_sort_key)[0]


def optimize_complete_lineup(
    *,
    squad: pd.DataFrame,
    projections: pd.DataFrame,
    policy: OpeningSquadObjectivePolicy,
    engine: SquadLegalityEngine,
    bank_units: int,
) -> Dict[str, Dict[str, Any]]:
    primary = primary_plan(
        squad=squad, projections=projections, policy=policy,
        engine=engine, bank_units=bank_units,
    )
    lineup_alt = lineup_alternative(
        primary=primary, squad=squad, projections=projections, policy=policy,
        engine=engine, bank_units=bank_units,
    )
    captain_alt = captaincy_alternative(
        primary=primary, squad=squad, projections=projections, policy=policy,
        engine=engine, bank_units=bank_units,
    )
    return {
        "primary": primary,
        "lineup_alternative": lineup_alt,
        "captaincy_alternative": captain_alt,
    }


def player_lookup(squad: pd.DataFrame, projections: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    pred = projection_lookup(projections)
    result: Dict[int, Dict[str, Any]] = {}
    for row in squad.to_dict(orient="records"):
        pid = int(row["player_id"])
        p = pred[pid]
        result[pid] = {
            "player_id": pid,
            "player_name": str(row.get("player_name") or ""),
            "web_name": str(row.get("web_name") or ""),
            "team_id": int(row["team_id"]),
            "team_name": str(row.get("team_name") or ""),
            "team_short_name": str(row.get("team_short_name") or ""),
            "position": str(row["position"]),
            "now_cost": int(row["now_cost"]),
            "predicted_points": float(p["predicted_points"]),
            "expected_minutes": float(p["expected_minutes"]),
            "start_probability": float(p["start_probability"]),
            "appearance_probability": float(p["appearance_probability"]),
            "fallback_used": bool(p["fallback_used"]),
            "fallback_level": int(p["fallback_level"]),
            "risk_flags": list(p["risk_flags"]),
            "manual_review_required": bool(p["manual_review_required"]),
        }
    return result


def explain_plan(plan: Mapping[str, Any], lookup: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    captain_id = int(plan["captain_player_id"])
    vice_id = int(plan["vice_captain_player_id"])
    captain = lookup[captain_id]
    vice = lookup[vice_id]
    starters = [lookup[int(v)] for v in plan["starting_player_ids"]]
    bench = [lookup[int(v)] for v in plan["bench_order"]]

    starter_decisions = []
    for player in sorted(
        starters,
        key=lambda row: (
            str(row["position"]),
            -float(row["predicted_points"]),
            int(row["player_id"]),
        ),
    ):
        starter_decisions.append({
            "player": dict(player),
            "reason": (
                "Starter in the highest Day100B risk-adjusted legal XI from the fixed "
                "Day101A squad. Expected points are balanced against expected minutes, "
                "start probability, fallback and uncertainty risk."
            ),
        })

    bench_decisions = []
    for index, player in enumerate(bench):
        if index == 0:
            reason = (
                "Substitute goalkeeper: the other goalkeeper wins the starting-GK role "
                "under the same risk-adjusted objective."
            )
        else:
            reason = (
                "Outfield substitute %s: ordered by Day100B bench-role utility; earlier "
                "bench slots carry more weight while preserving a legal XI." % index
            )
        bench_decisions.append({"slot": index, "player": dict(player), "reason": reason})

    return {
        "formation_reason": (
            "Formation emerges from the highest Day100B risk-adjusted legal role assignment; "
            "it is not hardcoded."
        ),
        "starter_decisions": starter_decisions,
        "bench_decisions": bench_decisions,
        "captain_reason": (
            "Captain maximizes the Day100B captain-bonus utility within the selected XI, "
            "using expected points plus the existing minutes/start, fallback and uncertainty risk treatment."
        ),
        "vice_reason": (
            "Vice is contingency-only in Day100B, so Day101B ranks non-captain starters by "
            "appearance probability, start probability, expected minutes, then risk-adjusted utility."
        ),
        "captain": dict(captain),
        "vice_captain": dict(vice),
        "bench": [dict(player) for player in bench],
    }


def plan_summary(plan: Mapping[str, Any], lookup: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "formation": plan["formation"],
        "starting_player_ids": [int(v) for v in plan["starting_player_ids"]],
        "bench_order": [int(v) for v in plan["bench_order"]],
        "bench_goalkeeper_id": int(plan["bench_order"][0]),
        "ordered_outfield_bench_ids": [int(v) for v in plan["bench_order"][1:]],
        "captain_player_id": int(plan["captain_player_id"]),
        "vice_captain_player_id": int(plan["vice_captain_player_id"]),
        "objective_value": float(plan["objective_evaluation"]["totals"]["objective_value"]),
        "gross_expected_points": float(plan["objective_evaluation"]["totals"]["gross_expected_points"]),
        "total_penalty": float(plan["objective_evaluation"]["totals"]["total_penalty"]),
        "manual_review_required": bool(plan["objective_evaluation"]["manual_review_required"]),
        "manual_review_reasons": list(plan["objective_evaluation"]["manual_review_reasons"]),
        "legality": {
            "valid": bool(plan["legality"]["valid"]),
            "issue_codes": list(plan["legality"]["issue_codes"]),
            "component_validity": dict(plan["legality"]["component_validity"]),
        },
        "objective_reconciliation": dict(plan["objective_reconciliation"]),
        "explanation": explain_plan(plan, lookup),
    }


def lineup_frame(plans: Mapping[str, Mapping[str, Any]], lookup: Mapping[int, Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for plan_name, plan in plans.items():
        roles = {int(pid): "starter" for pid in plan["starting_player_ids"]}
        for index, pid in enumerate(plan["bench_order"]):
            roles[int(pid)] = "bench_gk" if index == 0 else "bench_%s" % index
        for pid in sorted(lookup):
            player = lookup[pid]
            rows.append({
                "plan": plan_name,
                "player_id": pid,
                "player_name": player["player_name"],
                "web_name": player["web_name"],
                "team_short_name": player["team_short_name"],
                "position": player["position"],
                "now_cost": player["now_cost"],
                "role": roles[pid],
                "is_starter": roles[pid] == "starter",
                "is_captain": pid == int(plan["captain_player_id"]),
                "is_vice_captain": pid == int(plan["vice_captain_player_id"]),
                "formation": plan["formation"],
                "predicted_points": player["predicted_points"],
                "expected_minutes": player["expected_minutes"],
                "start_probability": player["start_probability"],
                "appearance_probability": player["appearance_probability"],
                "fallback_used": player["fallback_used"],
                "fallback_level": player["fallback_level"],
                "risk_flags": json.dumps(player["risk_flags"], separators=(",", ":")),
                "manual_review_required": player["manual_review_required"],
                "plan_objective_value": plan["objective_evaluation"]["totals"]["objective_value"],
                "is_day101b_selected_plan": plan_name == "primary",
                "preview_only": True,
            })
    return pd.DataFrame(rows)


def objective_frame(plans: Mapping[str, Mapping[str, Any]], lookup: Mapping[int, Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for plan_name, plan in plans.items():
        for payload in plan["objective_evaluation"]["by_player"].values():
            pid = int(payload["player_id"])
            totals = payload["totals"]
            rows.append({
                "plan": plan_name,
                "player_id": pid,
                "web_name": lookup[pid]["web_name"],
                "position": lookup[pid]["position"],
                "role": payload["role"],
                "is_captain": bool(payload["is_captain"]),
                "is_vice_captain": bool(payload["is_vice_captain"]),
                "gross_expected_points": totals["gross_expected_points"],
                "minutes_risk_penalty": totals["minutes_risk_penalty"],
                "start_risk_penalty": totals["start_risk_penalty"],
                "fallback_penalty": totals["fallback_penalty"],
                "uncertainty_penalty": totals["uncertainty_penalty"],
                "objective_value": totals["objective_value"],
            })
    return pd.DataFrame(rows)


def build_run_metadata_payload(
    source_report: Mapping[str, Any],
    source_metadata: Mapping[str, Mapping[str, Any]],
    policy: OpeningSquadObjectivePolicy,
    rules: Any,
) -> Dict[str, Any]:
    source_run = source_report.get("run_metadata") or {}
    versions = source_run.get("versions") or {}
    rules_versions = dict(versions.get("rules_versions") or {})
    rules_versions["squad_transfer"] = rules.rules_version
    parent = str(source_run.get("run_id") or "")
    return build_run_metadata(
        run_id=None,
        run_type="optimization",
        artifact_type=ARTIFACT_TYPE,
        source_seasons=list(source_run.get("source_seasons") or ["2025_26"]),
        target_season=str(source_report["target_season"]),
        target_gw=1,
        horizon=int(source_report["effective_horizon"]),
        as_of_time=str(source_report["as_of_time_utc"]),
        prediction_mode="pre_gw1_prior",
        created_at=utc_now(),
        model_version=str(versions.get("model_version") or "pre_gw1_heuristic_preview"),
        rules_versions=rules_versions,
        manifest_version=OPTIMIZER_VERSION,
        artifact_version=ARTIFACT_VERSION,
        additional_versions={
            "lineup_optimizer_version": OPTIMIZER_VERSION,
            "source_day101a_optimizer_version": str(source_report.get("optimizer_version") or "day101a_v1"),
            "objective_contract_version": policy.contract_version,
            "objective_policy_version": policy.policy_version,
            "objective_mode": str(source_report["objective_mode"]),
        },
        provenance={
            "producer": "ml.decision.optimize_opening_lineup",
            "inputs": provenance_inputs_from_file_metadata(source_metadata),
            "parent_run_ids": [parent] if parent else [],
            "notes": [
                "Day101B consumes the fixed Day101A Primary squad.",
                "Day100A re-validates Primary and both alternatives.",
                "Day100B supplies the role-weighted risk-adjusted objective.",
                "Official availability is handled upstream in Day76D/Day97A; Day101B consumes the resulting adjusted projections and does not apply a second availability penalty.",
                "All outputs remain preview_only and write-disabled.",
            ],
        },
    ).to_dict()


def build_report(
    source_report: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    plans: Mapping[str, Mapping[str, Any]],
    lookup: Mapping[int, Mapping[str, Any]],
    policy: OpeningSquadObjectivePolicy,
    engine: SquadLegalityEngine,
    source_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    summaries = {name: plan_summary(plan, lookup) for name, plan in plans.items()}
    blockers: List[str] = []
    for name, summary in summaries.items():
        if not summary["legality"]["valid"]:
            blockers.append("%s failed Day100A legality." % name)
        if not summary["objective_reconciliation"]["passed"]:
            blockers.append("%s failed Day100B reconciliation." % name)
    if set(summaries["primary"]["starting_player_ids"]) == set(summaries["lineup_alternative"]["starting_player_ids"]):
        blockers.append("Lineup alternative is not distinct.")
    if summaries["primary"]["captain_player_id"] == summaries["captaincy_alternative"]["captain_player_id"]:
        blockers.append("Captaincy alternative is not distinct.")
    passed = not blockers
    return {
        "created_at_utc": utc_now(),
        "artifact_type": ARTIFACT_TYPE,
        "optimizer_version": OPTIMIZER_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "target_season": source_report["target_season"],
        "target_gw": 1,
        "requested_horizon": int(source_report["requested_horizon"]),
        "effective_horizon": int(source_report["effective_horizon"]),
        "objective_mode": source_report["objective_mode"],
        "as_of_time_utc": source_report["as_of_time_utc"],
        "recommendation_status": RECOMMENDATION_STATUS,
        "preview_only": True,
        "audit_only": True,
        "production_approved": False,
        "writes_database": False,
        "writes_predictions_table": False,
        "writes_recommendations": False,
        "writes_squad_state": False,
        "source_day101a_run_id": str((source_report.get("run_metadata") or {}).get("run_id") or ""),
        "source_day97a_run_id": str(source_report.get("source_day97a_run_id") or ""),
        "run_metadata": dict(run_metadata),
        "source_artifacts": dict(source_metadata),
        "rules": engine.rules_metadata(),
        "objective_policy": policy.to_dict(),
        "fixed_squad": {
            "selected_player_ids": [int(v) for v in source_report["variants"]["primary"]["selected_player_ids"]],
            "total_cost_units": int(source_report["variants"]["primary"]["total_cost_units"]),
            "bank_units": int(source_report["variants"]["primary"]["bank_units"]),
            "membership_changed_by_day101b": False,
        },
        "plans": summaries,
        "availability_scope": {
            "expected_minutes_used": True,
            "start_probability_used": True,
            "appearance_probability_used_for_vice_and_secondary_review": True,
            "fallback_uncertainty_and_risk_flags_preserved": True,
            "official_chance_of_playing_next_round_propagated": True,
            "official_availability_consumed_via_adjusted_projection_inputs": True,
            "day101b_additional_official_availability_penalty_applied": False,
            "required_follow_up": None,
        },
        "scope_boundary": {
            "squad_membership_changed": False,
            "starting_xi_selected": True,
            "formation_selected": True,
            "bench_order_selected": True,
            "captain_selected": True,
            "vice_captain_selected": True,
            "captaincy_alternative_preserved": True,
            "lineup_alternative_preserved": True,
            "final_pre_deadline_snapshot_frozen": False,
        },
        "warnings": [
            "This is a pre-deadline preview, not the final frozen GW1 Model Team snapshot.",
            "GW2-GW5 player predictions remain unavailable in gw1_only_fallback mode.",
        ],
        "passed": passed,
        "ready_for_day101c": passed,
        "stop_point_satisfied": passed,
        "blockers": blockers,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    primary = report["plans"]["primary"]
    exp = primary["explanation"]
    lines = [
        "# Day101B — Lineup, Bench, Captain, and Vice Optimization",
        "",
        "- Run ID: `%s`" % report["run_metadata"]["run_id"],
        "- Formation: `%s`" % primary["formation"],
        "- Captain: `%s`" % exp["captain"]["web_name"],
        "- Vice-captain: `%s`" % exp["vice_captain"]["web_name"],
        "- Bench GK: `%s`" % exp["bench"][0]["web_name"],
        "- Outfield bench: `%s → %s → %s`" % tuple(item["web_name"] for item in exp["bench"][1:]),
        "- Objective value: `%s`" % primary["objective_value"],
        "- Day100A legal: `%s`" % primary["legality"]["valid"],
        "- Day100B reconciled: `%s`" % primary["objective_reconciliation"]["passed"],
        "",
        "## Alternatives",
        "",
        "- Lineup alternative formation: `%s`" % report["plans"]["lineup_alternative"]["formation"],
        "- Captaincy alternative captain ID: `%s`" % report["plans"]["captaincy_alternative"]["captain_player_id"],
        "",
        "## Availability disclosure",
        "",
        "> Expected minutes, start probability, appearance probability, eligibility, fallback, uncertainty, and risk flags are used/preserved. Official `chance_of_playing_next_round` is propagated upstream before Day101B; Day101B consumes the adjusted projection values and does not apply an additional availability penalty.",
        "",
        "## Stop point",
        "",
        "> The opening squad includes a complete legal GW1 lineup, bench order, captain, and vice-captain.",
        "",
        "Stop point satisfied: `%s`" % report["stop_point_satisfied"],
        "Ready for Day101C: `%s`" % report["ready_for_day101c"],
    ]
    return "\n".join(lines)


def write_outputs(
    artifact_root: Path,
    source_report: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    lineup_csv: pd.DataFrame,
    objective_csv: pd.DataFrame,
    report: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    storage = LocalArtifactStorage(artifact_root)
    defs = {
        "opening_lineup_csv": ("opening_lineup", "csv"),
        "opening_lineup_objective_breakdown_csv": ("opening_lineup_objective_breakdown", "csv"),
        "run_metadata_json": ("run_metadata", "json"),
        "opening_lineup_report_json": ("opening_lineup_report", "json"),
        "opening_lineup_report_md": ("opening_lineup_report", "md"),
    }
    keys = {
        name: build_immutable_artifact_key(
            artifact_type=ARTIFACT_TYPE,
            season=str(source_report["target_season"]),
            target_gw=1,
            as_of_time=str(source_report["as_of_time_utc"]),
            run_id=str(run_metadata["run_id"]),
            version=OPTIMIZER_VERSION,
            filename=filename,
            extension=extension,
        )
        for name, (filename, extension) in defs.items()
    }
    payloads = {
        "opening_lineup_csv": lineup_csv.to_csv(index=False),
        "opening_lineup_objective_breakdown_csv": objective_csv.to_csv(index=False),
        "run_metadata_json": json.dumps(run_metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    }
    stored: Dict[str, Dict[str, Any]] = {}
    for name, content in payloads.items():
        stored[name] = storage.write_immutable_text(keys[name], content).to_dict()
    report["artifacts"] = {"root": str(artifact_root.expanduser().resolve()), "keys": keys}
    for name, content in {
        "opening_lineup_report_json": json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "opening_lineup_report_md": build_markdown_report(report),
    }.items():
        stored[name] = storage.write_immutable_text(keys[name], content).to_dict()
    return stored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day101B fixed-squad GW1 lineup optimizer.")
    parser.add_argument("--opening-squad-report-json", required=True)
    parser.add_argument("--target-season", default="2026_27")
    parser.add_argument("--artifact-root", default="/private/tmp/fpl-artifacts")
    parser.add_argument("--squad-rules-config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.opening_squad_report_json).expanduser().resolve()
    source_report = load_json(report_path, "Day101A opening squad report")
    validate_day101a_report(source_report)
    if str(source_report["target_season"]) != str(args.target_season):
        raise OpeningLineupOptimizerError("Target season mismatch.")

    squad_path = resolve_day101a_artifact(source_report, "opening_squad_csv")
    projection_path = resolve_projection_source(source_report)
    squad = normalize_primary_squad(pd.read_csv(squad_path), source_report)
    projections = select_primary_projections(
        normalize_projection_rows(pd.read_csv(projection_path)),
        squad,
    )
    policy = OpeningSquadObjectivePolicy.from_mapping(
        source_report["variants"]["primary"]["objective_policy"]
    )
    rules = load_squad_transfer_rules(
        args.target_season,
        config_path=None if args.squad_rules_config is None else Path(args.squad_rules_config).expanduser().resolve(),
    )
    engine = SquadLegalityEngine(rules)
    bank_units = int(source_report["variants"]["primary"]["bank_units"])

    plans = optimize_complete_lineup(
        squad=squad, projections=projections, policy=policy,
        engine=engine, bank_units=bank_units,
    )
    lookup = player_lookup(squad, projections)

    source_run_id = str((source_report.get("run_metadata") or {}).get("run_id") or "")
    source_metadata = {
        "day101a_report": file_metadata(
            report_path, artifact_type="opening_squad_optimizer",
            run_id=source_run_id, version=str(source_report.get("optimizer_version") or "day101a_v1"),
        ),
        "day101a_opening_squad": file_metadata(
            squad_path, artifact_type="opening_squad_optimizer",
            run_id=source_run_id, version=str(source_report.get("optimizer_version") or "day101a_v1"),
        ),
        "day97a_optimizer_projection_rows": file_metadata(
            projection_path, artifact_type="player_prediction_horizon",
            run_id=str(source_report.get("source_day97a_run_id") or ""),
            version=str(((source_report.get("source_artifacts") or {}).get("optimizer_projection_rows") or {}).get("version") or "day97a_v1"),
        ),
    }
    run_metadata = build_run_metadata_payload(source_report, source_metadata, policy, rules)
    report = build_report(source_report, run_metadata, plans, lookup, policy, engine, source_metadata)
    if not report["passed"]:
        raise OpeningLineupOptimizerError("Day101B failed: %s" % report["blockers"])

    stored = write_outputs(
        Path(args.artifact_root),
        source_report,
        run_metadata,
        lineup_frame(plans, lookup),
        objective_frame(plans, lookup),
        report,
    )

    primary = report["plans"]["primary"]
    print("Day101B opening lineup optimizer complete.")
    print("run_id:", run_metadata["run_id"])
    print("formation:", primary["formation"])
    print("starting_player_ids:", primary["starting_player_ids"])
    print("bench_order:", primary["bench_order"])
    print("captain_player_id:", primary["captain_player_id"])
    print("vice_captain_player_id:", primary["vice_captain_player_id"])
    print("objective_value:", primary["objective_value"])
    print("immutable_artifacts:", len(stored))
    print("preview_only:", str(report["preview_only"]).lower())
    print("writes_database:", str(report["writes_database"]).lower())
    print("writes_squad_state:", str(report["writes_squad_state"]).lower())
    print("ready_for_day101c:", str(report["ready_for_day101c"]).lower())
    print("stop_point_satisfied:", str(report["stop_point_satisfied"]).lower())


if __name__ == "__main__":
    main()
