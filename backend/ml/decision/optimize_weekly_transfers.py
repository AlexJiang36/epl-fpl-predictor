from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from app.rules.squad import load_squad_transfer_rules, validate_squad
from app.rules.transfers import advance_free_transfer_bank, price_transfer_plan


OPTIMIZER_VERSION = "weekly_transfer_optimizer_v0"
OBJECTIVE_VERSION = "gw_points_xi_plus_captain_v0"
PLAYER_MODEL_NAME = "early_season_blend_player_v0"
VALID_POSITIONS = ("GKP", "DEF", "MID", "FWD")


class WeeklyTransferOptimizerError(RuntimeError):
    pass


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-first weekly FPL transfer optimizer. v0 compares ROLL against "
            "all legal one-transfer moves from the previous finalized Model Team."
        )
    )
    parser.add_argument("--season", required=True)
    parser.add_argument("--target-gw", type=int, required=True)
    parser.add_argument("--previous-squad-json", required=True)
    parser.add_argument("--market-csv", required=True)
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument(
        "--available-free-transfers",
        type=int,
        default=None,
        help=(
            "Explicit FT bank for the target-GW deadline. For GW2 only, if omitted, "
            "the optimizer bootstraps from the season weekly-accrual rule."
        ),
    )
    parser.add_argument(
        "--previous-transfer-state-json",
        default=None,
        help=(
            "Optional finalized transfer-state artifact from the previous GW. "
            "When supplied, purchase prices, bank, and FT state come from this artifact."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional explicit output directory. Defaults to private-planning/gw-pre/...",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of ranked one-transfer options to include in the compact CSV.",
    )
    return parser.parse_args()


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise WeeklyTransferOptimizerError("%s must be numeric." % label)
    if not math.isfinite(result):
        raise WeeklyTransferOptimizerError("%s must be finite." % label)
    return result


def int_value(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise WeeklyTransferOptimizerError("%s must be an integer." % label)
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise WeeklyTransferOptimizerError("%s must be an integer." % label)
    return result


def fpl_selling_price_units(purchase_price_units: int, current_price_units: int) -> int:
    purchase = int_value(purchase_price_units, "purchase_price_units")
    current = int_value(current_price_units, "current_price_units")
    if purchase < 0 or current < 0:
        raise WeeklyTransferOptimizerError("Prices must be non-negative.")
    if current <= purchase:
        return current
    return purchase + ((current - purchase) // 2)


def formation_counts() -> List[Tuple[int, int, int]]:
    results: List[Tuple[int, int, int]] = []
    for defenders in range(3, 6):
        for midfielders in range(2, 6):
            for forwards in range(1, 4):
                if defenders + midfielders + forwards == 10:
                    results.append((defenders, midfielders, forwards))
    return results


def best_lineup_for_predicted_points(squad: pd.DataFrame) -> Dict[str, Any]:
    required = {"fpl_player_id", "web_name", "position", "predicted_points"}
    missing = sorted(required - set(squad.columns))
    if missing:
        raise WeeklyTransferOptimizerError("Squad frame missing columns: %s" % missing)
    if len(squad) != 15:
        raise WeeklyTransferOptimizerError("Expected a 15-player squad, got %s." % len(squad))
    if squad["fpl_player_id"].duplicated().any():
        raise WeeklyTransferOptimizerError("Squad contains duplicate fpl_player_id values.")

    frame = squad.copy()
    frame["fpl_player_id"] = pd.to_numeric(frame["fpl_player_id"], errors="raise").astype(int)
    frame["predicted_points"] = pd.to_numeric(frame["predicted_points"], errors="coerce")
    if frame["predicted_points"].isna().any():
        raise WeeklyTransferOptimizerError("Squad contains missing predicted_points.")
    frame["position"] = frame["position"].astype(str).str.upper().str.strip()

    by_position: Dict[str, pd.DataFrame] = {}
    for position in VALID_POSITIONS:
        by_position[position] = (
            frame[frame["position"] == position]
            .sort_values(
                ["predicted_points", "fpl_player_id"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )

    if len(by_position["GKP"]) != 2:
        raise WeeklyTransferOptimizerError("A legal squad must contain two goalkeepers.")

    best: Optional[Dict[str, Any]] = None
    for defenders, midfielders, forwards in formation_counts():
        if len(by_position["DEF"]) < defenders:
            continue
        if len(by_position["MID"]) < midfielders:
            continue
        if len(by_position["FWD"]) < forwards:
            continue

        selected = pd.concat(
            [
                by_position["GKP"].head(1),
                by_position["DEF"].head(defenders),
                by_position["MID"].head(midfielders),
                by_position["FWD"].head(forwards),
            ],
            ignore_index=True,
        )
        xi_points = float(selected["predicted_points"].sum())
        captain_row = selected.sort_values(
            ["predicted_points", "fpl_player_id"],
            ascending=[False, True],
        ).iloc[0]
        captain_bonus = float(captain_row["predicted_points"])
        objective = xi_points + captain_bonus
        starter_ids = [int(v) for v in selected["fpl_player_id"].tolist()]

        candidate = {
            "formation": "1-%s-%s-%s" % (defenders, midfielders, forwards),
            "starting_player_ids": starter_ids,
            "starting_xi_predicted_points": xi_points,
            "captain_player_id": int(captain_row["fpl_player_id"]),
            "captain_name": str(captain_row["web_name"]),
            "captain_bonus_predicted_points": captain_bonus,
            "objective_points_before_transfer_cost": objective,
        }
        if best is None:
            best = candidate
            continue
        if candidate["objective_points_before_transfer_cost"] > best["objective_points_before_transfer_cost"] + 1e-12:
            best = candidate
        elif abs(
            candidate["objective_points_before_transfer_cost"]
            - best["objective_points_before_transfer_cost"]
        ) <= 1e-12:
            if tuple(candidate["starting_player_ids"]) < tuple(best["starting_player_ids"]):
                best = candidate

    if best is None:
        raise WeeklyTransferOptimizerError("No legal starting formation can be constructed.")
    return best


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def validate_prediction_inputs(
    market: pd.DataFrame,
    predictions: pd.DataFrame,
    manifest: Mapping[str, Any],
    season: str,
    target_gw: int,
) -> Dict[str, Any]:
    blockers: List[str] = []
    required_market = {
        "player_id",
        "fpl_player_id",
        "web_name",
        "position",
        "team_id",
        "team_name",
        "team_short_name",
        "now_cost",
        "status",
        "predicted_points",
    }
    required_prediction = {
        "target_season",
        "target_gw",
        "model_name",
        "fpl_player_id",
        "predicted_points",
        "production_ready",
    }
    missing_market = sorted(required_market - set(market.columns))
    missing_prediction = sorted(required_prediction - set(predictions.columns))
    if missing_market:
        blockers.append("market csv missing columns: %s" % missing_market)
    if missing_prediction:
        blockers.append("prediction csv missing columns: %s" % missing_prediction)
    if blockers:
        return {"valid": False, "blockers": blockers}

    for frame, label in ((market, "market"), (predictions, "prediction")):
        if frame["fpl_player_id"].duplicated().any():
            blockers.append("%s contains duplicate fpl_player_id." % label)

    if len(market) != len(predictions):
        blockers.append(
            "market/prediction row count mismatch: %s vs %s" % (len(market), len(predictions))
        )

    prediction_seasons = set(predictions["target_season"].astype(str))
    if prediction_seasons != {season}:
        blockers.append("prediction target_season mismatch: %s" % sorted(prediction_seasons))
    prediction_gws = set(pd.to_numeric(predictions["target_gw"], errors="coerce").dropna().astype(int))
    if prediction_gws != {int(target_gw)}:
        blockers.append("prediction target_gw mismatch: %s" % sorted(prediction_gws))
    prediction_models = set(predictions["model_name"].astype(str))
    if prediction_models != {PLAYER_MODEL_NAME}:
        blockers.append("unexpected prediction model(s): %s" % sorted(prediction_models))

    if str(manifest.get("season")) != season:
        blockers.append("prediction manifest season mismatch.")
    if int(manifest.get("target_gw") or -1) != int(target_gw):
        blockers.append("prediction manifest target_gw mismatch.")
    if manifest.get("status") != "PASS_PREVIEW":
        blockers.append("prediction manifest must have status=PASS_PREVIEW.")

    merged = market[["fpl_player_id", "predicted_points"]].merge(
        predictions[["fpl_player_id", "predicted_points"]],
        on="fpl_player_id",
        how="outer",
        suffixes=("_market", "_prediction"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        blockers.append("market/prediction player universes do not match.")
    both = merged[merged["_merge"] == "both"].copy()
    if not both.empty:
        diff = (
            pd.to_numeric(both["predicted_points_market"], errors="coerce")
            - pd.to_numeric(both["predicted_points_prediction"], errors="coerce")
        ).abs()
        if diff.isna().any() or float(diff.max()) > 1e-9:
            blockers.append("market predicted_points do not exactly match prediction csv.")

    production_ready = normalize_bool_series(predictions["production_ready"])
    preview_only_rows = (
        int(normalize_bool_series(predictions["preview_only"]).sum())
        if "preview_only" in predictions.columns
        else None
    )

    return {
        "valid": len(blockers) == 0,
        "blockers": blockers,
        "market_rows": int(len(market)),
        "prediction_rows": int(len(predictions)),
        "source_preview_production_ready_rows": int(production_ready.sum()),
        "source_preview_not_production_ready_rows": int((~production_ready).sum()),
        "source_preview_only_rows": preview_only_rows,
        "note": (
            "The immutable source prediction CSV is a PREVIEW artifact and may "
            "legitimately carry production_ready=false. The market CSV is expected "
            "to be exported from the canonical published prediction rows and must "
            "match the source preview values exactly."
        ),
    }


def load_previous_state(
    *,
    previous_squad_path: Path,
    previous_transfer_state_path: Optional[Path],
    target_gw: int,
    weekly_accrual: int,
) -> Dict[str, Any]:
    opening = json.loads(previous_squad_path.read_text(encoding="utf-8"))

    if previous_transfer_state_path is not None:
        state = json.loads(previous_transfer_state_path.read_text(encoding="utf-8"))
        if int(state.get("next_gameweek") or -1) != int(target_gw):
            raise WeeklyTransferOptimizerError(
                "previous transfer state next_gameweek does not match target_gw."
            )
        raw_players = list(state.get("squad") or [])
        if len(raw_players) != 15:
            raise WeeklyTransferOptimizerError(
                "previous transfer state must contain a 15-player squad."
            )
        bank_units = int_value(state.get("bank_units"), "bank_units")
        available_free_transfers = int_value(
            state.get("free_transfers_for_next_gameweek"),
            "free_transfers_for_next_gameweek",
        )
        purchase_prices = {
            int_value(p["fpl_player_id"], "fpl_player_id"): int_value(
                p["purchase_price_units"], "purchase_price_units"
            )
            for p in raw_players
        }
        previous_ids = sorted(purchase_prices)
        source_kind = "weekly_transfer_state"
    else:
        if int(target_gw) != 2:
            raise WeeklyTransferOptimizerError(
                "For target_gw>2, --previous-transfer-state-json is required."
            )
        if opening.get("final_pre_deadline_snapshot_frozen") is not True:
            raise WeeklyTransferOptimizerError(
                "GW2 bootstrap requires a frozen GW1 previous-squad artifact."
            )
        primary = opening.get("primary") or {}
        raw_players = list(primary.get("players") or [])
        if len(raw_players) != 15:
            raise WeeklyTransferOptimizerError(
                "GW1 primary opening squad must contain 15 players."
            )
        bank_units = int_value(primary.get("bank_units"), "primary.bank_units")
        purchase_prices = {
            int_value(p["fpl_player_id"], "fpl_player_id"): int_value(
                p["now_cost"], "opening purchase price"
            )
            for p in raw_players
        }
        previous_ids = sorted(purchase_prices)
        available_free_transfers = int(weekly_accrual)
        source_kind = "frozen_gw1_opening_squad"

    return {
        "source_kind": source_kind,
        "bank_units": bank_units,
        "available_free_transfers": available_free_transfers,
        "purchase_price_by_fpl_player_id": purchase_prices,
        "fpl_player_ids": previous_ids,
    }


def structural_legality_players(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "player_id": int(row["fpl_player_id"]),
            "position": str(row["position"]),
            "club_id": int(row["team_id"]),
            "price_units": int(row["now_cost"]),
        }
        for row in frame.to_dict(orient="records")
    ]


def incoming_is_candidate(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or "").strip().lower()
    if status == "u":
        return False
    try:
        points = float(row["predicted_points"])
    except (TypeError, ValueError):
        return False
    return math.isfinite(points)


def build_option_row(
    *,
    action: str,
    outgoing: Optional[Mapping[str, Any]],
    incoming: Optional[Mapping[str, Any]],
    bank_before: int,
    bank_after: int,
    transfer_plan: Mapping[str, Any],
    next_ft_state: Mapping[str, Any],
    lineup: Mapping[str, Any],
    baseline_objective: float,
) -> Dict[str, Any]:
    net_objective = (
        float(lineup["objective_points_before_transfer_cost"])
        - float(transfer_plan["points_cost"])
    )
    return {
        "action": action,
        "transfer_count": int(transfer_plan["transfer_count"]),
        "out_fpl_player_id": None if outgoing is None else int(outgoing["fpl_player_id"]),
        "out_name": None if outgoing is None else str(outgoing["web_name"]),
        "out_position": None if outgoing is None else str(outgoing["position"]),
        "out_current_price_units": None if outgoing is None else int(outgoing["now_cost"]),
        "out_selling_price_units": None if outgoing is None else int(outgoing["selling_price_units"]),
        "out_predicted_points": None if outgoing is None else float(outgoing["predicted_points"]),
        "in_fpl_player_id": None if incoming is None else int(incoming["fpl_player_id"]),
        "in_name": None if incoming is None else str(incoming["web_name"]),
        "in_team": None if incoming is None else str(incoming["team_name"]),
        "in_position": None if incoming is None else str(incoming["position"]),
        "in_price_units": None if incoming is None else int(incoming["now_cost"]),
        "in_predicted_points": None if incoming is None else float(incoming["predicted_points"]),
        "bank_before_units": int(bank_before),
        "bank_after_units": int(bank_after),
        "free_transfers_before": int(transfer_plan["available_free_transfers_before"]),
        "free_transfers_used": int(transfer_plan["free_transfers_used"]),
        "charged_transfers": int(transfer_plan["charged_transfers"]),
        "transfer_hit_points": int(transfer_plan["points_cost"]),
        "free_transfers_next_gameweek": int(next_ft_state["free_transfers_next_gameweek"]),
        "formation": str(lineup["formation"]),
        "starting_xi_predicted_points": float(lineup["starting_xi_predicted_points"]),
        "captain_player_id": int(lineup["captain_player_id"]),
        "captain_name": str(lineup["captain_name"]),
        "captain_bonus_predicted_points": float(lineup["captain_bonus_predicted_points"]),
        "objective_points_before_transfer_cost": float(
            lineup["objective_points_before_transfer_cost"]
        ),
        "net_objective_points": net_objective,
        "net_gain_vs_roll": net_objective - float(baseline_objective),
    }


def optimize_roll_vs_one_transfer(
    *,
    rules: Any,
    current_squad: pd.DataFrame,
    market: pd.DataFrame,
    purchase_price_by_id: Mapping[int, int],
    bank_units: int,
    available_free_transfers: int,
    target_gw: int,
) -> Dict[str, Any]:
    if len(current_squad) != 15:
        raise WeeklyTransferOptimizerError("Current squad must contain 15 players.")
    current = current_squad.copy()
    current["selling_price_units"] = [
        fpl_selling_price_units(
            int(purchase_price_by_id[int(pid)]),
            int(price),
        )
        for pid, price in zip(current["fpl_player_id"], current["now_cost"])
    ]

    structural = validate_squad(
        rules,
        structural_legality_players(current),
        budget_limit_units=1000000,
    )
    if not structural["valid"]:
        raise WeeklyTransferOptimizerError(
            "Previous squad is structurally illegal under target-season rules: %s"
            % structural["errors"]
        )

    baseline_lineup = best_lineup_for_predicted_points(current)
    roll_transfer_plan = price_transfer_plan(
        rules,
        transfer_count=0,
        available_free_transfers=available_free_transfers,
        phase="in_season",
        chip=None,
    )
    roll_next_ft = advance_free_transfer_bank(
        rules,
        free_transfers_after_transfers=int(
            roll_transfer_plan["free_transfers_after_transfers"]
        ),
        completed_gameweek=int(target_gw),
        phase="in_season",
        chip=None,
    )
    baseline_objective = float(
        baseline_lineup["objective_points_before_transfer_cost"]
    )
    options: List[Dict[str, Any]] = [
        build_option_row(
            action="ROLL",
            outgoing=None,
            incoming=None,
            bank_before=bank_units,
            bank_after=bank_units,
            transfer_plan=roll_transfer_plan,
            next_ft_state=roll_next_ft,
            lineup=baseline_lineup,
            baseline_objective=baseline_objective,
        )
    ]

    owned_ids = set(int(v) for v in current["fpl_player_id"].tolist())
    max_per_club = int(rules.squad["max_players_per_club"])

    for outgoing in current.to_dict(orient="records"):
        selling_price = int(outgoing["selling_price_units"])
        max_incoming_price = int(bank_units) + selling_price
        candidates = market[
            (market["position"].astype(str) == str(outgoing["position"]))
            & (~market["fpl_player_id"].astype(int).isin(owned_ids))
            & (pd.to_numeric(market["now_cost"], errors="coerce") <= max_incoming_price)
        ].copy()

        for incoming in candidates.to_dict(orient="records"):
            if not incoming_is_candidate(incoming):
                continue
            new_squad = current[
                current["fpl_player_id"].astype(int) != int(outgoing["fpl_player_id"])
            ].copy()
            incoming_frame = pd.DataFrame([incoming])
            incoming_frame["selling_price_units"] = int(incoming["now_cost"])
            new_squad = pd.concat([new_squad, incoming_frame], ignore_index=True)

            club_counts = new_squad["team_id"].astype(int).value_counts()
            if not club_counts.empty and int(club_counts.max()) > max_per_club:
                continue

            legality = validate_squad(
                rules,
                structural_legality_players(new_squad),
                budget_limit_units=1000000,
            )
            if not legality["valid"]:
                continue

            bank_after = (
                int(bank_units)
                + selling_price
                - int(incoming["now_cost"])
            )
            if bank_after < 0:
                continue

            transfer_plan = price_transfer_plan(
                rules,
                transfer_count=1,
                available_free_transfers=available_free_transfers,
                phase="in_season",
                chip=None,
            )
            next_ft = advance_free_transfer_bank(
                rules,
                free_transfers_after_transfers=int(
                    transfer_plan["free_transfers_after_transfers"]
                ),
                completed_gameweek=int(target_gw),
                phase="in_season",
                chip=None,
            )
            lineup = best_lineup_for_predicted_points(new_squad)
            options.append(
                build_option_row(
                    action="TRANSFER",
                    outgoing=outgoing,
                    incoming=incoming,
                    bank_before=bank_units,
                    bank_after=bank_after,
                    transfer_plan=transfer_plan,
                    next_ft_state=next_ft,
                    lineup=lineup,
                    baseline_objective=baseline_objective,
                )
            )

    ranked = pd.DataFrame(options)
    ranked["_action_tiebreak"] = ranked["action"].map({"ROLL": 0, "TRANSFER": 1}).fillna(9)
    ranked = ranked.sort_values(
        [
            "net_objective_points",
            "bank_after_units",
            "_action_tiebreak",
            "in_fpl_player_id",
            "out_fpl_player_id",
        ],
        ascending=[False, False, True, True, True],
        na_position="last",
    ).drop(columns=["_action_tiebreak"]).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    winner = ranked.iloc[0].to_dict()
    roll = ranked[ranked["action"] == "ROLL"].iloc[0].to_dict()

    return {
        "ranked_options": ranked,
        "winner": winner,
        "roll": roll,
        "baseline_lineup": baseline_lineup,
        "option_count": int(len(ranked)),
        "one_transfer_option_count": int((ranked["transfer_count"] == 1).sum()),
        "future_ft_option_value_monetized": False,
        "lineup_and_captain_are_objective_only": True,
        "final_lineup_selected": False,
        "final_captain_selected": False,
        "objective_scope": (
            "Target-GW starting XI predicted points plus one captain bonus, "
            "minus transfer hit. Future value of carrying an extra FT is shown "
            "as state but is not converted into points in v0."
        ),
    }


def recommended_state_preview(
    *,
    winner: Mapping[str, Any],
    current_squad: pd.DataFrame,
    market: pd.DataFrame,
    purchase_prices: Mapping[int, int],
    target_gw: int,
) -> Dict[str, Any]:
    frame = current_squad.copy()
    new_purchase = {int(k): int(v) for k, v in purchase_prices.items()}

    if str(winner["action"]) == "TRANSFER":
        out_id = int(winner["out_fpl_player_id"])
        in_id = int(winner["in_fpl_player_id"])
        frame = frame[frame["fpl_player_id"].astype(int) != out_id].copy()
        incoming = market[market["fpl_player_id"].astype(int) == in_id].copy()
        if len(incoming) != 1:
            raise WeeklyTransferOptimizerError(
                "Could not resolve recommended incoming player to exactly one market row."
            )
        frame = pd.concat([frame, incoming], ignore_index=True)
        new_purchase.pop(out_id, None)
        new_purchase[in_id] = int(winner["in_price_units"])

    if len(frame) != 15 or frame["fpl_player_id"].astype(int).duplicated().any():
        raise WeeklyTransferOptimizerError(
            "Recommended next-state preview must contain 15 unique players."
        )

    squad_state: List[Dict[str, Any]] = []
    for row in frame.sort_values("fpl_player_id").to_dict(orient="records"):
        pid = int(row["fpl_player_id"])
        squad_state.append(
            {
                "fpl_player_id": pid,
                "web_name": str(row["web_name"]),
                "position": str(row["position"]),
                "team_id": int(row["team_id"]),
                "team_name": str(row["team_name"]),
                "purchase_price_units": int(new_purchase[pid]),
                "current_price_units": int(row["now_cost"]),
            }
        )

    return {
        "state_contract": "weekly_transfer_state_preview_v0",
        "completed_gameweek": int(target_gw),
        "next_gameweek": int(target_gw) + 1,
        "bank_units": int(winner["bank_after_units"]),
        "free_transfers_for_next_gameweek": int(
            winner["free_transfers_next_gameweek"]
        ),
        "squad": squad_state,
        "finalized": False,
        "note": (
            "Preview only. Final weekly squad/transfer freeze must convert this "
            "into the authoritative next-GW state."
        ),
    }

def default_output_dir(season: str, target_gw: int, run_id: str) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    planning_root = repo_root.parent / "private-planning"
    return (
        planning_root
        / "gw-pre"
        / season
        / ("gw%02d" % int(target_gw))
        / "transfer-decision"
        / run_id
    )


def main() -> None:
    args = parse_args()
    previous_squad_path = Path(args.previous_squad_json).expanduser().resolve()
    market_path = Path(args.market_csv).expanduser().resolve()
    prediction_path = Path(args.prediction_csv).expanduser().resolve()
    manifest_path = Path(args.prediction_manifest).expanduser().resolve()
    previous_state_path = (
        None
        if not args.previous_transfer_state_json
        else Path(args.previous_transfer_state_json).expanduser().resolve()
    )

    for path in [
        previous_squad_path,
        market_path,
        prediction_path,
        manifest_path,
    ]:
        if not path.is_file():
            raise WeeklyTransferOptimizerError("Required input file not found: %s" % path)
    if previous_state_path is not None and not previous_state_path.is_file():
        raise WeeklyTransferOptimizerError(
            "previous transfer state file not found: %s" % previous_state_path
        )

    rules = load_squad_transfer_rules(args.season)
    weekly_accrual = int(
        rules.transfers["weekly"]["free_transfers_accrued_per_gameweek"]
    )
    previous_state = load_previous_state(
        previous_squad_path=previous_squad_path,
        previous_transfer_state_path=previous_state_path,
        target_gw=args.target_gw,
        weekly_accrual=weekly_accrual,
    )
    if args.available_free_transfers is not None:
        previous_state["available_free_transfers"] = int(args.available_free_transfers)

    market = pd.read_csv(market_path, low_memory=False)
    predictions = pd.read_csv(prediction_path, low_memory=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    validation = validate_prediction_inputs(
        market=market,
        predictions=predictions,
        manifest=manifest,
        season=args.season,
        target_gw=args.target_gw,
    )
    if not validation["valid"]:
        raise WeeklyTransferOptimizerError(
            "Prediction input validation failed: %s"
            % " | ".join(validation["blockers"])
        )

    for column in ("player_id", "fpl_player_id", "team_id", "now_cost"):
        market[column] = pd.to_numeric(market[column], errors="raise").astype(int)
    market["predicted_points"] = pd.to_numeric(
        market["predicted_points"], errors="raise"
    )
    market["position"] = market["position"].astype(str).str.upper().str.strip()

    previous_ids = set(int(v) for v in previous_state["fpl_player_ids"])
    current_squad = market[market["fpl_player_id"].isin(previous_ids)].copy()
    if len(current_squad) != 15 or set(current_squad["fpl_player_id"]) != previous_ids:
        raise WeeklyTransferOptimizerError(
            "Could not reconcile all 15 previous-squad players to current market."
        )

    optimizer = optimize_roll_vs_one_transfer(
        rules=rules,
        current_squad=current_squad,
        market=market,
        purchase_price_by_id=previous_state["purchase_price_by_fpl_player_id"],
        bank_units=int(previous_state["bank_units"]),
        available_free_transfers=int(previous_state["available_free_transfers"]),
        target_gw=args.target_gw,
    )
    ranked = optimizer["ranked_options"]
    winner = optimizer["winner"]

    run_id = "%s_%s_gw%s_%s" % (
        OPTIMIZER_VERSION,
        args.season,
        args.target_gw,
        utc_now_compact(),
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else default_output_dir(args.season, args.target_gw, run_id)
    )
    if out_dir.exists():
        raise WeeklyTransferOptimizerError("Output directory already exists: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    full_csv = out_dir / "all_roll_and_one_transfer_options.csv"
    compact_csv = out_dir / "top_transfer_options.csv"
    decision_json = out_dir / "transfer_decision_preview.json"
    state_json = out_dir / "next_gameweek_transfer_state_preview.json"
    manifest_json = out_dir / "run_manifest.json"
    summary_md = out_dir / "summary.md"

    ranked.to_csv(full_csv, index=False)
    ranked.head(max(1, int(args.top_n))).to_csv(compact_csv, index=False)

    decision = {
        "artifact_type": "weekly_transfer_decision_preview",
        "artifact_version": OPTIMIZER_VERSION,
        "season": args.season,
        "target_gw": int(args.target_gw),
        "objective_version": OBJECTIVE_VERSION,
        "objective_scope": optimizer["objective_scope"],
        "future_ft_option_value_monetized": False,
        "lineup_and_captain_are_objective_only": True,
        "final_lineup_selected": False,
        "final_captain_selected": False,
        "previous_state_source_kind": previous_state["source_kind"],
        "bank_before_units": int(previous_state["bank_units"]),
        "available_free_transfers_before": int(
            previous_state["available_free_transfers"]
        ),
        "roll_option": optimizer["roll"],
        "recommended_action_by_target_gw_objective": winner,
        "option_count": optimizer["option_count"],
        "one_transfer_option_count": optimizer["one_transfer_option_count"],
        "writes_database": False,
        "final_weekly_transfer_decision": False,
    }
    decision_json.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    state_preview = recommended_state_preview(
        winner=winner,
        current_squad=current_squad,
        market=market,
        purchase_prices=previous_state["purchase_price_by_fpl_player_id"],
        target_gw=args.target_gw,
    )
    state_json.write_text(
        json.dumps(state_preview, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    input_hashes = {
        "previous_squad_json": {
            "path": str(previous_squad_path),
            "sha256": sha256_file(previous_squad_path),
        },
        "market_csv": {
            "path": str(market_path),
            "sha256": sha256_file(market_path),
        },
        "prediction_csv": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
        },
        "prediction_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "rules_registry": {
            "path": str(rules.path),
            "sha256": rules.sha256,
        },
    }
    if previous_state_path is not None:
        input_hashes["previous_transfer_state_json"] = {
            "path": str(previous_state_path),
            "sha256": sha256_file(previous_state_path),
        }

    run_manifest = {
        "run_id": run_id,
        "artifact_type": "weekly_transfer_optimizer_run",
        "optimizer_version": OPTIMIZER_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "season": args.season,
        "target_gw": int(args.target_gw),
        "status": "PASS_PREVIEW",
        "player_model_name": PLAYER_MODEL_NAME,
        "prediction_input_validation": validation,
        "previous_state_source_kind": previous_state["source_kind"],
        "bank_before_units": int(previous_state["bank_units"]),
        "available_free_transfers_before": int(
            previous_state["available_free_transfers"]
        ),
        "option_count": optimizer["option_count"],
        "one_transfer_option_count": optimizer["one_transfer_option_count"],
        "winner_action": winner["action"],
        "winner_net_gain_vs_roll": winner["net_gain_vs_roll"],
        "future_ft_option_value_monetized": False,
        "input_hashes": input_hashes,
        "artifacts": {
            "all_options_csv": str(full_csv),
            "top_options_csv": str(compact_csv),
            "decision_json": str(decision_json),
            "next_state_preview_json": str(state_json),
            "summary_md": str(summary_md),
        },
        "writes_database": False,
        "writes_squad_state": False,
        "final_weekly_transfer_decision": False,
    }
    manifest_json.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if winner["action"] == "ROLL":
        recommendation_line = "ROLL / NO TRANSFER"
    else:
        recommendation_line = "%s -> %s" % (
            winner["out_name"],
            winner["in_name"],
        )

    summary_lines = [
        "# Weekly Transfer Optimizer Preview",
        "",
        "- Season: `%s`" % args.season,
        "- Target GW: `%s`" % args.target_gw,
        "- Status: `PASS_PREVIEW`",
        "- Recommendation by target-GW objective: **%s**" % recommendation_line,
        "- Bank before: `%.1fm`" % (int(previous_state["bank_units"]) / 10.0),
        "- Bank after winner: `%.1fm`" % (int(winner["bank_after_units"]) / 10.0),
        "- Free transfers before: `%s`" % int(previous_state["available_free_transfers"]),
        "- Free transfers next GW if ROLL: `%s`" % int(optimizer["roll"]["free_transfers_next_gameweek"]),
        "- Free transfers next GW if winner: `%s`" % int(winner["free_transfers_next_gameweek"]),
        "- Target-GW objective gain vs ROLL: `%.6f`" % float(winner["net_gain_vs_roll"]),
        "- Winner formation: `%s`" % winner["formation"],
        "- Objective-only provisional captain: `%s`" % winner["captain_name"],
        "",
        "> v0 does not convert the future option value of carrying an extra free transfer into points. "
        "The recommendation is therefore the winner of the target-GW XI + captain objective only.",
        "",
        "## Top 10",
        "",
    ]
    for row in ranked.head(10).to_dict(orient="records"):
        if row["action"] == "ROLL":
            label = "ROLL"
        else:
            label = "%s -> %s" % (row["out_name"], row["in_name"])
        summary_lines.append(
            "%s. `%s` — net gain vs roll `%.6f`, bank after `%.1fm`, FT next `%s`"
            % (
                int(row["rank"]),
                label,
                float(row["net_gain_vs_roll"]),
                int(row["bank_after_units"]) / 10.0,
                int(row["free_transfers_next_gameweek"]),
            )
        )
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("=== Weekly Transfer Optimizer Preview ===")
    print("status: PASS_PREVIEW")
    print("run_id:", run_id)
    print("output_dir:", out_dir)
    print("season:", args.season)
    print("target_gw:", args.target_gw)
    print("previous_state_source_kind:", previous_state["source_kind"])
    print("bank_before_units:", int(previous_state["bank_units"]))
    print("available_free_transfers_before:", int(previous_state["available_free_transfers"]))
    print("option_count:", optimizer["option_count"])
    print("one_transfer_option_count:", optimizer["one_transfer_option_count"])
    print("recommended_action:", recommendation_line)
    print("net_gain_vs_roll:", round(float(winner["net_gain_vs_roll"]), 6))
    print("bank_after_units:", int(winner["bank_after_units"]))
    print("free_transfers_next_gw_if_roll:", int(optimizer["roll"]["free_transfers_next_gameweek"]))
    print("free_transfers_next_gw_if_recommendation:", int(winner["free_transfers_next_gameweek"]))
    print("formation_after:", winner["formation"])
    print("objective_captain_after:", winner["captain_name"])
    print("final_lineup_selected: False")
    print("final_captain_selected: False")
    print("future_ft_option_value_monetized: False")
    print("database_write: False")
    print("final_weekly_transfer_decision: False")


if __name__ == "__main__":
    main()
