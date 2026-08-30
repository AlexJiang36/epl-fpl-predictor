from __future__ import annotations

from dataclasses import asdict, is_dataclass
from itertools import combinations, product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.rules.squad import (
    SquadTransferRules,
    load_squad_transfer_rules,
    validate_lineup,
    validate_squad,
)
from ml.contracts.squad_state import SquadState, squad_state_from_mapping
from ml.decision.free_transfer_ledger import (
    FreeTransferLedgerState,
    TransferFinancialEffect,
    transition_free_transfer_ledger,
)


OPTIMIZER_VERSION = "fpl_single_gw_transfer_optimizer_v1"
OBJECTIVE_VERSION = "single_gw_xi_plus_captain_minus_hits_v1"
VALID_POSITIONS: Tuple[str, ...] = ("GKP", "DEF", "MID", "FWD")


class SingleGWTransferOptimizerError(ValueError):
    """Raised when the single-GW optimizer input or candidate set is invalid."""


def _coerce_frozen_state(value: Any) -> SquadState:
    if isinstance(value, SquadState):
        state = value
    elif isinstance(value, Mapping):
        try:
            state = squad_state_from_mapping(value)
        except Exception as exc:
            raise SingleGWTransferOptimizerError(
                "previous_frozen_state is not a valid canonical SquadState: %s" % exc
            ) from exc
    else:
        raise SingleGWTransferOptimizerError(
            "previous_frozen_state must be a SquadState or canonical mapping."
        )

    if state.state_status != "frozen" or not state.immutable:
        raise SingleGWTransferOptimizerError(
            "Day127A requires the previous owned squad state to be frozen."
        )
    return state


def _require_int(value: Any, label: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SingleGWTransferOptimizerError("%s must be an integer." % label)
    result = int(value)
    if minimum is not None and result < minimum:
        raise SingleGWTransferOptimizerError(
            "%s must be at least %s." % (label, minimum)
        )
    return result


def _player_id(row: Mapping[str, Any], label: str) -> int:
    for key in ("fpl_player_id", "player_id", "id"):
        value = row.get(key)
        if value is not None:
            try:
                result = int(value)
            except (TypeError, ValueError):
                break
            if result > 0:
                return result
    raise SingleGWTransferOptimizerError("%s is missing a valid player ID." % label)


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise SingleGWTransferOptimizerError("%s must be numeric." % label)
    if result != result or result in (float("inf"), float("-inf")):
        raise SingleGWTransferOptimizerError("%s must be finite." % label)
    return result


def _prediction_points(row: Mapping[str, Any], label: str) -> float:
    for key in (
        "predicted_points",
        "adjusted_next_fixture_points",
        "production_player_model_points",
        "target_gw_predicted_points",
    ):
        if row.get(key) is not None:
            return _finite_float(row[key], "%s.%s" % (label, key))
    raise SingleGWTransferOptimizerError(
        "%s is missing target-GW predicted points." % label
    )


def _target_gw_value(row: Mapping[str, Any]) -> Optional[int]:
    for key in ("target_gw", "gw", "gameweek"):
        value = row.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _target_prediction_map(
    rows: Sequence[Mapping[str, Any]],
    target_gw: int,
) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise SingleGWTransferOptimizerError(
                "target_gw_predictions[%s] must be a mapping." % index
            )
        row_gw = _target_gw_value(raw)
        if row_gw is not None and row_gw != target_gw:
            continue

        pid = _player_id(raw, "target_gw_predictions[%s]" % index)
        if pid in result:
            raise SingleGWTransferOptimizerError(
                "Duplicate target-GW prediction for player_id=%s." % pid
            )
        result[pid] = {
            "player_id": pid,
            "predicted_points": _prediction_points(
                raw, "target_gw_predictions[%s]" % index
            ),
            "player_name": (
                raw.get("player_name")
                or raw.get("web_name")
                or raw.get("name")
            ),
        }
    return result


def _rule_player(player: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "player_id": int(player["fpl_player_id"]),
        "position": str(player["position"]),
        "club_id": int(player["club_id"]),
        "price_units": int(player["current_price_units"]),
    }


def _formation_count_options(rules: SquadTransferRules) -> List[Dict[str, int]]:
    bounds = rules.lineup["position_bounds"]
    ranges: List[Sequence[int]] = []
    for position in VALID_POSITIONS:
        position_bounds = bounds[position]
        ranges.append(
            range(
                int(position_bounds["min"]),
                int(position_bounds["max"]) + 1,
            )
        )

    starting_size = int(rules.lineup["starting_size"])
    options: List[Dict[str, int]] = []
    for values in product(*ranges):
        counts = dict(zip(VALID_POSITIONS, values))
        if sum(counts.values()) == starting_size:
            options.append(counts)
    return options


def reoptimize_lineup_and_captain(
    *,
    squad_players: Sequence[Mapping[str, Any]],
    prediction_by_id: Mapping[int, Mapping[str, Any]],
    rules: SquadTransferRules,
) -> Dict[str, Any]:
    """Pure single-GW XI/bench/captain/vice optimization.

    The objective is intentionally the Day127A baseline:
    starting-XI predicted points + one captain bonus.
    """

    if len(squad_players) != int(rules.squad["size"]):
        raise SingleGWTransferOptimizerError(
            "Lineup optimization requires exactly %s owned players."
            % int(rules.squad["size"])
        )

    player_by_id: Dict[int, Dict[str, Any]] = {}
    for raw in squad_players:
        pid = int(raw["fpl_player_id"])
        if pid in player_by_id:
            raise SingleGWTransferOptimizerError(
                "Squad contains duplicate player_id=%s." % pid
            )
        if pid not in prediction_by_id:
            raise SingleGWTransferOptimizerError(
                "Missing target-GW prediction for owned player_id=%s; "
                "missing predictions are never zero-filled." % pid
            )
        player = dict(raw)
        player["predicted_points"] = float(
            prediction_by_id[pid]["predicted_points"]
        )
        player_by_id[pid] = player

    structural = validate_squad(
        rules,
        [_rule_player(player) for player in player_by_id.values()],
        budget_limit_units=(
            sum(
                int(player["current_price_units"])
                for player in player_by_id.values()
            )
            + 1000
        ),
    )
    if not structural["valid"]:
        raise SingleGWTransferOptimizerError(
            "Candidate squad is structurally illegal: %s"
            % structural["errors"]
        )

    by_position: Dict[str, List[Dict[str, Any]]] = {}
    for position in VALID_POSITIONS:
        by_position[position] = sorted(
            [
                player
                for player in player_by_id.values()
                if str(player["position"]) == position
            ],
            key=lambda player: (
                -float(player["predicted_points"]),
                int(player["fpl_player_id"]),
            ),
        )

    best: Optional[Dict[str, Any]] = None
    for counts in _formation_count_options(rules):
        if any(
            len(by_position[position]) < counts[position]
            for position in VALID_POSITIONS
        ):
            continue

        starters: List[Dict[str, Any]] = []
        for position in VALID_POSITIONS:
            starters.extend(by_position[position][: counts[position]])

        starter_ids = [int(player["fpl_player_id"]) for player in starters]
        starter_set = set(starter_ids)
        bench = [
            player
            for player in player_by_id.values()
            if int(player["fpl_player_id"]) not in starter_set
        ]

        bench_goalkeepers = [
            player for player in bench if player["position"] == "GKP"
        ]
        outfield_bench = [
            player for player in bench if player["position"] != "GKP"
        ]
        if len(bench_goalkeepers) != 1 or len(outfield_bench) != 3:
            continue

        outfield_bench = sorted(
            outfield_bench,
            key=lambda player: (
                -float(player["predicted_points"]),
                int(player["fpl_player_id"]),
            ),
        )
        bench_order = [
            int(bench_goalkeepers[0]["fpl_player_id"])
        ] + [
            int(player["fpl_player_id"]) for player in outfield_bench
        ]

        captain_order = sorted(
            starters,
            key=lambda player: (
                -float(player["predicted_points"]),
                int(player["fpl_player_id"]),
            ),
        )
        captain_id = int(captain_order[0]["fpl_player_id"])
        vice_id = int(captain_order[1]["fpl_player_id"])

        legality = validate_lineup(
            rules,
            [_rule_player(player) for player in player_by_id.values()],
            starter_ids,
            bench_order,
            captain_id,
            vice_id,
        )
        if not legality["valid"]:
            continue

        xi_points = sum(float(player["predicted_points"]) for player in starters)
        captain_bonus = float(
            player_by_id[captain_id]["predicted_points"]
        )
        objective = xi_points + captain_bonus

        starting_counts = legality["starting_position_counts"]
        formation = "%s-%s-%s" % (
            int(starting_counts["DEF"]),
            int(starting_counts["MID"]),
            int(starting_counts["FWD"]),
        )

        candidate = {
            "formation": formation,
            "starting_player_ids": starter_ids,
            "bench_order": bench_order,
            "captain_player_id": captain_id,
            "vice_captain_player_id": vice_id,
            "starting_xi_predicted_points": xi_points,
            "captain_bonus_predicted_points": captain_bonus,
            "objective_points_before_transfer_cost": objective,
        }

        tie_key = (
            tuple(sorted(starter_ids)),
            tuple(bench_order),
            captain_id,
            vice_id,
        )
        if best is None:
            best = {**candidate, "_tie_key": tie_key}
            continue

        if objective > float(best["objective_points_before_transfer_cost"]) + 1e-12:
            best = {**candidate, "_tie_key": tie_key}
        elif abs(
            objective - float(best["objective_points_before_transfer_cost"])
        ) <= 1e-12 and tie_key < best["_tie_key"]:
            best = {**candidate, "_tie_key": tie_key}

    if best is None:
        raise SingleGWTransferOptimizerError(
            "No legal target-GW lineup exists for the candidate squad."
        )

    best.pop("_tie_key", None)
    return best


def _candidate_report_contract(
    report: Mapping[str, Any],
    *,
    season: str,
    state_kind: str,
    target_gw: int,
) -> None:
    if not isinstance(report, Mapping):
        raise SingleGWTransferOptimizerError(
            "transfer_candidates must be a Day126B candidate-report mapping."
        )
    if report.get("generator_scope") != "stateful_owned_squad_transfer_candidates":
        raise SingleGWTransferOptimizerError(
            "Day127A requires the Day126B stateful candidate scope."
        )
    if str(report.get("season")) != season:
        raise SingleGWTransferOptimizerError(
            "Transfer-candidate season does not match previous frozen state."
        )
    if str(report.get("state_kind")) != state_kind:
        raise SingleGWTransferOptimizerError(
            "Transfer-candidate state_kind does not match previous frozen state."
        )
    if int(report.get("target_gw") or -1) != int(target_gw):
        raise SingleGWTransferOptimizerError(
            "Transfer-candidate target_gw does not match next Gameweek=%s."
            % target_gw
        )

    safety = report.get("safety")
    if not isinstance(safety, Mapping):
        raise SingleGWTransferOptimizerError(
            "Transfer-candidate report is missing safety metadata."
        )
    forbidden_true = (
        "shadow_optimal_consumed",
        "opening_squad_rebuild_used",
        "general_transfer_targets_surface_used_as_state",
        "missing_predictions_zero_filled",
    )
    bad = [key for key in forbidden_true if safety.get(key) is not False]
    if bad:
        raise SingleGWTransferOptimizerError(
            "Transfer-candidate safety contract failed: %s." % bad
        )
    if not isinstance(report.get("pair_candidates"), Sequence):
        raise SingleGWTransferOptimizerError(
            "Transfer-candidate report must contain pair_candidates."
        )


def _edge_ids(edge: Mapping[str, Any]) -> Tuple[int, int]:
    try:
        out_id = int(
            edge.get("out_fpl_player_id")
            if edge.get("out_fpl_player_id") is not None
            else edge["out_player_id"]
        )
        in_id = int(
            edge.get("in_fpl_player_id")
            if edge.get("in_fpl_player_id") is not None
            else edge["in_player_id"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SingleGWTransferOptimizerError(
            "Candidate edge is missing valid OUT/IN player IDs."
        ) from exc
    if out_id <= 0 or in_id <= 0:
        raise SingleGWTransferOptimizerError(
            "Candidate edge player IDs must be positive."
        )
    return out_id, in_id


def _previous_players(state: SquadState) -> List[Dict[str, Any]]:
    return [
        {
            "fpl_player_id": int(player.fpl_player_id),
            "player_name": player.player_name,
            "position": str(player.position),
            "club_id": int(player.club_id),
            "purchase_price_units": int(player.purchase_price_units),
            "current_price_units": int(player.current_price_units),
            "selling_price_units": int(player.selling_price_units),
        }
        for player in state.players
    ]


def _apply_transfer_edges(
    *,
    previous_players: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    bank_before_units: int,
    rules: SquadTransferRules,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    players = {
        int(player["fpl_player_id"]): dict(player)
        for player in previous_players
    }

    outgoing_ids: List[int] = []
    incoming_ids: List[int] = []
    sales_units = 0
    purchases_units = 0

    for edge in edges:
        out_id, in_id = _edge_ids(edge)
        if out_id in outgoing_ids:
            raise SingleGWTransferOptimizerError(
                "Transfer plan repeats outgoing player_id=%s." % out_id
            )
        if in_id in incoming_ids:
            raise SingleGWTransferOptimizerError(
                "Transfer plan repeats incoming player_id=%s." % in_id
            )
        if out_id not in players:
            raise SingleGWTransferOptimizerError(
                "Outgoing player_id=%s is not in the previous frozen squad."
                % out_id
            )
        if in_id in players:
            raise SingleGWTransferOptimizerError(
                "Incoming player_id=%s is already owned." % in_id
            )

        outgoing = players[out_id]
        edge_position = str(edge.get("out_position") or outgoing["position"])
        in_position = str(edge.get("in_position") or "")
        if edge_position != outgoing["position"] or in_position != outgoing["position"]:
            raise SingleGWTransferOptimizerError(
                "Transfer edge is not position-compatible for player_id=%s."
                % out_id
            )

        sell = int(edge["out_selling_price_units"])
        if sell != int(outgoing["selling_price_units"]):
            raise SingleGWTransferOptimizerError(
                "Candidate OUT selling value does not match previous frozen state "
                "for player_id=%s." % out_id
            )
        buy = int(edge["in_price_units"])
        club_id = int(edge["in_club_id"])

        outgoing_ids.append(out_id)
        incoming_ids.append(in_id)
        sales_units += sell
        purchases_units += buy

        players.pop(out_id)
        players[in_id] = {
            "fpl_player_id": in_id,
            "player_name": edge.get("in_name"),
            "position": in_position,
            "club_id": club_id,
            "purchase_price_units": buy,
            "current_price_units": buy,
            "selling_price_units": buy,
        }

    bank_after = int(bank_before_units) + sales_units - purchases_units
    if bank_after < 0:
        raise SingleGWTransferOptimizerError(
            "Transfer plan is unaffordable after aggregate sale/buy reconciliation."
        )

    final_players = sorted(
        players.values(),
        key=lambda player: int(player["fpl_player_id"]),
    )
    if len(final_players) != int(rules.squad["size"]):
        raise SingleGWTransferOptimizerError(
            "Transfer plan does not preserve squad size."
        )

    current_market_value = sum(
        int(player["current_price_units"]) for player in final_players
    )
    legality = validate_squad(
        rules,
        [_rule_player(player) for player in final_players],
        budget_limit_units=current_market_value + bank_after,
    )
    if not legality["valid"]:
        raise SingleGWTransferOptimizerError(
            "Transfer plan produces an illegal squad: %s" % legality["errors"]
        )

    return final_players, bank_after, sales_units, purchases_units


def _transfer_payload(edge: Mapping[str, Any]) -> Dict[str, Any]:
    out_id, in_id = _edge_ids(edge)
    return {
        "out_fpl_player_id": out_id,
        "out_name": edge.get("out_name"),
        "out_position": edge.get("out_position"),
        "out_selling_price_units": int(edge["out_selling_price_units"]),
        "out_target_gw_predicted_points": edge.get(
            "out_horizon_predicted_points"
        ),
        "out_risk": dict(edge.get("out_risk") or {}),
        "out_role": dict(edge.get("out_role") or {}),
        "in_fpl_player_id": in_id,
        "in_name": edge.get("in_name"),
        "in_position": edge.get("in_position"),
        "in_club_id": int(edge["in_club_id"]),
        "in_price_units": int(edge["in_price_units"]),
        "in_target_gw_predicted_points": edge.get(
            "in_horizon_predicted_points"
        ),
        "in_risk": dict(edge.get("in_risk") or {}),
        "in_role": dict(edge.get("in_role") or {}),
        "candidate_projected_gain": float(edge["projected_gain"]),
        "candidate_legal_reasons": list(edge.get("legal_reasons") or []),
    }


def _option(
    *,
    action: str,
    edges: Sequence[Mapping[str, Any]],
    previous_players: Sequence[Mapping[str, Any]],
    bank_before_units: int,
    prediction_by_id: Mapping[int, Mapping[str, Any]],
    ledger_state: FreeTransferLedgerState,
    rules: SquadTransferRules,
    baseline_objective_before_cost: Optional[float],
) -> Dict[str, Any]:
    final_players, bank_after, sales_units, purchases_units = _apply_transfer_edges(
        previous_players=previous_players,
        edges=edges,
        bank_before_units=bank_before_units,
        rules=rules,
    )

    transfer_count = len(edges)
    financial_effect = TransferFinancialEffect(
        bank_before_units=int(bank_before_units),
        bank_after_units=int(bank_after),
        sales_units=int(sales_units),
        purchases_units=int(purchases_units),
        persistent=True,
        note="Day127A single-GW candidate plan; price/bank separate from FT count.",
    )
    transition = transition_free_transfer_ledger(
        ledger_state,
        transfer_count=transfer_count,
        completed_gameweek=ledger_state.gameweek,
        chip=None,
        financial_effect=financial_effect,
        rules=rules,
    )
    lineup = reoptimize_lineup_and_captain(
        squad_players=final_players,
        prediction_by_id=prediction_by_id,
        rules=rules,
    )

    before_cost = float(lineup["objective_points_before_transfer_cost"])
    hit_cost = int(transition.hit_points)
    net = before_cost - float(hit_cost)
    raw_gain = (
        0.0
        if baseline_objective_before_cost is None
        else before_cost - float(baseline_objective_before_cost)
    )
    net_gain = (
        0.0
        if baseline_objective_before_cost is None
        else net - float(baseline_objective_before_cost)
    )

    transfers = [_transfer_payload(edge) for edge in edges]
    transfer_ids = [
        (item["out_fpl_player_id"], item["in_fpl_player_id"])
        for item in transfers
    ]

    if action == "NO TRANSFER":
        explanation = [
            "No squad transfer is made.",
            "The current 15-player owned squad is re-evaluated for target-GW XI and captaincy.",
            "Transfer hit cost is 0.",
        ]
    else:
        explanation = [
            "%s transfer(s) are selected only from the Day126B stateful candidate set."
            % transfer_count,
            "The resulting 15-player squad is re-validated before scoring.",
            "Target-GW XI and captaincy are re-optimized after the transfers.",
            "Projected gain before transfer cost = %.6f points." % raw_gain,
            "Transfer hit cost = %s points." % hit_cost,
            "Net gain vs NO TRANSFER = %.6f points." % net_gain,
        ]

    return {
        "action": action,
        "transfer_count": transfer_count,
        "transfers": transfers,
        "transfer_ids": transfer_ids,
        "bank_before_units": int(bank_before_units),
        "bank_after_units": int(bank_after),
        "sales_units": int(sales_units),
        "purchases_units": int(purchases_units),
        "free_transfers_before": int(
            transition.available_free_transfers_before
        ),
        "free_transfers_used": int(transition.free_transfers_used),
        "charged_transfers": int(transition.charged_transfers),
        "transfer_hit_points": hit_cost,
        "free_transfers_next_gameweek": int(
            transition.available_free_transfers_next_gameweek
        ),
        "lineup": lineup,
        "projected_points_before_transfer_cost": before_cost,
        "projected_gain_before_cost_vs_no_transfer": raw_gain,
        "transfer_cost_points": hit_cost,
        "net_projected_points": net,
        "net_gain_vs_no_transfer": net_gain,
        "legal": True,
        "explanation": explanation,
    }


def _rank_key(option: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        -float(option["net_projected_points"]),
        int(option["transfer_count"]),
        -int(option["bank_after_units"]),
        tuple(
            sorted(
                (
                    int(item["out_fpl_player_id"]),
                    int(item["in_fpl_player_id"]),
                )
                for item in option["transfers"]
            )
        ),
    )


def optimize_single_gw_transfers(
    previous_frozen_state: Any,
    transfer_candidates: Mapping[str, Any],
    target_gw_predictions: Sequence[Mapping[str, Any]],
    free_transfer_ledger_state: FreeTransferLedgerState,
    *,
    max_transfers: int,
    rules: Optional[SquadTransferRules] = None,
) -> Dict[str, Any]:
    """Compare NO TRANSFER with legal Day126B transfer combinations.

    Day127A is intentionally single-GW only.  It does not run the opening-squad
    optimizer, value future FT optionality, activate chips, or write manager state.
    """

    previous = _coerce_frozen_state(previous_frozen_state)
    target_gw = int(previous.gameweek) + 1

    max_count = _require_int(max_transfers, "max_transfers", 0)
    if max_count > int(len(previous.players)):
        raise SingleGWTransferOptimizerError(
            "max_transfers cannot exceed squad size."
        )

    resolved_rules = rules or load_squad_transfer_rules(previous.season)
    if resolved_rules.effective_season != previous.season:
        raise SingleGWTransferOptimizerError(
            "Transfer policy season does not match previous frozen state."
        )

    if not isinstance(free_transfer_ledger_state, FreeTransferLedgerState):
        raise SingleGWTransferOptimizerError(
            "free_transfer_ledger_state must be a Day126A FreeTransferLedgerState."
        )
    if free_transfer_ledger_state.season != previous.season:
        raise SingleGWTransferOptimizerError(
            "FT ledger season does not match previous frozen state."
        )
    if free_transfer_ledger_state.state_kind != previous.state_kind:
        raise SingleGWTransferOptimizerError(
            "FT ledger state_kind does not match previous frozen state."
        )
    if free_transfer_ledger_state.gameweek != target_gw:
        raise SingleGWTransferOptimizerError(
            "FT ledger gameweek must equal next target Gameweek=%s." % target_gw
        )

    if previous.free_transfers.available_for_gameweek != target_gw:
        raise SingleGWTransferOptimizerError(
            "Previous frozen state FT availability does not point to target GW."
        )
    if previous.free_transfers.count != free_transfer_ledger_state.available_free_transfers:
        raise SingleGWTransferOptimizerError(
            "Previous frozen-state FT count and Day126A ledger disagree."
        )

    _candidate_report_contract(
        transfer_candidates,
        season=previous.season,
        state_kind=previous.state_kind,
        target_gw=target_gw,
    )

    prediction_by_id = _target_prediction_map(
        target_gw_predictions,
        target_gw,
    )
    previous_players = _previous_players(previous)

    no_transfer = _option(
        action="NO TRANSFER",
        edges=[],
        previous_players=previous_players,
        bank_before_units=previous.bank_units,
        prediction_by_id=prediction_by_id,
        ledger_state=free_transfer_ledger_state,
        rules=resolved_rules,
        baseline_objective_before_cost=None,
    )
    baseline = float(no_transfer["projected_points_before_transfer_cost"])

    # Normalize the NO TRANSFER comparison fields into the same exact structure
    # used for transfer candidates.
    no_transfer["projected_gain_before_cost_vs_no_transfer"] = 0.0
    no_transfer["net_gain_vs_no_transfer"] = 0.0

    pair_candidates = [
        dict(item) for item in transfer_candidates["pair_candidates"]
    ]
    pair_candidates.sort(
        key=lambda edge: (
            _edge_ids(edge)[0],
            _edge_ids(edge)[1],
        )
    )

    options: List[Dict[str, Any]] = [no_transfer]
    rejected_counts: Dict[str, int] = {}
    seen_transfer_sets = set()

    for transfer_count in range(1, max_count + 1):
        for combo in combinations(pair_candidates, transfer_count):
            out_ids = [_edge_ids(edge)[0] for edge in combo]
            in_ids = [_edge_ids(edge)[1] for edge in combo]

            if len(set(out_ids)) != len(out_ids):
                rejected_counts["duplicate_outgoing"] = (
                    rejected_counts.get("duplicate_outgoing", 0) + 1
                )
                continue
            if len(set(in_ids)) != len(in_ids):
                rejected_counts["duplicate_incoming"] = (
                    rejected_counts.get("duplicate_incoming", 0) + 1
                )
                continue

            transfer_set_key = (
                tuple(sorted(out_ids)),
                tuple(sorted(in_ids)),
            )
            if transfer_set_key in seen_transfer_sets:
                rejected_counts["duplicate_final_transfer_set"] = (
                    rejected_counts.get("duplicate_final_transfer_set", 0) + 1
                )
                continue

            try:
                candidate = _option(
                    action="TRANSFER",
                    edges=combo,
                    previous_players=previous_players,
                    bank_before_units=previous.bank_units,
                    prediction_by_id=prediction_by_id,
                    ledger_state=free_transfer_ledger_state,
                    rules=resolved_rules,
                    baseline_objective_before_cost=baseline,
                )
            except SingleGWTransferOptimizerError as exc:
                message = str(exc)
                if "illegal squad" in message:
                    reason = "illegal_final_squad"
                elif "unaffordable" in message:
                    reason = "unaffordable_final_plan"
                elif "Missing target-GW prediction" in message:
                    reason = "missing_target_gw_prediction"
                else:
                    reason = "invalid_candidate_plan"
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                continue

            seen_transfer_sets.add(transfer_set_key)
            options.append(candidate)

    ranked = sorted(options, key=_rank_key)
    for rank, option in enumerate(ranked, start=1):
        option["rank"] = rank

    winner = ranked[0]

    return {
        "optimizer_version": OPTIMIZER_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "season": previous.season,
        "target_gw": target_gw,
        "state_kind": previous.state_kind,
        "source_frozen_state_id": previous.state_id,
        "source_owned_state_fingerprint": previous.owned_state_fingerprint,
        "candidate_generator_version": transfer_candidates.get(
            "generator_version"
        ),
        "configured_max_transfers": max_count,
        "rules_version": resolved_rules.rules_version,
        "no_transfer": no_transfer,
        "winner": winner,
        "ranked_options": ranked,
        "counts": {
            "candidate_pair_edges": len(pair_candidates),
            "evaluated_legal_options": len(ranked),
            "evaluated_transfer_options": len(ranked) - 1,
        },
        "rejected_plan_summary": dict(sorted(rejected_counts.items())),
        "future_ft_option_value_monetized": False,
        "objective_scope": (
            "Single target-GW starting-XI predicted points plus one captain bonus "
            "minus transfer-hit points. NO TRANSFER uses the same evaluation structure. "
            "Future value of stored free transfers is exposed as state but is not "
            "converted into projected points in Day127A."
        ),
        "safety": {
            "writes_database": False,
            "writes_manager_state": False,
            "opening_squad_optimizer_used": False,
            "candidate_generator_bypassed": False,
            "shadow_optimal_consumed": False,
            "general_transfer_targets_used_as_state": False,
            "missing_predictions_zero_filled": False,
            "chips_activated": False,
            "single_gw_only": True,
            "no_transfer_first_class": True,
        },
    }
