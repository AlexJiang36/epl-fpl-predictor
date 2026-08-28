from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.rules.squad import SquadTransferRules, load_squad_transfer_rules


GENERATOR_VERSION = "fpl_transfer_candidate_generator_v1"
VALID_STATE_KINDS: Tuple[str, ...] = ("model_team", "team_alex")
VALID_POSITIONS: Tuple[str, ...] = ("GKP", "DEF", "MID", "FWD")


class TransferCandidateGeneratorError(ValueError):
    """Raised when stateful transfer-candidate inputs are invalid."""


@dataclass(frozen=True)
class CandidatePruningPolicy:
    """Transparent pruning only; no hidden football judgement."""

    max_out_candidates_per_position: Optional[int] = None
    max_in_candidates_per_position: Optional[int] = None
    max_pair_candidates_per_out: Optional[int] = 20
    minimum_projected_gain: Optional[float] = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_out_candidates_per_position",
            "max_in_candidates_per_position",
            "max_pair_candidates_per_out",
        ):
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise TransferCandidateGeneratorError(
                        "%s must be a positive integer or null." % field_name
                    )
        if self.minimum_projected_gain is not None:
            try:
                float(self.minimum_projected_gain)
            except (TypeError, ValueError):
                raise TransferCandidateGeneratorError(
                    "minimum_projected_gain must be numeric or null."
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    for method_name in ("to_dict", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            result = method()
            if isinstance(result, Mapping):
                return result
    raise TransferCandidateGeneratorError(
        "%s must be a mapping, dataclass, or expose to_dict()/as_dict()." % label
    )


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _identity(value: Any, label: str) -> str:
    if isinstance(value, bool) or value is None:
        raise TransferCandidateGeneratorError("%s has no valid identity." % label)
    if isinstance(value, (int, str)):
        text = str(value).strip()
        if text:
            return text
    raise TransferCandidateGeneratorError("%s has no valid identity." % label)


def _player_identity(row: Mapping[str, Any], label: str) -> Tuple[str, Any]:
    raw = _first_present(
        row,
        ("fpl_player_id", "player_id", "target_player_id", "id"),
    )
    return _identity(raw, label), raw


def _normalize_position(value: Any, label: str) -> str:
    position = "" if value is None else str(value).strip().upper()
    if position not in VALID_POSITIONS:
        raise TransferCandidateGeneratorError(
            "%s.position=%r is invalid; expected one of %s."
            % (label, value, VALID_POSITIONS)
        )
    return position


def _club_identity(row: Mapping[str, Any], label: str) -> Tuple[str, Any]:
    raw = _first_present(
        row,
        (
            "club_id",
            "team_id",
            "target_team_id",
            "team_short_name",
            "target_team_short_name",
        ),
    )
    return _identity(raw, "%s club" % label), raw


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TransferCandidateGeneratorError("%s must be non-negative." % label)
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise TransferCandidateGeneratorError("%s must be non-negative." % label)


def _price_millions_to_units(value: Any, label: str) -> int:
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise TransferCandidateGeneratorError("%s is not a valid price." % label)
    units = parsed * Decimal("10")
    if units < 0 or units != units.to_integral_value():
        raise TransferCandidateGeneratorError(
            "%s must resolve to exact tenths of £1m." % label
        )
    return int(units)


def _price_units(
    row: Mapping[str, Any],
    *,
    unit_keys: Sequence[str],
    million_keys: Sequence[str],
    label: str,
) -> int:
    value = _first_present(row, unit_keys)
    if value is not None:
        return _nonnegative_int(value, label)
    value = _first_present(row, million_keys)
    if value is not None:
        return _price_millions_to_units(value, label)
    raise TransferCandidateGeneratorError("%s is missing." % label)


def _optional_float(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    value = _first_present(row, keys)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[bool]:
    value = _first_present(row, keys)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        if "|" in value:
            raw = value.split("|")
        elif "," in value:
            raw = value.split(",")
        else:
            raw = [value]
        return sorted({item.strip() for item in raw if item.strip()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    return [str(value).strip()] if str(value).strip() else []


def _state_kind(mapping: Mapping[str, Any]) -> str:
    raw = _first_present(mapping, ("state_kind", "team_kind", "manager_kind"))
    if raw is None:
        raise TransferCandidateGeneratorError(
            "Owned squad state is missing state_kind."
        )
    normalized = str(raw).strip().lower()
    if normalized == "shadow_optimal":
        raise TransferCandidateGeneratorError(
            "shadow_optimal is diagnostic only and must never be used as owned squad state."
        )
    if normalized not in VALID_STATE_KINDS:
        raise TransferCandidateGeneratorError(
            "state_kind=%s is invalid; expected one of %s."
            % (normalized, VALID_STATE_KINDS)
        )
    return normalized


def _state_players(mapping: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    raw = _first_present(mapping, ("squad", "players", "owned_players"))
    if raw is None and isinstance(mapping.get("primary"), Mapping):
        raw = mapping["primary"].get("players")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TransferCandidateGeneratorError(
            "Owned squad state must contain a squad/players sequence."
        )
    result: List[Mapping[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise TransferCandidateGeneratorError(
                "owned squad row %s must be a mapping." % index
            )
        result.append(row)
    return result


def _state_bank_units(mapping: Mapping[str, Any]) -> int:
    raw = _first_present(mapping, ("bank_units", "bank_after_units"))
    if raw is not None:
        return _nonnegative_int(raw, "bank_units")
    raw = _first_present(mapping, ("bank", "bank_gbp_m"))
    if raw is not None:
        return _price_millions_to_units(raw, "bank")
    raise TransferCandidateGeneratorError("Owned squad state is missing bank.")


def _owned_player(
    raw: Mapping[str, Any],
    *,
    locked_keys: Sequence[str],
) -> Dict[str, Any]:
    key, original_id = _player_identity(raw, "owned player")
    position = _normalize_position(
        _first_present(raw, ("position", "target_position")),
        "owned player %s" % key,
    )
    club_key, club_raw = _club_identity(raw, "owned player %s" % key)
    selling_price_units = _price_units(
        raw,
        unit_keys=(
            "selling_price_units",
            "sale_price_units",
            "sell_price_units",
        ),
        million_keys=("selling_price", "sale_price", "sell_price"),
        label="owned player %s selling price" % key,
    )
    current_price_units: Optional[int]
    try:
        current_price_units = _price_units(
            raw,
            unit_keys=("current_price_units", "now_cost", "price_units"),
            million_keys=("current_price", "price"),
            label="owned player %s current price" % key,
        )
    except TransferCandidateGeneratorError:
        current_price_units = None

    locked = key in set(locked_keys)
    row_locked = _optional_bool(raw, ("locked", "is_locked", "transfer_locked"))
    if row_locked is True:
        locked = True

    return {
        "key": key,
        "player_id": original_id,
        "fpl_player_id": raw.get("fpl_player_id"),
        "player_name": _first_present(raw, ("web_name", "player_name", "name")),
        "position": position,
        "club_key": club_key,
        "club_id": club_raw,
        "team_name": _first_present(raw, ("team_name", "club_name")),
        "team_short_name": _first_present(
            raw, ("team_short_name", "club_short_name")
        ),
        "selling_price_units": selling_price_units,
        "current_price_units": current_price_units,
        "locked": locked,
    }


def _market_player(raw: Mapping[str, Any]) -> Dict[str, Any]:
    key, original_id = _player_identity(raw, "market player")
    position = _normalize_position(
        _first_present(raw, ("position", "target_position")),
        "market player %s" % key,
    )
    club_key, club_raw = _club_identity(raw, "market player %s" % key)
    price_units = _price_units(
        raw,
        unit_keys=("now_cost", "price_units", "in_price_units", "target_price_units"),
        million_keys=("price", "current_price"),
        label="market player %s price" % key,
    )
    return {
        "key": key,
        "player_id": original_id,
        "fpl_player_id": raw.get("fpl_player_id"),
        "player_name": _first_present(raw, ("web_name", "player_name", "name")),
        "position": position,
        "club_key": club_key,
        "club_id": club_raw,
        "team_name": _first_present(raw, ("team_name", "club_name")),
        "team_short_name": _first_present(
            raw, ("team_short_name", "club_short_name")
        ),
        "price_units": price_units,
        "status": raw.get("status"),
        "selection_eligible": _optional_bool(raw, ("selection_eligible",)),
        "eligibility_reason": raw.get("eligibility_reason"),
    }


def _prediction_groups(
    horizon_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(horizon_rows):
        if not isinstance(row, Mapping):
            raise TransferCandidateGeneratorError(
                "horizon_predictions[%s] must be a mapping." % index
            )
        key, _ = _player_identity(row, "horizon prediction row")
        grouped[key].append(row)

    result: Dict[str, Dict[str, Any]] = {}
    for key, rows in grouped.items():
        explicit_horizon = [
            _optional_float(
                row,
                (
                    "horizon_predicted_points",
                    "horizon_points",
                    "projected_horizon_points",
                ),
            )
            for row in rows
        ]
        explicit_horizon = [value for value in explicit_horizon if value is not None]

        if explicit_horizon:
            horizon_points = float(explicit_horizon[0])
        else:
            per_gw = [
                _optional_float(
                    row,
                    (
                        "predicted_points",
                        "adjusted_next_fixture_points",
                        "production_player_model_points",
                    ),
                )
                for row in rows
            ]
            valid = [value for value in per_gw if value is not None]
            horizon_points = float(sum(valid)) if valid else None

        def gw_sort_key(row: Mapping[str, Any]) -> Tuple[int, str]:
            raw_gw = _first_present(row, ("target_gw", "gw", "gameweek"))
            try:
                gw = int(raw_gw)
            except (TypeError, ValueError):
                gw = 999
            return gw, str(_first_present(row, ("player_id", "fpl_player_id")))

        ordered = sorted(rows, key=gw_sort_key)
        anchor = ordered[0]

        selection_values = [
            _optional_bool(row, ("selection_eligible",))
            for row in ordered
            if _optional_bool(row, ("selection_eligible",)) is not None
        ]
        selection_eligible = (
            all(selection_values) if selection_values else None
        )

        risk_flags: List[str] = []
        for row in ordered:
            risk_flags.extend(_string_list(row.get("risk_flags")))
        risk_flags = sorted(set(risk_flags))

        result[key] = {
            "horizon_predicted_points": horizon_points,
            "target_gw_predicted_points": _optional_float(
                anchor,
                (
                    "predicted_points",
                    "adjusted_next_fixture_points",
                    "production_player_model_points",
                ),
            ),
            "selection_eligible": selection_eligible,
            "eligibility_reason": anchor.get("eligibility_reason"),
            "risk": {
                "risk_flags": risk_flags,
                "fallback_used": any(
                    _optional_bool(row, ("fallback_used",)) is True
                    for row in ordered
                ),
                "manual_review_required": any(
                    _optional_bool(row, ("manual_review_required",)) is True
                    for row in ordered
                ),
                "prediction_confidence": _first_present(
                    anchor,
                    ("prediction_confidence", "confidence"),
                ),
                "data_quality_status": anchor.get("data_quality_status"),
                "uncertainty": _optional_float(
                    anchor,
                    (
                        "uncertainty",
                        "prediction_uncertainty",
                        "uncertainty_half_width",
                    ),
                ),
            },
            "role": {
                "expected_minutes": _optional_float(
                    anchor, ("expected_minutes",)
                ),
                "start_probability": _optional_float(
                    anchor, ("start_probability",)
                ),
                "appearance_probability": _optional_float(
                    anchor, ("appearance_probability",)
                ),
                "role_proxy": anchor.get("role_proxy"),
                "role_confidence": _first_present(
                    anchor, ("role_confidence", "prediction_confidence")
                ),
                "sample_reliability_tier": anchor.get(
                    "sample_reliability_tier"
                ),
            },
            "prediction_row_count": len(rows),
        }
    return result


def _validate_owned_squad(
    owned: Sequence[Mapping[str, Any]],
    rules: SquadTransferRules,
) -> None:
    required_size = int(rules.squad["size"])
    if len(owned) != required_size:
        raise TransferCandidateGeneratorError(
            "Owned squad has %s players; target-season policy requires %s."
            % (len(owned), required_size)
        )

    keys = [str(row["key"]) for row in owned]
    if len(keys) != len(set(keys)):
        raise TransferCandidateGeneratorError(
            "Owned squad contains duplicate player identities."
        )

    expected_quotas = {
        position: int(rules.squad["position_quotas"][position])
        for position in VALID_POSITIONS
    }
    actual_quotas = Counter(str(row["position"]) for row in owned)
    if dict(actual_quotas) != expected_quotas:
        raise TransferCandidateGeneratorError(
            "Owned squad position quotas %s do not match target-season policy %s."
            % (dict(actual_quotas), expected_quotas)
        )

    max_per_club = int(rules.squad["max_players_per_club"])
    club_counts = Counter(str(row["club_key"]) for row in owned)
    violations = {
        club: count
        for club, count in club_counts.items()
        if count > max_per_club
    }
    if violations:
        raise TransferCandidateGeneratorError(
            "Owned squad already violates club limit: %s." % violations
        )


def _candidate_risk_role(
    player: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "risk": dict(prediction.get("risk") or {}),
        "role": dict(prediction.get("role") or {}),
        "status": player.get("status"),
        "eligibility_reason": (
            prediction.get("eligibility_reason")
            or player.get("eligibility_reason")
        ),
    }


def _sort_out(candidate: Mapping[str, Any]) -> Tuple[float, str]:
    return (
        float(candidate["horizon_predicted_points"]),
        str(candidate["key"]),
    )


def _sort_in(candidate: Mapping[str, Any]) -> Tuple[float, int, str]:
    return (
        -float(candidate["horizon_predicted_points"]),
        int(candidate["price_units"]),
        str(candidate["key"]),
    )


def _sort_pair(candidate: Mapping[str, Any]) -> Tuple[float, float, int, str]:
    return (
        -float(candidate["projected_gain"]),
        -float(candidate["in_horizon_predicted_points"]),
        int(candidate["in_price_units"]),
        str(candidate["in_key"]),
    )


def _prune_by_position(
    rows: Sequence[Dict[str, Any]],
    *,
    limit: Optional[int],
    sort_key: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["position"])].append(row)

    kept: List[Dict[str, Any]] = []
    pruned: List[Dict[str, Any]] = []
    for position in VALID_POSITIONS:
        ordered = sorted(grouped.get(position, []), key=sort_key)
        if limit is None:
            kept.extend(ordered)
        else:
            kept.extend(ordered[:limit])
            for item in ordered[limit:]:
                removed = dict(item)
                removed["pruned_reason"] = (
                    "position_limit_%s" % limit
                )
                pruned.append(removed)
    return kept, pruned


def generate_transfer_candidates(
    squad_state: Any,
    current_market: Sequence[Mapping[str, Any]],
    horizon_predictions: Sequence[Mapping[str, Any]],
    *,
    locked_player_ids: Sequence[Any] = (),
    pruning_policy: Optional[CandidatePruningPolicy] = None,
    rules: Optional[SquadTransferRules] = None,
) -> Dict[str, Any]:
    """Generate legal, stateful 1-for-1 transfer candidate edges.

    This module intentionally stops before transfer optimization.  It does not
    compare full plans, re-optimize lineups, price hits, or mutate manager state.
    """

    state = _as_mapping(squad_state, "squad_state")
    state_kind = _state_kind(state)
    season = str(
        _first_present(state, ("season", "target_season", "effective_season"))
        or ""
    ).strip()
    if not season:
        raise TransferCandidateGeneratorError(
            "Owned squad state is missing season."
        )
    target_gw_raw = _first_present(state, ("gameweek", "gw", "target_gw"))
    target_gw: Optional[int]
    try:
        target_gw = int(target_gw_raw) if target_gw_raw is not None else None
    except (TypeError, ValueError):
        raise TransferCandidateGeneratorError("Owned squad state gameweek is invalid.")

    resolved_rules = rules or load_squad_transfer_rules(season)
    if str(resolved_rules.effective_season) != season:
        raise TransferCandidateGeneratorError(
            "Squad state season does not match loaded transfer policy."
        )

    policy = pruning_policy or CandidatePruningPolicy()
    locked_keys = [_identity(value, "locked_player_ids") for value in locked_player_ids]
    bank_units = _state_bank_units(state)

    owned = [
        _owned_player(row, locked_keys=locked_keys)
        for row in _state_players(state)
    ]
    _validate_owned_squad(owned, resolved_rules)

    market_rows: List[Dict[str, Any]] = []
    market_by_key: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(current_market):
        if not isinstance(raw, Mapping):
            raise TransferCandidateGeneratorError(
                "current_market[%s] must be a mapping." % index
            )
        normalized = _market_player(raw)
        if normalized["key"] in market_by_key:
            raise TransferCandidateGeneratorError(
                "current_market contains duplicate player identity %s."
                % normalized["key"]
            )
        market_rows.append(normalized)
        market_by_key[str(normalized["key"])] = normalized

    predictions = _prediction_groups(horizon_predictions)
    owned_keys = {str(row["key"]) for row in owned}
    club_counts = Counter(str(row["club_key"]) for row in owned)
    max_per_club = int(resolved_rules.squad["max_players_per_club"])

    excluded_outgoing: List[Dict[str, Any]] = []
    raw_out_candidates: List[Dict[str, Any]] = []

    for player in owned:
        key = str(player["key"])
        prediction = predictions.get(key)
        if player["locked"]:
            excluded_outgoing.append(
                {
                    "key": key,
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": player["position"],
                    "reason": "locked_player",
                }
            )
            continue
        if prediction is None or prediction.get("horizon_predicted_points") is None:
            excluded_outgoing.append(
                {
                    "key": key,
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": player["position"],
                    "reason": "missing_horizon_prediction_not_zero_filled",
                }
            )
            continue

        candidate = {
            **player,
            "horizon_predicted_points": float(
                prediction["horizon_predicted_points"]
            ),
            "target_gw_predicted_points": prediction.get(
                "target_gw_predicted_points"
            ),
            **_candidate_risk_role(player, prediction),
            "consideration_reasons": [
                "currently_owned",
                "not_locked",
                "horizon_prediction_available",
                "eligible_for_stateful_out_consideration",
            ],
        }
        raw_out_candidates.append(candidate)

    excluded_incoming: List[Dict[str, Any]] = []
    raw_in_candidates: List[Dict[str, Any]] = []

    for player in market_rows:
        key = str(player["key"])
        if key in owned_keys:
            excluded_incoming.append(
                {
                    "key": key,
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": player["position"],
                    "reason": "already_owned",
                }
            )
            continue

        prediction = predictions.get(key)
        if prediction is None or prediction.get("horizon_predicted_points") is None:
            excluded_incoming.append(
                {
                    "key": key,
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": player["position"],
                    "reason": "missing_horizon_prediction_not_zero_filled",
                }
            )
            continue

        eligibility = prediction.get("selection_eligible")
        if eligibility is None:
            eligibility = player.get("selection_eligible")
        if eligibility is not True:
            excluded_incoming.append(
                {
                    "key": key,
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": player["position"],
                    "reason": (
                        "selection_ineligible"
                        if eligibility is False
                        else "missing_explicit_selection_eligibility"
                    ),
                    "eligibility_reason": (
                        prediction.get("eligibility_reason")
                        or player.get("eligibility_reason")
                    ),
                }
            )
            continue

        candidate = {
            **player,
            "horizon_predicted_points": float(
                prediction["horizon_predicted_points"]
            ),
            "target_gw_predicted_points": prediction.get(
                "target_gw_predicted_points"
            ),
            **_candidate_risk_role(player, prediction),
            "consideration_reasons": [
                "not_currently_owned",
                "explicitly_selection_eligible",
                "horizon_prediction_available",
                "available_for_stateful_in_consideration",
            ],
        }
        raw_in_candidates.append(candidate)

    out_candidates, pruned_out = _prune_by_position(
        raw_out_candidates,
        limit=policy.max_out_candidates_per_position,
        sort_key=_sort_out,
    )
    in_candidates, pruned_in = _prune_by_position(
        raw_in_candidates,
        limit=policy.max_in_candidates_per_position,
        sort_key=_sort_in,
    )

    in_by_position: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in in_candidates:
        in_by_position[str(candidate["position"])].append(candidate)

    pair_candidates: List[Dict[str, Any]] = []
    rejection_counts: Counter = Counter()
    pair_pruned_count = 0

    for out_player in out_candidates:
        per_out: List[Dict[str, Any]] = []
        position = str(out_player["position"])
        available_units = bank_units + int(out_player["selling_price_units"])

        for in_player in in_by_position.get(position, []):
            if int(in_player["price_units"]) > available_units:
                rejection_counts["unaffordable"] += 1
                continue

            incoming_club = str(in_player["club_key"])
            outgoing_club = str(out_player["club_key"])
            resulting_incoming_club_count = (
                int(club_counts.get(incoming_club, 0))
                - (1 if outgoing_club == incoming_club else 0)
                + 1
            )
            if resulting_incoming_club_count > max_per_club:
                rejection_counts["club_limit"] += 1
                continue

            gain = float(in_player["horizon_predicted_points"]) - float(
                out_player["horizon_predicted_points"]
            )
            if (
                policy.minimum_projected_gain is not None
                and gain < float(policy.minimum_projected_gain)
            ):
                rejection_counts["below_minimum_projected_gain"] += 1
                continue

            bank_after = available_units - int(in_player["price_units"])
            per_out.append(
                {
                    "out_key": out_player["key"],
                    "out_player_id": out_player["player_id"],
                    "out_fpl_player_id": out_player.get("fpl_player_id"),
                    "out_name": out_player.get("player_name"),
                    "out_position": position,
                    "out_club_id": out_player.get("club_id"),
                    "out_selling_price_units": int(
                        out_player["selling_price_units"]
                    ),
                    "out_horizon_predicted_points": float(
                        out_player["horizon_predicted_points"]
                    ),
                    "out_risk": dict(out_player.get("risk") or {}),
                    "out_role": dict(out_player.get("role") or {}),
                    "in_key": in_player["key"],
                    "in_player_id": in_player["player_id"],
                    "in_fpl_player_id": in_player.get("fpl_player_id"),
                    "in_name": in_player.get("player_name"),
                    "in_position": position,
                    "in_club_id": in_player.get("club_id"),
                    "in_team_name": in_player.get("team_name"),
                    "in_team_short_name": in_player.get(
                        "team_short_name"
                    ),
                    "in_price_units": int(in_player["price_units"]),
                    "in_horizon_predicted_points": float(
                        in_player["horizon_predicted_points"]
                    ),
                    "in_risk": dict(in_player.get("risk") or {}),
                    "in_role": dict(in_player.get("role") or {}),
                    "bank_before_units": bank_units,
                    "affordable_budget_units": available_units,
                    "bank_after_units": bank_after,
                    "projected_gain": gain,
                    "legal_reasons": [
                        "same_position",
                        "incoming_not_owned",
                        "incoming_selection_eligible",
                        "selling_price_plus_bank_affordable",
                        "club_limit_preserved",
                        "owned_state_not_shadow_optimal",
                    ],
                    "candidate_scope": (
                        "stateful_owned_squad_single_swap_edge"
                    ),
                }
            )

        per_out = sorted(per_out, key=_sort_pair)
        if policy.max_pair_candidates_per_out is not None:
            pair_pruned_count += max(
                0,
                len(per_out) - policy.max_pair_candidates_per_out,
            )
            per_out = per_out[: policy.max_pair_candidates_per_out]
        pair_candidates.extend(per_out)

    pair_candidates = sorted(
        pair_candidates,
        key=lambda row: (
            str(row["out_position"]),
            str(row["out_key"]),
            *_sort_pair(row),
        ),
    )

    paired_out_keys = {str(row["out_key"]) for row in pair_candidates}
    paired_in_keys = {str(row["in_key"]) for row in pair_candidates}
    for candidate in out_candidates:
        candidate["has_legal_pair"] = str(candidate["key"]) in paired_out_keys
        if candidate["has_legal_pair"]:
            candidate["consideration_reasons"].append(
                "at_least_one_legal_replacement_pair"
            )
    for candidate in in_candidates:
        candidate["has_legal_pair"] = str(candidate["key"]) in paired_in_keys
        if candidate["has_legal_pair"]:
            candidate["consideration_reasons"].append(
                "at_least_one_legal_owned_squad_pair"
            )

    return {
        "generator_version": GENERATOR_VERSION,
        "generator_scope": "stateful_owned_squad_transfer_candidates",
        "season": season,
        "target_gw": target_gw,
        "state_kind": state_kind,
        "source_state_version": _first_present(
            state, ("state_version", "contract_version", "version")
        ),
        "rules": {
            "rules_version": resolved_rules.rules_version,
            "schema_version": resolved_rules.schema_version,
            "rules_sha256": resolved_rules.sha256,
            "max_players_per_club": max_per_club,
        },
        "bank_units": bank_units,
        "owned_player_count": len(owned),
        "pruning_policy": policy.to_dict(),
        "out_candidates": out_candidates,
        "in_candidates": in_candidates,
        "pair_candidates": pair_candidates,
        "excluded_outgoing": excluded_outgoing,
        "excluded_incoming": excluded_incoming,
        "pruned_outgoing": pruned_out,
        "pruned_incoming": pruned_in,
        "pair_rejection_summary": dict(sorted(rejection_counts.items())),
        "pair_pruned_count": pair_pruned_count,
        "counts": {
            "raw_out_candidates": len(raw_out_candidates),
            "out_candidates": len(out_candidates),
            "raw_in_candidates": len(raw_in_candidates),
            "in_candidates": len(in_candidates),
            "legal_pair_candidates": len(pair_candidates),
        },
        "safety": {
            "writes_database": False,
            "writes_manager_state": False,
            "shadow_optimal_consumed": False,
            "opening_squad_rebuild_used": False,
            "general_transfer_targets_surface_used_as_state": False,
            "missing_predictions_zero_filled": False,
        },
        "scope_notes": [
            "Shadow Optimal remains a separate diagnostic and is not an owned-state input.",
            "/transfers/targets may remain a reusable scouting surface but is not the stateful candidate source of truth.",
            "This generator stops before transfer optimization, hit pricing, lineup re-optimization, or manager-state mutation.",
            "Only legal one-for-one position-compatible edges are emitted; downstream planners may compose them into multi-transfer plans.",
        ],
    }
