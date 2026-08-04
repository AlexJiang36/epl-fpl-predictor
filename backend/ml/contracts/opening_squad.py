from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CONTRACT_VERSION = "fpl_opening_squad_objective_v1"
DEFAULT_POLICY_VERSION = "fast_lane_2026_27_v1"

HORIZON_GW1_ONLY = "gw1_only"
HORIZON_GW1_GW5 = "gw1_gw5"
HORIZON_VALUES = (HORIZON_GW1_ONLY, HORIZON_GW1_GW5)

RECOMMENDATION_PREVIEW_ONLY = "preview_only"
VICE_CAPTAIN_CONTINGENCY_ONLY = "contingency_only_no_deterministic_bonus"
DEFAULT_TIE_BREAKERS = (
    "objective_value_desc",
    "gw1_net_contribution_desc",
    "total_penalty_asc",
    "bank_units_desc_if_enabled",
    "canonical_player_ids_asc",
)
ROUND_DIGITS = 10
EPSILON = 1e-8


class OpeningSquadObjectiveError(ValueError):
    """Raised when an objective policy or evaluation input is invalid."""


def _round(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OpeningSquadObjectiveError("%s must be a mapping." % label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OpeningSquadObjectiveError("%s must be a sequence." % label)
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpeningSquadObjectiveError("%s must be a non-empty string." % label)
    return value.strip()


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise OpeningSquadObjectiveError("%s must be a boolean." % label)
    return value


def _float(
    value: Any,
    label: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool):
        raise OpeningSquadObjectiveError("%s must be numeric." % label)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise OpeningSquadObjectiveError("%s must be numeric." % label)
    if minimum is not None and result < minimum:
        raise OpeningSquadObjectiveError("%s must be at least %s." % (label, minimum))
    if maximum is not None and result > maximum:
        raise OpeningSquadObjectiveError("%s must be at most %s." % (label, maximum))
    return result


def _int(
    value: Any,
    label: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpeningSquadObjectiveError("%s must be an integer." % label)
    if minimum is not None and value < minimum:
        raise OpeningSquadObjectiveError("%s must be at least %s." % (label, minimum))
    if maximum is not None and value > maximum:
        raise OpeningSquadObjectiveError("%s must be at most %s." % (label, maximum))
    return int(value)


def _probability(value: Any, label: str) -> float:
    return _float(value, label, 0.0, 1.0)


def _player_id(value: Any, label: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise OpeningSquadObjectiveError(
            "%s must be an integer or non-empty string." % label
        )
    if isinstance(value, str) and not value.strip():
        raise OpeningSquadObjectiveError("%s must not be empty." % label)
    return value


def _discounts(value: Mapping[Any, Any]) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for raw_key, raw_value in value.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError):
            raise OpeningSquadObjectiveError(
                "gameweek_discounts keys must be integer Gameweeks."
            )
        result[key] = _float(
            raw_value,
            "gameweek_discounts[%s]" % key,
            0.0,
        )
    return result


@dataclass(frozen=True)
class RoleWeightPolicy:
    starter_weight: float = 1.0
    captain_bonus_weight: float = 1.0
    vice_captain_bonus_weight: float = 0.0
    vice_captain_treatment: str = VICE_CAPTAIN_CONTINGENCY_ONLY
    bench_goalkeeper_weight: float = 0.05
    bench_outfield_weights: Tuple[float, float, float] = (0.12, 0.06, 0.03)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "starter_weight",
            _float(self.starter_weight, "starter_weight", 0.0),
        )
        object.__setattr__(
            self,
            "captain_bonus_weight",
            _float(self.captain_bonus_weight, "captain_bonus_weight", 0.0),
        )
        object.__setattr__(
            self,
            "vice_captain_bonus_weight",
            _float(
                self.vice_captain_bonus_weight,
                "vice_captain_bonus_weight",
                0.0,
            ),
        )
        object.__setattr__(
            self,
            "vice_captain_treatment",
            _text(self.vice_captain_treatment, "vice_captain_treatment"),
        )
        object.__setattr__(
            self,
            "bench_goalkeeper_weight",
            _float(self.bench_goalkeeper_weight, "bench_goalkeeper_weight", 0.0),
        )
        raw = tuple(self.bench_outfield_weights)
        if len(raw) != 3:
            raise OpeningSquadObjectiveError(
                "bench_outfield_weights must contain exactly three values."
            )
        object.__setattr__(
            self,
            "bench_outfield_weights",
            tuple(
                _float(value, "bench_outfield_weights[%s]" % index, 0.0)
                for index, value in enumerate(raw)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["bench_outfield_weights"] = list(self.bench_outfield_weights)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoleWeightPolicy":
        mapping = _mapping(value, "role_weights")
        return cls(
            starter_weight=mapping.get("starter_weight", 1.0),
            captain_bonus_weight=mapping.get("captain_bonus_weight", 1.0),
            vice_captain_bonus_weight=mapping.get(
                "vice_captain_bonus_weight", 0.0
            ),
            vice_captain_treatment=mapping.get(
                "vice_captain_treatment", VICE_CAPTAIN_CONTINGENCY_ONLY
            ),
            bench_goalkeeper_weight=mapping.get("bench_goalkeeper_weight", 0.05),
            bench_outfield_weights=tuple(
                mapping.get("bench_outfield_weights", (0.12, 0.06, 0.03))
            ),
        )


@dataclass(frozen=True)
class RiskPenaltyPolicy:
    expected_minutes_target: float = 60.0
    minutes_shortfall_penalty_per_minute: float = 0.01
    start_probability_target: float = 0.70
    start_probability_shortfall_penalty: float = 0.75
    fallback_used_penalty: float = 0.35
    fallback_level_penalty: float = 0.15
    uncertainty_width_penalty: float = 0.08
    missing_uncertainty_penalty: float = 0.50

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_minutes_target",
            _float(self.expected_minutes_target, "expected_minutes_target", 0.0, 120.0),
        )
        object.__setattr__(
            self,
            "minutes_shortfall_penalty_per_minute",
            _float(
                self.minutes_shortfall_penalty_per_minute,
                "minutes_shortfall_penalty_per_minute",
                0.0,
            ),
        )
        object.__setattr__(
            self,
            "start_probability_target",
            _probability(self.start_probability_target, "start_probability_target"),
        )
        for field_name in (
            "start_probability_shortfall_penalty",
            "fallback_used_penalty",
            "fallback_level_penalty",
            "uncertainty_width_penalty",
            "missing_uncertainty_penalty",
        ):
            object.__setattr__(
                self,
                field_name,
                _float(getattr(self, field_name), field_name, 0.0),
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RiskPenaltyPolicy":
        mapping = _mapping(value, "risk_penalties")
        defaults = cls()
        return cls(
            **{
                field_name: mapping.get(field_name, getattr(defaults, field_name))
                for field_name in asdict(defaults)
            }
        )


@dataclass(frozen=True)
class ValueBankPolicy:
    """Optional secondary utility; defaults never force full budget spend."""

    value_bonus_weight: float = 0.0
    bank_bonus_per_unit: float = 0.0
    use_bank_as_tie_breaker: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value_bonus_weight",
            _float(self.value_bonus_weight, "value_bonus_weight", 0.0),
        )
        object.__setattr__(
            self,
            "bank_bonus_per_unit",
            _float(self.bank_bonus_per_unit, "bank_bonus_per_unit", 0.0),
        )
        object.__setattr__(
            self,
            "use_bank_as_tie_breaker",
            _bool(self.use_bank_as_tie_breaker, "use_bank_as_tie_breaker"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ValueBankPolicy":
        mapping = _mapping(value, "value_bank")
        return cls(
            value_bonus_weight=mapping.get("value_bonus_weight", 0.0),
            bank_bonus_per_unit=mapping.get("bank_bonus_per_unit", 0.0),
            use_bank_as_tie_breaker=mapping.get("use_bank_as_tie_breaker", False),
        )


@dataclass(frozen=True)
class OpeningSquadObjectivePolicy:
    target_season: str
    horizon_mode: str = HORIZON_GW1_GW5
    target_gw: int = 1
    contract_version: str = CONTRACT_VERSION
    policy_version: str = DEFAULT_POLICY_VERSION
    objective_name: str = "discounted_opening_squad_expected_points"
    gameweek_discounts: Mapping[int, float] = field(
        default_factory=lambda: {
            1: 1.0,
            2: 0.85,
            3: 0.7225,
            4: 0.614125,
            5: 0.52200625,
        }
    )
    role_weights: RoleWeightPolicy = field(default_factory=RoleWeightPolicy)
    risk_penalties: RiskPenaltyPolicy = field(default_factory=RiskPenaltyPolicy)
    value_bank: ValueBankPolicy = field(default_factory=ValueBankPolicy)
    allow_gw1_fallback: bool = True
    tie_breakers: Tuple[str, ...] = DEFAULT_TIE_BREAKERS
    recommendation_status: str = RECOMMENDATION_PREVIEW_ONLY
    writes_enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_season", _text(self.target_season, "target_season"))
        if self.horizon_mode not in HORIZON_VALUES:
            raise OpeningSquadObjectiveError(
                "horizon_mode must be one of %s." % (HORIZON_VALUES,)
            )
        object.__setattr__(self, "target_gw", _int(self.target_gw, "target_gw", 1, 38))
        if self.target_gw != 1:
            raise OpeningSquadObjectiveError(
                "Opening-squad Fast Lane policy requires target_gw=1."
            )
        object.__setattr__(
            self, "contract_version", _text(self.contract_version, "contract_version")
        )
        if self.contract_version != CONTRACT_VERSION:
            raise OpeningSquadObjectiveError(
                "Unsupported opening-squad objective contract_version=%s."
                % self.contract_version
            )
        object.__setattr__(self, "policy_version", _text(self.policy_version, "policy_version"))
        object.__setattr__(self, "objective_name", _text(self.objective_name, "objective_name"))

        discounts = _discounts(self.gameweek_discounts)
        missing = [gw for gw in self.requested_gameweeks if gw not in discounts]
        if missing:
            raise OpeningSquadObjectiveError(
                "gameweek_discounts is missing required Gameweeks: %s." % missing
            )
        object.__setattr__(self, "gameweek_discounts", discounts)

        if not isinstance(self.role_weights, RoleWeightPolicy):
            object.__setattr__(
                self, "role_weights", RoleWeightPolicy.from_mapping(self.role_weights)
            )
        if not isinstance(self.risk_penalties, RiskPenaltyPolicy):
            object.__setattr__(
                self,
                "risk_penalties",
                RiskPenaltyPolicy.from_mapping(self.risk_penalties),
            )
        if not isinstance(self.value_bank, ValueBankPolicy):
            object.__setattr__(
                self, "value_bank", ValueBankPolicy.from_mapping(self.value_bank)
            )

        object.__setattr__(
            self, "allow_gw1_fallback", _bool(self.allow_gw1_fallback, "allow_gw1_fallback")
        )
        normalized_tie_breakers = tuple(
            _text(value, "tie_breakers[%s]" % index)
            for index, value in enumerate(self.tie_breakers)
        )
        if not normalized_tie_breakers:
            raise OpeningSquadObjectiveError("tie_breakers must not be empty.")
        object.__setattr__(self, "tie_breakers", normalized_tie_breakers)
        object.__setattr__(
            self,
            "recommendation_status",
            _text(self.recommendation_status, "recommendation_status"),
        )
        if self.recommendation_status != RECOMMENDATION_PREVIEW_ONLY:
            raise OpeningSquadObjectiveError(
                "Fast Lane opening-squad objective must remain preview_only."
            )
        object.__setattr__(self, "writes_enabled", _bool(self.writes_enabled, "writes_enabled"))
        if self.writes_enabled:
            raise OpeningSquadObjectiveError(
                "Fast Lane opening-squad objective writes must remain disabled."
            )

    @property
    def requested_gameweeks(self) -> Tuple[int, ...]:
        return (1,) if self.horizon_mode == HORIZON_GW1_ONLY else (1, 2, 3, 4, 5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "policy_version": self.policy_version,
            "objective_name": self.objective_name,
            "target_season": self.target_season,
            "target_gw": self.target_gw,
            "horizon_mode": self.horizon_mode,
            "requested_gameweeks": list(self.requested_gameweeks),
            "gameweek_discounts": {
                str(key): value for key, value in sorted(self.gameweek_discounts.items())
            },
            "role_weights": self.role_weights.to_dict(),
            "risk_penalties": self.risk_penalties.to_dict(),
            "value_bank": self.value_bank.to_dict(),
            "allow_gw1_fallback": self.allow_gw1_fallback,
            "tie_breakers": list(self.tie_breakers),
            "recommendation_status": self.recommendation_status,
            "writes_enabled": self.writes_enabled,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "OpeningSquadObjectivePolicy":
        mapping = _mapping(value, "opening_squad_objective_policy")
        return cls(
            target_season=mapping.get("target_season"),
            horizon_mode=mapping.get("horizon_mode", HORIZON_GW1_GW5),
            target_gw=mapping.get("target_gw", 1),
            contract_version=mapping.get("contract_version", CONTRACT_VERSION),
            policy_version=mapping.get("policy_version", DEFAULT_POLICY_VERSION),
            objective_name=mapping.get(
                "objective_name", "discounted_opening_squad_expected_points"
            ),
            gameweek_discounts=mapping.get(
                "gameweek_discounts",
                {1: 1.0, 2: 0.85, 3: 0.7225, 4: 0.614125, 5: 0.52200625},
            ),
            role_weights=RoleWeightPolicy.from_mapping(mapping.get("role_weights", {})),
            risk_penalties=RiskPenaltyPolicy.from_mapping(
                mapping.get("risk_penalties", {})
            ),
            value_bank=ValueBankPolicy.from_mapping(mapping.get("value_bank", {})),
            allow_gw1_fallback=mapping.get("allow_gw1_fallback", True),
            tie_breakers=tuple(mapping.get("tie_breakers", DEFAULT_TIE_BREAKERS)),
            recommendation_status=mapping.get(
                "recommendation_status", RECOMMENDATION_PREVIEW_ONLY
            ),
            writes_enabled=mapping.get("writes_enabled", False),
        )

    @classmethod
    def from_json(cls, value: str) -> "OpeningSquadObjectivePolicy":
        try:
            payload = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise OpeningSquadObjectiveError(
                "Opening-squad objective JSON is invalid: %s" % exc
            )
        return cls.from_mapping(payload)


def build_default_opening_squad_objective_policy(
    target_season: str,
    horizon_mode: str = HORIZON_GW1_GW5,
) -> OpeningSquadObjectivePolicy:
    return OpeningSquadObjectivePolicy(
        target_season=target_season,
        horizon_mode=horizon_mode,
    )


def _normalize_projection_row(
    row: Mapping[str, Any],
    index: int,
) -> Dict[str, Any]:
    mapping = _mapping(row, "projection_rows[%s]" % index)
    label = "projection_rows[%s]" % index
    player_id = _player_id(mapping.get("player_id"), "%s.player_id" % label)
    target_gw = _int(mapping.get("target_gw"), "%s.target_gw" % label, 1, 38)
    predicted_points = _float(
        mapping.get("predicted_points"), "%s.predicted_points" % label
    )
    expected_minutes = _float(
        mapping.get("expected_minutes"),
        "%s.expected_minutes" % label,
        0.0,
        120.0,
    )
    start_probability = _probability(
        mapping.get("start_probability"), "%s.start_probability" % label
    )
    appearance_probability = _probability(
        mapping.get("appearance_probability"),
        "%s.appearance_probability" % label,
    )
    if start_probability > appearance_probability + EPSILON:
        raise OpeningSquadObjectiveError(
            "%s.start_probability cannot exceed appearance_probability." % label
        )

    if not _bool(
        mapping.get("selection_eligible"), "%s.selection_eligible" % label
    ):
        raise OpeningSquadObjectiveError("%s is not selection eligible." % label)

    has_fixture = _bool(mapping.get("has_fixture"), "%s.has_fixture" % label)
    fallback_used = _bool(
        mapping.get("fallback_used"), "%s.fallback_used" % label
    )
    raw_fallback_level = mapping.get("fallback_level")
    fallback_level = (
        0
        if raw_fallback_level is None
        else _int(raw_fallback_level, "%s.fallback_level" % label, 0)
    )
    if fallback_used and fallback_level <= 0:
        raise OpeningSquadObjectiveError(
            "%s fallback_used=True requires fallback_level > 0." % label
        )

    lower_raw = mapping.get("uncertainty_lower")
    upper_raw = mapping.get("uncertainty_upper")
    if (lower_raw is None) != (upper_raw is None):
        raise OpeningSquadObjectiveError(
            "%s uncertainty bounds must both be present or both be absent." % label
        )
    lower = None if lower_raw is None else _float(
        lower_raw, "%s.uncertainty_lower" % label
    )
    upper = None if upper_raw is None else _float(
        upper_raw, "%s.uncertainty_upper" % label
    )
    if lower is not None and upper is not None and lower > upper:
        raise OpeningSquadObjectiveError(
            "%s uncertainty_lower cannot exceed uncertainty_upper." % label
        )

    now_cost = _int(mapping.get("now_cost"), "%s.now_cost" % label, 1)
    position = _text(mapping.get("position"), "%s.position" % label).upper()
    if position not in ("GKP", "DEF", "MID", "FWD"):
        raise OpeningSquadObjectiveError("%s.position is invalid." % label)

    risk_flags = tuple(
        _text(value, "%s.risk_flags[%s]" % (label, flag_index))
        for flag_index, value in enumerate(
            _sequence(mapping.get("risk_flags", []), "%s.risk_flags" % label)
        )
    )

    return {
        "player_id": player_id,
        "target_gw": target_gw,
        "predicted_points": predicted_points,
        "expected_minutes": expected_minutes,
        "start_probability": start_probability,
        "appearance_probability": appearance_probability,
        "has_fixture": has_fixture,
        "fallback_used": fallback_used,
        "fallback_level": fallback_level,
        "uncertainty_lower": lower,
        "uncertainty_upper": upper,
        "now_cost": now_cost,
        "position": position,
        "risk_flags": risk_flags,
        "readiness_status": mapping.get("readiness_status"),
        "production_ready": mapping.get("production_ready"),
    }


def _normalize_projection_rows(
    projection_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[Tuple[Any, int], Dict[str, Any]], Dict[Any, str]]:
    rows = _sequence(projection_rows, "projection_rows")
    by_key: Dict[Tuple[Any, int], Dict[str, Any]] = {}
    positions: Dict[Any, str] = {}
    for index, row in enumerate(rows):
        normalized = _normalize_projection_row(row, index)
        key = (normalized["player_id"], normalized["target_gw"])
        if key in by_key:
            raise OpeningSquadObjectiveError(
                "Duplicate projection row for player_id=%s target_gw=%s." % key
            )
        by_key[key] = normalized
        existing = positions.get(normalized["player_id"])
        if existing is not None and existing != normalized["position"]:
            raise OpeningSquadObjectiveError(
                "Player %s has inconsistent positions across Gameweeks."
                % normalized["player_id"]
            )
        positions[normalized["player_id"]] = normalized["position"]
    if not by_key:
        raise OpeningSquadObjectiveError("projection_rows must not be empty.")
    return by_key, positions


def _normalize_plan(
    *,
    starting_player_ids: Sequence[Any],
    bench_order: Sequence[Any],
    captain_player_id: Any,
    vice_captain_player_id: Any,
    positions: Mapping[Any, str],
) -> Tuple[List[Any], List[Any], Dict[Any, Dict[str, Any]]]:
    starters = [
        _player_id(value, "starting_player_ids[%s]" % index)
        for index, value in enumerate(
            _sequence(starting_player_ids, "starting_player_ids")
        )
    ]
    bench = [
        _player_id(value, "bench_order[%s]" % index)
        for index, value in enumerate(_sequence(bench_order, "bench_order"))
    ]
    captain = _player_id(captain_player_id, "captain_player_id")
    vice = _player_id(vice_captain_player_id, "vice_captain_player_id")

    all_ids = starters + bench
    if len(set(all_ids)) != len(all_ids):
        raise OpeningSquadObjectiveError(
            "Starting and bench assignments must contain unique player IDs."
        )
    if captain not in starters:
        raise OpeningSquadObjectiveError("captain_player_id must be a starter.")
    if vice not in starters:
        raise OpeningSquadObjectiveError(
            "vice_captain_player_id must be a starter."
        )
    if captain == vice:
        raise OpeningSquadObjectiveError(
            "Captain and vice-captain must be different players."
        )

    missing = [player_id for player_id in all_ids if player_id not in positions]
    if missing:
        raise OpeningSquadObjectiveError(
            "Plan contains players with no projection rows: %s." % missing
        )
    if bench:
        if positions[bench[0]] != "GKP":
            raise OpeningSquadObjectiveError(
                "bench_order[0] must be the substitute goalkeeper."
            )
        misplaced = [
            player_id for player_id in bench[1:] if positions[player_id] == "GKP"
        ]
        if misplaced:
            raise OpeningSquadObjectiveError(
                "Outfield bench slots contain goalkeeper(s): %s." % misplaced
            )

    assignments: Dict[Any, Dict[str, Any]] = {}
    for player_id in starters:
        assignments[player_id] = {
            "role": "starter",
            "bench_index": None,
            "is_captain": player_id == captain,
            "is_vice_captain": player_id == vice,
        }
    for index, player_id in enumerate(bench):
        assignments[player_id] = {
            "role": "bench_goalkeeper" if index == 0 else "bench_outfield_%s" % index,
            "bench_index": index,
            "is_captain": False,
            "is_vice_captain": False,
        }
    return starters, bench, assignments


def _role_weight(
    policy: OpeningSquadObjectivePolicy,
    assignment: Mapping[str, Any],
) -> float:
    if assignment["role"] == "starter":
        weight = policy.role_weights.starter_weight
        if assignment["is_captain"]:
            weight += policy.role_weights.captain_bonus_weight
        if assignment["is_vice_captain"]:
            weight += policy.role_weights.vice_captain_bonus_weight
        return weight
    if assignment["role"] == "bench_goalkeeper":
        return policy.role_weights.bench_goalkeeper_weight
    outfield_index = int(assignment["bench_index"]) - 1
    if outfield_index < 0 or outfield_index >= 3:
        raise OpeningSquadObjectiveError(
            "Outfield bench index must be between 1 and 3."
        )
    return policy.role_weights.bench_outfield_weights[outfield_index]


def _effective_gameweeks(
    policy: OpeningSquadObjectivePolicy,
    player_ids: Sequence[Any],
    projections: Mapping[Tuple[Any, int], Mapping[str, Any]],
) -> Tuple[Tuple[int, ...], bool, List[Dict[str, Any]]]:
    missing = [
        {"player_id": player_id, "target_gw": gw}
        for player_id in player_ids
        for gw in policy.requested_gameweeks
        if (player_id, gw) not in projections
    ]
    if not missing:
        return policy.requested_gameweeks, False, []
    gw1_missing = [item for item in missing if item["target_gw"] == 1]
    if gw1_missing:
        raise OpeningSquadObjectiveError(
            "GW1 projections are missing for selected players: %s." % gw1_missing
        )
    if policy.horizon_mode == HORIZON_GW1_GW5 and policy.allow_gw1_fallback:
        return (1,), True, missing
    raise OpeningSquadObjectiveError(
        "Requested horizon is incomplete and GW1 fallback is disabled: %s."
        % missing
    )


def _empty_totals() -> Dict[str, float]:
    return {
        "gross_expected_points": 0.0,
        "minutes_risk_penalty": 0.0,
        "start_risk_penalty": 0.0,
        "fallback_penalty": 0.0,
        "uncertainty_penalty": 0.0,
        "value_bonus": 0.0,
        "bank_bonus": 0.0,
        "objective_value": 0.0,
    }


def _add_totals(
    totals: Dict[str, float],
    contribution: Mapping[str, float],
) -> None:
    for key in (
        "gross_expected_points",
        "minutes_risk_penalty",
        "start_risk_penalty",
        "fallback_penalty",
        "uncertainty_penalty",
        "value_bonus",
    ):
        totals[key] += float(contribution[key])


def _finalize_totals(totals: Dict[str, float]) -> Dict[str, float]:
    result = {key: _round(value) for key, value in totals.items()}
    result["total_penalty"] = _round(
        result["minutes_risk_penalty"]
        + result["start_risk_penalty"]
        + result["fallback_penalty"]
        + result["uncertainty_penalty"]
    )
    result["objective_value"] = _round(
        result["gross_expected_points"]
        - result["total_penalty"]
        + result["value_bonus"]
        + result["bank_bonus"]
    )
    return result


def evaluate_opening_squad_objective(
    policy: OpeningSquadObjectivePolicy,
    projection_rows: Sequence[Mapping[str, Any]],
    *,
    starting_player_ids: Sequence[Any],
    bench_order: Sequence[Any],
    captain_player_id: Any,
    vice_captain_player_id: Any,
    bank_units: int,
) -> Dict[str, Any]:
    """Evaluate one Day100A-validated plan using an explicit objective."""

    if not isinstance(policy, OpeningSquadObjectivePolicy):
        raise OpeningSquadObjectiveError(
            "policy must be an OpeningSquadObjectivePolicy."
        )
    normalized_bank = _int(bank_units, "bank_units", 0)
    projections, positions = _normalize_projection_rows(projection_rows)
    starters, bench, assignments = _normalize_plan(
        starting_player_ids=starting_player_ids,
        bench_order=bench_order,
        captain_player_id=captain_player_id,
        vice_captain_player_id=vice_captain_player_id,
        positions=positions,
    )
    selected_ids = starters + bench
    effective_gws, horizon_fallback_used, missing_horizon_rows = (
        _effective_gameweeks(policy, selected_ids, projections)
    )

    totals = _empty_totals()
    by_gameweek: Dict[str, Dict[str, Any]] = {}
    by_player: Dict[str, Dict[str, Any]] = {}
    review_reasons: List[Dict[str, Any]] = []

    if horizon_fallback_used:
        review_reasons.append(
            {
                "code": "gw1_gw5_horizon_incomplete",
                "message": (
                    "Primary GW1-GW5 objective fell back to GW1-only "
                    "because future projection rows are incomplete."
                ),
                "missing_projection_rows": missing_horizon_rows,
            }
        )

    for gw in effective_gws:
        gw_totals = _empty_totals()
        discount = float(policy.gameweek_discounts[gw])

        for player_id in selected_ids:
            row = projections[(player_id, gw)]
            assignment = assignments[player_id]
            role_weight = _role_weight(policy, assignment)
            weighted_discount = role_weight * discount

            gross = row["predicted_points"] * weighted_discount

            if not row["has_fixture"]:
                review_reasons.append(
                    {
                        "code": "player_fixture_missing_or_unconfirmed",
                        "message": (
                            "The projection row has has_fixture=False; "
                            "the supplied predicted points remain visible "
                            "for reconciliation, but the recommendation "
                            "requires explicit fixture review."
                        ),
                        "player_id": player_id,
                        "target_gw": gw,
                        "role": assignment["role"],
                        "predicted_points": _round(row["predicted_points"]),
                    }
                )

            minute_shortfall = max(
                0.0,
                policy.risk_penalties.expected_minutes_target
                - row["expected_minutes"],
            )
            minutes_penalty = (
                minute_shortfall
                * policy.risk_penalties.minutes_shortfall_penalty_per_minute
                * weighted_discount
            )
            start_shortfall = max(
                0.0,
                policy.risk_penalties.start_probability_target
                - row["start_probability"],
            )
            start_penalty = (
                start_shortfall
                * policy.risk_penalties.start_probability_shortfall_penalty
                * weighted_discount
            )

            fallback_penalty = 0.0
            if row["fallback_used"]:
                fallback_penalty = (
                    policy.risk_penalties.fallback_used_penalty
                    + row["fallback_level"]
                    * policy.risk_penalties.fallback_level_penalty
                ) * weighted_discount
                review_reasons.append(
                    {
                        "code": "player_projection_fallback_used",
                        "player_id": player_id,
                        "target_gw": gw,
                        "fallback_level": row["fallback_level"],
                    }
                )

            if (
                row["uncertainty_lower"] is None
                or row["uncertainty_upper"] is None
            ):
                uncertainty_width = None
                uncertainty_penalty = (
                    policy.risk_penalties.missing_uncertainty_penalty
                    * weighted_discount
                )
                review_reasons.append(
                    {
                        "code": "player_uncertainty_missing",
                        "player_id": player_id,
                        "target_gw": gw,
                    }
                )
            else:
                uncertainty_width = (
                    row["uncertainty_upper"] - row["uncertainty_lower"]
                )
                uncertainty_penalty = (
                    uncertainty_width
                    * policy.risk_penalties.uncertainty_width_penalty
                    * weighted_discount
                )

            value_bonus = (
                row["predicted_points"]
                / float(max(row["now_cost"], 1))
                * policy.value_bank.value_bonus_weight
                * weighted_discount
            )
            net = (
                gross
                - minutes_penalty
                - start_penalty
                - fallback_penalty
                - uncertainty_penalty
                + value_bonus
            )

            contribution = {
                "player_id": player_id,
                "target_gw": gw,
                "role": assignment["role"],
                "is_captain": assignment["is_captain"],
                "is_vice_captain": assignment["is_vice_captain"],
                "role_weight": _round(role_weight),
                "gameweek_discount": _round(discount),
                "predicted_points": _round(row["predicted_points"]),
                "expected_minutes": _round(row["expected_minutes"]),
                "start_probability": _round(row["start_probability"]),
                "has_fixture": row["has_fixture"],
                "fallback_used": row["fallback_used"],
                "fallback_level": row["fallback_level"],
                "uncertainty_width": (
                    None
                    if uncertainty_width is None
                    else _round(uncertainty_width)
                ),
                "gross_expected_points": _round(gross),
                "minutes_risk_penalty": _round(minutes_penalty),
                "start_risk_penalty": _round(start_penalty),
                "fallback_penalty": _round(fallback_penalty),
                "uncertainty_penalty": _round(uncertainty_penalty),
                "value_bonus": _round(value_bonus),
                "net_contribution": _round(net),
            }
            _add_totals(totals, contribution)
            _add_totals(gw_totals, contribution)

            player_key = str(player_id)
            if player_key not in by_player:
                by_player[player_key] = {
                    "player_id": player_id,
                    "role": assignment["role"],
                    "is_captain": assignment["is_captain"],
                    "is_vice_captain": assignment["is_vice_captain"],
                    "contributions": [],
                    "totals": _empty_totals(),
                }
            by_player[player_key]["contributions"].append(contribution)
            _add_totals(by_player[player_key]["totals"], contribution)

        by_gameweek[str(gw)] = {
            "target_gw": gw,
            "discount": _round(discount),
            "totals": _finalize_totals(gw_totals),
        }

    totals["bank_bonus"] = (
        normalized_bank * policy.value_bank.bank_bonus_per_unit
    )
    finalized_totals = _finalize_totals(totals)
    for player_payload in by_player.values():
        player_payload["totals"] = _finalize_totals(
            player_payload["totals"]
        )

    review_reasons = sorted(
        review_reasons,
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("player_id", "")),
            int(item.get("target_gw", 0) or 0),
        ),
    )

    result: Dict[str, Any] = {
        "contract_version": policy.contract_version,
        "policy_version": policy.policy_version,
        "objective_name": policy.objective_name,
        "target_season": policy.target_season,
        "target_gw": policy.target_gw,
        "requested_horizon_mode": policy.horizon_mode,
        "requested_gameweeks": list(policy.requested_gameweeks),
        "effective_gameweeks": list(effective_gws),
        "horizon_fallback_used": horizon_fallback_used,
        "recommendation_status": policy.recommendation_status,
        "writes_enabled": policy.writes_enabled,
        "manual_review_required": bool(review_reasons),
        "manual_review_reasons": review_reasons,
        "plan": {
            "starting_player_ids": list(starters),
            "bench_order": list(bench),
            "captain_player_id": captain_player_id,
            "vice_captain_player_id": vice_captain_player_id,
            "bank_units": normalized_bank,
            "canonical_player_ids": sorted(
                str(player_id) for player_id in selected_ids
            ),
        },
        "policy": policy.to_dict(),
        "totals": finalized_totals,
        "by_gameweek": by_gameweek,
        "by_player": {key: by_player[key] for key in sorted(by_player)},
        "explanation": {
            "primary_objective": (
                "discounted GW1-GW5 expected points"
                if policy.horizon_mode == HORIZON_GW1_GW5
                else "GW1 expected points"
            ),
            "fallback_objective": (
                "GW1 expected points with explicit manual review"
            ),
            "captain_treatment": (
                "captain receives starter weight plus captain bonus weight"
            ),
            "vice_captain_treatment": (
                policy.role_weights.vice_captain_treatment
            ),
            "bench_treatment": {
                "slot_0": "substitute goalkeeper weight",
                "slots_1_to_3": "ordered outfield substitute weights",
            },
            "risk_treatment": (
                "minutes, start probability, fallback, and uncertainty "
                "penalties are subtracted after role and GW weighting"
            ),
            "fixture_treatment": (
                "has_fixture=False is never silent: supplied prediction "
                "values remain in the decomposition, while the result is "
                "marked for explicit fixture/manual review"
            ),
            "budget_treatment": (
                "value and bank utility are optional; Fast Lane defaults "
                "are zero and do not force full-budget spending"
            ),
            "tie_breakers": list(policy.tie_breakers),
            "formula": (
                "objective = gross_expected_points - total_penalty "
                "+ value_bonus + bank_bonus"
            ),
        },
    }
    result["reconciliation"] = reconcile_objective_evaluation(result)
    result["tie_break_key"] = list(
        deterministic_objective_sort_key(result)
    )
    return result


def reconcile_objective_evaluation(
    evaluation: Mapping[str, Any],
) -> Dict[str, Any]:
    mapping = _mapping(evaluation, "evaluation")
    totals = _mapping(mapping.get("totals"), "evaluation.totals")
    gross = _float(
        totals.get("gross_expected_points"),
        "totals.gross_expected_points",
    )
    penalties = sum(
        _float(totals.get(field_name), "totals.%s" % field_name)
        for field_name in (
            "minutes_risk_penalty",
            "start_risk_penalty",
            "fallback_penalty",
            "uncertainty_penalty",
        )
    )
    value_bonus = _float(totals.get("value_bonus"), "totals.value_bonus")
    bank_bonus = _float(totals.get("bank_bonus"), "totals.bank_bonus")
    reported = _float(
        totals.get("objective_value"), "totals.objective_value"
    )
    recomputed = _round(gross - penalties + value_bonus + bank_bonus)
    difference = _round(reported - recomputed)
    return {
        "passed": abs(difference) <= EPSILON,
        "reported_objective_value": _round(reported),
        "recomputed_objective_value": recomputed,
        "difference": difference,
        "formula": (
            "gross_expected_points - minutes_risk_penalty "
            "- start_risk_penalty - fallback_penalty "
            "- uncertainty_penalty + value_bonus + bank_bonus"
        ),
    }


def deterministic_objective_sort_key(
    evaluation: Mapping[str, Any],
) -> Tuple[Any, ...]:
    """Return an ascending Python sort key for deterministic maximization."""

    mapping = _mapping(evaluation, "evaluation")
    totals = _mapping(mapping.get("totals"), "evaluation.totals")
    by_gameweek = _mapping(
        mapping.get("by_gameweek"), "evaluation.by_gameweek"
    )
    plan = _mapping(mapping.get("plan"), "evaluation.plan")
    policy = _mapping(mapping.get("policy"), "evaluation.policy")
    value_bank = _mapping(
        policy.get("value_bank"), "evaluation.policy.value_bank"
    )
    gw1_payload = _mapping(
        by_gameweek.get("1"), "evaluation.by_gameweek[1]"
    )
    gw1_totals = _mapping(
        gw1_payload.get("totals"),
        "evaluation.by_gameweek[1].totals",
    )

    bank_key = 0
    if bool(value_bank.get("use_bank_as_tie_breaker")):
        bank_key = -_int(
            plan.get("bank_units"), "evaluation.plan.bank_units", 0
        )
    canonical_ids = tuple(
        str(value)
        for value in _sequence(
            plan.get("canonical_player_ids"),
            "evaluation.plan.canonical_player_ids",
        )
    )
    return (
        -_float(
            totals.get("objective_value"),
            "evaluation.totals.objective_value",
        ),
        -_float(
            gw1_totals.get("objective_value"),
            "evaluation.by_gameweek[1].totals.objective_value",
        ),
        _float(
            totals.get("total_penalty"),
            "evaluation.totals.total_penalty",
        ),
        bank_key,
        canonical_ids,
    )
