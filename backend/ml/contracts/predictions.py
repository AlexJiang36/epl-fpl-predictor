"""Versioned prediction and decision output contracts.

Day76B keeps downstream consumers independent from model-specific CSV columns.
The contracts are pure Python, filesystem/database agnostic, and Python 3.9
compatible.  Legacy preview adapters are explicit and never upgrade preview
outputs to production-ready status.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PREDICTION_CONTRACT_VERSION = "fpl_prediction_contract_v1"

OUTPUT_MATCH = "match_prediction"
OUTPUT_SCORELINE = "scoreline_prediction"
OUTPUT_MINUTES = "minutes_prediction"
OUTPUT_PLAYER_EVENT = "player_event_prediction"
OUTPUT_PLAYER_POINTS = "player_points_prediction"
OUTPUT_RANKING = "ranking_output"
OUTPUT_DECISION = "decision_output"

SUPPORTED_OUTPUT_TYPES = (
    OUTPUT_MATCH,
    OUTPUT_SCORELINE,
    OUTPUT_MINUTES,
    OUTPUT_PLAYER_EVENT,
    OUTPUT_PLAYER_POINTS,
    OUTPUT_RANKING,
    OUTPUT_DECISION,
)

READINESS_PREVIEW_ONLY = "preview_only"
READINESS_VALIDATED = "validated"
READINESS_APPROVED = "approved"
READINESS_PRODUCTION = "production"
SUPPORTED_READINESS_STATUSES = (
    READINESS_PREVIEW_ONLY,
    READINESS_VALIDATED,
    READINESS_APPROVED,
    READINESS_PRODUCTION,
)

POSITION_VALUES = ("GKP", "DEF", "MID", "FWD")
RESULT_LABELS = ("H", "D", "A")

_SEASON_RE = re.compile(r"^\d{4}_\d{2}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class PredictionContractError(ValueError):
    """Raised when a standard output or legacy adapter violates the contract."""


def _required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise PredictionContractError("%s is required." % field_name)
    text = str(value).strip()
    if not text:
        raise PredictionContractError("%s must be non-empty." % field_name)
    return text


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _identifier(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if not _IDENTIFIER_RE.match(text):
        raise PredictionContractError(
            "%s must contain only letters, digits, underscore, dot, colon, or hyphen."
            % field_name
        )
    return text


def _season(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if not _SEASON_RE.match(text):
        raise PredictionContractError("%s must use YYYY_YY format." % field_name)
    start_year = int(text[:4])
    end_year = int(text[-2:])
    if (start_year + 1) % 100 != end_year:
        raise PredictionContractError("%s must represent consecutive seasons." % field_name)
    return text


def _int_value(
    value: Any,
    field_name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    optional: bool = False,
) -> Optional[int]:
    if value is None or value == "":
        if optional:
            return None
        raise PredictionContractError("%s is required." % field_name)
    if isinstance(value, bool):
        raise PredictionContractError("%s must be an integer, not bool." % field_name)
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise PredictionContractError("%s must be an integer." % field_name)
    if minimum is not None and number < minimum:
        raise PredictionContractError("%s must be >= %s." % (field_name, minimum))
    if maximum is not None and number > maximum:
        raise PredictionContractError("%s must be <= %s." % (field_name, maximum))
    return number


def _float_value(
    value: Any,
    field_name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    optional: bool = False,
) -> Optional[float]:
    if value is None or value == "":
        if optional:
            return None
        raise PredictionContractError("%s is required." % field_name)
    if isinstance(value, bool):
        raise PredictionContractError("%s must be numeric, not bool." % field_name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise PredictionContractError("%s must be numeric." % field_name)
    if not math.isfinite(number):
        raise PredictionContractError("%s must be finite." % field_name)
    if minimum is not None and number < minimum:
        raise PredictionContractError("%s must be >= %s." % (field_name, minimum))
    if maximum is not None and number > maximum:
        raise PredictionContractError("%s must be <= %s." % (field_name, maximum))
    return number


def _probability(value: Any, field_name: str, optional: bool = False) -> Optional[float]:
    return _float_value(value, field_name, 0.0, 1.0, optional=optional)


def _strict_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes"):
            return True
        if normalized in ("false", "0", "no"):
            return False
    raise PredictionContractError("%s must be boolean." % field_name)


def _legacy_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return _strict_bool(value, "legacy_bool")


def _utc_iso(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise PredictionContractError("%s must be ISO-8601." % field_name)
    if parsed.tzinfo is None:
        raise PredictionContractError("%s must be timezone-aware." % field_name)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_seasons(value: Any) -> Tuple[str, ...]:
    values: List[Any]
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        raise PredictionContractError("source_seasons must be a sequence or comma-separated string.")
    if not values:
        raise PredictionContractError("source_seasons must be non-empty.")
    normalized = tuple(_season(item, "source_seasons") for item in values)
    if len(set(normalized)) != len(normalized):
        raise PredictionContractError("source_seasons must not contain duplicates.")
    return normalized


def _string_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    if value is None or value == "":
        return tuple()
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        raw = list(value)
    else:
        raise PredictionContractError("%s must be a sequence or comma-separated string." % field_name)
    return tuple(_required_text(item, field_name) for item in raw)


def _text_mapping(value: Any, field_name: str) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PredictionContractError("%s must be a mapping." % field_name)
    out: Dict[str, str] = {}
    for key, item in value.items():
        out[_identifier(key, "%s key" % field_name)] = _required_text(
            item, "%s value" % field_name
        )
    return dict(sorted(out.items()))


def _json_safe_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PredictionContractError("%s must be a mapping." % field_name)
    return dict(value)


def _mapping_from_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise PredictionContractError("Expected a mapping or object with to_dict().")


def _dataclass_to_dict(value: Any) -> Dict[str, Any]:
    result = asdict(value)
    return result


@dataclass(frozen=True)
class PredictionContext:
    """Shared provenance and scope for every standard output row."""

    output_type: str
    source_seasons: Tuple[str, ...]
    target_season: str
    target_gw: Optional[int]
    as_of_time: str
    prediction_mode: str
    prediction_scope: str
    run_id: str
    model_name: str
    model_version: str
    feature_version: Optional[str] = None
    rules_versions: Mapping[str, str] = field(default_factory=dict)
    contract_version: str = PREDICTION_CONTRACT_VERSION
    source_artifact_version: Optional[str] = None

    def __post_init__(self) -> None:
        if self.output_type not in SUPPORTED_OUTPUT_TYPES:
            raise PredictionContractError("Unsupported output_type=%s." % self.output_type)
        object.__setattr__(self, "source_seasons", _source_seasons(self.source_seasons))
        object.__setattr__(self, "target_season", _season(self.target_season, "target_season"))
        object.__setattr__(
            self,
            "target_gw",
            _int_value(self.target_gw, "target_gw", 1, 38, optional=True),
        )
        object.__setattr__(self, "as_of_time", _utc_iso(self.as_of_time, "as_of_time"))
        object.__setattr__(self, "prediction_mode", _identifier(self.prediction_mode, "prediction_mode"))
        object.__setattr__(self, "prediction_scope", _identifier(self.prediction_scope, "prediction_scope"))
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "model_name", _identifier(self.model_name, "model_name"))
        object.__setattr__(self, "model_version", _identifier(self.model_version, "model_version"))
        object.__setattr__(self, "feature_version", _optional_text(self.feature_version))
        object.__setattr__(self, "rules_versions", _text_mapping(self.rules_versions, "rules_versions"))
        object.__setattr__(self, "contract_version", _identifier(self.contract_version, "contract_version"))
        object.__setattr__(self, "source_artifact_version", _optional_text(self.source_artifact_version))

    @classmethod
    def from_run_metadata(
        cls,
        metadata: Any,
        output_type: str,
        model_name: str,
        model_version: str,
        prediction_scope: str,
    ) -> "PredictionContext":
        raw = _mapping_from_object(metadata)
        versions = raw.get("versions") or {}
        if not isinstance(versions, Mapping):
            raise PredictionContractError("run metadata versions must be a mapping.")
        scope = raw.get("scope") or {}
        if not isinstance(scope, Mapping):
            scope = {}
        source_seasons = raw.get("source_seasons") or scope.get("source_seasons")
        target_season = raw.get("target_season") or scope.get("season")
        target_gw = raw.get("target_gw")
        if target_gw is None:
            target_gw = scope.get("gw")
        return cls(
            output_type=output_type,
            source_seasons=_source_seasons(source_seasons),
            target_season=target_season,
            target_gw=target_gw,
            as_of_time=raw.get("as_of_time_utc") or raw.get("as_of_time"),
            prediction_mode=raw.get("prediction_mode"),
            prediction_scope=prediction_scope,
            run_id=raw.get("run_id"),
            model_name=model_name,
            model_version=model_version,
            feature_version=versions.get("feature_version"),
            rules_versions=versions.get("rules_versions") or {},
            contract_version=PREDICTION_CONTRACT_VERSION,
            source_artifact_version=versions.get("artifact_version"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class SafetyMetadata:
    """Required uncertainty, fallback, calibration, and readiness metadata."""

    data_quality_status: str
    calibration_status: str
    guardrail_status: str
    readiness_status: str = READINESS_PREVIEW_ONLY
    confidence_score: Optional[float] = None
    confidence_label: Optional[str] = None
    uncertainty_lower: Optional[float] = None
    uncertainty_upper: Optional[float] = None
    fallback_used: bool = False
    fallback_level: Optional[int] = None
    fallback_reason: Optional[str] = None
    risk_flags: Tuple[str, ...] = field(default_factory=tuple)
    production_ready: bool = False
    prediction_write_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_quality_status", _identifier(self.data_quality_status, "data_quality_status"))
        object.__setattr__(self, "calibration_status", _identifier(self.calibration_status, "calibration_status"))
        object.__setattr__(self, "guardrail_status", _identifier(self.guardrail_status, "guardrail_status"))
        if self.readiness_status not in SUPPORTED_READINESS_STATUSES:
            raise PredictionContractError(
                "Unsupported readiness_status=%s." % self.readiness_status
            )
        object.__setattr__(
            self,
            "confidence_score",
            _probability(self.confidence_score, "confidence_score", optional=True),
        )
        object.__setattr__(self, "confidence_label", _optional_text(self.confidence_label))
        object.__setattr__(
            self,
            "uncertainty_lower",
            _float_value(self.uncertainty_lower, "uncertainty_lower", optional=True),
        )
        object.__setattr__(
            self,
            "uncertainty_upper",
            _float_value(self.uncertainty_upper, "uncertainty_upper", optional=True),
        )
        if (
            self.uncertainty_lower is not None
            and self.uncertainty_upper is not None
            and self.uncertainty_lower > self.uncertainty_upper
        ):
            raise PredictionContractError("uncertainty_lower cannot exceed uncertainty_upper.")
        object.__setattr__(self, "fallback_used", _strict_bool(self.fallback_used, "fallback_used"))
        object.__setattr__(
            self,
            "fallback_level",
            _int_value(self.fallback_level, "fallback_level", 0, optional=True),
        )
        object.__setattr__(self, "fallback_reason", _optional_text(self.fallback_reason))
        object.__setattr__(self, "risk_flags", _string_tuple(self.risk_flags, "risk_flags"))
        object.__setattr__(self, "production_ready", _strict_bool(self.production_ready, "production_ready"))
        object.__setattr__(
            self,
            "prediction_write_allowed",
            _strict_bool(self.prediction_write_allowed, "prediction_write_allowed"),
        )
        if self.prediction_write_allowed and not self.production_ready:
            raise PredictionContractError(
                "prediction_write_allowed=True requires production_ready=True."
            )
        if self.readiness_status == READINESS_PRODUCTION and not self.production_ready:
            raise PredictionContractError(
                "readiness_status=production requires production_ready=True."
            )
        if self.production_ready and self.readiness_status == READINESS_PREVIEW_ONLY:
            raise PredictionContractError(
                "production_ready=True is incompatible with preview_only readiness."
            )
        if self.fallback_used and self.fallback_level is None:
            raise PredictionContractError(
                "fallback_used=True requires fallback_level."
            )

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class PlayerPredictionManifestContract:
    """Stable preview-only view of the Day72B player manifest.

    ``preview_schema_consumption_allowed`` authorizes only explicit adaptation
    of the validated preview rows into the Day76B schema.  It does not turn the
    preview into an approved recommendation, public prediction, or writable
    production artifact.
    """

    context: PredictionContext
    manifest_version: str
    artifact_type: str
    passed: bool
    preview_schema_consumption_allowed: bool
    original_optimizer_input_ready: bool
    ops_preview_ready: bool
    blockers: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    artifact_fingerprints: Mapping[str, Any] = field(default_factory=dict)
    readiness_status: str = READINESS_PREVIEW_ONLY
    production_ready: bool = False
    prediction_write_allowed: bool = False

    def __post_init__(self) -> None:
        if self.context.output_type != OUTPUT_PLAYER_POINTS:
            raise PredictionContractError(
                "Player manifest requires output_type=%s." % OUTPUT_PLAYER_POINTS
            )
        object.__setattr__(
            self,
            "manifest_version",
            _identifier(self.manifest_version, "manifest_version"),
        )
        artifact_type = _identifier(self.artifact_type, "artifact_type")
        if artifact_type != "pre_gw1_player_prediction_manifest":
            raise PredictionContractError(
                "Unsupported Day72B artifact_type=%s." % artifact_type
            )
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(self, "passed", _strict_bool(self.passed, "passed"))
        object.__setattr__(
            self,
            "preview_schema_consumption_allowed",
            _strict_bool(
                self.preview_schema_consumption_allowed,
                "preview_schema_consumption_allowed",
            ),
        )
        object.__setattr__(
            self,
            "original_optimizer_input_ready",
            _strict_bool(
                self.original_optimizer_input_ready,
                "original_optimizer_input_ready",
            ),
        )
        object.__setattr__(
            self,
            "ops_preview_ready",
            _strict_bool(self.ops_preview_ready, "ops_preview_ready"),
        )
        object.__setattr__(self, "blockers", _string_tuple(self.blockers, "blockers"))
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, "warnings"))
        object.__setattr__(
            self,
            "artifact_fingerprints",
            _json_safe_mapping(self.artifact_fingerprints, "artifact_fingerprints"),
        )
        if self.readiness_status != READINESS_PREVIEW_ONLY:
            raise PredictionContractError(
                "Day72B manifest adapter must remain preview_only."
            )
        object.__setattr__(
            self,
            "production_ready",
            _strict_bool(self.production_ready, "production_ready"),
        )
        object.__setattr__(
            self,
            "prediction_write_allowed",
            _strict_bool(
                self.prediction_write_allowed, "prediction_write_allowed"
            ),
        )
        if self.production_ready or self.prediction_write_allowed:
            raise PredictionContractError(
                "Day72B preview manifest cannot be promoted to production or writes."
            )
        if self.preview_schema_consumption_allowed and (not self.passed or self.blockers):
            raise PredictionContractError(
                "Preview schema consumption requires a passing manifest with no blockers."
            )

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class MatchPredictionOutput:
    context: PredictionContext
    safety: SafetyMetadata
    fixture_id: int
    home_team_id: int
    away_team_id: int
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    predicted_result_label: str
    fpl_fixture_id: Optional[int] = None
    kickoff_time: Optional[str] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.context.output_type != OUTPUT_MATCH:
            raise PredictionContractError("Match output requires output_type=%s." % OUTPUT_MATCH)
        object.__setattr__(self, "fixture_id", _int_value(self.fixture_id, "fixture_id", 1))
        object.__setattr__(self, "home_team_id", _int_value(self.home_team_id, "home_team_id", 1))
        object.__setattr__(self, "away_team_id", _int_value(self.away_team_id, "away_team_id", 1))
        if self.home_team_id == self.away_team_id:
            raise PredictionContractError("home_team_id and away_team_id must differ.")
        probabilities = (
            _probability(self.home_win_probability, "home_win_probability"),
            _probability(self.draw_probability, "draw_probability"),
            _probability(self.away_win_probability, "away_win_probability"),
        )
        object.__setattr__(self, "home_win_probability", probabilities[0])
        object.__setattr__(self, "draw_probability", probabilities[1])
        object.__setattr__(self, "away_win_probability", probabilities[2])
        if abs(sum(probabilities) - 1.0) > 1e-5:
            raise PredictionContractError("Match probabilities must sum to 1.")
        label = _required_text(self.predicted_result_label, "predicted_result_label")
        if label not in RESULT_LABELS:
            raise PredictionContractError("predicted_result_label must be H, D, or A.")
        object.__setattr__(self, "predicted_result_label", label)
        object.__setattr__(self, "fpl_fixture_id", _int_value(self.fpl_fixture_id, "fpl_fixture_id", 1, optional=True))
        object.__setattr__(self, "kickoff_time", _optional_text(self.kickoff_time))
        object.__setattr__(self, "home_team_name", _optional_text(self.home_team_name))
        object.__setattr__(self, "away_team_name", _optional_text(self.away_team_name))

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class ScorelinePredictionOutput:
    context: PredictionContext
    safety: SafetyMetadata
    fixture_id: int
    home_team_id: int
    away_team_id: int
    expected_home_goals: float
    expected_away_goals: float
    predicted_scoreline: Optional[str] = None
    scoreline_home_win_probability: Optional[float] = None
    scoreline_draw_probability: Optional[float] = None
    scoreline_away_win_probability: Optional[float] = None
    fpl_fixture_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.context.output_type != OUTPUT_SCORELINE:
            raise PredictionContractError("Scoreline output requires output_type=%s." % OUTPUT_SCORELINE)
        object.__setattr__(self, "fixture_id", _int_value(self.fixture_id, "fixture_id", 1))
        object.__setattr__(self, "home_team_id", _int_value(self.home_team_id, "home_team_id", 1))
        object.__setattr__(self, "away_team_id", _int_value(self.away_team_id, "away_team_id", 1))
        object.__setattr__(self, "expected_home_goals", _float_value(self.expected_home_goals, "expected_home_goals", 0.0))
        object.__setattr__(self, "expected_away_goals", _float_value(self.expected_away_goals, "expected_away_goals", 0.0))
        object.__setattr__(self, "predicted_scoreline", _optional_text(self.predicted_scoreline))
        probability_values = (
            _probability(self.scoreline_home_win_probability, "scoreline_home_win_probability", optional=True),
            _probability(self.scoreline_draw_probability, "scoreline_draw_probability", optional=True),
            _probability(self.scoreline_away_win_probability, "scoreline_away_win_probability", optional=True),
        )
        object.__setattr__(self, "scoreline_home_win_probability", probability_values[0])
        object.__setattr__(self, "scoreline_draw_probability", probability_values[1])
        object.__setattr__(self, "scoreline_away_win_probability", probability_values[2])
        present = [value is not None for value in probability_values]
        if any(present) and not all(present):
            raise PredictionContractError("Scoreline probabilities must be all present or all absent.")
        if all(present) and abs(sum(value for value in probability_values if value is not None) - 1.0) > 1e-5:
            raise PredictionContractError("Scoreline probabilities must sum to 1.")
        object.__setattr__(self, "fpl_fixture_id", _int_value(self.fpl_fixture_id, "fpl_fixture_id", 1, optional=True))

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class MinutesPredictionOutput:
    context: PredictionContext
    safety: SafetyMetadata
    player_id: int
    appearance_probability: float
    start_probability: float
    conditional_minutes_if_appears: float
    expected_minutes: float
    minutes_lower_bound: float
    minutes_upper_bound: float
    fixture_id: Optional[int] = None
    fpl_player_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.context.output_type != OUTPUT_MINUTES:
            raise PredictionContractError("Minutes output requires output_type=%s." % OUTPUT_MINUTES)
        object.__setattr__(self, "player_id", _int_value(self.player_id, "player_id", 1))
        object.__setattr__(self, "appearance_probability", _probability(self.appearance_probability, "appearance_probability"))
        object.__setattr__(self, "start_probability", _probability(self.start_probability, "start_probability"))
        if self.start_probability > self.appearance_probability + 1e-8:
            raise PredictionContractError("start_probability cannot exceed appearance_probability.")
        object.__setattr__(self, "conditional_minutes_if_appears", _float_value(self.conditional_minutes_if_appears, "conditional_minutes_if_appears", 0.0, 120.0))
        object.__setattr__(self, "expected_minutes", _float_value(self.expected_minutes, "expected_minutes", 0.0, 120.0))
        object.__setattr__(self, "minutes_lower_bound", _float_value(self.minutes_lower_bound, "minutes_lower_bound", 0.0, 120.0))
        object.__setattr__(self, "minutes_upper_bound", _float_value(self.minutes_upper_bound, "minutes_upper_bound", 0.0, 120.0))
        if not (self.minutes_lower_bound <= self.expected_minutes <= self.minutes_upper_bound):
            raise PredictionContractError("expected_minutes must lie within its bounds.")
        object.__setattr__(self, "fixture_id", _int_value(self.fixture_id, "fixture_id", 1, optional=True))
        object.__setattr__(self, "fpl_player_id", _int_value(self.fpl_player_id, "fpl_player_id", 1, optional=True))

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class PlayerEventPredictionOutput:
    context: PredictionContext
    safety: SafetyMetadata
    player_id: int
    expected_goals: float
    expected_assists: float
    clean_sheet_probability: float
    expected_bonus: float
    expected_other_points: float
    fixture_id: Optional[int] = None
    fpl_player_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.context.output_type != OUTPUT_PLAYER_EVENT:
            raise PredictionContractError("Event output requires output_type=%s." % OUTPUT_PLAYER_EVENT)
        object.__setattr__(self, "player_id", _int_value(self.player_id, "player_id", 1))
        object.__setattr__(self, "expected_goals", _float_value(self.expected_goals, "expected_goals", 0.0))
        object.__setattr__(self, "expected_assists", _float_value(self.expected_assists, "expected_assists", 0.0))
        object.__setattr__(self, "clean_sheet_probability", _probability(self.clean_sheet_probability, "clean_sheet_probability"))
        object.__setattr__(self, "expected_bonus", _float_value(self.expected_bonus, "expected_bonus"))
        object.__setattr__(self, "expected_other_points", _float_value(self.expected_other_points, "expected_other_points"))
        object.__setattr__(self, "fixture_id", _int_value(self.fixture_id, "fixture_id", 1, optional=True))
        object.__setattr__(self, "fpl_player_id", _int_value(self.fpl_player_id, "fpl_player_id", 1, optional=True))

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class PlayerPointsPredictionOutput:
    """Stable player-points row consumed by rankings and opening-squad logic."""

    context: PredictionContext
    safety: SafetyMetadata
    player_id: int
    team_id: int
    position: str
    price: float
    now_cost: int
    predicted_points: float
    expected_minutes: float
    start_probability: float
    appearance_probability: float
    has_fixture: bool
    player_status: str
    status_cutoff_valid: bool
    status_hard_guardrail_applied: bool
    selection_eligible: bool
    eligibility_reason: str
    fixture_id: Optional[int] = None
    fpl_fixture_id: Optional[int] = None
    fpl_player_id: Optional[int] = None
    player_name: Optional[str] = None
    web_name: Optional[str] = None
    team_name: Optional[str] = None
    team_short_name: Optional[str] = None
    opponent_team_id: Optional[int] = None
    opponent_name: Optional[str] = None
    is_home: Optional[bool] = None
    minutes_lower_bound: Optional[float] = None
    minutes_upper_bound: Optional[float] = None
    raw_expected_points: Optional[float] = None
    expected_appearance_points: Optional[float] = None
    expected_goals: Optional[float] = None
    expected_goal_points: Optional[float] = None
    expected_assists: Optional[float] = None
    expected_assist_points: Optional[float] = None
    clean_sheet_probability: Optional[float] = None
    expected_clean_sheet_points: Optional[float] = None
    expected_bonus: Optional[float] = None
    expected_other_points: Optional[float] = None
    component_accounting_status: Optional[str] = None
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context.output_type != OUTPUT_PLAYER_POINTS:
            raise PredictionContractError("Player-points output requires output_type=%s." % OUTPUT_PLAYER_POINTS)
        object.__setattr__(self, "player_id", _int_value(self.player_id, "player_id", 1))
        object.__setattr__(self, "team_id", _int_value(self.team_id, "team_id", 1))
        position = _required_text(self.position, "position").upper()
        if position not in POSITION_VALUES:
            raise PredictionContractError("position must be one of %s." % (POSITION_VALUES,))
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "price", _float_value(self.price, "price", 0.0))
        object.__setattr__(self, "now_cost", _int_value(self.now_cost, "now_cost", 0))
        if abs(self.price * 10.0 - self.now_cost) > 0.51:
            raise PredictionContractError("price and now_cost are inconsistent.")
        object.__setattr__(self, "predicted_points", _float_value(self.predicted_points, "predicted_points"))
        object.__setattr__(self, "expected_minutes", _float_value(self.expected_minutes, "expected_minutes", 0.0, 120.0))
        object.__setattr__(self, "start_probability", _probability(self.start_probability, "start_probability"))
        object.__setattr__(self, "appearance_probability", _probability(self.appearance_probability, "appearance_probability"))
        if self.start_probability > self.appearance_probability + 1e-8:
            raise PredictionContractError("start_probability cannot exceed appearance_probability.")
        object.__setattr__(self, "has_fixture", _strict_bool(self.has_fixture, "has_fixture"))
        object.__setattr__(
            self, "player_status", _required_text(self.player_status, "player_status").lower()
        )
        object.__setattr__(
            self,
            "status_cutoff_valid",
            _strict_bool(self.status_cutoff_valid, "status_cutoff_valid"),
        )
        object.__setattr__(
            self,
            "status_hard_guardrail_applied",
            _strict_bool(
                self.status_hard_guardrail_applied,
                "status_hard_guardrail_applied",
            ),
        )
        object.__setattr__(
            self,
            "selection_eligible",
            _strict_bool(self.selection_eligible, "selection_eligible"),
        )
        object.__setattr__(
            self,
            "eligibility_reason",
            _identifier(self.eligibility_reason, "eligibility_reason"),
        )
        if self.selection_eligible and not self.status_cutoff_valid:
            raise PredictionContractError(
                "selection_eligible=True requires status_cutoff_valid=True."
            )
        if self.selection_eligible and self.status_hard_guardrail_applied:
            raise PredictionContractError(
                "selection_eligible=True is incompatible with a hard status guardrail."
            )
        object.__setattr__(self, "fixture_id", _int_value(self.fixture_id, "fixture_id", 1, optional=True))
        object.__setattr__(self, "fpl_fixture_id", _int_value(self.fpl_fixture_id, "fpl_fixture_id", 1, optional=True))
        object.__setattr__(self, "fpl_player_id", _int_value(self.fpl_player_id, "fpl_player_id", 1, optional=True))
        if self.has_fixture and self.fixture_id is None:
            raise PredictionContractError("has_fixture=True requires fixture_id.")
        object.__setattr__(self, "player_name", _optional_text(self.player_name))
        object.__setattr__(self, "web_name", _optional_text(self.web_name))
        object.__setattr__(self, "team_name", _optional_text(self.team_name))
        object.__setattr__(self, "team_short_name", _optional_text(self.team_short_name))
        object.__setattr__(self, "opponent_team_id", _int_value(self.opponent_team_id, "opponent_team_id", 1, optional=True))
        object.__setattr__(self, "opponent_name", _optional_text(self.opponent_name))
        if self.is_home is not None:
            object.__setattr__(self, "is_home", _strict_bool(self.is_home, "is_home"))
        for field_name in (
            "minutes_lower_bound",
            "minutes_upper_bound",
            "raw_expected_points",
            "expected_appearance_points",
            "expected_goals",
            "expected_goal_points",
            "expected_assists",
            "expected_assist_points",
            "clean_sheet_probability",
            "expected_clean_sheet_points",
            "expected_bonus",
            "expected_other_points",
        ):
            value = getattr(self, field_name)
            if field_name == "clean_sheet_probability":
                normalized = _probability(value, field_name, optional=True)
            elif field_name in ("minutes_lower_bound", "minutes_upper_bound"):
                normalized = _float_value(value, field_name, 0.0, 120.0, optional=True)
            elif field_name in ("expected_goals", "expected_assists"):
                normalized = _float_value(value, field_name, 0.0, optional=True)
            else:
                normalized = _float_value(value, field_name, optional=True)
            object.__setattr__(self, field_name, normalized)
        if (
            self.minutes_lower_bound is not None
            and self.minutes_upper_bound is not None
            and not self.minutes_lower_bound <= self.expected_minutes <= self.minutes_upper_bound
        ):
            raise PredictionContractError("expected_minutes must lie within minutes bounds.")
        object.__setattr__(self, "component_accounting_status", _optional_text(self.component_accounting_status))
        object.__setattr__(self, "extensions", _json_safe_mapping(self.extensions, "extensions"))

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)

    def to_optimizer_row(self) -> Dict[str, Any]:
        """Return the intentionally small stable input used by Fast Lane optimizers."""
        return {
            "contract_version": self.context.contract_version,
            "target_season": self.context.target_season,
            "target_gw": self.context.target_gw,
            "as_of_time": self.context.as_of_time,
            "prediction_mode": self.context.prediction_mode,
            "prediction_scope": self.context.prediction_scope,
            "run_id": self.context.run_id,
            "model_name": self.context.model_name,
            "model_version": self.context.model_version,
            "player_id": self.player_id,
            "fpl_player_id": self.fpl_player_id,
            "player_name": self.player_name,
            "web_name": self.web_name,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "team_short_name": self.team_short_name,
            "position": self.position,
            "price": self.price,
            "now_cost": self.now_cost,
            "fixture_id": self.fixture_id,
            "opponent_team_id": self.opponent_team_id,
            "opponent_name": self.opponent_name,
            "is_home": self.is_home,
            "has_fixture": self.has_fixture,
            "player_status": self.player_status,
            "status_cutoff_valid": self.status_cutoff_valid,
            "status_hard_guardrail_applied": self.status_hard_guardrail_applied,
            "selection_eligible": self.selection_eligible,
            "eligibility_reason": self.eligibility_reason,
            "appearance_probability": self.appearance_probability,
            "start_probability": self.start_probability,
            "expected_minutes": self.expected_minutes,
            "predicted_points": self.predicted_points,
            "uncertainty_lower": self.safety.uncertainty_lower,
            "uncertainty_upper": self.safety.uncertainty_upper,
            "confidence_score": self.safety.confidence_score,
            "confidence_label": self.safety.confidence_label,
            "fallback_used": self.safety.fallback_used,
            "fallback_level": self.safety.fallback_level,
            "risk_flags": list(self.safety.risk_flags),
            "data_quality_status": self.safety.data_quality_status,
            "calibration_status": self.safety.calibration_status,
            "guardrail_status": self.safety.guardrail_status,
            "readiness_status": self.safety.readiness_status,
            "production_ready": self.safety.production_ready,
        }


@dataclass(frozen=True)
class RankingOutput:
    context: PredictionContext
    safety: SafetyMetadata
    player_id: int
    position: str
    metric_name: str
    metric_value: float
    rank: int
    horizon_gws: Tuple[int, ...]
    fpl_player_id: Optional[int] = None
    explanation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context.output_type != OUTPUT_RANKING:
            raise PredictionContractError("Ranking output requires output_type=%s." % OUTPUT_RANKING)
        object.__setattr__(self, "player_id", _int_value(self.player_id, "player_id", 1))
        position = _required_text(self.position, "position").upper()
        if position not in POSITION_VALUES:
            raise PredictionContractError("Unsupported ranking position.")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "metric_name", _identifier(self.metric_name, "metric_name"))
        object.__setattr__(self, "metric_value", _float_value(self.metric_value, "metric_value"))
        object.__setattr__(self, "rank", _int_value(self.rank, "rank", 1))
        gws = tuple(_int_value(item, "horizon_gws", 1, 38) for item in self.horizon_gws)
        if not gws or len(set(gws)) != len(gws):
            raise PredictionContractError("horizon_gws must be non-empty and unique.")
        object.__setattr__(self, "horizon_gws", gws)
        object.__setattr__(self, "fpl_player_id", _int_value(self.fpl_player_id, "fpl_player_id", 1, optional=True))
        object.__setattr__(self, "explanation", _json_safe_mapping(self.explanation, "explanation"))

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class DecisionOutput:
    context: PredictionContext
    safety: SafetyMetadata
    decision_id: str
    decision_type: str
    objective_name: str
    objective_value: float
    selected_entity_ids: Tuple[int, ...]
    rejected_constraints: Tuple[str, ...] = field(default_factory=tuple)
    explanation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context.output_type != OUTPUT_DECISION:
            raise PredictionContractError("Decision output requires output_type=%s." % OUTPUT_DECISION)
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        object.__setattr__(self, "decision_type", _identifier(self.decision_type, "decision_type"))
        object.__setattr__(self, "objective_name", _identifier(self.objective_name, "objective_name"))
        object.__setattr__(self, "objective_value", _float_value(self.objective_value, "objective_value"))
        selected = tuple(_int_value(item, "selected_entity_ids", 1) for item in self.selected_entity_ids)
        if not selected or len(set(selected)) != len(selected):
            raise PredictionContractError("selected_entity_ids must be non-empty and unique.")
        object.__setattr__(self, "selected_entity_ids", selected)
        object.__setattr__(self, "rejected_constraints", _string_tuple(self.rejected_constraints, "rejected_constraints"))
        object.__setattr__(self, "explanation", _json_safe_mapping(self.explanation, "explanation"))

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


CONTRACT_FIELD_REQUIREMENTS: Mapping[str, Mapping[str, Tuple[str, ...]]] = {
    OUTPUT_MATCH: {
        "mandatory": (
            "context",
            "safety",
            "fixture_id",
            "home_team_id",
            "away_team_id",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "predicted_result_label",
        ),
        "optional": ("fpl_fixture_id", "kickoff_time", "home_team_name", "away_team_name"),
    },
    OUTPUT_SCORELINE: {
        "mandatory": (
            "context",
            "safety",
            "fixture_id",
            "home_team_id",
            "away_team_id",
            "expected_home_goals",
            "expected_away_goals",
        ),
        "optional": (
            "predicted_scoreline",
            "scoreline_home_win_probability",
            "scoreline_draw_probability",
            "scoreline_away_win_probability",
            "fpl_fixture_id",
        ),
    },
    OUTPUT_MINUTES: {
        "mandatory": (
            "context",
            "safety",
            "player_id",
            "appearance_probability",
            "start_probability",
            "conditional_minutes_if_appears",
            "expected_minutes",
            "minutes_lower_bound",
            "minutes_upper_bound",
        ),
        "optional": ("fixture_id", "fpl_player_id"),
    },
    OUTPUT_PLAYER_EVENT: {
        "mandatory": (
            "context",
            "safety",
            "player_id",
            "expected_goals",
            "expected_assists",
            "clean_sheet_probability",
            "expected_bonus",
            "expected_other_points",
        ),
        "optional": ("fixture_id", "fpl_player_id"),
    },
    OUTPUT_PLAYER_POINTS: {
        "mandatory": (
            "context",
            "safety",
            "player_id",
            "team_id",
            "position",
            "price",
            "now_cost",
            "predicted_points",
            "expected_minutes",
            "start_probability",
            "appearance_probability",
            "has_fixture",
            "player_status",
            "status_cutoff_valid",
            "status_hard_guardrail_applied",
            "selection_eligible",
            "eligibility_reason",
        ),
        "optional": (
            "fixture_id",
            "fpl_player_id",
            "opponent_team_id",
            "is_home",
            "component fields",
            "extensions",
        ),
    },
    OUTPUT_RANKING: {
        "mandatory": (
            "context",
            "safety",
            "player_id",
            "position",
            "metric_name",
            "metric_value",
            "rank",
            "horizon_gws",
        ),
        "optional": ("fpl_player_id", "explanation"),
    },
    OUTPUT_DECISION: {
        "mandatory": (
            "context",
            "safety",
            "decision_id",
            "decision_type",
            "objective_name",
            "objective_value",
            "selected_entity_ids",
        ),
        "optional": ("rejected_constraints", "explanation"),
    },
}


def contract_field_requirements(output_type: str) -> Mapping[str, Tuple[str, ...]]:
    if output_type not in CONTRACT_FIELD_REQUIREMENTS:
        raise PredictionContractError("Unsupported output_type=%s." % output_type)
    return CONTRACT_FIELD_REQUIREMENTS[output_type]


def _legacy_context(
    row: Mapping[str, Any],
    output_type: str,
    source_seasons: Any,
    as_of_time: Any,
    run_id: Any,
    feature_version: Optional[str] = None,
    rules_versions: Optional[Mapping[str, str]] = None,
) -> PredictionContext:
    return PredictionContext(
        output_type=output_type,
        source_seasons=_source_seasons(source_seasons),
        target_season=row.get("target_season"),
        target_gw=row.get("target_gw"),
        as_of_time=as_of_time,
        prediction_mode=row.get("prediction_mode"),
        prediction_scope=row.get("prediction_scope"),
        run_id=run_id,
        model_name=row.get("model_name"),
        model_version=row.get("model_version"),
        feature_version=feature_version,
        rules_versions=rules_versions or {},
    )


def _legacy_preview_safety(
    row: Mapping[str, Any],
    fallback_used: bool,
    fallback_level: Optional[int] = None,
    fallback_reason: Optional[str] = None,
    confidence_score: Optional[float] = None,
    confidence_label: Optional[str] = None,
    uncertainty_lower: Optional[float] = None,
    uncertainty_upper: Optional[float] = None,
    risk_flags: Any = None,
) -> SafetyMetadata:
    if _legacy_bool(row.get("production_ready"), False):
        raise PredictionContractError(
            "Legacy preview adapter refuses production_ready=True."
        )
    if _legacy_bool(row.get("prediction_write_allowed"), False):
        raise PredictionContractError(
            "Legacy preview adapter refuses prediction_write_allowed=True."
        )
    return SafetyMetadata(
        data_quality_status=row.get("data_quality_status") or "legacy_unknown",
        calibration_status=row.get("calibration_status") or "legacy_unknown",
        guardrail_status=row.get("guardrail_status") or "legacy_unknown",
        readiness_status=READINESS_PREVIEW_ONLY,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        uncertainty_lower=uncertainty_lower,
        uncertainty_upper=uncertainty_upper,
        fallback_used=fallback_used,
        fallback_level=fallback_level,
        fallback_reason=fallback_reason,
        risk_flags=_string_tuple(risk_flags, "risk_flags"),
        production_ready=False,
        prediction_write_allowed=False,
    )


def adapt_day70a_match_preview(
    row: Mapping[str, Any],
    as_of_time: str,
    run_id: str,
    feature_version: Optional[str] = None,
) -> MatchPredictionOutput:
    """Adapt one Day70A row; missing provenance must be supplied explicitly."""
    fallback_used = _legacy_bool(row.get("any_team_fallback_applied"), False)
    context = _legacy_context(
        row=row,
        output_type=OUTPUT_MATCH,
        source_seasons=row.get("source_season"),
        as_of_time=as_of_time,
        run_id=run_id,
        feature_version=feature_version,
    )
    safety = _legacy_preview_safety(
        row,
        fallback_used=fallback_used,
        fallback_level=1 if fallback_used else None,
        fallback_reason="team_feature_fallback" if fallback_used else None,
        confidence_score=_float_value(row.get("confidence_score"), "confidence_score", 0.0, 1.0, optional=True),
    )
    return MatchPredictionOutput(
        context=context,
        safety=safety,
        fixture_id=row.get("fixture_id"),
        fpl_fixture_id=row.get("fpl_fixture_id"),
        kickoff_time=row.get("kickoff_time"),
        home_team_id=row.get("home_team_id"),
        away_team_id=row.get("away_team_id"),
        home_team_name=row.get("home_team_name") or row.get("home_team_short_name"),
        away_team_name=row.get("away_team_name") or row.get("away_team_short_name"),
        home_win_probability=row.get("home_win_probability"),
        draw_probability=row.get("draw_probability"),
        away_win_probability=row.get("away_win_probability"),
        predicted_result_label=row.get("predicted_result_label"),
    )


def adapt_day70c_scoreline_preview(
    row: Mapping[str, Any],
    as_of_time: str,
    run_id: str,
    feature_version: Optional[str] = None,
) -> ScorelinePredictionOutput:
    """Adapt one Day70C row while preserving preview-only safety."""
    fallback_used = _legacy_bool(row.get("any_team_fallback_applied"), False)
    context = _legacy_context(
        row=row,
        output_type=OUTPUT_SCORELINE,
        source_seasons=row.get("source_season"),
        as_of_time=as_of_time,
        run_id=run_id,
        feature_version=feature_version,
    )
    safety = _legacy_preview_safety(
        row,
        fallback_used=fallback_used,
        fallback_level=1 if fallback_used else None,
        fallback_reason="team_feature_fallback" if fallback_used else None,
    )
    top_scoreline = row.get("top_1_scoreline")
    return ScorelinePredictionOutput(
        context=context,
        safety=safety,
        fixture_id=row.get("fixture_id"),
        fpl_fixture_id=row.get("fpl_fixture_id"),
        home_team_id=row.get("home_team_id"),
        away_team_id=row.get("away_team_id"),
        expected_home_goals=row.get("expected_home_goals"),
        expected_away_goals=row.get("expected_away_goals"),
        predicted_scoreline=top_scoreline,
        scoreline_home_win_probability=row.get("scoreline_home_win_probability"),
        scoreline_draw_probability=row.get("scoreline_draw_probability"),
        scoreline_away_win_probability=row.get("scoreline_away_win_probability"),
    )


def adapt_day72b_player_prediction_manifest(
    manifest: Mapping[str, Any],
) -> PlayerPredictionManifestContract:
    """Adapt the Day72B manifest without granting production approval.

    A passing Day72B manifest may authorize explicit conversion of its Day72A
    rows into the stable Day76B preview schema.  The adapter preserves the
    original ``ready_for_opening_squad_optimizer_input`` flag, which remains
    false for Day72B v1, and never treats schema consumption as recommendation
    approval.
    """
    raw = _mapping_from_object(manifest)
    artifact_type = _required_text(raw.get("artifact_type"), "artifact_type")
    manifest_version = _required_text(
        raw.get("manifest_version"), "manifest_version"
    )
    model_name = _required_text(raw.get("model_name"), "model_name")
    model_version = _required_text(raw.get("model_version"), "model_version")
    prediction_scope = _required_text(
        raw.get("prediction_scope"), "prediction_scope"
    )

    nested_metadata = raw.get("run_metadata")
    if nested_metadata is not None:
        if not isinstance(nested_metadata, Mapping):
            raise PredictionContractError("run_metadata must be a mapping.")
        nested_raw = _mapping_from_object(nested_metadata)
        nested_versions = nested_raw.get("versions") or {}
        if not isinstance(nested_versions, Mapping):
            raise PredictionContractError(
                "run_metadata.versions must be a mapping."
            )
        context = PredictionContext.from_run_metadata(
            nested_metadata,
            output_type=OUTPUT_PLAYER_POINTS,
            model_name=model_name,
            model_version=model_version,
            prediction_scope=prediction_scope,
        )
        identity_pairs = (
            ("run_id", raw.get("run_id"), context.run_id),
            ("target_season", raw.get("target_season"), context.target_season),
            ("target_gw", raw.get("target_gw"), context.target_gw),
            ("prediction_mode", raw.get("resolved_prediction_mode"), context.prediction_mode),
            ("model_version", nested_versions.get("model_version"), model_version),
            ("feature_version", raw.get("player_feature_version"), context.feature_version),
            ("artifact_version", nested_versions.get("artifact_version"), manifest_version),
            ("manifest_version", nested_versions.get("manifest_version"), manifest_version),
        )
        for field_name, top_value, nested_value in identity_pairs:
            if top_value is not None and str(top_value) != str(nested_value):
                raise PredictionContractError(
                    "Day72B top-level %s does not match run_metadata."
                    % field_name
                )
        top_sources = _source_seasons(raw.get("source_seasons"))
        if top_sources != context.source_seasons:
            raise PredictionContractError(
                "Day72B top-level source_seasons do not match run_metadata."
            )
        top_scoring_version = raw.get("scoring_rules_version")
        nested_scoring_version = context.rules_versions.get("scoring")
        if (
            top_scoring_version is not None
            and str(top_scoring_version) != str(nested_scoring_version)
        ):
            raise PredictionContractError(
                "Day72B top-level scoring_rules_version does not match run_metadata."
            )
        top_as_of = raw.get("as_of_time")
        if top_as_of is not None and _utc_iso(top_as_of, "as_of_time") != context.as_of_time:
            raise PredictionContractError(
                "Day72B top-level as_of_time does not match run_metadata."
            )
    else:
        context = PredictionContext(
            output_type=OUTPUT_PLAYER_POINTS,
            source_seasons=_source_seasons(raw.get("source_seasons")),
            target_season=raw.get("target_season"),
            target_gw=raw.get("target_gw"),
            as_of_time=raw.get("as_of_time"),
            prediction_mode=raw.get("resolved_prediction_mode"),
            prediction_scope=prediction_scope,
            run_id=raw.get("run_id"),
            model_name=model_name,
            model_version=model_version,
            feature_version=raw.get("player_feature_version"),
            rules_versions={"scoring": raw.get("scoring_rules_version")},
            source_artifact_version=manifest_version,
        )

    unsafe_flags = (
        "writes_database",
        "ready_for_public_prediction",
        "ready_for_prediction_write",
        "ready_for_production_write",
    )
    enabled_unsafe = [
        name for name in unsafe_flags if _legacy_bool(raw.get(name), False)
    ]
    if enabled_unsafe:
        raise PredictionContractError(
            "Day72B preview manifest contains unsafe enabled flags: %s."
            % ", ".join(enabled_unsafe)
        )

    blockers = _string_tuple(raw.get("blockers") or (), "blockers")
    warnings = _string_tuple(raw.get("warnings") or (), "warnings")
    passed = _legacy_bool(raw.get("passed"), False)
    manifest_ready = _legacy_bool(
        raw.get("ready_for_pre_gw1_player_prediction_manifest"), False
    )
    ops_preview_ready = _legacy_bool(raw.get("ready_for_ops_preview"), False)
    original_optimizer_ready = _legacy_bool(
        raw.get("ready_for_opening_squad_optimizer_input"), False
    )

    safety_contract = raw.get("safety_contract") or {}
    if not isinstance(safety_contract, Mapping):
        raise PredictionContractError("safety_contract must be a mapping.")
    row_gate = safety_contract.get("row_level_write_gate") or {}
    if not isinstance(row_gate, Mapping):
        raise PredictionContractError(
            "safety_contract.row_level_write_gate must be a mapping."
        )
    write_gate_passed = _legacy_bool(row_gate.get("write_gate_passed"), False)
    preview_schema_consumption_allowed = bool(
        passed
        and manifest_ready
        and ops_preview_ready
        and not blockers
        and write_gate_passed
    )

    return PlayerPredictionManifestContract(
        context=context,
        manifest_version=manifest_version,
        artifact_type=artifact_type,
        passed=passed,
        preview_schema_consumption_allowed=preview_schema_consumption_allowed,
        original_optimizer_input_ready=original_optimizer_ready,
        ops_preview_ready=ops_preview_ready,
        blockers=blockers,
        warnings=warnings,
        artifact_fingerprints=raw.get("artifact_fingerprints") or {},
        readiness_status=READINESS_PREVIEW_ONLY,
        production_ready=False,
        prediction_write_allowed=False,
    )


def adapt_day72a_player_points_preview(row: Mapping[str, Any]) -> PlayerPointsPredictionOutput:
    """Adapt Day72A to the stable player-points contract used by Fast Lane."""
    fallback_level = _int_value(row.get("fallback_level"), "fallback_level", 0, optional=True)
    fallback_policy = _optional_text(row.get("fallback_policy_used"))
    normalized_policy = (fallback_policy or "").strip().lower()
    fallback_used = bool(
        (fallback_level or 0) > 0
        or normalized_policy not in ("", "none", "no_fallback", "not_used", "full_prior")
    )
    source_seasons = row.get("source_seasons")
    context = _legacy_context(
        row=row,
        output_type=OUTPUT_PLAYER_POINTS,
        source_seasons=source_seasons,
        as_of_time=row.get("as_of_time"),
        run_id=row.get("run_id"),
        feature_version=row.get("player_feature_version"),
        rules_versions={"scoring": row.get("scoring_rules_version")},
    )
    risk_flags = row.get("risk_flags")
    safety = _legacy_preview_safety(
        row,
        fallback_used=fallback_used,
        fallback_level=fallback_level if fallback_used else None,
        fallback_reason=row.get("fallback_reason") or fallback_policy,
        confidence_label=row.get("prediction_confidence"),
        uncertainty_lower=None,
        uncertainty_upper=None,
        risk_flags=risk_flags,
    )
    known_fields = {
        "source_seasons",
        "target_season",
        "target_gw",
        "as_of_time",
        "prediction_mode",
        "prediction_scope",
        "run_id",
        "model_name",
        "model_version",
        "player_feature_version",
        "scoring_rules_version",
        "player_id",
        "fpl_player_id",
        "player_name",
        "web_name",
        "team_id",
        "team_name",
        "team_short_name",
        "position",
        "price",
        "now_cost",
        "fixture_id",
        "fpl_fixture_id",
        "opponent_team_id",
        "opponent_team_name",
        "opponent_short_name",
        "is_home",
        "has_fixture",
        "status",
        "status_cutoff_valid",
        "status_hard_guardrail_applied",
        "selection_eligible",
        "eligibility_reason",
        "appearance_probability",
        "start_probability",
        "expected_minutes",
        "minutes_lower_bound",
        "minutes_upper_bound",
        "raw_expected_points",
        "final_predicted_points",
        "predicted_points",
        "expected_appearance_points",
        "expected_goals",
        "expected_goal_points",
        "expected_assists",
        "expected_assist_points",
        "clean_sheet_probability",
        "expected_clean_sheet_points",
        "expected_bonus",
        "expected_other_points",
        "component_accounting_status",
        "risk_flags",
        "fallback_level",
        "fallback_policy_used",
        "fallback_reason",
        "data_quality_status",
        "prediction_confidence",
        "prediction_write_allowed",
        "production_ready",
        "calibration_status",
        "guardrail_status",
    }
    status_cutoff_valid = _strict_bool(
        row.get("status_cutoff_valid"), "status_cutoff_valid"
    )
    status_hard_guardrail_applied = _strict_bool(
        row.get("status_hard_guardrail_applied"),
        "status_hard_guardrail_applied",
    )
    derived_eligible = bool(
        status_cutoff_valid and not status_hard_guardrail_applied
    )
    if row.get("selection_eligible") is None:
        selection_eligible = derived_eligible
    else:
        selection_eligible = _strict_bool(
            row.get("selection_eligible"), "selection_eligible"
        )
    eligibility_reason = row.get("eligibility_reason")
    if eligibility_reason is None:
        if selection_eligible:
            eligibility_reason = "eligible_preview_status"
        elif status_hard_guardrail_applied:
            eligibility_reason = "status_hard_guardrail_applied"
        else:
            eligibility_reason = "status_cutoff_invalid"

    extensions = {key: value for key, value in row.items() if key not in known_fields}
    return PlayerPointsPredictionOutput(
        context=context,
        safety=safety,
        player_id=row.get("player_id"),
        fpl_player_id=row.get("fpl_player_id"),
        player_name=row.get("player_name"),
        web_name=row.get("web_name"),
        team_id=row.get("team_id"),
        team_name=row.get("team_name"),
        team_short_name=row.get("team_short_name"),
        position=row.get("position"),
        price=row.get("price"),
        now_cost=row.get("now_cost"),
        fixture_id=row.get("fixture_id"),
        fpl_fixture_id=row.get("fpl_fixture_id"),
        opponent_team_id=row.get("opponent_team_id"),
        opponent_name=row.get("opponent_team_name") or row.get("opponent_short_name"),
        is_home=row.get("is_home"),
        has_fixture=row.get("has_fixture"),
        player_status=row.get("status") or "unknown",
        status_cutoff_valid=status_cutoff_valid,
        status_hard_guardrail_applied=status_hard_guardrail_applied,
        selection_eligible=selection_eligible,
        eligibility_reason=eligibility_reason,
        appearance_probability=row.get("appearance_probability"),
        start_probability=row.get("start_probability"),
        expected_minutes=row.get("expected_minutes"),
        minutes_lower_bound=row.get("minutes_lower_bound"),
        minutes_upper_bound=row.get("minutes_upper_bound"),
        predicted_points=(
            row.get("final_predicted_points")
            if row.get("final_predicted_points") is not None
            else row.get("predicted_points")
        ),
        raw_expected_points=row.get("raw_expected_points"),
        expected_appearance_points=row.get("expected_appearance_points"),
        expected_goals=row.get("expected_goals"),
        expected_goal_points=row.get("expected_goal_points"),
        expected_assists=row.get("expected_assists"),
        expected_assist_points=row.get("expected_assist_points"),
        clean_sheet_probability=row.get("clean_sheet_probability"),
        expected_clean_sheet_points=row.get("expected_clean_sheet_points"),
        expected_bonus=row.get("expected_bonus"),
        expected_other_points=row.get("expected_other_points"),
        component_accounting_status=row.get("component_accounting_status"),
        extensions=extensions,
    )


def adapt_legacy_db_player_prediction(
    row: Mapping[str, Any],
    context: PredictionContext,
    team_id: int,
    position: str,
    price: float,
    now_cost: int,
) -> PlayerPointsPredictionOutput:
    """Explicit minimal adapter for legacy DB Prediction rows.

    Missing minutes, probability, uncertainty, and fallback evidence are made
    visible instead of guessed.  The result is always preview-only.
    """
    safety = SafetyMetadata(
        data_quality_status="legacy_db_minimal_fields",
        calibration_status="legacy_unknown",
        guardrail_status="legacy_unknown",
        readiness_status=READINESS_PREVIEW_ONLY,
        fallback_used=True,
        fallback_level=99,
        fallback_reason="legacy_db_missing_standard_metadata",
        risk_flags=("missing_minutes_contract", "missing_uncertainty_contract"),
        production_ready=False,
        prediction_write_allowed=False,
    )
    return PlayerPointsPredictionOutput(
        context=context,
        safety=safety,
        player_id=row.get("player_id"),
        team_id=team_id,
        position=position,
        price=price,
        now_cost=now_cost,
        predicted_points=row.get("predicted_points"),
        expected_minutes=0.0,
        start_probability=0.0,
        appearance_probability=0.0,
        has_fixture=False,
        player_status="unknown",
        status_cutoff_valid=False,
        status_hard_guardrail_applied=False,
        selection_eligible=False,
        eligibility_reason="legacy_status_unknown",
        extensions={"legacy_created_at": row.get("created_at")},
    )
