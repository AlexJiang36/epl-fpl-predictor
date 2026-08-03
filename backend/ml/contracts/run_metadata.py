from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


CONTRACT_VERSION = "fpl_run_metadata_v1"
VALIDATION_VERSION = "day75b_v1"

RUN_TYPES = (
    "feature",
    "model",
    "prediction",
    "evaluation",
    "optimization",
    "publishing",
)

_RUN_TYPE_SET = set(RUN_TYPES)
_SEASON_RE = re.compile(r"^(?P<start>\d{4})_(?P<end>\d{2})$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PREDICTION_MODE_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

_REQUIRED_VERSION_FIELDS_BY_RUN_TYPE = {
    "feature": ("feature_version", "artifact_version"),
    "model": ("feature_version", "model_version", "artifact_version"),
    "prediction": (
        "feature_version",
        "model_version",
        "manifest_version",
        "artifact_version",
        "rules_versions",
    ),
    "evaluation": (
        "model_version",
        "manifest_version",
        "artifact_version",
    ),
    "optimization": (
        "model_version",
        "manifest_version",
        "artifact_version",
        "rules_versions",
    ),
    "publishing": ("manifest_version", "artifact_version"),
}

_TARGET_GW_REQUIRED_RUN_TYPES = {
    "feature",
    "prediction",
    "evaluation",
    "optimization",
    "publishing",
}


class RunMetadataError(ValueError):
    """Raised when run metadata is invalid or cannot be normalized."""


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunMetadataError("%s must be a non-empty string." % label)
    return value.strip()


def _optional_text(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    return _nonempty_text(value, label)


def _normalize_identifier(value: Any, label: str) -> str:
    text = _nonempty_text(value, label)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise RunMetadataError(
            "%s must contain only letters, digits, '.', '_', ':', or '-'." % label
        )
    return text


def _normalize_optional_identifier(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    return _normalize_identifier(value, label)


def parse_utc_datetime(value: Union[str, datetime], label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise RunMetadataError("%s must be a non-empty ISO-8601 timestamp." % label)
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise RunMetadataError(
                "%s must be a valid ISO-8601 timestamp: %s" % (label, value)
            ) from exc
    else:
        raise RunMetadataError("%s must be a datetime or ISO-8601 string." % label)

    if parsed.tzinfo is None:
        raise RunMetadataError("%s must include a timezone offset." % label)

    return parsed.astimezone(timezone.utc)


def format_utc_datetime(value: Union[str, datetime], label: str) -> str:
    return parse_utc_datetime(value, label).isoformat().replace("+00:00", "Z")


def utc_now() -> str:
    return format_utc_datetime(datetime.now(timezone.utc), "utc_now")


def normalize_season(value: Any, label: str) -> str:
    text = _nonempty_text(value, label)
    match = _SEASON_RE.fullmatch(text)
    if match is None:
        raise RunMetadataError(
            "%s must use the season format YYYY_YY, for example 2025_26." % label
        )

    start = int(match.group("start"))
    end = int(match.group("end"))
    if end != (start + 1) % 100:
        raise RunMetadataError(
            "%s must describe consecutive seasons; got %s." % (label, text)
        )
    return text


def _season_start(season: str) -> int:
    return int(season.split("_", 1)[0])


def _normalize_source_seasons(values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise RunMetadataError("source_seasons must be a sequence, not one string.")

    normalized = tuple(
        normalize_season(value, "source_seasons[%s]" % index)
        for index, value in enumerate(values)
    )
    if not normalized:
        raise RunMetadataError("source_seasons must contain at least one season.")
    if len(set(normalized)) != len(normalized):
        raise RunMetadataError("source_seasons must not contain duplicates.")

    starts = [_season_start(value) for value in normalized]
    if starts != sorted(starts):
        raise RunMetadataError("source_seasons must be ordered oldest to newest.")
    return normalized


def _normalize_target_gw(value: Optional[Any]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RunMetadataError("target_gw must be an integer.")
    try:
        target_gw = int(value)
    except (TypeError, ValueError) as exc:
        raise RunMetadataError("target_gw must be an integer.") from exc
    if target_gw < 1 or target_gw > 38:
        raise RunMetadataError("target_gw must be between 1 and 38.")
    return target_gw


def _normalize_horizon(value: Optional[Any]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RunMetadataError("horizon must be an integer.")
    try:
        horizon = int(value)
    except (TypeError, ValueError) as exc:
        raise RunMetadataError("horizon must be an integer.") from exc
    if horizon < 1:
        raise RunMetadataError("horizon must be at least 1.")
    return horizon


def _normalize_optional_nonnegative_int(
    value: Optional[Any],
    label: str,
) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RunMetadataError("%s must be a nonnegative integer." % label)
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise RunMetadataError("%s must be a nonnegative integer." % label) from exc
    if normalized < 0:
        raise RunMetadataError("%s must be a nonnegative integer." % label)
    return normalized


def _sorted_text_mapping(
    value: Optional[Mapping[str, Any]],
    label: str,
) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RunMetadataError("%s must be an object mapping names to versions." % label)

    normalized: Dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _normalize_identifier(str(raw_key), "%s key" % label)
        normalized[key] = _normalize_identifier(raw_value, "%s.%s" % (label, key))
    return {key: normalized[key] for key in sorted(normalized)}


@dataclass(frozen=True)
class RunVersions:
    feature_version: Optional[str] = None
    model_version: Optional[str] = None
    rules_versions: Mapping[str, str] = field(default_factory=dict)
    manifest_version: Optional[str] = None
    artifact_version: Optional[str] = None
    additional_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_version",
            _normalize_optional_identifier(self.feature_version, "feature_version"),
        )
        object.__setattr__(
            self,
            "model_version",
            _normalize_optional_identifier(self.model_version, "model_version"),
        )
        object.__setattr__(
            self,
            "rules_versions",
            _sorted_text_mapping(self.rules_versions, "rules_versions"),
        )
        object.__setattr__(
            self,
            "manifest_version",
            _normalize_optional_identifier(self.manifest_version, "manifest_version"),
        )
        object.__setattr__(
            self,
            "artifact_version",
            _normalize_optional_identifier(self.artifact_version, "artifact_version"),
        )
        object.__setattr__(
            self,
            "additional_versions",
            _sorted_text_mapping(self.additional_versions, "additional_versions"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "rules_versions": dict(self.rules_versions),
            "manifest_version": self.manifest_version,
            "artifact_version": self.artifact_version,
            "additional_versions": dict(self.additional_versions),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunVersions":
        if not isinstance(value, Mapping):
            raise RunMetadataError("versions must be an object.")
        return cls(
            feature_version=value.get("feature_version"),
            model_version=value.get("model_version"),
            rules_versions=value.get("rules_versions") or {},
            manifest_version=value.get("manifest_version"),
            artifact_version=value.get("artifact_version"),
            additional_versions=value.get("additional_versions") or {},
        )


@dataclass(frozen=True)
class ProvenanceInput:
    name: str
    path: Optional[str] = None
    uri: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    exists: Optional[bool] = None
    artifact_type: Optional[str] = None
    run_id: Optional[str] = None
    version: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_identifier(self.name, "input.name"))
        object.__setattr__(self, "path", _optional_text(self.path, "input.path"))
        object.__setattr__(self, "uri", _optional_text(self.uri, "input.uri"))
        object.__setattr__(
            self,
            "artifact_type",
            _normalize_optional_identifier(self.artifact_type, "input.artifact_type"),
        )
        object.__setattr__(self, "run_id", _optional_text(self.run_id, "input.run_id"))
        object.__setattr__(
            self,
            "version",
            _normalize_optional_identifier(self.version, "input.version"),
        )
        object.__setattr__(
            self,
            "size_bytes",
            _normalize_optional_nonnegative_int(self.size_bytes, "input.size_bytes"),
        )

        if self.sha256 is not None:
            digest = _nonempty_text(self.sha256, "input.sha256").lower()
            if not _SHA256_RE.fullmatch(digest):
                raise RunMetadataError(
                    "input.sha256 must be a 64-character lowercase hex digest."
                )
            object.__setattr__(self, "sha256", digest)

        if self.exists is not None and not isinstance(self.exists, bool):
            raise RunMetadataError("input.exists must be true, false, or null.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "exists": self.exists,
            "artifact_type": self.artifact_type,
            "run_id": self.run_id,
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProvenanceInput":
        if not isinstance(value, Mapping):
            raise RunMetadataError("Each provenance input must be an object.")
        return cls(
            name=value.get("name"),
            path=value.get("path"),
            uri=value.get("uri"),
            sha256=value.get("sha256"),
            size_bytes=value.get("size_bytes"),
            exists=value.get("exists"),
            artifact_type=value.get("artifact_type"),
            run_id=value.get("run_id"),
            version=value.get("version"),
        )

    @classmethod
    def from_file_metadata(
        cls,
        name: str,
        value: Mapping[str, Any],
    ) -> "ProvenanceInput":
        if not isinstance(value, Mapping):
            raise RunMetadataError("File metadata for %s must be an object." % name)
        return cls(
            name=name,
            path=value.get("path"),
            sha256=value.get("sha256"),
            size_bytes=value.get("size_bytes"),
            exists=value.get("exists"),
        )


@dataclass(frozen=True)
class RunProvenance:
    producer: str
    inputs: Tuple[ProvenanceInput, ...] = field(default_factory=tuple)
    parent_run_ids: Tuple[str, ...] = field(default_factory=tuple)
    code_version: Optional[str] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "producer",
            _normalize_identifier(self.producer, "provenance.producer"),
        )

        normalized_inputs: List[ProvenanceInput] = []
        for item in self.inputs:
            if isinstance(item, ProvenanceInput):
                normalized_inputs.append(item)
            elif isinstance(item, Mapping):
                normalized_inputs.append(ProvenanceInput.from_mapping(item))
            else:
                raise RunMetadataError(
                    "provenance.inputs must contain ProvenanceInput objects or mappings."
                )
        names = [item.name for item in normalized_inputs]
        if len(set(names)) != len(names):
            raise RunMetadataError("provenance input names must be unique.")
        object.__setattr__(
            self,
            "inputs",
            tuple(sorted(normalized_inputs, key=lambda item: item.name)),
        )

        parent_run_ids = tuple(
            _nonempty_text(value, "provenance.parent_run_ids")
            for value in self.parent_run_ids
        )
        if len(set(parent_run_ids)) != len(parent_run_ids):
            raise RunMetadataError("provenance.parent_run_ids must be unique.")
        object.__setattr__(self, "parent_run_ids", parent_run_ids)

        object.__setattr__(
            self,
            "code_version",
            _normalize_optional_identifier(self.code_version, "provenance.code_version"),
        )
        object.__setattr__(
            self,
            "notes",
            tuple(_nonempty_text(value, "provenance.notes") for value in self.notes),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "producer": self.producer,
            "inputs": [item.to_dict() for item in self.inputs],
            "parent_run_ids": list(self.parent_run_ids),
            "code_version": self.code_version,
            "notes": list(self.notes),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunProvenance":
        if not isinstance(value, Mapping):
            raise RunMetadataError("provenance must be an object.")
        raw_inputs = value.get("inputs") or []
        if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, (str, bytes)):
            raise RunMetadataError("provenance.inputs must be an array.")
        return cls(
            producer=value.get("producer"),
            inputs=tuple(ProvenanceInput.from_mapping(item) for item in raw_inputs),
            parent_run_ids=tuple(value.get("parent_run_ids") or []),
            code_version=value.get("code_version"),
            notes=tuple(value.get("notes") or []),
        )


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    run_type: str
    artifact_type: str
    source_seasons: Tuple[str, ...]
    target_season: str
    target_gw: Optional[int]
    horizon: Optional[int]
    as_of_time_utc: str
    prediction_mode: str
    created_at_utc: str
    versions: RunVersions
    provenance: RunProvenance
    contract_version: str = CONTRACT_VERSION
    started_at_utc: Optional[str] = None
    completed_at_utc: Optional[str] = None

    def __post_init__(self) -> None:
        run_id = _nonempty_text(self.run_id, "run_id")
        if not _RUN_ID_RE.fullmatch(run_id):
            raise RunMetadataError(
                "run_id must be 3-256 characters and contain only letters, "
                "digits, '.', '_', ':', or '-'."
            )
        object.__setattr__(self, "run_id", run_id)

        run_type = _nonempty_text(self.run_type, "run_type")
        if run_type not in _RUN_TYPE_SET:
            raise RunMetadataError("run_type must be one of: %s." % ", ".join(RUN_TYPES))
        object.__setattr__(self, "run_type", run_type)
        object.__setattr__(
            self,
            "artifact_type",
            _normalize_identifier(self.artifact_type, "artifact_type"),
        )
        object.__setattr__(
            self,
            "source_seasons",
            _normalize_source_seasons(self.source_seasons),
        )
        object.__setattr__(
            self,
            "target_season",
            normalize_season(self.target_season, "target_season"),
        )
        object.__setattr__(self, "target_gw", _normalize_target_gw(self.target_gw))
        object.__setattr__(self, "horizon", _normalize_horizon(self.horizon))

        mode = _nonempty_text(self.prediction_mode, "prediction_mode")
        if not _PREDICTION_MODE_RE.fullmatch(mode):
            raise RunMetadataError("prediction_mode must use lowercase snake_case.")
        object.__setattr__(self, "prediction_mode", mode)

        object.__setattr__(
            self,
            "as_of_time_utc",
            format_utc_datetime(self.as_of_time_utc, "as_of_time_utc"),
        )
        object.__setattr__(
            self,
            "created_at_utc",
            format_utc_datetime(self.created_at_utc, "created_at_utc"),
        )
        if self.started_at_utc is not None:
            object.__setattr__(
                self,
                "started_at_utc",
                format_utc_datetime(self.started_at_utc, "started_at_utc"),
            )
        if self.completed_at_utc is not None:
            object.__setattr__(
                self,
                "completed_at_utc",
                format_utc_datetime(self.completed_at_utc, "completed_at_utc"),
            )

        versions = self.versions
        if isinstance(versions, Mapping):
            versions = RunVersions.from_mapping(versions)
        if not isinstance(versions, RunVersions):
            raise RunMetadataError("versions must be RunVersions or an object.")
        object.__setattr__(self, "versions", versions)

        provenance = self.provenance
        if isinstance(provenance, Mapping):
            provenance = RunProvenance.from_mapping(provenance)
        if not isinstance(provenance, RunProvenance):
            raise RunMetadataError("provenance must be RunProvenance or an object.")
        object.__setattr__(self, "provenance", provenance)

        object.__setattr__(
            self,
            "contract_version",
            _normalize_identifier(self.contract_version, "contract_version"),
        )

        errors = _validate_normalized_metadata(self)
        if errors:
            raise RunMetadataError("; ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "run_id": self.run_id,
            "run_type": self.run_type,
            "artifact_type": self.artifact_type,
            "source_seasons": list(self.source_seasons),
            "target_season": self.target_season,
            "target_gw": self.target_gw,
            "horizon": self.horizon,
            "as_of_time_utc": self.as_of_time_utc,
            "prediction_mode": self.prediction_mode,
            "timestamps": {
                "created_at_utc": self.created_at_utc,
                "started_at_utc": self.started_at_utc,
                "completed_at_utc": self.completed_at_utc,
            },
            "versions": self.versions.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunMetadata":
        if not isinstance(value, Mapping):
            raise RunMetadataError("run metadata must be an object.")
        timestamps = value.get("timestamps") or {}
        if not isinstance(timestamps, Mapping):
            raise RunMetadataError("timestamps must be an object.")
        return cls(
            contract_version=value.get("contract_version", CONTRACT_VERSION),
            run_id=value.get("run_id"),
            run_type=value.get("run_type"),
            artifact_type=value.get("artifact_type"),
            source_seasons=tuple(value.get("source_seasons") or []),
            target_season=value.get("target_season"),
            target_gw=value.get("target_gw"),
            horizon=value.get("horizon"),
            as_of_time_utc=value.get("as_of_time_utc"),
            prediction_mode=value.get("prediction_mode"),
            created_at_utc=timestamps.get("created_at_utc"),
            started_at_utc=timestamps.get("started_at_utc"),
            completed_at_utc=timestamps.get("completed_at_utc"),
            versions=RunVersions.from_mapping(value.get("versions") or {}),
            provenance=RunProvenance.from_mapping(value.get("provenance") or {}),
        )

    @classmethod
    def from_json(cls, value: str) -> "RunMetadata":
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RunMetadataError("run metadata JSON is invalid.") from exc
        return cls.from_mapping(parsed)


def _validate_normalized_metadata(metadata: RunMetadata) -> List[str]:
    errors: List[str] = []

    if any(
        _season_start(value) > _season_start(metadata.target_season)
        for value in metadata.source_seasons
    ):
        errors.append("source_seasons must not be later than target_season.")

    if metadata.run_type in _TARGET_GW_REQUIRED_RUN_TYPES and metadata.target_gw is None:
        errors.append("target_gw is required for run_type=%s." % metadata.run_type)

    if metadata.target_gw is not None:
        if metadata.horizon is None:
            errors.append("horizon is required when target_gw is set.")
        elif metadata.target_gw + metadata.horizon - 1 > 38:
            errors.append("target_gw + horizon - 1 must not exceed Gameweek 38.")
    elif metadata.horizon is not None:
        errors.append("horizon must be null when target_gw is null.")

    as_of = parse_utc_datetime(metadata.as_of_time_utc, "as_of_time_utc")
    created = parse_utc_datetime(metadata.created_at_utc, "created_at_utc")
    if as_of > created:
        errors.append("as_of_time_utc must not be later than created_at_utc.")

    started = (
        parse_utc_datetime(metadata.started_at_utc, "started_at_utc")
        if metadata.started_at_utc is not None
        else None
    )
    completed = (
        parse_utc_datetime(metadata.completed_at_utc, "completed_at_utc")
        if metadata.completed_at_utc is not None
        else None
    )
    if started is not None and completed is not None and started > completed:
        errors.append("started_at_utc must not be later than completed_at_utc.")
    if started is not None and started > created:
        errors.append("started_at_utc must not be later than created_at_utc.")
    if completed is not None and completed > created:
        errors.append("completed_at_utc must not be later than created_at_utc.")

    version_mapping = metadata.versions.to_dict()
    for field_name in _REQUIRED_VERSION_FIELDS_BY_RUN_TYPE[metadata.run_type]:
        value = version_mapping[field_name]
        if field_name == "rules_versions":
            if not value:
                errors.append(
                    "rules_versions must be non-empty for run_type=%s."
                    % metadata.run_type
                )
        elif value is None:
            errors.append(
                "%s is required for run_type=%s."
                % (field_name, metadata.run_type)
            )

    if metadata.run_id in metadata.provenance.parent_run_ids:
        errors.append("run_id must not list itself as a parent run.")

    return errors


def validate_run_metadata(
    value: Union[RunMetadata, Mapping[str, Any], str],
) -> List[str]:
    try:
        if isinstance(value, RunMetadata):
            metadata = value
        elif isinstance(value, str):
            metadata = RunMetadata.from_json(value)
        elif isinstance(value, Mapping):
            metadata = RunMetadata.from_mapping(value)
        else:
            return ["run metadata must be RunMetadata, a mapping, or JSON text."]
        return _validate_normalized_metadata(metadata)
    except RunMetadataError as exc:
        return [str(exc)]


def assert_valid_run_metadata(
    value: Union[RunMetadata, Mapping[str, Any], str],
) -> RunMetadata:
    if isinstance(value, RunMetadata):
        metadata = value
    elif isinstance(value, str):
        metadata = RunMetadata.from_json(value)
    elif isinstance(value, Mapping):
        metadata = RunMetadata.from_mapping(value)
    else:
        raise RunMetadataError(
            "run metadata must be RunMetadata, a mapping, or JSON text."
        )

    errors = _validate_normalized_metadata(metadata)
    if errors:
        raise RunMetadataError("; ".join(errors))
    return metadata


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not slug:
        raise RunMetadataError("run ID segment cannot be empty.")
    return slug


def build_run_id(
    run_type: str,
    target_season: str,
    target_gw: Optional[int],
    created_at: Union[str, datetime],
    artifact_type: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    normalized_type = _nonempty_text(run_type, "run_type")
    if normalized_type not in _RUN_TYPE_SET:
        raise RunMetadataError("run_type must be one of: %s." % ", ".join(RUN_TYPES))
    season = normalize_season(target_season, "target_season")
    gw = _normalize_target_gw(target_gw)
    timestamp = parse_utc_datetime(created_at, "created_at").strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    artifact_segment = _slug(artifact_type or normalized_type)
    gw_segment = "season" if gw is None else "gw%s" % gw

    if suffix is None:
        payload = "|".join(
            [normalized_type, artifact_segment, season, gw_segment, timestamp]
        )
        suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]

    return "%s_%s_%s_%s_%s" % (
        artifact_segment,
        season,
        gw_segment,
        timestamp,
        _slug(suffix),
    )


def build_run_metadata(
    *,
    run_id: Optional[str],
    run_type: str,
    artifact_type: str,
    source_seasons: Sequence[str],
    target_season: str,
    target_gw: Optional[int],
    horizon: Optional[int],
    as_of_time: Union[str, datetime],
    prediction_mode: str,
    created_at: Union[str, datetime],
    feature_version: Optional[str] = None,
    model_version: Optional[str] = None,
    rules_versions: Optional[Mapping[str, str]] = None,
    manifest_version: Optional[str] = None,
    artifact_version: Optional[str] = None,
    additional_versions: Optional[Mapping[str, str]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    started_at: Optional[Union[str, datetime]] = None,
    completed_at: Optional[Union[str, datetime]] = None,
    contract_version: str = CONTRACT_VERSION,
) -> RunMetadata:
    normalized_created_at = format_utc_datetime(created_at, "created_at")
    resolved_run_id = run_id or build_run_id(
        run_type=run_type,
        target_season=target_season,
        target_gw=target_gw,
        created_at=normalized_created_at,
        artifact_type=artifact_type,
    )

    provenance_value = provenance or {
        "producer": artifact_type,
        "inputs": [],
        "parent_run_ids": [],
        "notes": [],
    }

    return RunMetadata(
        contract_version=contract_version,
        run_id=resolved_run_id,
        run_type=run_type,
        artifact_type=artifact_type,
        source_seasons=tuple(source_seasons),
        target_season=target_season,
        target_gw=target_gw,
        horizon=horizon,
        as_of_time_utc=format_utc_datetime(as_of_time, "as_of_time"),
        prediction_mode=prediction_mode,
        created_at_utc=normalized_created_at,
        started_at_utc=(
            format_utc_datetime(started_at, "started_at")
            if started_at is not None
            else None
        ),
        completed_at_utc=(
            format_utc_datetime(completed_at, "completed_at")
            if completed_at is not None
            else None
        ),
        versions=RunVersions(
            feature_version=feature_version,
            model_version=model_version,
            rules_versions=rules_versions or {},
            manifest_version=manifest_version,
            artifact_version=artifact_version,
            additional_versions=additional_versions or {},
        ),
        provenance=RunProvenance.from_mapping(provenance_value),
    )


def provenance_inputs_from_file_metadata(
    values: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        ProvenanceInput.from_file_metadata(name, metadata).to_dict()
        for name, metadata in sorted(values.items())
    ]


def _example_base(run_type: str, artifact_type: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "run_id": None,
        "run_type": run_type,
        "artifact_type": artifact_type,
        "source_seasons": ["2024_25"],
        "target_season": "2025_26",
        "target_gw": 1,
        "horizon": 1,
        "as_of_time": "2025-08-10T12:00:00Z",
        "prediction_mode": "pre_gw1_prior",
        "created_at": "2025-08-10T12:05:00Z",
        "artifact_version": "example_v1",
        "provenance": {
            "producer": "ml.contracts.run_metadata",
            "inputs": [],
            "parent_run_ids": [],
            "notes": ["Day75B deterministic example."],
        },
    }

    if run_type == "feature":
        kwargs["feature_version"] = "feature_v1"
    elif run_type == "model":
        kwargs["target_gw"] = None
        kwargs["horizon"] = None
        kwargs["prediction_mode"] = "historical_training"
        kwargs["feature_version"] = "feature_v1"
        kwargs["model_version"] = "model_v1"
    elif run_type == "prediction":
        kwargs["feature_version"] = "feature_v1"
        kwargs["model_version"] = "model_v1"
        kwargs["rules_versions"] = {"scoring": "scoring_v1"}
        kwargs["manifest_version"] = "manifest_v1"
    elif run_type == "evaluation":
        kwargs["model_version"] = "model_v1"
        kwargs["manifest_version"] = "manifest_v1"
    elif run_type == "optimization":
        kwargs["model_version"] = "model_v1"
        kwargs["rules_versions"] = {"squad_transfer": "rules_v1"}
        kwargs["manifest_version"] = "manifest_v1"
    elif run_type == "publishing":
        kwargs["manifest_version"] = "manifest_v1"

    return kwargs


def run_deterministic_examples() -> Dict[str, Any]:
    examples: List[Dict[str, Any]] = []

    for run_type in RUN_TYPES:
        metadata = build_run_metadata(
            **_example_base(run_type, "example_%s_artifact" % run_type)
        )
        round_trip = RunMetadata.from_json(metadata.to_json())
        errors = validate_run_metadata(round_trip)
        examples.append(
            {
                "name": "valid_%s_run" % run_type,
                "expected_valid": True,
                "actual_valid": not errors,
                "errors": errors,
            }
        )

    invalid_cases = []

    chronology = _example_base("prediction", "invalid_chronology")
    chronology["as_of_time"] = "2025-08-10T12:06:00Z"
    invalid_cases.append(
        (
            "invalid_future_as_of",
            chronology,
            "as_of_time_utc must not be later than created_at_utc",
        )
    )

    overflow = _example_base("optimization", "invalid_horizon")
    overflow["target_gw"] = 38
    overflow["horizon"] = 2
    invalid_cases.append(
        (
            "invalid_horizon_overflow",
            overflow,
            "must not exceed Gameweek 38",
        )
    )

    missing_rules = _example_base("prediction", "invalid_missing_rules")
    missing_rules["rules_versions"] = {}
    invalid_cases.append(
        (
            "invalid_missing_prediction_rules",
            missing_rules,
            "rules_versions must be non-empty",
        )
    )

    for name, kwargs, expected_fragment in invalid_cases:
        try:
            build_run_metadata(**kwargs)
            errors = []
            actual_valid = True
        except RunMetadataError as exc:
            errors = [str(exc)]
            actual_valid = False
        examples.append(
            {
                "name": name,
                "expected_valid": False,
                "actual_valid": actual_valid,
                "expected_error_fragment": expected_fragment,
                "errors": errors,
                "error_fragment_present": any(
                    expected_fragment in error for error in errors
                ),
            }
        )

    passed_count = 0
    for example in examples:
        if example["expected_valid"]:
            passed = example["actual_valid"]
        else:
            passed = (
                not example["actual_valid"]
                and example.get("error_fragment_present", False)
            )
        example["passed"] = passed
        passed_count += int(passed)

    return {
        "examples": examples,
        "passed_count": passed_count,
        "total_count": len(examples),
        "passed": passed_count == len(examples),
    }


def build_validation_report() -> Dict[str, Any]:
    example_result = run_deterministic_examples()
    passed = bool(example_result["passed"])
    return {
        "created_at": utc_now(),
        "validation_version": VALIDATION_VERSION,
        "contract_version": CONTRACT_VERSION,
        "artifact_type": "standard_run_metadata_contract_validation",
        "passed": passed,
        "audit_only": True,
        "writes_database": False,
        "supported_run_types": list(RUN_TYPES),
        "ready_for_feature_run_metadata": passed,
        "ready_for_model_run_metadata": passed,
        "ready_for_prediction_run_metadata": passed,
        "ready_for_evaluation_run_metadata": passed,
        "ready_for_optimization_run_metadata": passed,
        "ready_for_publishing_run_metadata": passed,
        "ready_for_chronology_validation": passed,
        "ready_for_consistent_serialization": passed,
        "deterministic_examples": example_result,
        "blockers": [] if passed else [
            "One or more deterministic run-metadata examples failed."
        ],
        "warnings": [
            "Day75B proves compatibility with one existing artifact only; broad artifact migration is intentionally deferred."
        ],
    }


def write_json(report: Mapping[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_markdown(report: Mapping[str, Any], out_md: str) -> None:
    examples = report["deterministic_examples"]
    lines = [
        "# Day75B Standard Run Metadata Contract Validation",
        "",
        "- Created at: `%s`" % report["created_at"],
        "- Validation version: `%s`" % report["validation_version"],
        "- Contract version: `%s`" % report["contract_version"],
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `%s`" % report["audit_only"],
        "- Writes database: `%s`" % report["writes_database"],
        "",
        "## Deterministic examples",
        "",
        "- Passed: `%s/%s`"
        % (examples["passed_count"], examples["total_count"]),
        "",
    ]
    for example in examples["examples"]:
        lines.append("- `%s`: `%s`" % (example["name"], example["passed"]))
    lines.extend(
        [
            "",
            "## Blockers",
            "",
            "- %s" % (report["blockers"] or "None"),
            "",
            "## Warnings",
            "",
        ]
    )
    for warning in report["warnings"]:
        lines.append("- %s" % warning)

    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Day75B standard run-metadata contract. "
            "This command is read-only and does not write to the database."
        )
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser.parse_args()


def print_report(report: Mapping[str, Any]) -> None:
    examples = report["deterministic_examples"]
    print("=== Day75B Standard Run Metadata Contract ===")
    print("validation_version:", report["validation_version"])
    print("contract_version:", report["contract_version"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    for key in (
        "ready_for_feature_run_metadata",
        "ready_for_model_run_metadata",
        "ready_for_prediction_run_metadata",
        "ready_for_evaluation_run_metadata",
        "ready_for_optimization_run_metadata",
        "ready_for_publishing_run_metadata",
        "ready_for_chronology_validation",
        "ready_for_consistent_serialization",
    ):
        print("%s:" % key, report[key])
    print()
    print(
        "Deterministic examples: %s/%s passed"
        % (examples["passed_count"], examples["total_count"])
    )
    print()
    print("Blockers:", report["blockers"] or "none")
    print("Warnings:", report["warnings"] or "none")


def main() -> None:
    args = parse_args()
    report = build_validation_report()
    if args.out_json:
        write_json(report, args.out_json)
        print("saved_json:", str(Path(args.out_json)))
    if args.out_md:
        write_markdown(report, args.out_md)
        print("saved_md:", str(Path(args.out_md)))
    print_report(report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
