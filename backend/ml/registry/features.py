"""Feature registry contracts for model/feature compatibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FEATURE_REGISTRY_VERSION = "fpl_feature_registry_v1"
FEATURE_STATUSES = ("draft", "validated", "active", "deprecated", "archived")


class FeatureRegistryError(ValueError):
    pass


def _text(value: Any, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise FeatureRegistryError("%s must be non-empty." % field_name)
    return str(value).strip()


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tuple_text(value: Any, field_name: str, allow_empty: bool = True) -> Tuple[str, ...]:
    if value is None:
        values: Sequence[Any] = []
    elif isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise FeatureRegistryError("%s must be a sequence." % field_name)
    result = tuple(_text(item, field_name) for item in values)
    if not allow_empty and not result:
        raise FeatureRegistryError("%s must be non-empty." % field_name)
    if len(set(result)) != len(result):
        raise FeatureRegistryError("%s must not contain duplicates." % field_name)
    return result


def _mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise FeatureRegistryError("%s must be a mapping." % field_name)
    return dict(value)


@dataclass(frozen=True)
class FeatureRegistryEntry:
    feature_name: str
    feature_version: str
    task_types: Tuple[str, ...]
    prediction_modes: Tuple[str, ...]
    required_columns: Tuple[str, ...]
    optional_columns: Tuple[str, ...] = field(default_factory=tuple)
    source_artifact_types: Tuple[str, ...] = field(default_factory=tuple)
    compatibility_tags: Tuple[str, ...] = field(default_factory=tuple)
    status: str = "draft"
    schema_version: str = FEATURE_REGISTRY_VERSION
    production_ready: bool = False
    artifact_key: Optional[str] = None
    artifact_fingerprint: Optional[str] = None
    notes: Optional[str] = None
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_name", _text(self.feature_name, "feature_name"))
        object.__setattr__(self, "feature_version", _text(self.feature_version, "feature_version"))
        object.__setattr__(self, "task_types", _tuple_text(self.task_types, "task_types", allow_empty=False))
        object.__setattr__(self, "prediction_modes", _tuple_text(self.prediction_modes, "prediction_modes", allow_empty=False))
        object.__setattr__(self, "required_columns", _tuple_text(self.required_columns, "required_columns", allow_empty=False))
        object.__setattr__(self, "optional_columns", _tuple_text(self.optional_columns, "optional_columns"))
        overlap = set(self.required_columns).intersection(self.optional_columns)
        if overlap:
            raise FeatureRegistryError("Columns cannot be both required and optional: %s" % sorted(overlap))
        object.__setattr__(self, "source_artifact_types", _tuple_text(self.source_artifact_types, "source_artifact_types"))
        object.__setattr__(self, "compatibility_tags", _tuple_text(self.compatibility_tags, "compatibility_tags"))
        if self.status not in FEATURE_STATUSES:
            raise FeatureRegistryError("Unsupported status=%s." % self.status)
        object.__setattr__(self, "schema_version", _text(self.schema_version, "schema_version"))
        if not isinstance(self.production_ready, bool):
            raise FeatureRegistryError("production_ready must be bool.")
        if self.production_ready and self.status not in ("validated", "active"):
            raise FeatureRegistryError("production_ready features must be validated or active.")
        object.__setattr__(self, "artifact_key", _optional_text(self.artifact_key))
        object.__setattr__(self, "artifact_fingerprint", _optional_text(self.artifact_fingerprint))
        if self.artifact_fingerprint is not None and self.artifact_key is None:
            raise FeatureRegistryError("artifact_fingerprint requires artifact_key.")
        object.__setattr__(self, "notes", _optional_text(self.notes))
        object.__setattr__(self, "extensions", _mapping(self.extensions, "extensions"))

    @property
    def registry_key(self) -> str:
        return "%s:%s" % (self.feature_name, self.feature_version)

    def supports(self, task_type: str, prediction_mode: str) -> bool:
        return task_type in self.task_types and prediction_mode in self.prediction_modes

    def validate_columns(self, columns: Iterable[str]) -> Tuple[str, ...]:
        available = set(str(column) for column in columns)
        return tuple(column for column in self.required_columns if column not in available)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureRegistryEntry":
        return cls(**dict(value))


class FeatureRegistry:
    def __init__(self, entries: Iterable[FeatureRegistryEntry] = ()) -> None:
        self._entries: Dict[str, FeatureRegistryEntry] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: FeatureRegistryEntry, replace: bool = False) -> None:
        if not isinstance(entry, FeatureRegistryEntry):
            raise FeatureRegistryError("entry must be FeatureRegistryEntry.")
        key = entry.registry_key
        if key in self._entries and not replace:
            raise FeatureRegistryError("Feature registry key already exists: %s" % key)
        self._entries[key] = entry

    def get(self, feature_name: str, feature_version: str) -> FeatureRegistryEntry:
        key = "%s:%s" % (feature_name, feature_version)
        if key not in self._entries:
            raise FeatureRegistryError("Unknown feature registry key: %s" % key)
        return self._entries[key]

    def list_entries(self, status: Optional[str] = None) -> Tuple[FeatureRegistryEntry, ...]:
        entries = sorted(self._entries.values(), key=lambda item: item.registry_key)
        if status is not None:
            entries = [entry for entry in entries if entry.status == status]
        return tuple(entries)

    def compatible_entries(self, task_type: str, prediction_mode: str) -> Tuple[FeatureRegistryEntry, ...]:
        return tuple(
            entry
            for entry in self.list_entries()
            if entry.supports(task_type, prediction_mode)
            and entry.status not in ("deprecated", "archived")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_version": FEATURE_REGISTRY_VERSION,
            "entries": [entry.to_dict() for entry in self.list_entries()],
        }


def adapt_legacy_feature_snapshot_metadata(
    value: Mapping[str, Any],
    task_types: Sequence[str],
    prediction_modes: Sequence[str],
    required_columns: Sequence[str],
) -> FeatureRegistryEntry:
    """Convert old feature-snapshot metadata without claiming production readiness."""
    feature_name = value.get("snapshot_type") or value.get("artifact_type") or "legacy_feature_snapshot"
    feature_version = value.get("feature_version") or "legacy_unknown"
    extensions = {
        "legacy_snapshot_id": value.get("snapshot_id"),
        "legacy_created_at": value.get("created_at"),
        "legacy_source_path": value.get("source_path"),
    }
    return FeatureRegistryEntry(
        feature_name=str(feature_name),
        feature_version=str(feature_version),
        task_types=tuple(task_types),
        prediction_modes=tuple(prediction_modes),
        required_columns=tuple(required_columns),
        optional_columns=tuple(),
        source_artifact_types=("legacy_feature_snapshot",),
        compatibility_tags=("legacy_adapter", "unverified_schema"),
        status="draft",
        production_ready=False,
        notes="Explicit adapter for legacy feature snapshot metadata.",
        extensions=extensions,
    )
