"""Model registry contracts with explicit feature/output compatibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


MODEL_REGISTRY_VERSION = "fpl_model_registry_v1"
MODEL_STATUSES = ("draft", "experimental", "validated", "active", "deprecated", "archived")


class ModelRegistryError(ValueError):
    pass


def _text(value: Any, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise ModelRegistryError("%s must be non-empty." % field_name)
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
        raise ModelRegistryError("%s must be a sequence." % field_name)
    result = tuple(_text(item, field_name) for item in values)
    if not allow_empty and not result:
        raise ModelRegistryError("%s must be non-empty." % field_name)
    if len(set(result)) != len(result):
        raise ModelRegistryError("%s must not contain duplicates." % field_name)
    return result


def _text_mapping(value: Any, field_name: str) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ModelRegistryError("%s must be a mapping." % field_name)
    result: Dict[str, str] = {}
    for key, item in value.items():
        result[_text(key, "%s key" % field_name)] = _text(item, "%s value" % field_name)
    return dict(sorted(result.items()))


def _float_mapping(value: Any, field_name: str) -> Dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ModelRegistryError("%s must be a mapping." % field_name)
    result: Dict[str, float] = {}
    for key, item in value.items():
        try:
            result[_text(key, "%s key" % field_name)] = float(item)
        except (TypeError, ValueError):
            raise ModelRegistryError("%s values must be numeric." % field_name)
    return dict(sorted(result.items()))


def _mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ModelRegistryError("%s must be a mapping." % field_name)
    return dict(value)


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_name: str
    model_version: str
    task_type: str
    output_contract_type: str
    output_contract_version: str
    supported_prediction_modes: Tuple[str, ...]
    feature_versions: Mapping[str, str]
    status: str = "experimental"
    calibration_status: str = "not_calibrated"
    guardrail_status: str = "not_validated"
    fallback_policy: Optional[str] = None
    production_ready: bool = False
    is_active: bool = False
    is_production_default: bool = False
    artifact_key: Optional[str] = None
    artifact_fingerprint: Optional[str] = None
    metrics_summary: Mapping[str, float] = field(default_factory=dict)
    selected_reason: Optional[str] = None
    notes: Optional[str] = None
    compatibility_tags: Tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = MODEL_REGISTRY_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_name", _text(self.model_name, "model_name"))
        object.__setattr__(self, "model_version", _text(self.model_version, "model_version"))
        object.__setattr__(self, "task_type", _text(self.task_type, "task_type"))
        object.__setattr__(self, "output_contract_type", _text(self.output_contract_type, "output_contract_type"))
        object.__setattr__(self, "output_contract_version", _text(self.output_contract_version, "output_contract_version"))
        object.__setattr__(self, "supported_prediction_modes", _tuple_text(self.supported_prediction_modes, "supported_prediction_modes", allow_empty=False))
        object.__setattr__(self, "feature_versions", _text_mapping(self.feature_versions, "feature_versions"))
        if not self.feature_versions:
            raise ModelRegistryError("feature_versions must be non-empty.")
        if self.status not in MODEL_STATUSES:
            raise ModelRegistryError("Unsupported status=%s." % self.status)
        object.__setattr__(self, "calibration_status", _text(self.calibration_status, "calibration_status"))
        object.__setattr__(self, "guardrail_status", _text(self.guardrail_status, "guardrail_status"))
        object.__setattr__(self, "fallback_policy", _optional_text(self.fallback_policy))
        for field_name in ("production_ready", "is_active", "is_production_default"):
            if not isinstance(getattr(self, field_name), bool):
                raise ModelRegistryError("%s must be bool." % field_name)
        if self.is_production_default and not self.is_active:
            raise ModelRegistryError("is_production_default=True requires is_active=True.")
        if self.is_production_default and not self.production_ready:
            raise ModelRegistryError("Production default must be production_ready.")
        if self.production_ready and self.status not in ("validated", "active"):
            raise ModelRegistryError("production_ready models must be validated or active.")
        object.__setattr__(self, "artifact_key", _optional_text(self.artifact_key))
        object.__setattr__(self, "artifact_fingerprint", _optional_text(self.artifact_fingerprint))
        if self.artifact_fingerprint is not None and self.artifact_key is None:
            raise ModelRegistryError("artifact_fingerprint requires artifact_key.")
        object.__setattr__(self, "metrics_summary", _float_mapping(self.metrics_summary, "metrics_summary"))
        object.__setattr__(self, "selected_reason", _optional_text(self.selected_reason))
        object.__setattr__(self, "notes", _optional_text(self.notes))
        object.__setattr__(self, "compatibility_tags", _tuple_text(self.compatibility_tags, "compatibility_tags"))
        object.__setattr__(self, "schema_version", _text(self.schema_version, "schema_version"))
        object.__setattr__(self, "extensions", _mapping(self.extensions, "extensions"))

    @property
    def registry_key(self) -> str:
        return "%s:%s" % (self.model_name, self.model_version)

    def supports_prediction_mode(self, prediction_mode: str) -> bool:
        return prediction_mode in self.supported_prediction_modes

    def accepts_feature(self, feature_name: str, feature_version: str) -> bool:
        return self.feature_versions.get(feature_name) == feature_version

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelRegistryEntry":
        return cls(**dict(value))


class ModelRegistry:
    def __init__(self, entries: Iterable[ModelRegistryEntry] = ()) -> None:
        self._entries: Dict[str, ModelRegistryEntry] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: ModelRegistryEntry, replace: bool = False) -> None:
        if not isinstance(entry, ModelRegistryEntry):
            raise ModelRegistryError("entry must be ModelRegistryEntry.")
        key = entry.registry_key
        if key in self._entries and not replace:
            raise ModelRegistryError("Model registry key already exists: %s" % key)
        self._entries[key] = entry

    def get(self, model_name: str, model_version: str) -> ModelRegistryEntry:
        key = "%s:%s" % (model_name, model_version)
        if key not in self._entries:
            raise ModelRegistryError("Unknown model registry key: %s" % key)
        return self._entries[key]

    def list_entries(self, status: Optional[str] = None) -> Tuple[ModelRegistryEntry, ...]:
        entries = sorted(self._entries.values(), key=lambda item: item.registry_key)
        if status is not None:
            entries = [entry for entry in entries if entry.status == status]
        return tuple(entries)

    def compatible_entries(
        self,
        task_type: str,
        prediction_mode: str,
        feature_versions: Mapping[str, str],
        require_production_ready: bool = False,
    ) -> Tuple[ModelRegistryEntry, ...]:
        matches = []
        for entry in self.list_entries():
            if entry.task_type != task_type:
                continue
            if not entry.supports_prediction_mode(prediction_mode):
                continue
            if require_production_ready and not entry.production_ready:
                continue
            if any(feature_versions.get(name) != version for name, version in entry.feature_versions.items()):
                continue
            if entry.status in ("deprecated", "archived"):
                continue
            matches.append(entry)
        return tuple(matches)

    def production_default(self, task_type: str, prediction_mode: str) -> Optional[ModelRegistryEntry]:
        matches = [
            entry
            for entry in self.list_entries()
            if entry.task_type == task_type
            and entry.supports_prediction_mode(prediction_mode)
            and entry.is_production_default
        ]
        if len(matches) > 1:
            raise ModelRegistryError("Multiple production defaults for task/mode.")
        return matches[0] if matches else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_version": MODEL_REGISTRY_VERSION,
            "entries": [entry.to_dict() for entry in self.list_entries()],
        }


def adapt_legacy_model_metadata(
    value: Mapping[str, Any],
    model_version: str,
    output_contract_type: str,
    output_contract_version: str,
    supported_prediction_modes: Sequence[str],
) -> ModelRegistryEntry:
    """Adapt old ModelMetadataArtifact without trusting old active/default flags.

    The old fields are preserved in extensions, but the adapted entry remains
    non-production until a Day76B-compatible manifest proves compatibility.
    """
    feature_version = value.get("feature_version") or "legacy_unknown"
    old_default = bool(value.get("is_production_default", False))
    old_active = bool(value.get("is_active", False))
    return ModelRegistryEntry(
        model_name=value.get("model_name"),
        model_version=model_version,
        task_type=value.get("task_type"),
        output_contract_type=output_contract_type,
        output_contract_version=output_contract_version,
        supported_prediction_modes=tuple(supported_prediction_modes),
        feature_versions={"legacy_feature_set": str(feature_version)},
        status="experimental" if value.get("status") == "active" else str(value.get("status") or "experimental"),
        calibration_status="legacy_unknown",
        guardrail_status="legacy_unknown",
        fallback_policy="legacy_unverified",
        production_ready=False,
        is_active=False,
        is_production_default=False,
        metrics_summary=value.get("metrics_summary") or {},
        selected_reason=value.get("selected_reason"),
        notes="Explicit adapter; old active/default state is not promoted automatically.",
        compatibility_tags=("legacy_adapter", "requires_manifest_validation"),
        extensions={
            "legacy_is_active": old_active,
            "legacy_is_production_default": old_default,
            "legacy_training_window_start_gw": value.get("training_window_start_gw"),
            "legacy_training_window_end_gw": value.get("training_window_end_gw"),
            "legacy_evaluation_start_gw": value.get("evaluation_start_gw"),
            "legacy_evaluation_end_gw": value.get("evaluation_end_gw"),
            "legacy_updated_at": value.get("updated_at"),
        },
    )
