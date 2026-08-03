from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Optional, Sequence, Union


ARTIFACT_LAYOUT_VERSION = "fpl_artifact_paths_v1"
LAYOUT_PREFIX = "v1"
IMMUTABLE_PREFIX = "immutable"
POINTER_PREFIX = "pointers"

SUPPORTED_POINTER_KINDS = ("latest", "active")

LEGACY_FEATURE_SNAPSHOT_PREFIX = "feature_snapshots"
LEGACY_DECISION_RUN_PREFIX = "decision_runs"
LEGACY_RUN_SNAPSHOT_PREFIX = "run_snapshots"
LEGACY_MODEL_METADATA_PREFIX = "model_metadata"
LEGACY_OFFLINE_DATASET_PREFIX = "offline_datasets"
LEGACY_EVALUATION_PREFIX = "evaluations"

_SEASON_RE = re.compile(r"^(?P<start>\d{4})_(?P<end>\d{2})$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]*$")
_EXTENSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArtifactPathError(ValueError):
    """Raised when an artifact key or path component is unsafe or invalid."""


TimestampValue = Union[str, datetime]


def _nonempty_text(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ArtifactPathError("%s must be non-empty." % label)
    return text


def _safe_segment(value: Any, label: str) -> str:
    text = _nonempty_text(value, label)
    if text in {".", ".."}:
        raise ArtifactPathError("%s cannot be '.' or '..'." % label)
    if "/" in text or "\\" in text:
        raise ArtifactPathError("%s cannot contain path separators." % label)
    if not _SEGMENT_RE.fullmatch(text):
        raise ArtifactPathError(
            "%s must contain only letters, numbers, '.', '_', '-', or '='." % label
        )
    return text


def _safe_extension(value: Any) -> str:
    text = _nonempty_text(value, "extension").lstrip(".")
    if not _EXTENSION_RE.fullmatch(text):
        raise ArtifactPathError(
            "extension must contain only letters, numbers, '.', '_', or '-'."
        )
    return text.lower()


def normalize_season(value: Any) -> str:
    season = _nonempty_text(value, "season")
    match = _SEASON_RE.fullmatch(season)
    if match is None:
        raise ArtifactPathError("season must use YYYY_YY format.")

    start_year = int(match.group("start"))
    end_year = int(match.group("end"))
    if end_year != (start_year + 1) % 100:
        raise ArtifactPathError("season must identify consecutive years.")
    return season


def normalize_target_gw(value: Optional[Any]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ArtifactPathError("target_gw must be an integer from 1 to 38.")
    try:
        gw = int(value)
    except (TypeError, ValueError):
        raise ArtifactPathError("target_gw must be an integer from 1 to 38.")
    if gw < 1 or gw > 38:
        raise ArtifactPathError("target_gw must be an integer from 1 to 38.")
    return gw


def parse_utc_datetime(value: TimestampValue, label: str = "as_of_time") -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = _nonempty_text(value, label)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            raise ArtifactPathError("%s must be an ISO-8601 timestamp." % label)

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactPathError("%s must include timezone information." % label)
    return parsed.astimezone(timezone.utc)


def format_path_timestamp(value: TimestampValue) -> str:
    parsed = parse_utc_datetime(value)
    if parsed.microsecond:
        return parsed.strftime("%Y%m%dT%H%M%S%fZ")
    return parsed.strftime("%Y%m%dT%H%M%SZ")


def assert_relative_object_key(value: Any, label: str = "key") -> str:
    text = _nonempty_text(value, label).replace("\\", "/")
    if text.startswith("/"):
        raise ArtifactPathError("%s must be relative." % label)

    path = PurePosixPath(text)
    parts = path.parts
    if not parts:
        raise ArtifactPathError("%s must be non-empty." % label)
    if any(part in {"", ".", ".."} for part in parts):
        raise ArtifactPathError("%s cannot contain empty, '.', or '..' segments." % label)

    for index, part in enumerate(parts):
        _safe_segment(part, "%s segment %s" % (label, index))
    return path.as_posix()


def normalize_prefix(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    text = _nonempty_text(value, "prefix").replace("\\", "/").strip("/")
    if not text:
        return None
    return assert_relative_object_key(text, "prefix")


def _gw_segment(target_gw: Optional[int]) -> str:
    return "season" if target_gw is None else "%02d" % target_gw


@dataclass(frozen=True)
class ImmutableArtifactPath:
    artifact_type: str
    season: str
    target_gw: Optional[int]
    as_of_time: TimestampValue
    run_id: str
    version: str
    filename: str = "artifact"
    extension: str = "json"

    def key(self) -> str:
        artifact_type = _safe_segment(self.artifact_type, "artifact_type")
        season = normalize_season(self.season)
        target_gw = normalize_target_gw(self.target_gw)
        run_id = _safe_segment(self.run_id, "run_id")
        version = _safe_segment(self.version, "version")
        filename = _safe_segment(self.filename, "filename")
        extension = _safe_extension(self.extension)
        as_of_segment = format_path_timestamp(self.as_of_time)

        path = PurePosixPath(
            LAYOUT_PREFIX,
            IMMUTABLE_PREFIX,
            artifact_type,
            "season=%s" % season,
            "gw=%s" % _gw_segment(target_gw),
            "as_of=%s" % as_of_segment,
            "version=%s" % version,
            "run_id=%s" % run_id,
            "%s.%s" % (filename, extension),
        )
        return assert_relative_object_key(path.as_posix())


@dataclass(frozen=True)
class MutablePointerPath:
    pointer_kind: str
    artifact_type: str
    season: str
    target_gw: Optional[int]
    pointer_name: str = "current"
    extension: str = "json"

    def key(self) -> str:
        pointer_kind = _safe_segment(self.pointer_kind, "pointer_kind").lower()
        if pointer_kind not in SUPPORTED_POINTER_KINDS:
            raise ArtifactPathError(
                "pointer_kind must be one of: %s."
                % ", ".join(SUPPORTED_POINTER_KINDS)
            )

        artifact_type = _safe_segment(self.artifact_type, "artifact_type")
        season = normalize_season(self.season)
        target_gw = normalize_target_gw(self.target_gw)
        pointer_name = _safe_segment(self.pointer_name, "pointer_name")
        extension = _safe_extension(self.extension)

        path = PurePosixPath(
            LAYOUT_PREFIX,
            POINTER_PREFIX,
            pointer_kind,
            artifact_type,
            "season=%s" % season,
            "gw=%s" % _gw_segment(target_gw),
            "%s.%s" % (pointer_name, extension),
        )
        return assert_relative_object_key(path.as_posix())


def build_immutable_artifact_key(
    *,
    artifact_type: str,
    season: str,
    target_gw: Optional[int],
    as_of_time: TimestampValue,
    run_id: str,
    version: str,
    filename: str = "artifact",
    extension: str = "json",
) -> str:
    return ImmutableArtifactPath(
        artifact_type=artifact_type,
        season=season,
        target_gw=target_gw,
        as_of_time=as_of_time,
        run_id=run_id,
        version=version,
        filename=filename,
        extension=extension,
    ).key()


def build_pointer_key(
    *,
    pointer_kind: str,
    artifact_type: str,
    season: str,
    target_gw: Optional[int],
    pointer_name: str = "current",
    extension: str = "json",
) -> str:
    return MutablePointerPath(
        pointer_kind=pointer_kind,
        artifact_type=artifact_type,
        season=season,
        target_gw=target_gw,
        pointer_name=pointer_name,
        extension=extension,
    ).key()


def build_immutable_key_from_run_metadata(
    metadata: Mapping[str, Any],
    *,
    version: Optional[str] = None,
    filename: str = "artifact",
    extension: str = "json",
) -> str:
    if not isinstance(metadata, Mapping):
        raise ArtifactPathError("metadata must be a mapping.")

    versions = metadata.get("versions")
    if version is None and isinstance(versions, Mapping):
        version = (
            versions.get("artifact_version")
            or versions.get("manifest_version")
            or versions.get("model_version")
            or versions.get("feature_version")
        )

    if version is None:
        raise ArtifactPathError(
            "version is required when metadata does not provide an artifact version."
        )

    as_of_time = metadata.get("as_of_time_utc")
    if as_of_time is None:
        as_of_time = metadata.get("as_of_time")

    return build_immutable_artifact_key(
        artifact_type=_nonempty_text(metadata.get("artifact_type"), "artifact_type"),
        season=_nonempty_text(metadata.get("target_season"), "target_season"),
        target_gw=metadata.get("target_gw"),
        as_of_time=as_of_time,
        run_id=_nonempty_text(metadata.get("run_id"), "run_id"),
        version=version,
        filename=filename,
        extension=extension,
    )


def is_immutable_key(key: Any) -> bool:
    normalized = assert_relative_object_key(key)
    prefix = PurePosixPath(LAYOUT_PREFIX, IMMUTABLE_PREFIX).as_posix() + "/"
    return normalized.startswith(prefix)


def is_pointer_key(key: Any) -> bool:
    normalized = assert_relative_object_key(key)
    prefix = PurePosixPath(LAYOUT_PREFIX, POINTER_PREFIX).as_posix() + "/"
    return normalized.startswith(prefix)


def _legacy_json_key(prefix: str, identifier: Any, label: str) -> str:
    safe_identifier = _safe_segment(identifier, label)
    return assert_relative_object_key(
        PurePosixPath(prefix, "%s.json" % safe_identifier).as_posix()
    )


def legacy_feature_snapshot_key(snapshot_id: str) -> str:
    return _legacy_json_key(
        LEGACY_FEATURE_SNAPSHOT_PREFIX,
        snapshot_id,
        "snapshot_id",
    )


def legacy_decision_run_key(run_id: str) -> str:
    return _legacy_json_key(LEGACY_DECISION_RUN_PREFIX, run_id, "run_id")


def legacy_run_snapshot_key(snapshot_id: str) -> str:
    return _legacy_json_key(LEGACY_RUN_SNAPSHOT_PREFIX, snapshot_id, "snapshot_id")


def legacy_model_metadata_key(model_name: str) -> str:
    return _legacy_json_key(LEGACY_MODEL_METADATA_PREFIX, model_name, "model_name")


def legacy_offline_dataset_key(filename: str) -> str:
    safe_filename = _safe_segment(filename, "filename")
    return assert_relative_object_key(
        PurePosixPath(LEGACY_OFFLINE_DATASET_PREFIX, safe_filename).as_posix()
    )


def legacy_evaluation_key(filename: str) -> str:
    safe_filename = _safe_segment(filename, "filename")
    return assert_relative_object_key(
        PurePosixPath(LEGACY_EVALUATION_PREFIX, safe_filename).as_posix()
    )


def legacy_key_candidates(
    *,
    artifact_kind: str,
    identifier: str,
) -> Sequence[str]:
    normalized_kind = _safe_segment(artifact_kind, "artifact_kind").lower()
    builders = {
        "feature_snapshot": legacy_feature_snapshot_key,
        "decision_run": legacy_decision_run_key,
        "run_snapshot": legacy_run_snapshot_key,
        "model_metadata": legacy_model_metadata_key,
        "offline_dataset": legacy_offline_dataset_key,
        "evaluation": legacy_evaluation_key,
    }
    builder = builders.get(normalized_kind)
    if builder is None:
        raise ArtifactPathError(
            "Unsupported legacy artifact_kind: %s." % artifact_kind
        )
    return (builder(identifier),)
