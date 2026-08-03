from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Protocol, Union

from ml.artifacts.paths import (
    assert_relative_object_key,
    is_immutable_key,
    is_pointer_key,
    normalize_prefix,
)


class ArtifactStorageError(RuntimeError):
    """Base error for artifact storage operations."""


class ArtifactNotFoundError(FileNotFoundError, ArtifactStorageError):
    """Raised when an artifact key does not exist."""


class ImmutableArtifactExistsError(FileExistsError, ArtifactStorageError):
    """Raised when an immutable artifact key already exists."""


@dataclass(frozen=True)
class ArtifactFingerprint:
    algorithm: str
    hexdigest: str
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "hexdigest": self.hexdigest,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class StoredArtifact:
    key: str
    size_bytes: int
    sha256: str
    immutable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "immutable": self.immutable,
        }


@dataclass(frozen=True)
class ArtifactListing:
    key: str
    size_bytes: int
    modified_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "size_bytes": self.size_bytes,
            "modified_at_utc": self.modified_at_utc,
        }


class ArtifactStorage(Protocol):
    """Small storage interface intended to support local and cloud adapters."""

    def exists(self, key: str) -> bool:
        ...

    def write_immutable_bytes(self, key: str, data: bytes) -> StoredArtifact:
        ...

    def write_mutable_bytes(self, key: str, data: bytes) -> StoredArtifact:
        ...

    def read_bytes(self, key: str) -> bytes:
        ...

    def list(self, prefix: Optional[str] = None) -> List[ArtifactListing]:
        ...

    def fingerprint(self, key: str) -> ArtifactFingerprint:
        ...


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _format_mtime(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


class LocalArtifactStorage:
    """Filesystem-backed implementation of the artifact storage interface."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        normalized = assert_relative_object_key(key)
        candidate = (self.root / Path(*normalized.split("/"))).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise ArtifactStorageError(
                "Artifact key resolves outside the configured storage root."
            )
        return candidate

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def write_immutable_bytes(self, key: str, data: bytes) -> StoredArtifact:
        normalized = assert_relative_object_key(key)
        if not is_immutable_key(normalized):
            raise ArtifactStorageError(
                "Immutable writes require a key under v1/immutable/."
            )
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes.")

        path = self._resolve(normalized)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise ImmutableArtifactExistsError(
                "Immutable artifact already exists: %s" % normalized
            )
        except Exception:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
            raise

        return StoredArtifact(
            key=normalized,
            size_bytes=len(data),
            sha256=_sha256_bytes(data),
            immutable=True,
        )

    def write_mutable_bytes(self, key: str, data: bytes) -> StoredArtifact:
        normalized = assert_relative_object_key(key)
        if not is_pointer_key(normalized):
            raise ArtifactStorageError(
                "Mutable writes require a key under v1/pointers/."
            )
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes.")

        path = self._resolve(normalized)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=".%s." % path.name,
            suffix=".tmp",
            dir=str(path.parent),
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp_path), str(path))
        except Exception:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

        return StoredArtifact(
            key=normalized,
            size_bytes=len(data),
            sha256=_sha256_bytes(data),
            immutable=False,
        )

    def write_immutable_text(
        self,
        key: str,
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> StoredArtifact:
        if not isinstance(text, str):
            raise TypeError("text must be str.")
        return self.write_immutable_bytes(key, text.encode(encoding))

    def write_mutable_text(
        self,
        key: str,
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> StoredArtifact:
        if not isinstance(text, str):
            raise TypeError("text must be str.")
        return self.write_mutable_bytes(key, text.encode(encoding))

    def write_immutable_json(
        self,
        key: str,
        value: Any,
        *,
        indent: Optional[int] = 2,
    ) -> StoredArtifact:
        text = json.dumps(
            value,
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        if indent is not None:
            text += "\n"
        return self.write_immutable_text(key, text)

    def write_mutable_json(
        self,
        key: str,
        value: Any,
        *,
        indent: Optional[int] = 2,
    ) -> StoredArtifact:
        text = json.dumps(
            value,
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        if indent is not None:
            text += "\n"
        return self.write_mutable_text(key, text)

    def read_bytes(self, key: str) -> bytes:
        normalized = assert_relative_object_key(key)
        path = self._resolve(normalized)
        if not path.is_file():
            raise ArtifactNotFoundError("Artifact not found: %s" % normalized)
        return path.read_bytes()

    def read_text(self, key: str, *, encoding: str = "utf-8") -> str:
        return self.read_bytes(key).decode(encoding)

    def read_json(self, key: str) -> Any:
        return json.loads(self.read_text(key))

    def list(self, prefix: Optional[str] = None) -> List[ArtifactListing]:
        normalized_prefix = normalize_prefix(prefix)
        if normalized_prefix is None:
            search_root = self.root
        else:
            search_root = self._resolve(normalized_prefix)

        if not search_root.exists():
            return []

        if search_root.is_file():
            candidates = [search_root]
        else:
            candidates = [path for path in search_root.rglob("*") if path.is_file()]

        listings: List[ArtifactListing] = []
        for path in candidates:
            relative = path.relative_to(self.root).as_posix()
            stat = path.stat()
            listings.append(
                ArtifactListing(
                    key=relative,
                    size_bytes=stat.st_size,
                    modified_at_utc=_format_mtime(stat.st_mtime),
                )
            )
        return sorted(listings, key=lambda item: item.key)

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        return [item.key for item in self.list(prefix)]

    def fingerprint(self, key: str) -> ArtifactFingerprint:
        normalized = assert_relative_object_key(key)
        path = self._resolve(normalized)
        if not path.is_file():
            raise ArtifactNotFoundError("Artifact not found: %s" % normalized)

        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size_bytes += len(chunk)

        return ArtifactFingerprint(
            algorithm="sha256",
            hexdigest=digest.hexdigest(),
            size_bytes=size_bytes,
        )

    def read_first_existing_json(self, keys: List[str]) -> Any:
        if not keys:
            raise ArtifactStorageError("At least one compatibility key is required.")

        checked: List[str] = []
        for key in keys:
            normalized = assert_relative_object_key(key)
            checked.append(normalized)
            if self.exists(normalized):
                return self.read_json(normalized)

        raise ArtifactNotFoundError(
            "No compatible artifact found. Checked: %s" % ", ".join(checked)
        )
