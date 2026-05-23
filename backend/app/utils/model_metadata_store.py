from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from app.schemas.model_metadata import ModelMetadataArtifact


BASE_DIR = Path("artifacts/model_metadata")


def _ensure_base_dir() -> Path:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_DIR


def _artifact_path(model_name: str) -> Path:
    safe_name = model_name.replace("/", "_")
    return _ensure_base_dir() / f"{safe_name}.json"


def save_model_metadata_artifact(artifact: ModelMetadataArtifact) -> Path:
    path = _artifact_path(artifact.model_name)
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path


def load_model_metadata_artifact(model_name: str) -> ModelMetadataArtifact:
    path = _artifact_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {model_name}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelMetadataArtifact(**data)


def list_model_metadata_artifacts(limit: int = 50) -> List[ModelMetadataArtifact]:
    base = _ensure_base_dir()
    paths = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    results: List[ModelMetadataArtifact] = []
    for path in paths[:limit]:
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append(ModelMetadataArtifact(**data))

    return results


def maybe_load_model_metadata_artifact(model_name: str) -> Optional[ModelMetadataArtifact]:
    path = _artifact_path(model_name)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelMetadataArtifact(**data)