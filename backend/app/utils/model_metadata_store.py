from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from app.schemas.model_metadata import ModelMetadataArtifact


ARTIFACT_DIR = Path("artifacts/model_metadata")


def _artifact_to_jsonable(artifact: ModelMetadataArtifact):
    try:
        return artifact.model_dump(mode="json")
    except AttributeError:
        return json.loads(artifact.json())


def _parse_artifact(data) -> ModelMetadataArtifact:
    try:
        return ModelMetadataArtifact.model_validate(data)
    except AttributeError:
        return ModelMetadataArtifact.parse_obj(data)


def get_model_metadata_artifact_path(model_name: str) -> Path:
    return ARTIFACT_DIR / f"{model_name}.json"


def save_model_metadata_artifact(artifact: ModelMetadataArtifact) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = get_model_metadata_artifact_path(artifact.model_name)
    payload = _artifact_to_jsonable(artifact)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {path}")
    return path


def maybe_load_model_metadata_artifact(model_name: str) -> Optional[ModelMetadataArtifact]:
    path = get_model_metadata_artifact_path(model_name)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    return _parse_artifact(data)


def list_model_metadata_artifacts(task_type: Optional[str] = None) -> List[ModelMetadataArtifact]:
    if not ARTIFACT_DIR.exists():
        return []

    out: List[ModelMetadataArtifact] = []
    for path in sorted(ARTIFACT_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            artifact = _parse_artifact(data)
        except Exception:
            continue

        if task_type and artifact.task_type != task_type:
            continue
        out.append(artifact)

    return out
