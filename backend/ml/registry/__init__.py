"""Versioned feature and model registry contracts."""

from ml.registry.features import (
    FEATURE_REGISTRY_VERSION,
    FeatureRegistry,
    FeatureRegistryEntry,
    FeatureRegistryError,
    adapt_legacy_feature_snapshot_metadata,
)
from ml.registry.models import (
    MODEL_REGISTRY_VERSION,
    ModelRegistry,
    ModelRegistryEntry,
    ModelRegistryError,
    adapt_legacy_model_metadata,
)

__all__ = [
    "FEATURE_REGISTRY_VERSION",
    "FeatureRegistry",
    "FeatureRegistryEntry",
    "FeatureRegistryError",
    "adapt_legacy_feature_snapshot_metadata",
    "MODEL_REGISTRY_VERSION",
    "ModelRegistry",
    "ModelRegistryEntry",
    "ModelRegistryError",
    "adapt_legacy_model_metadata",
]
