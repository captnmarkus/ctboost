"""Objective-specific weighting helpers for CTBoost."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..core import Pool, _clone_pool
from .schema import (
    _is_binary_classification_objective,
    _is_classification_objective,
    _is_ranking_objective,
)

def _combine_weight_arrays(*arrays: Optional[np.ndarray]) -> Optional[np.ndarray]:
    combined: Optional[np.ndarray] = None
    for array in arrays:
        if array is None:
            continue
        resolved = np.asarray(array, dtype=np.float32)
        combined = resolved.copy() if combined is None else combined * resolved
    return combined

def _apply_sample_weight_to_pool(pool: Pool, sample_weight: Optional[Any]) -> Pool:
    if sample_weight is None:
        return pool

    resolved_weight = np.asarray(sample_weight, dtype=np.float32)
    if resolved_weight.ndim != 1:
        raise ValueError("sample_weight must be a 1D array")
    if resolved_weight.shape[0] != len(pool):
        raise ValueError("sample_weight size must match the number of rows")

    combined_weight = _combine_weight_arrays(pool.weight, resolved_weight)
    return _clone_pool(
        pool,
        weight=combined_weight,
        releasable_feature_storage=True,
    )

def _balanced_class_weight(labels: np.ndarray) -> Dict[Any, float]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_count = unique_labels.size
    sample_count = labels.shape[0]
    return {
        label.item() if hasattr(label, "item") else label: float(sample_count / (class_count * count))
        for label, count in zip(unique_labels, counts)
    }

def _resolve_class_weight_map(
    labels: np.ndarray,
    class_weights: Optional[Any],
    auto_class_weights: Optional[str],
) -> Optional[Dict[Any, float]]:
    if class_weights is not None and auto_class_weights is not None:
        raise ValueError("class_weights and auto_class_weights cannot be used together")

    if auto_class_weights is not None:
        normalized = str(auto_class_weights).lower()
        if normalized != "balanced":
            raise ValueError("auto_class_weights must be 'balanced'")
        return _balanced_class_weight(labels)

    if class_weights is None:
        return None

    if isinstance(class_weights, Mapping):
        return {
            key.item() if hasattr(key, "item") else key: float(value)
            for key, value in class_weights.items()
        }

    weight_values = np.asarray(class_weights, dtype=np.float32).reshape(-1)
    unique_labels = np.unique(labels)
    if weight_values.shape[0] != unique_labels.shape[0]:
        raise ValueError("class_weights must match the number of unique labels")
    return {
        label.item() if hasattr(label, "item") else label: float(weight)
        for label, weight in zip(unique_labels.tolist(), weight_values.tolist())
    }

def _classification_weight_multiplier(
    labels: np.ndarray,
    objective_name: str,
    *,
    class_weights: Optional[Any] = None,
    auto_class_weights: Optional[str] = None,
    scale_pos_weight: Optional[float] = None,
) -> Optional[np.ndarray]:
    if (
        class_weights is None
        and auto_class_weights is None
        and scale_pos_weight is None
    ):
        return None

    if not _is_classification_objective(objective_name):
        raise ValueError("class weighting parameters require a classification objective")

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("classification labels must be a 1D array")

    multiplier = np.ones(labels.shape[0], dtype=np.float32)
    class_weight_map = _resolve_class_weight_map(labels, class_weights, auto_class_weights)
    if class_weight_map is not None:
        unique_labels = {
            label.item() if hasattr(label, "item") else label for label in np.unique(labels).tolist()
        }
        if set(class_weight_map) != unique_labels:
            raise ValueError("class_weights must provide a weight for every class in the data")
        for label_value, weight in class_weight_map.items():
            if weight <= 0.0:
                raise ValueError("class weights must be positive")
            multiplier[labels == label_value] *= float(weight)

    if scale_pos_weight is not None:
        if not _is_binary_classification_objective(objective_name):
            raise ValueError("scale_pos_weight is only supported for binary classification")
        scale_value = float(scale_pos_weight)
        if scale_value <= 0.0:
            raise ValueError("scale_pos_weight must be positive")
        positive_mask = labels == 1
        multiplier[positive_mask] *= scale_value

    return multiplier

def _apply_objective_weights(pool: Pool, params: Mapping[str, Any], objective_name: str) -> Pool:
    objective_weight = _classification_weight_multiplier(
        np.asarray(pool.label),
        objective_name,
        class_weights=params.get("class_weights"),
        auto_class_weights=params.get("auto_class_weights"),
        scale_pos_weight=params.get("scale_pos_weight"),
    )
    if objective_weight is None:
        return pool

    combined_weight = _combine_weight_arrays(pool.weight, objective_weight)
    return _clone_pool(
        pool,
        weight=combined_weight,
        releasable_feature_storage=True,
    )

