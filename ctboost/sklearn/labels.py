"""Label and eval-set helpers for ctboost.sklearn."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core import Pool
from ..training import _normalize_eval_sets, _pool_from_data_and_label

def _encode_class_labels(raw_labels: Any, label_to_index: Dict[Any, int], label_name: str) -> np.ndarray:
    label_array = np.asarray(raw_labels)
    if label_array.ndim != 1:
        raise ValueError("classification labels must be a 1D array")

    try:
        encoded = [label_to_index[label] for label in label_array]
    except KeyError as exc:
        raise ValueError(f"{label_name} contains labels not seen in the training data") from exc
    return np.asarray(encoded, dtype=np.float32)

def _resolve_classifier_eval_pool(
    eval_set: Any, label_to_index: Dict[Any, int]
) -> Optional[Pool]:
    normalized = _normalize_eval_sets(eval_set)
    if not normalized:
        return None
    resolved = []
    for entry in normalized:
        if isinstance(entry, Pool):
            encoded_labels = _encode_class_labels(entry.label, label_to_index, "eval_set")
            resolved.append(_pool_from_data_and_label(entry, encoded_labels))
            continue
        if len(entry) == 2:
            X_eval, y_eval = entry
            resolved.append(
                _pool_from_data_and_label(
                    X_eval,
                    _encode_class_labels(y_eval, label_to_index, "eval_set"),
                )
            )
        else:
            X_eval, y_eval, group_id = entry
            resolved.append(
                _pool_from_data_and_label(
                    X_eval,
                    _encode_class_labels(y_eval, label_to_index, "eval_set"),
                    group_id=group_id,
                )
            )
    return resolved[0] if len(resolved) == 1 else resolved

def _encode_classifier_eval_set(eval_set: Any, label_to_index: Dict[Any, int]) -> Optional[Any]:
    normalized = _normalize_eval_sets(eval_set)
    if not normalized:
        return None
    resolved = []
    for entry in normalized:
        if isinstance(entry, Pool):
            resolved.append(entry)
            continue
        if len(entry) == 2:
            X_eval, y_eval = entry
            resolved.append((X_eval, _encode_class_labels(y_eval, label_to_index, "eval_set")))
        else:
            X_eval, y_eval, group_id = entry
            resolved.append((X_eval, _encode_class_labels(y_eval, label_to_index, "eval_set"), group_id))
    return resolved[0] if len(resolved) == 1 else resolved

def _resolve_ranker_eval_pool(eval_set: Any) -> Optional[Pool]:
    normalized = _normalize_eval_sets(eval_set)
    if not normalized:
        return None
    resolved = []
    for entry in normalized:
        if isinstance(entry, Pool):
            resolved.append(entry)
            continue
        if len(entry) != 3:
            raise TypeError("ranking eval_set entries must be Pools or (X, y, group_id) tuples")
        X_eval, y_eval, group_id = entry
        resolved.append(_pool_from_data_and_label(X_eval, y_eval, group_id=group_id))
    return resolved[0] if len(resolved) == 1 else resolved
