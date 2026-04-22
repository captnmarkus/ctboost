"""Schema helpers for ctboost.training.booster."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _borders_from_quantization_schema(schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if schema is None:
        return None
    cut_offsets = list(schema["cut_offsets"])
    cut_values = list(schema["cut_values"])
    feature_borders = [
        cut_values[cut_offsets[index] : cut_offsets[index + 1]]
        for index in range(len(cut_offsets) - 1)
    ]
    nan_mode_lookup = {0: "Forbidden", 1: "Min", 2: "Max"}
    nan_modes = schema.get("nan_modes")
    if nan_modes is None:
        default_mode = nan_mode_lookup.get(int(schema["nan_mode"]), "Min")
        nan_mode_by_feature = [default_mode] * len(feature_borders)
    else:
        nan_mode_by_feature = [nan_mode_lookup.get(int(mode), "Min") for mode in nan_modes]
    return {
        "feature_borders": feature_borders,
        "nan_mode_by_feature": nan_mode_by_feature,
        "num_bins_per_feature": list(schema["num_bins_per_feature"]),
        "categorical_mask": list(schema["categorical_mask"]),
        "missing_value_mask": list(schema["missing_value_mask"]),
    }
