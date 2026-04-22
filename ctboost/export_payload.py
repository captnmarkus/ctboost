"""Payload helpers for standalone CTBoost exports."""

from __future__ import annotations

from typing import Any, Optional

from ._version import __version__


def _normalize_export_format(export_format: Optional[str]) -> str:
    resolved = "python" if export_format is None else str(export_format).strip().lower()
    if resolved in {"python", "standalone_python", "py"}:
        return "python"
    if resolved in {"json_predictor", "predictor_json", "json"}:
        return "json_predictor"
    raise ValueError("export_format must be one of: python, standalone_python, json_predictor")


def _standalone_python_payload(handle: Any, *, expects_prepared_features: bool) -> dict[str, Any]:
    state = dict(handle.export_state())
    quantization_schema = state.get("quantization_schema")
    if quantization_schema is None:
        raise ValueError("standalone export requires a trained model with a quantization schema")
    trees = list(state.get("trees", []))
    if not trees:
        raise ValueError("standalone export requires a trained model with at least one tree")
    return {
        "ctboost_version": __version__,
        "objective_name": str(handle.objective_name()),
        "learning_rate": float(handle.learning_rate()),
        "tree_learning_rates": []
        if not hasattr(handle, "tree_learning_rates")
        else [float(value) for value in handle.tree_learning_rates()],
        "prediction_dimension": int(handle.prediction_dimension()),
        "num_features": len(list(quantization_schema["num_bins_per_feature"])),
        "expects_prepared_features": bool(expects_prepared_features),
        "quantization_schema": dict(quantization_schema),
        "trees": trees,
    }
