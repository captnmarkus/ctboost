"""Payload helpers for standalone CTBoost exports."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ._version import __version__
from .inference_manifest import _json_ready, build_inference_manifest

JSON_PREDICTOR_FORMAT = "ctboost-json-predictor"
JSON_PREDICTOR_FORMAT_VERSION = 2


def _normalize_export_format(export_format: Optional[str]) -> str:
    resolved = "python" if export_format is None else str(export_format).strip().lower()
    if resolved in {"python", "standalone_python", "py"}:
        return "python"
    if resolved in {"json_predictor", "predictor_json", "json"}:
        return "json_predictor"
    if resolved in {"cpp", "c++", "cc", "standalone_cpp", "c_api"}:
        return "cpp"
    if resolved in {"onnx"}:
        return "onnx"
    raise ValueError(
        "export_format must be one of: python, standalone_python, json_predictor, cpp, onnx"
    )


def _standalone_python_payload(
    handle: Any,
    *,
    expects_prepared_features: bool,
    data_schema: Optional[Mapping[str, Any]] = None,
    feature_pipeline_state: Optional[Mapping[str, Any]] = None,
    artifact_kind: str = "json_predictor",
    class_labels: Optional[Sequence[Any]] = None,
    estimator_name: Optional[str] = None,
) -> dict[str, Any]:
    if (
        artifact_kind == "json_predictor"
        and feature_pipeline_state is not None
        and not expects_prepared_features
    ):
        pipeline_format = feature_pipeline_state.get(
            "feature_pipeline_format_version"
        )
        key_encoding = feature_pipeline_state.get(
            "categorical_key_encoding_version"
        )
        if pipeline_format != 3 or key_encoding != 2:
            raise ValueError(
                "raw-feature JSON export requires feature-pipeline format 3 and "
                "categorical key encoding 2; export prepared features or refit the "
                "pipeline with CTBoost 0.1.55 or newer"
            )
    state = dict(handle.export_state())
    quantization_schema = state.get("quantization_schema")
    if quantization_schema is None:
        raise ValueError("standalone export requires a trained model with a quantization schema")
    trees = list(state.get("trees", []))
    if not trees:
        raise ValueError("standalone export requires a trained model with at least one tree")
    resolved_class_labels = (
        None if class_labels is None else _json_ready(list(class_labels))
    )
    objective_name = str(handle.objective_name())
    normalized_objective = objective_name.strip().lower()
    prediction_dimension = int(handle.prediction_dimension())
    base_score = (
        []
        if not hasattr(handle, "base_score")
        else [float(value) for value in handle.base_score()]
    )
    if not base_score:
        base_score = [0.0] * prediction_dimension
    if resolved_class_labels is not None:
        if normalized_objective in {"logloss", "binary_logloss", "binary:logistic"}:
            expected_class_count = 2
        elif normalized_objective in {"multiclass", "softmax", "softmaxloss"}:
            expected_class_count = int(handle.prediction_dimension())
        else:
            raise ValueError("class_labels are only valid for classification exports")
        if len(resolved_class_labels) != expected_class_count:
            raise ValueError(
                "class_labels size must match the exported probability dimension "
                f"({expected_class_count})"
            )
    payload = {
        "format": JSON_PREDICTOR_FORMAT,
        "format_version": JSON_PREDICTOR_FORMAT_VERSION,
        "ctboost_version": __version__,
        "objective_name": objective_name,
        "learning_rate": float(handle.learning_rate()),
        "tree_learning_rates": []
        if not hasattr(handle, "tree_learning_rates")
        else [float(value) for value in handle.tree_learning_rates()],
        "base_score": base_score,
        "prediction_dimension": prediction_dimension,
        "num_features": len(list(quantization_schema["num_bins_per_feature"])),
        "expects_prepared_features": bool(expects_prepared_features),
        # Prepared predictors never execute this state. Keep its descriptive
        # fingerprint in the manifest without embedding potentially sensitive
        # fitted preprocessing values in the scoring payload.
        "feature_pipeline_state": (
            _json_ready(dict(feature_pipeline_state))
            if feature_pipeline_state is not None and not expects_prepared_features
            else None
        ),
        "quantization_schema": dict(quantization_schema),
        "trees": trees,
        "class_labels": resolved_class_labels,
    }
    payload["inference_manifest"] = build_inference_manifest(
        payload,
        data_schema=data_schema,
        feature_pipeline_state=feature_pipeline_state,
        expects_prepared_features=expects_prepared_features,
        artifact_kind=artifact_kind,
        class_labels=resolved_class_labels,
        estimator_name=estimator_name,
    )
    return payload
