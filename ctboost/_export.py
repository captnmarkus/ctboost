"""Standalone export facade for trained CTBoost models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from .export_codegen import _standalone_python_source
from .export_cpp import standalone_cpp_source
from .export_onnx import save_onnx_model
from .export_payload import _normalize_export_format, _standalone_python_payload
from .export_runtime import ExportedPredictor, load_exported_predictor
from .inference_manifest import save_inference_manifest

PathLike = Union[str, Path]


def export_model(
    path: PathLike,
    handle: Any,
    *,
    export_format: Optional[str] = None,
    feature_pipeline: Any = None,
    prepared_features: bool = False,
    data_schema: Optional[Mapping[str, Any]] = None,
    class_labels: Optional[Sequence[Any]] = None,
    estimator_name: Optional[str] = None,
) -> None:
    destination = Path(path)
    inferred_format = export_format
    if inferred_format is None:
        if destination.suffix.lower() in {".cpp", ".cc", ".cxx"}:
            inferred_format = "cpp"
        elif destination.suffix.lower() == ".onnx":
            inferred_format = "onnx"
        elif destination.suffix.lower() == ".json":
            inferred_format = "json_predictor"
    resolved_format = _normalize_export_format(inferred_format)
    if feature_pipeline is not None and not prepared_features:
        raise ValueError(
            "standalone python export currently supports numeric or already-prepared features only; "
            "pass prepared_features=True to export a scorer that expects transformed features"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _standalone_python_payload(
        handle,
        expects_prepared_features=prepared_features or feature_pipeline is None,
        data_schema=data_schema,
        feature_pipeline_state=(
            None if feature_pipeline is None else feature_pipeline.to_state()
        ),
        artifact_kind=(
            "standalone_python"
            if resolved_format == "python"
            else "standalone_cpp"
            if resolved_format == "cpp"
            else "onnx"
            if resolved_format == "onnx"
            else "json_predictor"
        ),
        class_labels=class_labels,
        estimator_name=estimator_name,
    )
    if resolved_format == "python":
        destination.write_text(_standalone_python_source(payload), encoding="utf-8")
        return
    if resolved_format == "cpp":
        destination.write_text(standalone_cpp_source(payload), encoding="utf-8")
        return
    if resolved_format == "onnx":
        save_onnx_model(destination, payload)
        return
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def get_inference_manifest(
    handle: Any,
    *,
    feature_pipeline: Any = None,
    prepared_features: bool = False,
    data_schema: Optional[Mapping[str, Any]] = None,
    class_labels: Optional[Sequence[Any]] = None,
    estimator_name: Optional[str] = None,
) -> dict[str, Any]:
    """Return a versioned deployment contract for a trained CTBoost model."""
    payload = _standalone_python_payload(
        handle,
        expects_prepared_features=prepared_features,
        data_schema=data_schema,
        feature_pipeline_state=(
            None if feature_pipeline is None else feature_pipeline.to_state()
        ),
        artifact_kind="ctboost_model",
        class_labels=class_labels,
        estimator_name=estimator_name,
    )
    return dict(payload["inference_manifest"])


def export_inference_manifest(
    path: PathLike,
    handle: Any,
    *,
    feature_pipeline: Any = None,
    prepared_features: bool = False,
    data_schema: Optional[Mapping[str, Any]] = None,
    class_labels: Optional[Sequence[Any]] = None,
    estimator_name: Optional[str] = None,
) -> None:
    """Write a standalone JSON inference contract for a trained CTBoost model."""
    save_inference_manifest(
        path,
        get_inference_manifest(
            handle,
            feature_pipeline=feature_pipeline,
            prepared_features=prepared_features,
            data_schema=data_schema,
            class_labels=class_labels,
            estimator_name=estimator_name,
        ),
    )
