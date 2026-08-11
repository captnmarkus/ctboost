"""Stable CTBoost model serialization helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from . import _core

JSON_MODEL_SUFFIXES = {".json", ".ctb", ".ctboost"}
PICKLE_MODEL_SUFFIXES = {".pkl", ".pickle"}
MODEL_SCHEMA_VERSION = 2
SUPPORTED_MODEL_SCHEMA_VERSIONS = frozenset({1, MODEL_SCHEMA_VERSION})


def _validate_model_schema_version(document: Dict[str, Any]) -> None:
    version = document.get("schema_version")
    if version not in SUPPORTED_MODEL_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported CTBoost model schema version: {version!r}")


def _serialize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            "__ctboost_type__": "dict",
            "items": [[_serialize_json_value(key), _serialize_json_value(item)] for key, item in value.items()],
        }
    if isinstance(value, tuple):
        return {
            "__ctboost_type__": "tuple",
            "items": [_serialize_json_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_serialize_json_value(item) for item in value]
    return value


def _deserialize_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_deserialize_json_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    if value.get("__ctboost_type__") == "dict":
        return {
            _deserialize_json_value(key): _deserialize_json_value(item)
            for key, item in value["items"]
        }
    if value.get("__ctboost_type__") == "tuple":
        return tuple(_deserialize_json_value(item) for item in value["items"])
    return {key: _deserialize_json_value(item) for key, item in value.items()}


def _feature_pipeline_state_from_document(
    document: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    artifact_type = document.get("artifact_type")
    if artifact_type == "ctboost.booster":
        pipeline_state = document.get("feature_pipeline_state")
    elif artifact_type == "ctboost.estimator":
        fitted_state = document.get("fitted_state")
        pipeline_state = (
            fitted_state.get("feature_pipeline_state")
            if isinstance(fitted_state, dict)
            else None
        )
    else:
        pipeline_state = None
    if pipeline_state is None:
        return None
    decoded = _deserialize_json_value(pipeline_state)
    if not isinstance(decoded, dict):
        raise ValueError("CTBoost feature pipeline state must be a mapping")
    return decoded


def _validate_model_document(document: Dict[str, Any]) -> None:
    _validate_model_schema_version(document)
    pipeline_state = _feature_pipeline_state_from_document(document)
    if pipeline_state is None or document.get("schema_version") != 1:
        return

    format_version = pipeline_state.get("feature_pipeline_format_version", 1)
    encoding_version = pipeline_state.get("categorical_key_encoding_version", 1)
    if not isinstance(format_version, int) or not isinstance(encoding_version, int):
        raise ValueError("CTBoost feature pipeline versions must be integers")
    if format_version >= 3 or encoding_version >= 2:
        raise ValueError(
            "CTBoost model schema version 1 cannot contain feature pipeline "
            f"format {format_version} or categorical key encoding {encoding_version}"
        )


def _normalize_model_format(path: Path, model_format: Optional[str]) -> str:
    if model_format is None or model_format == "auto":
        suffix = path.suffix.lower()
        if suffix in PICKLE_MODEL_SUFFIXES:
            return "pickle"
        if suffix in JSON_MODEL_SUFFIXES:
            return "json"
        return "json"

    normalized = str(model_format).lower()
    if normalized not in {"json", "pickle"}:
        raise ValueError("model_format must be one of: auto, json, pickle")
    return normalized


def _looks_like_json(path: Path) -> bool:
    with path.open("rb") as stream:
        prefix = stream.read(32).lstrip()
    return prefix.startswith(b"{")


def _booster_document(
    handle: Any,
    *,
    feature_pipeline_state: Optional[Dict[str, Any]] = None,
    training_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    document = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact_type": "ctboost.booster",
        "booster_state": dict(handle.export_state()),
    }
    if feature_pipeline_state is not None:
        document["feature_pipeline_state"] = _serialize_json_value(feature_pipeline_state)
    if training_state is not None:
        document["training_state"] = _serialize_json_value(training_state)
    return document


def _booster_from_document(document: Dict[str, Any]) -> Any:
    _validate_model_document(document)
    if document.get("artifact_type") != "ctboost.booster":
        raise ValueError("JSON model does not contain a CTBoost booster")
    return _core.GradientBooster.from_state(document["booster_state"])


def save_booster(
    path: Path,
    handle: Any,
    *,
    model_format: Optional[str] = None,
    feature_pipeline_state: Optional[Dict[str, Any]] = None,
    training_state: Optional[Dict[str, Any]] = None,
) -> None:
    resolved_format = _normalize_model_format(path, model_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = _booster_document(
        handle,
        feature_pipeline_state=feature_pipeline_state,
        training_state=training_state,
    )
    if resolved_format == "json":
        with path.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
        return

    with path.open("wb") as stream:
        pickle.dump(document, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_booster_document(path: Path) -> Dict[str, Any]:
    use_json = path.suffix.lower() in JSON_MODEL_SUFFIXES or (
        path.suffix.lower() not in PICKLE_MODEL_SUFFIXES and _looks_like_json(path)
    )
    if use_json:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    else:
        with path.open("rb") as stream:
            loaded = pickle.load(stream)
        if isinstance(loaded, _core.GradientBooster):
            document = _booster_document(loaded)
        elif isinstance(loaded, dict):
            document = loaded
        else:
            raise TypeError("serialized model does not contain a CTBoost booster")

    _validate_model_document(document)
    if document.get("artifact_type") != "ctboost.booster":
        raise ValueError("JSON model does not contain a CTBoost booster")
    if "feature_pipeline_state" in document:
        document["feature_pipeline_state"] = _deserialize_json_value(document["feature_pipeline_state"])
    if "training_state" in document:
        document["training_state"] = _deserialize_json_value(document["training_state"])
    return document


def load_booster(path: Path) -> Any:
    return _booster_from_document(load_booster_document(path))


def save_estimator(
    path: Path,
    *,
    estimator_class: str,
    init_params: Dict[str, Any],
    fitted_state: Dict[str, Any],
    model_format: Optional[str] = None,
) -> None:
    resolved_format = _normalize_model_format(path, model_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_format == "json":
        serializable_state = {key: value for key, value in fitted_state.items() if key != "python_object"}
        document = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "artifact_type": "ctboost.estimator",
            "estimator_class": estimator_class,
            "init_params": init_params,
            "fitted_state": serializable_state,
        }
        with path.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
        return

    with path.open("wb") as stream:
        pickle.dump(fitted_state["python_object"], stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_estimator_document(path: Path) -> Optional[Dict[str, Any]]:
    use_json = path.suffix.lower() in JSON_MODEL_SUFFIXES or (
        path.suffix.lower() not in PICKLE_MODEL_SUFFIXES and _looks_like_json(path)
    )
    if not use_json:
        return None

    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    _validate_model_document(document)
    if document.get("artifact_type") != "ctboost.estimator":
        raise ValueError("JSON model does not contain a CTBoost estimator")
    return document
