"""Stable CTBoost model serialization helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from . import _core

JSON_MODEL_SUFFIXES = {".json", ".ctb", ".ctboost"}
PICKLE_MODEL_SUFFIXES = {".pkl", ".pickle"}
MODEL_SCHEMA_VERSION = 1


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
) -> Dict[str, Any]:
    document = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact_type": "ctboost.booster",
        "booster_state": dict(handle.export_state()),
    }
    if feature_pipeline_state is not None:
        document["feature_pipeline_state"] = _serialize_json_value(feature_pipeline_state)
    return document


def _booster_from_document(document: Dict[str, Any]) -> Any:
    if document.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported CTBoost model schema version")
    if document.get("artifact_type") != "ctboost.booster":
        raise ValueError("JSON model does not contain a CTBoost booster")
    return _core.GradientBooster.from_state(document["booster_state"])


def save_booster(
    path: Path,
    handle: Any,
    *,
    model_format: Optional[str] = None,
    feature_pipeline_state: Optional[Dict[str, Any]] = None,
) -> None:
    resolved_format = _normalize_model_format(path, model_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = _booster_document(handle, feature_pipeline_state=feature_pipeline_state)
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

    if document.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported CTBoost model schema version")
    if document.get("artifact_type") != "ctboost.booster":
        raise ValueError("JSON model does not contain a CTBoost booster")
    if "feature_pipeline_state" in document:
        document["feature_pipeline_state"] = _deserialize_json_value(document["feature_pipeline_state"])
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
    if document.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported CTBoost model schema version")
    if document.get("artifact_type") != "ctboost.estimator":
        raise ValueError("JSON model does not contain a CTBoost estimator")
    return document
