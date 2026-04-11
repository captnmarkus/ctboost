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


def _booster_document(handle: Any) -> Dict[str, Any]:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact_type": "ctboost.booster",
        "booster_state": dict(handle.export_state()),
    }


def _booster_from_document(document: Dict[str, Any]) -> Any:
    if document.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported CTBoost model schema version")
    if document.get("artifact_type") != "ctboost.booster":
        raise ValueError("JSON model does not contain a CTBoost booster")
    return _core.GradientBooster.from_state(document["booster_state"])


def save_booster(path: Path, handle: Any, *, model_format: Optional[str] = None) -> None:
    resolved_format = _normalize_model_format(path, model_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_format == "json":
        with path.open("w", encoding="utf-8") as stream:
            json.dump(_booster_document(handle), stream, indent=2, sort_keys=True)
        return

    with path.open("wb") as stream:
        pickle.dump(handle, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_booster(path: Path) -> Any:
    use_json = path.suffix.lower() in JSON_MODEL_SUFFIXES or (
        path.suffix.lower() not in PICKLE_MODEL_SUFFIXES and _looks_like_json(path)
    )
    if use_json:
        with path.open("r", encoding="utf-8") as stream:
            return _booster_from_document(json.load(stream))

    with path.open("rb") as stream:
        handle = pickle.load(stream)
    if not isinstance(handle, _core.GradientBooster):
        raise TypeError("serialized model does not contain a CTBoost booster")
    return handle


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
