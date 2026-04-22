"""Standalone export facade for trained CTBoost models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from .export_codegen import _standalone_python_source
from .export_payload import _normalize_export_format, _standalone_python_payload
from .export_runtime import ExportedPredictor, load_exported_predictor

PathLike = Union[str, Path]


def export_model(
    path: PathLike,
    handle: Any,
    *,
    export_format: Optional[str] = None,
    feature_pipeline: Any = None,
    prepared_features: bool = False,
) -> None:
    resolved_format = _normalize_export_format(export_format)
    if feature_pipeline is not None and not prepared_features:
        raise ValueError(
            "standalone python export currently supports numeric or already-prepared features only; "
            "pass prepared_features=True to export a scorer that expects transformed features"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _standalone_python_payload(
        handle,
        expects_prepared_features=prepared_features or feature_pipeline is None,
    )
    if resolved_format == "python":
        destination.write_text(_standalone_python_source(payload), encoding="utf-8")
        return
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
