"""Reuse the pandas category codes assigned by an estimator's training frame."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np

from .normalization import _is_pandas_categorical_series, _is_pandas_dataframe, pd


def _category_label(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bool, int)) or (
        isinstance(value, float) and math.isfinite(value)
    ):
        return value
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return ["datetime", value.isoformat()]
    if isinstance(value, date):
        return ["date", value.isoformat()]
    if isinstance(value, timedelta):
        return ["timedelta", value.days, value.seconds, value.microseconds, getattr(value, "nanoseconds", 0)]
    # Custom hashable labels need a stable, distinct textual representation.
    # Never deserialize an object or execute model-supplied code.
    return [type(value).__module__, type(value).__qualname__, repr(value)]


def _category_key(label: Any) -> Any:
    return tuple(label) if isinstance(label, list) else label


def pandas_category_schema(frame: Any) -> Any:
    if not _is_pandas_dataframe(frame):
        return None
    schema = {}
    for name, column in frame.items():
        if _is_pandas_categorical_series(column):
            levels = column.astype("category").cat.categories
            labels = [_category_label(value) for value in levels]
            if len({_category_key(label) for label in labels}) != len(labels):
                raise ValueError(f"categorical labels in {name!r} have ambiguous representations")
            schema[str(name)] = {"pandas_category_labels": labels}
    return schema or None


def align_pandas_categories(frame: Any, schema: Any) -> Any:
    if not isinstance(schema, dict) or not schema or not _is_pandas_dataframe(frame):
        return frame
    aligned = frame
    for index, (name, column) in enumerate(frame.items()):
        contract = schema.get(str(name))
        if not isinstance(contract, dict) or "pandas_category_labels" not in contract:
            continue
        if not _is_pandas_categorical_series(column):
            # A numeric DataFrame, like an array or Pool, can carry prepared codes.
            continue
        levels = contract["pandas_category_labels"]
        categorical = column.astype("category").cat
        incoming = [_category_label(v) for v in categorical.categories]
        if incoming == levels:
            continue
        positions = {_category_key(label): code for code, label in enumerate(levels)}
        # pandas uses -1 for missing values. The last entry also maps unseen
        # categories to this existing route without extending the training vocabulary.
        remap = [positions.get(_category_key(label), -1) for label in incoming]
        mapped = np.asarray([*remap, -1], dtype=np.int64)[categorical.codes.to_numpy()]
        if aligned is frame:
            aligned = frame.copy(deep=False)
        aligned.isetitem(index, pd.Categorical.from_codes(mapped, categories=range(len(levels))))
    return aligned
