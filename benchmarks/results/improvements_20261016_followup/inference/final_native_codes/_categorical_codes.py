"""Private fast path for wide batches of independent categorical encoders."""
from __future__ import annotations

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


MAX_DICTIONARY = 32
OBJECT_DTYPE = np.dtype(object)
MINIMUM_ROWS = 1000
MINIMUM_INPUT_COLUMNS = 32


def build_code_plan(state):
    """Derive output-source association from fitted state, outside prediction."""
    width = state.get("n_features_in_")
    if width is None or width < MINIMUM_INPUT_COLUMNS or any(state.get(key) for key in (
        "one_hot_states", "combination_states", "text_states", "embedding_states"
    )):
        return None
    sources = [entry["source_index"] for entry in state.get("categorical_states", [])]
    numeric = state.get("numeric_indices", [])
    if (len(sources) != width or set(sources) != set(range(width))
            or len(numeric) != width or set(numeric) != set(range(width))):
        return None
    output_sources = list(numeric)
    for entry in state.get("ctr_states", []):
        if len(entry["source_indices"]) != 1 or entry["source_indices"][0] not in sources:
            return None
        output_sources.extend([entry["source_indices"][0]] * len(entry["output_names"]))
    if len(output_sources) != len(state.get("output_feature_names_", [])):
        return None
    return width, tuple(output_sources)


def try_code_transform(pipeline, data, plan):
    """Return exact transformed tuple or None to use the established path."""
    if (plan is None or pd is None or type(data) is not pd.DataFrame
            or plan[0] < MINIMUM_INPUT_COLUMNS
            or len(data) < MINIMUM_ROWS or data.shape[1] != plan[0]):
        return None
    column_names = data.columns
    if (type(column_names) is not pd.Index or column_names.dtype != OBJECT_DTYPE
            or not all(type(name) is str for name in column_names.values)):
        return None
    # Preserve pandas names: _extract_frame also ignores explicit names for
    # DataFrames. Delay string conversion until after eligibility is known.
    dictionaries = []
    # Reject a late unsupported column before materializing a Series for each
    # earlier column. The row/width guards amortize this public metadata scan.
    for dtype in data.dtypes:
        if type(dtype) is not pd.CategoricalDtype:
            return None
        categories = dtype.categories
        if (type(categories) is not pd.Index
                or categories.dtype != OBJECT_DTYPE
                or len(categories) > MAX_DICTIONARY):
            return None
        dictionary = categories.values
        if not all(type(value) is str for value in dictionary):
            return None
        dictionaries.append(dictionary)

    codes = []
    for _, series in data.items():
        array = series.array
        if type(array) is not pd.Categorical:
            return None
        codes.append(array.codes)

    names = [str(name) for name in data.columns]
    encoded, categories, output_names = pipeline._native._transform_categorical_codes(
        codes,
        dictionaries,
        names,
    )
    return encoded, list(categories), list(output_names)

