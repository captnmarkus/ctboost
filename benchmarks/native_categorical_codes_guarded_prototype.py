"""Isolated inference-only categorical dictionary experiment; not production API.

The builder copies an existing isolated runtime and patches only its Python
pipeline. The checkout package and native binary remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_variant(base, destination):
    base, destination = Path(base).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Preserve existing variants; choose a new destination")
    shutil.copytree(base / "ctboost", destination / "ctboost")
    if (base / "build_receipt.json").exists():
        shutil.copyfile(base / "build_receipt.json", destination / "build_receipt.json")
    source = destination / "ctboost" / "feature_pipeline.py"
    text = source.read_text(encoding="utf-8")
    replacements = [
        ("from . import _core\n", "from . import _core\nfrom ._native_categorical_codes_prototype import build_code_plan, try_code_transform\n"),
        ("        self._categorical_only = False\n", "        self._categorical_only = False\n        self._categorical_code_plan = None\n"),
        ("        state = dict(self._native.to_state())\n", "        state = dict(self._native.to_state())\n        self._categorical_code_plan = build_code_plan(state)\n"),
        ("        raw_matrix, resolved_feature_names = self._extract_frame(\n            data,\n            feature_names,\n            allow_numeric=self._numeric_only,",
         "        if self._categorical_code_plan is not None:\n            candidate = try_code_transform(self, data, self._categorical_code_plan)\n            if candidate is not None:\n                return candidate\n        raw_matrix, resolved_feature_names = self._extract_frame(\n            data,\n            feature_names,\n            allow_numeric=self._numeric_only,")]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"Expected exactly one source marker: {old!r}")
        text = text.replace(old, new)
    source.write_text(text, encoding="utf-8")
    helper = destination / "ctboost" / "_native_categorical_codes_prototype.py"
    shutil.copyfile(__file__, helper)
    native = list((destination / "ctboost").glob("_core*.pyd"))
    receipt = {"base": str(base), "variant": str(destination),
               "base_pipeline_sha256": sha256(base / "ctboost" / "feature_pipeline.py"),
               "candidate_pipeline_sha256": sha256(source), "prototype_sha256": sha256(helper),
               "native": [{"path": str(path), "sha256": sha256(path)} for path in native],
               "scope": "Native categorical-code gather experiment; no production edit"}
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_variant(args.base, args.destination), indent=2))
