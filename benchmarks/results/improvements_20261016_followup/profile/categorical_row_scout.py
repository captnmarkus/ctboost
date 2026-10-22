"""Diagnostic single-row categorical extraction; no production mutation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--scout", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.meta_path = [f for f in sys.meta_path if type(f).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(args.package.resolve()))
    import ctboost
    import numpy as np
    import pandas as pd
    import psutil
    assert Path(ctboost.__file__).resolve().is_relative_to(args.package.resolve())
    assert Path(ctboost._core.__file__).resolve().is_relative_to(args.package.resolve())
    psutil.Process().cpu_affinity([6])
    sys.path.append(str(args.root.resolve()))
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame
    from ctboost.feature_pipeline import FeaturePipeline
    baseline_extract = FeaturePipeline._extract_frame

    def original(frame):
        return frame.to_numpy(dtype=object, copy=False)

    def row(frame):
        return frame.iloc[0].to_numpy(dtype=object, copy=False).reshape(1, -1)

    def guard(frame, string_only):
        return len(frame) == 1 and all(
            isinstance(dtype, pd.CategoricalDtype)
            and dtype.categories.dtype == np.dtype(object)
            and (not string_only or dtype.categories.inferred_type == "string")
            for dtype in frame.dtypes
        )

    def candidate(frame, string_only=True):
        return row(frame) if guard(frame, string_only) else original(frame)

    def patched(frame, feature_names=None, *, allow_numeric=False):
        if isinstance(frame, pd.DataFrame) and guard(frame, True):
            return row(frame), [str(name) for name in frame.columns]
        return baseline_extract(frame, feature_names, allow_numeric=allow_numeric)

    def inspect_cells(old, new):
        differences = []
        for index, (a, b) in enumerate(zip(old.flat, new.flat)):
            missing_a, missing_b = bool(pd.isna(a)), bool(pd.isna(b))
            if type(a) is not type(b) or missing_a != missing_b or (not missing_a and a != b):
                differences.append({"cell": index, "old_type": type(a).__name__, "old": repr(a),
                                    "new_type": type(b).__name__, "new": repr(b)})
        return differences

    def diagnostic(call, repetitions=5):
        call()
        elapsed = []
        for _ in range(repetitions):
            begin = time.perf_counter_ns()
            call()
            elapsed.append((time.perf_counter_ns() - begin) / 1e6)
        return {"median_ms": float(np.median(elapsed)), "samples_ms": elapsed}

    synthetic = {
        "string": pd.DataFrame({"a": pd.Categorical(["x"]), "b": pd.Categorical(["y"])}),
        "string_missing": pd.DataFrame({"a": pd.Categorical(["x"]), "b": pd.Categorical([None], categories=["x", "y"])}),
        "all_string_missing": pd.DataFrame({"a": pd.Categorical([None], categories=["x"]), "b": pd.Categorical([None], categories=["y"])}),
        "empty_vocabulary": pd.DataFrame({"a": pd.Categorical([None]), "b": pd.Categorical(["x"])}),
        "same_string_vocabulary": pd.DataFrame({"a": pd.Categorical(["x"], categories=["x", "y"]), "b": pd.Categorical([None], categories=["x", "y"])}),
        "integer": pd.DataFrame({"a": pd.Categorical([2**62+1]), "b": pd.Categorical([3])}),
        "integer_float": pd.DataFrame({"a": pd.Categorical([2**62+1]), "b": pd.Categorical([1.5])}),
        "integer_missing": pd.DataFrame({"a": pd.Categorical([2**62+1]), "b": pd.Categorical([None], categories=[1, 2])}),
        "float32": pd.DataFrame({"a": pd.Categorical(np.array([1.25], dtype=np.float32)), "b": pd.Categorical(np.array([-0.0], dtype=np.float32))}),
        "integer_string": pd.DataFrame({"a": pd.Categorical([2**62+1]), "b": pd.Categorical(["x"])}),
        "mixed_object_vocabulary": pd.DataFrame({"a": pd.Categorical([2**62+1], categories=pd.Index([2**62+1, "x"], dtype=object)), "b": pd.Categorical([1.5], categories=pd.Index([1.5, "y"], dtype=object))}),
        "bool_integer": pd.DataFrame({"a": pd.Categorical([True]), "b": pd.Categorical([5])}),
        "numeric_plain": pd.DataFrame({"a": [2**62+1], "b": [1.5]}),
        "mixed_plain": pd.DataFrame({"a": [2**62+1], "b": pd.Categorical(["x"])}),
    }
    result = {"purpose": "Correctness and concurrent diagnostic, not a benchmark speedup claim",
              "python_version": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
              "native_sha256": digest(ctboost._core.__file__), "script_sha256": digest(__file__),
              "synthetic": [], "real": []}
    for name, frame in synthetic.items():
        expected = original(frame)
        record = {"name": name, "object_guard": guard(frame, False), "string_guard": guard(frame, True)}
        for variant, fn in (("unguarded_row", row), ("object_guarded", lambda f: candidate(f, False)), ("string_guarded", candidate)):
            try:
                differences = inspect_cells(expected, fn(frame))
                record[variant] = {"exact_cells": not differences, "differences": differences}
            except Exception as exc:
                record[variant] = {"error": f"{type(exc).__name__}: {exc}"}
        assert record["string_guarded"]["exact_cells"]
        result["synthetic"].append(record)
    for case in ("hiva_agnostic", "splice"):
        model_path = args.scout / "fits" / case / "ctboost_default/model.pkl"
        data_path = args.scout / "data" / case / "development.pkl"
        with model_path.open("rb") as stream:
            payload = CompatibleUnpickler(stream).load()
        model, categories = payload["model"], payload["categorical_columns"]
        del payload
        with data_path.open("rb") as stream:
            payload = CompatibleUnpickler(stream).load()
        frame = payload["X"].iloc[:1]
        del payload  # Do not inspect or score labels.
        normalized = normalize_tabarena_frame(frame, categorical_columns=categories)[0]
        old_matrix = original(normalized)
        new_matrix = candidate(normalized)
        differences = inspect_cells(old_matrix, new_matrix)
        assert not differences
        old_transform = model._feature_pipeline.transform_array(normalized)[0]
        old_prediction = model.predict_proba(normalized)
        FeaturePipeline._extract_frame = staticmethod(patched)
        try:
            new_transform = model._feature_pipeline.transform_array(normalized)[0]
            new_prediction = model.predict_proba(normalized)
            np.testing.assert_array_equal(old_transform.view(np.uint32), new_transform.view(np.uint32))
            np.testing.assert_array_equal(old_prediction.view(np.uint8), new_prediction.view(np.uint8))
            reference = np.load(args.reference / f"{case}-1.npy")
            np.testing.assert_array_equal(new_prediction.view(np.uint8), reference.view(np.uint8))
            candidate_raw = diagnostic(lambda: model.predict_proba(normalized))
        finally:
            FeaturePipeline._extract_frame = staticmethod(baseline_extract)
        baseline_raw = diagnostic(lambda: model.predict_proba(normalized))
        record = {"case": case, "columns": len(frame.columns), "model_sha256": digest(model_path),
                  "input_sha256": digest(data_path), "object_guard": guard(normalized, False),
                  "string_guard": guard(normalized, True), "exact_cell_types_values_missing": True,
                  "transformed_bits_exact": True, "prediction_bits_exact": True,
                  "original_extract": diagnostic(lambda: original(normalized)),
                  "unguarded_row_extract": diagnostic(lambda: row(normalized)),
                  "object_guarded_extract": diagnostic(lambda: candidate(normalized, False)),
                  "string_guarded_extract": diagnostic(lambda: candidate(normalized, True)),
                  "baseline_raw_library": baseline_raw, "candidate_raw_library": candidate_raw}
        result["real"].append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["synthetic"], indent=2), flush=True)


if __name__ == "__main__":
    main()
