"""Diagnostic cProfile of fixed hiva/splice prediction preparation.

No fitting, label inspection, accuracy scoring, or benchmark speedup claims.
Profiles may overlap accuracy fits; native method internals remain opaque.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import platform
import pstats
import sys


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def profile_call(call, calls, destination):
    call()  # warm the same method before the diagnostic
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(calls):
        call()
    profiler.disable()
    profiler.dump_stats(str(destination.with_suffix(".prof")))
    stats = pstats.Stats(profiler)
    functions = []
    for (filename, line, function), (primitive, total, own, cumulative, _) in stats.stats.items():
        functions.append({"file": filename, "line": line, "function": function,
                          "primitive_calls": primitive, "total_calls": total,
                          "self_ms_per_request": 1000 * own / calls,
                          "cumulative_ms_per_request": 1000 * cumulative / calls})
    text = io.StringIO()
    printable = pstats.Stats(profiler, stream=text)
    printable.sort_stats("cumulative").print_stats(45)
    printable.sort_stats("tottime").print_stats(35)
    destination.with_suffix(".txt").write_text(text.getvalue(), encoding="utf-8")
    return {"calls": calls, "profile_total_ms_per_request": 1000 * stats.total_tt / calls,
            "total_function_calls": stats.total_calls,
            "top_cumulative": sorted(functions, key=lambda f: -f["cumulative_ms_per_request"])[:45],
            "top_self": sorted(functions, key=lambda f: -f["self_ms_per_request"])[:35],
            "all_functions": functions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--scout", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=3)
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "CTBOOST_HIST_THREADS"):
        os.environ[variable] = "1"
    sys.meta_path = [finder for finder in sys.meta_path
                     if type(finder).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(args.package.resolve()))
    import ctboost
    import numpy as np
    import psutil
    assert Path(ctboost.__file__).resolve().is_relative_to(args.package.resolve())
    assert Path(ctboost._core.__file__).resolve().is_relative_to(args.package.resolve())
    psutil.Process().cpu_affinity([6])
    sys.path.append(str(args.root.resolve()))
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame
    build = json.loads((args.package / "build_receipt.json").read_text())
    assert digest(ctboost._core.__file__) == build["native_sha256"]
    previous = json.loads((args.reference.parent / "combined-1.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": "Diagnostic attribution, not benchmark timing or an optimization comparison",
        "limitations": "cProfile overhead; concurrent accuracy fits; native methods opaque; scope totals not additive",
        "python": sys.executable, "python_version": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "psutil")},
        "package": str(Path(ctboost.__file__).resolve()), "build_info": ctboost.build_info(),
        "native_sha256": digest(ctboost._core.__file__), "build_receipt": build,
        "script_sha256": digest(__file__), "cpu_affinity": psutil.Process().cpu_affinity(),
        "source_hashes": {str(path.relative_to(args.package)): digest(path)
                          for path in sorted((args.package / "ctboost").rglob("*.py"))},
        "normalizer_sha256": digest(args.root / "benchmarks/tabarena/ctboost_model.py"),
        "records": [],
    }
    for case in ("hiva_agnostic", "splice"):
        model_path = args.scout / "fits" / case / "ctboost_default/model.pkl"
        data_path = args.scout / "data" / case / "development.pkl"
        with model_path.open("rb") as stream:
            payload = CompatibleUnpickler(stream).load()
        model, categories = payload["model"], payload["categorical_columns"]
        del payload
        with data_path.open("rb") as stream:
            container = CompatibleUnpickler(stream).load()
        source_frame = container["X"]
        del container  # Labels are never inspected or scored.
        raw = source_frame.iloc[np.arange(1000) % len(source_frame)].copy().reset_index(drop=True)
        del source_frame
        pipeline = model._feature_pipeline
        native = model.get_booster()._handle
        predict = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        normalize = lambda frame: normalize_tabarena_frame(frame, categorical_columns=categories)[0]
        state = pipeline.to_state()
        state_counts = {key: len(state.get(key, [])) for key in
                        ("categorical_states", "one_hot_states", "combination_states",
                         "ctr_states", "text_states", "embedding_states")}
        del state
        expected_transform = np.load(args.reference / f"{case}-transformed.npy")
        for rows in (1, 64, 1000):
            batch = raw.iloc[:rows]
            normalized = normalize(batch)
            matrix, input_names = pipeline._extract_frame(normalized, allow_numeric=pipeline._numeric_only)
            transformed, transformed_cats, output_names = pipeline.transform_array(normalized)
            expected = np.load(args.reference / f"{case}-{rows}.npy")
            np.testing.assert_array_equal(np.ascontiguousarray(transformed).view(np.uint32),
                                          np.ascontiguousarray(expected_transform[:rows]).view(np.uint32))
            prepared = model._transform_prediction_pool(normalized)
            np.testing.assert_array_equal(np.ascontiguousarray(predict(normalized)).view(np.uint8),
                                          np.ascontiguousarray(expected).view(np.uint8))
            np.testing.assert_array_equal(np.ascontiguousarray(predict(prepared)).view(np.uint8),
                                          np.ascontiguousarray(expected).view(np.uint8))
            np.testing.assert_array_equal(np.asarray(native.predict(prepared._handle)).view(np.uint32),
                                          np.asarray(native._predict_uncached(prepared._handle)).view(np.uint32))
            metadata = next(record for record in previous["records"] if record["case"] == case and record["rows"] == rows)
            record = {"case": case, "rows": rows, "model_sha256": digest(model_path),
                      "input_file_sha256": digest(data_path),
                      "input_columns": batch.shape[1], "transformed_columns": transformed.shape[1],
                      "categorical_columns": len(categories), "pipeline_state_counts": state_counts,
                      "input_dtype_counts": {str(k): int(v) for k, v in batch.dtypes.astype(str).value_counts().items()},
                      "transformed_bytes": transformed.nbytes, "transformed_fortran_contiguous": transformed.flags.f_contiguous,
                      "matrix_dtype": str(matrix.dtype), "trees": metadata["trees"], "nodes": metadata["nodes"],
                      "active_features": metadata["active_features"], "prediction_bits_exact": True,
                      "transform_bits_exact": True,
                      "estimator_booster_pipeline_same_object": pipeline is model.get_booster()._feature_pipeline,
                      "scopes": {}}
            scopes = {
                "raw_benchmark": lambda: predict(normalize(batch)),
                "raw_library": lambda: predict(normalized),
                "benchmark_normalize": lambda: normalize(batch),
                "validate_raw": lambda: model._validate_prediction_input(normalized),
                "dataframe_to_object_matrix": lambda: pipeline._extract_frame(normalized, allow_numeric=pipeline._numeric_only),
                "native_transform": lambda: pipeline._native.transform_array(matrix, input_names),
                "public_transform": lambda: pipeline.transform_array(normalized),
                "pool_construction": lambda: ctboost.Pool(transformed, cat_features=transformed_cats, feature_names=output_names),
                "prediction_pool": lambda: model._transform_prediction_pool(normalized),
                "prepared_sklearn": lambda: predict(prepared),
                "native_prediction": lambda: native.predict(prepared._handle),
            }
            for name, call in scopes.items():
                destination = args.output / f"{case}-{rows}-{name}"
                record["scopes"][name] = profile_call(call, args.calls, destination)
            report["records"].append(record)
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"case": case, "rows": rows, "scopes_ms_per_call": {
                name: values["profile_total_ms_per_request"] for name, values in record["scopes"].items()}}), flush=True)


if __name__ == "__main__":
    main()
