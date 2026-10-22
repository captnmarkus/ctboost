"""Final single-model serving diagnostic with immutable all-batch parity receipts.

No fitting, quality recomputation, or selection. Run --check-only before timing;
each timed pass must bind to that exact source/runtime/model/input reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import sys
import time

for _name in ("CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import fair_accuracy_scout as accuracy
from benchmarks import fixed_temperature


def measure(call):
    for _ in range(3):
        call()
    started = time.perf_counter()
    for _ in range(5):
        call()
    count = min(1000, max(2, int(.05 / max((time.perf_counter() - started) / 5, 1e-6))))
    samples = []
    for _ in range(7):
        started = time.perf_counter()
        for _ in range(count):
            call()
        samples.append(1000 * (time.perf_counter() - started) / count)
    return dict(median_ms=statistics.median(samples), block_ms=samples, calls_per_block=count)


def assert_bits(actual, expected):
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Frozen output shape or dtype changed")
    np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint8),
                                  np.ascontiguousarray(expected).view(np.uint8))


def checked_dependencies(plan):
    dependencies = plan.get("dependencies", {})
    for path, expected in dependencies.items():
        if accuracy.digest(path) != expected:
            raise ValueError(f"Frozen accuracy dependency changed: {path}")
    for module in (fixed_temperature, accuracy):
        path = Path(module.__file__).resolve()
        values = [value for key, value in dependencies.items() if Path(key).resolve() == path]
        if values != [accuracy.digest(path)]:
            raise ValueError(f"Accuracy plan must freeze the exact helper: {path.name}")
    return dependencies


def model_rounds(payload):
    model = payload["model"]
    if payload["library"] == "ctboost":
        return int(model.num_iterations_trained)
    if payload["library"] == "xgboost":
        return int(model.num_boosted_rounds())
    return int(model.current_iteration())


def validate_payload(payload, library, problem):
    if (not isinstance(payload, dict) or "components" in payload or "format_version" in payload
            or not {"library", "problem", "schema", "model"}.issubset(payload)
            or payload["library"] != library or payload["problem"] != problem
            or not isinstance(payload["schema"], dict)):
        raise ValueError("Expected one native single-model payload for the frozen task")


def quality_metadata(model_quality, seed):
    """Report the loaded seed's quality separately from any multi-seed mean."""
    errors = model_quality.get("errors", {})
    if "errors" in model_quality:
        if str(seed) not in errors:
            raise ValueError("Frozen quality must identify the loaded single model's seed")
        error = errors[str(seed)]
    elif "error" in model_quality:
        error = model_quality["error"]
    else:
        raise ValueError("Frozen quality must identify the loaded single model's seed")
    return {"development_error": error, "development_seed": seed,
            "development_mean_error": errors.get("mean"),
            "development_quality_scope": "loaded single model; any multi-seed mean is separate metadata"}


def prediction_hash(model_quality, seed):
    """Resolve generic primary receipts or explicitly named seed companions."""
    named = f"predictions{seed}_sha256"
    generic = "predictions_sha256"
    if named in model_quality:
        expected = model_quality[named]
        if generic in model_quality and model_quality[generic] != expected:
            raise ValueError("Conflicting generic and seed-specific prediction hashes")
        field = named
    elif generic in model_quality:
        if any(key.startswith("predictions") and key.endswith("_sha256") and key != generic
               for key in model_quality):
            raise ValueError("Ambiguous prediction hashes for the loaded seed")
        field, expected = generic, model_quality[generic]
    else:
        raise ValueError("No frozen prediction hash for the loaded seed")
    if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("Invalid frozen prediction hash")
    return field, expected


def native_margins(payload, frame, *, xgb_inplace=False):
    model, library = payload["model"], payload["library"]
    if library == "xgboost":
        if xgb_inplace:
            return model.inplace_predict(frame, predict_type="margin")
        import xgboost as xgb
        return model.predict(xgb.DMatrix(frame, enable_categorical=True), output_margin=True)
    if library == "lightgbm":
        return model.predict(frame, raw_score=True, num_threads=1)
    return model.predict(frame)


def predict_normalized(payload, frame, *, xgb_inplace=False):
    return fixed_temperature.apply_temperature(payload, native_margins(payload, frame, xgb_inplace=xgb_inplace))


def predict_raw(payload, frame, *, xgb_inplace=False):
    normalized, _ = accuracy.normalize(frame, payload["schema"])
    return predict_normalized(payload, normalized, xgb_inplace=xgb_inplace)


def check_batch(payload, raw, expected=None):
    """Guard the exact batch's XGB margins before opting into inplace timing."""
    import numpy as np
    normalized, _ = accuracy.normalize(raw, payload["schema"])
    original = native_margins(payload, normalized, xgb_inplace=False)
    guard = None
    if payload["library"] == "xgboost":
        inplace = native_margins(payload, normalized, xgb_inplace=True)
        assert_bits(inplace, original)
        guard = {"bit_exact": True, "rows": len(raw), "shape": list(np.asarray(original).shape),
                 "dtype": str(np.asarray(original).dtype),
                 "margins_sha256": hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest()}
    reference = fixed_temperature.apply_temperature(payload, original)
    assert_bits(predict_raw(payload, raw, xgb_inplace=True), reference)
    assert_bits(predict_normalized(payload, normalized, xgb_inplace=True), reference)
    if expected is not None:
        assert_bits(reference, expected)
    return normalized, reference, guard


def verify_reference(reference, common):
    if not reference.get("check_only") or any(reference.get(key) != value for key, value in common.items()):
        raise ValueError("Timing source/runtime/models/batches differ from the checked reference")
    keys = [(row["dataset"], row["rows"], row["library"]) for row in reference["records"]]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate checked reference records")
    return dict(zip(keys, reference["records"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package-root", "dependencies", "accuracy-results", "accuracy-artifacts", "output"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--order", nargs=3, choices=accuracy.LIBRARIES, default=list(accuracy.LIBRARIES))
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 64, 1000])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--reference", type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    destination = args.output / f"{args.label}.json"
    if (destination.exists() or set(args.order) != set(accuracy.LIBRARIES)
            or not args.rows or len(set(args.rows)) != len(args.rows) or min(args.rows) <= 0
            or not args.check_only and args.reference is None):
        raise ValueError("Use fresh outputs, positive distinct batches, all libraries, and a checked timing reference")
    runtime = accuracy.bootstrap(args)
    import numpy as np
    import psutil
    psutil.Process().cpu_affinity([6])
    plan_path, frozen_path, quality_path = [args.accuracy_results / name for name in
                                           ("plan.json", "frozen_selections.json", "development.json")]
    plan, frozen, quality = (accuracy.read(path) for path in (plan_path, frozen_path, quality_path))
    expected_keys = {(case["dataset_name"], library) for case in plan["cases"] for library in accuracy.LIBRARIES}
    if (len(plan["cases"]) != 12 or len(expected_keys) != 36 or len(frozen["selections"]) != 36
            or {(row["dataset"], row["library"]) for row in frozen["selections"]} != expected_keys
            or not frozen["all_fits_successful"] or frozen["plan_sha256"] != accuracy.digest(plan_path)
            or quality["plan_sha256"] != accuracy.digest(plan_path)
            or quality["frozen_selections_sha256"] != accuracy.digest(frozen_path)):
        raise ValueError("Expected an intact frozen12-task/36-single-model study")
    if type(plan.get("seed")) is not int:
        raise ValueError("The frozen plan must identify the loaded single-model seed")
    dependencies = checked_dependencies(plan)
    common = dict(runtime=runtime, runner_sha256=accuracy.digest(__file__),
                  preprocessing_runner_sha256=accuracy.digest(accuracy.__file__),
                  temperature_helper_sha256=accuracy.digest(fixed_temperature.__file__),
                  selection_sha256=accuracy.digest(frozen_path), plan_sha256=accuracy.digest(plan_path),
                  development_quality_sha256=accuracy.digest(quality_path),
                  benchmark_batches=args.rows, affinity=[6], prediction_seed=plan["seed"])
    reference = accuracy.read(args.reference) if args.reference else None
    reference_records = verify_reference(reference, common) if reference else {}
    if reference and len(reference_records) != len(expected_keys) * len(args.rows):
        raise ValueError("Checked reference is incomplete")
    records = []
    for case in plan["cases"]:
        dataset = case["dataset_name"]
        source = accuracy.load_role(case, "development")["X"]
        quality_case = next(row for row in quality["results"] if row["dataset"] == dataset)
        models = {}
        for library in args.order:
            selected = next(row for row in frozen["selections"] if (row["dataset"], row["library"]) == (dataset, library))
            model_path = Path(selected["model_path"])
            if accuracy.digest(model_path) != selected["model_sha256"] or model_path.stat().st_size != selected["model_bytes"]:
                raise ValueError("Frozen single-model file or size changed")
            with model_path.open("rb") as stream:
                payload = pickle.load(stream)
            validate_payload(payload, library, case["problem_type"])
            if selected.get("seed", plan["seed"]) != plan["seed"]:
                raise ValueError("Selected model seed differs from the frozen plan")
            expected_path = args.accuracy_artifacts / "development" / dataset / f"{library}.npz"
            quality_model = quality_case["models"][library]
            prediction_field, expected_hash = prediction_hash(quality_model, plan["seed"])
            if accuracy.digest(expected_path) != expected_hash:
                raise ValueError("Saved single-model development predictions changed")
            with np.load(expected_path) as saved:
                expected = saved["predictions"]
            _, _, full_guard = check_batch(payload, source, expected)
            models[library] = payload, selected, full_guard, expected, prediction_field
        for rows in args.rows:
            indices = np.arange(rows) % len(source)
            raw = source.iloc[indices].reset_index(drop=True)
            for library in args.order:
                payload, selected, full_guard, expected, prediction_field = models[library]
                normalized, _, batch_guard = check_batch(payload, raw, expected[indices])
                record = dict(dataset=dataset, problem_type=case["problem_type"], rows=rows,
                    source_rows=len(source), input_columns=source.shape[1], library=library,
                    model_sha256=selected["model_sha256"], model_bytes=selected["model_bytes"],
                    native_model_calls=1, retained_rounds=model_rounds(payload), temperature=payload.get("temperature"),
                    **quality_metadata(quality_case["models"][library], plan["seed"]),
                    development_prediction_hash_field=prediction_field,
                    reference_max_abs_difference=0., exact_prediction_parity=True,
                    xgb_full_source_inplace_guard=full_guard, xgb_batch_inplace_guard=batch_guard,
                    input_sha256=case["roles"]["development"]["sha256"],
                    row_selection_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest())
                if reference and reference_records.get((dataset, rows, library)) != record:
                    raise ValueError("Case or model metadata differs from the checked timing reference")
                if not args.check_only:
                    record["raw"] = measure(lambda: predict_raw(payload, raw, xgb_inplace=True))
                    record["normalized_frame"] = measure(lambda: predict_normalized(payload, normalized, xgb_inplace=True))
                records.append(record)
            print(json.dumps(dict(dataset=dataset, rows=rows, complete=True)), flush=True)
    receipt = dict(**common, label=args.label, library_order=args.order, check_only=args.check_only,
        reference_sha256=accuracy.digest(args.reference) if args.reference else None,
        verified_accuracy_dependencies=dependencies, accuracy_runtime=plan["runtime"],
        development_target_met=quality["development_target_met"],
        protocol="One pass of the planned three balanced library orders; one CPU/thread;3warmups;5calibration calls;7blocks targeting50ms; batches cycle development rows.",
        scope="Raw includes one shared frozen vocabulary normalization, one complete model call and its temperature exactly once. Normalized-frame excludes only shared normalization. Full-source and each timed batch have byte-exact reference/raw/normalized checks; XGB inplace margins are checked against DMatrix on each exact input batch.",
        limitations="Selected models differ in quality/capacity; no equal-quality or leaderboard claim. No fitting, model/seed selection, new quality scoring, or official-test reads. Storage is serialized bytes, not RSS.",
        records=records)
    args.output.mkdir(parents=True, exist_ok=True)
    accuracy.write(destination, receipt)


if __name__ == "__main__":
    main()
