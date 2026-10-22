"""Frozen two-component serving-cost diagnostic; no fitting or quality selection.

Run --check-only first. Timed passes require that immutable reference receipt and
include exactly two complete component predictions and the frozen averaging rule.
"""
from __future__ import annotations

import argparse
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
from benchmarks import fixed_seed_ensemble as ensemble
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
    for module in (ensemble, fixed_temperature, accuracy):
        path = Path(module.__file__).resolve()
        values = [value for key, value in dependencies.items() if Path(key).resolve() == path]
        if values != [accuracy.digest(path)]:
            raise ValueError(f"Accuracy plan must freeze the exact helper: {path.name}")
    return dependencies


def component_rounds(payload):
    model = payload["model"]
    if payload["library"] == "ctboost":
        return int(model.num_iterations_trained)
    if payload["library"] == "xgboost":
        return int(model.num_boosted_rounds())
    return int(model.current_iteration())


def component_metadata(payload, selected, frozen_components, dataset):
    """Connect embedded native components to the independently frozen files."""
    components = ensemble.validate(payload)
    if [record["seed"] for record in components] != selected["seed_pair"]:
        raise ValueError("Ensemble seed order changed")
    result = []
    for record in components:
        matches = [item for item in frozen_components if
                   (item["dataset"], item["library"], item["seed"]) ==
                   (dataset, payload["library"], record["seed"])]
        if len(matches) != 1:
            raise ValueError("Expected one frozen record for each component")
        source = matches[0]
        selection_matches = [item for item in selected["components"] if item["seed"] == record["seed"]]
        if (len(selection_matches) != 1 or selection_matches[0]["model_sha256"] != source["model_sha256"]
                or record["model_id"] != f"{dataset}/{payload['library']}/seed{record['seed']}"
                or record["model_sha256"] != source["model_sha256"]
                or accuracy.digest(source["model_path"]) != source["model_sha256"]
                or Path(source["model_path"]).stat().st_size != source["model_bytes"]):
            raise ValueError("Frozen component identity, file, or size changed")
        result.append({"seed": record["seed"], "model_id": record["model_id"],
                       "model_sha256": record["model_sha256"], "model_bytes": source["model_bytes"],
                       "retained_rounds": component_rounds(record["payload"]),
                       "temperature": record["payload"].get("temperature")})
    return result


def check_batch(payload, raw, expected=None):
    """Check both timed paths; XGB component and mixture parity covers this batch."""
    normalized, _ = accuracy.normalize(raw, payload["schema"])
    guard = ensemble.verify_xgb_inplace(payload, normalized) if payload["library"] == "xgboost" else None
    reference = ensemble.predict_normalized(payload, normalized, xgb_inplace=False)
    raw_result = ensemble.predict(payload, raw, xgb_inplace=True)
    normalized_result = ensemble.predict_normalized(payload, normalized, xgb_inplace=True)
    assert_bits(raw_result, reference)
    assert_bits(normalized_result, reference)
    if expected is not None:
        assert_bits(reference, expected)
    return normalized, reference, guard


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
    if (plan["protocol"] != "fixed_two_seed_ensemble_20261016" or len(plan["cases"]) != 12
            or len(expected_keys) != 36 or len(frozen["selections"]) != 36 or len(frozen["components"]) != 72
            or {(row["dataset"], row["library"]) for row in frozen["selections"]} != expected_keys
            or not frozen["all_fits_successful"] or frozen["plan_sha256"] != accuracy.digest(plan_path)
            or quality["plan_sha256"] != accuracy.digest(plan_path)
            or quality["frozen_selections_sha256"] != accuracy.digest(frozen_path)):
        raise ValueError("Expected an intact frozen12-task/36-ensemble study")
    dependencies = checked_dependencies(plan)
    common = dict(runtime=runtime, runner_sha256=accuracy.digest(__file__),
                  preprocessing_runner_sha256=accuracy.digest(accuracy.__file__),
                  ensemble_helper_sha256=accuracy.digest(ensemble.__file__),
                  temperature_helper_sha256=accuracy.digest(fixed_temperature.__file__),
                  selection_sha256=accuracy.digest(frozen_path), plan_sha256=accuracy.digest(plan_path),
                  development_quality_sha256=accuracy.digest(quality_path),
                  benchmark_batches=args.rows, affinity=[6])
    reference = accuracy.read(args.reference) if args.reference else None
    if reference:
        if not reference["check_only"] or any(reference[key] != value for key, value in common.items()):
            raise ValueError("Timing source/runtime/models/batches differ from the checked reference")
    reference_records = {(row["dataset"], row["rows"], row["library"]): row for row in reference["records"]} if reference else {}
    records = []
    pair = "+".join(map(str, plan["seed_pair"]))
    for case in plan["cases"]:
        dataset = case["dataset_name"]
        source = accuracy.load_role(case, "development")["X"]
        quality_case = next(row for row in quality["results"] if row["dataset"] == dataset)
        models = {}
        for library in args.order:
            selected = next(row for row in frozen["selections"] if (row["dataset"], row["library"]) == (dataset, library))
            if accuracy.digest(selected["model_path"]) != selected["model_sha256"]:
                raise ValueError("Frozen ensemble file changed")
            with Path(selected["model_path"]).open("rb") as stream:
                payload = pickle.load(stream)
            if (payload["library"] != library or payload["problem"] != case["problem_type"]
                    or selected["seed_pair"] != plan["seed_pair"]):
                raise ValueError("Ensemble task, library, or seed pair differs")
            if case["problem_type"] != "regression" and selected["shared_encoded_label_ids"] != list(range(case["num_classes"])):
                raise ValueError("Frozen class order differs")
            members = component_metadata(payload, selected, frozen["components"], dataset)
            expected_path = args.accuracy_artifacts / "development" / dataset / f"{library}.npz"
            if accuracy.digest(expected_path) != quality_case["models"][library]["predictions_sha256"]:
                raise ValueError("Saved ensemble development predictions changed")
            with np.load(expected_path) as saved:
                expected = saved["predictions"]
            _, actual, full_guard = check_batch(payload, source, expected)
            models[library] = payload, selected, members, full_guard, expected
        for rows in args.rows:
            indices = np.arange(rows) % len(source)
            raw = source.iloc[indices].reset_index(drop=True)
            for library in args.order:
                payload, selected, members, full_guard, expected = models[library]
                normalized, actual, batch_guard = check_batch(payload, raw, expected[indices])
                record = dict(dataset=dataset, problem_type=case["problem_type"], rows=rows,
                    source_rows=len(source), input_columns=source.shape[1], library=library,
                    model_sha256=selected["model_sha256"], model_bytes=selected["model_bytes"],
                    components=members, native_model_calls=2, seed_pair=selected["seed_pair"],
                    retained_rounds=[member["retained_rounds"] for member in members],
                    total_retained_rounds=sum(member["retained_rounds"] for member in members),
                    component_source_bytes=sum(member["model_bytes"] for member in members),
                    output_kind=payload["output_kind"], development_error=quality_case["models"][library]["errors"][pair],
                    development_pair=pair, reference_max_abs_difference=0., exact_prediction_parity=True,
                    xgb_full_source_inplace_guard=full_guard, xgb_batch_inplace_guard=batch_guard,
                    input_sha256=case["roles"]["development"]["sha256"],
                    row_selection_sha256=accuracy.hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest())
                if reference and reference_records.get((dataset, rows, library)) != record:
                    raise ValueError("Case or component metadata differs from the checked timing reference")
                if not args.check_only:
                    record["raw"] = measure(lambda: ensemble.predict(payload, raw, xgb_inplace=True))
                    record["normalized_frame"] = measure(lambda: ensemble.predict_normalized(payload, normalized, xgb_inplace=True))
                records.append(record)
            print(json.dumps(dict(dataset=dataset, rows=rows, complete=True)), flush=True)
    receipt = dict(**common, label=args.label, library_order=args.order, check_only=args.check_only,
        reference_sha256=accuracy.digest(args.reference) if args.reference else None,
        verified_accuracy_dependencies=dependencies, accuracy_runtime=plan["runtime"],
        development_target_met=quality["development_target_met"],
        protocol="One pass of the planned three balanced library orders; one CPU/thread;3warmups;5calibration calls;7blocks targeting50ms; batches cycle development rows.",
        scope="Raw includes one shared frozen vocabulary normalization, two full component model calls with their own fitted preprocessing, each component temperature once, and probability/raw averaging. Normalized-frame excludes only shared normalization. XGB inplace is byte-checked for both components and ensemble on full source and every timed batch.",
        limitations="Failed-quality fixed ensemble policy; serving-cost diagnostic only. Models differ in quality/capacity. No fitting, model/seed selection, new quality scoring, or official-test reads. Storage is serialized bytes, not RSS.",
        records=records)
    args.output.mkdir(parents=True, exist_ok=True)
    accuracy.write(destination, receipt)


if __name__ == "__main__":
    main()
