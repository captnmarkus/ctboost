"""Matched latency with verified frozen dependencies and temperature rules.

No fitting or model selection. All libraries receive the same development rows
and train-only vocabulary normalization; quality is reported beside latency.
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

for _name in ("CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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


def predict_normalized(payload, frame):
    if payload["library"] == "xgboost":
        margins = payload["model"].inplace_predict(frame, predict_type="margin")
    elif payload["library"] == "lightgbm":
        margins = payload["model"].predict(frame, raw_score=True, num_threads=1)
    else:
        margins = payload["model"].predict(frame)
    # Both timed paths include this exact shared validation, float64 conversion,
    # and division whenever the frozen model payload specifies a temperature.
    return fixed_temperature.apply_temperature(payload, margins)


def predict_raw(payload, frame):
    normalized, _ = accuracy.normalize(frame, payload["schema"])
    return predict_normalized(payload, normalized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--accuracy-results", type=Path, required=True)
    parser.add_argument("--accuracy-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--order", nargs=3, choices=accuracy.LIBRARIES, default=list(accuracy.LIBRARIES))
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 64, 1000])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.package_root, args.dependencies = args.package_root.resolve(), args.dependencies.resolve()
    destination = args.output / f"{args.label}.json"
    if destination.exists() or set(args.order) != set(accuracy.LIBRARIES):
        raise ValueError("Refusing to overwrite receipt or use an incomplete library order")
    runtime = accuracy.bootstrap(args)
    import numpy as np
    import psutil
    psutil.Process().cpu_affinity([6])
    plan_path = args.accuracy_results / "plan.json"
    frozen_path = args.accuracy_results / "frozen_selections.json"
    quality_path = args.accuracy_results / "development.json"
    plan, frozen, quality = (accuracy.read(path) for path in (plan_path, frozen_path, quality_path))
    if frozen["plan_sha256"] != accuracy.digest(plan_path):
        raise ValueError("Selection plan does not match")
    if quality["frozen_selections_sha256"] != accuracy.digest(frozen_path):
        raise ValueError("Quality report does not match frozen selections")
    helper_hash = accuracy.digest(fixed_temperature.__file__)
    dependencies = plan.get("dependencies", {})
    for path, expected_hash in dependencies.items():
        if accuracy.digest(path) != expected_hash:
            raise ValueError(f"Frozen accuracy dependency changed: {path}")
    frozen_helper = plan.get("temperature_helper_sha256", plan.get("helper_sha256"))
    if frozen_helper is None:
        helper_path = Path(fixed_temperature.__file__).resolve()
        matches = [value for path, value in dependencies.items()
                   if Path(path).resolve() == helper_path]
        if len(matches) != 1:
            raise ValueError("Frozen accuracy plan must identify the temperature helper")
        frozen_helper = matches[0]
    if frozen_helper != helper_hash:
        raise ValueError("Temperature helper does not match the frozen accuracy plan")
    records = []
    for case in plan["cases"]:
        dataset = case["dataset_name"]
        development = accuracy.load_role(case, "development")
        source = development["X"]
        quality_case = next(row for row in quality["results"] if row["dataset"] == dataset)
        models = {}
        for library in args.order:
            selected = next(row for row in frozen["selections"]
                            if row["dataset"] == dataset and row["library"] == library)
            if accuracy.digest(selected["model_path"]) != selected["model_sha256"]:
                raise ValueError("Selected model changed")
            with Path(selected["model_path"]).open("rb") as stream:
                payload = pickle.load(stream)
            actual = np.asarray(predict_raw(payload, source))
            expected_path = args.accuracy_artifacts / "development" / dataset / f"{library}.npz"
            if accuracy.digest(expected_path) != quality_case["models"][library]["predictions_sha256"]:
                raise ValueError("Frozen predictions changed")
            with np.load(expected_path) as saved:
                expected = saved["predictions"]
            # In-place XGBoost is checked against the accuracy study's DMatrix
            # output. CTBoost and every calibrated output must preserve bits;
            # compare logical C order because LightGBM may return F-order views.
            if library == "ctboost" or payload.get("temperature") is not None:
                assert actual.shape == expected.shape and actual.dtype == expected.dtype
                np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint8),
                                              np.ascontiguousarray(expected).view(np.uint8))
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            models[library] = payload, selected, float(np.max(np.abs(actual - expected), initial=0))
        for rows in args.rows:
            raw = source.iloc[np.arange(rows) % len(source)].reset_index(drop=True)
            for library in args.order:
                payload, selected, max_difference = models[library]
                normalized, _ = accuracy.normalize(raw, payload["schema"])
                record = dict(dataset=dataset, problem_type=case["problem_type"], rows=rows,
                              source_rows=len(source), input_columns=source.shape[1], library=library,
                              model_sha256=selected["model_sha256"], model_bytes=selected["model_bytes"],
                              retained_rounds=selected["best"]["retained_rounds"],
                              temperature=payload.get("temperature"),
                              development_error=quality_case["models"][library]["error"],
                              reference_max_abs_difference=max_difference,
                              input_sha256=case["roles"]["development"]["sha256"])
                if not args.check_only:
                    record["raw"] = measure(lambda: predict_raw(payload, raw))
                    record["normalized_frame"] = measure(lambda: predict_normalized(payload, normalized))
                records.append(record)
            print(json.dumps(dict(dataset=dataset, rows=rows, complete=True)), flush=True)
    receipt = dict(label=args.label, runtime=runtime, library_order=args.order,
                   affinity=psutil.Process().cpu_affinity(), runner_sha256=accuracy.digest(__file__),
                   preprocessing_runner_sha256=accuracy.digest(accuracy.__file__),
                   temperature_helper_sha256=helper_hash,
                   verified_accuracy_dependencies=dependencies,
                   selection_sha256=accuracy.digest(frozen_path), plan_sha256=accuracy.digest(plan_path),
                   development_quality_sha256=accuracy.digest(quality_path),
                   protocol="One CPU/thread;3warmups;5calibration calls;7blocks targeting50ms; batches cycle development rows.",
                   scope="Raw includes identical train-only vocabulary normalization and each library's fastest checked standard DataFrame prediction. Both paths apply the exact frozen temperature helper after margins; normalized-frame excludes only shared normalization.",
                   limitations="Selected models differ in quality/capacity; reused development data, no equal-quality or leaderboard claim. No fitting, holdout selection, or official test reads.",
                   records=records)
    args.output.mkdir(parents=True, exist_ok=True)
    accuracy.write(destination, receipt)


if __name__ == "__main__":
    main()
