"""Compare installed CTBoost wheels on identical saved models and input batches.

Run with an isolated Python and ``-I``. Synthetic cases diagnose inference only;
CatBoost uses the same tree/depth budget, not an accuracy-matched tuned model.
No preparation of the input batch is excluded from headline prediction timings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import statistics
import time
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def setup(cpus):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "CTBOOST_HIST_THREADS"):
        os.environ[name] = str(len(cpus))
    import psutil
    psutil.Process().cpu_affinity(cpus)


def runtime():
    import ctboost
    native = next(Path(ctboost.__file__).parent.glob("_core*.*"))
    return {
        "ctboost_path": ctboost.__file__,
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "feature_pipeline_sha256": hashlib.sha256(
            (Path(ctboost.__file__).parent / "feature_pipeline.py").read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("ctboost", "catboost", "numpy", "pandas")},
    }


def prepare(args):
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier

    from ctboost import CTBoostClassifier

    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name, cardinality, classes in (
        ("numeric_binary", 0, 2), ("categorical_100", 100, 2),
        ("categorical_2000", 2000, 2), ("numeric_multiclass", 0, 5),
    ):
        rng = np.random.default_rng(47)
        numeric = rng.normal(size=(4000, 12)).astype("float32")
        frame = pd.DataFrame(numeric, columns=[f"x{i}" for i in range(12)])
        categories = []
        for i in range(4 if cardinality else 0):
            column = f"cat{i}"
            categories.append(column)
            frame[column] = pd.Categorical([f"v{x}" for x in rng.integers(0, cardinality, 4000)])
        if classes == 2:
            target = (numeric[:, 0] + .7 * numeric[:, 1] * numeric[:, 2]
                      + .4 * rng.normal(size=4000) > 0).astype(int)
        else:
            target = (numeric[:, :classes] + .4 * rng.normal(size=(4000, classes))).argmax(axis=1)
        case = args.output / name
        case.mkdir()
        with (case / "input.pkl").open("wb") as stream:
            pickle.dump(frame.iloc[3000:].copy(), stream, protocol=5)
        ct = CTBoostClassifier(
            iterations=args.iterations, learning_rate=.05, max_depth=6,
            ordered_ctr=True, max_cat_threshold=64, cat_features=categories or None,
            random_seed=47, task_type="CPU", verbose=False,
        )
        start = time.perf_counter()
        ct.fit(frame.iloc[:3000], target[:3000])
        ct_fit = time.perf_counter() - start
        ct.save_model(case / "ctboost.json")
        np.save(case / "expected.npy", ct.predict_proba(frame.iloc[3000:]))
        cb = CatBoostClassifier(
            iterations=args.iterations, learning_rate=.05, depth=6,
            cat_features=categories, random_seed=47, task_type="CPU",
            thread_count=len(args.cpus), verbose=False, allow_writing_files=False,
        )
        start = time.perf_counter()
        cb.fit(frame.iloc[:3000], target[:3000])
        cb_fit = time.perf_counter() - start
        cb.save_model(str(case / "catboost.cbm"))
        record = {"case": name, "train_rows": 3000, "prediction_rows": 1000,
                  "classes": classes, "categorical_cardinality": cardinality,
                  "ctboost_fit_seconds": ct_fit, "catboost_fit_seconds": cb_fit}
        records.append(record)
        print(json.dumps(record), flush=True)
    write(args.output / "models.json", {
        "purpose": "Synthetic inference diagnostic; no accuracy or Elo claim",
        "iterations": args.iterations, "depth": 6, "runtime": runtime(),
        "cases": records, "cpu_affinity": args.cpus,
    })


def measure_call(call):
    # Block timing avoids coarse CPU timer resolution on Windows.
    for _ in range(3):
        call()
    start = time.perf_counter()
    for _ in range(10):
        call()
    estimate = max((time.perf_counter() - start) / 10, 1e-6)
    count = min(1000, max(5, int(.1 / estimate)))
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(count):
            call()
        samples.append((time.perf_counter() - start) * 1000 / count)
    return {"median_ms": statistics.median(samples), "block_ms": samples,
            "calls_per_block": count}


def measure(args):
    import numpy as np
    import psutil
    from catboost import CatBoostClassifier

    from ctboost import CTBoostClassifier

    output = args.output / f"timings-{args.label}.json"
    if output.exists():
        raise ValueError("Preserve previous measurements; use a new label")
    manifest = json.loads((args.output / "models.json").read_text(encoding="utf-8"))
    records = []
    for spec in manifest["cases"]:
        case = args.output / spec["case"]
        with (case / "input.pkl").open("rb") as stream:
            frame = pickle.load(stream)
        ct = CTBoostClassifier.load_model(case / "ctboost.json")
        cb = CatBoostClassifier()
        cb.load_model(str(case / "catboost.cbm"))
        expected = np.load(case / "expected.npy")
        artifact_hashes = {
            name: hashlib.sha256((case / name).read_bytes()).hexdigest()
            for name in ("ctboost.json", "catboost.cbm", "input.pkl", "expected.npy")
        }
        rss_before = psutil.Process().memory_info().rss
        start = time.perf_counter()
        cold_prediction = ct.predict_proba(frame)
        cold_ms = (time.perf_counter() - start) * 1000
        rss_after = psutil.Process().memory_info().rss
        np.testing.assert_array_equal(cold_prediction, expected)
        for rows in (1, 128, 1000):
            batch = frame.iloc[:rows]
            prepared = ct._transform_prediction_pool(batch)
            np.testing.assert_array_equal(ct.predict_proba(batch), expected[:rows])
            np.testing.assert_array_equal(ct.predict_proba(prepared), expected[:rows])
            record = {
                "case": spec["case"], "rows": rows,
                "artifact_hashes": artifact_hashes,
                "ctboost": measure_call(lambda ct=ct, batch=batch: ct.predict_proba(batch)),
                "catboost": measure_call(lambda cb=cb, batch=batch: cb.predict_proba(batch, thread_count=len(args.cpus))),
                "ctboost_prepared_pool_diagnostic": measure_call(lambda ct=ct, prepared=prepared: ct.predict_proba(prepared)),
                "ctboost_exact_prediction_parity": True,
                "ctboost_cold_1000_ms": cold_ms,
                "ctboost_first_prediction_rss_delta_bytes": rss_after - rss_before,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    write(output, {"runtime": runtime(), "cpu_affinity": args.cpus,
                   "concurrent_hpo_running": args.concurrent_hpo,
                   "headline_includes_input_preprocessing": True,
                   "protocol": "3 warmups; 7 blocks targeting 100ms each; median per-call wall time",
                   "results": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "measure"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--cpus", type=int, nargs="+", default=[6, 14])
    parser.add_argument("--concurrent-hpo", action="store_true")
    options = parser.parse_args()
    setup(options.cpus)
    (prepare if options.command == "prepare" else measure)(options)
