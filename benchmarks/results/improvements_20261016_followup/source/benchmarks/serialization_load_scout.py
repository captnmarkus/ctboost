"""Compare unchanged native model documents with compact JSON and pickle.

Preparation reuses already assessed phase-one models. Measurement loads each
artifact in a fresh process, with warm OS caches (not cold-disk latency).
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def python_sources(package):
    return {str(path.relative_to(package)): digest(path)
            for path in sorted((package / "ctboost").rglob("*.py"))}


def runtime(package):
    sys.meta_path = [item for item in sys.meta_path
                     if type(item).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(package.resolve()))
    import ctboost
    import psutil
    assert Path(ctboost.__file__).resolve().is_relative_to(package.resolve())
    assert Path(ctboost._core.__file__).resolve().is_relative_to(package.resolve())
    psutil.Process().cpu_affinity([6])
    return ctboost


def prepare(args):
    import numpy as np
    ctboost = runtime(args.package)
    args.output.mkdir(parents=True, exist_ok=False)
    args.artifacts.mkdir(parents=True, exist_ok=False)
    accuracy = json.loads((args.accuracy / "plan.json").read_text())
    fits = json.loads((args.accuracy / "fits.json").read_text())["records"]
    requests = [("breast_cancer_20261016", "baseline"),
                ("digits_20261016", "baseline"), ("digits_20261016", "vector")]
    plan = {
        "purpose": "File formatting and fresh-process loading of previously assessed fixed models",
        "timing_scope": "Load only and first prediction separately; imports/data reads excluded; warm OS file cache",
        "order": [["pretty_json", "compact_json", "pickle"],
                  ["compact_json", "pickle", "pretty_json"],
                  ["pickle", "pretty_json", "compact_json"]],
        "cpu_affinity": [6], "python": sys.executable, "python_version": platform.python_version(),
        "package": str(args.package.resolve()), "native_sha256": digest(ctboost._core.__file__),
        "python_sources": python_sources(args.package.resolve()),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "psutil")},
        "script_sha256": digest(__file__), "cases": [],
    }
    # Save protocol before any artifact measurement. No model or policy is fitted.
    write(args.output / "protocol.json", {**plan, "requests": requests})
    for case_id, variant in requests:
        case = next(item for item in accuracy["cases"] if item["id"] == case_id)
        fit = next(item for item in fits if item["case"] == case_id and item["variant"] == variant)
        source = Path(fit["model_path"])
        assert digest(source) == fit["model_sha256"]
        data_path = Path(case["roles"]["test"]["path"])
        assert digest(data_path) == case["roles"]["test"]["sha256"]
        prefix = args.artifacts / (case_id + "_" + variant)
        with source.open(encoding="utf-8") as stream:
            document = json.load(stream)
        compact = prefix.with_suffix(".json")
        with compact.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, separators=(",", ":"), sort_keys=True)
        binary = prefix.with_suffix(".pkl")
        with compact.open(encoding="utf-8") as stream:
            assert json.load(stream) == document
        del document
        gc.collect()
        with np.load(data_path) as data:
            X = data["X"].copy()
        input_path = prefix.with_suffix(".input.npy")
        expected_path = prefix.with_suffix(".expected.npy")
        np.save(input_path, X)
        model = ctboost.Booster.load_model(source)
        model.save_model(binary, model_format="pickle")
        np.save(expected_path, model.predict(X))
        del model
        artifacts = {}
        for name, path in (("pretty_json", source), ("compact_json", compact), ("pickle", binary)):
            artifacts[name] = {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}
        plan["cases"].append({"id": case_id + "_" + variant, "artifacts": artifacts,
                              "input": str(input_path.resolve()), "input_sha256": digest(input_path),
                              "expected": str(expected_path.resolve()), "expected_sha256": digest(expected_path)})
        print("prepared " + plan["cases"][-1]["id"], flush=True)
    write(args.output / "plan.json", plan)


def worker(args):
    import numpy as np
    import psutil
    ctboost = runtime(args.package)
    plan = json.loads((args.output / "plan.json").read_text())
    assert digest(ctboost._core.__file__) == plan["native_sha256"]
    assert python_sources(args.package.resolve()) == plan["python_sources"]
    assert {name: importlib.metadata.version(name) for name in ("numpy", "psutil")} == plan["versions"]
    case = next(item for item in plan["cases"] if item["id"] == args.case)
    artifact = case["artifacts"][args.format]
    # Hashing is deliberately outside the timed region; this warms OS caches.
    assert digest(artifact["path"]) == artifact["sha256"]
    assert digest(case["input"]) == case["input_sha256"]
    assert digest(case["expected"]) == case["expected_sha256"]
    X, expected = np.load(case["input"]), np.load(case["expected"])
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    model = ctboost.Booster.load_model(artifact["path"])
    load_seconds = time.perf_counter() - started
    rss_loaded = process.memory_info().rss
    started = time.perf_counter()
    actual = model.predict(X)
    first_predict_seconds = time.perf_counter() - started
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    print(json.dumps({"case": args.case, "format": args.format, "load_seconds": load_seconds,
                      "first_predict_seconds": first_predict_seconds, "raw_bits_exact": True,
                      "rows": len(X), "rss_before": rss_before, "rss_loaded": rss_loaded,
                      "peak_working_set_bytes": process.memory_info().peak_wset}))


def measure(args):
    plan = json.loads((args.output / "plan.json").read_text())
    assert digest(__file__) == plan["script_sha256"]
    results = []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for block, order in enumerate(plan["order"]):
        for case in plan["cases"]:
            for name in order:
                result = subprocess.run([plan["python"], str(Path(__file__).resolve()), "worker",
                                         "--output", str(args.output.resolve()), "--package", plan["package"],
                                         "--case", case["id"], "--format", name],
                                        env=env, check=True, capture_output=True, text=True)
                record = json.loads(result.stdout.strip().splitlines()[-1])
                record["block"] = block
                results.append(record)
                write(args.output / "measurements.json", {"plan_sha256": digest(args.output / "plan.json"),
                                                           "records": results})
                print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "worker", "measure"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--accuracy", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--format")
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
