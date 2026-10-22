"""Fixed seed947 check for threshold-selected union models; no new selection."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import time
import traceback

BINARY_PATH = Path(__file__).with_name("fair_accuracy_threshold_scout.py")
AGGREGATE_PATH = Path(__file__).with_name("fair_accuracy_seed_check.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


binary = load_module("fixed_binary_refit", BINARY_PATH)
aggregation = load_module("original_seed_aggregation", AGGREGATE_PATH)
base, POLICY, TEMPERATURE_PATH = binary.base, binary.POLICY, binary.union.TEMPERATURE_PATH
predict, aggregate = binary.predict, aggregation.aggregate


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve previous attempts; use fresh paths")
    parent = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (parent["wave"] != 12 or parent["seed"] != 47 or parent["runtime"] != runtime
            or not frozen["all_fits_successful"] or not development["development_target_met"]
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")
            or parent["script_sha256"] != base.digest(BINARY_PATH)
            or parent["scipy_version"] != importlib.metadata.version("scipy")):
        raise ValueError("The intact mandatory union development gate must pass first")
    if frozen["selected_training_sha256"] != base.digest(args.previous / "selected_training.json"):
        raise ValueError("Frozen TRAIN-selected controls changed")
    dependencies = dict(parent["dependencies"])
    for path, expected in dependencies.items():
        if base.digest(path) != expected:
            raise ValueError("Parent dependency changed")
    for path in (BINARY_PATH, AGGREGATE_PATH):
        dependencies[str(path)] = base.digest(path)
    dependencies.update({str(args.previous / name): base.digest(args.previous / name)
                         for name in ("plan.json", "frozen_selections.json", "development.json", "selected_training.json")})
    jobs = []
    for job in frozen["jobs"]:
        original = next(s for s in frozen["selections"] if s["dataset"] == job["dataset"] and s["library"] == job["library"])
        if base.digest(original["model_path"]) != original["model_sha256"]:
            raise ValueError("Parent model changed")
        jobs.append({**job, "original_selection": original, "binary_refit": original.get("binary_refit")})
    expected = {(case["dataset_name"], library) for case in parent["cases"] for library in base.LIBRARIES}
    if (len(parent["cases"]) != 12 or len(expected) != 36 or len(jobs) != 36 or
            {(job["dataset"], job["library"]) for job in jobs} != expected or len(frozen["selections"]) != 36 or
            {(selected["dataset"], selected["library"]) for selected in frozen["selections"]} != expected):
        raise ValueError("Expected36 unique mandatory refits")
    plan = {"protocol": "threshold_union_fixed_seed947_20261016", "created_at": base.now(),
        "script_sha256": base.digest(__file__), "dependencies": dependencies, "runtime": runtime,
        "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous),
        "cases": parent["cases"], "jobs": jobs, "seed": 947, "resources": parent["resources"],
        "official_test_scored": False, "selection": "None: reuse exact union, requested rounds and post-fit controls; only learner seed changes"}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "fits": 36})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["runtime"] != base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy") or args.previous != Path(plan["previous"])):
        raise ValueError("Frozen code/runtime/parent changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def run(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    if (args.output / "fits").exists():
        raise ValueError("A started attempt is terminal; preserve it")
    (args.output / "fits").mkdir()
    selections = []
    for job in plan["jobs"]:
        while (args.artifacts / "pause_fits").exists():
            time.sleep(.5)
        case = next(case for case in plan["cases"] if case["dataset_name"] == job["dataset"])
        record = {"dataset": job["dataset"], "library": job["library"], "status": "failed", "seed": 947,
                  "original_selection": job["original_selection"], "started_at": base.now()}
        try:
            payload, details, frame = binary.fit_policy(job, case, 947)
            path = args.artifacts / "models" / job["dataset"] / (job["library"] + ".pkl")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".candidate.pkl")
            with temporary.open("wb") as stream:
                pickle.dump(payload, stream, protocol=5)
            with temporary.open("rb") as stream:
                restored = pickle.load(stream)
            np.testing.assert_array_equal(predict(payload, frame), predict(restored, frame))
            temporary.replace(path)
            record.update(status="ok", refit_details=details, model_path=str(path), model_sha256=base.digest(path),
                          model_bytes=path.stat().st_size)
        except Exception:
            record["error"] = traceback.format_exc()
        record["finished_at"] = base.now()
        selections.append(record)
        base.write(args.output / "fits" / job["dataset"] / (job["library"] + ".json"), record)
        print({"dataset": job["dataset"], "library": job["library"], "status": record["status"]}, flush=True)
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "selections": selections, "all_fits_successful": all(s["status"] == "ok" for s in selections),
        "development_scored": False, "official_test_scored": False})


def development(args):
    import numpy as np
    plan = checked(args)
    frozen = base.read(args.output / "frozen_selections.json")
    if frozen["plan_sha256"] != base.digest(args.output / "plan.json") or not frozen["all_fits_successful"]:
        raise ValueError("All36 refits must succeed before DEV")
    if (args.output / "development.json").exists():
        raise ValueError("Preserve existing assessment")
    original = base.read(Path(plan["previous"]) / "development.json")
    results = []
    for case in plan["cases"]:
        data = base.load_role(case, "development")
        original_row = next(row for row in original["results"] if row["dataset"] == case["dataset_name"])
        row = {"dataset": case["dataset_name"], "problem_type": case["problem_type"], "models": {}}
        for library in base.LIBRARIES:
            selected = next(s for s in frozen["selections"] if s["dataset"] == case["dataset_name"] and s["library"] == library)
            if base.digest(selected["model_path"]) != selected["model_sha256"]:
                raise ValueError("Selected model changed")
            with Path(selected["model_path"]).open("rb") as stream:
                payload = pickle.load(stream)
            predictions = predict(payload, data["X"])
            error947 = base.error(predictions, data["y"], case["problem_type"])
            error47 = original_row["models"][library]["error"]
            path = args.artifacts / "development" / case["dataset_name"] / (library + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, predictions=predictions, labels=data["y"], outer_train_positions=data["outer_train_positions"])
            row["models"][library] = {"errors": {"47": error47, "947": error947, "mean": (error47 + error947) / 2},
                                      "predictions947_sha256": base.digest(path), "selection947": selected}
        results.append(row)
    summary, passed = aggregate(results)
    report = {"created_at": base.now(), "plan_sha256": base.digest(args.output / "plan.json"),
        "frozen_selections_sha256": base.digest(args.output / "frozen_selections.json"), "development_target_met": passed,
        "official_test_scored": False, "results": results, "summary": summary}
    base.write(args.output / "development.json", report)
    print({"development_target_met": passed, "summary": [s for s in summary if s["family"] == "all"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "development": development}[args.stage](args)


