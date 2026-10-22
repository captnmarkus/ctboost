"""Fixed seed569 check of all36 cap-expansion policies; no new selection."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import time
import traceback

BINARY_PATH = Path(__file__).with_name("fair_accuracy_scalar_refresh.py")
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


def aggregate(results):
    import numpy as np
    summaries = []
    for seed in ("47", "569", "mean"):
        for comparator in ("xgboost", "lightgbm"):
            for family in ("all", "binary", "regression", "multiclass"):
                rows = [r for r in results if family == "all" or r["problem_type"] == family]
                gains = [(r["models"][comparator]["errors"][seed] - r["models"]["ctboost"]["errors"][seed])
                         / max(r["models"][comparator]["errors"][seed], 1e-12) for r in rows]
                summaries.append({"seed": seed, "comparator": comparator, "family": family, "datasets": len(rows),
                    "median_relative_error_gain": float(np.median(gains)), "wins": sum(g > 1e-12 for g in gains),
                    "ties": sum(abs(g) <= 1e-12 for g in gains), "worst_relative_regression": max(0, -min(gains))})
    passed = all(s["median_relative_error_gain"] >= 0 and s["wins"] + s["ties"] >= 6
                 for s in summaries if s["seed"] == "mean" and s["family"] == "all")
    return summaries, passed


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve previous attempts; use fresh paths")
    parent = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (parent["wave"] != 17 or parent["seed"] != 47 or parent["runtime"] != runtime
            or not frozen["all_fits_successful"] or not development["development_target_met"]
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")
            or parent["script_sha256"] != base.digest(BINARY_PATH)
            or parent["scipy_version"] != importlib.metadata.version("scipy")):
        raise ValueError("The intact mandatory union development gate must pass first")
    controls = base.read(args.previous / "scalar_controls.json")
    if (parent["protocol"] != "scalar_correction_refresh_clean_refit_20261016"
            or not controls["all_fits_successful"]
            or controls["plan_sha256"] != base.digest(args.previous / "plan.json")
            or frozen["scalar_controls_sha256"] != base.digest(args.previous / "scalar_controls.json")):
        raise ValueError("Expected frozen independent-STOP scalar controls")
    expected_active = {(c["dataset_name"], lib) for c in parent["cases"] if c["problem_type"] in ("binary", "regression") for lib in base.LIBRARIES}
    if len(controls["choices"]) != 24 or {(c["dataset"], c["library"]) for c in controls["choices"]} != expected_active:
        raise ValueError("Expected twenty-four unique scalar controls")
    choices = {(c["dataset"], c["library"]): c["correction"] for c in controls["choices"]}
    original_jobs = {(j["dataset"], j["library"]): j for j in parent["jobs"]}
    problems = {c["dataset_name"]: c["problem_type"] for c in parent["cases"]}
    for job in frozen["jobs"]:
        key = (job["dataset"], job["library"])
        expected_job = binary.refreshed_job(original_jobs[key], problems[job["dataset"]], choices[key]) if key in choices else original_jobs[key]
        if job != expected_job:
            raise ValueError("Only newly frozen scalar controls may change the fresh-seed jobs")
    dependencies = dict(parent["dependencies"])
    for path, expected in dependencies.items():
        if base.digest(path) != expected:
            raise ValueError("Parent dependency changed")
    for path in (BINARY_PATH, AGGREGATE_PATH):
        dependencies[str(path)] = base.digest(path)
    dependencies.update({str(args.previous / name): base.digest(args.previous / name)
                         for name in ("plan.json", "frozen_selections.json", "development.json", "scalar_controls.json")})
    prepared = {"expanded_data": parent["expanded_data"]}
    if len(prepared["expanded_data"]) != 4 or {s["dataset"] for s in prepared["expanded_data"]} != set(binary.EXPAND_DATASETS):
        raise ValueError("Expected all four inherited expanded sources")
    for source in prepared["expanded_data"]:
        if base.digest(source["path"]) != source["sha256"]:
            raise ValueError("Expanded fitting data changed")
        dependencies[source["path"]] = source["sha256"]
    jobs = []
    for job in frozen["jobs"]:
        original = next(s for s in frozen["selections"] if s["dataset"] == job["dataset"] and s["library"] == job["library"])
        if base.digest(original["model_path"]) != original["model_sha256"]:
            raise ValueError("Parent model changed")
        if job["dataset"] in binary.EXPAND_DATASETS:
            source = next(item for item in prepared["expanded_data"] if item["dataset"] == job["dataset"])
            if job["expanded_source"] != source:
                raise ValueError("Conditional seed must use the same expanded rows")
        elif job.get("expanded_source") is not None:
            raise ValueError("Ineligible task changed fitting rows")
        jobs.append({**job, "original_selection": original})
    expected = {(case["dataset_name"], library) for case in parent["cases"] for library in base.LIBRARIES}
    if (len(parent["cases"]) != 12 or len(expected) != 36 or len(jobs) != 36 or
            {(job["dataset"], job["library"]) for job in jobs} != expected or len(frozen["selections"]) != 36 or
            {(selected["dataset"], selected["library"]) for selected in frozen["selections"]} != expected):
        raise ValueError("Expected36 unique mandatory refits")
    plan = {"protocol": "scalar_refresh_fixed_seed569_20261016", "created_at": base.now(),
        "script_sha256": base.digest(__file__), "dependencies": dependencies, "runtime": runtime,
        "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous),
        "cases": parent["cases"], "jobs": jobs, "seed": 569,
        "resources": parent["resources"],
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


def fit(args):
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    job = next(j for j in plan["jobs"] if (j["dataset"], j["library"]) == (args.dataset, args.library))
    case = next(c for c in plan["cases"] if c["dataset_name"] == args.dataset)
    path = args.output / "fits" / args.dataset / (args.library + ".json")
    if path.exists():
        raise ValueError("Preserve every started fit")
    record = {"dataset": args.dataset, "library": args.library, "status": "failed", "seed": 569,
              "original_selection": job["original_selection"], "started_at": base.now()}
    base.write(path, record)
    try:
        payload, details, frame = binary.fit_policy(job, case, 569)
        model_path = args.artifacts / "models" / args.dataset / (args.library + ".pkl")
        record.update(status="ok", refit_details=details, **binary.cv.threshold.persist(payload, model_path, frame))
    except Exception:
        record["error"] = traceback.format_exc()
    record["finished_at"] = base.now()
    base.write(path, binary.union.refit.plain_receipt(record))


def run(args):
    import subprocess
    import sys
    plan = checked(args)
    if (args.output / "fits").exists() or (args.output / "logs").exists():
        raise ValueError("A started attempt is terminal; preserve it")
    (args.output / "logs").mkdir()
    selections = []
    for job in plan["jobs"]:
        while (args.artifacts / "pause_fits").exists():
            time.sleep(.5)
        command = [sys.executable, "-I", str(Path(__file__).resolve()), "fit"]
        for name in ("output", "artifacts", "package_root", "dependencies", "previous"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        command.extend(["--dataset", job["dataset"], "--library", job["library"]])
        name = f"{job['dataset']}-{job['library']}"
        with (args.output / "logs" / (name + ".txt")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            resource = binary.cv.watchdog.watch(process, maximum_seconds=plan["resources"]["maximum_job_seconds"],
                                                maximum_rss_bytes=plan["resources"]["maximum_job_rss_bytes"])
        base.write(args.output / "logs" / (name + "-resource.json"), resource)
        if not resource["cleanup_complete"]:
            raise RuntimeError("Owned worker cleanup failed; stop dispatch")
        path = args.output / "fits" / job["dataset"] / (job["library"] + ".json")
        record = base.read(path) if path.exists() else {"dataset": job["dataset"], "library": job["library"],
            "status": "failed", "seed": 569, "original_selection": job["original_selection"], "error": "not_completed"}
        if resource["exit_code"] != 0 or resource["resource_failure"] is not None:
            record.update(status="failed", resource=resource)
        base.write(path, record)
        selections.append(record)
        print({"dataset": job["dataset"], "library": job["library"], "status": record["status"]}, flush=True)
    expected = {(case["dataset_name"], library) for case in plan["cases"] for library in base.LIBRARIES}
    success = (len(selections) == 36 and {(s["dataset"], s["library"]) for s in selections} == expected
               and all(s["status"] == "ok" for s in selections))
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "selections": selections, "all_fits_successful": success,
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
            error569 = base.error(predictions, data["y"], case["problem_type"])
            error47 = original_row["models"][library]["error"]
            path = args.artifacts / "development" / case["dataset_name"] / (library + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, predictions=predictions, labels=data["y"], outer_train_positions=data["outer_train_positions"])
            row["models"][library] = {"errors": {"47": error47, "569": error569, "mean": (error47 + error569) / 2},
                                      "predictions569_sha256": base.digest(path), "selection569": selected}
        results.append(row)
    summary, passed = aggregate(results)
    report = {"created_at": base.now(), "plan_sha256": base.digest(args.output / "plan.json"),
        "frozen_selections_sha256": base.digest(args.output / "frozen_selections.json"), "development_target_met": passed,
        "official_test_scored": False, "results": results, "summary": summary}
    base.write(args.output / "development.json", report)
    print({"development_target_met": passed, "summary": [s for s in summary if s["family"] == "all"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "fit", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--library", choices=base.LIBRARIES)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "fit": fit, "development": development}[args.stage](args)





