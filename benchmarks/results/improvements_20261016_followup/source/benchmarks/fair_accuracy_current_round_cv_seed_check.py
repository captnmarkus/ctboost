"""Inactive draft: fixed seed677 for all36 selected current-data CV policies."""
from __future__ import annotations
import copy
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import traceback

MAIN_PATH = Path(__file__).with_name("fair_accuracy_current_round_cv.py")
spec = importlib.util.spec_from_file_location("current_cv_seed_main", MAIN_PATH)
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
base, POLICY, TEMPERATURE_PATH, predict = main.base, main.POLICY, main.TEMPERATURE_PATH, main.predict
PROTOCOL = "full_current_round_cv_fixed_seed677_20261016"


def aggregate(results):
    import numpy as np
    summaries = []
    for seed in ("47", "677", "mean"):
        for comparator in ("xgboost", "lightgbm"):
            for family in ("all", "binary", "regression", "multiclass"):
                rows = [row for row in results if family == "all" or row["problem_type"] == family]
                gains = [(row["models"][comparator]["errors"][seed] - row["models"]["ctboost"]["errors"][seed])
                         / max(row["models"][comparator]["errors"][seed], 1e-12) for row in rows]
                summaries.append({"seed": seed, "comparator": comparator, "family": family, "datasets": len(rows),
                    "median_relative_error_gain": float(np.median(gains)), "wins": sum(gain > 1e-12 for gain in gains),
                    "ties": sum(abs(gain) <= 1e-12 for gain in gains), "worst_relative_regression": max(0, -min(gains))})
    passed = all(row["median_relative_error_gain"] >= 0 and row["wins"] + row["ties"] >= 6
                 for row in summaries if row["seed"] == "mean" and row["family"] == "all")
    return summaries, passed


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve previous attempts")
    document = base.read(args.previous / "plan.json")
    parent_args = copy.copy(args)
    parent_args.output, parent_args.previous = args.previous, Path(document["previous"])
    parent = main.checked(parent_args)
    frozen = base.read(args.previous / "frozen_selections.json")
    report = base.read(args.previous / "development.json")
    if (parent["seed"] != 47 or parent["conditional_seed"] != 677 or parent["runtime"] != runtime
            or frozen["all_fits_successful"] is not True or report["development_target_met"] is not True
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or report["plan_sha256"] != base.digest(args.previous / "plan.json")
            or report["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")
            or frozen["selected_cv_sha256"] != base.digest(args.previous / "selected_cv.json")
            or any(doc.get("official_test_scored") is not False for doc in (parent, frozen, report))):
        raise ValueError("The complete primary47 gate must pass before fixed677 refits")
    main.reject_outer_access(args.previous.parent)
    jobs = main.selected_jobs(parent_args, parent)
    if jobs != frozen["jobs"]:
        raise ValueError("Only CV-selected current policies may be replicated")
    main.expected_pairs(parent["cases"], jobs, frozen["selections"])
    original = {(row["dataset"], row["library"]): row for row in frozen["selections"]}
    dependencies = dict(parent["dependencies"])
    main.add_files(dependencies, args.previous)
    for selected in original.values():
        if base.digest(selected["model_path"]) != selected["model_sha256"]:
            raise ValueError("Primary selected model changed")
        dependencies[selected["model_path"]] = selected["model_sha256"]
    dependencies[str(MAIN_PATH)] = base.digest(MAIN_PATH)
    dependencies[str(Path(__file__).resolve())] = base.digest(__file__)
    plan = {"protocol": PROTOCOL, "seed": 677, "created_at": base.now(), "script_sha256": base.digest(__file__),
        "dependencies": dependencies, "runtime": runtime, "scipy_version": importlib.metadata.version("scipy"),
        "previous": str(args.previous), "cases": parent["cases"], "resources": parent["resources"],
        "expanded_data": parent["expanded_data"], "jobs": [{**job, "original_selection": original[job["dataset"], job["library"]]} for job in jobs],
        "selection": "None: only learner seed changes; fixed counts/corrections/T and exact current fitting rows",
        "official_test_scored": False}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["protocol"] != PROTOCOL or plan["seed"] != 677 or plan["script_sha256"] != base.digest(__file__)
            or plan["runtime"] != base.bootstrap(args) or plan["scipy_version"] != importlib.metadata.version("scipy")
            or Path(plan["previous"]) != args.previous):
        raise ValueError("Frozen677 implementation/runtime/parent changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen677 dependency changed")
    main.resources.validate_resources(plan["resources"])
    return plan


def seed_job(args):
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity([plan["resources"]["cpu"]])
    if type(args.job_index) is not int or not 0 <= args.job_index < 36:
        raise ValueError("Invalid frozen seed job index")
    job = plan["jobs"][args.job_index]
    case = next(case for case in plan["cases"] if case["dataset_name"] == job["dataset"])
    path = main.job_path(args, "seed_job", args.job_index)
    if path.exists():
        raise ValueError("Preserve a started seed fit")
    record = {"dataset": job["dataset"], "library": job["library"], "seed": 677, "status": "failed",
              "original_selection": job["original_selection"], "started_at": base.now(),
              "native_fit_started": False, "native_fit_completed": False}
    base.write(path, record)
    def progress(event):
        record[event] = True
        base.write(path, record)
    try:
        payload, details, probe = main.fit_current(job, case, 677, plan, progress)
        record.update(status="ok", refit_details=details, **main.cv.threshold.persist(payload,
            args.artifacts / "models" / job["dataset"] / (job["library"] + ".pkl"), probe))
    except Exception:
        record["error"] = traceback.format_exc()
    record["finished_at"] = base.now()
    base.write(path, main.union.refit.plain_receipt(record))


def run(args):
    plan = checked(args)
    records = main.dispatch(args, plan, "seed_job", [{"dataset": job["dataset"], "library": job["library"], "seed": 677}
        for job in plan["jobs"]], script=Path(__file__))
    success = len(records) == 36 and all(row["status"] == "ok" for row in records)
    if success:
        main.expected_pairs(plan["cases"], plan["jobs"], records)
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "selections": records, "all_fits_successful": success,
        "completion_counts": main.completion_counts(records, 36), "official_test_scored": False})


def development(args):
    import numpy as np
    plan = checked(args)
    frozen = base.read(args.output / "frozen_selections.json")
    if frozen["plan_sha256"] != base.digest(args.output / "plan.json") or frozen["all_fits_successful"] is not True:
        raise ValueError("All36 fresh677 fits must succeed")
    main.expected_pairs(plan["cases"], plan["jobs"], frozen["selections"])
    if (args.output / "development.json").exists():
        raise ValueError("Preserve the existing assessment")
    original = base.read(args.previous / "development.json")
    results = []
    for case in plan["cases"]:
        data = base.load_role(case, "development")
        old = next(row for row in original["results"] if row["dataset"] == case["dataset_name"])
        row = {"dataset": case["dataset_name"], "problem_type": case["problem_type"], "models": {}}
        for library in base.LIBRARIES:
            selected = next(row for row in frozen["selections"] if (row["dataset"], row["library"]) == (case["dataset_name"], library))
            if base.digest(selected["model_path"]) != selected["model_sha256"]:
                raise ValueError("Seed677 model changed")
            with Path(selected["model_path"]).open("rb") as stream:
                payload = pickle.load(stream)
            predictions = predict(payload, data["X"])
            error677, error47 = base.error(predictions, data["y"], case["problem_type"]), old["models"][library]["error"]
            path = args.artifacts / "development" / case["dataset_name"] / (library + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, predictions=predictions, labels=data["y"], outer_train_positions=data["outer_train_positions"])
            row["models"][library] = {"errors": {"47": error47, "677": error677, "mean": (error47 + error677) / 2},
                                     "predictions677_sha256": base.digest(path), "selection677": selected}
        results.append(row)
    summary, passed = aggregate(results)
    base.write(args.output / "development.json", {"created_at": base.now(), "plan_sha256": base.digest(args.output / "plan.json"),
        "frozen_selections_sha256": base.digest(args.output / "frozen_selections.json"), "development_target_met": passed,
        "official_test_scored": False, "results": results, "summary": summary})


if __name__ == "__main__":
    args = main.parser(("prepare", "run", "seed_job", "development")).parse_args()
    globals()[args.stage](args)
