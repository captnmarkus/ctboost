"""Refresh scalar correction controls using fresh clean TRAIN and union models."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

PARENT_PATH = Path(__file__).with_name("fair_accuracy_multiclass_refit.py")
HELPER_PATH = Path(__file__).with_name("fixed_clean_refit_check.py")
REGRESSION_PATH = Path(__file__).with_name("fixed_leaf_refit.py")
BINARY_PATH = Path(__file__).with_name("fixed_binary_leaf_refit.py")
SEED_PATH = Path(__file__).with_name("fair_accuracy_scalar_refresh_seed_check.py")
OUTER_PATH = Path(__file__).with_name("fair_accuracy_scalar_refresh_outer_test.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_scalar_refresh.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


parent = load("scalar_refresh_parent", PARENT_PATH)
base, union, cv, predict = parent.base, parent.union, parent.cv, parent.predict
EXPAND_DATASETS = parent.EXPAND_DATASETS


def helper(problem):
    if problem not in ("binary", "regression"):
        raise ValueError("Only scalar binary/regression correction is supported")
    return load("scalar_leaf_helper", BINARY_PATH if problem == "binary" else REGRESSION_PATH)


def checker():
    return load("clean_refit_checker", HELPER_PATH)


def clean_job(job):
    result = copy.deepcopy(job)
    result.update(leaf_refit=None, binary_refit=None, multiclass_refit=None)
    return result


def clean_provenance(job):
    clean = clean_job(job)
    controls = {key: clean.get(key) for key in ("parameters", "rounds", "leaf_refit", "binary_refit", "multiclass_refit", "temperature")}
    return {"old_scalar_corrections_disabled_before_fit": True, "clean_fit_controls": controls,
            "clean_fit_controls_sha256": hashlib.sha256(json.dumps(controls, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def correction_field(problem):
    if problem not in ("binary", "regression"):
        raise ValueError("Expected scalar problem")
    return "binary_refit" if problem == "binary" else "leaf_refit"


def refreshed_job(job, problem, correction):
    result = clean_job(job)
    result[correction_field(problem)] = correction
    return result


def solve_coefficients(problem, design, labels, penalty, solver):
    if problem == "regression":
        return union.refit.solve_coefficients(design, labels, penalty, solver)
    return helper(problem).solve_coefficients(design, labels, penalty, **solver)

def selection_key(score, controls):
    return (score, 0, 0., 0.) if controls is None else (score, 1, -controls["lambda"], controls["shrinkage"])


def fitting_data(job, case, *, train_only):
    import numpy as np
    result = parent.fitting_data(job, case, train_only=train_only)
    X, labels, schema, train, stop, receipt = result
    if case["problem_type"] == "binary":
        if not np.array_equal(np.unique(labels), [0, 1]) or not np.array_equal(np.unique(stop["y"]), [0, 1]):
            raise ValueError("Both fitting and STOP must retain both binary classes")
    return result

def fold_candidate(payload, design, coefficients, shrinkage, frames):
    import numpy as np
    h = helper(payload["problem"])
    updates = h.corrections_by_tree(design["leaf_ids_by_tree"], coefficients, shrinkage)
    candidate, fold = h.fold_corrections(payload, updates)
    fold = union.refit.plain_receipt(fold)
    checks = {name: h.verify_fold(payload, candidate, frame, updates, fold,
              design["training_prepared"] if name == "fitting" else None) for name, frame in frames.items()}
    expected = design["baseline"] + shrinkage * (design["Z"] @ coefficients)
    np.testing.assert_allclose(h.predict(candidate, frames["fitting"], design["training_prepared"]),
                               expected, rtol=5e-5, atol=1e-5)
    return candidate, union.refit.plain_receipt({"fold": fold, "checks": checks,
        "independent_sparse_raw_margin_oracle": True, "preprocessing": design["preprocessing_receipt"]})

def corrected(payload, X, labels, controls, verification_frames):
    problem, h = payload["problem"], helper(payload["problem"])
    memory = checker().memory_preflight(payload, len(X), max_working_bytes=controls["max_working_bytes"], solver_kind=problem)
    design = h.prepare_design(payload, X, labels)
    coefficients, solve = solve_coefficients(problem, design, labels, controls["lambda"], controls["solver"])
    candidate, checks = fold_candidate(payload, design, coefficients, controls["shrinkage"], {"fitting": X, **verification_frames})
    return candidate, union.refit.plain_receipt({"memory": memory, "solve": solve, **checks})

def fit_policy(job, case, seed):
    """Fresh fixed-seed models: remove old scalar corrections before every fit."""
    if case["problem_type"] == "multiclass":
        return parent.fit_policy(job, case, seed)
    payload, details, frame = parent.fit_policy(clean_job(job), case, seed)
    controls = job.get(correction_field(case["problem_type"]))
    if controls is not None:
        X, labels, schema, train, stop, receipt = fitting_data(job, case, train_only=False)
        if schema != payload["schema"]:
            raise ValueError("Fresh clean model and correction vocabulary differ")
        verification = {name: base.normalize(role["X"], schema)[0]
                        for name, role in (("original_train", train), ("original_stop", stop))}
        payload, correction = corrected(payload, X, labels, controls, verification)
        details.update(scalar_refresh=correction, scalar_fitting_data=receipt)
    details["old_scalar_corrections_disabled_before_fit"] = True
    return payload, details, frame

def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve all earlier attempts")
    prior = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (prior["wave"] != 16 or prior["seed"] != 47 or prior["runtime"] != runtime
            or prior["script_sha256"] != base.digest(PARENT_PATH) or not frozen["all_fits_successful"]
            or development["development_target_met"] or prior["scipy_version"] != importlib.metadata.version("scipy")
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")
            ):
        raise ValueError("Expected intact failed wave16 policy")
    previous_args = copy.copy(args)
    previous_args.output, previous_args.previous = args.previous, Path(prior["previous"])
    parent.checked(previous_args)
    parent.controls_document(previous_args, prior)
    if frozen["multiclass_controls_sha256"] != base.digest(args.previous / "multiclass_controls.json"):
        raise ValueError("Parent multiclass controls changed")
    expanded = {"expanded_data": prior["expanded_data"]}
    expected = {(case["dataset_name"], library) for case in prior["cases"] for library in base.LIBRARIES}
    if len(prior["cases"]) != 12 or len(expected) != 36:
        raise ValueError("Expected twelve tasks")
    for records in (frozen["jobs"], frozen["selections"]):
        if len(records) != 36 or {(r["dataset"], r["library"]) for r in records} != expected:
            raise ValueError("Expected all36 unique parent policies")
    dependencies = dict(prior["dependencies"])
    for path, digest in dependencies.items():
        if base.digest(path) != digest:
            raise ValueError("Parent dependency changed")
    for path in (PARENT_PATH, HELPER_PATH, REGRESSION_PATH, BINARY_PATH, SEED_PATH, OUTER_PATH, POLICY):
        dependencies[str(path)] = base.digest(path)
    for name in ("plan.json", "frozen_selections.json", "development.json", "multiclass_controls.json"):
        dependencies[str(args.previous / name)] = base.digest(args.previous / name)
    for selected in frozen["selections"]:
        if base.digest(selected["model_path"]) != selected["model_sha256"]:
            raise ValueError("Parent model changed")
        dependencies[selected["model_path"]] = selected["model_sha256"]
    for source in expanded["expanded_data"]:
        dependencies[source["path"]] = source["sha256"]
    jobs = []
    for job in frozen["jobs"]:
        params = job["parameters"]
        for key in ("sample_weight", "class_weight", "class_weights", "baseline", "base_margin",
                    "monotone_constraints", "interaction_constraints"):
            if params.get(key) not in (None, "", [], (), {}):
                raise ValueError("Weighted/constrained/external baseline models unsupported")
        if params.get("max_leaf_weight", 0) or params.get("max_delta_step", 0):
            raise ValueError("Constrained leaf values unsupported")
        if params.get("scale_pos_weight", 1) != 1 or params.get("is_unbalance", False) or params.get("sigmoid", 1) != 1:
            raise ValueError("Weighted or non-unit binary logits unsupported")
        jobs.append(copy.deepcopy(job))
    policy = base.read(POLICY)
    plan = {"protocol": "scalar_correction_refresh_clean_refit_20261016", "wave": 17, "seed": 47,
        "created_at": base.now(), "script_sha256": base.digest(__file__), "dependencies": dependencies,
        "runtime": runtime, "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous),
        "cases": prior["cases"], "jobs": jobs, "base_selections": frozen["selections"],
        "expanded_data": expanded["expanded_data"], "target": prior["target"],
        "grid": [{"lambda": penalty, "shrinkage": shrink} for penalty in (1., 10., 100., 1000.) for shrink in (.25, 1.)],
        "solvers": {"binary": {"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9, "gradient_gate": 1e-6},
                    "regression": {"atol": 1e-8, "btol": 1e-8, "iter_lim": 2000}},
        "resources": {"workers": 1, "threads_per_fit": 1, "cpus": [4], "maximum_job_seconds": 1200,
                      "maximum_job_rss_bytes": 8 * 1024**3, "maximum_named_array_bytes": 4 * 1024**3},
        "selection": policy["selection"], "limitations": policy["limitations"], "conditional_seed": 569,
        "official_test_scored": False}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "new_train_anchors": 24,
           "selection_solves": 96, "nonzero_candidates": 192, "new_clean_union_fits": 24, "carried_models": 12})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["runtime"] != base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy") or args.previous != Path(plan["previous"])):
        raise ValueError("Frozen implementation/runtime/parent changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def selection_job(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    job = next(j for j in plan["jobs"] if (j["dataset"], j["library"]) == (args.dataset, args.library))
    case = next(c for c in plan["cases"] if c["dataset_name"] == args.dataset)
    if case["problem_type"] not in ("binary", "regression"):
        raise ValueError("Only scalar jobs may enter selection")
    directory = args.output / "selection_fits" / args.dataset / args.library
    path = directory / "job.json"
    if directory.exists():
        raise ValueError("Preserve a started job")
    record = {"dataset": args.dataset, "library": args.library, "status": "failed", "started_at": base.now()}
    base.write(path, record)
    try:
        X, labels, schema, train, stop, receipt = fitting_data(job, case, train_only=True)
        payload, fit = union.fit_fixed(clean_job(job), case, X, labels, schema, 47)
        if job.get("temperature") is not None:
            raise ValueError("Scalar identity must not contain temperature")
        stop_frame = base.normalize(stop["X"], schema)[0]
        anchor = cv.threshold.persist(payload, args.artifacts / "anchors" / args.dataset / (args.library + ".pkl"), stop["X"])
        base.write(directory / "anchor.json", union.refit.plain_receipt({"fit": fit, "model": anchor,
            "fitting_data": receipt, "temperature": job.get("temperature"), "status": "ok", **clean_provenance(job)}))
        identity_prediction = predict(payload, stop["X"])
        identity_error = base.error(identity_prediction, stop["y"], case["problem_type"])
        np.savez_compressed(directory / "identity.npz", predictions=identity_prediction, labels=stop["y"], row_ids=stop["outer_train_positions"])
        h = helper(case["problem_type"])
        memory = checker().memory_preflight(payload, len(X), max_working_bytes=plan["resources"]["maximum_named_array_bytes"], solver_kind=case["problem_type"])
        base.write(directory / "memory.json", memory)
        design = h.prepare_design(payload, X, labels)
        chosen = {"dataset": args.dataset, "library": args.library, "correction": None,
            "stop_error": identity_error, "anchor": anchor, "fitting_data": receipt, "fit": fit}
        best_key, cached_penalty, coefficients, solve, failed_solve = selection_key(identity_error, None), None, None, None, None
        candidates = []
        for index, controls in enumerate(plan["grid"]):
            item = {"index": index, "controls": controls, "status": "failed", "started_at": base.now()}
            try:
                if cached_penalty != controls["lambda"]:
                    cached_penalty, failed_solve = controls["lambda"], None
                    try:
                        coefficients, solve = solve_coefficients(case["problem_type"], design, labels, cached_penalty, plan["solvers"][case["problem_type"]])
                    except Exception:
                        failed_solve = traceback.format_exc()
                if failed_solve:
                    raise ValueError(failed_solve)
                candidate, evidence = fold_candidate(payload, design, coefficients, controls["shrinkage"],
                                                       {"fitting": X, "stop": stop_frame})
                prediction = predict(candidate, stop["X"])
                restored = pickle.loads(pickle.dumps(candidate, protocol=5))
                np.testing.assert_array_equal(predict(restored, stop["X"]), prediction)
                score = base.error(prediction, stop["y"], case["problem_type"])
                pred_path = directory / (f"candidate_{index}.npz")
                np.savez_compressed(pred_path, predictions=prediction, labels=stop["y"], row_ids=stop["outer_train_positions"])
                item.update(status="ok", stop_error=score, solve=solve, evidence=evidence, predictions_sha256=base.digest(pred_path))
                key = selection_key(score, controls)
                if key < best_key:
                    best_key = key
                    chosen = {**chosen, "correction": {**controls, "solver": plan["solvers"][case["problem_type"]],
                        "max_working_bytes": plan["resources"]["maximum_named_array_bytes"]}, "stop_error": score}
            except Exception:
                item["error"] = traceback.format_exc()
            item["finished_at"] = base.now()
            candidates.append(item)
            base.write(directory / f"candidate_{index}.json", union.refit.plain_receipt(item))
        record.update(status="ok" if len(candidates) == 8 and all(c["status"] == "ok" for c in candidates) else "failed",
            choice=chosen, candidates=candidates, memory=memory, anchor=anchor, fitting_data=receipt, fit=fit)
    except Exception:
        record["error"] = traceback.format_exc()
    record["finished_at"] = base.now()
    base.write(path, union.refit.plain_receipt(record))


def dispatch(args, stage, jobs, destination):
    plan = checked(args)
    logdir = args.output / (stage + "_logs")
    if logdir.exists():
        raise ValueError("Preserve a started stage")
    logdir.mkdir()
    records = []
    for job in jobs:
        while (args.artifacts / "pause_fits").exists():
            time.sleep(.5)
        command = [sys.executable, "-I", str(Path(__file__).resolve()), stage]
        for name in ("output", "artifacts", "package_root", "dependencies", "previous"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        command += ["--dataset", job["dataset"], "--library", job["library"]]
        name = job["dataset"] + "-" + job["library"]
        with (logdir / (name + ".txt")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            resource = cv.watchdog.watch(process, maximum_seconds=plan["resources"]["maximum_job_seconds"],
                maximum_rss_bytes=plan["resources"]["maximum_job_rss_bytes"])
        base.write(logdir / (name + "-resource.json"), resource)
        if not resource["cleanup_complete"]:
            raise RuntimeError("Owned worker cleanup failed; stop further dispatch")
        path = args.output / destination / job["dataset"] / job["library"] / "job.json"
        record = base.read(path) if path.exists() else {"dataset": job["dataset"], "library": job["library"], "status": "failed", "error": "not_completed"}
        if resource["exit_code"] != 0 or resource["resource_failure"] is not None:
            record.update(status="failed", resource=resource)
        base.write(path, record)
        records.append(record)
        print({"stage": stage, "dataset": job["dataset"], "library": job["library"], "status": record["status"]}, flush=True)
    return records


def active_jobs(plan):
    tasks = {case["dataset_name"] for case in plan["cases"] if case["problem_type"] in ("binary", "regression")}
    jobs = [job for job in plan["jobs"] if job["dataset"] in tasks]
    expected = {(task, library) for task in tasks for library in base.LIBRARIES}
    if len(tasks) != 8 or len(jobs) != 24 or {(j["dataset"], j["library"]) for j in jobs} != expected:
        raise ValueError("Expected twenty-four unique scalar jobs")
    return jobs


def select(args):
    plan = checked(args)
    records = dispatch(args, "selection_job", active_jobs(plan), "selection_fits")
    success = len(records) == 24 and all(r["status"] == "ok" for r in records)
    choices = [r["choice"] for r in records if r["status"] == "ok"]
    base.write(args.output / "scalar_controls.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "all_fits_successful": success, "choices": choices,
        "new_anchor_fits_planned": 24,
        "new_anchor_fits_completed": sum((args.output / "selection_fits" / j["dataset"] / j["library"] / "anchor.json").exists() for j in active_jobs(plan)),
        "selection_solves_planned": 96, "nonidentity_candidate_evaluations_planned": 192,
        "identity_evaluations_planned": 24,
        "transfer_started": False, "development_scored": False})


def controls_document(args, plan):
    document = base.read(args.output / "scalar_controls.json")
    expected = {(j["dataset"], j["library"]) for j in active_jobs(plan)}
    if (not document["all_fits_successful"] or document["plan_sha256"] != base.digest(args.output / "plan.json")
            or len(document["choices"]) != 24 or {(c["dataset"], c["library"]) for c in document["choices"]} != expected):
        raise ValueError("All twenty-four choices and192 candidates must succeed before transfer")
    return document


def transfer_job(args):
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    controls = controls_document(args, plan)
    choice = next(c for c in controls["choices"] if (c["dataset"], c["library"]) == (args.dataset, args.library))
    original = next(c for c in plan["base_selections"] if (c["dataset"], c["library"]) == (args.dataset, args.library))
    job = next(c for c in plan["jobs"] if (c["dataset"], c["library"]) == (args.dataset, args.library))
    case = next(c for c in plan["cases"] if c["dataset_name"] == args.dataset)
    path = args.output / "transfer_fits" / args.dataset / args.library / "job.json"
    if path.exists():
        raise ValueError("Preserve a started transfer")
    record = {"dataset": args.dataset, "library": args.library, "status": "failed", "started_at": base.now()}
    base.write(path, record)
    try:
        # Always fit a fresh clean union model, including identity choices.
        payload, details, verification_frame = parent.fit_policy(clean_job(job), case, 47)
        clean_fit = {"fit": details, "status": "native_fit_completed", **clean_provenance(job)}
        base.write(path.parent / "clean_fit.json", union.refit.plain_receipt(clean_fit))
        if base.digest(original["model_path"]) != original["model_sha256"]:
            raise ValueError("Parent comparison model changed")
        with Path(original["model_path"]).open("rb") as stream:
            original_payload = pickle.load(stream)
        topology = checker().verify_clean_refit(payload, original_payload)
        new_job = refreshed_job(job, case["problem_type"], choice["correction"])
        clean_fit.update(topology=topology, status="ok")
        base.write(path.parent / "clean_fit.json", union.refit.plain_receipt(clean_fit))
        if choice["correction"] is not None:
            X, labels, schema, train, stop, receipt = fitting_data(job, case, train_only=False)
            if schema != payload["schema"]:
                raise ValueError("Clean fitting vocabulary differs from exact reconstruction")
            verification = {name: base.normalize(role["X"], schema)[0]
                for name, role in (("original_train", train), ("original_stop", stop))}
            payload, correction = corrected(payload, X, labels, choice["correction"], verification)
            details.update(scalar_refresh=correction, scalar_fitting_data=receipt)
        saved = cv.threshold.persist(payload, args.artifacts / "models" / args.dataset / (args.library + ".pkl"), verification_frame)
        selection = {"dataset": args.dataset, "library": args.library, "source_wave": 17, "seed": 47,
            "base_selection": original, "independent_stop_selection": choice, "fixed_job": new_job,
            "requested_rounds": job["rounds"], "best": {"parameters": job["parameters"], "retained_rounds": details["retained_rounds"]},
            "leaf_refit": new_job["leaf_refit"], "binary_refit": new_job["binary_refit"], "temperature": None,
            "multiclass_refit": None, "clean_refit_check": topology, "refit_details": details, **saved}
        record.update(status="ok", selection=selection)
    except Exception:
        record["error"] = traceback.format_exc()
    record["finished_at"] = base.now()
    base.write(path, union.refit.plain_receipt(record))


def transfer(args):
    plan = checked(args)
    controls = controls_document(args, plan)
    controls_hash = base.digest(args.output / "scalar_controls.json")
    records = dispatch(args, "transfer_job", active_jobs(plan), "transfer_fits")
    if controls_hash != base.digest(args.output / "scalar_controls.json"):
        raise ValueError("Selected controls changed during fixed transfer")
    active = {(j["dataset"], j["library"]) for j in active_jobs(plan)}
    carried = [s for s in plan["base_selections"] if (s["dataset"], s["library"]) not in active]
    selections = carried + [r["selection"] for r in records if r["status"] == "ok"]
    choices = {(c["dataset"], c["library"]): c for c in controls["choices"]}
    problems = {c["dataset_name"]: c["problem_type"] for c in plan["cases"]}
    jobs = [refreshed_job(job, problems[job["dataset"]], choices[job["dataset"], job["library"]]["correction"])
        if (job["dataset"], job["library"]) in choices else job for job in plan["jobs"]]
    expected = {(c["dataset_name"], library) for c in plan["cases"] for library in base.LIBRARIES}
    success = (len(carried) == 12 and len(selections) == 36 and len(jobs) == 36
        and {(s["dataset"], s["library"]) for s in selections} == expected and all(r["status"] == "ok" for r in records))
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "scalar_controls_sha256": controls_hash, "created_at": base.now(), "jobs": jobs,
        "selections": selections, "all_fits_successful": success, "new_tree_fits_planned": 48,
        "new_anchor_fits_completed": controls["new_anchor_fits_completed"],
        "new_clean_union_fits_completed": sum((args.output / "transfer_fits" / j["dataset"] / j["library"] / "clean_fit.json").exists() for j in active_jobs(plan)),
        "carried_models": 12,
        "official_test_scored": False})


def development(args):
    plan = checked(args)
    controls_document(args, plan)
    frozen = base.read(args.output / "frozen_selections.json")
    if not frozen["all_fits_successful"] or frozen["scalar_controls_sha256"] != base.digest(args.output / "scalar_controls.json"):
        raise ValueError("All mandatory transfers and carries must succeed")
    base.checked, base.predict = checked, predict
    base.development(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "select", "selection_job", "transfer", "transfer_job", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--library", choices=base.LIBRARIES)
    arguments = parser.parse_args()
    globals()[arguments.stage](arguments)
