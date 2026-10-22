"""Retry scalar corrections using verified clean TRAIN anchors and fresh union models."""
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
ATTEMPT_PATH = Path(__file__).with_name("fair_accuracy_scalar_refresh.py")
NUMERICAL_PATH = Path(__file__).with_name("ridge_numerical_diagnostic_extended.py")
DIAGNOSTIC = Path(__file__).parent / "results/improvements_20261016_followup/ridge_numerical_diagnostic_extended"
HELPER_PATH = Path(__file__).with_name("fixed_clean_refit_check.py")
REGRESSION_PATH = Path(__file__).with_name("fixed_leaf_refit.py")
BINARY_PATH = Path(__file__).with_name("fixed_binary_leaf_refit.py")
SEED_PATH = Path(__file__).with_name("fair_accuracy_scalar_numerical_seed_check.py")
OUTER_PATH = Path(__file__).with_name("fair_accuracy_scalar_numerical_outer_test.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_scalar_numerical.json"


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
        import numpy as np
        numerical = load("certified_numerical_ridge", NUMERICAL_PATH)
        delta, receipt = numerical.solve(design["Z"], np.asarray(labels, dtype=np.float64) - design["baseline"],
            penalty, solver["variant"], iterations=solver["iterations"])
        if not receipt["success"]:
            raise ValueError(f"Frozen ridge certificate/status failed: {receipt}")
        return delta, receipt
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
    if problem == "regression":
        extra = 256 * (len(X) + memory["parameters_upper_bound"])
        memory["extra_certificate_vector_bytes"] = extra
        if memory["refit_array_budget_bytes"] + extra > controls["max_working_bytes"]:
            raise MemoryError("Frozen regression certificate reserve exceeded")
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

def verify_binary_reuse(record, anchor, choice, directory, grid, dataset, library):
    candidates = record["candidates"]
    if (record["status"] != "ok" or len(candidates) != 8
            or {c["index"] for c in candidates} != set(range(8))
            or any(c["status"] != "ok" for c in candidates)):
        raise ValueError("Every reused binary grid must contain all eight successful indices")
    if record["choice"] != choice or record["anchor"] != anchor["model"]:
        raise ValueError("Reused binary choice or anchor differs from frozen evidence")
    for candidate in candidates:
        index = candidate["index"]
        if (candidate["controls"] != grid[index]
                or candidate.get("dataset", dataset) != dataset
                or candidate.get("library", library) != library
                or base.read(directory / f"candidate_{index}.json") != candidate
                or base.digest(directory / f"candidate_{index}.npz") != candidate["predictions_sha256"]):
            raise ValueError("Reused binary candidate identity, controls or evidence changed")
    return {str(path): base.digest(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve earlier attempts")
    attempt = load("failed_scalar_attempt", ATTEMPT_PATH)
    prior = base.read(args.previous / "plan.json")
    failed = base.read(args.previous / "scalar_controls.json")
    failure_summary = base.read(args.previous / "summary.json")
    if (prior["wave"] != 17 or failed["all_fits_successful"] or prior["runtime"] != runtime
            or failed["plan_sha256"] != base.digest(args.previous / "plan.json")
            or failure_summary["source_sha256"]["scalar_controls.json"] != base.digest(args.previous / "scalar_controls.json")):
        raise ValueError("Expected unchanged failed wave17")
    previous_args = copy.copy(args)
    previous_args.output, previous_args.previous = args.previous, Path(prior["previous"])
    attempt.checked(previous_args)
    diagnostic = base.read(DIAGNOSTIC / "plan.json")
    result = base.read(DIAGNOSTIC / "numerics/receipt.json")
    resource = base.read(DIAGNOSTIC / "resource.json")
    if (diagnostic["script_sha256"] != base.digest(NUMERICAL_PATH) or diagnostic["runtime"] != runtime
            or result["status"] != "completed" or len(result["variants"]) != 1
            or not result["variants"][0]["success"] or resource["exit_code"] != 0
            or resource["resource_failure"] is not None or not resource["cleanup_complete"]):
        raise ValueError("The prospectively fixed numerical diagnostic must certify before retry")
    dependencies = dict(prior["dependencies"])
    for path, expected in diagnostic["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Numerical diagnostic dependency changed")
        dependencies[path] = expected
    for path in (ATTEMPT_PATH, NUMERICAL_PATH, SEED_PATH, OUTER_PATH, POLICY,
                 args.previous / "plan.json", args.previous / "scalar_controls.json", args.previous / "summary.json"):
        dependencies[str(path)] = base.digest(path)
    for path in sorted(DIAGNOSTIC.rglob("*")):
        if path.is_file():
            dependencies[str(path)] = base.digest(path)
    problems = {case["dataset_name"]: case["problem_type"] for case in prior["cases"]}
    anchors, reused_binary = [], []
    for job in active_jobs(prior):
        directory = args.previous / "selection_fits" / job["dataset"] / job["library"]
        anchor_path, job_path, identity_path = directory / "anchor.json", directory / "job.json", directory / "identity.npz"
        anchor, record = base.read(anchor_path), base.read(job_path)
        if (anchor["status"] != "ok" or not anchor["old_scalar_corrections_disabled_before_fit"]
                or anchor["clean_fit_controls_sha256"] != clean_provenance(job)["clean_fit_controls_sha256"]
                or base.digest(anchor["model"]["model_path"]) != anchor["model"]["model_sha256"]
                or anchor["fit"]["requested_rounds"] != job["rounds"]):
            raise ValueError("Reused clean TRAIN anchor changed")
        item = {"dataset": job["dataset"], "library": job["library"], "anchor": anchor,
                "anchor_path": str(anchor_path), "anchor_sha256": base.digest(anchor_path),
                "identity_path": str(identity_path), "identity_sha256": base.digest(identity_path)}
        anchors.append(item)
        for path in (anchor_path, job_path, identity_path, Path(anchor["model"]["model_path"])):
            dependencies[str(path)] = base.digest(path)
        if problems[job["dataset"]] == "binary":
            old_choice = next(choice for choice in failed["choices"]
                              if (choice["dataset"], choice["library"]) == (job["dataset"], job["library"]))
            files = verify_binary_reuse(record, anchor, old_choice, directory, prior["grid"], job["dataset"], job["library"])
            dependencies.update(files)
            reused_binary.append({"dataset": job["dataset"], "library": job["library"],
                                  "job_path": str(job_path), "job_sha256": base.digest(job_path),
                                  "choice": record["choice"], "artifact_sha256": files})
    if len(anchors) != 24 or len(reused_binary) != 12:
        raise ValueError("Expected 24 anchors and 12 complete binary grids")
    policy = base.read(POLICY)
    plan = {**prior, "protocol": "certified_numerical_scalar_refresh_retry_20261016", "wave": "17b",
        "created_at": base.now(), "script_sha256": base.digest(__file__), "dependencies": dependencies,
        "previous": str(args.previous), "latest_complete_policy": prior["previous"],
        "source_anchors": anchors, "reused_binary": reused_binary,
        "solvers": {"binary": prior["solvers"]["binary"],
                    "regression": {"variant": "unscaled_lsqr_10000", "iterations": 10000}},
        "selection": policy["selection"], "limitations": policy["limitations"],
        "diagnostic_plan_sha256": base.digest(DIAGNOSTIC / "plan.json"),
        "diagnostic_receipt_sha256": base.digest(DIAGNOSTIC / "numerics/receipt.json"),
        "work": {"reused_anchors": 24, "reused_binary_solves": 48, "new_regression_solves": 48,
                 "new_nonidentity_candidates": 96, "reused_nonidentity_candidates": 96,
                 "new_train_tree_fits": 0, "mandatory_new_union_tree_fits": 24},
        "resource_scope": "One regression penalty solve plus two native shrinkage checks per 1200-second owned worker; union and seed workers retain the same limit"}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "new_penalty_jobs": 48, "reused_binary_grids": 12})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["runtime"] != base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy") or args.previous != Path(plan["previous"])):
        raise ValueError("Frozen implementation/runtime/parent changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def penalty_job(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    job = next(j for j in plan["jobs"] if (j["dataset"], j["library"]) == (args.dataset, args.library))
    case = next(c for c in plan["cases"] if c["dataset_name"] == args.dataset)
    if case["problem_type"] != "regression" or args.penalty_index not in range(4):
        raise ValueError("Expected a fixed regression penalty")
    source = next(s for s in plan["source_anchors"] if (s["dataset"], s["library"]) == (args.dataset, args.library))
    directory = args.output / "penalty_fits" / args.dataset / args.library / str(args.penalty_index)
    path = directory / "job.json"
    if directory.exists():
        raise ValueError("Preserve started penalty job")
    record = {"dataset": args.dataset, "library": args.library, "penalty_index": args.penalty_index,
              "status": "failed", "started_at": base.now(), "new_native_tree_fits": 0}
    base.write(path, record)
    try:
        X, labels, schema, train, stop, receipt = fitting_data(job, case, train_only=True)
        anchor = source["anchor"]
        if receipt != anchor["fitting_data"] or schema != anchor["fitting_data"]["schema"]:
            raise ValueError("Reused anchor fitting rows or vocabulary differ")
        with Path(anchor["model"]["model_path"]).open("rb") as stream:
            payload = pickle.load(stream)
        identity_prediction = predict(payload, stop["X"])
        with np.load(source["identity_path"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(identity_prediction, saved["predictions"])
            np.testing.assert_array_equal(stop["y"], saved["labels"])
            np.testing.assert_array_equal(stop["outer_train_positions"], saved["row_ids"])
        identity_error = base.error(identity_prediction, stop["y"], "regression")
        memory = checker().memory_preflight(payload, len(X), max_working_bytes=plan["resources"]["maximum_named_array_bytes"], solver_kind="regression")
        extra = 256 * (len(X) + memory["parameters_upper_bound"])
        memory["extra_certificate_vector_bytes"] = extra
        if memory["refit_array_budget_bytes"] + extra > plan["resources"]["maximum_named_array_bytes"]:
            raise MemoryError("Frozen certificate vector reserve exceeded")
        base.write(directory / "memory.json", memory)
        design = helper("regression").prepare_design(payload, X, labels)
        grid = plan["grid"][2 * args.penalty_index:2 * args.penalty_index + 2]
        coefficients, solve = solve_coefficients("regression", design, labels, grid[0]["lambda"], plan["solvers"]["regression"])
        base.write(directory / "solve.json", union.refit.plain_receipt(solve))
        candidates = []
        for local_index, controls in enumerate(grid):
            index = 2 * args.penalty_index + local_index
            item = {"index": index, "controls": controls, "status": "failed"}
            try:
                candidate, evidence = fold_candidate(payload, design, coefficients, controls["shrinkage"],
                    {"fitting": X, "stop": base.normalize(stop["X"], schema)[0]})
                prediction = predict(candidate, stop["X"])
                np.testing.assert_array_equal(prediction, predict(pickle.loads(pickle.dumps(candidate, protocol=5)), stop["X"]))
                pred_path = directory / f"candidate_{index}.npz"
                np.savez_compressed(pred_path, predictions=prediction, labels=stop["y"], row_ids=stop["outer_train_positions"])
                item.update(status="ok", stop_error=base.error(prediction, stop["y"], "regression"), evidence=evidence,
                            predictions_sha256=base.digest(pred_path))
            except Exception:
                item["error"] = traceback.format_exc()
            candidates.append(item)
            base.write(directory / f"candidate_{index}.json", union.refit.plain_receipt(item))
        record.update(status="ok" if all(c["status"] == "ok" for c in candidates) else "failed",
            candidates=candidates, solve=solve, memory=memory, identity_error=identity_error,
            fitting_data=receipt, anchor=anchor["model"], fit=anchor["fit"], reused_anchor=source)
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
    problems = {c["dataset_name"]: c["problem_type"] for c in plan["cases"]}
    regression = [job for job in active_jobs(plan) if problems[job["dataset"]] == "regression"]
    logdir = args.output / "penalty_logs"
    if logdir.exists():
        raise ValueError("Preserve started penalty stage")
    logdir.mkdir()
    records = []
    for job in regression:
        for index in range(4):
            while (args.artifacts / "pause_fits").exists():
                time.sleep(.5)
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "penalty_job"]
            for name in ("output", "artifacts", "package_root", "dependencies", "previous"):
                command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
            command += ["--dataset", job["dataset"], "--library", job["library"], "--penalty-index", str(index)]
            name = job["dataset"] + "-" + job["library"] + "-" + str(index)
            with (logdir / (name + ".txt")).open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                resource = cv.watchdog.watch(process, maximum_seconds=plan["resources"]["maximum_job_seconds"],
                    maximum_rss_bytes=plan["resources"]["maximum_job_rss_bytes"])
            base.write(logdir / (name + "-resource.json"), resource)
            if not resource["cleanup_complete"]:
                raise RuntimeError("Unsafe worker cleanup; stop dispatch")
            path = args.output / "penalty_fits" / job["dataset"] / job["library"] / str(index) / "job.json"
            record = base.read(path) if path.exists() else {"dataset": job["dataset"], "library": job["library"], "penalty_index": index, "status": "failed"}
            if resource["exit_code"] != 0 or resource["resource_failure"] is not None:
                record.update(status="failed", resource=resource)
            base.write(path, record)
            records.append(record)
            print({"dataset": job["dataset"], "library": job["library"], "penalty_index": index, "status": record["status"]}, flush=True)
    choices = [copy.deepcopy(item["choice"]) for item in plan["reused_binary"]]
    for job in regression:
        rows = [r for r in records if (r["dataset"], r["library"]) == (job["dataset"], job["library"])]
        if len(rows) != 4 or {r["penalty_index"] for r in rows} != set(range(4)) or any(r["status"] != "ok" for r in rows):
            continue
        if len({r["identity_error"] for r in rows}) != 1:
            raise ValueError("Repeated frozen identity differs")
        first = rows[0]
        choice = {"dataset": job["dataset"], "library": job["library"], "correction": None,
            "stop_error": first["identity_error"], "anchor": first["anchor"], "fit": first["fit"], "fitting_data": first["fitting_data"]}
        best = selection_key(choice["stop_error"], None)
        for item in [c for row in rows for c in row["candidates"]]:
            key = selection_key(item["stop_error"], item["controls"])
            if key < best:
                best = key
                choice.update(stop_error=item["stop_error"], correction={**item["controls"],
                    "solver": plan["solvers"]["regression"], "max_working_bytes": plan["resources"]["maximum_named_array_bytes"]})
        choices.append(choice)
    expected = {(j["dataset"], j["library"], i) for j in regression for i in range(4)}
    success = (len(regression) == 12 and len(records) == 48 and len(choices) == 24
        and {(r["dataset"], r["library"], r["penalty_index"]) for r in records} == expected
        and all(r["status"] == "ok" for r in records))
    base.write(args.output / "scalar_controls.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "all_fits_successful": success, "choices": choices,
        "new_anchor_fits_completed": 0, "reused_anchor_fits": 24, "reused_binary_grids": 12,
        "new_regression_solves_planned": 48, "new_nonidentity_evaluations_planned": 96,
        "reused_nonidentity_evaluations": 96, "unique_identity_candidates": 24,
        "regression_identity_prediction_checks": sum("identity_error" in r for r in records),
        "transfer_started": False, "development_scored": False})


def controls_document(args, plan):
    document = base.read(args.output / "scalar_controls.json")
    expected = {(j["dataset"], j["library"]) for j in active_jobs(plan)}
    if (not document["all_fits_successful"] or document["plan_sha256"] != base.digest(args.output / "plan.json")
            or len(document["choices"]) != 24 or {(c["dataset"], c["library"]) for c in document["choices"]} != expected):
        raise ValueError("All twenty-four choices and 192 candidates must succeed before transfer")
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
        selection = {"dataset": args.dataset, "library": args.library, "source_wave": "17b", "seed": 47,
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
        "selections": selections, "all_fits_successful": success, "new_tree_fits_planned": 24, "reused_anchor_fits": 24,
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
    parser.add_argument("stage", choices=("prepare", "select", "penalty_job", "transfer", "transfer_job", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--penalty-index", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--library", choices=base.LIBRARIES)
    arguments = parser.parse_args()
    globals()[arguments.stage](arguments)
