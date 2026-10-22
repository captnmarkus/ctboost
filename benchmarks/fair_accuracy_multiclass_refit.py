"""TRAIN-only multiclass leaf correction, independent STOP choice, fixed transfer."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

PARENT_PATH = Path(__file__).with_name("fair_accuracy_houses_cap_v2.py")
HELPER_PATH = Path(__file__).with_name("fixed_multiclass_leaf_refit.py")
RECOVERY_PATH = Path(__file__).with_name("fair_accuracy_houses_report_recovery.py")
SEED_PATH = Path(__file__).with_name("fair_accuracy_multiclass_seed_check.py")
OUTER_PATH = Path(__file__).with_name("fair_accuracy_multiclass_outer_test.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_multiclass.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


parent = load("multiclass_parent", PARENT_PATH)
base, union, cv, predict = parent.base, parent.union, parent.cv, parent.predict
EXPAND_DATASETS = parent.EXPAND_DATASETS


def helper():
    return load("multiclass_leaf_helper", HELPER_PATH)


def selection_key(score, controls):
    return (score, 0, 0., 0.) if controls is None else (score, 1, -controls["lambda"], controls["shrinkage"])


def fitting_data(job, case, *, train_only):
    """Exact raw fitting order; SDSS anchor omits the original STOP block."""
    import numpy as np
    import pandas as pd
    train, stop = (base.load_role(case, name) for name in ("train", "stop"))
    nt, ns = len(train["y"]), len(stop["y"])
    ti, si = train["outer_train_positions"], stop["outer_train_positions"]
    if np.intersect1d(ti, si).size:
        raise ValueError("TRAIN and STOP overlap")
    source = job.get("expanded_source")
    if source:
        if source["dataset"] != case["dataset_name"] or base.digest(source["path"]) != source["sha256"]:
            raise ValueError("Expanded fitting source changed")
        with Path(source["path"]).open("rb") as stream:
            expanded = pickle.load(stream)
        if expanded["receipt"] != source["receipt"]:
            raise ValueError("Expanded source receipt changed")
        raw, labels, ids = expanded["X"], expanded["y"], expanded["outer_train_positions"]
        if not np.array_equal(ids[:nt], ti) or not np.array_equal(ids[nt:nt + ns], si):
            raise ValueError("Expanded source has a different TRAIN/STOP prefix")
        offset = 0
        for role in (train, stop):
            count = len(role["y"])
            pd.testing.assert_frame_equal(raw.iloc[offset:offset + count].reset_index(drop=True),
                role["X"].reset_index(drop=True), check_dtype=False, check_categorical=False, check_exact=True)
            np.testing.assert_array_equal(labels[offset:offset + count], role["y"])
            offset += count
        positions = np.r_[np.arange(nt), np.arange(nt + ns, len(ids))] if train_only else np.arange(len(ids))
        raw, labels, ids = raw.iloc[positions].reset_index(drop=True), labels[positions], ids[positions]
    else:
        raw = train["X"].reset_index(drop=True) if train_only else pd.concat([train["X"], stop["X"]], ignore_index=True)
        labels = train["y"] if train_only else np.concatenate([train["y"], stop["y"]])
        ids = ti if train_only else np.concatenate([ti, si])
    if len(np.unique(ids)) != len(ids) or (train_only and np.intersect1d(ids, si).size):
        raise ValueError("Fitting row identity/independent STOP violation")
    if case["problem_type"] == "multiclass":
        expected = np.arange(case["num_classes"])
        if not np.array_equal(np.unique(labels), expected) or not np.array_equal(np.unique(stop["y"]), expected):
            raise ValueError("Every fitting and STOP role must retain the common class order")
    X, schema = base.normalize(raw)
    receipt = {"rows": len(labels), "original_train_rows": nt, "original_stop_rows": ns,
        "stop_excluded": train_only, "outside_cap_rows": len(labels) - nt - (0 if train_only else ns),
        "row_ids_sha256": base.hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "source_sha256": source["sha256"] if source else None,
        "row_order": "TRAIN then ascending outside-cap" if train_only else "TRAIN, STOP, ascending outside-cap",
        "schema": schema, "vocabulary_fit_on": "actual fitting rows only"}
    return X, labels, schema, train, stop, receipt


def fold_candidate(payload, design, coefficients, shrinkage, frames):
    import numpy as np
    h = helper()
    updates = h.corrections_by_tree(design, coefficients, shrinkage)
    candidate, fold = h.fold_corrections(payload, updates)
    fold = union.refit.plain_receipt(fold)
    checks = {name: h.verify_fold(payload, candidate, frame, updates, fold,
              design["training_prepared"] if name == "fitting" else None) for name, frame in frames.items()}
    expected = design["baseline"].copy()
    for index, matrix in enumerate(design["matrices"]):
        left, right = design["offsets"][index:index + 2]
        expected[:, index] += shrinkage * (matrix @ coefficients[left:right])
    np.testing.assert_allclose(h.predict_raw(candidate, frames["fitting"], design["training_prepared"]),
                               expected, rtol=h.FIXED_RTOL, atol=h.FIXED_ATOL)
    return candidate, union.refit.plain_receipt({"fold": fold, "checks": checks,
        "independent_sparse_raw_margin_oracle": True, "preprocessing": design["preprocessing_receipt"]})


def corrected(payload, X, labels, controls, verification_frames):
    h = helper()
    memory = h.memory_preflight(payload, len(X), max_working_bytes=controls["max_working_bytes"])
    design = h.prepare_design(payload, X, labels, max_working_bytes=controls["max_working_bytes"])
    coefficients, solve = h.solve_coefficients(design, labels, controls["lambda"], **controls["solver"])
    candidate, checks = fold_candidate(payload, design, coefficients, controls["shrinkage"],
                                       {"fitting": X, **verification_frames})
    return candidate, union.refit.plain_receipt({"memory": memory, "solve": solve, **checks})


def fit_policy(job, case, seed):
    """Fixed fresh-seed policy: preserve all four expanded fitting sources."""
    payload, details, frame = parent.fit_policy(job, case, seed)
    if job.get("multiclass_refit") is not None:
        X, labels, schema, train, stop, receipt = fitting_data(job, case, train_only=False)
        if schema != payload["schema"]:
            raise ValueError("Fresh model and correction fitting vocabulary differ")
        verification = {name: base.normalize(role["X"], schema)[0]
                        for name, role in (("original_train", train), ("original_stop", stop))}
        payload, correction = corrected(payload, X, labels, job["multiclass_refit"], verification)
        details.update(multiclass_refit=correction, multiclass_fitting_data=receipt)
    return payload, details, frame


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve all earlier attempts")
    prior = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (prior["wave"] != 15 or prior["seed"] != 47 or prior["runtime"] != runtime
            or prior["script_sha256"] != base.digest(PARENT_PATH) or not frozen["all_fits_successful"]
            or development["development_target_met"] or prior["scipy_version"] != importlib.metadata.version("scipy")
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")
            or development["report_recovery"]["script_sha256"] != base.digest(RECOVERY_PATH)
            or not development["report_recovery"]["predictions_bit_exact"]):
        raise ValueError("Expected intact failed wave15b policy")
    data_args = copy.copy(args)
    data_args.output = args.previous
    expanded = parent.prepared(data_args, prior)
    if frozen["jobs"] != expanded["jobs"] or frozen["prepared_data_sha256"] != base.digest(args.previous / "prepared_data.json"):
        raise ValueError("Parent fitting sources changed")
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
    for path in (PARENT_PATH, HELPER_PATH, RECOVERY_PATH, SEED_PATH, OUTER_PATH, POLICY):
        dependencies[str(path)] = base.digest(path)
    for name in ("plan.json", "frozen_selections.json", "development.json", "prepared_data.json", "acquired_source.json",
                 "authorized_rows.json", "report_recovery_started.json", "report_recovery_completed.json", "development_report_failed.json"):
        dependencies[str(args.previous / name)] = base.digest(args.previous / name)
    for selected in frozen["selections"]:
        if base.digest(selected["model_path"]) != selected["model_sha256"]:
            raise ValueError("Parent model changed")
        dependencies[selected["model_path"]] = selected["model_sha256"]
    for source in expanded["expanded_data"]:
        dependencies[source["path"]] = source["sha256"]
    jobs = []
    for job in frozen["jobs"]:
        if job.get("multiclass_refit") is not None:
            raise ValueError("This study starts from an uncorrected multiclass policy")
        params = job["parameters"]
        for key in ("sample_weight", "class_weight", "class_weights", "baseline", "base_margin",
                    "monotone_constraints", "interaction_constraints"):
            if params.get(key) not in (None, "", [], (), {}):
                raise ValueError("Weighted/constrained/external baseline models unsupported")
        if params.get("max_leaf_weight", 0) or params.get("max_delta_step", 0):
            raise ValueError("Constrained leaf values unsupported")
        jobs.append({**job, "multiclass_refit": None})
    policy = base.read(POLICY)
    plan = {"protocol": "multiclass_leaf_stop_selection_fixed_transfer_20261016", "wave": 16, "seed": 47,
        "created_at": base.now(), "script_sha256": base.digest(__file__), "dependencies": dependencies,
        "runtime": runtime, "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous),
        "cases": prior["cases"], "jobs": jobs, "base_selections": frozen["selections"],
        "expanded_data": expanded["expanded_data"], "target": prior["target"],
        "grid": [{"lambda": penalty, "shrinkage": shrink} for penalty in (1., 10., 100., 1000.) for shrink in (.25, 1.)],
        "solver": {"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9, "gradient_gate": 1e-6},
        "resources": {"workers": 1, "threads_per_fit": 1, "cpus": [4], "maximum_job_seconds": 1200,
                      "maximum_job_rss_bytes": 8 * 1024**3, "maximum_named_array_bytes": 2 * 1024**3},
        "selection": policy["selection"], "limitations": policy["limitations"], "conditional_seed": 457,
        "official_test_scored": False}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "new_train_anchors": 12,
           "selection_solves": 48, "nonzero_candidates": 96, "union_transfers": 12, "carried_models": 24})


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
    if case["problem_type"] != "multiclass":
        raise ValueError("Only multiclass jobs may enter selection")
    directory = args.output / "selection_fits" / args.dataset / args.library
    path = directory / "job.json"
    if directory.exists():
        raise ValueError("Preserve a started job")
    record = {"dataset": args.dataset, "library": args.library, "status": "failed", "started_at": base.now()}
    base.write(path, record)
    try:
        X, labels, schema, train, stop, receipt = fitting_data(job, case, train_only=True)
        payload, fit = union.fit_fixed(job, case, X, labels, schema, 47)
        if job.get("temperature") is not None:
            payload["temperature"] = job["temperature"]
        stop_frame = base.normalize(stop["X"], schema)[0]
        anchor = cv.threshold.persist(payload, args.artifacts / "anchors" / args.dataset / (args.library + ".pkl"), stop["X"])
        base.write(directory / "anchor.json", union.refit.plain_receipt({"fit": fit, "model": anchor,
            "fitting_data": receipt, "temperature": job.get("temperature"), "status": "ok"}))
        identity_prediction = predict(payload, stop["X"])
        identity_error = base.error(identity_prediction, stop["y"], "multiclass")
        np.savez_compressed(directory / "identity.npz", predictions=identity_prediction, labels=stop["y"], row_ids=stop["outer_train_positions"])
        h = helper()
        memory = h.memory_preflight(payload, len(X), max_working_bytes=plan["resources"]["maximum_named_array_bytes"])
        base.write(directory / "memory.json", memory)
        design = h.prepare_design(payload, X, labels, max_working_bytes=plan["resources"]["maximum_named_array_bytes"])
        chosen = {"dataset": args.dataset, "library": args.library, "multiclass_refit": None,
            "stop_error": identity_error, "anchor": anchor, "fitting_data": receipt, "fit": fit}
        best_key, cached_penalty, coefficients, solve, failed_solve = selection_key(identity_error, None), None, None, None, None
        candidates = []
        for index, controls in enumerate(plan["grid"]):
            item = {"index": index, "controls": controls, "status": "failed", "started_at": base.now()}
            try:
                if cached_penalty != controls["lambda"]:
                    cached_penalty, failed_solve = controls["lambda"], None
                    try:
                        coefficients, solve = h.solve_coefficients(design, labels, cached_penalty, **plan["solver"])
                    except Exception:
                        failed_solve = traceback.format_exc()
                if failed_solve:
                    raise ValueError(failed_solve)
                candidate, evidence = fold_candidate(payload, design, coefficients, controls["shrinkage"],
                                                       {"fitting": X, "stop": stop_frame})
                prediction = predict(candidate, stop["X"])
                restored = pickle.loads(pickle.dumps(candidate, protocol=5))
                np.testing.assert_array_equal(predict(restored, stop["X"]), prediction)
                score = base.error(prediction, stop["y"], "multiclass")
                pred_path = directory / (f"candidate_{index}.npz")
                np.savez_compressed(pred_path, predictions=prediction, labels=stop["y"], row_ids=stop["outer_train_positions"])
                item.update(status="ok", stop_error=score, solve=solve, evidence=evidence, predictions_sha256=base.digest(pred_path))
                key = selection_key(score, controls)
                if key < best_key:
                    best_key = key
                    chosen = {**chosen, "multiclass_refit": {**controls, "solver": plan["solver"],
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
    tasks = {case["dataset_name"] for case in plan["cases"] if case["problem_type"] == "multiclass"}
    jobs = [job for job in plan["jobs"] if job["dataset"] in tasks]
    expected = {(task, library) for task in tasks for library in base.LIBRARIES}
    if len(tasks) != 4 or len(jobs) != 12 or {(j["dataset"], j["library"]) for j in jobs} != expected:
        raise ValueError("Expected twelve unique multiclass jobs")
    return jobs


def select(args):
    plan = checked(args)
    records = dispatch(args, "selection_job", active_jobs(plan), "selection_fits")
    success = len(records) == 12 and all(r["status"] == "ok" for r in records)
    choices = [r["choice"] for r in records if r["status"] == "ok"]
    base.write(args.output / "multiclass_controls.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "all_fits_successful": success, "choices": choices,
        "new_anchor_fits_planned": 12,
        "new_anchor_fits_completed": sum((args.output / "selection_fits" / j["dataset"] / j["library"] / "anchor.json").exists() for j in active_jobs(plan)),
        "selection_solves_planned": 48, "nonidentity_candidate_evaluations_planned": 96,
        "identity_evaluations_planned": 12,
        "transfer_started": False, "development_scored": False})


def controls_document(args, plan):
    document = base.read(args.output / "multiclass_controls.json")
    expected = {(j["dataset"], j["library"]) for j in active_jobs(plan)}
    if (not document["all_fits_successful"] or document["plan_sha256"] != base.digest(args.output / "plan.json")
            or len(document["choices"]) != 12 or {(c["dataset"], c["library"]) for c in document["choices"]} != expected):
        raise ValueError("All twelve choices and96 candidates must succeed before transfer")
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
        selection = {**original, "source_wave": 16, "base_selection": original,
            "multiclass_refit": choice["multiclass_refit"], "independent_stop_selection": choice}
        if choice["multiclass_refit"] is not None:
            if base.digest(original["model_path"]) != original["model_sha256"]:
                raise ValueError("Parent transfer model changed")
            with Path(original["model_path"]).open("rb") as stream:
                payload = pickle.load(stream)
            X, labels, schema, train, stop, receipt = fitting_data(job, case, train_only=False)
            if schema != payload["schema"]:
                raise ValueError("Parent fitting vocabulary differs from exact reconstruction")
            verification = {name: base.normalize(role["X"], schema)[0]
                for name, role in (("original_train", train), ("original_stop", stop))}
            candidate, correction = corrected(payload, X, labels, choice["multiclass_refit"], verification)
            saved = cv.threshold.persist(candidate, args.artifacts / "models" / args.dataset / (args.library + ".pkl"), stop["X"])
            selection.update(saved, fixed_multiclass_refit=correction, multiclass_fitting_data=receipt)
        record.update(status="ok", selection=selection)
    except Exception:
        record["error"] = traceback.format_exc()
    record["finished_at"] = base.now()
    base.write(path, union.refit.plain_receipt(record))


def transfer(args):
    plan = checked(args)
    controls = controls_document(args, plan)
    controls_hash = base.digest(args.output / "multiclass_controls.json")
    records = dispatch(args, "transfer_job", active_jobs(plan), "transfer_fits")
    if controls_hash != base.digest(args.output / "multiclass_controls.json"):
        raise ValueError("Selected controls changed during fixed transfer")
    active = {(j["dataset"], j["library"]) for j in active_jobs(plan)}
    carried = [s for s in plan["base_selections"] if (s["dataset"], s["library"]) not in active]
    selections = carried + [r["selection"] for r in records if r["status"] == "ok"]
    choices = {(c["dataset"], c["library"]): c for c in controls["choices"]}
    jobs = [{**job, "multiclass_refit": choices[job["dataset"], job["library"]]["multiclass_refit"]}
        if (job["dataset"], job["library"]) in choices else job for job in plan["jobs"]]
    expected = {(c["dataset_name"], library) for c in plan["cases"] for library in base.LIBRARIES}
    success = (len(carried) == 24 and len(selections) == 36 and len(jobs) == 36
        and {(s["dataset"], s["library"]) for s in selections} == expected and all(r["status"] == "ok" for r in records))
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "multiclass_controls_sha256": controls_hash, "created_at": base.now(), "jobs": jobs,
        "selections": selections, "all_fits_successful": success, "new_tree_fits": 12, "carried_models": 24,
        "official_test_scored": False})


def development(args):
    plan = checked(args)
    controls_document(args, plan)
    frozen = base.read(args.output / "frozen_selections.json")
    if not frozen["all_fits_successful"] or frozen["multiclass_controls_sha256"] != base.digest(args.output / "multiclass_controls.json"):
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
