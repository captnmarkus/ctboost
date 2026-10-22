"""Five-fold selection of fixed model length, followed by mandatory union refit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time
import traceback

THRESHOLD_PATH = Path(__file__).with_name("fair_accuracy_threshold_scout.py")
SEED_PATH = Path(__file__).with_name("fair_accuracy_round_cv_seed_check.py")
OUTER_PATH = Path(__file__).with_name("fair_accuracy_round_cv_outer_test.py")
WATCHDOG_PATH = Path(__file__).with_name("owned_process_watchdog.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_round_cv.json"
spec = importlib.util.spec_from_file_location("round_cv_threshold", THRESHOLD_PATH)
threshold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(threshold)
base, union = threshold.base, threshold.union
predict, fit_policy = threshold.predict, threshold.fit_policy
watchdog_spec = importlib.util.spec_from_file_location("round_cv_watchdog", WATCHDOG_PATH)
watchdog = importlib.util.module_from_spec(watchdog_spec)
watchdog_spec.loader.exec_module(watchdog)


def index_hash(values):
    import numpy as np
    return base.hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def raw_union(case):
    import numpy as np
    import pandas as pd
    train, stop = base.load_role(case, "train"), base.load_role(case, "stop")
    ids = np.concatenate([train["outer_train_positions"], stop["outer_train_positions"]])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Original TRAIN and STOP overlap or contain duplicates")
    # Never slice the union-normalized representation into folds.
    frame = pd.concat([train["X"], stop["X"]], ignore_index=True)
    labels = np.concatenate([train["y"], stop["y"]])
    return frame, labels, ids, {"row_order": "TRAIN followed by STOP", "rows": len(ids),
        "ordered_ids_sha256": index_hash(ids), "source_roles": {name: case["roles"][name] for name in ("train", "stop")}}


def make_folds(case, labels, ids, split_seed):
    import numpy as np
    from sklearn.model_selection import KFold, StratifiedKFold
    classification = case["problem_type"] != "regression"
    splitter = (StratifiedKFold(5, shuffle=True, random_state=split_seed) if classification else
                KFold(5, shuffle=True, random_state=split_seed))
    assignments = np.full(len(labels), -1, dtype=np.int8)
    records = []
    expected_classes = np.arange(case["num_classes"]) if classification else None
    for fold, (fit_ids, valid_ids) in enumerate(splitter.split(np.zeros(len(labels)), labels if classification else None)):
        assignments[valid_ids] = fold
        if classification:
            np.testing.assert_array_equal(np.unique(labels[fit_ids]), expected_classes)
            np.testing.assert_array_equal(np.unique(labels[valid_ids]), expected_classes)
        records.append({"fold": fold, "train_rows": len(fit_ids), "validation_rows": len(valid_ids),
            "train_index_sha256": index_hash(fit_ids), "validation_index_sha256": index_hash(valid_ids),
            "train_class_counts": np.bincount(labels[fit_ids].astype(int)).tolist() if classification else None,
            "validation_class_counts": np.bincount(labels[valid_ids].astype(int)).tolist() if classification else None})
    if np.any(assignments < 0):
        raise ValueError("Every union row must be held out exactly once")
    return {"dataset": case["dataset_name"], "split_seed": split_seed, "assignments": assignments.tolist(),
            "ordered_union_ids_sha256": index_hash(ids), "folds": records}


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve every previous attempt")
    parent = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (parent["wave"] != 12 or parent["seed"] != 47 or parent["runtime"] != runtime
            or parent["script_sha256"] != base.digest(THRESHOLD_PATH)
            or parent["scipy_version"] != importlib.metadata.version("scipy") or not frozen["all_fits_successful"]
            or development["development_target_met"] or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")
            or frozen["selected_training_sha256"] != base.digest(args.previous / "selected_training.json")):
        raise ValueError("Expected intact failed wave12 gate")
    expected = {(case["dataset_name"], library) for case in parent["cases"] for library in base.LIBRARIES}
    for records in (frozen["jobs"], frozen["selections"]):
        if len(records) != 36 or {(r["dataset"], r["library"]) for r in records} != expected:
            raise ValueError("Expected36 unique parent policies")
    if len(parent["cases"]) != 12 or len(expected) != 36:
        raise ValueError("Expected12 unique tasks")
    for selected in frozen["selections"]:
        if base.digest(selected["model_path"]) != selected["model_sha256"]:
            raise ValueError("Parent native model changed")
    dependencies = dict(parent["dependencies"])
    for path, expected_hash in dependencies.items():
        if base.digest(path) != expected_hash:
            raise ValueError("Parent implementation changed")
    for path in (THRESHOLD_PATH, SEED_PATH, OUTER_PATH, WATCHDOG_PATH, POLICY):
        dependencies[str(path)] = base.digest(path)
    dependencies.update({str(args.previous / name): base.digest(args.previous / name)
                         for name in ("plan.json", "frozen_selections.json", "development.json", "selected_training.json")})
    manifests = []
    for case in parent["cases"]:
        import numpy as np
        frame, labels, ids, receipt = raw_union(case)
        manifest = {**make_folds(case, labels, ids, 13013), "data": receipt, "feature_names": list(frame.columns)}
        assignments = np.asarray(manifest["assignments"])
        for fold in manifest["folds"]:
            _, schema = base.normalize(frame.iloc[np.flatnonzero(assignments != fold["fold"])])
            fold["train_category_schema"] = schema
        manifests.append(manifest)
    jobs = [{**job, "anchor_rounds": job["rounds"], "round_candidates": [max(1, (job["rounds"] + 1) // 2), job["rounds"], 2 * job["rounds"]]}
            for job in frozen["jobs"]]
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "fold_manifest.json", manifests)
    plan = {"protocol": "fivefold_fixed_round_union_20261016", "wave": 13, "created_at": base.now(),
        "script_sha256": base.digest(__file__), "dependencies": dependencies, "runtime": runtime,
        "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous), "seed": 47,
        "cases": parent["cases"], "jobs": jobs, "fold_manifest_sha256": base.digest(args.output / "fold_manifest.json"),
        "resources": {"workers": 2, "threads_per_fit": 1, "cpus": [4, 8],
                      "maximum_job_seconds": 1200, "maximum_job_rss_bytes": 8 * 1024 ** 3},
        "fold_models_retained": False, "target": parent["target"], "official_test_scored": False,
        "selection": "Five shared folds of raw original TRAIN+STOP. Independently fit each of3requested counts on each fold, no slicing/earlystop. Recompute frozen corrections on foldTRAIN, retain fixedT. Mean fold primaryerror selectscount; tiespreferfewerrequestedroundsthenprofileindex. Keepduplicateroundindices. Freeze36choices beforemandatory36unionrefits; no original-versus-CV fallback.",
        "limitations": "Inherited tree/correction/T controls already used originalSTOP, so CV is further adaptive tuning, not an unbiased performance estimate. Different libraries' anchors imply different compute and model capacity despite equal15fits/job. Fixed corrections/T can be suboptimal on folds. Fold models are hashed and prediction-roundtripped but not retained; all predictions, native counts and receipts remain. No pooled-fold AUC because margin scales differ."}
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "fold_fits": 540, "mandatory_union_fits": 36,
           "duplicate_round_indices": sum(3 - len(set(job["round_candidates"])) for job in jobs)})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["runtime"] != base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy") or args.previous != Path(plan["previous"])
            or plan["fold_manifest_sha256"] != base.digest(args.output / "fold_manifest.json")):
        raise ValueError("Frozen implementation/runtime/folds changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def fit_fold(job, case, raw_train, labels, raw_validation, seed):
    """Fresh fold pipeline and model; the validation labels are not provided."""
    X, schema = base.normalize(raw_train)
    Xv, _ = base.normalize(raw_validation, schema)
    # Assert the learned vocabulary uses observed fold-training values only,
    # including when pandas retains unused category metadata in the raw slice.
    for name, values in schema.items():
        observed = sorted(raw_train[name].dropna().unique().tolist(), key=lambda value: (type(value).__name__, str(value)))
        if values != observed:
            raise ValueError("Fold vocabulary leaked outside fold TRAIN")
    payload, details = union.fit_fixed(job, case, X, labels, schema, seed)
    payload, correction = threshold.correct(payload, job, X, labels, Xv)
    if job.get("temperature") is not None:
        payload = {**payload, "temperature": job["temperature"]}
    details.update(correction=correction, temperature=job.get("temperature"),
                   fold_category_counts={name: len(values) for name, values in schema.items()},
                   fold_category_schema=schema, feature_names=list(raw_train.columns))
    return payload, details


def fit(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity([plan["resources"]["cpus"][args.slot]])
    job = next(j for j in plan["jobs"] if (j["dataset"], j["library"]) == (args.dataset, args.library))
    case = next(c for c in plan["cases"] if c["dataset_name"] == args.dataset)
    manifest = next(m for m in base.read(args.output / "fold_manifest.json") if m["dataset"] == args.dataset)
    raw, labels, ids, receipt = raw_union(case)
    if receipt != manifest["data"] or index_hash(ids) != manifest["ordered_union_ids_sha256"]:
        raise ValueError("Union row provenance changed")
    assignments = np.asarray(manifest["assignments"])
    directory = args.output / "fits" / args.dataset / args.library
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for index, rounds in enumerate(job["round_candidates"]):
        for fold in range(5):
            while (args.artifacts / "pause_fits").exists():
                time.sleep(.5)
            record = {"dataset": args.dataset, "library": args.library, "profile_index": index,
                      "fold": fold, "requested_rounds": rounds, "status": "failed", "started_at": base.now()}
            try:
                train_ids, valid_ids = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
                expected = manifest["folds"][fold]
                if index_hash(train_ids) != expected["train_index_sha256"] or index_hash(valid_ids) != expected["validation_index_sha256"]:
                    raise ValueError("Fold membership changed")
                if case["problem_type"] != "regression":
                    np.testing.assert_array_equal(np.unique(labels[train_ids]), np.arange(case["num_classes"]))
                    np.testing.assert_array_equal(np.unique(labels[valid_ids]), np.arange(case["num_classes"]))
                payload, details = fit_fold({**job, "rounds": rounds}, case, raw.iloc[train_ids], labels[train_ids], raw.iloc[valid_ids], plan["seed"])
                if details["fold_category_schema"] != expected["train_category_schema"] or details["feature_names"] != manifest["feature_names"]:
                    raise ValueError("Fold feature schema differs from frozen manifest")
                predictions = predict(payload, raw.iloc[valid_ids])
                error = base.error(predictions, labels[valid_ids], case["problem_type"])
                encoded = pickle.dumps(payload, protocol=5)
                restored = pickle.loads(encoded)
                np.testing.assert_array_equal(predictions, predict(restored, raw.iloc[valid_ids]))
                pred_path = args.artifacts / "fold_predictions" / args.dataset / args.library / f"profile{index}-fold{fold}.npz"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pred_path, predictions=predictions, labels=labels[valid_ids],
                                    union_positions=valid_ids, outer_train_positions=ids[valid_ids])
                record.update(details, status="ok", validation_error=error, model_sha256=base.hashlib.sha256(encoded).hexdigest(),
                    model_bytes=len(encoded), model_retained=False, prediction_path=str(pred_path), prediction_sha256=base.digest(pred_path))
                del payload, restored, encoded
                gc.collect()
            except Exception:
                record["error"] = traceback.format_exc()
            record["finished_at"] = base.now()
            records.append(record)
            base.write(directory / f"profile{index}-fold{fold}.json", union.refit.plain_receipt(record))
    successful = len(records) == 15 and all(r["status"] == "ok" for r in records)
    scores = [{"profile_index": index, "requested_rounds": rounds,
               "mean_fold_error": float(np.mean([r["validation_error"] for r in records if r["profile_index"] == index]))}
              for index, rounds in enumerate(job["round_candidates"])] if successful else []
    best = min(scores, key=lambda row: (row["mean_fold_error"], row["requested_rounds"], row["profile_index"])) if scores else None
    base.write(directory / "selection.json", {"dataset": args.dataset, "library": args.library, "all_fits_successful": successful,
        "scores": scores, "best": best, "job": {**job, "rounds": best["requested_rounds"]} if best else None})


def run(args):
    plan = checked(args)
    if (args.output / "logs").exists() or (args.output / "fits").exists():
        raise ValueError("Preserve every started run")
    (args.output / "logs").mkdir()
    abort = threading.Event()
    dispatch_lock = threading.Lock()
    def worker(slot):
        for job in plan["jobs"][slot::2]:
            if abort.is_set():
                return
            while (args.artifacts / "pause_fits").exists():
                if abort.is_set():
                    return
                time.sleep(.5)
            name = f"{job['dataset']}-{job['library']}"
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "fit", "--output", str(args.output),
                "--artifacts", str(args.artifacts), "--package-root", str(args.package_root), "--dependencies", str(args.dependencies),
                "--previous", str(args.previous), "--dataset", job["dataset"], "--library", job["library"], "--slot", str(slot)]
            with (args.output / "logs" / (name + ".txt")).open("w", encoding="utf-8") as log:
                with dispatch_lock:
                    if abort.is_set():
                        return
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                resource = watchdog.watch(process, maximum_seconds=plan["resources"]["maximum_job_seconds"],
                    maximum_rss_bytes=plan["resources"]["maximum_job_rss_bytes"])
            base.write(args.output / "logs" / (name + "-resource.json"), resource)
            if not resource["cleanup_complete"]:
                with dispatch_lock:
                    abort.set()
                raise RuntimeError("Worker tree cleanup failed; no further jobs may start")
            print({"completed": job["dataset"], "library": job["library"], "exit_code": process.returncode,
                   "failure": resource["resource_failure"]}, flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, range(2)))
    choices, failures = [], []
    for job in plan["jobs"]:
        path = args.output / "fits" / job["dataset"] / job["library"] / "selection.json"
        resource = base.read(args.output / "logs" / f"{job['dataset']}-{job['library']}-resource.json")
        for profile, rounds in enumerate(job["round_candidates"]):
            for fold in range(5):
                receipt = path.parent / f"profile{profile}-fold{fold}.json"
                if not receipt.exists():
                    base.write(receipt, {"dataset": job["dataset"], "library": job["library"], "profile_index": profile,
                        "fold": fold, "requested_rounds": rounds, "status": "failed", "failure": "not_completed",
                        "resource": resource, "fit_completion_unknown": True})
        if path.exists() and resource["exit_code"] == 0 and resource["resource_failure"] is None:
            choice = base.read(path)
            choices.append(choice)
            if not choice["all_fits_successful"]:
                failures.append({"dataset": job["dataset"], "library": job["library"], "reason": "failed_fold_fit"})
        else:
            failures.append({"dataset": job["dataset"], "library": job["library"], "reason": "missing_or_failed_job", "resource": resource})
    expected = {(j["dataset"], j["library"]) for j in plan["jobs"]}
    complete = not failures and len(choices) == 36 and {(c["dataset"], c["library"]) for c in choices} == expected
    base.write(args.output / "selected_cv.json", {"plan_sha256": base.digest(args.output / "plan.json"), "created_at": base.now(),
        "choices": choices, "failures": failures, "all_fits_successful": complete, "development_scored": False})


def refit(args):
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity([plan["resources"]["cpus"][0]])
    selected_path = args.output / "selected_cv.json"
    selected_hash, selected = base.digest(selected_path), base.read(selected_path)
    expected = {(j["dataset"], j["library"]) for j in plan["jobs"]}
    if (selected["plan_sha256"] != base.digest(args.output / "plan.json") or not selected["all_fits_successful"]
            or len(selected["choices"]) != 36 or {(c["dataset"], c["library"]) for c in selected["choices"]} != expected):
        raise ValueError("All540 fold fits and36 choices must pass before union refit")
    if (args.output / "refits").exists():
        raise ValueError("Preserve every started refit")
    (args.output / "refits").mkdir()
    selections, jobs, success = [], [], True
    for choice in selected["choices"]:
        job = choice["job"]
        jobs.append(job)
        case = next(c for c in plan["cases"] if c["dataset_name"] == job["dataset"])
        record = {"dataset": job["dataset"], "library": job["library"], "status": "failed"}
        try:
            payload, details, frame = fit_policy(job, case, plan["seed"])
            path = args.artifacts / "models" / job["dataset"] / (job["library"] + ".pkl")
            selection = {"dataset": job["dataset"], "library": job["library"], "source_wave": 13,
                "cv_selection": choice, "best": {"parameters": job["parameters"], "retained_rounds": details["retained_rounds"], "stop_error": None},
                "refit_details": details, "leaf_refit": job.get("leaf_refit"), "binary_refit": job.get("binary_refit"),
                "temperature": job.get("temperature"), **threshold.persist(payload, path, frame)}
            selections.append(selection)
            record.update(status="ok", selection=selection)
        except Exception:
            success = False
            record["error"] = traceback.format_exc()
        base.write(args.output / "refits" / job["dataset"] / (job["library"] + ".json"), union.refit.plain_receipt(record))
        print({key: record[key] for key in ("dataset", "library", "status")}, flush=True)
    if selected_hash != base.digest(selected_path):
        raise ValueError("CV choices changed during union refit")
    success = success and len(selections) == 36 and {(s["dataset"], s["library"]) for s in selections} == expected
    base.write(args.output / "frozen_selections.json", union.refit.plain_receipt({"plan_sha256": base.digest(args.output / "plan.json"),
        "selected_cv_sha256": selected_hash, "created_at": base.now(), "jobs": jobs, "selections": selections,
        "all_fits_successful": success, "development_scored": False, "official_test_scored": False}))


def development(args):
    frozen = base.read(args.output / "frozen_selections.json")
    if not frozen["all_fits_successful"] or frozen["selected_cv_sha256"] != base.digest(args.output / "selected_cv.json"):
        raise ValueError("All mandatory policies and CV choices must be intact")
    base.checked, base.predict = checked, predict
    base.development(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "fit", "refit", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--library", choices=base.LIBRARIES)
    parser.add_argument("--slot", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "fit": fit, "refit": refit, "development": development}[args.stage](args)
