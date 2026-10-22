"""Inactive executable draft: current-data fivefold fixed-policy round selection.

No stage has been run or frozen. Activation requires a complete, numerically
successful17b policy with a failed primary or linked569 mean gate. The primary
47 policy is always the anchor. Existing source/data/pipeline semantics remain
unchanged; only requested round count is selected by fold primary-error means.
"""
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

HERE = Path(__file__).resolve().parent
PARENT_PATH = HERE / "fair_accuracy_scalar_numerical_retry.py"
PARENT_SEED_PATH = HERE / "fair_accuracy_scalar_numerical_seed_check.py"
SEED_PATH = HERE / "fair_accuracy_current_round_cv_seed_check.py"
OUTER_PATH = HERE / "fair_accuracy_current_round_cv_outer_test.py"
POLICY = HERE / "fair_accuracy_current_round_cv_policy.json"
MEMORY_PATH = HERE / "results/improvements_20261016_followup/full_current_cv_memory/estimates.json"
PROTOCOL = "full_current_fivefold_round_cv_20261016"
SPLIT_SEED, SECOND_SEED = 18018, 677


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parent = load("current_cv_parent", PARENT_PATH)
support = load("current_cv_support", HERE / "fair_accuracy_current_round_cv_support.py")
adapter = load("current_cv_backend", HERE / "fair_accuracy_current_round_cv_backend.py")
resources = load("current_cv_resources", HERE / "fair_accuracy_current_round_cv_resources.py")
base, union, cv, predict = parent.base, parent.union, parent.cv, parent.predict
TEMPERATURE_PATH = union.TEMPERATURE_PATH


def expected_pairs(cases, jobs, selections=None):
    expected = {(case["dataset_name"], library) for case in cases for library in base.LIBRARIES}
    if len(cases) != 12 or len(expected) != 36:
        raise ValueError("Expected twelve distinct tasks")
    for records in (jobs,) if selections is None else (jobs, selections):
        if len(records) != 36 or {(row["dataset"], row["library"]) for row in records} != expected:
            raise ValueError("Expected36 unique task/library records")
    return expected


def validate_admission(prior, frozen, development, controls, hashes, linked_seed=None):
    """Pure gate checks; linked_seed is already source/runtime/hash verified."""
    if (prior.get("wave") != "17b" or prior.get("protocol") != "certified_numerical_scalar_refresh_retry_20261016"
            or prior.get("seed") != 47 or frozen.get("all_fits_successful") is not True
            or controls.get("all_fits_successful") is not True
            or frozen.get("new_clean_union_fits_completed") != 24 or frozen.get("carried_models") != 12
            or frozen.get("plan_sha256") != hashes["plan"] or controls.get("plan_sha256") != hashes["plan"]
            or frozen.get("scalar_controls_sha256") != hashes["controls"]
            or development.get("plan_sha256") != hashes["plan"]
            or development.get("frozen_selections_sha256") != hashes["frozen"]):
        raise ValueError("Admission requires a complete numerically successful17b policy")
    if any(document.get("official_test_scored") is not False for document in (prior, frozen, development)):
        raise ValueError("Official-test access prohibits a new selection wave")
    expected_pairs(prior["cases"], frozen["jobs"], frozen["selections"])
    if development.get("development_target_met") is False:
        if linked_seed is not None:
            raise ValueError("Primary failure does not permit selecting a seed-derived parent")
        return "primary47_failed"
    if development.get("development_target_met") is not True or linked_seed is None:
        raise ValueError("A passing primary gate requires a complete linked569 mean-gate failure")
    seed_plan, seed_frozen, seed_report = linked_seed
    if (seed_plan.get("seed") != 569 or seed_plan.get("protocol") != "certified_scalar_retry_fixed_seed569_20261016"
            or seed_frozen.get("all_fits_successful") is not True or seed_report.get("development_target_met") is not False
            or any(doc.get("official_test_scored") is not False for doc in linked_seed)
            or any(row.get("status") != "ok" for row in seed_frozen["selections"])):
        raise ValueError("Linked569 check must finish successfully and fail its mean quality gate")
    expected_pairs(prior["cases"], seed_plan["jobs"], seed_frozen["selections"])
    originals = {(row["dataset"], row["library"]): row for row in frozen["selections"]}
    for row in seed_frozen["selections"]:
        if row["original_selection"] != originals[row["dataset"], row["library"]]:
            raise ValueError("Linked569 models are not linked to the exact primary47 policy")
    return "linked569_mean_failed_primary47_anchor"


def reject_outer_access(report_root):
    # This task's report tree is one adaptive panel. Even a started one-shot
    # assessment closes admission; never read any official dataset here.
    for name in ("assessment_started.json", "assessment.json", "assessment_failure.json"):
        if next(Path(report_root).rglob(name), None) is not None:
            raise ValueError("Official assessment was accessed in this study")


def raw_union(job, case):
    def expanded(source):
        with Path(source["path"]).open("rb") as stream:
            return pickle.load(stream)
    return support.reconstruct_raw_union(case, job, load_role=base.load_role, load_expanded=expanded, digest=base.digest)


def add_files(dependencies, directory):
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            dependencies[str(path)] = base.digest(path)


def admit(args, runtime):
    prior = base.read(args.previous / "plan.json")
    parent_args = copy.copy(args)
    parent_args.output, parent_args.previous = args.previous, Path(prior["previous"])
    prior = parent.checked(parent_args)
    if prior["runtime"] != runtime or prior["scipy_version"] != importlib.metadata.version("scipy"):
        raise ValueError("Parent numerical runtime changed")
    controls = parent.controls_document(parent_args, prior)
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    hashes = {key: base.digest(args.previous / name) for key, name in
              (("plan", "plan.json"), ("controls", "scalar_controls.json"), ("frozen", "frozen_selections.json"))}
    linked = None
    dependencies = dict(prior["dependencies"])
    if args.failed_seed is not None:
        seed = load("current_cv_failed_seed_parent", PARENT_SEED_PATH)
        seed_args = copy.copy(args)
        seed_args.output, seed_args.previous = args.failed_seed, args.previous
        seed_plan = seed.checked(seed_args)
        seed_frozen = base.read(args.failed_seed / "frozen_selections.json")
        seed_report = base.read(args.failed_seed / "development.json")
        if (Path(seed_plan["previous"]) != args.previous or seed_plan["runtime"] != runtime
                or seed_plan["script_sha256"] != base.digest(PARENT_SEED_PATH)
                or seed_frozen["plan_sha256"] != base.digest(args.failed_seed / "plan.json")
                or seed_report["plan_sha256"] != base.digest(args.failed_seed / "plan.json")
                or seed_report["frozen_selections_sha256"] != base.digest(args.failed_seed / "frozen_selections.json")):
            raise ValueError("The linked seed569 source/runtime/evidence changed")
        linked = seed_plan, seed_frozen, seed_report
        dependencies.update(seed_plan["dependencies"])
        add_files(dependencies, args.failed_seed)
    branch = validate_admission(prior, frozen, development, controls, hashes, linked)
    reject_outer_access(args.previous.parent)
    choices = {(row["dataset"], row["library"]): row["correction"] for row in controls["choices"]}
    old_jobs = {(row["dataset"], row["library"]): row for row in prior["jobs"]}
    problems = {case["dataset_name"]: case["problem_type"] for case in prior["cases"]}
    sources = {row["dataset"]: row for row in prior["expanded_data"]}
    if set(sources) != support.EXPANDED_DATASETS or len(prior["expanded_data"]) != 4:
        raise ValueError("All four authorized expanded sources must be preserved")
    for job in frozen["jobs"]:
        key = job["dataset"], job["library"]
        expected = parent.refreshed_job(old_jobs[key], problems[job["dataset"]], choices[key]) if key in choices else old_jobs[key]
        if job != expected or job.get("expanded_source") != sources.get(job["dataset"]):
            raise ValueError("Current policy or expanded fitting source changed")
    for source in sources.values():
        if base.digest(source["path"]) != source["sha256"]:
            raise ValueError("Authorized fitting source changed")
        dependencies[source["path"]] = source["sha256"]
    for document in (frozen,) if linked is None else (frozen, linked[1]):
        for selected in document["selections"]:
            if base.digest(selected["model_path"]) != selected["model_sha256"]:
                raise ValueError("Parent selected model changed")
            dependencies[selected["model_path"]] = selected["model_sha256"]
    add_files(dependencies, args.previous)
    return prior, frozen, dependencies, branch


def prepare(args):
    import numpy as np
    import psutil
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve previous attempts; choose new directories")
    resource = resources.validate_resources({name: getattr(args, name) for name in
        ("cpu", "maximum_named_array_bytes", "maximum_job_rss_bytes", "maximum_job_seconds",
         "headroom_reserve_bytes", "headroom_wait_seconds", "headroom_poll_seconds")})
    if resource["cpu"] not in psutil.Process().cpu_affinity():
        raise ValueError("Chosen CPU is unavailable")
    prior, frozen, dependencies, admission = admit(args, runtime)
    snapshot = psutil.virtual_memory()
    snapshot_time = base.now()
    memory = base.read(MEMORY_PATH)
    for path, expected in memory["source_sha256"].items():
        path = (base.ROOT / path).resolve()
        if base.digest(path) != expected:
            raise ValueError("Prospective metadata memory source changed")
        dependencies[str(path)] = expected
    bounds = resources.check_named_bounds(memory, prior["cases"], frozen["jobs"], resource["maximum_named_array_bytes"])
    manifests = []
    for case in prior["cases"]:
        job = next(job for job in frozen["jobs"] if job["dataset"] == case["dataset_name"])
        raw, labels, ids, receipt = raw_union(job, case)
        if len(labels) < 5 or (case["problem_type"] != "regression" and np.bincount(labels.astype(int)).min() < 5):
            raise ValueError("Every class/population needs five nonempty held-out folds")
        manifest = cv.make_folds(case, labels, ids, SPLIT_SEED)
        manifest.update(data=receipt, feature_names=list(raw.columns))
        assignments = np.asarray(manifest["assignments"])
        for fold in manifest["folds"]:
            train_ids = np.flatnonzero(assignments != fold["fold"])
            valid_ids = np.flatnonzero(assignments == fold["fold"])
            _, schema = base.normalize(raw.iloc[train_ids])
            fold.update(train_category_schema=schema, train_row_ids_sha256=support.index_hash(ids[train_ids]),
                        validation_row_ids_sha256=support.index_hash(ids[valid_ids]))
        manifests.append(manifest)
    sources = (Path(__file__), PARENT_PATH, PARENT_SEED_PATH, SEED_PATH, OUTER_PATH, POLICY, MEMORY_PATH,
               Path(support.__file__), Path(adapter.__file__), Path(resources.__file__), Path(cv.__file__), cv.WATCHDOG_PATH)
    dependencies.update({str(path): base.digest(path) for path in sources})
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "fold_manifest.json", manifests)
    policy = base.read(POLICY)
    plan = {"protocol": PROTOCOL, "wave": 18, "seed": 47, "split_seed": SPLIT_SEED, "conditional_seed": SECOND_SEED,
        "created_at": base.now(), "script_sha256": base.digest(__file__), "dependencies": dependencies,
        "runtime": runtime, "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous),
        "failed_seed": str(args.failed_seed) if args.failed_seed else None, "admission": admission,
        "cases": prior["cases"], "jobs": frozen["jobs"], "expanded_data": prior["expanded_data"],
        "fold_manifest_sha256": base.digest(args.output / "fold_manifest.json"), "resources": resource,
        "memory_bounds": bounds, "quiet_baseline_memory": {"captured_at": snapshot_time, "available_bytes": snapshot.available, "total_bytes": snapshot.total,
            "context": "Activation coordinator must ensure no fits/builds/timings overlap this snapshot"},
        "target": prior["target"], "selection": policy["selection"], "limitations": policy["limitations"],
        "fold_models_retained": False, "official_test_scored": False}
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "fold_fits": 540, "mandatory_union_fits": 36})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["protocol"] != PROTOCOL or plan["script_sha256"] != base.digest(__file__)
            or plan["runtime"] != base.bootstrap(args) or plan["scipy_version"] != importlib.metadata.version("scipy")
            or Path(plan["previous"]) != args.previous or plan["split_seed"] != SPLIT_SEED or plan["conditional_seed"] != SECOND_SEED
            or plan["fold_manifest_sha256"] != base.digest(args.output / "fold_manifest.json")):
        raise ValueError("Frozen current-CV implementation/runtime/folds changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen current-CV dependency changed")
    resources.validate_resources(plan["resources"])
    return plan


def backend(plan, progress=None):
    result = adapter.make_backend(scalar=parent, multiclass=parent.parent,
                                  max_working_bytes=plan["resources"]["maximum_named_array_bytes"])
    if progress is not None:
        original = result.fit_clean
        def fit_clean(*args, **kwargs):
            progress("native_fit_started")
            fitted = original(*args, **kwargs)
            progress("native_fit_completed")
            return fitted
        result.fit_clean = fit_clean
    return result


def selected_jobs(args, plan):
    document = base.read(args.output / "selected_cv.json")
    expected_pairs(plan["cases"], [choice["job"] for choice in document["choices"]])
    if document["plan_sha256"] != base.digest(args.output / "plan.json") or document["all_fits_successful"] is not True:
        raise ValueError("All540 folds and36 choices must succeed before union refitting")
    if len(document["fold_receipt_sha256"]) != 540:
        raise ValueError("Every fold receipt must remain frozen")
    for path, expected in document["fold_receipt_sha256"].items():
        if base.digest(path) != expected:
            raise ValueError("A selected fold receipt changed")
    expected_paths = {str(job_path(args, "fold_job", index)) for index in range(540)}
    if set(document["fold_receipt_sha256"]) != expected_paths:
        raise ValueError("Selection must bind exactly the frozen fold workers")
    records = [base.read(job_path(args, "fold_job", index)) for index in range(540)]
    if document["choices"] != support.choose_rounds(plan["jobs"], records):
        raise ValueError("Selected policies differ from the complete frozen fold criterion")
    return [choice["job"] for choice in document["choices"]]


def fit_current(job, case, seed, plan, progress=None):
    raw, labels, ids, data_receipt = raw_union(job, case)
    probe = raw.iloc[:min(64, len(raw))]
    payload, _, details = support.fit_fold(job, case, raw, labels, probe, seed, backend(plan, progress))
    details.update(fitting_data=data_receipt, verification_scope="Fitting-row prefix for reload only; no selection score")
    return payload, details, probe


def job_path(args, stage, index):
    return args.output / (stage + "s") / f"{index:04d}.json"


def fit_job(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity([plan["resources"]["cpu"]])
    stage, index = args.stage, args.job_index
    if stage not in ("fold_job", "union_job") or type(index) is not int or not 0 <= index < (540 if stage == "fold_job" else 36):
        raise ValueError("Invalid frozen job index or stage")
    if stage == "fold_job":
        identity = support.enumerate_fit_jobs(plan["jobs"])[index]
        job = next(job for job in plan["jobs"] if (job["dataset"], job["library"]) == (identity["dataset"], identity["library"]))
        job = {**job, "rounds": identity["requested_rounds"]}
    else:
        job = selected_jobs(args, plan)[index]
        identity = {"dataset": job["dataset"], "library": job["library"], "seed": 47, "requested_rounds": job["rounds"]}
    case = next(case for case in plan["cases"] if case["dataset_name"] == job["dataset"])
    path = job_path(args, stage, index)
    if path.exists():
        raise ValueError("Preserve every started fit")
    record = {**identity, "status": "failed", "started_at": base.now(), "native_fit_started": False, "native_fit_completed": False}
    base.write(path, record)
    def progress(event):
        record[event] = True
        base.write(path, record)
    try:
        if stage == "fold_job":
            raw, labels, ids, receipt = raw_union(job, case)
            manifest = next(row for row in base.read(args.output / "fold_manifest.json") if row["dataset"] == job["dataset"])
            if receipt != manifest["data"] or support.index_hash(ids) != manifest["ordered_union_ids_sha256"]:
                raise ValueError("Current fitting row manifest changed")
            assignments = np.asarray(manifest["assignments"])
            train_ids, valid_ids = np.flatnonzero(assignments != identity["fold"]), np.flatnonzero(assignments == identity["fold"])
            fold = manifest["folds"][identity["fold"]]
            for values, key in ((train_ids, "train_index_sha256"), (valid_ids, "validation_index_sha256"),
                                (ids[train_ids], "train_row_ids_sha256"), (ids[valid_ids], "validation_row_ids_sha256")):
                if support.index_hash(values) != fold[key]:
                    raise ValueError("Frozen fold row identities changed")
            if case["problem_type"] != "regression":
                for values in (labels[train_ids], labels[valid_ids]):
                    np.testing.assert_array_equal(np.unique(values), np.arange(case["num_classes"]))
            payload, predictions, details = support.fit_fold(job, case, raw.iloc[train_ids], labels[train_ids], raw.iloc[valid_ids], 47, backend(plan, progress))
            if details["fold_category_schema"] != fold["train_category_schema"] or details["feature_names"] != manifest["feature_names"]:
                raise ValueError("Fold schema changed")
            prediction_path = args.artifacts / "fold_predictions" / f"{index:04d}.npz"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(prediction_path, predictions=predictions, labels=labels[valid_ids],
                                union_positions=valid_ids, outer_train_positions=ids[valid_ids])
            record.update(details, validation_error=base.error(predictions, labels[valid_ids], case["problem_type"]),
                          prediction_path=str(prediction_path), prediction_sha256=base.digest(prediction_path))
        else:
            payload, details, probe = fit_current(job, case, 47, plan, progress)
            saved = cv.threshold.persist(payload, args.artifacts / "models" / job["dataset"] / (job["library"] + ".pkl"), probe)
            record["selection"] = {**identity, "source_wave": 18, "fixed_job": job,
                "best": {"parameters": job["parameters"], "retained_rounds": details["retained_rounds"]},
                "refit_details": details, **{key: job.get(key) for key in (*support.CORRECTION_FIELDS, "temperature")}, **saved}
        record["status"] = "ok"
    except Exception:
        record["error"] = traceback.format_exc()
    record["finished_at"] = base.now()
    base.write(path, union.refit.plain_receipt(record))


def dispatch(args, plan, stage, identities, *, script=None):
    """Exactly one fit/correction per owned process; terminal failure stops work."""
    import psutil
    script = Path(__file__).resolve() if script is None else Path(script).resolve()
    logs = args.output / (stage + "_logs")
    if logs.exists():
        raise ValueError("Preserve every started stage")
    logs.mkdir(parents=True)
    records, terminal = [], None
    for index, identity in enumerate(identities):
        while terminal is None and (args.artifacts / "pause_fits").exists():
            time.sleep(.5)
        path = job_path(args, stage, index)
        if terminal is not None:
            record = {**identity, "status": "failed", "failure": "not_started_after_terminal_failure", "cause": terminal}
            base.write(path, record)
            records.append(record)
            continue
        headroom = resources.wait_for_headroom(plan["resources"], available=lambda: psutil.virtual_memory().available)
        base.write(logs / f"{index:04d}-headroom.json", headroom)
        if not headroom["sufficient"]:
            terminal = "insufficient_system_headroom"
            record = {**identity, "status": "failed", "failure": terminal, "headroom": headroom}
        else:
            command = [sys.executable, "-I", str(script), stage]
            for name in ("output", "artifacts", "package_root", "dependencies", "previous"):
                command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
            command += ["--job-index", str(index)]
            with (logs / f"{index:04d}.txt").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                resource = cv.watchdog.watch(process, maximum_seconds=plan["resources"]["maximum_job_seconds"],
                                             maximum_rss_bytes=plan["resources"]["maximum_job_rss_bytes"])
            base.write(logs / f"{index:04d}-resource.json", resource)
            if not resource["cleanup_complete"]:
                raise RuntimeError("Owned-process cleanup incomplete; no further dispatch")
            record = base.read(path) if path.exists() else {**identity, "status": "failed", "failure": "worker_did_not_complete"}
            if resource["exit_code"] != 0 or resource["resource_failure"] is not None:
                record.update(status="failed", resource=resource)
            if record["status"] != "ok":
                terminal = "owned_fit_or_resource_failed"
        base.write(path, record)
        records.append(record)
        print({"stage": stage, "index": index, "status": record["status"]}, flush=True)
    return records


def completion_counts(records, planned):
    """Separate native returns from fully verified fit/correction/persistence jobs."""
    return {"planned_jobs": planned, "native_fits_started": sum(row.get("native_fit_started") is True for row in records),
            "native_fits_completed": sum(row.get("native_fit_completed") is True for row in records),
            "fully_verified_jobs": sum(row["status"] == "ok" for row in records),
            "limits": "Completed means the native fit returned and passed clean-fit checks. A terminated fit without this checkpoint is not counted."}


def run_folds(args):
    plan = checked(args)
    records = dispatch(args, plan, "fold_job", support.enumerate_fit_jobs(plan["jobs"]))
    base.write(args.output / "fold_completion.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "all_fits_successful": len(records) == 540 and all(row["status"] == "ok" for row in records),
        "completion_counts": completion_counts(records, 540),
        "records": [str(job_path(args, "fold_job", index)) for index in range(len(records))]})


def select(args):
    plan = checked(args)
    if (args.output / "selected_cv.json").exists():
        raise ValueError("Preserve the existing selection")
    completion = base.read(args.output / "fold_completion.json")
    if completion["plan_sha256"] != base.digest(args.output / "plan.json") or completion["all_fits_successful"] is not True:
        raise ValueError("All540 fold processes must succeed")
    records = [base.read(job_path(args, "fold_job", index)) for index in range(540)]
    for row in records:
        if base.digest(row["prediction_path"]) != row["prediction_sha256"]:
            raise ValueError("Fold prediction artifact changed")
    choices = support.choose_rounds(plan["jobs"], records)
    base.write(args.output / "selected_cv.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "all_fits_successful": True, "choices": choices, "development_scored": False,
        "fold_receipt_sha256": {str(job_path(args, "fold_job", index)): base.digest(job_path(args, "fold_job", index)) for index in range(540)}})


def run_union(args):
    plan = checked(args)
    jobs = selected_jobs(args, plan)
    selected_hash = base.digest(args.output / "selected_cv.json")
    records = dispatch(args, plan, "union_job", [{"dataset": job["dataset"], "library": job["library"], "seed": 47} for job in jobs])
    if selected_hash != base.digest(args.output / "selected_cv.json"):
        raise ValueError("CV choices changed during union refits")
    selections = [row["selection"] for row in records if row["status"] == "ok"]
    success = len(selections) == 36 and all(row["status"] == "ok" for row in records)
    if success:
        expected_pairs(plan["cases"], jobs, selections)
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "selected_cv_sha256": selected_hash, "jobs": jobs, "selections": selections, "all_fits_successful": success,
        "created_at": base.now(), "fold_completion_counts": base.read(args.output / "fold_completion.json")["completion_counts"],
        "union_completion_counts": completion_counts(records, 36), "official_test_scored": False})


def development(args):
    plan = checked(args)
    frozen = base.read(args.output / "frozen_selections.json")
    expected_pairs(plan["cases"], frozen["jobs"], frozen["selections"])
    if (frozen["all_fits_successful"] is not True or frozen["plan_sha256"] != base.digest(args.output / "plan.json")
            or frozen["selected_cv_sha256"] != base.digest(args.output / "selected_cv.json")
            or frozen["jobs"] != selected_jobs(args, plan)):
        raise ValueError("Every mandatory current-union fit must succeed")
    base.checked, base.predict = checked, predict
    base.development(args)


def parser(stages):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=stages)
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        result.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    result.add_argument("--job-index", type=int)
    return result


if __name__ == "__main__":
    cli = parser(("prepare", "run_folds", "fold_job", "select", "run_union", "union_job", "development"))
    cli.add_argument("--failed-seed", type=lambda value: Path(value).resolve())
    for name in ("cpu", "maximum-named-array-bytes", "maximum-job-rss-bytes", "headroom-reserve-bytes"):
        cli.add_argument("--" + name, type=int)
    for name in ("maximum-job-seconds", "headroom-wait-seconds", "headroom-poll-seconds"):
        cli.add_argument("--" + name, type=float)
    args = cli.parse_args()
    if args.stage in ("fold_job", "union_job"):
        fit_job(args)
    else:
        globals()[args.stage](args)
