"""Inactive runner tests using synthetic rows and fake learners only.

No real cache/model payloads, native fits, quality assessments or protocol
freezes are performed. Files under tmp_path are synthetic stage artifacts.
"""
from __future__ import annotations
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import importlib.util


def load(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name + "_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = load("fair_accuracy_current_round_cv")
seed = load("fair_accuracy_current_round_cv_seed_check")
outer = load("fair_accuracy_current_round_cv_outer_test")
res = main.resources


def resources():
    return dict(cpu=0, maximum_named_array_bytes=6 * 2**30, maximum_job_rss_bytes=10 * 2**30,
                headroom_reserve_bytes=4 * 2**30, maximum_job_seconds=1200,
                headroom_wait_seconds=300, headroom_poll_seconds=5, workers=1, threads_per_fit=1)


def panel():
    cases = [{"dataset_name": f"synthetic{i}", "problem_type": ("regression", "binary", "multiclass")[i // 4],
              "num_classes": 3 if i >= 8 else 2} for i in range(12)]
    jobs = [{"dataset": case["dataset_name"], "library": library, "rounds": 2, "parameters": {},
             **{field: None for field in main.support.CORRECTION_FIELDS}, "temperature": None}
            for case in cases for library in main.base.LIBRARIES]
    return cases, jobs


def admission():
    cases, jobs = panel()
    prior = dict(wave="17b", protocol="certified_numerical_scalar_refresh_retry_20261016", seed=47,
                 official_test_scored=False, cases=cases)
    frozen = dict(all_fits_successful=True, new_clean_union_fits_completed=24, carried_models=12,
                  plan_sha256="plan", scalar_controls_sha256="controls", official_test_scored=False,
                  jobs=jobs, selections=[dict(dataset=j["dataset"], library=j["library"], model_sha256="a"*64) for j in jobs])
    dev = dict(plan_sha256="plan", frozen_selections_sha256="frozen", official_test_scored=False, development_target_met=False)
    controls = dict(all_fits_successful=True, plan_sha256="plan")
    hashes = dict(plan="plan", controls="controls", frozen="frozen")
    return prior, frozen, dev, controls, hashes


def linked(frozen):
    return (dict(seed=569, protocol="certified_scalar_retry_fixed_seed569_20261016", official_test_scored=False,
                 jobs=frozen["jobs"]),
            dict(all_fits_successful=True, official_test_scored=False, selections=[
                dict(dataset=s["dataset"], library=s["library"], status="ok", original_selection=copy.deepcopy(s))
                for s in frozen["selections"]]),
            dict(development_target_met=False, official_test_scored=False))


def test_both_admission_branches_always_anchor_primary47():
    docs = admission()
    assert main.validate_admission(*docs) == "primary47_failed"
    docs[2]["development_target_met"] = True
    assert main.validate_admission(*docs, linked(docs[1])) == "linked569_mean_failed_primary47_anchor"


@pytest.mark.parametrize("defect", ["native_failure", "numerical_failure", "incomplete_transfer", "parent_hash", "controls_hash",
    "official_access", "primary_pass", "wrong_seed", "seed_pass", "seed_failure", "seed_original", "extra_seed_after_failure"])
def test_incomplete_or_passing_policy_cannot_enter_new_cv(defect):
    docs = admission()
    secondary = None
    if defect in ("wrong_seed", "seed_pass", "seed_failure", "seed_original"):
        docs[2]["development_target_met"] = True
        secondary = linked(docs[1])
        if defect == "wrong_seed": secondary[0]["seed"] = 947
        if defect == "seed_pass": secondary[2]["development_target_met"] = True
        if defect == "seed_failure": secondary[1]["selections"][0]["status"] = "failed"
        if defect == "seed_original": secondary[1]["selections"][0]["original_selection"]["model_sha256"] = "b"*64
    elif defect == "native_failure": docs[1]["all_fits_successful"] = False
    elif defect == "numerical_failure": docs[3]["all_fits_successful"] = False
    elif defect == "incomplete_transfer": docs[1]["new_clean_union_fits_completed"] = 23
    elif defect == "parent_hash": docs[1]["plan_sha256"] = "changed"
    elif defect == "controls_hash": docs[1]["scalar_controls_sha256"] = "changed"
    elif defect == "official_access": docs[2]["official_test_scored"] = True
    elif defect == "primary_pass": docs[2]["development_target_met"] = True
    else: secondary = linked(docs[1])
    with pytest.raises(ValueError): main.validate_admission(*docs, secondary)


@pytest.mark.parametrize("key,value", [("cpu", -1), ("cpu", True), ("maximum_named_array_bytes", 0),
    ("maximum_job_rss_bytes", 0), ("headroom_reserve_bytes", 0), ("maximum_job_seconds", 1201),
    ("headroom_wait_seconds", 0), ("headroom_poll_seconds", float("nan")),
    ("headroom_poll_seconds", 301), ("maximum_named_array_bytes", 10*2**30), ("workers", 2), ("threads_per_fit", 2)])
def test_resources_are_explicit_positive_and_fixed(key, value):
    values = resources(); values[key] = value
    with pytest.raises(ValueError): res.validate_resources(values)


@pytest.mark.parametrize("eventual", [False, True])
def test_headroom_wait_is_bounded_and_records_allowance_and_reserve(eventual):
    values = resources(); values.update(headroom_wait_seconds=12, headroom_poll_seconds=5)
    clock = [0.]; waits = []
    def sleep(seconds): waits.append(seconds); clock[0] += seconds
    result = res.wait_for_headroom(values, clock=lambda: clock[0], sleep=sleep,
        available=lambda: (14*2**30 if eventual and clock[0] >= 10 else 13*2**30))
    assert result["sufficient"] is eventual and result["required_available_bytes"] == 14*2**30
    assert result["headroom_reserve_bytes"] == 4*2**30 and result["process_rss_allowance_bytes"] == 10*2**30
    assert waits == ([5, 5] if eventual else [5, 5, 2]) and clock[0] <= 12


def memory_fixture(problem, library, vector=False):
    case = dict(dataset_name="synthetic", problem_type=problem, num_classes=3,
                roles=dict(train=dict(rows=13), stop=dict(rows=8)))
    params = dict(max_depth=3, num_leaves=7)
    if vector: params["multi_strategy"] = "multi_output_tree"
    job = dict(dataset="synthetic", library=library, parameters=params, rounds=5,
               **{main.support.CORRECTION_FIELD[problem]: dict(solver={"maxcor": 20})})
    classes = 3 if problem == "multiclass" else 1
    rows = []
    for multiplier, count in (("0.5", 3), ("1", 5), ("2", 10)):
        for stage, n in (("fivefold_train_max", 17), ("full_union", 21)):
            rows.append(dict(dataset="synthetic", library=library, multiplier=multiplier, stage=stage,
                current_requested_rounds=5, requested_rounds=count, problem=problem, classes=classes,
                vector_leaves=vector, fitting_rows=n, full_fitting_rows=21,
                physical_trees_per_requested_round=classes if not vector else 1,
                maximum_native_leaves_per_tree=7 if library == "lightgbm" else 8,
                named_total_bytes=90 if stage == "fivefold_train_max" else 150))
    return dict(estimates=rows), [case], [job]


@pytest.mark.parametrize("problem,library,vector", [(p, lib, vector) for p in ("regression", "binary", "multiclass")
    for lib in main.base.LIBRARIES for vector in ((False, True) if p == "multiclass" and lib != "lightgbm" else (False,))])
def test_all_scalar_vector_library_bounds_include_maximum_union(problem, library, vector):
    report, cases, jobs = memory_fixture(problem, library, vector)
    assert res.check_named_bounds(report, cases, jobs, 150)[0]["profiles_checked"] == 6
    with pytest.raises(MemoryError): res.check_named_bounds(report, cases, jobs, 100)
    report["estimates"].pop()
    with pytest.raises(ValueError): res.check_named_bounds(report, cases, jobs, 200)


def test_metadata_bound_rejects_unplanned_shape_or_solver():
    report, cases, jobs = memory_fixture("multiclass", "ctboost", True)
    jobs[0]["multiclass_refit"]["solver"]["maxcor"] = 30
    with pytest.raises(ValueError): res.check_named_bounds(report, cases, jobs, 200)
    jobs[0]["multiclass_refit"]["solver"]["maxcor"] = 20
    jobs[0]["rounds"] = 6
    with pytest.raises(ValueError): res.check_named_bounds(report, cases, jobs, 200)


def synthetic_args(tmp_path):
    return SimpleNamespace(output=tmp_path / "output", artifacts=tmp_path / "artifacts", previous=tmp_path / "parent",
                           package_root=tmp_path / "runtime", dependencies=tmp_path / "deps", job_index=0, stage="fold_job")


@pytest.mark.parametrize("failure", ["headroom", "worker", "cleanup"])
def test_dispatch_stops_after_terminal_failure_and_owns_one_process(tmp_path, monkeypatch, failure):
    args, launched = synthetic_args(tmp_path), []
    plan = dict(resources=resources())
    monkeypatch.setattr(res, "wait_for_headroom", lambda *a, **k: dict(sufficient=failure != "headroom"))
    class Process:
        def __init__(self, command, **kwargs):
            launched.append(command)
            main.base.write(main.job_path(args, "fold_job", 0), dict(status="failed", native_fit_started=True,
                native_fit_completed=True, error="synthetic correction rejection after fake native return"))
    monkeypatch.setattr(main.subprocess, "Popen", Process)
    monkeypatch.setattr(main.cv.watchdog, "watch", lambda *a, **k: dict(cleanup_complete=failure != "cleanup", exit_code=0, resource_failure=None))
    identities = [dict(dataset="synthetic", library="ctboost", seed=47, profile_index=0, fold=i) for i in range(3)]
    if failure == "cleanup":
        with pytest.raises(RuntimeError, match="cleanup"): main.dispatch(args, plan, "fold_job", identities)
    else:
        records = main.dispatch(args, plan, "fold_job", identities)
        assert [row["status"] for row in records] == ["failed"]*3
        assert records[1]["failure"] == "not_started_after_terminal_failure"
        counts = main.completion_counts(records, 3)
        assert counts["fully_verified_jobs"] == 0 and counts["native_fits_completed"] == int(failure == "worker")
    assert len(launched) == int(failure != "headroom")
    if launched: assert launched[0][-2:] == ["--job-index", "0"]


def fake_predict(payload, frame):
    values = frame["num"].to_numpy(dtype=float)
    if payload["problem"] == "multiclass": return np.stack([values, -values, values*.2], axis=1)
    return values


def fake_backend(plan, progress=None):
    def normalize(frame, schema=None): return frame.copy(), {} if schema is None else schema
    def fit(clean, case, frame, labels, schema, seed):
        if progress: progress("native_fit_started")
        payload = dict(problem=case["problem_type"], library=clean["library"], schema=schema,
                       model="fake learner; no native fit", seed=seed)
        if progress: progress("native_fit_completed")
        return payload, dict(requested_rounds=clean["rounds"], retained_rounds=clean["rounds"], early_stopping=False)
    return SimpleNamespace(normalize=normalize, fit_clean=fit, predict=fake_predict)


def test_actual_worker_persistence_select_union_and_development_handoff(tmp_path, monkeypatch):
    """One real runner fold worker +36 union workers use fake learners, never native fits."""
    args = synthetic_args(tmp_path)
    cases, jobs = panel()
    plan = dict(cases=cases, jobs=jobs, resources=resources(), limitations="synthetic stage integration")
    main.base.write(args.output / "plan.json", plan)
    monkeypatch.setattr(main, "checked", lambda args: plan)
    monkeypatch.setattr(main, "backend", fake_backend)
    monkeypatch.setattr(main, "predict", fake_predict)
    monkeypatch.setattr(main.cv.threshold, "predict", fake_predict)
    # base.development rewires these names; restore even when the test fails.
    monkeypatch.setattr(main.base, "checked", main.base.checked)
    monkeypatch.setattr(main.base, "predict", main.base.predict)
    raw = pd.DataFrame(dict(num=np.linspace(-1, 1, 30)))
    ids = np.arange(30, dtype=np.int64)
    def labels(case): return np.linspace(-.5, .5, 30) if case["problem_type"] == "regression" else np.arange(30) % case["num_classes"]
    def fitting(job, case): return raw.copy(), labels(case), ids.copy(), dict(synthetic=True)
    monkeypatch.setattr(main, "raw_union", fitting)
    manifests = []
    for case in cases:
        manifest = main.cv.make_folds(case, labels(case), ids, main.SPLIT_SEED)
        manifest.update(data=dict(synthetic=True), feature_names=["num"])
        assignments = np.asarray(manifest["assignments"])
        for fold in manifest["folds"]:
            fold.update(train_category_schema={}, train_row_ids_sha256=main.support.index_hash(ids[assignments != fold["fold"]]),
                        validation_row_ids_sha256=main.support.index_hash(ids[assignments == fold["fold"]]))
        manifests.append(manifest)
    main.base.write(args.output / "fold_manifest.json", manifests)
    main.fit_job(args)
    first = main.base.read(main.job_path(args, "fold_job", 0))
    assert first["status"] == "ok" and first["reload_bit_exact"] and first["native_fit_completed"]
    with np.load(first["prediction_path"]) as saved:
        assert len(saved["predictions"]) == 6
        np.testing.assert_array_equal(saved["outer_train_positions"], np.flatnonzero(np.asarray(manifests[0]["assignments"]) == 0))
    # The other539 records are synthetic receipts, not fits or measured results.
    records = []
    for i, identity in enumerate(main.support.enumerate_fit_jobs(jobs)):
        record = {**first, **identity, "retained_rounds": identity["requested_rounds"], "validation_error": .5}
        main.base.write(main.job_path(args, "fold_job", i), record); records.append(record)
    main.base.write(args.output / "fold_completion.json", dict(plan_sha256=main.base.digest(args.output / "plan.json"),
        all_fits_successful=True, completion_counts=main.completion_counts(records, 540)))
    main.select(args)
    selected = main.selected_jobs(args, plan)
    assert all(job["rounds"] == 1 for job in selected)
    def dispatch(actual_args, actual_plan, stage, identities):
        assert stage == "union_job"
        output = []
        for i in range(len(identities)):
            worker = copy.copy(actual_args); worker.stage, worker.job_index = stage, i
            main.fit_job(worker)
            output.append(main.base.read(main.job_path(actual_args, stage, i)))
        return output
    monkeypatch.setattr(main, "dispatch", dispatch)
    main.run_union(args)
    frozen = main.base.read(args.output / "frozen_selections.json")
    assert frozen["all_fits_successful"] and len(frozen["selections"]) == 36
    assert frozen["union_completion_counts"]["fully_verified_jobs"] == 36
    accesses = []
    def role(case, name):
        accesses.append(name)
        offset = dict(train=0, stop=100, development=200)[name]
        return dict(X=raw, y=labels(case), outer_train_positions=ids+offset)
    monkeypatch.setattr(main.base, "load_role", role)
    main.development(args)
    report = main.base.read(args.output / "development.json")
    assert len(report["results"]) == 12 and report["official_test_scored"] is False
    assert set(accesses) == {"train", "stop", "development"}
    changed = main.base.read(main.job_path(args, "fold_job", 0)); changed["validation_error"] = .1
    main.base.write(main.job_path(args, "fold_job", 0), changed)
    with pytest.raises(ValueError, match="receipt changed"): main.selected_jobs(args, plan)


@pytest.mark.parametrize("stage,index", [("fold_job", -1), ("fold_job", 540), ("union_job", -1), ("union_job", 36), ("fold_job", None)])
def test_worker_index_is_checked_before_any_fit(stage, index, tmp_path, monkeypatch):
    args = synthetic_args(tmp_path); args.stage, args.job_index = stage, index
    monkeypatch.setattr(main, "checked", lambda args: dict(resources=resources()))
    with pytest.raises(ValueError, match="job index"): main.fit_job(args)


def test_seed_aggregation_is_twelve_task_means_not_twentyfour_independent_tasks():
    cases, _ = panel()
    rows = []
    for case in cases:
        rows.append(dict(dataset=case["dataset_name"], problem_type=case["problem_type"], models={
            lib: dict(errors={"47": .1, "677": .3, "mean": .2}) for lib in main.base.LIBRARIES}))
    summary, passed = seed.aggregate(rows)
    assert passed and all(row["datasets"] == 12 for row in summary if row["family"] == "all")
    assert all(row["datasets"] == 4 for row in summary if row["family"] != "all")
    assert set(row["seed"] for row in summary) == {"47", "677", "mean"}


def test_assessment_marker_closes_admission_and_one_shot_before_data_access(tmp_path, monkeypatch):
    args = synthetic_args(tmp_path)
    main.base.write(args.output / "assessment_started.json", dict(synthetic=True))
    with pytest.raises(ValueError, match="accessed"): main.reject_outer_access(tmp_path)
    monkeypatch.setattr(outer, "checked", lambda args: {})
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("must not open dataset"))
    with pytest.raises(ValueError, match="already started"): outer.assess(args)


def test_split_metadata_rejects_overlap_without_loading_data(tmp_path):
    path = tmp_path / "splits.arff"
    path.write_text("@relation split\n@attribute type string\n@attribute rowid integer\n@attribute repeat integer\n@attribute fold integer\n@data\nTRAIN,0,0,0\nTEST,0,0,0\n")
    with pytest.raises(ValueError, match="overlap"): outer.splits(path)
