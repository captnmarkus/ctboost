"""Synthetic clean-refit protocol integration; never reads real benchmark data."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "benchmarks/fair_accuracy_scalar_numerical_retry.py"
    spec = importlib.util.spec_from_file_location("tested_scalar_refresh_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(module, tmp_path, problem):
    rng = np.random.default_rng(116)
    frame = pd.DataFrame({"x": rng.normal(size=96), "z": rng.normal(size=96),
        "category": ["a", "b", "c"] * 20 + ["STOP_ONLY"] * 12 + ["extra", "a", "b"] * 8})
    target = 2 * frame.x.to_numpy() + frame.z.to_numpy()
    labels = (target > 0).astype(np.int32) if problem == "binary" else target + rng.normal(0, .2, 96)
    dataset = "GiveMeSomeCredit" if problem == "binary" else "diamonds"
    roles = {}
    for name, rows in (("train", np.arange(60)), ("stop", np.arange(60, 72))):
        path = tmp_path / (name + ".pkl")
        with path.open("wb") as stream:
            pickle.dump({"X": frame.iloc[rows].copy(), "y": labels[rows], "outer_train_positions": rows}, stream)
        roles[name] = {"path": str(path), "sha256": module.base.digest(path)}
    receipt = {"synthetic": True}
    expanded = {"X": frame, "y": labels, "outer_train_positions": np.arange(96),
        "verification_frame": frame.iloc[60:72], "receipt": receipt}
    path = tmp_path / "expanded.pkl"
    with path.open("wb") as stream:
        pickle.dump(expanded, stream)
    source = {"dataset": dataset, "path": str(path), "sha256": module.base.digest(path), "receipt": receipt}
    return {"dataset_name": dataset, "problem_type": problem, "num_classes": 2 if problem == "binary" else 0, "roles": roles}, source


def params(library, problem):
    if library == "ctboost":
        return {"iterations": 99, "learning_rate": .1, "max_depth": 3, "min_data_in_leaf": 2,
            "alpha": .5, "lambda_l2": 1, "ordered_ctr": True, "verbose": False, "random_seed": 47, "subsample": .8}
    if library == "xgboost":
        return {"objective": "binary:logistic" if problem == "binary" else "reg:squarederror",
            "tree_method": "hist", "nthread": 1, "max_depth": 3, "eta": .1, "seed": 47, "subsample": .8}
    return {"objective": "binary" if problem == "binary" else "regression", "num_threads": 1,
        "num_leaves": 8, "min_data_in_leaf": 2, "learning_rate": .1, "verbosity": -1,
        "seed": 47, "deterministic": True, "force_col_wise": True, "bagging_fraction": .8, "bagging_freq": 1}


def controls(problem):
    solver = ({"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9, "gradient_gate": 1e-6} if problem == "binary"
              else {"variant": "unscaled_lsqr_10000", "iterations": 10000})
    return {"lambda": 10., "shrinkage": .25, "solver": solver, "max_working_bytes": 4 * 1024**3}


def job_for(module, case, source, library):
    # The opposite-problem correction is deliberately present: BOTH must be removed.
    return {"dataset": case["dataset_name"], "library": library, "parameters": params(library, case["problem_type"]),
        "rounds": 5, "expanded_source": source, "leaf_refit": {"old": True}, "binary_refit": {"old": True},
        "temperature": None, "multiclass_refit": None}


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("problem", ["binary", "regression"])
def test_fresh_anchor_and_fixed_seed_pipeline(tmp_path, monkeypatch, library, problem):
    module = runner()
    case, source = fixture(module, tmp_path, problem)
    job = job_for(module, case, source, library)
    original = copy.deepcopy(job)
    X, labels, schema, train, stop, receipt = module.fitting_data(job, case, train_only=True)
    assert len(labels) == 84 and receipt["outside_cap_rows"] == 24 and receipt["stop_excluded"]
    assert "STOP_ONLY" not in schema["category"] and "extra" in schema["category"]
    clean_job = module.clean_job(job)
    assert clean_job["leaf_refit"] is None and clean_job["binary_refit"] is None
    anchor, _ = module.union.fit_fixed(clean_job, case, X, labels, schema, 47)
    candidate, evidence = module.corrected(anchor, X, labels, controls(problem),
        {"stop": module.base.normalize(stop["X"], schema)[0]})
    assert evidence["checks"]["fitting"]["leaf_routing_exact"]
    assert evidence["independent_sparse_raw_margin_oracle"]
    assert np.isfinite(module.base.error(module.predict(candidate, stop["X"]), stop["y"], problem))
    refreshed = module.refreshed_job(job, problem, controls(problem))
    guard = module.checker()
    monkeypatch.setattr(guard, "verify_clean_refit", lambda *_: pytest.fail("new seed must not compare topology against seed 47"))
    monkeypatch.setattr(module, "checker", lambda: guard)
    final, details, verification = module.fit_policy(refreshed, case, 569)
    assert details["requested_rounds"] == details["retained_rounds"] == 5
    assert not details["early_stopping"] and details["old_scalar_corrections_disabled_before_fit"]
    assert details["parameters"]["random_seed" if library == "ctboost" else "seed"] == 569
    assert details["scalar_fitting_data"]["rows"] == 96
    assert "STOP_ONLY" in final["schema"]["category"]
    np.testing.assert_array_equal(module.predict(final, verification), module.predict(pickle.loads(pickle.dumps(final)), verification))
    # A separately fitted seed-569 clean model plus ONE correction gives the same prediction.
    clean569, _, _ = module.parent.fit_policy(module.clean_job(job), case, 569)
    allX, ally, _, _, _, _ = module.fitting_data(job, case, train_only=False)
    expected, _ = module.corrected(clean569, allX, ally, controls(problem), {})
    np.testing.assert_array_equal(module.predict(final, verification), module.predict(expected, verification))
    assert job == original
    if library == "xgboost":
        clean47, _, _ = module.parent.fit_policy(module.clean_job(job), case, 47)
        assert not np.array_equal(module.predict(clean47, verification), module.predict(clean569, verification))


def test_identity_union_is_fresh_and_clean_not_parent(tmp_path, monkeypatch):
    module = runner()
    case, source = fixture(module, tmp_path, "regression")
    job = job_for(module, case, source, "ctboost")
    clean, _, verification = module.parent.fit_policy(module.clean_job(job), case, 47)
    X, labels, _, _, _, _ = module.fitting_data(job, case, train_only=False)
    old, _ = module.corrected(clean, X, labels, controls("regression"), {})
    old_path = tmp_path / "old.pkl"
    old_path.write_bytes(pickle.dumps(old, protocol=5))
    original = {"dataset": case["dataset_name"], "library": "ctboost", "model_path": str(old_path),
        "model_sha256": module.base.digest(old_path)}
    plan = {"resources": {"cpus": [4]}, "cases": [case], "base_selections": [original], "jobs": [job]}
    monkeypatch.setattr(module, "checked", lambda _: plan)
    monkeypatch.setattr(module, "controls_document", lambda *_: {"choices": [{"dataset": job["dataset"], "library": "ctboost", "correction": None}]})
    real_fit = module.parent.fit_policy
    calls = []
    def fresh(received, *args):
        assert received["leaf_refit"] is None and received["binary_refit"] is None
        calls.append(copy.deepcopy(received))
        return real_fit(received, *args)
    monkeypatch.setattr(module.parent, "fit_policy", fresh)
    monkeypatch.setattr(module, "corrected", lambda *_: pytest.fail("identity does not solve"))
    args = SimpleNamespace(dataset=job["dataset"], library="ctboost", output=tmp_path / "result", artifacts=tmp_path / "artifacts")
    module.transfer_job(args)
    directory = args.output / "transfer_fits" / job["dataset"] / "ctboost"
    result = module.base.read(directory / "job.json")
    assert result["status"] == "ok", result.get("error")
    assert len(calls) == 1 and result["selection"]["model_path"] != str(old_path)
    saved = pickle.loads(Path(result["selection"]["model_path"]).read_bytes())
    np.testing.assert_array_equal(module.predict(saved, verification), module.predict(clean, verification))
    assert not np.array_equal(module.predict(saved, verification), module.predict(old, verification))
    assert module.base.digest(old_path) == original["model_sha256"]
    receipt = module.base.read(directory / "clean_fit.json")
    assert receipt["status"] == "ok" and receipt["old_scalar_corrections_disabled_before_fit"]
    assert len(receipt["clean_fit_controls_sha256"]) == 64


def test_penalty_worker_reuses_anchor_and_checks_two_native_candidates(tmp_path, monkeypatch):
    module = runner()
    case, source = fixture(module, tmp_path, "regression")
    job = job_for(module, case, source, "lightgbm")
    X, labels, schema, train, stop, receipt = module.fitting_data(job, case, train_only=True)
    payload, details = module.union.fit_fixed(module.clean_job(job), case, X, labels, schema, 47)
    model_path = tmp_path / "anchor.pkl"
    model_path.write_bytes(pickle.dumps(payload, protocol=5))
    identity_path = tmp_path / "identity.npz"
    np.savez_compressed(identity_path, predictions=module.predict(payload, stop["X"]),
        labels=stop["y"], row_ids=stop["outer_train_positions"])
    anchor = {"fitting_data": receipt, "fit": details, "model": {"model_path": str(model_path),
        "model_sha256": module.base.digest(model_path)}}
    plan = {"resources": {"cpus": [4], "maximum_named_array_bytes": 4 * 1024**3},
        "jobs": [job], "cases": [case], "solvers": {"regression": controls("regression")["solver"]},
        "source_anchors": [{"dataset": job["dataset"], "library": job["library"], "anchor": anchor,
            "identity_path": str(identity_path)}],
        "grid": [{"lambda": p, "shrinkage": s} for p in (1., 10., 100., 1000.) for s in (.25, 1.)]}
    monkeypatch.setattr(module, "checked", lambda _: plan)
    monkeypatch.setattr(module.union, "fit_fixed", lambda *_: pytest.fail("penalty worker must reuse its anchor"))
    args = SimpleNamespace(dataset=job["dataset"], library=job["library"], penalty_index=0,
        output=tmp_path / "result", artifacts=tmp_path / "artifacts")
    module.penalty_job(args)
    record = module.base.read(args.output / "penalty_fits" / job["dataset"] / job["library"] / "0/job.json")
    assert record["status"] == "ok", record.get("error")
    assert record["new_native_tree_fits"] == 0 and len(record["candidates"]) == 2
    assert record["solve"]["success"] and all(c["status"] == "ok" for c in record["candidates"])
    assert module.base.digest(model_path) == anchor["model"]["model_sha256"]



def test_ties_and_clean_control_provenance():
    module = runner()
    assert module.selection_key(.2, None) < module.selection_key(.2, {"lambda": 1000., "shrinkage": .25})
    assert module.selection_key(.2, {"lambda": 1000., "shrinkage": .25}) < module.selection_key(.2, {"lambda": 10., "shrinkage": .25})
    assert module.selection_key(.2, {"lambda": 10., "shrinkage": .25}) < module.selection_key(.2, {"lambda": 10., "shrinkage": 1.})
    job = {"parameters": {"seed": 47}, "rounds": 5, "leaf_refit": {"old": 1}, "binary_refit": {"old": 2}}
    assert module.clean_provenance(job) == module.clean_provenance({**job, "leaf_refit": None, "binary_refit": None})


def test_seed569_mc_delegation_and_all_four_sources(monkeypatch):
    module = runner()
    received = []
    def record(job, case, seed):
        received.append((copy.deepcopy(job), copy.deepcopy(case), seed))
        return {"sentinel": True}, {}, "frame"
    monkeypatch.setattr(module.parent, "fit_policy", record)
    for dataset in (*module.EXPAND_DATASETS, "churn"):
        problem = "multiclass" if dataset == "SDSS17" else "regression"
        job = {"dataset": dataset, "expanded_source": {"immutable": dataset} if dataset in module.EXPAND_DATASETS else None,
            "rounds": 123, "multiclass_refit": {"keep": True} if problem == "multiclass" else None,
            "binary_refit": None, "leaf_refit": None, "temperature": .75 if problem == "multiclass" else None}
        case = {"dataset_name": dataset, "problem_type": problem}
        module.fit_policy(job, case, 569)
        assert received[-1] == (job, case, 569)


def test_report_contract_and_seed_companions():
    module = runner()
    policy = json.loads(module.POLICY.read_text(encoding="utf-8"))
    assert isinstance(policy["limitations"], str) and isinstance(policy["selection"], str)
    seed = module.load("test_scalar_seed", module.SEED_PATH)
    outer = module.load("test_scalar_outer", module.OUTER_PATH)
    assert seed.BINARY_PATH == Path(module.__file__) and outer.SEED_PATH == module.SEED_PATH
    rows = [{"problem_type": family, "models": {lib: {"errors": {"47": .2, "569": .2, "mean": .2}} for lib in module.base.LIBRARIES}}
            for family in ("binary", "regression", "multiclass") for _ in range(4)]
    summaries, passed = seed.aggregate(rows)
    assert passed and {s["seed"] for s in summaries} == {"47", "569", "mean"}


@pytest.mark.parametrize("kind", ["index", "controls", "dataset", "library", "prediction", "choice", "valid"])
def test_binary_reuse_requires_exact_complete_evidence(tmp_path, kind):
    module = runner()
    grid = [{"lambda": p, "shrinkage": s} for p in (1., 10., 100., 1000.) for s in (.25, 1.)]
    choice = {"dataset": "binary", "library": "xgboost", "correction": None}
    anchor = {"model": {"model_path": "immutable", "model_sha256": "a" * 64}}
    record = {"status": "ok", "choice": choice, "anchor": anchor["model"], "candidates": []}
    for i, control in enumerate(grid):
        path = tmp_path / f"candidate_{i}.npz"
        np.savez_compressed(path, predictions=np.array([i, i+1.], dtype=float))
        candidate = {"index": i, "controls": control, "status": "ok", "predictions_sha256": module.base.digest(path)}
        module.base.write(tmp_path / f"candidate_{i}.json", candidate)
        record["candidates"].append(candidate)
    expected = copy.deepcopy(choice)
    if kind == "index": record["candidates"][-1]["index"] = 0
    if kind == "controls": record["candidates"][0]["controls"] = {"lambda": 999., "shrinkage": .25}
    if kind in ("dataset", "library"): record["candidates"][0][kind] = "wrong"
    if kind == "prediction": (tmp_path / "candidate_0.npz").write_bytes(b"changed")
    if kind == "choice": expected["correction"] = {"lambda": 10.}
    if kind == "valid":
        assert len(module.verify_binary_reuse(record, anchor, expected, tmp_path, grid, "binary", "xgboost")) == 16
    else:
        with pytest.raises(ValueError):
            module.verify_binary_reuse(record, anchor, expected, tmp_path, grid, "binary", "xgboost")


def test_regression_certificate_failure_cannot_fold(monkeypatch):
    module = runner()
    original_load = module.load
    fake = SimpleNamespace(solve=lambda *_args, **_kwargs: (np.array([0.]), {"success": False, "status": 7}))
    monkeypatch.setattr(module, "load", lambda name, path: fake if path == module.NUMERICAL_PATH else original_load(name, path))
    with pytest.raises(ValueError, match="certificate/status failed"):
        module.solve_coefficients("regression", {"Z": "matrix", "baseline": np.zeros(1)},
            np.ones(1), 1., controls("regression")["solver"])


@pytest.mark.parametrize("missing", [False, True])
def test_all_48_penalties_and_reused_binary_grids_form_barrier(tmp_path, monkeypatch, missing):
    module = runner()
    cases = [{"dataset_name": f"{problem}{i}", "problem_type": problem}
             for problem in ("binary", "regression", "multiclass") for i in range(4)]
    jobs = [{"dataset": c["dataset_name"], "library": lib} for c in cases for lib in module.base.LIBRARIES]
    reused = [{"choice": {**j, "correction": None, "stop_error": .5}}
              for j in jobs if j["dataset"].startswith("binary")]
    grid = [{"lambda": p, "shrinkage": s} for p in (1., 10., 100., 1000.) for s in (.25, 1.)]
    plan = {"cases": cases, "jobs": jobs, "reused_binary": reused, "grid": grid,
        "solvers": {"regression": controls("regression")["solver"]},
        "resources": {"maximum_job_seconds": 1200, "maximum_job_rss_bytes": 8 * 1024**3,
            "maximum_named_array_bytes": 4 * 1024**3}}
    args = SimpleNamespace(output=tmp_path / "output", artifacts=tmp_path / "artifacts",
        package_root=tmp_path, dependencies=tmp_path, previous=tmp_path)
    module.base.write(args.output / "plan.json", plan)
    for j in jobs:
        if not j["dataset"].startswith("regression"): continue
        for index in range(4):
            if missing and j["dataset"] == "regression0" and j["library"] == "ctboost" and index == 0: continue
            candidates = [{"index": 2*index+k, "controls": grid[2*index+k], "status": "ok", "stop_error": .4}
                for k in range(2)]
            record = {**j, "penalty_index": index, "status": "ok", "identity_error": .5,
                "anchor": {}, "fit": {}, "fitting_data": {}, "candidates": candidates}
            module.base.write(args.output / "penalty_fits" / j["dataset"] / j["library"] / str(index) / "job.json", record)
    monkeypatch.setattr(module, "checked", lambda _: plan)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module.cv.watchdog, "watch", lambda *_args, **_kwargs:
        {"cleanup_complete": True, "exit_code": 0, "resource_failure": None})
    module.select(args)
    result = module.base.read(args.output / "scalar_controls.json")
    assert result["all_fits_successful"] == (not missing)
    assert len(result["choices"]) == (23 if missing else 24)
    assert result["choices"][:12] == [r["choice"] for r in reused]
    assert result["new_anchor_fits_completed"] == 0 and result["new_regression_solves_planned"] == 48
    if missing:
        with pytest.raises(ValueError): module.controls_document(args, plan)
    else:
        assert all(c["correction"]["lambda"] == 1000. and c["correction"]["shrinkage"] == .25 for c in result["choices"][12:])
        assert len(module.controls_document(args, plan)["choices"]) == 24
