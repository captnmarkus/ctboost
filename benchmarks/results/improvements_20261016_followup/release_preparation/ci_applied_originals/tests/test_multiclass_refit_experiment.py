"""Synthetic runner integration; no real study data are loaded."""
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
    path = Path(__file__).resolve().parents[1] / "benchmarks/fair_accuracy_multiclass_refit.py"
    spec = importlib.util.spec_from_file_location("tested_multiclass_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(module, tmp_path):
    labels = np.tile(np.arange(3), 24)
    frame = pd.DataFrame({"number": labels + np.sin(np.arange(72)) * .2,
        "category": ["a", "b", "c"] * 16 + ["STOP_ONLY"] * 12 + ["extra", "a", "b"] * 4})
    roles = {}
    for name, rows in (("train", np.arange(48)), ("stop", np.arange(48, 60))):
        path = tmp_path / (name + ".pkl")
        with path.open("wb") as stream:
            pickle.dump({"X": frame.iloc[rows].copy(), "y": labels[rows], "outer_train_positions": rows}, stream)
        roles[name] = {"path": str(path), "sha256": module.base.digest(path)}
    receipt = {"synthetic": True}
    expanded = {"X": frame, "y": labels, "outer_train_positions": np.arange(72),
        "verification_frame": frame.iloc[48:60], "receipt": receipt}
    path = tmp_path / "expanded.pkl"
    with path.open("wb") as stream:
        pickle.dump(expanded, stream)
    source = {"dataset": "SDSS17", "path": str(path), "sha256": module.base.digest(path), "receipt": receipt}
    return {"dataset_name": "SDSS17", "problem_type": "multiclass", "num_classes": 3, "roles": roles}, source


def params(library, vector):
    if library == "ctboost":
        return {"iterations": 99, "learning_rate": .1, "max_depth": 2, "min_data_in_leaf": 2,
            "alpha": .5, "lambda_l2": 1, "ordered_ctr": True, "verbose": False, "random_seed": 47,
            "multi_strategy": "multi_output_tree" if vector else "one_output_per_tree"}
    if library == "xgboost":
        return {"objective": "multi:softprob", "num_class": 3, "tree_method": "hist", "nthread": 1,
            "max_depth": 2, "eta": .1, "seed": 47, "multi_strategy": "multi_output_tree" if vector else "one_output_per_tree"}
    return {"objective": "multiclass", "num_class": 3, "num_threads": 1, "num_leaves": 4,
        "min_data_in_leaf": 2, "learning_rate": .1, "verbosity": -1, "seed": 47, "deterministic": True, "force_col_wise": True}


def controls():
    return {"lambda": 10., "shrinkage": .25, "solver": {"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9,
        "gradient_gate": 1e-6}, "max_working_bytes": 2 * 1024**3}


@pytest.mark.parametrize("library,vector", [("ctboost", False), ("ctboost", True), ("xgboost", False), ("xgboost", True), ("lightgbm", False)])
def test_anchor_stop_and_seed_transfer_pipeline(tmp_path, library, vector):
    module = runner()
    case, source = fixture(module, tmp_path)
    job = {"dataset": "SDSS17", "library": library, "parameters": params(library, vector), "rounds": 4,
        "expanded_source": source, "leaf_refit": None, "binary_refit": None, "temperature": .75,
        "multiclass_refit": controls()}
    snapshot = copy.deepcopy(job)
    X, labels, schema, train, stop, receipt = module.fitting_data(job, case, train_only=True)
    assert len(labels) == 60 and receipt["outside_cap_rows"] == 12 and receipt["stop_excluded"]
    assert "STOP_ONLY" not in schema["category"] and "extra" in schema["category"]
    anchor, fit = module.union.fit_fixed(job, case, X, labels, schema, 47)
    anchor["temperature"] = .75
    normalized_stop = module.base.normalize(stop["X"], schema)[0]
    candidate, evidence = module.corrected(anchor, X, labels, controls(), {"stop": normalized_stop})
    assert evidence["checks"]["fitting"]["leaf_routing_exact"]
    assert evidence["checks"]["stop"]["temperature_applied_once"]
    assert np.isfinite(module.base.error(module.predict(candidate, stop["X"]), stop["y"], "multiclass"))
    final, details, verification = module.fit_policy(job, case, 457)
    assert details["requested_rounds"] == details["retained_rounds"] == 4
    assert not details["early_stopping"]
    assert details["parameters"]["random_seed" if library == "ctboost" else "seed"] == 457
    assert details["multiclass_fitting_data"]["rows"] == 72
    assert "STOP_ONLY" in final["schema"]["category"]
    assert final["temperature"] == .75
    np.testing.assert_array_equal(module.predict(final, verification), module.predict(pickle.loads(pickle.dumps(final)), verification))
    assert job == snapshot


def test_identity_and_nonzero_tie_order():
    module = runner()
    assert module.selection_key(.2, None) < module.selection_key(.2, {"lambda": 1000., "shrinkage": .25})
    assert module.selection_key(.2, {"lambda": 1000., "shrinkage": .25}) < module.selection_key(.2, {"lambda": 10., "shrinkage": .25})
    assert module.selection_key(.2, {"lambda": 10., "shrinkage": .25}) < module.selection_key(.2, {"lambda": 10., "shrinkage": 1.})


def test_identity_transfer_keeps_parent_artifact(tmp_path, monkeypatch):
    module = runner()
    original = {"dataset": "toy", "library": "ctboost", "model_path": "unchanged.pkl", "model_sha256": "f" * 64, "model_bytes": 123}
    job = {"dataset": "toy", "library": "ctboost"}
    choice = {"dataset": "toy", "library": "ctboost", "multiclass_refit": None}
    plan = {"resources": {"cpus": [4]}, "cases": [{"dataset_name": "toy", "problem_type": "multiclass"}],
        "base_selections": [original], "jobs": [job]}
    monkeypatch.setattr(module, "checked", lambda _: plan)
    monkeypatch.setattr(module, "controls_document", lambda *_: {"choices": [choice]})
    monkeypatch.setattr(module, "corrected", lambda *_: pytest.fail("identity must not solve or fold"))
    args = SimpleNamespace(dataset="toy", library="ctboost", output=tmp_path, artifacts=tmp_path / "artifacts")
    module.transfer_job(args)
    result = module.base.read(tmp_path / "transfer_fits/toy/ctboost/job.json")
    assert result["status"] == "ok"
    assert {key: result["selection"][key] for key in original} == original


def test_all_expanded_sources_and_other_controls_are_forwarded_for_fresh_seed(monkeypatch):
    module = runner()
    received = []
    def record(job, case, seed):
        received.append((copy.deepcopy(job), case["dataset_name"], seed))
        return "payload", "details", "frame"
    monkeypatch.setattr(module.parent, "fit_policy", record)
    for dataset in (*module.EXPAND_DATASETS, "churn"):
        job = {"dataset": dataset, "expanded_source": {"immutable": dataset} if dataset in module.EXPAND_DATASETS else None,
            "rounds": 123, "multiclass_refit": None, "binary_refit": {"preserved": True}, "temperature": .75}
        assert module.fit_policy(job, {"dataset_name": dataset}, 457) == ("payload", "details", "frame")
        assert received[-1] == (job, dataset, 457)


def test_frozen_report_contract_and_seed_companions():
    module = runner()
    policy = json.loads(module.POLICY.read_text(encoding="utf-8"))
    assert isinstance(policy["limitations"], str) and isinstance(policy["selection"], str)
    seed = module.load("test_multiclass_seed", module.SEED_PATH)
    outer = module.load("test_multiclass_outer", module.OUTER_PATH)
    assert seed.BINARY_PATH == Path(module.__file__)
    assert outer.SEED_PATH == module.SEED_PATH
    rows = [{"problem_type": family, "models": {lib: {"errors": {"47": .2, "457": .2, "mean": .2}} for lib in module.base.LIBRARIES}}
            for family in ("binary", "regression", "multiclass") for _ in range(4)]
    summaries, passed = seed.aggregate(rows)
    assert passed and {s["seed"] for s in summaries} == {"47", "457", "mean"}
