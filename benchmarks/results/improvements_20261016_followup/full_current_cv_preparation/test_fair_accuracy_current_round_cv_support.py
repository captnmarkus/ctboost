"""Synthetic orchestration checks; no dataset/model loading or native fitting."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("current_round_support", Path(__file__).with_name("fair_accuracy_current_round_cv_support.py"))
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def raw_fixture(dataset="toy", expanded=False):
    categories = ["a", "b", "outside", "unused"]
    def frame(values):
        return pd.DataFrame({"cat": pd.Categorical(values, categories=categories), "num": np.arange(len(values), dtype=float)})
    train = {"X": frame(["a", "b"]), "y": np.array([0., 1.]), "outer_train_positions": np.array([4, 1])}
    stop = {"X": frame(["b"]), "y": np.array([2.]), "outer_train_positions": np.array([6])}
    roles, access, hashes = {"train": train, "stop": stop}, [], {}
    case = {"dataset_name": dataset, "problem_type": "regression", "num_classes": 0, "roles": {}}
    for name, role in roles.items():
        digest = ("a" if name == "train" else "b") * 64
        case["roles"][name] = {"path": name, "sha256": digest, "rows": len(role["y"]),
                               "index_sha256": support.index_hash(role["outer_train_positions"])}
        hashes[name] = digest
    # These roles deliberately cannot be loaded by the callbacks below.
    case["roles"].update(development={"path": "forbidden"}, confirmation={"path": "forbidden"})
    job = {"dataset": dataset, "library": "ctboost", "rounds": 3, "parameters": {}}
    payload = None
    if expanded:
        ids = np.array([4, 1, 6, 7, 9])
        receipt = {"rows": 5, "added_rows": 2, "additional_positions_sha256": support.index_hash(ids[3:]),
                   ("authorized_rows_sha256" if dataset == "houses" else "expanded_positions_sha256"): support.index_hash(ids)}
        raw = pd.concat([train["X"], stop["X"], frame(["outside", "a"])], ignore_index=True)
        payload = {"X": raw, "y": np.array([0., 1., 2., 3., 4.]), "outer_train_positions": ids, "receipt": receipt}
        job["expanded_source"] = {"dataset": dataset, "path": "expanded", "sha256": "c" * 64,
                                  "receipt": copy.deepcopy(receipt)}
        hashes["expanded"] = "c" * 64
    def load_role(case, name):
        access.append(name)
        return roles[name]
    def load_expanded(source):
        access.append("expanded")
        return payload
    return SimpleNamespace(case=case, job=job, roles=roles, expanded=payload, access=access, hashes=hashes,
                           callbacks=dict(load_role=load_role, load_expanded=load_expanded, digest=hashes.__getitem__))


@pytest.mark.parametrize("dataset", ["toy", *sorted(support.EXPANDED_DATASETS)])
def test_reconstructs_only_authorized_raw_rows_without_normalizing(dataset):
    fixture = raw_fixture(dataset, dataset != "toy")
    raw, labels, ids, receipt = support.reconstruct_raw_union(fixture.case, fixture.job, **fixture.callbacks)
    assert fixture.access == (["train", "stop", "expanded"] if dataset != "toy" else ["train", "stop"])
    np.testing.assert_array_equal(ids, [4, 1, 6, 7, 9] if dataset != "toy" else [4, 1, 6])
    np.testing.assert_array_equal(labels, np.arange(len(ids)))
    assert list(raw.cat.cat.categories) == ["a", "b", "outside", "unused"]
    assert not receipt["normalization_performed"]


@pytest.mark.parametrize("change", ["source_hash", "role_hash", "role_ids", "prefix_value", "prefix_label",
                                    "authorized_id", "order", "duplicate", "receipt", "feature_order"])
def test_reconstruction_rejects_changed_source_or_original_prefix(change):
    fixture = raw_fixture("diamonds", True)
    if change == "source_hash":
        fixture.hashes["expanded"] = "d" * 64
    elif change == "role_hash":
        fixture.hashes["train"] = "d" * 64
    elif change == "role_ids":
        fixture.roles["train"]["outer_train_positions"][0] = 3
    elif change == "prefix_value":
        fixture.expanded["X"].loc[0, "num"] = 42
    elif change == "prefix_label":
        fixture.expanded["y"][0] = 42
    elif change == "authorized_id":
        fixture.expanded["outer_train_positions"][-1] = 10  # excluded ID cannot replace an authorized one
    elif change == "order":
        fixture.expanded["outer_train_positions"][-2:] = [9, 7]
    elif change == "duplicate":
        fixture.expanded["outer_train_positions"][-1] = 1
    elif change == "receipt":
        fixture.expanded["receipt"]["added_rows"] = 3
    else:
        fixture.expanded["X"] = fixture.expanded["X"][["num", "cat"]]
    with pytest.raises((ValueError, AssertionError)):
        support.reconstruct_raw_union(fixture.case, fixture.job, **fixture.callbacks)
    if change == "source_hash":
        assert fixture.access == ["train", "stop"]
    if change == "role_hash":
        assert fixture.access == []


def test_source_order_check_is_independent_of_authorized_hash():
    fixture = raw_fixture("houses", True)
    ids = fixture.expanded["outer_train_positions"]
    ids[-2:] = [9, 7]
    for receipt in (fixture.expanded["receipt"], fixture.job["expanded_source"]["receipt"]):
        receipt["authorized_rows_sha256"] = support.index_hash(ids)
        receipt["additional_positions_sha256"] = support.index_hash(ids[3:])
    with pytest.raises(ValueError, match="ascending"):
        support.reconstruct_raw_union(fixture.case, fixture.job, **fixture.callbacks)


def test_expanded_policy_cannot_silently_revert_to_the_old_cap():
    fixture = raw_fixture("diamonds", True)
    del fixture.job["expanded_source"]
    with pytest.raises(ValueError, match="old capped union"):
        support.reconstruct_raw_union(fixture.case, fixture.job, **fixture.callbacks)


def all_jobs():
    return [{"dataset": f"task{task}", "library": library, "rounds": 1 if task == 0 else 9,
             "parameters": {"learning_rate": .05}, "leaf_refit": {"lambda": 100}, "temperature": None}
            for task in range(12) for library in support.LIBRARIES]


def complete_records(jobs):
    return [{**identity, "status": "ok", "reload_bit_exact": True,
             "retained_rounds": identity["requested_rounds"], "model_sha256": "a" * 64,
             "prediction_sha256": "b" * 64, "validation_error": .5}
            for identity in support.enumerate_fit_jobs(jobs)]


def test_540_distinct_fit_identities_and_duplicate_profile_tie_rule():
    jobs = all_jobs()
    before = copy.deepcopy(jobs)
    identities = support.enumerate_fit_jobs(jobs)
    assert len(identities) == 540
    assert [identities[i]["requested_rounds"] for i in (0, 5, 10)] == [1, 1, 2]
    choices = support.choose_rounds(jobs, complete_records(jobs))
    assert len(choices) == 36
    assert choices[0]["best"]["profile_index"] == 0
    assert choices[3]["job"]["rounds"] == 5
    assert choices[3]["job"]["parameters"] == before[3]["parameters"]
    assert choices[3]["job"]["leaf_refit"] == before[3]["leaf_refit"]
    assert jobs == before


def test_selection_uses_equal_fold_mean_not_pooled_or_row_weights():
    jobs = all_jobs()
    records = complete_records(jobs)
    for record in records[:15]:
        record["validation_rows"] = 10000 if record["fold"] == 0 else 1
        record["validation_error"] = ([0., 1., 1., 1., 1.] if record["profile_index"] == 0 else [.7] * 5)[record["fold"]]
    choice = support.choose_rounds(jobs, records)[0]
    assert choice["best"]["profile_index"] == 1
    assert choice["scores"][0]["mean_fold_error"] == .8


@pytest.mark.parametrize("failure", ["missing", "duplicate", "seed", "rounds", "retained", "failed", "reload", "nan", "hash"])
def test_selection_requires_complete_successful_frozen_identities(failure):
    jobs, records = all_jobs(), complete_records(all_jobs())
    if failure == "missing":
        records.pop()
    elif failure == "duplicate":
        records[-1] = records[0].copy()
    else:
        field, value = {"seed": ("seed", 947), "rounds": ("requested_rounds", 100), "retained": ("retained_rounds", 0),
                        "failed": ("status", "failed"), "reload": ("reload_bit_exact", False),
                        "nan": ("validation_error", float("nan")), "hash": ("prediction_sha256", "missing")}[failure]
        records[0][field] = value
    with pytest.raises(ValueError):
        support.choose_rounds(jobs, records)


def backend_fixture(problem="regression", correction=True):
    calls = []
    train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a"], categories=["a", "b", "heldout", "unused"]),
                          "num": [0., 1., 2.]})
    valid = pd.DataFrame({"cat": pd.Categorical(["heldout"], categories=train.cat.cat.categories), "num": [8.]})
    labels = np.array([0., 1., 2.] if problem != "binary" else [0., 1., 0.])
    case = {"dataset_name": "toy", "problem_type": problem, "num_classes": 3 if problem == "multiclass" else 2}
    job = {"dataset": "toy", "library": "ctboost", "parameters": {"learning_rate": .1}, "rounds": 7,
           "temperature": 2. if problem == "multiclass" else None,
           **{field: None for field in support.CORRECTION_FIELDS}}
    if correction:
        job[support.CORRECTION_FIELD[problem]] = {"lambda": 100., "shrinkage": .25, "solver": {"unbound_test_adapter": True}}
    def normalize(frame, schema=None):
        calls.append(("normalize_fit" if schema is None else "normalize_apply", frame.cat.tolist()))
        if schema is None:
            schema = {"cat": sorted(frame.cat.dropna().unique().tolist())}
        result = frame.copy()
        result["cat"] = pd.Categorical(frame.cat, categories=schema["cat"])
        return result, schema
    def fit_clean(clean, case, X, y, schema, seed):
        calls.append(("fit", X.cat.tolist(), y.tolist(), seed))
        assert all(clean.get(field) is None for field in (*support.CORRECTION_FIELDS, "temperature"))
        return {"library": clean["library"], "problem": problem, "schema": schema, "model": {"offset": 0.}}, {
            "requested_rounds": clean["rounds"], "retained_rounds": clean["rounds"], "early_stopping": False}
    def apply_fixed_correction(payload, actual_problem, controls, X, y, frames):
        calls.append(("correct", X.cat.tolist(), y.tolist(), controls.copy()))
        assert actual_problem == problem
        assert payload.get("temperature") == job["temperature"]  # before the fixed-T softmax solve
        assert list(frames) == ["validation"] and frames["validation"].cat.isna().all()
        # A spy for ordered replay receives precisely this fold-training frame.
        calls.append(("ordered_replay", X.cat.tolist(), y.tolist()))
        result = copy.deepcopy(payload)
        result["model"]["offset"] = 1.
        return result, {"synthetic_adapter_only": True}
    def predict(payload, frame):
        calls.append(("predict", frame.cat.tolist()))
        raw = np.full((len(frame), case["num_classes"]) if problem == "multiclass" else len(frame),
                      4. + payload["model"]["offset"])
        return raw / payload["temperature"] if payload.get("temperature") is not None else raw
    backend = SimpleNamespace(normalize=normalize, fit_clean=fit_clean, apply_fixed_correction=apply_fixed_correction, predict=predict)
    return SimpleNamespace(train=train, valid=valid, labels=labels, case=case, job=job, calls=calls, backend=backend)


@pytest.mark.parametrize("problem", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("correction", [False, True])
def test_fold_local_vocabulary_labels_ordered_replay_and_single_temperature(problem, correction):
    fixture = backend_fixture(problem, correction)
    original = copy.deepcopy(fixture.job)
    payload, predictions, receipt = support.fit_fold(fixture.job, fixture.case, fixture.train, fixture.labels,
                                                    fixture.valid, 47, fixture.backend)
    assert receipt["fold_category_schema"] == {"cat": ["a", "b"]}
    assert receipt["validation_labels_received"] is False and receipt["reload_bit_exact"]
    assert [call[0] for call in fixture.calls].count("correct") == int(correction)
    assert [call[0] for call in fixture.calls].count("ordered_replay") == int(correction)
    assert fixture.calls[2] == ("fit", ["a", "b", "a"], fixture.labels.tolist(), 47)
    np.testing.assert_array_equal(predictions, np.full(predictions.shape, (5. if correction else 4.) / (2. if problem == "multiclass" else 1.)))
    assert fixture.job == original


@pytest.mark.parametrize("defect", ["leaky_schema", "temperature_during_fit", "changed_temperature", "changed_schema",
                                   "stopping", "wrong_correction", "wrong_rounds", "reload"])
def test_fold_rejects_backend_or_control_contract_violations(defect):
    fixture = backend_fixture("multiclass")
    backend = fixture.backend
    if defect == "leaky_schema":
        original = backend.normalize
        backend.normalize = lambda frame, schema=None: original(frame, {"cat": ["a", "b", "heldout"]} if schema is None else schema)
    elif defect == "temperature_during_fit":
        original = backend.fit_clean
        def fit(*args):
            payload, details = original(*args)
            payload["temperature"] = 2.
            return payload, details
        backend.fit_clean = fit
    elif defect in ("changed_temperature", "changed_schema"):
        original = backend.apply_fixed_correction
        def correct(*args):
            payload, receipt = original(*args)
            payload["temperature" if defect == "changed_temperature" else "schema"] = 4. if defect == "changed_temperature" else {}
            return payload, receipt
        backend.apply_fixed_correction = correct
    elif defect == "stopping":
        fixture.job["parameters"]["early_stopping_rounds"] = 2
    elif defect == "wrong_correction":
        fixture.job["leaf_refit"] = {"lambda": 1}
    elif defect == "wrong_rounds":
        original = backend.fit_clean
        def fit(*args):
            payload, details = original(*args)
            details["retained_rounds"] = 3
            return payload, details
        backend.fit_clean = fit
    else:
        original, count = backend.predict, [0]
        def predict(*args):
            count[0] += 1
            return original(*args) + (1 if count[0] == 2 else 0)
        backend.predict = predict
    with pytest.raises(ValueError):
        support.fit_fold(fixture.job, fixture.case, fixture.train, fixture.labels, fixture.valid, 47, backend)


@pytest.mark.parametrize("defect", ["missing_class", "empty_train", "empty_validation", "seed"])
def test_fold_rejects_unsupported_membership_before_fit(defect):
    fixture = backend_fixture("multiclass")
    seed = 47
    if defect == "missing_class":
        fixture.labels[-1] = 1
    elif defect == "empty_train":
        fixture.train, fixture.labels = fixture.train.iloc[:0], fixture.labels[:0]
    elif defect == "empty_validation":
        fixture.valid = fixture.valid.iloc[:0]
    else:
        seed = True
    with pytest.raises((ValueError, AssertionError)):
        support.fit_fold(fixture.job, fixture.case, fixture.train, fixture.labels, fixture.valid, seed, fixture.backend)
    assert not fixture.calls
