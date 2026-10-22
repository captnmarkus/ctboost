"""Synthetic only: no real cache reads, model fits, prediction scores or timers."""
import copy
import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks"
spec = importlib.util.spec_from_file_location("cap_draft_test_main", DIRECTORY / "fair_accuracy_cap_expansion.py")
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
data = main.data


def fixture(problem="binary"):
    count, cap = 120, 80
    labels = np.arange(count) / 7 if problem == "regression" else np.array(["z", "a", "m"] * 40)
    encoded = labels if problem == "regression" else LabelEncoder().fit_transform(labels)
    selected, _ = train_test_split(np.arange(count), train_size=cap, random_state=47,
                                   stratify=encoded if problem != "regression" else None)
    selected.sort()
    original = {"X": pd.DataFrame({"row_id": np.arange(count),
                                   "category": pd.Categorical(["a", "b", "c"] * 40)}), "y": labels}
    parts = {"train": selected[:50], "stop": selected[50:60], "development": selected[60:70], "confirmation": selected[70:]}
    meta = {name: {"rows": len(ids), "index_sha256": data.index_hash(ids), "sha256": name * 4}
            for name, ids in parts.items()}
    roles = {name: {"X": original["X"].iloc[ids].copy(), "y": encoded[ids], "outer_train_positions": ids.copy()}
             for name, ids in parts.items() if name != "confirmation"}
    case = {"dataset_name": "diamonds" if problem == "regression" else "SDSS17", "problem_type": problem,
            "num_classes": 3, "original_rows": count, "capped_rows": cap, "num_features": 2,
            "task_id": 123, "source_sha256": "a" * 64,
            "roles": {name: copy.deepcopy(meta[name]) for name in roles}}
    return case, original, roles, meta, parts, selected


@pytest.mark.parametrize("problem", ["binary", "multiclass", "regression"])
def test_reconstruction_keeps_exact_prefix_and_excludes_all_held_out_rows(problem):
    case, original, roles, metadata, parts, selected = fixture(problem)
    result = data.expand_rows(case, original, roles, metadata, cap_rows=80)
    additions = np.array(sorted(set(range(120)) - set(selected)))
    expected = np.concatenate([parts["train"], parts["stop"], additions])
    np.testing.assert_array_equal(result["outer_train_positions"], expected)
    np.testing.assert_array_equal(result["X"]["row_id"], expected)
    assert len(result["X"]) == 100
    assert not set(expected) & set(np.concatenate([parts["development"], parts["confirmation"]]))
    assert result["receipt"]["class_vocabulary"] == (None if problem == "regression" else ["a", "m", "z"])


@pytest.mark.parametrize("mutation", ["duplicate", "overlap", "outside_cap", "wrong_order", "negative", "float_ids", "confirmation_hash", "confirmation_count", "wrong_frame", "wrong_labels", "wrong_dtype", "wrong_columns"])
def test_invalid_membership_or_source_values_fail_closed(mutation):
    case, original, roles, metadata, parts, selected = fixture()
    if mutation in ("duplicate", "overlap", "outside_cap", "wrong_order", "negative", "float_ids"):
        ids = roles["train"]["outer_train_positions"]
        if mutation == "duplicate":
            ids[0] = ids[1]
        elif mutation == "overlap":
            ids[0] = parts["stop"][0]
        elif mutation == "outside_cap":
            ids[0] = next(i for i in range(120) if i not in selected)
        elif mutation == "wrong_order":
            ids[:] = ids[::-1]
        elif mutation == "negative":
            ids[0] = -1
        else:
            roles["train"]["outer_train_positions"] = ids.astype(float)
    elif mutation == "confirmation_hash":
        metadata["confirmation"]["index_sha256"] = "f" * 64
    elif mutation == "confirmation_count":
        metadata["confirmation"]["rows"] += 1
    elif mutation == "wrong_frame":
        roles["train"]["X"].iloc[0, 0] += 1
    elif mutation == "wrong_labels":
        roles["stop"]["y"][0] = 99
    elif mutation == "wrong_dtype":
        roles["train"]["X"]["row_id"] = roles["train"]["X"]["row_id"].astype(float)
    else:
        roles["train"]["X"].columns = ["different", "category"]
    with pytest.raises((ValueError, AssertionError)):
        data.expand_rows(case, original, roles, metadata, cap_rows=80)


def test_no_confirmation_payload_or_unapproved_dataset_accepted():
    case, original, roles, metadata, *_ = fixture()
    with pytest.raises(ValueError, match="approved"):
        data.expand_rows(case, original, {**roles, "confirmation": {}}, metadata, cap_rows=80)
    case["dataset_name"] = "houses"
    with pytest.raises(ValueError, match="approved"):
        data.expand_rows(case, original, roles, metadata, cap_rows=80)


def test_prepared_receipt_requires_source_hash_and_no_test_rows():
    case, *_ = fixture()
    receipt = {"dataset": case["dataset_name"], "task_id": case["task_id"], "problem_type": case["problem_type"],
               "contains_outer_test_rows": False, "data_sha256": "a" * 64, "protocol_sha256": "b" * 64,
               "rows_outer_train": 120, "features": 2, "outer_train_indices_sha256": "c" * 64}
    result = data.verify_prepared_receipt(case, receipt, source_hash="a" * 64, protocol_hash="b" * 64)
    assert "length_uint64" in result["outer_train_index_hash_encoding"]
    assert data.index_hash([1, 3]) != data.index_hash([1, 3], length_prefix=True)
    for key, value in (("contains_outer_test_rows", True), ("data_sha256", "d" * 64), ("protocol_sha256", "d" * 64)):
        with pytest.raises(ValueError, match="official-TRAIN"):
            data.verify_prepared_receipt(case, {**receipt, key: value}, source_hash="a" * 64, protocol_hash="b" * 64)


def policy_fixture():
    names = list(data.EXPAND_DATASETS) + ["houses"] + [f"unchanged{i}" for i in range(8)]
    jobs = [{"dataset": name, "library": library, "rounds": 17, "parameters": {"control": 123},
             "leaf_refit": None, "binary_refit": None, "temperature": 1.25}
            for name in names for library in main.LIBRARIES]
    selections = [{"dataset": j["dataset"], "library": j["library"], "cv_selection": {"job": copy.deepcopy(j)},
                   "model_path": f"{j['dataset']}/{j['library']}.pkl", "model_sha256": "a" * 64}
                  for j in jobs]
    return {"cases": [{"dataset_name": name} for name in names]}, {"jobs": jobs, "selections": selections}


def test_nine_fixed_refits_and_twenty_seven_exact_carries():
    plan, frozen = policy_fixture()
    snapshot = copy.deepcopy(frozen)
    refits, carries = main.partition_policy(plan, frozen)
    assert len(refits) == 9 and len(carries) == 27
    assert frozen == snapshot
    assert carries == [s for s in frozen["selections"] if s["dataset"] not in data.EXPAND_DATASETS]
    assert all(j["rounds"] == 17 and j["parameters"] == {"control": 123} and j["temperature"] == 1.25 for j in refits)
    frozen["selections"][0]["cv_selection"]["job"]["rounds"] = 99
    with pytest.raises(ValueError, match="controls"):
        main.partition_policy(plan, frozen)


def gate_fixture():
    hashes = {name: str(i) * 64 for i, name in enumerate(("plan.json", "frozen_selections.json", "development.json", "selected_cv.json", "fold_manifest.json"), 1)}
    plan = {"wave": 13, "seed": 47, "fold_manifest_sha256": hashes["fold_manifest.json"],
            "runtime": {"native": "fixed"}, "scipy_version": "same"}
    frozen = {"all_fits_successful": True, "plan_sha256": hashes["plan.json"], "selected_cv_sha256": hashes["selected_cv.json"]}
    quality = {"plan_sha256": hashes["plan.json"], "frozen_selections_sha256": hashes["frozen_selections.json"], "development_target_met": False}
    return (plan, frozen, quality), hashes


def test_only_failed_initial_or_bound_terminal_gate_can_activate():
    anchor, hashes = gate_fixture()
    assert main.verify_terminal_gate(anchor, hashes) == "wave13_initial_seed47_failure"
    anchor[2]["development_target_met"] = True
    with pytest.raises(ValueError, match="failed terminal"):
        main.verify_terminal_gate(anchor, hashes)
    directory = DIRECTORY / "synthetic-anchor"
    t_hashes = {"plan.json": "a" * 64, "frozen_selections.json": "b" * 64}
    t_plan = {"protocol": "round_cv_union_fixed_seed137_20261016", "seed": 137,
              "runtime": anchor[0]["runtime"], "scipy_version": "same", "script_sha256": "c" * 64,
              "previous": str(directory), "dependencies": {str((directory / key).resolve()): value for key, value in hashes.items()}}
    t_frozen = {"all_fits_successful": True, "plan_sha256": t_hashes["plan.json"]}
    t_quality = {"development_target_met": False, "plan_sha256": t_hashes["plan.json"],
                 "frozen_selections_sha256": t_hashes["frozen_selections.json"],
                 "summary": [{"seed": "mean", "family": "all", "comparator": lib,
                              "median_relative_error_gain": -.01, "wins": 5, "ties": 0}
                             for lib in ("xgboost", "lightgbm")]}
    terminal = t_plan, t_frozen, t_quality, t_hashes
    assert main.verify_terminal_gate(anchor, hashes, terminal, anchor_directory=directory,
                                      terminal_script_hash="c" * 64) == "wave13_linked_seed137_mean_failure"
    t_plan["runtime"] = {}
    with pytest.raises(ValueError, match="linked"):
        main.verify_terminal_gate(anchor, hashes, terminal, anchor_directory=directory, terminal_script_hash="c" * 64)


def test_normalization_and_correction_never_receive_excluded_rows():
    case, original, roles, metadata, parts, _ = fixture()
    expanded = data.expand_rows(case, original, roles, metadata, cap_rows=80)
    excluded = set(np.concatenate([parts["development"], parts["confirmation"]]))
    calls = []
    def normalize(frame, schema=None):
        assert not set(frame["row_id"]) & excluded
        calls.append(("normalize", len(frame)))
        return frame.copy(), {"category": list(frame["category"].dropna().unique())} if schema is None else schema
    def fit_fixed(job, current_case, X, y, schema, seed):
        assert seed == 47 and len(X) == len(y) == 100 and current_case is case
        calls.append(("fit", job["rounds"]))
        return {"library": "ctboost", "problem": "multiclass", "schema": schema}, {"retained_rounds": job["rounds"]}
    def correct(payload, job, X, y, verification):
        assert "temperature" not in payload
        assert not set(X["row_id"]) & excluded and not set(verification["row_id"]) & excluded
        calls.append(("correct", len(X)))
        return payload, {"fixed": True}
    cv = SimpleNamespace(base=SimpleNamespace(normalize=normalize), union=SimpleNamespace(fit_fixed=fit_fixed),
                         threshold=SimpleNamespace(correct=correct))
    job = {"rounds": 17, "temperature": 1.25, "parameters": {"fixed": 1}}
    snapshot = copy.deepcopy(job)
    payload, details, verification = main.fit_expansion(job, case, expanded, 47, cv)
    assert payload["temperature"] == 1.25 and details["correction"] == {"fixed": True}
    assert job == snapshot and calls == [("normalize", 100), ("normalize", 10), ("fit", 17), ("correct", 100)]


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("problem", ["binary", "regression", "multiclass"])
def test_native_expanded_dispatch_and_seed_controls(tmp_path, library, problem):
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(112)
    raw = pd.DataFrame({"value": rng.normal(size=90), "category": pd.Categorical(["a", "b", "c"] * 30)})
    labels = raw["value"].to_numpy() * 2 if problem == "regression" else np.arange(90) % (3 if problem == "multiclass" else 2)
    dataset = {"binary": "GiveMeSomeCredit", "regression": "diamonds", "multiclass": "SDSS17"}[problem]
    case = {"dataset_name": dataset, "problem_type": problem, "num_classes": 3 if problem == "multiclass" else 2}
    expanded = {"X": raw, "y": labels, "verification_frame": raw.iloc[:8], "receipt": {"synthetic": True, "rows": 90}}
    path = tmp_path / "expanded.pkl"
    path.write_bytes(pickle.dumps(expanded, protocol=5))
    source = {"dataset": dataset, "path": str(path), "sha256": main.base.digest(path), "receipt": expanded["receipt"]}
    if library == "ctboost":
        params = {"learning_rate": .1, "max_depth": 3, "min_data_in_leaf": 3, "alpha": .5,
                  "ordered_ctr": True, "verbose": False,
                  "multi_strategy": "multi_output_tree" if problem == "multiclass" else "one_output_per_tree"}
    elif library == "xgboost":
        params = {"objective": {"binary": "binary:logistic", "regression": "reg:squarederror", "multiclass": "multi:softprob"}[problem],
                  "tree_method": "hist", "eta": .1, "max_depth": 3, "nthread": 1}
        if problem == "multiclass":
            params["num_class"] = 3
    else:
        params = {"objective": {"binary": "binary", "regression": "regression", "multiclass": "multiclass"}[problem],
                  "learning_rate": .1, "num_leaves": 7, "min_data_in_leaf": 3, "num_threads": 1,
                  "verbosity": -1, "deterministic": True, "force_col_wise": True}
        if problem == "multiclass":
            params["num_class"] = 3
    job = {"dataset": dataset, "library": library, "parameters": params, "rounds": 4,
           "expanded_source": source, "leaf_refit": None, "binary_refit": None,
           "temperature": .75 if problem == "multiclass" else None}
    if problem != "multiclass":
        job["binary_refit" if problem == "binary" else "leaf_refit"] = {"lambda": 10., "shrinkage": .25,
            "solver": ({"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9, "gradient_gate": 1e-6} if problem == "binary"
                       else {"atol": 1e-8, "btol": 1e-8, "iter_lim": 2000})}
    snapshot = copy.deepcopy(job)
    for seed in (47, 239):
        payload, details, frame = main.fit_policy(job, case, seed)
        assert details["requested_rounds"] == details["retained_rounds"] == 4
        assert details["early_stopping"] is False
        assert details["expanded_data"] == expanded["receipt"]
        predictions = main.predict(payload, frame)
        assert np.isfinite(predictions).all()
        np.testing.assert_array_equal(predictions, main.predict(pickle.loads(pickle.dumps(payload, protocol=5)), frame))
        assert payload.get("temperature") == job["temperature"]
    assert job == snapshot
    job["expanded_source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        main.fit_policy(job, case, 239)



