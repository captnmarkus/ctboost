"""Inactive scalar-regression absolute-leaf experiment; not a training API.

The fixed physical-leaf basis includes unobserved leaves and duplicates. Solve
0.5*||y-b-Zw||^2 + 0.5*lambda*||w||^2 with a free intercept, then serve
(1-q)*F0 + q*(b+Zw) by replacing native outputs. Native rounding is checked
separately. A caller must freeze reference provenance, solver settings, resources
and selection policy before any real use. This module has no fitting or CLI.
Historical statistics remain unchanged and stale: artifacts are inference only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import re

import numpy as np

from benchmarks import fixed_leaf_refit as legacy

INACTIVE = True
FIXED_RTOL = 5e-5
FIXED_ATOL = 1e-5


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=True).encode()).hexdigest()


def _float32(value):
    value = float(value)
    if not math.isfinite(value) or abs(value) > np.finfo(np.float32).max:
        raise ValueError("Nonfinite or overflowing native float32 output")
    return float(np.float32(value))


def _finite_vector(values, size, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite {name} with the declared size")
    return values


def _walk_named(value, name):
    if isinstance(value, dict):
        if name in value:
            yield value[name]
        for child in value.values():
            yield from _walk_named(child, name)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_named(child, name)


def _nonzero_constraint(value):
    if isinstance(value, str):
        value = value.strip().strip("()[]")
        return bool(value) and any(float(part) != 0 for part in re.split(r"[, ]+", value))
    return any(_nonzero_constraint(item) if isinstance(item, (str, list)) else float(item) != 0
               for item in value)


def _xgb_guard(document, config):
    learner = document["learner"]
    booster = learner["gradient_booster"]
    parameters = learner["learner_model_param"]
    if (learner["objective"]["name"] != "reg:squarederror" or booster["name"] != "gbtree"
            or int(parameters["num_class"]) != 0 or int(parameters.get("num_target", 1)) != 1):
        raise ValueError("Expected ordinary scalar squared-error XGBoost")
    # 3.4.1 can serialize DART with name=gbtree; inspect effective weights too.
    if any(len(values) for values in _walk_named(booster, "weight_drop")):
        raise ValueError("Weighted/DART XGBoost is unsupported")
    for controls in _walk_named(config, "dart_train_param"):
        if float(controls.get("rate_drop", 0)) != 0 or int(controls.get("one_drop", 0)) != 0:
            raise ValueError("Active XGBoost dropout is unsupported")
    for key in ("monotone_constraints",):
        if any(_nonzero_constraint(value) for value in _walk_named(config, key)):
            raise ValueError("Constrained XGBoost is unsupported")
    if any(str(value).strip() not in ("", "[]") for value in _walk_named(config, "interaction_constraints")):
        raise ValueError("Constrained XGBoost is unsupported")
    if any(float(value) != 0 for value in _walk_named(config, "max_delta_step")):
        raise ValueError("Constrained XGBoost leaf outputs are unsupported")
    if any(int(value) != 1 for value in _walk_named(config, "num_parallel_tree")):
        raise ValueError("Parallel-tree/RF XGBoost is unsupported")
    if any(str(value) != "cpu" for value in _walk_named(config, "device")):
        raise ValueError("This experiment requires CPU prediction")
    trees = booster["model"]["trees"]
    if len(booster["model"]["tree_info"]) != len(trees) or any(booster["model"]["tree_info"]):
        raise ValueError("Invalid scalar XGBoost tree groups")
    for tree in trees:
        if (int(tree["tree_param"]["size_leaf_vector"]) > 1
                or int(tree["tree_param"].get("num_deleted", 0)) != 0):
            raise ValueError("Vector/deleted XGBoost nodes are unsupported")


def _lgb_parameters(text):
    # Model text retains native controls after reload, unlike Python .params.
    result = {}
    for name, value in re.findall(r"^\[([^:\]\r\n]+): ([^\r\n]*)\]\r?$", text, flags=re.MULTILINE):
        if name in result:
            raise ValueError("Duplicate native LightGBM parameter")
        result[name] = value.strip()
    return result


def _lgb_guard(document, parameters):
    if (document["objective"].split()[0] != "regression"
            or int(document["num_tree_per_iteration"]) != 1
            or int(document["num_class"]) != 1 or document.get("average_output", False)):
        raise ValueError("Expected ordinary scalar squared-error LightGBM")
    if parameters.get("boosting") != "gbdt" or parameters.get("linear_tree") != "0":
        raise ValueError("Missing/unsupported LightGBM boosting or linear-tree controls")
    if (_nonzero_constraint(parameters.get("monotone_constraints", ""))
            or parameters.get("interaction_constraints", "") not in ("", "[]")
            or float(parameters.get("max_delta_step", 0)) != 0
            or parameters.get("pred_early_stop", "0") != "0"
            or parameters.get("device_type", "cpu") != "cpu"
            or parameters.get("reg_sqrt", "0") != "0"):
        raise ValueError("Unsupported constrained/nonidentity LightGBM prediction")
    for tree in document["tree_info"]:
        if any(bool(value) for value in _walk_named(tree, "is_linear")):
            raise ValueError("Linear LightGBM leaves are unsupported")


def _nonoutput_document(library, document):
    if library == "ctboost":
        result = legacy._ct_topology(document)
        result["base_score"] = [None]
    elif library == "xgboost":
        result = legacy._xgb_topology(document)
        result["learner"]["learner_model_param"]["base_score"] = None
    else:
        result = legacy._lgb_topology(document)
    return result


def _python_state(payload):
    # Preserve ordered schema and all prior correction/provenance metadata too.
    result = [(key, value) for key, value in payload.items()
              if key not in ("model", "fixed_absolute_leaf_refit")]
    model = payload["model"]
    if payload["library"] == "ctboost":
        pipeline = model._feature_pipeline
        result += [("native_pipeline", None if pipeline is None else pipeline.to_state()),
                   ("data_schema", model.data_schema), ("training_metadata", model._training_metadata)]
    return _hash(result)


def extract_native(payload):
    """Extract every physical leaf in prediction units, without reading data."""
    if payload.get("problem") != "regression" or payload.get("temperature") is not None:
        raise ValueError("Only scalar identity-link regression without temperature is supported")
    if "fixed_absolute_leaf_refit" in payload:
        raise ValueError("Absolute refit cannot be applied twice")
    model, library = payload["model"], payload["library"]
    if type(payload.get("schema")) is not dict:
        raise ValueError("An explicit ordered schema is required")
    maps, outputs, rates = [], [], []
    if library == "ctboost":
        document = model._handle.export_state()
        if (document["objective_name"].lower() != "rmse" or model.prediction_dimension != 1
                or document["boosting_type"] != "GradientBoosting" or document["task_type"] != "CPU"
                or any(document["monotone_constraints"]) or document["interaction_constraints"]
                or document["max_leaf_weight"] != 0):
            raise ValueError("Unsupported CTBoost objective, constraints or prediction mode")
        base = _finite_vector(document["base_score"], 1, "CTBoost bias")
        bias = _float32(base[0])
        schedule = document["tree_learning_rates"]
        if schedule and len(schedule) != len(document["trees"]):
            raise ValueError("Invalid scalar CTBoost learning-rate schedule")
        for index, tree in enumerate(document["trees"]):
            rate = _float32(schedule[index] if schedule else document["learning_rate"])
            if rate < np.finfo(np.float32).tiny:
                raise ValueError("Zero/subnormal/negative CTBoost rate is unsupported")
            ids = np.array([i for i, node in enumerate(tree["nodes"]) if node["is_leaf"]], dtype=np.int64)
            if any(tree["nodes"][i].get("leaf_weights") for i in ids):
                raise ValueError("Vector leaves are unsupported")
            values = [_float32(rate * _float32(tree["nodes"][i]["leaf_weight"])) for i in ids]
            maps.append(ids); outputs.extend(values); rates.append(rate)
    elif library == "xgboost":
        import xgboost as xgb
        if xgb.__version__ != "3.4.1":
            raise ValueError("This inactive helper requires XGBoost3.4.1")
        document = legacy._xgb_document(model)
        _xgb_guard(document, json.loads(model.save_config()))
        encoded = document["learner"]["learner_model_param"]["base_score"]
        if not isinstance(encoded, str):
            raise ValueError("Unsupported XGBoost base-score encoding")
        base = _finite_vector(np.atleast_1d(json.loads(encoded)), 1, "XGBoost bias")
        bias = _float32(base[0])
        for tree in document["learner"]["gradient_booster"]["model"]["trees"]:
            ids = np.array([i for i, child in enumerate(tree["left_children"]) if child == -1], dtype=np.int64)
            maps.append(ids); outputs.extend(_float32(tree["split_conditions"][i]) for i in ids)
    elif library == "lightgbm":
        import lightgbm as lgb
        if lgb.__version__ != "4.7.0":
            raise ValueError("This inactive helper requires LightGBM4.7.0")
        document = model.dump_model(num_iteration=-1)
        _lgb_guard(document, _lgb_parameters(model.model_to_string(num_iteration=-1)))
        if 0 < model.best_iteration < len(document["tree_info"]):
            raise ValueError("Default partial-iteration LightGBM prediction is unsupported")
        bias = 0.0  # Native initial constant is already present in effective leaves.
        for tree, info in enumerate(document["tree_info"]):
            ids = np.arange(info["num_leaves"], dtype=np.int64)
            maps.append(ids); outputs.extend(model.get_leaf_output(tree, int(i)) for i in ids)
    else:
        raise ValueError("Unknown library")
    if not maps or any(len(ids) == 0 for ids in maps):
        raise ValueError("At least one nonempty native tree is required")
    theta = _finite_vector(outputs, sum(map(len, maps)), "native leaf outputs")
    return {"library": library, "bias": bias, "theta": theta,
            "leaf_ids_by_tree": maps, "rates": rates, "document": document,
            "native_sha256": _hash(document), "python_state_sha256": _python_state(payload),
            "nonoutput_sha256": _hash(_nonoutput_document(library, document))}


def _input_frame(frame):
    import pandas as pd
    if type(frame) is not pd.DataFrame or len(frame) == 0:
        raise ValueError("An explicit nonempty normalized DataFrame is required; no row baselines")


def predict(payload, frame, training_prepared=None):
    _input_frame(frame)
    if payload["library"] == "lightgbm":
        values = payload["model"].predict(frame, raw_score=True, num_iteration=-1, num_threads=1)
        return _finite_vector(values, len(frame), "predictions")
    return legacy.predict(payload, frame, training_prepared)


def leaf_indices(payload, frame, training_prepared=None):
    _input_frame(frame)
    if payload["library"] == "lightgbm":
        values = np.asarray(payload["model"].predict(frame, pred_leaf=True, num_iteration=-1, num_threads=1))
        if values.ndim == 1:
            values = values[:, None]
        if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Invalid native leaf IDs")
        return values.astype(np.int64)
    return legacy.leaf_indices(payload, frame, training_prepared)


def indicator_matrix(leaves, maps):
    """Canonical CSR with every native leaf column, including zero columns."""
    from scipy.sparse import csr_matrix
    leaves = np.asarray(leaves)
    if (leaves.ndim != 2 or not leaves.shape[0] or leaves.shape[1] != len(maps)
            or not np.isfinite(leaves).all() or (leaves < 0).any()
            or not np.equal(leaves, np.floor(leaves)).all()):
        raise ValueError("Invalid leaf-matrix shape")
    columns = sum(len(ids) for ids in maps)
    indices = np.empty(leaves.shape, dtype=np.int32 if columns < 2**31 else np.int64)
    offset = 0
    for tree, ids in enumerate(maps):
        ids = np.asarray(ids)
        if (ids.ndim != 1 or not len(ids) or not np.isfinite(ids).all()
                or (ids < 0).any() or not np.equal(ids, np.floor(ids)).all()
                or not np.all(ids[1:] > ids[:-1])):
            raise ValueError("Leaf maps must be sorted unique nonnegative IDs")
        positions = np.searchsorted(ids, leaves[:, tree])
        if (positions >= len(ids)).any() or not np.array_equal(ids[positions], leaves[:, tree]):
            raise ValueError("A routed leaf is absent from the complete native map")
        indices[:, tree] = positions + offset
        offset += len(ids)
    indptr = np.arange(0, leaves.size + 1, leaves.shape[1], dtype=np.int64)
    return csr_matrix((np.ones(leaves.size, dtype=np.float64), indices.ravel(), indptr),
                      shape=(len(leaves), columns))


def independent_leaf_sum(leaves, maps, coefficients, bias):
    """Independent dictionary gather plus math.fsum, never the sparse product."""
    coefficients = _finite_vector(coefficients, sum(map(len, maps)), "coefficients")
    if not math.isfinite(bias):
        raise ValueError("Nonfinite bias")
    leaves = np.asarray(leaves)
    if (leaves.ndim != 2 or leaves.shape[1] != len(maps) or not np.isfinite(leaves).all()
            or (leaves < 0).any() or not np.equal(leaves, np.floor(leaves)).all()):
        raise ValueError("Tree count changed")
    lookup, offset = [], 0
    for ids in maps:
        ids = np.asarray(ids)
        if (ids.ndim != 1 or not len(ids) or not np.isfinite(ids).all() or (ids < 0).any()
                or not np.equal(ids, np.floor(ids)).all() or not np.all(ids[1:] > ids[:-1])):
            raise ValueError("Invalid full native leaf map")
        lookup.append(dict(zip(map(int, ids), coefficients[offset:offset + len(ids)])))
        offset += len(ids)
    try:
        result = np.array([math.fsum([bias] + [lookup[t][int(leaf)] for t, leaf in enumerate(row)])
                           for row in leaves], dtype=np.float64)
    except KeyError as exc:
        raise ValueError("Unknown native leaf") from exc
    return _finite_vector(result, len(leaves), "independent leaf sum")


def _magnitude(native):
    offset, result = 0, abs(native["bias"])
    for ids in native["leaf_ids_by_tree"]:
        result += float(np.max(np.abs(native["theta"][offset:offset + len(ids)])))
        offset += len(ids)
    if not math.isfinite(result):
        raise ValueError("Magnitude bound overflowed")
    return result


def _gamma(operations, epsilon):
    product = operations * epsilon
    if product >= 1:
        raise ValueError("Too many operations for the rounding bound")
    return product / (1 - product)


def _check_oracle(actual, expected, bound):
    difference = float(np.max(np.abs(actual - expected), initial=0))
    if not math.isfinite(difference) or difference > bound:
        raise AssertionError("Native prediction exceeds the analytical rounding bound")
    np.testing.assert_allclose(actual, expected, rtol=FIXED_RTOL, atol=FIXED_ATOL)
    return difference


def memory_preflight(native, rows, *, max_working_bytes):
    if (type(rows) is not int or rows <= 0 or type(max_working_bytes) is not int
            or max_working_bytes <= 0):
        raise ValueError("Explicit positive row and named-array budgets required")
    columns, trees = len(native["theta"]), len(native["leaf_ids_by_tree"])
    entries = rows * trees
    arrays = {"native_leaf_results_and_copy": 16 * entries,
              "csr_data_indices_indptr": 16 * entries + 8 * (rows + 1),
              "construction_and_gather_temporary": 24 * entries + 32 * rows,
              "lsqr_centered_vectors_certificate_budget": 8 * (24 * rows + 24 * columns)}
    total = sum(arrays.values())
    if total > max_working_bytes:
        raise MemoryError("Frozen absolute-refit named-array budget exceeded")
    return {"rows": rows, "physical_trees": trees, "all_native_leaf_columns": columns,
            "array_budgets_bytes": arrays, "named_array_bytes": total,
            "max_working_bytes": max_working_bytes,
            "excluded": ["models", "frames", "ordered preprocessing", "Python objects", "allocator overhead"],
            "separate_process_tree_rss_guard_required": True}


def prepare_design(payload, frame, labels, *, max_working_bytes):
    _input_frame(frame)
    native = extract_native(payload)
    budget = memory_preflight(native, len(frame), max_working_bytes=max_working_bytes)
    y = _finite_vector(labels, len(frame), "training labels")
    prepared = None
    receipt = {"ordered_training_transform_replayed": False, "memory_preflight": budget}
    if payload["library"] == "ctboost" and payload["model"]._feature_pipeline is not None:
        import ctboost
        pipeline = payload["model"]._feature_pipeline
        original = pipeline.to_state()
        replay = ctboost.FeaturePipeline.from_state(original)
        transformed, categorical, names = replay.fit_transform_array(frame, labels)
        if _hash(replay.to_state()) != _hash(original):
            raise ValueError("Ordered TRAIN replay differs from the fitted pipeline state")
        prepared = ctboost.Pool(transformed, cat_features=categorical, feature_names=names)
        prepared._feature_pipeline = pipeline
        receipt.update(ordered_training_transform_replayed=True,
                       pipeline_sha256=_hash(original), ordered_ctr=bool(original.get("ordered_ctr", False)))
    leaves = leaf_indices(payload, frame, prepared)
    Z = indicator_matrix(leaves, native["leaf_ids_by_tree"])
    baseline = predict(payload, frame, prepared)
    expected = independent_leaf_sum(leaves, native["leaf_ids_by_tree"], native["theta"], native["bias"])
    epsilon = np.finfo(np.float64 if payload["library"] == "lightgbm" else np.float32).eps
    bound = 4 * _gamma(2 * leaves.shape[1] + 8, epsilon) * _magnitude(native) + 1e-12
    difference = _check_oracle(baseline, expected, bound)
    receipt.update(native_reconstruction_difference=difference, native_reconstruction_bound=bound,
                   training_leaf_sha256=hashlib.sha256(leaves.tobytes()).hexdigest(),
                   column_order="physical tree then every sorted native leaf ID; duplicates retained",
                   unobserved_columns=int(np.count_nonzero(np.asarray(Z.sum(axis=0)).ravel() == 0)))
    return {"Z": Z, "native": native, "labels": y.copy(), "baseline": baseline,
            "training_leaves": leaves, "training_prepared": prepared, "receipt": receipt}


def solve_coefficients(design, labels, penalty, *, settings, limits):
    from benchmarks.absolute_leaf_ridge_numerics import solve_absolute_ridge
    y = _finite_vector(labels, design["Z"].shape[0], "training labels")
    if not np.array_equal(y, design["labels"]):
        raise ValueError("The solver labels differ from the ordered TRAIN replay")
    weights, bias, receipt = solve_absolute_ridge(design["Z"], y, penalty, settings=settings, limits=limits)
    if not receipt["success"]:
        raise ValueError("Absolute-leaf solver did not satisfy its frozen certificate")
    unsupported = np.asarray(design["Z"].sum(axis=0)).ravel() == 0
    if np.any(weights[unsupported] != 0):
        raise AssertionError("Unobserved absolute-leaf coefficients must be exactly zero")
    return weights, bias, receipt


def _without_receipt(payload):
    return {key: value for key, value in payload.items() if key != "fixed_absolute_leaf_refit"}


def fold_absolute(payload, design, weights, bias, *, shrinkage):
    """Fold an absolute solution into all leaves and bias, with strict q0 bypass."""
    native = extract_native(payload)
    if (native["native_sha256"] != design["native"]["native_sha256"]
            or native["python_state_sha256"] != design["native"]["python_state_sha256"]):
        raise ValueError("Design/native reference provenance changed")
    saved = design["native"]
    if (len(saved["leaf_ids_by_tree"]) != len(native["leaf_ids_by_tree"])
            or any(not np.array_equal(a, b) for a, b in zip(saved["leaf_ids_by_tree"], native["leaf_ids_by_tree"]))
            or not np.array_equal(saved["theta"], native["theta"]) or saved["bias"] != native["bias"]
            or design["Z"].shape[1] != len(native["theta"])):
        raise ValueError("Design native leaf map or effective outputs changed")
    weights = _finite_vector(weights, len(native["theta"]), "absolute coefficients")
    q, bias = float(shrinkage), float(bias)
    if not math.isfinite(q) or not 0 <= q <= 1 or not math.isfinite(bias):
        raise ValueError("Finite intercept and shrinkage in [0,1] required")
    if np.any(weights[np.asarray(design["Z"].sum(axis=0)).ravel() == 0] != 0):
        raise ValueError("Unobserved leaves require an absolute coefficient of zero")
    if q == 0:
        return payload, {"identity_bypass": True, "original_payload_returned": True,
                         "native_sha256": native["native_sha256"], "shrinkage": 0.0}
    theta = (1 - q) * native["theta"] + q * weights
    target_bias = (1 - q) * native["bias"] + q * bias
    if not np.isfinite(theta).all() or not math.isfinite(target_bias):
        raise ValueError("Blended outputs overflowed")
    library, original = payload["library"], payload["model"]
    document = copy.deepcopy(native["document"])
    offset = 0
    if library == "ctboost":
        import ctboost
        document["base_score"] = [_float32(target_bias)]
        for tree, ids, rate in zip(document["trees"], native["leaf_ids_by_tree"], native["rates"]):
            for leaf in ids:
                tree["nodes"][int(leaf)]["leaf_weight"] = _float32(theta[offset] / rate)
                offset += 1
        model = ctboost.Booster(ctboost._core.GradientBooster.from_state(document),
            feature_pipeline=original._feature_pipeline,
            training_metadata=copy.deepcopy(original._training_metadata))
    elif library == "xgboost":
        import xgboost as xgb
        learner = document["learner"]
        encoded = json.loads(learner["learner_model_param"]["base_score"])
        value = _float32(target_bias)
        learner["learner_model_param"]["base_score"] = json.dumps([value] if isinstance(encoded, list) else value)
        for tree, ids in zip(learner["gradient_booster"]["model"]["trees"], native["leaf_ids_by_tree"]):
            for leaf in ids:
                tree["split_conditions"][int(leaf)] = _float32(theta[offset])
                offset += 1
        model = xgb.Booster()
        model.load_model(bytearray(json.dumps(document).encode()))
        model.set_param({"nthread": 1, "device": "cpu"})
    else:
        import lightgbm as lgb
        model = lgb.Booster(model_str=original.model_to_string(num_iteration=-1))
        for tree, ids in enumerate(native["leaf_ids_by_tree"]):
            for leaf in ids:
                value = float(theta[offset] + (target_bias if tree == 0 else 0))
                if not math.isfinite(value):
                    raise ValueError("LightGBM bias placement overflowed")
                model.set_leaf_output(tree, int(leaf), value)
                offset += 1
    candidate = dict(payload, model=model)
    after = extract_native(candidate)
    if (native["nonoutput_sha256"] != after["nonoutput_sha256"]
            or native["python_state_sha256"] != after["python_state_sha256"]):
        raise AssertionError("A nonoutput field, schema or history changed")
    expected_storage = theta.copy()
    if library == "lightgbm":
        expected_storage[:len(native["leaf_ids_by_tree"][0])] += target_bias
        stored_bias_target = 0.0
    else:
        stored_bias_target = target_bias
    offset, coefficient_error = 0, 0.0
    for ids in native["leaf_ids_by_tree"]:
        coefficient_error += float(np.max(np.abs(after["theta"][offset:offset + len(ids)]
                                                 - expected_storage[offset:offset + len(ids)])))
        offset += len(ids)
    bias_error = abs(after["bias"] - stored_bias_target)
    epsilon = np.finfo(np.float64 if library == "lightgbm" else np.float32).eps
    operations = 2 * len(native["leaf_ids_by_tree"]) + 8
    rounding = _gamma(operations, epsilon) * (_magnitude(after) + (1 - q) * _magnitude(native))
    oracle_magnitude = abs(bias) + float(np.abs(weights).sum()) + _magnitude(native)
    oracle_rounding = _gamma(operations, np.finfo(np.float64).eps) * oracle_magnitude
    bound = 4 * (coefficient_error + bias_error + rounding + oracle_rounding) + 1e-12
    if not math.isfinite(bound):
        raise ValueError("Prediction roundoff bound overflowed")
    receipt = {"identity_bypass": False, "shrinkage": q, "inference_only": True,
        "native_sha256_before": native["native_sha256"], "native_sha256_after": after["native_sha256"],
        "nonoutput_sha256_before": native["nonoutput_sha256"], "nonoutput_sha256_after": after["nonoutput_sha256"],
        "python_state_sha256": native["python_state_sha256"], "physical_trees": len(native["leaf_ids_by_tree"]),
        "native_leaves": len(weights), "unobserved_absolute_weight": 0.0,
        "bias_gauge": "zero; blended intercept added to every tree0 leaf" if library == "lightgbm" else "explicit native effective base_score",
        "coefficient_rounding_absolute_bound": coefficient_error, "bias_rounding_absolute_bound": bias_error,
        "prediction_oracle_absolute_tolerance": bound, "fixed_rtol": FIXED_RTOL, "fixed_atol": FIXED_ATOL,
        "historical_statistics": "Preserved without recomputation; warm start/resume unsupported"}
    return dict(candidate, fixed_absolute_leaf_refit=receipt), receipt


def verify_fold(before, after, frame, design, weights, bias, receipt, training_prepared=None):
    old_leaves = leaf_indices(before, frame, training_prepared)
    np.testing.assert_array_equal(old_leaves, leaf_indices(after, frame, training_prepared))
    old = predict(before, frame, training_prepared)
    actual = predict(after, frame, training_prepared)
    if receipt["identity_bypass"]:
        if after is not before:
            raise AssertionError("q0 must return the original payload object")
        np.testing.assert_array_equal(actual, old)
        return {"identity_bytes_exact": True, "routes_exact": True}
    native = design["native"]
    absolute = independent_leaf_sum(old_leaves, native["leaf_ids_by_tree"], weights, float(bias))
    q = receipt["shrinkage"]
    expected = (1 - q) * old + q * absolute
    difference = _check_oracle(actual, expected, receipt["prediction_oracle_absolute_tolerance"])
    current = extract_native(_without_receipt(after))
    if (current["native_sha256"] != receipt["native_sha256_after"]
            or current["nonoutput_sha256"] != native["nonoutput_sha256"]
            or current["python_state_sha256"] != native["python_state_sha256"]):
        raise AssertionError("The folded artifact provenance changed")
    reconstructed = independent_leaf_sum(old_leaves, current["leaf_ids_by_tree"], current["theta"], current["bias"])
    reconstruction_difference = _check_oracle(actual, reconstructed, receipt["prediction_oracle_absolute_tolerance"])
    return {"routes_exact": True, "blend_oracle_max_absolute_difference": difference,
            "native_reconstruction_max_absolute_difference": reconstruction_difference,
            "independent_dictionary_fsum_oracle": True}


def verify_reload(payload, frames):
    """Require native serialization and pickle prediction/route byte parity."""
    original = payload["model"]
    library = payload["library"]
    if library == "ctboost":
        import ctboost
        # The native state document is the JSON serialization payload. Python's
        # JSON roundtrip also exercises finite output storage and list coercion.
        state = json.loads(json.dumps(original._handle.export_state(), allow_nan=True))
        model = ctboost.Booster(ctboost._core.GradientBooster.from_state(state),
            feature_pipeline=original._feature_pipeline,
            training_metadata=copy.deepcopy(original._training_metadata))
    elif library == "xgboost":
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(original.save_raw(raw_format="json"))
        model.set_param({"nthread": 1, "device": "cpu"})
    else:
        import lightgbm as lgb
        model = lgb.Booster(model_str=original.model_to_string(num_iteration=-1))
    restored = [dict(payload, model=model), pickle.loads(pickle.dumps(payload, protocol=5))]
    reference = extract_native(_without_receipt(payload))
    for candidate in restored:
        restored_native = extract_native(_without_receipt(candidate))
        if (restored_native["native_sha256"] != reference["native_sha256"]
                or restored_native["python_state_sha256"] != reference["python_state_sha256"]):
            raise AssertionError("Native reload changed preserved fields or outputs")
        for frame in frames:
            left, right = predict(payload, frame), predict(candidate, frame)
            if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
                raise AssertionError("Reloaded prediction bytes differ")
            np.testing.assert_array_equal(leaf_indices(payload, frame), leaf_indices(candidate, frame))
    return {"native_serialization_prediction_bytes_exact": True, "pickle_prediction_bytes_exact": True,
            "reload_leaf_routes_exact": True, "frames_checked": len(frames)}
