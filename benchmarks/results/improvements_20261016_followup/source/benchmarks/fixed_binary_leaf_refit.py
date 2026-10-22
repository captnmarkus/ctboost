"""Experimental binary-logistic fixed-topology leaf correction, not a public API.

The caller freezes the logistic-offset ridge protocol before fitting. This module reconstructs
the original ordered training transform, constructs a sparse leaf-indicator
matrix, and folds contribution-unit corrections into ordinary native models.
It never selects features, cuts, trees, or a correction using development data.
Refitted artifacts are for inference; their historical training statistics are
not recomputed and must not be used as a resumed-training experiment.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=True).encode()).hexdigest()


def _binary(payload):
    if payload["problem"] != "binary":
        raise ValueError("This experiment supports binary logistic scalar margins only")


def predict(payload, normalized_frame, training_prepared=None):
    _binary(payload)
    model = payload["model"]
    if payload["library"] == "ctboost":
        data = normalized_frame if training_prepared is None else training_prepared
        result = model.predict(data)
    elif payload["library"] == "xgboost":
        import xgboost as xgb
        result = model.predict(xgb.DMatrix(normalized_frame, enable_categorical=True),
                               output_margin=True)
    elif payload["library"] == "lightgbm":
        result = model.predict(normalized_frame, raw_score=True, num_threads=1)
    else:
        raise ValueError("Unknown library")
    result = np.asarray(result, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("Expected finite scalar predictions")
    return result


def leaf_indices(payload, normalized_frame, training_prepared=None):
    _binary(payload)
    model = payload["model"]
    if payload["library"] == "ctboost":
        data = normalized_frame if training_prepared is None else training_prepared
        values = model.predict_leaf_index(data)
    elif payload["library"] == "xgboost":
        import xgboost as xgb
        values = model.predict(xgb.DMatrix(normalized_frame, enable_categorical=True), pred_leaf=True)
    elif payload["library"] == "lightgbm":
        values = model.predict(normalized_frame, pred_leaf=True, num_threads=1)
    else:
        raise ValueError("Unknown library")
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if (values.ndim != 2 or not np.isfinite(values).all() or
            (values < 0).any() or not np.equal(values, np.floor(values)).all()):
        raise ValueError("Expected nonnegative integral leaf IDs")
    return values.astype(np.int64)


def indicator_matrix(leaves):
    """Columns concatenate sorted observed leaf IDs for each physical tree."""
    from scipy.sparse import csr_matrix
    leaves = np.asarray(leaves, dtype=np.int64)
    if leaves.ndim != 2 or not all(leaves.shape) or (leaves < 0).any():
        raise ValueError("A nonempty two-dimensional leaf matrix is required")
    maps = [np.unique(leaves[:, tree]) for tree in range(leaves.shape[1])]
    columns = sum(len(ids) for ids in maps)
    indices = np.empty(leaves.shape, dtype=np.int32 if columns < 2**31 else np.int64)
    offset = 0
    for tree, ids in enumerate(maps):
        indices[:, tree] = offset + np.searchsorted(ids, leaves[:, tree])
        offset += len(ids)
    indptr = np.arange(0, leaves.size + 1, leaves.shape[1], dtype=np.int64)
    matrix = csr_matrix((np.ones(leaves.size), indices.ravel(), indptr),
                        shape=(leaves.shape[0], columns))
    return matrix, maps


def prepare_design(payload, Xtrain_normalized, ytrain):
    _binary(payload)
    y = np.asarray(ytrain, dtype=np.float64)
    if y.shape != (len(Xtrain_normalized),) or not np.isfinite(y).all():
        raise ValueError("Training labels must be a finite vector")
    prepared = None
    receipt = {"training_rows": len(y), "ordered_training_transform_replayed": False}
    if payload["library"] == "ctboost":
        import ctboost
        pipeline = payload["model"]._feature_pipeline
        if pipeline is not None:
            original = pipeline.to_state()
            replay = ctboost.FeaturePipeline.from_state(original)
            transformed, categorical, names = replay.fit_transform_array(Xtrain_normalized, ytrain)
            if replay.to_state() != original:
                raise ValueError("Replayed training pipeline does not match the saved fitted state")
            prepared = ctboost.Pool(transformed, cat_features=categorical, feature_names=names)
            # The verified fitted state is identical; attach the original object
            # to preserve the normal prepared-Pool schema/identity contract.
            prepared._feature_pipeline = pipeline
            receipt.update(ordered_training_transform_replayed=True,
                           ordered_ctr=bool(original.get("ordered_ctr", False)),
                           pipeline_state_sha256=_hash(original),
                           training_transform_sha256=hashlib.sha256(
                               np.ascontiguousarray(transformed).tobytes()).hexdigest())
    leaves = leaf_indices(payload, Xtrain_normalized, prepared)
    baseline = predict(payload, Xtrain_normalized, prepared)
    matrix, maps = indicator_matrix(leaves)
    receipt.update(design_shape=list(matrix.shape), design_nonzeros=int(matrix.nnz),
                   column_order="physical tree order, then sorted observed native leaf IDs",
                   unobserved_leaf_correction=0,
                   training_leaf_ids_sha256=hashlib.sha256(leaves.tobytes()).hexdigest())
    return {"Z": matrix, "baseline": baseline, "leaf_ids_by_tree": maps,
            "training_leaves": leaves, "training_prepared": prepared,
            "preprocessing_receipt": receipt}


def corrections_by_tree(leaf_ids_by_tree, coefficients, shrinkage=1.0):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if (coefficients.shape != (sum(len(ids) for ids in leaf_ids_by_tree),) or
            not np.isfinite(coefficients).all() or not np.isfinite(shrinkage)):
        raise ValueError("Invalid correction vector")
    result, offset = [], 0
    for ids in leaf_ids_by_tree:
        result.append({int(leaf): float(shrinkage * coefficients[offset + j])
                       for j, leaf in enumerate(ids)})
        offset += len(ids)
    return result


def logistic_objective(coefficients, matrix, baseline, labels, penalty):
    """Mean logistic loss plus lambda/(2*n) times squared corrections."""
    from scipy.special import expit
    margins = baseline + matrix @ coefficients
    n = len(labels)
    value = (np.logaddexp(0., (1. - 2. * labels) * margins).sum()
             + .5 * penalty * (coefficients @ coefficients)) / n
    gradient = (np.asarray(matrix.T @ (expit(margins) - labels)).ravel()
                + penalty * coefficients) / n
    return float(value), gradient


def solve_coefficients(design, labels, penalty, *, maxiter=2000,
                       ftol=1e-13, gtol=1e-9, gradient_gate=1e-6):
    """Optimize offset logits with positive ridge, without changing tree routes.

    Every penalty starts at zero. The maximum absolute gradient of the mean
    objective must pass a fixed gate; solver status alone is insufficient.
    """
    from scipy.optimize import minimize
    labels = np.asarray(labels, dtype=np.float64)
    baseline = np.asarray(design["baseline"], dtype=np.float64)
    matrix = design["Z"]
    if (labels.ndim != 1 or not len(labels) or labels.shape != baseline.shape
            or matrix.shape[0] != len(labels) or matrix.shape[1] == 0
            or not np.isin(labels, [0., 1.]).all()
            or not np.isfinite(baseline).all() or not np.isfinite(matrix.data).all()
            or not np.isfinite(penalty) or penalty <= 0):
        raise ValueError("Expected finite binary labels, margins, design and positive ridge")
    zero = np.zeros(matrix.shape[1], dtype=np.float64)
    args = (matrix, baseline, labels, float(penalty))
    initial, _ = logistic_objective(zero, *args)
    result = minimize(logistic_objective, zero, args=args, method="L-BFGS-B", jac=True,
                      options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol,
                               "maxls": 50, "maxcor": 20})
    final, gradient = logistic_objective(result.x, *args)
    norm = float(np.max(np.abs(gradient)))
    if (not result.success or not np.isfinite(result.x).all()
            or not np.isfinite(final) or not np.isfinite(norm)
            or final > initial + 1e-12 or norm > gradient_gate):
        raise RuntimeError(f"Logistic correction did not converge: status={result.status}, "
                           f"gradient={norm}, initial={initial}, final={final}; {result.message}")
    return result.x, {"method": "L-BFGS-B", "penalty": float(penalty),
        "objective": "sum binary logistic loss with fixed baseline offset + lambda/2 * ||delta||^2, scaled by rows",
        "iterations": int(result.nit), "evaluations": int(result.nfev), "status": int(result.status),
        "message": str(result.message), "initial_objective": initial, "final_objective": final,
        "maximum_absolute_mean_gradient": norm, "gradient_gate": gradient_gate,
        "maxiter": maxiter, "ftol": ftol, "gtol": gtol, "maxls": 50, "maxcor": 20,
        "initial_coefficients": "all zero", "rows": len(labels), "columns": matrix.shape[1]}


def correction_values(leaves, deltas_per_tree):
    leaves = np.asarray(leaves)
    if leaves.ndim != 2 or leaves.shape[1] != len(deltas_per_tree):
        raise ValueError("Tree count changed")
    result = np.zeros(leaves.shape[0], dtype=np.float64)
    for tree, mapping in enumerate(deltas_per_tree):
        result += np.fromiter((mapping.get(int(leaf), 0.0) for leaf in leaves[:, tree]),
                              dtype=np.float64, count=len(result))
    return result


def _ct_topology(state):
    result = copy.deepcopy(state)
    for tree in result["trees"]:
        for node in tree["nodes"]:
            if node["is_leaf"]:
                node.pop("leaf_weight", None)
                node.pop("leaf_weights", None)
    return result


def _xgb_document(model):
    return json.loads(bytes(model.save_raw(raw_format="json")))


def _xgb_topology(document):
    result = copy.deepcopy(document)
    for tree in result["learner"]["gradient_booster"]["model"]["trees"]:
        for node, child in enumerate(tree["left_children"]):
            if child == -1:
                tree["split_conditions"][node] = None
    return result


def _lgb_topology(document):
    result = copy.deepcopy(document)
    def strip(node):
        if "leaf_value" in node:
            node.pop("leaf_value")
        else:
            strip(node["left_child"])
            strip(node["right_child"])
    for tree in result["tree_info"]:
        strip(tree["tree_structure"])
    return result


def fold_corrections(payload, deltas_per_tree):
    """Return an inference artifact, changing only existing scalar leaf values."""
    _binary(payload)
    if any(not np.isfinite(value) for values in deltas_per_tree for value in values.values()):
        raise ValueError("Leaf corrections must be finite")
    original, library = payload["model"], payload["library"]
    maximum_coefficient_error, old_magnitude, new_magnitude = 0.0, 0.0, 0.0
    if library == "ctboost":
        import ctboost
        state = original._handle.export_state()
        if (state["objective_name"].lower() != "logloss" or
                original.prediction_dimension != 1 or
                any(state["monotone_constraints"]) or state["max_leaf_weight"] > 0 or
                state["boosting_type"] != "GradientBoosting"):
            raise ValueError("Unsupported constrained/non-scalar/non-Logloss CTBoost model")
        if len(state["trees"]) != len(deltas_per_tree):
            raise ValueError("Tree count changed")
        before_hash = _hash(_ct_topology(state))
        rates = state["tree_learning_rates"]
        for index, (tree, mapping) in enumerate(zip(state["trees"], deltas_per_tree)):
            rate = np.float32(rates[index] if rates else state["learning_rate"])
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("Cannot fold through a zero/invalid learning rate")
            valid = {i for i, node in enumerate(tree["nodes"]) if node["is_leaf"]}
            if not set(mapping).issubset(valid):
                raise ValueError("Correction references a nonexistent leaf")
            errors, old_values, new_values = [], [], []
            for leaf in valid:
                node = tree["nodes"][leaf]
                if node.get("leaf_weights"):
                    raise ValueError("Vector leaves are outside this binary experiment")
                old = float(np.float32(rate * np.float32(node["leaf_weight"])))
                delta = mapping.get(leaf, 0.0)
                if delta:
                    node["leaf_weight"] = float(np.float32(float(node["leaf_weight"]) + delta / float(rate)))
                new = float(np.float32(rate * np.float32(node["leaf_weight"])))
                errors.append(abs((new - old) - delta))
                old_values.append(abs(old)); new_values.append(abs(new))
            maximum_coefficient_error += max(errors, default=0)
            old_magnitude += max(old_values, default=0)
            new_magnitude += max(new_values, default=0)
        model = ctboost.Booster(ctboost._core.GradientBooster.from_state(state),
                               feature_pipeline=original._feature_pipeline,
                               training_metadata=copy.deepcopy(original._training_metadata))
        after_hash = _hash(_ct_topology(model._handle.export_state()))
        base_magnitude = sum(abs(float(value)) for value in state["base_score"])
        epsilon = np.finfo(np.float32).eps
    elif library == "xgboost":
        import xgboost as xgb
        document = _xgb_document(original)
        learner = document["learner"]
        if learner["objective"]["name"] != "binary:logistic" or learner["gradient_booster"]["name"] != "gbtree":
            raise ValueError("Only scalar binary-logistic gbtree models are supported")
        trees = learner["gradient_booster"]["model"]["trees"]
        if len(trees) != len(deltas_per_tree):
            raise ValueError("Tree count changed")
        before_hash = _hash(_xgb_topology(document))
        for tree, mapping in zip(trees, deltas_per_tree):
            if int(tree["tree_param"]["size_leaf_vector"]) > 1:
                raise ValueError("Vector leaves are outside this binary experiment")
            valid = {i for i, child in enumerate(tree["left_children"]) if child == -1}
            if not set(mapping).issubset(valid):
                raise ValueError("Correction references a nonexistent leaf")
            errors, old_values, new_values = [], [], []
            for leaf in valid:
                old = float(np.float32(tree["split_conditions"][leaf]))
                delta = mapping.get(leaf, 0.0)
                new = float(np.float32(old + delta))
                tree["split_conditions"][leaf] = new
                errors.append(abs((new - old) - delta))
                old_values.append(abs(old)); new_values.append(abs(new))
            maximum_coefficient_error += max(errors, default=0)
            old_magnitude += max(old_values, default=0)
            new_magnitude += max(new_values, default=0)
        model = xgb.Booster()
        model.load_model(bytearray(json.dumps(document).encode()))
        model.set_param({"nthread": 1})
        after_hash = _hash(_xgb_topology(_xgb_document(model)))
        base = learner["learner_model_param"]["base_score"]
        base_values = json.loads(base) if isinstance(base, str) else base
        base_probabilities = np.atleast_1d(np.asarray(base_values, dtype=np.float64))
        if not ((base_probabilities > 0) & (base_probabilities < 1)).all():
            raise ValueError("Expected binary-logistic base probabilities in (0,1)")
        base_magnitude = float(np.abs(np.log(base_probabilities) - np.log1p(-base_probabilities)).sum())
        epsilon = np.finfo(np.float32).eps
    elif library == "lightgbm":
        import lightgbm as lgb
        document = original.dump_model()
        if document["objective"].split()[0] != "binary" or document["num_tree_per_iteration"] != 1:
            raise ValueError("Only scalar binary-logistic LightGBM models are supported")
        sigmoid = [float(token.split(":", 1)[1]) for token in document["objective"].split()
                   if token.startswith("sigmoid:")]
        if len(sigmoid) > 1 or any(value != 1. for value in sigmoid):
            raise ValueError("LightGBM binary sigmoid must be 1 for unit-scale logits")
        if len(document["tree_info"]) != len(deltas_per_tree):
            raise ValueError("Tree count changed")
        before_hash = _hash(_lgb_topology(document))
        model = lgb.Booster(model_str=original.model_to_string())
        for tree, (info, mapping) in enumerate(zip(document["tree_info"], deltas_per_tree)):
            valid = set(range(info["num_leaves"]))
            if not set(mapping).issubset(valid):
                raise ValueError("Correction references a nonexistent leaf")
            errors, old_values, new_values = [], [], []
            for leaf in valid:
                old = original.get_leaf_output(tree, leaf)
                delta = mapping.get(leaf, 0.0)
                new = old + delta
                if delta:
                    model.set_leaf_output(tree, leaf, new)
                errors.append(abs((new - old) - delta))
                old_values.append(abs(old)); new_values.append(abs(new))
            maximum_coefficient_error += max(errors, default=0)
            old_magnitude += max(old_values, default=0)
            new_magnitude += max(new_values, default=0)
        after_hash = _hash(_lgb_topology(model.dump_model()))
        base_magnitude, epsilon = 0.0, np.finfo(np.float64).eps
    else:
        raise ValueError("Unknown library")
    if before_hash != after_hash:
        raise AssertionError("Fields other than scalar leaf outputs changed")
    if not np.isfinite([old_magnitude, new_magnitude, maximum_coefficient_error]).all():
        raise ValueError("Leaf correction overflowed")
    operations = 2 * len(deltas_per_tree) + 4
    gamma = operations * epsilon / (1 - operations * epsilon)
    tolerance = 4 * (maximum_coefficient_error + gamma * (old_magnitude + new_magnitude + 2 * base_magnitude)) + 1e-12
    receipt = {"library": library, "topology_sha256_before": before_hash,
               "topology_sha256_after": after_hash, "physical_trees": len(deltas_per_tree),
               "coefficient_rounding_absolute_bound": maximum_coefficient_error,
               "prediction_oracle_absolute_tolerance": tolerance,
               "inference_only": True,
               "historical_training_metadata": "Preserved, not recomputed; no resume-training claim"}
    result = dict(payload, model=model, fixed_leaf_refit=receipt)
    return result, receipt


def verify_fold(before, after, frame, deltas_per_tree, receipt, training_prepared=None):
    leaves = leaf_indices(before, frame, training_prepared)
    np.testing.assert_array_equal(leaves, leaf_indices(after, frame, training_prepared))
    expected = predict(before, frame, training_prepared) + correction_values(leaves, deltas_per_tree)
    actual = predict(after, frame, training_prepared)
    difference = float(np.max(np.abs(actual - expected), initial=0))
    if difference > receipt["prediction_oracle_absolute_tolerance"]:
        raise AssertionError("Native folded prediction differs from leaf-indicator correction oracle")
    # Retain the separately predeclared fixed tolerance as an additional gate;
    # the analytical rounding bound never silently relaxes that protocol.
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-5)
    return {"leaf_routing_exact": True, "oracle_max_absolute_difference": difference,
            "oracle_absolute_tolerance": receipt["prediction_oracle_absolute_tolerance"],
            "fixed_tolerance_rtol": 5e-5, "fixed_tolerance_atol": 1e-5,
            "oracle_difference_divided_by_max_one_or_prediction_magnitude":
                difference / max(1., float(np.max(np.abs(expected), initial=0)))}
