"""Experimental multiclass fixed-leaf correction. No dataset/runner entry point.

Only existing native leaf outputs change. All margins/corrections are raw units;
the inherited temperature divides once in the objective and served predictions.
Historical training statistics are retained, so results are inference artifacts.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

DRAFT_ONLY = False
FIXED_RTOL = 5e-5
FIXED_ATOL = 1e-5


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=True).encode()).hexdigest()


def _temperature(payload):
    value = payload.get("temperature")
    value = 1. if value is None else float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Expected finite positive fixed temperature")
    return value


def _metadata(payload):
    if payload["problem"] != "multiclass":
        raise ValueError("Only multiclass softmax models are supported")
    model, library = payload["model"], payload["library"]
    _temperature(payload)
    if library == "ctboost":
        document = model._handle.export_state()
        classes = int(model.prediction_dimension)
        if (document["objective_name"].lower() != "multiclass" or classes < 2
                or document["boosting_type"] != "GradientBoosting"
                or any(document["monotone_constraints"]) or document["max_leaf_weight"] > 0
                or document.get("use_gpu", False)):
            raise ValueError("Unsupported CTBoost objective, constraints or prediction mode")
        vector = document["multi_strategy"] == "multi_output_tree"
        tree_classes = tuple(-1 if vector else tree % classes
                             for tree in range(len(document["trees"])))
    elif library == "xgboost":
        import xgboost as xgb
        if xgb.__version__ != "3.4.1":
            raise ValueError("This draft requires XGBoost3.4.1 serialization")
        document = json.loads(bytes(model.save_raw(raw_format="json")))
        learner = document["learner"]
        classes = int(learner["learner_model_param"]["num_class"])
        if (learner["objective"]["name"] != "multi:softprob" or classes < 2
                or learner["gradient_booster"]["name"] != "gbtree"):
            raise ValueError("Expected multiclass softprob gbtree")
        trees = learner["gradient_booster"]["model"]
        tree_classes = tuple(-1 if int(tree["tree_param"]["size_leaf_vector"]) > 1 else int(group)
                             for tree, group in zip(trees["trees"], trees["tree_info"]))
        if len(tree_classes) != len(trees["trees"]):
            raise ValueError("Invalid XGBoost tree-group mapping")
        for tree in trees["trees"]:
            width = int(tree["tree_param"]["size_leaf_vector"])
            if width > 1:
                if width != classes or "leaf_weights" not in tree:
                    raise ValueError("Unsupported vector leaf width/storage")
                leaves = [i for i, child in enumerate(tree["left_children"]) if child == -1]
                slots = [tree["right_children"][i] for i in leaves]
                if sorted(slots) != list(range(len(leaves))) or len(tree["leaf_weights"]) != classes * len(leaves):
                    raise ValueError("Invalid XGBoost vector leaf mapping")
    elif library == "lightgbm":
        document = model.dump_model(num_iteration=-1)
        classes = int(document["num_class"])
        if (document["objective"].split()[0] != "multiclass" or classes < 2
                or document["num_tree_per_iteration"] != classes
                or document.get("average_output", False)
                or any(document.get("monotone_constraints", []))):
            raise ValueError("Expected unconstrained native multiclass softmax LightGBM")
        tree_classes = tuple(tree % classes for tree in range(len(document["tree_info"])))
    else:
        raise ValueError("Unknown library")
    if (not tree_classes or any(value < -1 or value >= classes for value in tree_classes)
            or (-1 in tree_classes and any(value != -1 for value in tree_classes))):
        raise ValueError("Mixed scalar/vector or invalid class support is outside this draft")
    return classes, tree_classes, document


def predict_raw(payload, frame, training_prepared=None):
    model, library = payload["model"], payload["library"]
    if library == "ctboost":
        values = model.predict(frame if training_prepared is None else training_prepared)
    elif library == "xgboost":
        import xgboost as xgb
        values = model.predict(xgb.DMatrix(frame, enable_categorical=True), output_margin=True)
    elif library == "lightgbm":
        values = model.predict(frame, raw_score=True, num_threads=1, num_iteration=-1)
    else:
        raise ValueError("Unknown library")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(frame) or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite multiclass raw margins")
    return values


def predict(payload, frame, training_prepared=None):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return predict_raw(payload, frame, training_prepared) / _temperature(payload)


def leaf_indices(payload, frame, training_prepared=None):
    model, library = payload["model"], payload["library"]
    if library == "ctboost":
        values = model.predict_leaf_index(frame if training_prepared is None else training_prepared)
    elif library == "xgboost":
        import xgboost as xgb
        values = model.predict(xgb.DMatrix(frame, enable_categorical=True), pred_leaf=True, strict_shape=False)
    elif library == "lightgbm":
        values = model.predict(frame, pred_leaf=True, num_threads=1, num_iteration=-1)
    else:
        raise ValueError("Unknown library")
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if (values.ndim != 2 or values.shape[0] != len(frame) or not np.isfinite(values).all()
            or (values < 0).any() or not np.equal(values, np.floor(values)).all()):
        raise ValueError("Expected native nonnegative integral leaf IDs")
    return values.astype(np.int64)


def _indicator(leaves, maps):
    from scipy.sparse import csr_matrix
    rows, trees = leaves.shape
    columns = sum(len(ids) for ids in maps)
    indices = np.empty((rows, trees), dtype=np.int32 if columns < 2**31 else np.int64)
    offset = 0
    for tree, ids in enumerate(maps):
        indices[:, tree] = offset + np.searchsorted(ids, leaves[:, tree])
        offset += len(ids)
    return csr_matrix((np.ones(indices.size), indices.ravel(),
                       np.arange(0, indices.size + 1, trees, dtype=np.int64)), shape=(rows, columns))


def make_design(leaves, baseline, tree_classes, temperature=1.):
    leaves, baseline = np.asarray(leaves), np.asarray(baseline, dtype=np.float64)
    temperature = _temperature({"temperature": temperature})
    if (leaves.ndim != 2 or not all(leaves.shape) or leaves.dtype.kind not in "iu"
            or (leaves < 0).any() or baseline.ndim != 2 or baseline.shape[0] != leaves.shape[0]
            or baseline.shape[1] < 2 or not np.isfinite(baseline).all()
            or type(tree_classes) is not tuple or len(tree_classes) != leaves.shape[1]):
        raise ValueError("Invalid sparse multiclass design")
    classes = baseline.shape[1]
    if (any(type(c) is not int or c < -1 or c >= classes for c in tree_classes)
            or (-1 in tree_classes and any(c != -1 for c in tree_classes))):
        raise ValueError("Invalid/mixed native class support")
    if -1 not in tree_classes and set(tree_classes) != set(range(classes)):
        raise ValueError("Every class needs native scalar trees")
    maps = [np.unique(leaves[:, tree]) for tree in range(leaves.shape[1])]
    vector = tree_classes[0] == -1
    shared = _indicator(leaves, maps) if vector else None
    matrices, trees_by_class, offsets = [], [], [0]
    for c in range(classes):
        trees = [tree for tree, support in enumerate(tree_classes) if support in (-1, c)]
        matrix = shared if vector else _indicator(leaves[:, trees], [maps[tree] for tree in trees])
        matrices.append(matrix)
        trees_by_class.append(trees)
        offsets.append(offsets[-1] + matrix.shape[1])
    return {"matrices": matrices, "offsets": offsets, "maps": maps,
            "trees_by_class": trees_by_class, "tree_classes": tree_classes,
            "baseline": baseline, "temperature": temperature, "training_leaves": leaves}


def sparse_memory_estimate(rows, classes, tree_classes, leaf_counts, *, maxcor=20):
    """Conservative named array budgets before native leaf prediction/design.

    This is refit workspace accounting, not a bound on whole-process RSS. Native
    model objects, input frames, categorical transforms and allocator overhead
    are excluded explicitly. Observed leaf counts are bounded by min(rows,L).
    Vector class blocks share one CSR; scalar blocks partition physical trees.
    """
    if (type(rows) is not int or rows <= 0 or type(classes) is not int or classes < 2
            or type(maxcor) is not int or maxcor <= 0
            or len(tree_classes) != len(leaf_counts) or not leaf_counts
            or any(type(count) is not int or count <= 0 for count in leaf_counts)
            or any(type(c) is not int or c < -1 or c >= classes for c in tree_classes)
            or (-1 in tree_classes and any(c != -1 for c in tree_classes))):
        raise ValueError("Invalid sparse memory dimensions")
    vector = tree_classes[0] == -1
    if not vector and set(tree_classes) != set(range(classes)):
        raise ValueError("Every class needs native scalar trees")
    trees = len(tree_classes)
    observed = [min(rows, count) for count in leaf_counts]
    leaf_columns = sum(observed)
    parameters = leaf_columns * classes if vector else leaf_columns
    nonzeros = rows * trees
    # int64 is budgeted even when scipy safely downcasts CSR indices/indptr.
    matrix_count = 1 if vector else classes
    array_bytes = {
        "native_leaf_result_and_int64_copy": 16 * nonzeros,
        "native_raw_result_and_float64_copy": 16 * rows * classes,
        "observed_leaf_maps": 8 * leaf_columns,
        "resident_csr_data_indices_indptr": 16 * nonzeros + 8 * (rows + 1) * matrix_count,
        # Fancy-indexed scalar leaf subset, CSR construction indices/copies and
        # one-column sort/search work; deliberately budget worst physical width.
        "design_construction_temporary_budget": 24 * nonzeros + 32 * rows,
        "objective_margin_probability_workspace_budget": 8 * rows * (8 * classes + 4),
        "lbfgsb_float_and_integer_work_arrays":
            8 * (2 * maxcor * parameters + 5 * parameters + 11 * maxcor * maxcor + 8 * maxcor)
            + 4 * 3 * parameters,
        "optimizer_vector_budget": 8 * 12 * parameters + 4 * parameters,
    }
    total = sum(array_bytes.values())
    return {"rows": rows, "classes": classes, "physical_trees": trees,
            "vector_leaves": vector, "unique_csr_matrices": matrix_count,
            "native_leaf_count": sum(leaf_counts), "observed_leaf_columns_upper_bound": leaf_columns,
            "parameters_upper_bound": parameters, "csr_nonzeros": nonzeros,
            "maxcor": maxcor, "array_budgets_bytes": array_bytes,
            "refit_array_budget_bytes": total,
            "scope": "named conservative refit arrays, not whole-process RSS",
            "excluded": ["native model objects", "input frames", "categorical transform workspace",
                         "Python object/allocator overhead"],
            "no_dense_hessian": True}


def _memory_preflight(library, rows, classes, supports, document, max_working_bytes):
    if library == "ctboost":
        counts = [sum(bool(node["is_leaf"]) for node in tree["nodes"]) for tree in document["trees"]]
    elif library == "xgboost":
        counts = [sum(child == -1 for child in tree["left_children"])
                  for tree in document["learner"]["gradient_booster"]["model"]["trees"]]
    else:
        counts = [int(tree["num_leaves"]) for tree in document["tree_info"]]
    estimate = sparse_memory_estimate(rows, classes, supports, counts, maxcor=20)
    if type(max_working_bytes) is not int or max_working_bytes <= 0:
        raise ValueError("Expected a positive frozen refit-array memory budget")
    estimate["max_working_bytes"] = max_working_bytes
    if estimate["refit_array_budget_bytes"] > max_working_bytes:
        raise MemoryError("Frozen refit-array memory budget exceeded")
    return estimate


def memory_preflight(payload, rows, *, max_working_bytes):
    """Require the runner's frozen refit-array budget before allocating design."""
    classes, supports, document = _metadata(payload)
    return _memory_preflight(payload["library"], rows, classes, supports, document, max_working_bytes)


def prepare_design(payload, frame, labels, *, max_working_bytes=None):
    classes, supports, document = _metadata(payload)
    preflight = None if max_working_bytes is None else _memory_preflight(
        payload["library"], len(frame), classes, supports, document, max_working_bytes)
    del document
    labels = _labels(labels, len(frame), classes)
    prepared = None
    receipt = {"ordered_training_transform_replayed": False, "training_rows": len(frame),
               "memory_preflight": preflight}
    if payload["library"] == "ctboost" and payload["model"]._feature_pipeline is not None:
        import ctboost
        pipeline = payload["model"]._feature_pipeline
        state = pipeline.to_state()
        replay = ctboost.FeaturePipeline.from_state(state)
        transformed, categorical, names = replay.fit_transform_array(frame, labels)
        if replay.to_state() != state:
            raise ValueError("Replayed training pipeline does not match saved state")
        prepared = ctboost.Pool(transformed, cat_features=categorical, feature_names=names)
        prepared._feature_pipeline = pipeline
        receipt.update(ordered_training_transform_replayed=True,
                       ordered_ctr=bool(state.get("ordered_ctr", False)), pipeline_state_sha256=_hash(state),
                       training_transform_sha256=hashlib.sha256(np.ascontiguousarray(transformed).tobytes()).hexdigest())
    leaves = leaf_indices(payload, frame, prepared)
    baseline = predict_raw(payload, frame, prepared)
    if baseline.shape[1] != classes:
        raise ValueError("Native class dimension changed")
    design = make_design(leaves, baseline, supports, _temperature(payload))
    receipt.update(parameters=design["offsets"][-1], physical_trees=len(supports),
                   coefficient_order="class, physical tree, sorted observed native leaf ID",
                   unobserved_leaf_correction=0,
                   training_leaf_ids_sha256=hashlib.sha256(leaves.tobytes()).hexdigest())
    design.update(training_prepared=prepared, preprocessing_receipt=receipt)
    return design


def _labels(labels, rows, classes):
    labels = np.asarray(labels)
    if (labels.shape != (rows,) or labels.dtype.kind not in "iuf" or not np.isfinite(labels).all()
            or not np.equal(labels, np.floor(labels)).all() or (labels < 0).any() or (labels >= classes).any()):
        raise ValueError("Expected integral multiclass labels in range")
    return labels.astype(np.int64)


def softmax_objective(coefficients, design, labels, penalty):
    from scipy.special import logsumexp
    coefficients = np.asarray(coefficients, dtype=np.float64)
    baseline, temperature = design["baseline"], design["temperature"]
    rows, classes = baseline.shape
    labels = _labels(labels, rows, classes)
    if (coefficients.shape != (design["offsets"][-1],) or not np.isfinite(coefficients).all()
            or not np.isfinite(penalty) or penalty <= 0):
        raise ValueError("Expected finite coefficients and positive ridge")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        margins = baseline.copy()
        for c, matrix in enumerate(design["matrices"]):
            start, end = design["offsets"][c:c + 2]
            margins[:, c] += matrix @ coefficients[start:end]
        logits = margins / temperature
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_normalizer = logsumexp(shifted, axis=1)
        loss = (log_normalizer - shifted[np.arange(rows), labels]).sum()
        probabilities = np.exp(shifted - log_normalizer[:, None])
        probabilities[np.arange(rows), labels] -= 1.
        gradient = np.empty_like(coefficients)
        for c, matrix in enumerate(design["matrices"]):
            start, end = design["offsets"][c:c + 2]
            gradient[start:end] = (matrix.T @ (probabilities[:, c] / temperature)
                                   + penalty * coefficients[start:end]) / rows
        value = (loss + .5 * penalty * (coefficients @ coefficients)) / rows
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite softmax objective or gradient")
    return float(value), gradient


def solve_coefficients(design, labels, penalty, *, maxiter=2000, ftol=1e-13,
                       gtol=1e-9, gradient_gate=1e-6):
    from scipy.optimize import minimize
    zero = np.zeros(design["offsets"][-1], dtype=np.float64)
    args = (design, labels, penalty)
    initial, _ = softmax_objective(zero, *args)
    result = minimize(softmax_objective, zero, args=args, method="L-BFGS-B", jac=True,
                      options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol, "maxls": 50, "maxcor": 20})
    final, gradient = softmax_objective(result.x, *args)
    norm = float(np.max(np.abs(gradient)))
    if (not result.success or not np.isfinite(result.x).all() or final > initial + 1e-12
            or not np.isfinite(norm) or norm > gradient_gate):
        raise RuntimeError(f"Softmax correction did not converge: status={result.status}, gradient={norm}")
    return result.x, {"method": "L-BFGS-B", "penalty": float(penalty), "temperature": design["temperature"],
        "objective": "(sum softmax CE((raw baseline+A delta)/fixed T)+lambda/2*||delta||^2)/rows",
        "iterations": int(result.nit), "evaluations": int(result.nfev), "status": int(result.status),
        "initial_objective": initial, "final_objective": final, "maximum_absolute_mean_gradient": norm,
        "gradient_gate": gradient_gate, "maxiter": maxiter, "ftol": ftol, "gtol": gtol,
        "maxls": 50, "maxcor": 20, "initial_coefficients": "all zero", "dense_hessian": False}


def corrections_by_tree(design, coefficients, shrinkage=1.):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if (coefficients.shape != (design["offsets"][-1],) or not np.isfinite(coefficients).all()
            or not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1):
        raise ValueError("Invalid coefficients/shrinkage")
    classes = design["baseline"].shape[1]
    result = [{int(leaf): np.zeros(classes) for leaf in ids} for ids in design["maps"]]
    for c, trees in enumerate(design["trees_by_class"]):
        position = design["offsets"][c]
        for tree in trees:
            for leaf in design["maps"][tree]:
                result[tree][int(leaf)][c] = shrinkage * coefficients[position]
                position += 1
        if position != design["offsets"][c + 1]:
            raise ValueError("Coefficient layout changed")
    return result


def correction_values(leaves, updates, classes):
    leaves = np.asarray(leaves)
    if leaves.ndim != 2 or leaves.shape[1] != len(updates):
        raise ValueError("Tree count changed")
    result = np.zeros((len(leaves), classes))
    for tree, mapping in enumerate(updates):
        if not mapping:
            continue
        ids = np.asarray(sorted(mapping), dtype=np.int64)
        values = np.asarray([mapping[int(leaf)] for leaf in ids], dtype=np.float64)
        positions = np.searchsorted(ids, leaves[:, tree])
        found = positions < len(ids)
        found[found] &= ids[positions[found]] == leaves[found, tree]
        result[found] += values[positions[found]]
    return result


def _topology(document, library):
    result = copy.deepcopy(document)
    if library == "ctboost":
        for tree in result["trees"]:
            for node in tree["nodes"]:
                if node["is_leaf"]:
                    node.pop("leaf_weight", None)
                    node.pop("leaf_weights", None)
    elif library == "xgboost":
        for tree in result["learner"]["gradient_booster"]["model"]["trees"]:
            if int(tree["tree_param"]["size_leaf_vector"]) > 1:
                tree["leaf_weights"] = [None] * len(tree["leaf_weights"])
            else:
                for leaf, child in enumerate(tree["left_children"]):
                    if child == -1:
                        tree["split_conditions"][leaf] = None
    else:
        def strip(node):
            if "leaf_value" in node:
                node.pop("leaf_value")
            else:
                strip(node["left_child"])
                strip(node["right_child"])
        for tree in result["tree_info"]:
            strip(tree["tree_structure"])
    return result


def fold_corrections(payload, updates):
    classes, supports, document = _metadata(payload)
    library, original = payload["library"], payload["model"]
    if len(updates) != len(supports):
        raise ValueError("Tree count changed")
    for support, mapping in zip(supports, updates):
        for leaf, delta in mapping.items():
            delta = np.asarray(delta)
            if type(leaf) is not int or delta.shape != (classes,) or not np.isfinite(delta).all():
                raise ValueError("Expected finite class-vector leaf corrections")
            if support != -1 and np.any(np.delete(delta, support) != 0):
                raise ValueError("Scalar tree correction changed native class support")
    before_hash = _hash(_topology(document, library))
    if library == "ctboost":
        trees = document["trees"]
    elif library == "xgboost":
        trees = document["learner"]["gradient_booster"]["model"]["trees"]
    else:
        import lightgbm as lgb
        trees = document["tree_info"]
        model = lgb.Booster(model_str=original.model_to_string(num_iteration=-1))
    error_sum, old_sum, new_sum = np.zeros(classes), np.zeros(classes), np.zeros(classes)
    zero = np.zeros(classes)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for t, (tree, support, mapping) in enumerate(zip(trees, supports, updates)):
            if library == "ctboost":
                valid = [i for i, node in enumerate(tree["nodes"]) if node["is_leaf"]]
                iteration = t if support == -1 else t // classes
                rates = document["tree_learning_rates"]
                rate = np.float32(rates[iteration] if rates else document["learning_rate"])
                if not np.isfinite(rate) or rate <= 0:
                    raise ValueError("Invalid learning rate for leaf fold")
            elif library == "xgboost":
                valid = [i for i, child in enumerate(tree["left_children"]) if child == -1]
            else:
                valid = list(range(tree["num_leaves"]))
            if not set(mapping).issubset(valid):
                raise ValueError("Correction references a nonexistent leaf")
            errors, olds, news = [], [], []
            for leaf in valid:
                delta = np.asarray(mapping.get(leaf, zero), dtype=np.float64)
                old, new = np.zeros(classes), np.zeros(classes)
                for c in (range(classes) if support == -1 else (support,)):
                    if library == "ctboost":
                        node = tree["nodes"][leaf]
                        weight = node["leaf_weights"][c] if support == -1 else node["leaf_weight"]
                        old[c] = np.float32(rate * np.float32(weight))
                        changed = np.float32(float(weight) + delta[c] / float(rate)) if delta[c] else np.float32(weight)
                        if support == -1:
                            node["leaf_weights"][c] = float(changed)
                        else:
                            node["leaf_weight"] = float(changed)
                        new[c] = np.float32(rate * changed)
                    elif library == "xgboost":
                        key = "leaf_weights" if support == -1 else "split_conditions"
                        index = classes * tree["right_children"][leaf] + c if support == -1 else leaf
                        old[c] = np.float32(tree[key][index])
                        new[c] = np.float32(old[c] + delta[c])
                        tree[key][index] = float(new[c])
                    else:
                        old[c] = original.get_leaf_output(t, leaf)
                        new[c] = old[c] + delta[c]
                        if delta[c]:
                            model.set_leaf_output(t, leaf, new[c])
                errors.append(np.abs(new - old - delta)); olds.append(np.abs(old)); news.append(np.abs(new))
            error_sum += np.max(errors, axis=0)
            old_sum += np.max(olds, axis=0)
            new_sum += np.max(news, axis=0)
    if library == "ctboost":
        import ctboost
        model = ctboost.Booster(ctboost._core.GradientBooster.from_state(document),
                               feature_pipeline=original._feature_pipeline,
                               training_metadata=copy.deepcopy(original._training_metadata))
        base = np.asarray(document["base_score"], dtype=np.float64)
    elif library == "xgboost":
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(bytearray(json.dumps(document).encode()))
        model.set_param({"nthread": 1})
        base = np.atleast_1d(np.asarray(json.loads(document["learner"]["learner_model_param"]["base_score"]), dtype=np.float64))
    else:
        base = np.zeros(classes)
    candidate = dict(payload, model=model)
    _, after_supports, after = _metadata(candidate)
    after_hash = _hash(_topology(after, library))
    if before_hash != after_hash or supports != after_supports:
        raise AssertionError("Native fields other than leaf outputs changed")
    epsilon = np.finfo(np.float64 if library == "lightgbm" else np.float32).eps
    operations = 2 * len(updates) + 4
    if operations * epsilon >= 1:
        raise ValueError("Rounding bound outside supported range")
    gamma = operations * epsilon / (1 - operations * epsilon)
    bounds = 4 * (error_sum + gamma * (old_sum + new_sum + 2 * np.abs(base))) + 1e-12
    if not np.isfinite(bounds).all():
        raise ValueError("Nonfinite rounding bound")
    receipt = {"draft_only": DRAFT_ONLY, "library": library,
               "topology_sha256_before": before_hash, "topology_sha256_after": after_hash,
               "physical_trees": len(updates), "tree_classes": supports,
               "coefficient_rounding_absolute_bound": error_sum.tolist(),
               "prediction_oracle_absolute_tolerance_by_class": bounds.tolist(),
               "fixed_rtol": FIXED_RTOL, "fixed_atol": FIXED_ATOL,
               "temperature": _temperature(payload), "inference_only": True}
    candidate["fixed_multiclass_leaf_refit"] = receipt
    return candidate, receipt


def verify_fold(before, after, frame, updates, receipt, training_prepared=None):
    from scipy.special import softmax
    leaves = leaf_indices(before, frame, training_prepared)
    np.testing.assert_array_equal(leaves, leaf_indices(after, frame, training_prepared))
    raw = predict_raw(before, frame, training_prepared)
    expected = raw + correction_values(leaves, updates, raw.shape[1])
    actual = predict_raw(after, frame, training_prepared)
    differences = np.max(np.abs(actual - expected), axis=0)
    if (differences > np.asarray(receipt["prediction_oracle_absolute_tolerance_by_class"])).any():
        raise AssertionError("Native margins differ from leaf oracle rounding bound")
    np.testing.assert_allclose(actual, expected, rtol=FIXED_RTOL, atol=FIXED_ATOL)
    temperature = _temperature(before)
    if temperature != _temperature(after):
        raise AssertionError("Fixed temperature changed")
    calibrated = predict(after, frame, training_prepared)
    np.testing.assert_array_equal(calibrated, actual / temperature)
    np.testing.assert_allclose(softmax(calibrated, axis=1), softmax(expected / temperature, axis=1),
                               rtol=FIXED_RTOL, atol=FIXED_ATOL)
    return {"leaf_routing_exact": True, "temperature_applied_once": True,
            "oracle_max_absolute_difference_by_class": differences.tolist()}
