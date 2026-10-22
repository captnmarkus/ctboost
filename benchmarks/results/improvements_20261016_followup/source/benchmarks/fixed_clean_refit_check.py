"""Pure guards for fresh binary/regression refits before replacement corrections.

No fitting, scoring, real data access or modification of the frozen refit helpers.
Matching nonleaf state cannot alone prove a model was freshly fitted: the runner
must construct clean models with inherited corrections disabled and freeze that
provenance. These checks never subtract an old correction from its fitted leaves.
"""
from __future__ import annotations

import hashlib
import json

from benchmarks import fixed_leaf_refit as frozen


def _hash(value):
    # Native exports may contain identical NaN training metadata or split markers.
    # Canonical JSON supplies a stable equality representation for those values.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=True).encode()).hexdigest()


def _snapshot(payload):
    library, problem, model = payload["library"], payload["problem"], payload["model"]
    if problem not in ("binary", "regression") or payload.get("temperature") is not None:
        raise ValueError("Expected binary/regression payload without temperature")
    schema = payload["schema"]
    if type(schema) is not dict:
        raise ValueError("Expected ordered category-schema dictionary")
    result = {"library": library, "problem": problem,
              "ordered_schema_sha256": _hash(list(schema.items()))}
    if library == "ctboost":
        document = model._handle.export_state()
        expected = "logloss" if problem == "binary" else "rmse"
        if (document["objective_name"].lower() != expected or model.prediction_dimension != 1
                or document["boosting_type"] != "GradientBoosting"
                or any(node.get("leaf_weights") for tree in document["trees"] for node in tree["nodes"])):
            raise ValueError("Expected scalar CTBoost model with matching native objective")
        counts = [sum(bool(node["is_leaf"]) for node in tree["nodes"]) for tree in document["trees"]]
        structural = frozen._ct_topology(document)
        pipeline = model._feature_pipeline
        result.update(feature_pipeline_sha256=_hash(None if pipeline is None else pipeline.to_state()),
                      data_schema_sha256=_hash(model.data_schema),
                      training_metadata_sha256=_hash(model._training_metadata))
    elif library == "xgboost":
        document = frozen._xgb_document(model)
        learner = document["learner"]
        expected = "binary:logistic" if problem == "binary" else "reg:squarederror"
        if (learner["objective"]["name"] != expected or learner["gradient_booster"]["name"] != "gbtree"
                or int(learner["learner_model_param"]["num_class"]) != 0):
            raise ValueError("Expected scalar XGBoost model with matching native objective")
        trees = learner["gradient_booster"]["model"]["trees"]
        if any(int(tree["tree_param"]["size_leaf_vector"]) > 1 for tree in trees):
            raise ValueError("Vector leaves are outside clean scalar refit checks")
        counts = [sum(child == -1 for child in tree["left_children"]) for tree in trees]
        structural = frozen._xgb_topology(document)
    elif library == "lightgbm":
        document = model.dump_model(num_iteration=-1)
        expected = "binary" if problem == "binary" else "regression"
        if document["objective"].split()[0] != expected or document["num_tree_per_iteration"] != 1:
            raise ValueError("Expected scalar LightGBM model with matching native objective")
        counts = [int(tree["num_leaves"]) for tree in document["tree_info"]]
        structural = frozen._lgb_topology(document)
    else:
        raise ValueError("Unknown library")
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("Expected nonempty native trees")
    result.update(native_state_sha256=_hash(document),
                  nonleaf_state_sha256=_hash(structural),
                  physical_trees=len(counts), native_leaf_counts=counts)
    return result


def verify_clean_refit(clean_payload, parent_payload):
    """Require exact schema/pipeline/nonleaf equality, allowing only leaf outputs.

    Full native export/dump fields, including bias, cut tables, feature names,
    categorical routes and historical training statistics, remain in the hash.
    Only scalar leaf outputs are stripped by the frozen regression helper.
    Python correction receipts in the parent are intentionally irrelevant.
    """
    if any(key in clean_payload for key in ("fixed_leaf_refit", "fixed_multiclass_leaf_refit")):
        raise ValueError("Clean refit payload still carries an inherited correction receipt")
    clean, parent = _snapshot(clean_payload), _snapshot(parent_payload)
    for name in ("library", "problem", "ordered_schema_sha256", "nonleaf_state_sha256",
                 "physical_trees", "native_leaf_counts", "feature_pipeline_sha256",
                 "data_schema_sha256", "training_metadata_sha256"):
        if clean.get(name) != parent.get(name):
            raise ValueError(f"Clean refit differs from parent in {name}")
    return {"exact_clean_refit_identity_except_scalar_leaf_outputs": True,
            "library": clean["library"], "problem": clean["problem"],
            "physical_trees": clean["physical_trees"], "total_native_leaves": sum(clean["native_leaf_counts"]),
            "clean_native_state_sha256": clean["native_state_sha256"],
            "parent_native_state_sha256": parent["native_state_sha256"],
            "nonleaf_state_sha256": clean["nonleaf_state_sha256"],
            "ordered_schema_sha256": clean["ordered_schema_sha256"],
            "feature_pipeline_sha256": clean.get("feature_pipeline_sha256"),
            "data_schema_sha256": clean.get("data_schema_sha256"),
            "training_metadata_sha256": clean.get("training_metadata_sha256"),
            "leaf_outputs_ignored": True, "old_correction_subtracted": False,
            "limitation": "Fresh fit and disabled-correction provenance must also be enforced by the runner"}


def memory_preflight(payload, rows, *, max_working_bytes, solver_kind):
    """Bound named scalar-refit array budgets before leaf prediction/design.

    This intentionally excludes native models, frames, pipeline transforms and
    Python/allocator overhead; a separate process-tree RSS guard is required.
    """
    if (type(rows) is not int or rows <= 0 or type(max_working_bytes) is not int
            or max_working_bytes <= 0 or solver_kind not in ("binary", "regression")
            or solver_kind != payload["problem"]):
        raise ValueError("Invalid frozen scalar memory preflight")
    snapshot = _snapshot(payload)
    counts = snapshot["native_leaf_counts"]
    trees = len(counts)
    columns = sum(min(rows, count) for count in counts)
    nonzeros = rows * trees
    arrays = {
        "native_leaf_result_and_int64_copy": 16 * nonzeros,
        "native_raw_result_and_float64_copy": 16 * rows,
        "observed_leaf_maps": 8 * columns,
        "resident_csr_data_indices_indptr": 16 * nonzeros + 8 * (rows + 1),
        "design_construction_temporary_budget": 24 * nonzeros + 32 * rows,
    }
    if solver_kind == "binary":
        maxcor = 20
        arrays.update(objective_workspace_budget=8 * 12 * rows,
            lbfgsb_float_and_integer_work_arrays=
                8 * (2 * maxcor * columns + 5 * columns + 11 * maxcor * maxcor + 8 * maxcor)
                + 4 * 3 * columns,
            optimizer_vector_budget=8 * 12 * columns + 4 * columns)
    else:
        arrays["lsqr_residual_and_vector_workspace_budget"] = 8 * (8 * rows + 8 * columns)
    total = sum(arrays.values())
    if total > max_working_bytes:
        raise MemoryError("Frozen scalar refit-array memory budget exceeded")
    return {"solver_kind": solver_kind, "rows": rows, "physical_trees": trees,
            "native_leaf_count": sum(counts), "parameters_upper_bound": columns,
            "csr_nonzeros": nonzeros, "array_budgets_bytes": arrays,
            "refit_array_budget_bytes": total, "max_working_bytes": max_working_bytes,
            "scope": "named conservative refit arrays, not whole-process RSS",
            "excluded": ["native model objects", "input frames", "categorical transform workspace",
                         "Python object/allocator overhead"], "no_dense_hessian": True}
