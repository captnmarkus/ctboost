"""Read-only selected-model diagnostics on training/stopping rows, never test."""

import argparse
import importlib.util
from pathlib import Path
import pickle

SOURCE = Path(__file__).with_name("fair_accuracy_scout.py")
spec = importlib.util.spec_from_file_location("accuracy_base", SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def tree_depth(nodes, index=0):
    if nodes[index]["is_leaf"]:
        return 0
    return 1 + max(tree_depth(nodes, nodes[index][key]) for key in ("left_child", "right_child"))


def run(args):
    import numpy as np
    plan = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    if base.bootstrap(args) != plan["runtime"] or frozen["plan_sha256"] != base.digest(args.previous / "plan.json"):
        raise ValueError("Runtime or frozen evidence changed")
    if args.output.exists():
        raise ValueError("Preserve the previous diagnostics")
    rows = []
    for case in plan["cases"]:
        train, stop = base.load_role(case, "train"), base.load_role(case, "stop")
        for library in base.LIBRARIES:
            chosen = next(s for s in frozen["selections"] if s["dataset"] == case["dataset_name"] and s["library"] == library)
            if base.digest(chosen["model_path"]) != chosen["model_sha256"]:
                raise ValueError("Selected model changed")
            with Path(chosen["model_path"]).open("rb") as stream:
                payload = pickle.load(stream)
            row = {"dataset": case["dataset_name"], "library": library,
                "model_sha256": chosen["model_sha256"], "model_pickle_bytes": chosen["model_bytes"],
                "source_wave": chosen["source_wave"], "config_index": chosen["best"]["config_index"],
                "retained_rounds": chosen["best"]["retained_rounds"],
                "train_error_at_inference_transform": base.error(base.predict(payload, train["X"]), train["y"], case["problem_type"]),
                "stop_error": base.error(base.predict(payload, stop["X"]), stop["y"], case["problem_type"]),
                "input_features": train["X"].shape[1], "categorical_input_columns": len(payload["schema"]),
                "train_vocabulary_sizes": {str(k): len(v) for k, v in payload["schema"].items()}}
            row["stop_minus_train_error"] = row["stop_error"] - row["train_error_at_inference_transform"]
            if library == "ctboost":
                state = payload["model"]._handle.export_state()
                trees = state["trees"]
                leaves = [sum(node["is_leaf"] for node in tree["nodes"]) for tree in trees]
                pipeline = payload["model"]._feature_pipeline
                pipeline_state = pipeline.to_state() if pipeline is not None else {}
                row.update(physical_trees=len(trees), physical_nodes=sum(len(t["nodes"]) for t in trees),
                    root_only_trees=sum(len(t["nodes"]) == 1 for t in trees), mean_leaves=float(np.mean(leaves)),
                    mean_realized_depth=float(np.mean([tree_depth(t["nodes"]) for t in trees])),
                    transformed_features=len(state["quantization_schema"]["num_bins_per_feature"]),
                    pipeline_state_counts={key: len(pipeline_state.get(key, [])) for key in ("categorical_states", "one_hot_states", "ctr_states")})
            rows.append(row)
    base.write(args.output, {"created_at": base.now(), "script_sha256": base.digest(__file__),
        "plan_sha256": base.digest(args.previous / "plan.json"), "frozen_selections_sha256": base.digest(args.previous / "frozen_selections.json"),
        "no_test_access": True, "caveat": "Training-row predictions use the inference transform, including CTBoost full fitted category statistics, not the ordered optimization-row transform. These gaps are descriptive, not unbiased training-loss estimates or a causal attribution. Pickle bytes include library-specific persistence contracts.", "results": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "previous", "package-root", "dependencies"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    run(parser.parse_args())
