"""Bounded NumPy GEMM routing experiment for an unchanged CTBoost ensemble.

Prepare fixtures, then measure only when other timed experiments are idle.
This numeric, scalar-output prototype is deliberately not a production backend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

for _name in ("CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def runtime(package_root):
    # An editable-install finder otherwise takes precedence over sys.path.
    sys.meta_path = [finder for finder in sys.meta_path
                     if type(finder).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(package_root))
    import ctboost
    from threadpoolctl import threadpool_info
    for path in (ctboost.__file__, ctboost._core.__file__):
        if not Path(path).resolve().is_relative_to(package_root):
            raise RuntimeError("CTBoost resolved outside the frozen isolated package")
    return ctboost, {
        "python": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(), "package_path": ctboost.__file__,
        "native_path": ctboost._core.__file__, "native_sha256": digest(ctboost._core.__file__),
        "ctboost_version": ctboost.__version__,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "psutil", "threadpoolctl")},
        "blas": [{key: item.get(key) for key in ("internal_api", "version", "architecture", "num_threads", "filepath")}
                 for item in threadpool_info() if item["user_api"] == "blas"],
        "python_sources_sha256": {str(path.relative_to(package_root)): digest(path)
                                   for path in sorted((package_root / "ctboost").rglob("*.py"))},
    }


class TensorRouter:
    def __init__(self, state):
        if len(state["base_score"]) != 1 or state["multi_strategy"] != "one_output_per_tree":
            raise ValueError("This bounded prototype supports scalar outputs only")
        self.schema = state["quantization_schema"]
        if any(self.schema["categorical_mask"]):
            raise ValueError("This bounded experiment covers numeric features only")
        self.base_score = np.float32(state["base_score"][0])
        self.trees = []
        self.active_features = set()
        for tree_index, tree in enumerate(state["trees"]):
            nodes = tree["nodes"]
            internal = [i for i, node in enumerate(nodes) if not node["is_leaf"]]
            leaves = [i for i, node in enumerate(nodes) if node["is_leaf"]]
            if len(leaves) > 128 or len(nodes) > 255:
                raise ValueError("Tree exceeds the predeclared 128-leaf/255-node bound")
            positions = {node_id: index for index, node_id in enumerate(internal)}
            leaf_positions = {node_id: index for index, node_id in enumerate(leaves)}
            signs = np.zeros((len(internal), len(leaves)), dtype=np.float32)
            maximum_depth = 0
            pending = [(0, [])]
            while pending:
                node_id, path = pending.pop()
                node = nodes[node_id]
                if node["is_leaf"]:
                    maximum_depth = max(maximum_depth, len(path))
                    for parent, sign in path:
                        signs[positions[parent], leaf_positions[node_id]] = sign
                else:
                    pending.append((int(node["left_child"]), path + [(node_id, 1)]))
                    pending.append((int(node["right_child"]), path + [(node_id, -1)]))
            # Sums of these small integers are exactly representable in FP32.
            if maximum_depth >= 2**24:
                raise ValueError("Path sums would exceed exact float32 integer range")
            features = np.asarray([nodes[i]["split_feature_id"] for i in internal], dtype=np.intp)
            self.active_features.update(features.tolist())
            rates = state["tree_learning_rates"]
            rate = np.float32(rates[tree_index] if tree_index < len(rates) else state["learning_rate"])
            weights = np.asarray([nodes[i]["leaf_weight"] for i in leaves], dtype=np.float32)
            self.trees.append({
                "features": features,
                "thresholds": np.asarray([nodes[i]["split_bin_index"] for i in internal], dtype=np.uint16),
                "signs": signs,
                "required_left": (signs > 0).sum(axis=0).astype(np.float32),
                "leaf_ids": np.asarray(leaves, dtype=np.int32),
                "updates": np.multiply(rate, weights, dtype=np.float32),
                "maximum_depth": maximum_depth,
            })

    def quantize(self, data):
        from ctboost.explain import _quantize_feature
        data = np.asarray(data, dtype=np.float32)
        bins = np.zeros(data.shape, dtype=np.uint16)
        for feature in sorted(self.active_features):
            bins[:, feature] = _quantize_feature(data[:, feature], self.schema, feature)
        return bins

    def predict_bins(self, bins, return_leaves=False):
        prediction = np.full(len(bins), self.base_score, dtype=np.float32)
        leaves = np.empty((len(bins), len(self.trees)), dtype=np.int32) if return_leaves else None
        for tree_index, tree in enumerate(self.trees):
            if tree["features"].size:
                decisions = (bins[:, tree["features"]] <= tree["thresholds"]).astype(np.float32)
                scores = decisions @ tree["signs"]
                matches = scores == tree["required_left"]
                if return_leaves and not np.all(matches.sum(axis=1) == 1):
                    raise AssertionError("Each row must match exactly one original leaf")
                selected = matches.argmax(axis=1)
            else:
                selected = np.zeros(len(bins), dtype=np.intp)
            # Sequential FP32 addition in original tree order, not a BLAS reduction.
            np.add(prediction, tree["updates"][selected], out=prediction)
            if return_leaves:
                leaves[:, tree_index] = tree["leaf_ids"][selected]
        return (prediction, leaves) if return_leaves else prediction

    def predict(self, data):
        return self.predict_bins(self.quantize(data))

    def estimate(self, rows, features):
        sizes = [(len(tree["features"]), len(tree["leaf_ids"])) for tree in self.trees]
        arrays = [value for tree in self.trees for value in tree.values() if isinstance(value, np.ndarray)]
        return {
            "compiled_array_bytes": sum(array.nbytes for array in arrays),
            "prepared_bins_bytes": rows * features * 2,
            "score_only_temporary_array_upper_bound_bytes": rows * features * 2 + rows * 4 +
                max((rows * (internal * 7 + leaves * 5 + 20) for internal, leaves in sizes), default=0),
            "gemm_multiply_accumulates": rows * sum(internal * leaves for internal, leaves in sizes),
            "all_internal_comparisons": rows * sum(internal for internal, _ in sizes),
            "native_path_comparisons_upper_bound": rows * sum(tree["maximum_depth"] for tree in self.trees),
            "matrix_dimensions_per_tree": [[rows, internal, leaves] for internal, leaves in sizes],
            "estimate_note": "Array storage only; excludes Python objects, allocator overhead, BLAS workspace and native caches. Tree-at-a-time matrices, no forest block diagonal.",
        }


def prepare(args):
    if args.output.exists():
        raise ValueError("Use a fresh output directory to preserve previous attempts")
    ctboost, identity = runtime(args.package_root)
    rng = np.random.default_rng(20261016)
    train = rng.normal(size=(1024, 12)).astype(np.float32)
    target = (2 * train[:, 0] + np.sin(3 * train[:, 1]) + train[:, 2] * train[:, 3]
              + 0.1 * rng.normal(size=len(train))).astype(np.float32)
    train[rng.random(train.shape) < 0.015] = np.nan
    params = {"objective": "RMSE", "iterations": 64, "learning_rate": 0.05,
              "max_depth": 5, "max_bins": 64, "alpha": 0.05, "random_seed": 20261016,
              "nan_mode": "Min", "task_type": "CPU", "boost_from_average": True}
    model = ctboost.train(ctboost.Pool(train, target), params)
    args.output.mkdir(parents=True)
    model_path = args.output / "model.ctb"
    model.save_model(model_path)
    state = model._handle.export_state()
    router = TensorRouter(state)
    data = rng.normal(size=(4096, 12)).astype(np.float32)
    data[rng.random(data.shape) < 0.015] = np.nan
    # Include exact bin boundaries and adjacent FP32 values in every measured batch.
    for feature in sorted(router.active_features):
        begin, end = router.schema["cut_offsets"][feature:feature + 2]
        if end > begin:
            cut = np.float32(router.schema["cut_values"][begin])
            data[0, feature] = cut
            data[1, feature] = np.nextafter(cut, np.float32(-np.inf))
            data[2, feature] = np.nextafter(cut, np.float32(np.inf))
    data[3, :] = np.nan
    np.save(args.output / "input.npy", data)
    parity = []
    for batch in (1, 64, 4096):
        raw, leaves = router.predict_bins(router.quantize(data[:batch]), return_leaves=True)
        expected = model.predict(data[:batch])
        np.testing.assert_array_equal(raw.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(leaves, model.predict_leaf_index(data[:batch]))
        parity.append({"batch": batch, "leaf_ids_exact": True, "raw_margin_bits_exact": True})
    write(args.output / "plan.json", {
        "protocol": "tensor_routing_20261016_v1", "script_sha256": digest(__file__),
        "runtime": identity, "model_sha256": digest(model_path),
        "input_sha256": digest(args.output / "input.npy"), "params": params,
        "model_trees": len(state["trees"]), "model_nodes": sum(len(t["nodes"]) for t in state["trees"]),
        "batches": [1, 64, 4096], "cpu": args.cpu, "blocks": 7,
        "calls_per_block": {"1": 30, "64": 10, "4096": 3}, "parity": parity,
        "scopes": {
            "native_array": "Native model.predict(float32 ndarray): includes Pool creation and native quantization; fitted model cache warm.",
            "native_pool": "Native model.predict(prebuilt Pool): includes native quantization; excludes Pool construction.",
            "tensor_array": "NumPy router: includes numeric quantization, all-node comparisons, GEMM, leaf lookup and ordered sum; excludes static model compilation.",
            "tensor_bins": "NumPy router: prepared uint16 bins; includes all-node comparisons, GEMM, leaf lookup and ordered sum; excludes quantization and static compilation.",
        },
        "limitations": "One small deterministic synthetic regression model; numeric scalar outputs only, no categorical/vector outputs, raw feature pipeline or GPU. Timings do not measure accuracy or asymptotic OpenAI algorithms. Prepared bins are an advantage unavailable in the measured native API; compare scope explicitly.",
    })
    print(json.dumps({"prepared": str(args.output), "parity": parity}))


def measure(args):
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits
    ctboost, identity = runtime(args.package_root)
    plan = json.loads((args.output / "plan.json").read_text(encoding="utf-8"))
    if (args.output / "results.json").exists():
        raise ValueError("Preserve measured results; use a fresh experiment directory")
    if plan["script_sha256"] != digest(__file__) or plan["runtime"] != identity:
        raise ValueError("Frozen runner or runtime changed")
    for file, key in (("model.ctb", "model_sha256"), ("input.npy", "input_sha256")):
        if digest(args.output / file) != plan[key]:
            raise ValueError("Frozen model or fixture changed")
    psutil.Process().cpu_affinity([plan["cpu"]])
    model = ctboost.Booster.load_model(args.output / "model.ctb")
    router = TensorRouter(model._handle.export_state())
    data = np.load(args.output / "input.npy")
    results = []
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if any(pool["num_threads"] != 1 for pool in pools):
            raise RuntimeError("Thread pools are not restricted to one thread")
        for batch in plan["batches"]:
            rows = data[:batch]
            pool, bins = ctboost.Pool(rows), router.quantize(rows)
            expected = model.predict(pool)
            raw, leaves = router.predict_bins(bins, return_leaves=True)
            np.testing.assert_array_equal(raw.view(np.uint32), expected.view(np.uint32))
            np.testing.assert_array_equal(leaves, model.predict_leaf_index(pool))
            methods = {"native_array": lambda: model.predict(rows), "native_pool": lambda: model.predict(pool),
                       "tensor_array": lambda: router.predict(rows), "tensor_bins": lambda: router.predict_bins(bins)}
            timings = {name: [] for name in methods}
            for method in methods.values():
                for _ in range(3):
                    np.testing.assert_array_equal(method().view(np.uint32), expected.view(np.uint32))
            calls = plan["calls_per_block"][str(batch)]
            for block in range(plan["blocks"]):
                order = list(methods) if block % 2 == 0 else list(reversed(methods))
                for name in order:
                    started = time.perf_counter_ns()
                    for _ in range(calls):
                        methods[name]()
                    timings[name].append((time.perf_counter_ns() - started) / calls / 1e6)
            medians = {name: statistics.median(values) for name, values in timings.items()}
            results.append({"batch": batch, "milliseconds_per_call_blocks": timings, "median_ms": medians,
                            "tensor_array_over_native_array": medians["tensor_array"] / medians["native_array"],
                            "tensor_bins_over_native_pool": medians["tensor_bins"] / medians["native_pool"],
                            "raw_margin_bits_exact": True, "leaf_ids_exact": True,
                            "estimates": router.estimate(batch, data.shape[1])})
    write(args.output / "results.json", {"plan_sha256": digest(args.output / "plan.json"),
          "runtime": identity, "controlled_thread_pools": pools, "cpu_affinity": psutil.Process().cpu_affinity(),
          "results": results, "scopes": plan["scopes"], "limitations": plan["limitations"]})
    print(json.dumps([{key: row[key] for key in ("batch", "median_ms", "tensor_array_over_native_array")}
                      for row in results], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "measure"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--cpu", type=int, default=6)
    arguments = parser.parse_args()
    arguments.output = arguments.output.resolve()
    arguments.package_root = arguments.package_root.resolve()
    {"prepare": prepare, "measure": measure}[arguments.stage](arguments)
