"""Deterministic deferred baseline-resume limitation; synthetic data only."""
import hashlib
import json
from pathlib import Path
import sys

import ctboost
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


cases = []
native = Path(ctboost.__file__).parent
native = next(iter(list(native.glob("_core*.pyd")) + list(native.glob("_core*.so"))))
native_hash = digest(native)
for root_only in (False, True):
    for objective, strategy in (
        ("RMSE", "one_output_per_tree"),
        ("LogLoss", "one_output_per_tree"),
        ("MultiClass", "one_output_per_tree"),
        ("MultiClass", "multi_output_tree"),
    ):
        rng = np.random.default_rng(471)
        X = rng.normal(size=(96, 4)).astype(np.float32)
        if objective == "MultiClass":
            y = np.argmax(X[:, :3], axis=1).astype(np.float32)
            baseline = np.tile(np.array([0.17, -0.31, 0.23], dtype=np.float32), (96, 1))
        else:
            y = 2.7 * X[:, 0] - 1.3 * X[:, 1] + 0.21 * X[:, 2]
            if objective == "LogLoss":
                y = (y > 0).astype(np.float32)
            baseline = np.full(96, -0.34948291, dtype=np.float32)
        pool = ctboost.Pool(X, y, weight=np.linspace(0.3, 1.7, 96, dtype=np.float32), baseline=baseline)
        params = dict(objective=objective, learning_rate=0.123456789, max_depth=2,
                      min_data_in_leaf=1000 if root_only else 0, alpha=1.0, random_seed=19)
        if objective == "MultiClass":
            params.update(num_classes=3, multi_strategy=strategy)
        full = ctboost.train(pool, params, num_boost_round=7, eval_set=pool)
        initial = ctboost.train(pool, params, num_boost_round=3, eval_set=pool)
        resumed = ctboost.train(pool, params, num_boost_round=4, eval_set=pool, init_model=initial)
        a, b = np.asarray(full.predict(pool)), np.asarray(resumed.predict(pool))
        loss_a, loss_b = np.asarray(full.loss_history), np.asarray(resumed.loss_history)
        cases.append(dict(
            objective=objective, strategy=strategy, root_only=root_only,
            prediction_elements=a.size,
            prediction_bit_mismatches=int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
            max_absolute_prediction_difference=float(np.max(np.abs(a.astype(float) - b.astype(float)))),
            max_absolute_loss_history_difference=float(np.max(np.abs(loss_a - loss_b))),
            trees_equal=full._handle.export_state()["trees"] == resumed._handle.export_state()["trees"],
            uninterrupted_loss_history=loss_a.tolist(), resumed_loss_history=loss_b.tolist(),
            uninterrupted_train_eval_equal=full.loss_history == full.eval_loss_history,
            resumed_train_eval_equal=resumed.loss_history == resumed.eval_loss_history,
        ))
assert digest(native) == native_hash, "Installed native changed during verification"
receipt = dict(
    purpose="Deferred nonzero Pool.baseline warm-start rounding limitation; no production change",
    synthetic_only=True, no_benchmark_data=True,
    ctboost_version=ctboost.__version__, numpy_version=np.__version__, python=sys.version,
    ctboost_path=ctboost.__file__, native_path=str(native), native_sha256=native_hash,
    verifier_sha256=digest(__file__),
    mechanism="Fresh training starts with baseline then adds tree updates; resume reconstructs trees then adds baseline. Float addition order differs.",
    cases=cases,
)
target = Path(sys.argv[1])
target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps([{
    key: item[key] for key in ("objective", "strategy", "root_only", "prediction_bit_mismatches",
                               "max_absolute_prediction_difference", "max_absolute_loss_history_difference", "trees_equal")
} for item in cases], indent=2))
