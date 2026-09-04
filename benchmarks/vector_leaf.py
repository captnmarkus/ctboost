"""Compare scalar and vector storage using identical conditional tree training.

Run: python -m benchmarks.vector_leaf --rows 3000 --classes 8 --rounds 24
Timings are local measurements, not general performance guarantees.
"""

import argparse
import json
import time

import numpy as np

import ctboost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--seed", type=int, default=418)
    args = parser.parse_args()
    if args.rows < 100 or args.classes < 3 or args.rounds < 1:
        parser.error("rows >= 100, classes >= 3 and rounds >= 1 are required")
    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.rows, 12)).astype(np.float32)
    coefficients = rng.normal(size=(4, args.classes))
    y = np.argmax(X[:, :4] @ coefficients, axis=1).astype(np.float32)
    split = int(0.8 * args.rows)
    train_pool = ctboost.Pool(X[:split], y[:split])
    rows = []
    reference = None
    for strategy in ["one_output_per_tree", "multi_output_tree"]:
        start = time.perf_counter()
        booster = ctboost.train(
            train_pool,
            {
                "objective": "MultiClass",
                "num_classes": args.classes,
                "multi_strategy": strategy,
                "max_depth": 4,
                "alpha": 0.05,
                "learning_rate": 0.15,
                "random_seed": args.seed,
            },
            num_boost_round=args.rounds,
        )
        train_seconds = time.perf_counter() - start
        # Warm the prediction path before reporting the median of five runs.
        prediction = booster.predict(X[split:])
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            booster.predict(X[split:])
            timings.append(time.perf_counter() - start)
        if reference is None:
            reference = prediction
        else:
            np.testing.assert_array_equal(prediction, reference)
        state = booster._handle.export_state()
        log_prob = prediction.astype(np.float64)
        log_prob -= log_prob.max(axis=1, keepdims=True)
        log_prob -= np.log(np.exp(log_prob).sum(axis=1, keepdims=True))
        rows.append(
            {
                "multi_strategy": strategy,
                "train_seconds": train_seconds,
                "predict_seconds_median": float(np.median(timings)),
                "trees": len(state["trees"]),
                "splits": sum(
                    not n["is_leaf"] for t in state["trees"] for n in t["nodes"]
                ),
                "state_json_bytes": len(
                    json.dumps(state, separators=(",", ":")).encode()
                ),
                "held_out_logloss": float(
                    -log_prob[np.arange(len(log_prob)), y[split:].astype(int)].mean()
                ),
            }
        )
    print(
        json.dumps(
            {"config": vars(args), "predictions_identical": True, "results": rows},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
