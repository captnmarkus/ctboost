# Bounded tensor-routing experiment, 16 October 2026

The NumPy GEMM evaluator preserves the original leaf IDs and every raw-score
float32 bit on all tested batches, but is slower than native CTBoost. It remains
an experiment; no production backend or training logic changed.

| Rows | Native ndarray ms | Native Pool ms | Tensor ndarray ms | Tensor prepared bins ms | Tensor/native ndarray |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.03439 | 0.00395 | 0.63074 | 0.47306 | 18.34x slower |
| 64 | 0.07160 | 0.03535 | 0.85607 | 0.66079 | 11.96x slower |
| 4096 | 5.24040 | 5.17187 | 11.13183 | 9.51990 | 2.12x slower |

Each cell is the median of seven warmed timing blocks. Method order reverses
between successive blocks; each block performs 30, 10, or 3 calls for batches
1, 64, and 4096 respectively. All loaded thread pools were restricted to one
thread, with process affinity on logical CPU 6. These measurements ran after
the separate native ABBA inference experiment finished and before subsequent
accuracy export checks. Raw timing blocks are in [results.json](results.json).

The ndarray paths include their respective input preparation and quantization.
The native Pool path excludes Pool construction but still quantizes input.
The tensor prepared-bin path excludes quantization, giving it a scope advantage;
it is nevertheless 1.84x slower than native Pool prediction at 4096 rows.
Model loading and static tensor compilation are excluded from every warm result.
This numeric-only fixture has no categorical/text feature pipeline.

## Construction and parity

The unchanged learner trained 64 scalar regression trees with 1,280 total nodes
on 1,024 deterministic synthetic rows and 12 numeric features. Parameters and
fixture/model hashes were frozen in [plan.json](plan.json) before timing.
The model uses depth at most five, 64 bins, and missing-value mode Min. Input
probes include exact quantization boundaries, adjacent float32 values, and an
all-NaN row. Native leaf IDs and raw float32 bit patterns agree at batches
1, 64, and 4096; all four timed paths also pass raw-bit parity before timing.

For each tree, the prototype forms a row-by-internal-node matrix `P` of left
decisions. Its internal-node-by-leaf matrix `S` contains +1 for a required left
branch, -1 for a required right branch, and zero elsewhere. `P @ S` equals the
number of required left branches exactly when a row matches that leaf's path.
Each incorrect branch reduces the score by one. All operands and intermediate
path sums are exactly representable small integers in float32. The prototype
checks that each row matches exactly one leaf and gathers the original leaf ID.
Leaf updates are multiplied in float32 once and added in original tree order.

Feature selection uses quantized-bin gathers, avoiding the `0 * NaN` problem
that raw-input multiplication by a one-hot feature-selection matrix would cause.
Only numeric, scalar-output trees are accepted. The evaluator caps trees at
128 leaves/255 nodes and processes one tree at a time, avoiding a dense
forest-wide block-diagonal matrix.

At 4096 rows the matrices perform **33,628,160 multiply-accumulates**, plus
2,490,368 internal-node comparisons. Native traversal needs at most 1,171,456
path comparisons for this model. The compiled NumPy arrays occupy 46,984 bytes,
prepared bins occupy 98,304 bytes, and the conservative temporary-array estimate
is 1,249,280 bytes. These estimates exclude Python objects, allocator overhead,
BLAS workspace, and native model/cache memory. They are work/storage counts,
not equivalent-operation counts or measured peak RSS.

## Runtime and reproduction

The run used Python 3.10.14, NumPy 1.26.4, and its OpenBLAS 0.3.23.dev on Windows.
The separately loaded SciPy BLAS was also constrained to one thread. CTBoost
reports 0.1.61 and uses the freshly built baseline native binary SHA-256
`41ee0e2c10704693f18ae86f77039f49a6a6ddecc6cf77fe310df2121b2495af`.
The runtime, Python-source hashes, model hash, thread-pool metadata, and script
hash are retained in the plan and results. The selected optimized native build
was not used, keeping this experiment independent of optimization selection.

Run the source script with a fresh output directory and the same isolated
package/runtime, after other performance work is idle:

```powershell
& PYTHON -I benchmarks/tensor_routing_scout.py prepare --output NEW_OUTPUT --package-root ISOLATED_PACKAGE --cpu 6
& PYTHON -I benchmarks/tensor_routing_scout.py measure --output NEW_OUTPUT --package-root ISOLATED_PACKAGE --cpu 6
```

The synthetic `model.ctb` and `input.npy` remain local ignored artifacts;
the deterministic preparation command recreates them. Changing the script or
runtime after preparation requires a fresh experiment directory.

This result tests one CPU tensor formulation on one synthetic model. It does
not rule out GPU acceleration, compiled tensor kernels, other shapes, or better
matrix algorithms. It does show that converting this CTBoost model into dense
matrix operations with the available BLAS does not make it faster. This is not
an implementation or empirical validation of OpenAI's asymptotic multiplication
results, and supplies no accuracy or leaderboard evidence.
