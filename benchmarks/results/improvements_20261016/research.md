# CTBoost improvement research — 16 October 2026

This investigation preserves CTBoost's conditional-inference design: test and
select a feature before optimizing its cut. The starting checkout is
`9e68e6d5a6fe4a402d7b465092f6690bdf979a94` (Python package version 0.1.61).
Execution optimizations must preserve fitted trees, routing, and score arithmetic.
Learning experiments remain opt-in and require held-out evaluation.

## The recent multiplication results

The user's reference is to a real, recent release. OpenAI published its
[mathematics collection on 6 October 2026](https://openai.com/index/sharing-ai-progress-in-mathematics/).
The [repository describes differing verification stages](https://github.com/openai/math/blob/main/README.md);
publication, formalization, independent review, and a useful software implementation
are separate claims.

| Result | Statement | Relevance to CTBoost |
|---|---|---|
| [Integer multiplication below n log n, 23 September 2026](https://github.com/openai/math/blob/main/preprints/Integer-multiplication-below-n-log-n-September-23-2026/README.md) | Claims exact multiplication of two n-bit integers in O(n (log n)^(1 − 2^-182)) bit operations on a fixed multitape Turing machine. | Concerns arbitrarily long integers. CTBoost's 8/16-bit bins and fixed-width indices do not become faster through this asymptotic construction. |
| [Matrix exponent at most 9/4, 2 October 2026](https://github.com/openai/math/blob/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/README.md) | Bounds the complex matrix-multiplication exponent by 2.25: O_epsilon(n^(2.25 + epsilon)) arithmetic operations. | Does not establish a faster finite-size BLAS kernel. CTBoost prediction is tree routing, not a large dense matrix product. |

The [official result catalogue](https://github.com/openai/math/blob/main/CONTENTS.md)
did not link a Lean formalization for the integer result when checked on
16 October. Treat that result as a preprint claim. The
[matrix formalization's stated scope](https://github.com/openai/math/blob/main/lean/docs/107.md)
explicitly distinguishes arithmetic complexity from bit complexity and practical
crossover sizes. This investigation has not independently verified either proof.

## What the code actually does

`src/core/booster_predict.cpp` already caches learning-rate-weighted leaf updates.
The warm traversal repeatedly loads a node, reads an input bin, compares a
threshold or tests a category bit, chooses a child, and eventually adds a leaf
update. Reducing dependent memory accesses and branches is more directly relevant
than reducing multiplication complexity. Cached nodes occupy at most 32 bytes;
matching scalar multiclass trees already share traversal in the cache.

At the baseline, `src/core/booster_prediction.cpp` quantizes active features for
each prediction request, allocates a full feature-by-row bin buffer, and calls
`ApplyQuantizationSchema`, which copies schema vectors. The retained
[inference change](inference/README.md) removes that metadata copy from cached
prediction. The full buffer allocation and Python/native pipeline's transformation
of all output columns remain opportunities; their removal was not measured here.

`src/core/multivariate_statistics.cpp` limits joint response dimension to 32 and
reuses a node's covariance factorization across candidate features. Its small
linear-algebra operations do not justify adopting an asymptotic dense-matrix
algorithm without profiling and a finite-size benchmark.

The [September inference archive](../inference_accuracy_20260908/inference/README.md)
already demonstrated a preprocessing-heavy case: hiva's model had only three
root trees, while raw-frame inference took hundreds of milliseconds. That
historical diagnostic supports measuring preprocessing separately; its timings
are not fresh measurements from this investigation.

## Engineering directions from primary research

1. **Specialize and compact the existing CPU evaluator.** Hoist numeric versus
   categorical routing and scalar versus vector-output decisions out of row
   loops. Next, test a smaller private node representation. NVIDIA's
   [FIL redesign](https://developer.nvidia.com/blog/supercharge-tree-based-model-inference-with-forest-inference-library-in-nvidia-cuml/)
   provides implementation evidence for compact nodes and layout tuning. Its
   reported speedups are not predictions for CTBoost.
2. **Reduce repeated preparation.** Evaluate a prediction-only bin buffer that
   avoids copying schema metadata, followed by a schema-bound prepared-input
   API and selective pipeline execution. Preserve prefix-specific feature use,
   missing-value exceptions, categorical mapping, and input validation.
3. **Test bitset evaluation on suitable trees.**
   [QuickScorer](https://iris.unive.it/handle/10278/3661259) and
   [RapidScorer](https://ai.stanford.edu/~wzou/kdd_rapidscorer.pdf) motivate
   cache-aware bitsets and SIMD evaluation of the same learned trees. Any
   prototype needs categorical/missing routing, original leaf-ID parity, and
   model-size-dependent dispatch. A prior CTBoost eight-row packet experiment
   was rejected; another packet implementation needs new evidence.
4. **Reserve tensor compilation for a measured large-batch use case.**
   [Hummingbird](https://arxiv.org/abs/2010.04804) compiles traditional models
   into tensor computations. This is a plausible bridge to matrix kernels,
   but evaluating all node/path conditions can add substantial work and memory.
   Measure conversion, batching, routing equivalence, and accumulation order.
5. **Improve quality per unit of model size.** Test existing safeguarded leaf
   refinement and coupled multiclass leaf estimation; select options and
   iteration budgets using validation data only. Compact vector leaves store
   the existing shared multiclass partition once. Storage compaction alone
   does not improve predictive quality.

The original
[conditional-inference framework](https://www.zeileis.org/papers/Hothorn%2BHornik%2BZeileis-2006.pdf)
addresses statistical variable selection and stopping. It does not imply that
every boosted implementation must beat XGBoost or CatBoost on predictive loss.
Any ranking claim requires comparable tuning, splits, resource budgets, metrics,
and complete benchmark coverage.

## Measurement requirements

- Build the baseline and candidate from the same source revision and compiler
  flags; record native binary hashes. Initial checks found native 0.1.58 under
  the default Python 3.10 and native 0.1.60 under `.venv` Python 3.12, despite
  the checkout's Python version being 0.1.61. Neither is the study baseline.
- Use identical saved models for execution comparisons, including one-row and
  large batches. Include raw-input preparation in headline latency; report
  prepared-Pool measurements separately.
- Preserve bitwise raw scores and physical leaf IDs. Include wide bins,
  categorical values, missing values, iteration prefixes, and vector outputs.
- Keep accuracy choices on validation data and score held-out data only after
  freezing choices. Report regressions and costs, not only winning tasks.
- Treat local timing and the small accuracy scout as development evidence,
  without relabeling historical TabArena results or claiming an Elo increase.
