# CTBoost inference and accuracy investigation — 16 October 2026

This work retains the conditional-inference feature-test-before-cut design.
It adds a targeted inference optimization, tests existing learning options,
and evaluates whether matrix-based execution helps with the current CPU workload.
Production learning defaults and model formats are unchanged.

## Results

| Experiment | Observed result | Decision |
|---|---|---|
| Avoid copying quantization metadata in cached prediction | Four ordinary synthetic models: essentially unchanged end-to-end latency (0.998x). Separate 512-feature stress case: 1.48x faster for one raw row, 1.14x for 64 rows. | Retained as a targeted memory-allocation improvement. |
| Specialize numeric/categorical traversal | No aggregate improvement; several prepared-input workloads regressed. | Rejected; patch and negative results retained. |
| Compile unchanged trees into NumPy matrix operations | Exact leaf IDs and raw float32 bits; 2.12–18.34x slower at tested CPU batch sizes. | Experimental only. |
| Validation-select existing leaf refinement, learning rate and model size | Mean held-out loss reductions of 16.78%, 1.19%, and 27.73% on three small datasets, two splits each. | Promising exploratory policy; no default or ranking claim. |
| Existing vector-leaf layout | Identical digits predictions with 90% fewer physical trees/nodes and 89.17% smaller native JSON. | Useful existing storage option; no warm-speed claim from this test. |
| Existing pickle persistence | One digits model: 449.32 MB JSON to 62.51 MB pickle, exact native predictions. | Practical file-size reduction; load latency was not measured. |

The retained code change is in `src/core/booster_prediction.cpp`,
`src/core/booster_predict.cpp`, and `src/core/booster_internal.hpp`. Cached score
and leaf-index prediction use a bin-only internal histogram. The original
materialization, comparisons, tree order and float additions remain unchanged.
GPU, uncached/FMA scoring, contributions, and other full-histogram consumers
retain their quantization metadata. No root-only allocation shortcut or new
matrix backend was added to the library.

The wide inference result is a deliberately constructed diagnostic with 512
numeric features, four active features and 64 depth-two trees. It is excluded
from the four-model aggregate. Its prepared-Pool path improved 3.98x for one
row, but that number excludes input construction. Raw timings include normal
input processing; fixture construction and model loading are excluded.
All timed candidate outputs matched the fresh baseline and uncached helper
bit for bit. These Windows CPU measurements do not establish general parity
with CatBoost or a broad latency improvement.

The accuracy policy searched three or four CTBoost configurations per split
against one fixed XGBoost configuration. It remained worse than XGBoost on
breast cancer and diabetes, and slightly better on digits on dataset-mean loss.
One selected split regressed by 1.06%; applying the fast-refinement arm without
validation selection regressed by 21.34% on another comparison. These are
exploratory, historically reused small datasets, not independent TabArena
confirmation. Learning rate, leaf estimation, storage, and checkpoint selection
are mixed factors; the gains cannot be attributed solely to leaf refinement.

## Multiplication research

The requested OpenAI results are real: the
[6 October mathematics release](https://openai.com/index/sharing-ai-progress-in-mathematics/)
includes an integer-multiplication preprint claiming
`O(n (log n)^(1 - 2^-182))` bit complexity, and a complex matrix-multiplication
bound `omega <= 2.25`. These asymptotic results do not establish a faster
machine-word instruction or a useful dense kernel at CTBoost's sizes. The
[matrix formalization](https://github.com/openai/math/blob/main/lean/docs/107.md)
explicitly excludes practical crossover claims. See the
[research review](research.md) for primary sources, verification scope, and
the distinction between the mathematical results and our measured NumPy prototype.

CTBoost's cached CPU traversal is dominated by binning, memory access, branching,
and leaf additions. Dense conversion increases work: the tensor diagnostic
requires 33.6 million multiply-accumulates plus all-node comparisons at 4096
rows, versus at most 1.17 million comparisons along native tree paths. It is
not an implementation or a test of OpenAI's asymptotic algorithm.

## Evidence and reproduction

- [Inference](inference/README.md): both native candidates, ABBA timing order,
  all raw timing blocks, separate wide diagnostic, source/build hashes,
  frozen runners, and the rejected patch.
- [Accuracy](accuracy/README.md): frozen train/validation/test roles, all 26 fits,
  validation choices frozen before assessment, complete results, model-size
  follow-up, and reproduction commands.
- [Tensor execution](tensor/README.md): deterministic model, exact routing and
  arithmetic checks, timing scopes, work/storage estimates, and reproduction.
- [Validation](validation/README.md): regression tests and local native builds.

Exact copies of the three new top-level benchmark runners are also retained
under `source/`, with byte-preserving Git attributes for this evidence archive.
The archive is excluded from wheel packaging, including ignored local model
and array fixtures that may be present alongside the reports.

The baseline was freshly built from commit
`9e68e6d5a6fe4a402d7b465092f6690bdf979a94`, with MSVC Release flags and a fixed
Python ABI. Initial local imports combined Python package metadata 0.1.61 with
native binaries 0.1.58 (Python 3.10) and 0.1.60 (Python 3.12). Those stale
binaries were excluded from all performance comparisons. Local rebuild receipts
and backups document their replacement. Generated models and arrays remain
under ignored local experiment directories; the reports disclose where exact
reproduction requires those original fixtures.

## Recommended next experiments

1. Measure selective feature-pipeline execution and zero-active-feature
   allocation avoidance on preprocessing-heavy real models, including the
   previously identified hiva case. Preserve schema validation, prefix behavior,
   and missing-value exceptions.
2. Test a smaller prediction-node representation or a bounded bitset evaluator
   on varied real model sizes. Preserve original leaf IDs and summation order;
   retain the conventional evaluator where the alternative loses.
3. Predeclare a broader quality/latency panel with learning-rate-only ablations,
   coupled leaves, vector storage, and validation-selected round budgets.
   Give competing libraries comparable search budgets and reserve confirmation
   data. Existing grouped feature tests remain opt-in candidates, with their
   earlier failed gates disclosed.
4. Investigate compact native persistence separately from warm inference.
   Repeated category-route arrays and pretty-printed JSON create large files;
   any representation change needs backward-compatible readers and measured
   cold loading, memory, and output parity.

Faster execution does not itself raise predictive accuracy, and conditional
feature selection does not guarantee the highest benchmark rank. The next
promotion decision should use evidence from the broader, frozen protocol.
