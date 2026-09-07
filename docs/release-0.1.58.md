# CTBoost 0.1.58

CTBoost 0.1.58 includes the optional compact multiclass vector leaves and
correctness fixes documented in [0.1.57](release-0.1.57.md), plus two fixes found
during cross-platform CI. Publication of 0.1.57 to PyPI and GitHub Releases was
withheld; its Git tag remains available for provenance.

- Vector updates now preserve scalar accumulation rounding on platforms that
  contract floating-point multiplication and addition. This prevents small
  prediction differences from changing later boosting rounds. Conditional
  feature tests, split selection, and the default scalar strategy remain unchanged.
- Windows C++ export tests explicitly expose the selected compiler's runtime
  directory while loading the generated DLL. They still compile exported code
  and compare its predictions with the native model.

Model and predictor formats are unchanged from 0.1.57. See the
[vector-leaf guide](guides/vector-leaves.md) for supported workflows and boundaries.

The completed [0.1.58 TabArena-Lite evaluation](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-hpo25-0.1.58)
covers all 51 datasets at `r0f0`, with the default plus 25 frozen HPO
configurations and eight-fold bagging: 1,326 parent results and 10,608 child
fits, with zero imputed CTBoost tasks.

| Evaluation | Lite Elo |
|---|---:|
| Default | 1161.9 |
| Tuned | 1262.8 |
| Tuned + ensemble | 1296.9 |

These scores use an 87-row comparison roster. This is a author-run Lite HPO25
evaluation, not a full 200-configuration run, TabArena-Full result, or official
leaderboard entry. The workers' 4 CPUs and 28 GB RAM make their timings non-comparable
to canonical TabArena runtimes. See [benchmark status](benchmarks.md) for
provenance and the earlier 0.1.56 result, which used a different comparison roster.
