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

The latest [published TabArena-Lite results](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-0.1.56)
measure 0.1.56. There is no measured 0.1.58 TabArena result yet; the requested
25-configuration HPO evaluation is pending. See [benchmark status](benchmarks.md)
for the scope and provenance of the existing results.
