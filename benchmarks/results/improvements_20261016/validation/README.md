# Validation — 16 October 2026

The retained inference change passed **175 Python tests** on Windows,
CPython 3.10.14, with the rebuilt MSVC 19.50 CPU Release module:

- 147 prediction, float-arithmetic, vector-leaf, multiclass, basic training and
  training-API tests.
- 28 sparse-input, sklearn, persistence and exact-SHAP integration tests.
- The direct C++ prediction-cache regression test passed separately.

The tests cover routing and leaf identities, bin widths, categorical and missing
values, iteration prefixes, cache invalidation, baselines, empty/boundary batches,
vector/scalar arithmetic, and full-histogram consumers. This was a targeted
regression run, not the entire repository suite or a GPU/Linux/ARM validation.
The new benchmark scripts also passed Python compilation; `git diff --check`
reported no whitespace errors.

[CPython 3.10 receipt](cp310_validation.json), [focused test output](focused-tests.txt),
[integration test output](integration-tests.txt), and their JUnit XML companions
record the loaded module and exact results. The ordinary checkout import now
loads the candidate binary, SHA-256
`e1f5f5a6e29d9f643189fe87e27cecf66248b9ddd65d1617321768e5320084ed`,
with both Python and native version 0.1.61. The prior module is backed up under
`.tmp/improvements-20261016/`.

The first direct CTest launch could not initialize embedded Python's filesystem
codec. The successful rerun explicitly set `PYTHONHOME` to the linked Python
3.10 installation for that subprocess; no test assertion or source fix was
needed. The exact command is in the receipt.

Re-run the Python checks against a chosen isolated package with the archived
`run_tests.py`, followed by the same test filenames recorded in the JUnit reports.
Native build/compiler flags are recorded in the inference build receipts.

The [Python 3.12 integration report](cp312.md) and
[build receipt](cp312_build_receipt.json) document the separately rebuilt and
activated `.venv` native module. It passed 90 raw-bit prediction parity checks
and its direct native CTest, using NumPy 2.5.2 and the same frozen production
source. These checks do not rerun or replace the CPython 3.10 timing measurements.
Both normal local Python environments now report matching Python/native 0.1.61.
