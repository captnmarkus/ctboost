# Validation receipts

The full pre-follow-up baseline run imported the phase-one native extension
`e1f5f5a6e29d9f643189fe87e27cecf66248b9ddd65d1617321768e5320084ed`
from this checkout using CPython 3.10.14. It finished with **2,041 passed,
31 skipped, one failed**. Its failure was an existing process-lifecycle race in
`test_synthetic_process_failure_continues_next_parent_and_never_retries`:
`local_hpo_controller.process_rss_bytes` queried children after the synthetic
worker had exited, raising `psutil.NoSuchProcess`. The isolated rerun passed.
Both outcomes are preserved; the baseline suite is not reported as fully green.

The first full-suite attempt was interrupted because its temporary test launcher
lacked a Windows multiprocessing entry-point guard, causing spawned workers to
re-enter pytest. `full-initial.log` is incomplete and is not counted as a passing
run. The archived `../run_tests.py` launcher fixes that harness problem and pins
both the imported Python package and native extension.

After changing the two JSON writers, the existing booster, classifier,
feature-pipeline and vector-leaf persistence suites passed: **44 tests**.
`compact-json.log` and `compact-json.xml` record that run.

The composed retained implementation passed the complete CPython 3.10 suite:
**2,070 passed, 31 skipped in 198.94 seconds**, using native SHA-256
`d069a028a65f35656d977dfc808e5f43f113cba59c81cb19e16b26f05dedab24`.
See `full-composed.log` and `full-composed.xml`. The baseline process-lifecycle
race did not recur in this run; its original failure remains recorded above.

Python 3.12.11 with NumPy 2.0.2 and pandas 2.3.3 also passed **146 focused tests**
covering categorical conversion, key encoding, persistence, prediction caches
and arithmetic. Its native and source identities are recorded in
`../profile/final_cp312_validation.json`. Both native CPU builds passed their
230-check CTest suites, with receipts under the inference archive.

Any subsequently retained implementation candidate requires its own validation;
these results describe the composed implementation above.
CUDA and unavailable optional integration environments are outside these CPU
builds; skip reasons remain in the full pytest output.
