# Mean CTR precision and legacy compatibility

New feature pipelines use format 5 and double-precision Mean CTR priors, category
totals and ordered prefix totals. Final features remain checked float32 values.
This fixes demonstrable cancellation and accumulation drift, including a finite
mean whose total exceeds the float32 range. For example, the float32 accumulator
can lose the middle target in `[1e8, 1, -1e8]`; four float32 targets near `3e38`
overflow a float32 total despite having a finite float32 mean. Double accumulation
is improved precision, not exact summation for arbitrary dynamic ranges.

The conditional feature test, cut search, tree structure, objective and metric
code are unchanged. This evidence establishes a numerical correction and saved
model compatibility; it does **not** establish an accuracy or latency improvement.

**Adoption status:** the validated CP310 and CP312 binaries are active in the
checkout, with prior binaries backed up. The frozen accuracy runtime is unchanged.
The prospectively frozen [final timing policy v2](../final_runner_preparation/selection_policy_v2.json)
uses this exact CP312 runtime; `timing-policy-alignment.json` independently checks
its native, Python, source and legacy-parity provenance. No new timing was run.

## Compatibility contract

- Formats 1–4 load stored CTR values through float32, including its historical
  rounding and range behavior. Their subsequent accumulation remains float32.
- Loaded format 4 preserves prediction, ordered replay and warm-start behavior.
  Formats 1/2 canonicalize to 3; explicitly refitting formats 1–3 still upgrades
  smoothing to 4, while retaining legacy accumulation.
- Fresh pipelines use format 5. JSON and pickle retain double statistics, and
  Python raw-feature exports accept format 5. Older runtimes reject format 5.
- Frequency CTR and embedding arithmetic are unchanged. A fresh pipeline is
  required to obtain the new Mean arithmetic; old models are not silently refit.

## Checks and observed limits

The isolated CP312 build passed 103 focused checks with three optional export
skips, and 125 wider pipeline, embedding, persistence and warm-start checks.
The final precision file passed all 38 cases, including two added raw-export
tests for cancellation and large finite totals. Both native builds passed all
230 checks. Counts from these targeted runs overlap and are not summed.

The independent previous-runtime oracle captures 16 cases spanning formats 1–4,
regression, binary and multiclass targets, Frequency and categorical combinations.
The candidate matches loaded states, transformed query bytes, ordered replay
bytes and replayed states exactly. A persisted old model also matches prediction
bytes and warm-start continuation's complete native and pipeline state.

For all 12 frozen wave-16 CT models, complete original TRAIN-frame raw margins,
1/64/1000-row predictions, transformed-feature bytes, pipeline state and schema
hashes match the previous final runtime exactly. No fitting or scoring was done;
DEV, confirmation and official-test inputs were not opened. Real models stay in
their existing local cache; the archived receipts bind their hashes and inputs.

The fully consistent CP310 full run recorded **2484 passed, 125 skipped and 41
failed**. Those failures were resolved without a production-code change:

1. One distributed-pipeline test still expected newly fitted format 4. Its
   expectation was changed to 5 and it passed independently on CP310 and CP312.
2. Forty controller cases inherited CPU affinity narrowed by earlier scientific
   worker tests. Both controller modules passed all 55 checks with two worker
   slots available. The repository now has an autouse fixture that restores each
   test's incoming affinity. A normal checkout pytest invocation, without the
   external validation plugin, passed the known affinity-changing scientific
   test followed by both controller modules: **56 passed**.

This provides passing coverage for **2525 cases from that full collection**, with
**125 explicit skips**. It is not a zero-failure full-run receipt. Nineteen of the
skips are enumerated by the validation bootstrap because the CP310 environment
lacks the complete scientific comparison stack. Trying its existing LightGBM
dependency produced native ctypes access violations, while XGBoost 3.2 lacks the
frozen 3.4 vector-layout contract. All 34 checks in those four experiment modules
passed in the supported CP312 environment. Frozen scientific sources were not
edited. The latest ridge-diagnostic test source changed during the full run and
has separate validation; this archive does not claim coverage of that latest file.

The initial mixed-runtime full run, version-expectation failures, dependency
probe, embedded-Python CTest setup errors, affinity diagnostics and successful
repairs are preserved. `validation.json` records the precise interpretation;
`manifest.json` hashes the evidence.

## Runtime and source provenance

The previous isolated inference packages and immutable accuracy runtime remain
unchanged. New packages are under
`.tmp/improvements-20261016/ctr-precision/runtime-cp310` and `runtime-cp312`.
The checkout's CP310 and CP312 binaries were replaced only after focused checks,
with exact prior-byte backups and an activation receipt. Other Python ABIs were
not changed.

| Build | Native SHA-256 |
| --- | --- |
| CP310 | `21ed48b6234bf570413fd4336ffdf76279b0ff9e3d9003cbcf0b34620e2aee17` |
| CP312 | `04adb21b0e4cea8046581a23b62585b6c8d631561d96a417ad87bc3c55041613` |

The builds use MSVC 19.50 Release `/O2 /Ob2 /DNDEBUG` and `/EHsc`, CUDA disabled,
two compiler workers, and an isolated 110-file native source snapshot. See the
build receipts, flags, source hashes and CTR-only patch. `sdist-membership.json`
checks every changed source plus the new precision tests and affinity fixture
byte-for-byte inside a fresh source distribution; no wheel or native rebuild was
performed for that packaging check.


## Later complete CP310 validation

A new full `tests/` run on the active21ed runtime completed after the affinity and format-expectation repairs: **2,552 passed, eight failed,125 skipped**,2,685 collected,210.61 seconds. This is not an all-green suite claim. All eight failures came from the newly collected frozen `test_scalar_numerical_retry_experiment.py` research module importing missing LightGBM through its common three-library helper. The same complete module had already passed21 checks in the exact CP312 c5 comparison runtime; its test/helper hashes match this run. No source edits, dependency installs, additional skips, or rerun followed these failures.

Every controller check passed, the repository affinity fixture restored the four allowed CPUs, and all recorded source hashes stayed unchanged. The original19-node validation exclusion policy and all previous run/repair receipts are preserved. [The new receipt and classification](final_full_cp310/classification.json), full log, source hashes, exact scientific-runtime receipt and snapshots are separate artifacts. The owned process exited before quiet CV activation; no inference timing or new native build was performed.
