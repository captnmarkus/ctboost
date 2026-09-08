# CTBoost 0.1.61 release-candidate verification

This archive records release hardening of the CPU inference changes developed
on 8 September 2026. It includes failed checks and superseded candidates as
well as the corrected candidate. It does not record a published release or an
official TabArena result.

The current candidate passed 1,916 local tests, all 13 CPU CI jobs and the
complete 26-wheel release dry run. [verification.json](verification.json)
records the consolidated outcome. Pending-validation annotations in older
phase folders describe their historical status when captured; final CI results
are recorded in [ci_final](ci_final/).

The final production source is commit
`5780b41c7321c70e508dccf34e2f76f2503d1175` on
`improve/inference-accuracy-20260908`. Later documentation and evidence commits
may follow without changing executable source. Version 0.1.61 is consistent
across Python, CMake, R and JVM metadata.

## Evidence and corrections

- [Platform validation](platform_validation/) preserves local build, test and
  package receipts, together with the earlier Linux/macOS native-link failures
  and GNU FMA/ARM test failures. Final CI results are recorded separately from
  those failed runs.
- [Final CI](ci_final/) records the 13 successful CPU platform/compiler jobs
  and the complete release-workflow validation separately from superseded runs.
- [Final inference timing](inference_timing_phase3/) reuses the same 14 saved
  models in public/candidate/candidate/public order. The geometric mean speedup
  is 1.7918x, with all 14 tasks faster and exact prediction parity. All raw
  blocks, both timing pairs, unequal model sizes and control drift are retained.
- [Loss optima](loss_optima/) demonstrates the zero-weight Huber initialization
  and exact-fit Quantile fixes against public 0.1.60, with corrected-wheel tests.
- [Final preprocessing compatibility](pipeline_phase3/) checks 323 cases on
  each of NumPy 1.26/pandas 2.1 and NumPy 2.5/pandas 3.0. The
  [earlier preprocessing archive](pipeline/) retains the initial failures,
  compatibility fixes and limited conversion microtimings.
- [Final saved-model compatibility](model_compatibility_phase3/) verifies all
  14 public-release models on 6,610 already unsealed development rows, including
  raw and prepared predictions, repeated 1,000-row batches and serialized
  state. [Earlier phases](model_compatibility/) remain intact.
- [Training arithmetic](training_arithmetic/) explains the GNU contraction
  mismatch and why the correction reuses existing inference arithmetic instead
  of forcing fused operations. It also documents a separate, deferred nonzero
  `Pool.baseline` warm-start limitation.

The finalized local Windows/Python 3.12 CPU wheel has SHA-256
`b9fb7b598395a64380ee2f5a2395269ef567169edd151916790087dd71177f94`.
Its native extension has SHA-256
`5b91bb3139dd54ab05f5c8dea2b1bd1caabf0b9dcb2a2e76cafb01b30281e1b5`.
The initially announced `b3c83491...` archive hash was read before ZIP
finalization. All 181 installed payload files, excluding installer-rewritten
`RECORD`, were subsequently checked against the completed wheel. The saved-model
verifier's initial identity rejection is retained; its successful run used the
completed artifact's identity.

## Scope

Conditional feature selection still precedes cut optimization. Learning defaults
and saved-model formats are unchanged. The loss and GNU training-arithmetic
corrections can change newly trained models; native clients must rebuild.

The [earlier performance and accuracy studies](../inference_accuracy_20260908/)
remain a distinct development archive. Failed quality gates did not result in
new defaults or a merged joint-cut prototype. These checks establish no Elo
increase or general accuracy parity with XGBoost. Timing comparisons across
different CatBoost and CTBoost models are not accuracy-matched comparisons.

The frozen Full HPO25 run continues to use public 0.1.60 and its original
sources, environments and plan. This work did not change that run or the
TabArena pull request, fit additional evaluation models, open reserved
confirmation rows, publish a new dataset or send external messages.

Full test logs include environment-specific skips. CUDA-enabled wheel smoke
checks on hosted CPU runners do not validate execution on GPU hardware.
Historical failures must not be read as current failures, and a successful
single local CPU wheel does not establish the complete release-wheel matrix.

`manifest.json` binds every retained file by SHA-256. Models, benchmark rows,
prediction arrays, executable build artifacts and virtual environments remain
outside this Git archive.

Run `python -I verify_archive.py` from this directory to check the complete
retained file inventory and hashes without reading any benchmark data.
