# Real-input inference follow-up

The retained exact-builtin missing-value check reduces repeated Python/NumPy type checks during categorical preprocessing. A separate cached-prediction change skips allocating bin storage when the selected tree prefix has no active features. Both preserve tree learning, split selection, prediction arithmetic, and model state.

Five previously saved real-data development fixtures were measured at 1, 64, and 1,000 rows, cycling the available rows where necessary. The categorical preprocessing change improved 1,000-row raw prediction on hiva from **310.54 to 233.81 ms (1.328x)** and splice from **17.76 to 14.44 ms (1.230x)**. The 15-case raw geometric speedup was 1.057x. Prepared-Pool and native-only geometric ratios were 1.007x and 1.015x; these small differences do not establish a traversal improvement.

**Hiva is not root-only:** its saved model contains three trees, 147 nodes, and 22 active transformed features out of 6,468. None of the five timed models exercises the new zero-active-feature allocation shortcut, so that shortcut cannot explain the measured gains. Hiva's normalized input cells are exact Python strings, as recorded in `hiva-cell-types.json`.

| Case | Rows | Raw baseline ms | Raw candidate ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| hiva_agnostic | 1 | 37.480 | 36.639 | 1.023x |
| hiva_agnostic | 64 | 54.637 | 50.657 | 1.079x |
| hiva_agnostic | 1000 | 310.538 | 233.806 | 1.328x |
| coil2000_insurance_policies | 1 | 0.382 | 0.382 | 0.999x |
| coil2000_insurance_policies | 64 | 0.718 | 0.699 | 1.028x |
| coil2000_insurance_policies | 1000 | 5.148 | 4.873 | 1.056x |
| splice | 1 | 1.643 | 1.612 | 1.020x |
| splice | 64 | 2.914 | 2.689 | 1.084x |
| splice | 1000 | 17.760 | 14.439 | 1.230x |
| GiveMeSomeCredit | 1 | 0.194 | 0.192 | 1.008x |
| GiveMeSomeCredit | 64 | 0.345 | 0.343 | 1.006x |
| GiveMeSomeCredit | 1000 | 1.607 | 1.599 | 1.005x |
| SDSS17 | 1 | 0.309 | 0.307 | 1.004x |
| SDSS17 | 64 | 0.504 | 0.496 | 1.015x |
| SDSS17 | 1000 | 3.114 | 3.023 | 1.030x |

The frozen protocol used baseline–candidate–candidate–baseline order, one logical CPU (6), one native/BLAS/OpenMP thread, three warmups, five calibration calls, and seven timing blocks targeting 50 ms each. Each arm in the table is the geometric mean of two pass medians. Raw blocks and both ordering-specific ratios are retained in the four pass receipts and `comparison.json`. Other agents paused heavy work during timing; operating-system activity was not controlled. There is no unchanged competitor control in this experiment.

Raw timing includes the existing TabArena frame normalization and the model's normal preprocessing and prediction. Prepared and native timing exclude preprocessing and serve only as diagnostics. All passes verify full 1,000-row transformed feature arrays byte for byte, raw and prepared outputs byte for byte, and native prediction scores against the uncached implementation. This is a saved-model development diagnostic, not a new quality evaluation or a claim about equal-quality XGBoost/LightGBM inference.

The baseline is the accepted phase-one metadata-copy optimization, not the original checkout. The candidate adds the zero-active allocation shortcut and builtin scalar missing-value fast path. Build receipts record source and binary hashes and MSVC Release flags. All arms use Python 3.10.14, NumPy 1.26.4, pandas 2.3.3, and the same loader. The NumPy 2 fixture pickle compatibility mapping is explicit in the runner. Targeted checks passed: 83 prediction tests, 133 key-codec/numeric/CTR inference tests, and 230 native smoke checks.

The local source fixtures are under `.tmp/inference-accuracy-20260908/accuracy-grouped-scout-v1`, and isolated runtime packages under `.tmp/improvements-20261016/followup-native/{baseline,root-only,combined}`. The models and input frames are not redistributed here. Every pass records their hashes, so reproducing these exact measurements requires those local fixtures or independently regenerating them with the earlier study. The root-only package was built and retained separately; it was not timed in this five-case experiment.

`runner.py` is the frozen copy of `benchmarks/inference_real_followup.py` used for these measurements. It expects to run from the repository benchmark directory (or otherwise have the repository on its Python path) so it can import the existing adapter. Example invocation from the repository root:

```powershell
python -I benchmarks/inference_real_followup.py --package .tmp/improvements-20261016/followup-native/baseline --scout .tmp/inference-accuracy-20260908/accuracy-grouped-scout-v1 --output .tmp/improvements-20261016/followup-native/real --label baseline-1
python -I benchmarks/results/improvements_20261016_followup/inference/summarize.py benchmarks/results/improvements_20261016_followup/inference
```

The runner refuses to overwrite receipts or reference arrays. Initial reference generation needs `--reference --check-only` and a new label/output directory. `summarize.py` reproduces `comparison.json` exactly from the four retained receipts.
