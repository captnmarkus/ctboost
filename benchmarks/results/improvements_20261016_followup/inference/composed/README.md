# Composed serving changes

The final composition of builtin missing-value checks, direct exact-string key conversion, the metadata-gated one-row categorical-frame path, and zero-active-feature bin allocation avoidance passed a new **A–B–C–C–B–A** experiment. A was the phase-one metadata-copy baseline; B was the previously accepted missing-value/allocation implementation; C contained the composed changes. Training, tree selection, arithmetic order, and serialization semantics remained unchanged. The compact JSON writer was present in C but no model loading/saving was timed.

All five fixtures and all batches passed the original full 1,000-row transformed-array and prediction byte comparisons, plus native-versus-uncached score checks. The earlier one-row fallback regressions disappeared after the guard used fitted pipeline metadata to reject numeric/mixed pipelines cheaply. These are measurements of the composition, not products of separate component speedups.

| Fixture | Rows | Phase-one baseline ms | Composed ms | Raw speedup |
| --- | ---: | ---: | ---: | ---: |
| hiva_agnostic | 1 | 37.067 | 27.957 | 1.326x |
| hiva_agnostic | 64 | 55.018 | 47.210 | 1.165x |
| hiva_agnostic | 1000 | 309.282 | 212.896 | 1.453x |
| coil2000_insurance_policies | 1 | 0.377 | 0.377 | 1.001x |
| coil2000_insurance_policies | 64 | 0.715 | 0.684 | 1.046x |
| coil2000_insurance_policies | 1000 | 5.130 | 4.760 | 1.078x |
| splice | 1 | 1.612 | 1.378 | 1.169x |
| splice | 64 | 2.860 | 2.621 | 1.091x |
| splice | 1000 | 17.563 | 13.656 | 1.286x |
| GiveMeSomeCredit | 1 | 0.192 | 0.191 | 1.009x |
| GiveMeSomeCredit | 64 | 0.342 | 0.340 | 1.004x |
| GiveMeSomeCredit | 1000 | 1.590 | 1.595 | 0.997x |
| SDSS17 | 1 | 0.306 | 0.303 | 1.007x |
| SDSS17 | 64 | 0.496 | 0.484 | 1.025x |
| SDSS17 | 1000 | 3.133 | 3.021 | 1.037x |

Across these 15 points, phase-one-to-composed geometric ratios were **1.105x raw**, 1.009x prepared, and 1.014x native. Against B, they were 1.050x raw, 1.003x prepared, and 0.998x native. The gains are concentrated in categorical preprocessing; the small prepared/native differences do not establish a traversal improvement. Hiva still contains three nontrivial trees and 22 active features, so the root-only shortcut does not explain its gains.

Both mirrored comparisons reproduced the principal improvements. The raw hiva-1,000 ratios were 1.460x and 1.446x; splice-1,000 ratios were 1.291x and 1.282x. Against B, one-row numeric/mixed fallback ratios were 1.010x on coil, 1.004x on GiveMeSomeCredit, and 1.011x on SDSS17. All initial candidate and rejected-node evidence remains in the sibling `candidates/` directory.

The full composed CPython 3.10 suite subsequently passed **2,070 tests with 31 skips** in 198.94 seconds. CPython 3.12/NumPy 2 focused interoperability checks passed 146 tests. Both native CTest smoke programs passed 230 checks. The native packages are retained locally at `.tmp/improvements-20261016/followup-native/composed-cp310` and `composed-cp312`, and their native files were copied into the checkout after preserving backups. Binary/source/Python hashes and compiler flags are recorded in the build receipts. CPython 3.12's embedded smoke program requires the linked interpreter's `PYTHONHOME` and the environment's NumPy site-packages in `PYTHONPATH`; the native CMake `ctest.exe` was used to avoid invoking a different Python launcher with that home.

The same frozen real-data runner and protocol from the parent directory apply: one CPU/thread, coordinated quiet measurement interval, seven block medians, preserved raw blocks, and reused saved development fixtures. These are local serving measurements, not new model quality or equal-quality library-rank evidence. `../summarize.py . --baseline original --candidate composed` and `--baseline accepted-final --candidate composed` reproduce the two summaries.
