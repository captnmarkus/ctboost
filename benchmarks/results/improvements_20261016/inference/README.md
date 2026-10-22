# CPU inference experiments, 2026-10-16

The retained change removes per-call copies of immutable quantization metadata from cached CPU score and leaf-index prediction. It preserves bin materialization, tree traversal, float addition order, serialized tree indices, and conditional-inference training. GPU and uncached/FMA score paths and contribution prediction still receive the complete histogram metadata.

**This is a targeted allocation improvement, not a general inference-speed win.** Across four existing synthetic models and batches 1, 64, and 4096, the raw-input geometric speedup was **0.998x** and the prepared-Pool diagnostic was **1.014x**. The unchanged CatBoost control ratio was 0.983x; dividing raw speedup by that control gives 1.016x. This small difference is not a meaningful broad speed claim.

A separate deliberately wide schema-copy stress case improved single-row raw prediction by **1.478x** and prepared-Pool prediction by **3.980x**. It uses a fixed, manually constructed model with 512 numeric columns, 255 borders per column, four active columns, 64 trees of depth two, and one or 64 input rows. It is excluded from the four-model aggregate.

| Wide rows | Path | Baseline microseconds | Candidate microseconds | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | raw | 47.00 | 31.81 | 1.478x |
| 1 | prepared | 15.77 | 3.96 | 3.980x |
| 64 | raw | 111.82 | 97.94 | 1.142x |
| 64 | prepared | 33.35 | 20.59 | 1.620x |

| Fitted case | Rows | Raw speedup | Prepared speedup | Unchanged CatBoost ratio |
| --- | ---: | ---: | ---: | ---: |
| numeric_binary | 1 | 0.990x | 1.017x | 0.985x |
| numeric_binary | 64 | 0.995x | 1.015x | 0.991x |
| numeric_binary | 4096 | 1.010x | 1.006x | 0.981x |
| numeric_multiclass | 1 | 1.003x | 1.019x | 0.976x |
| numeric_multiclass | 64 | 0.987x | 0.997x | 0.964x |
| numeric_multiclass | 4096 | 0.980x | 0.982x | 0.984x |
| categorical_100 | 1 | 1.000x | 1.024x | 0.975x |
| categorical_100 | 64 | 0.990x | 1.015x | 0.979x |
| categorical_100 | 4096 | 1.015x | 1.012x | 0.992x |
| categorical_2000 | 1 | 1.002x | 1.041x | 0.995x |
| categorical_2000 | 64 | 0.996x | 1.034x | 0.991x |
| categorical_2000 | 4096 | 1.013x | 1.008x | 0.986x |

Every timed pass checked all raw and prepared output bytes against the same fresh-source baseline references and checked native scores against the legacy uncached helper. No labels or model quality were evaluated in these experiments. The fitted cases are three scalar binary models and one five-class scalar-tree model; vector-leaf correctness is covered separately by the repository prediction tests.

Both experiments used baseline–candidate–candidate–baseline order, one logical CPU (6), and one declared native/BLAS/OpenMP thread. Each timing has three warmups, ten calibration calls, and seven blocks targeting 50 ms per block. Preparation/model loading is excluded; raw-input prediction includes normal preprocessing, whereas prepared-Pool timing is diagnostic. Each reported arm is the geometric mean of its two pass medians. Raw blocks and both ordering-specific speedups are retained. Other agents held heavy work during timing; ordinary operating-system activity remains uncontrolled.

The four saved models and 1,000 input rows come from the earlier synthetic diagnostic. Batches of 4,096 cycle those same rows in order. These are reused development fixtures, not an independent holdout or leaderboard benchmark. The runtime was Python 3.10.14, NumPy 1.26.4, pandas 2.3.3, CatBoost 1.2.10, and CPU-only MSVC 19.50.35737 Release with normal `/EHsc /O2 /Ob2 /DNDEBUG` flags. The frozen pandas fixtures were created under NumPy 2; the runner explicitly maps NumPy `_core` pickle module names to the corresponding NumPy 1 modules. Both arms use the identical loader and dependencies.

An earlier numeric/categorical traversal-specialization experiment was rejected and removed from production: raw geometric speedup 0.992x, prepared 0.962x, unchanged CatBoost ratio 0.988x, and control-adjusted raw 1.004x. Some cases improved, but categorical prepared and large-batch multiclass prediction regressed. Its complete negative evidence is under `timing/` and the rejected source change is `candidate-specialized.patch`. Accepted bins-only measurements are under `timing-bins/`; the wide diagnostic is under `wide/`.

The baseline was freshly built from clean source commit `9e68e6d5a6fe4a402d7b465092f6690bdf979a94`, before source changes. The checkout initially loaded stale native modules (0.1.58 under Python 3.10 and 0.1.60 under Python 3.12) despite Python package metadata 0.1.61. These stale binaries were not used as the performance baseline. Build receipts record exact native and changed-source hashes. The isolated importer disables the existing editable-import finder and asserts the resolved package path.

To reproduce this protocol, build baseline and candidate CPU Release packages in separate directories using the same compiler and Python ABI, then provide their paths through `--package`. `build.cmd` records this host’s exact compiler/include/library environment. Use `measure_inference.py --fixtures FIXTURES --output NEW_OUTPUT --package PACKAGE --label LABEL`; first create references with `--reference --check-only` under the baseline. Run the four planned labels in order. The exact fitted models, input pickles, and reference arrays remain local; their hashes are in every receipt. New fixtures can be created with the archived preparation procedure in `benchmarks/inference_latency.py prepare --iterations 400`, but are a new experiment and need fresh references and timings. No claim of byte-identical reproduction without the original fixtures is made.

The stress case is independently reconstructible: run `measure_wide.py --prepare --check-only --package BASELINE --output NEW_WIDE --label prepare`, then measure its four planned baseline/candidate labels without `--prepare`. The source contains all deterministic state and input-generation parameters. Use fresh output directories and labels. Scripts and plans were frozen before the timed passes; receipts record their SHA-256 values. `summarize_bins.py` and `summarize_inference.py` recompute the fitted-case aggregates from archived raw measurements.
