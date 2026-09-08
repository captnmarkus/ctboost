# Final-source saved-model inference timing

Four passes used baseline-final-final-baseline order on the same saved 14 CTBoost models and 14 AutoGluon CatBoost models. All predictions matched both archived public reference arrays bit for bit in all passes. No models were fitted; only the same 1,000 cyclic development-fold-6 rows were used. Confirmation fold 7 and outer-test rows remained unopened.

The final CTBoost candidate was faster on 14/14 datasets; the median per-dataset speedup was 1.87x. The unchanged CatBoost control had a median baseline-period/final-period ratio of 1.02x. The median CTBoost speedup divided by that task's control ratio was 1.73x. These describe this selected local panel, not general population speed or statistical significance.

| Dataset | Public CTBoost ms | Final CTBoost ms | Speedup | CatBoost control ratio | Final / CatBoost |
| --- | ---: | ---: | ---: | ---: | ---: |
| blood-transfusion-service-center | 0.466 | 0.376 | 1.24x | 1.07x | 0.53x |
| coil2000_insurance_policies | 7.934 | 7.224 | 1.10x | 1.06x | 0.95x |
| GiveMeSomeCredit | 2.998 | 2.002 | 1.50x | 1.12x | 2.41x |
| QSAR_fish_toxicity | 2.124 | 1.624 | 1.31x | 0.93x | 1.24x |
| wine_quality | 5.013 | 4.001 | 1.25x | 0.84x | 1.00x |
| diamonds | 7.708 | 6.533 | 1.18x | 0.92x | 0.65x |
| MIC | 122.894 | 42.094 | 2.92x | 0.96x | 0.38x |
| SDSS17 | 10.425 | 4.614 | 2.26x | 0.83x | 0.26x |
| anneal | 58.879 | 19.636 | 3.00x | 0.91x | 0.49x |
| hiva_agnostic | 716.480 | 407.934 | 1.76x | 1.04x | 0.23x |
| maternal_health_risk | 2.961 | 1.087 | 2.72x | 1.54x | 0.26x |
| splice | 50.463 | 24.540 | 2.06x | 1.13x | 0.35x |
| students_dropout_and_academic_success | 25.193 | 10.383 | 2.43x | 1.04x | 0.42x |
| website_phishing | 9.690 | 4.909 | 1.97x | 1.00x | 0.34x |

Each cell uses the median of its two pass medians. Speedup is public/final CTBoost time; values above one favor the final candidate. The CatBoost control ratio is its baseline-period time divided by its final-period time, despite using the same CatBoost model and package in every pass. Final/CatBoost compares the two libraries during final passes; values below one favor CTBoost. The fitted tree counts differ across libraries and are recorded per model in the raw summaries. No equal-quality or equal-tree-count comparison is claimed.

The final wheel is `b9fb7b598395a64380ee2f5a2395269ef567169edd151916790087dd71177f94`, native `5b91bb3139dd54ab05f5c8dea2b1bd1caabf0b9dcb2a2e76cafb01b30281e1b5`, from production commit `5780b41c7321c70e508dccf34e2f76f2503d1175`. Its installed Python file hashes and native bytes match the separate phase-three 14-model compatibility receipt. The public CTBoost native is `06796f386ccbf5b51b3134ca11a391ac6ef719ab92fcb509f3f689e77abbf63c`. The explicit import loads only the final CTBoost package into the unchanged public baseline environment. Distribution metadata intentionally still reports ctboost 0.1.60 while the actual candidate module reports 0.1.61; every runtime receipt records both, with the actual package path and hashes. NumPy, pandas, scikit-learn, AutoGluon and CatBoost dependencies remain the same across arms.

CPU affinity was logical CPU 6, with all declared native/BLAS/OpenMP thread budgets set to one. The original frozen `measure_call` function performs three warmups, ten calibration calls and seven timed blocks. Normalization and model preprocessing are included; model loading and row selection are excluded. Cold latency means first prediction after each model is loaded, and its raw value is retained. RSS is the whole process after warmup, not an isolated model allocation or peak-memory measurement. Pass wall times, all seven block values, adaptive call counts, model/data/reference hashes and prediction-value hashes are preserved under `raw/`. Source data, labels, models and prediction arrays are excluded.

Repeated controls reveal remaining system variation. The median final/initial baseline CTBoost time was 0.93x, and the same CatBoost last/first ratio was 0.92x. `comparison.json` retains every dataset's two ordering-specific speedups, both CTBoost repeat ratios, control ratios and raw pass medians. ABBA reduces simple ordering bias but does not eliminate shared-system noise. The panel was chosen earlier for development work and is not an independent holdout or leaderboard benchmark.

An independent calculation uses geometric means across each arm's two pass medians and then across the 14 task ratios. Its overall public/final CTBoost speedup is 1.79x; all 14 tasks are faster. The geometric speedups for the baseline-then-final and final-then-baseline pairs are 1.83x and 1.76x, respectively, with all 14 tasks faster in each ordering. The geometric final-CTBoost/CatBoost ratio is 0.53x, with CTBoost faster on 11 of 14 tasks. The unchanged CatBoost middle-pass/endpoint-pass ratio is 0.988x, with task ratios ranging from 0.660x to 1.202x. These control ratios use final/baseline direction, the reciprocal direction of the table's control column. The small aggregate drift does not remove the larger task-specific variation.

`root-independent-summary.json` preserves the independently produced receipt byte for byte. `geometric_summary.py` reproduces it from the archived raw summaries and checks agreement. The 1.79x geometric aggregate and the earlier 1.87x median aggregate answer different aggregation choices; neither drops any task. The table and `comparison.json` retain their original two-pass-median definition.

The plan and source hashes were frozen before pass 1. The adapter has no fitting entry point. Exact adapter bytes are archived as `measure.py`, with the orchestration and archiving scripts. The temporary untracked `benchmarks/inference_release_timing.py` is removed after archiving so this research helper is not shipped in the wheel. To reproduce, restore that exact source to the original recorded location in a separate research checkout, provide the existing hashed fixtures and matching environments, and choose a fresh output directory; existing evidence is never overwritten. `manifest.json` hashes all evidence files; `SHA256SUMS` additionally covers the manifest. Local Git attributes preserve exact bytes.
