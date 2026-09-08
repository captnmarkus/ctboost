# Real-data inference diagnostic, 2026-09-08

The final Windows candidate reduced warm prediction latency on **14/14 tasks**, with a **1.76x geometric mean speedup** over public CTBoost 0.1.60. All 14 saved CTBoost models produced exactly identical prediction arrays under both wheels. These are development measurements under active background training, with substantial shared-resource contention. Compilation and test activity also varied between sequential passes.

Each row below is the median of seven warm blocks for a 1,000-row batch, including frame normalization, the full CTBoost feature pipeline, and AutoGluon CatBoost preprocessing. Both libraries use one thread and logical CPU 6. Loading models and constructing the repeated batch are excluded. Cold first-prediction measurements and every raw block are retained in [the four measurement records](measurements/). RSS fields are absolute process memory, not isolated model-cache size.

| Dataset | Original development rows | Public CTBoost ms | Final CTBoost ms | CatBoost ms | Public/final | Trees CTB/CB |
|---|---:|---:|---:|---:|---:|---:|
| blood-transfusion-service-center | 62 | 0.527 | 0.485 | 0.694 | 1.09x | 3/37 |
| coil2000_insurance_policies | 818 | 10.137 | 8.393 | 9.076 | 1.21x | 129/205 |
| GiveMeSomeCredit | 1250 | 3.576 | 2.433 | 1.158 | 1.47x | 105/60 |
| QSAR_fish_toxicity | 75 | 2.782 | 1.563 | 1.445 | 1.78x | 343/218 |
| wine_quality | 541 | 4.834 | 4.100 | 4.061 | 1.18x | 286/1111 |
| diamonds | 1250 | 8.828 | 6.842 | 10.851 | 1.29x | 367/2333 |
| MIC | 141 | 142.450 | 44.799 | 125.123 | 3.18x | 800/133 |
| SDSS17 | 1250 | 11.427 | 5.601 | 19.020 | 2.04x | 1062/864 |
| anneal | 74 | 64.947 | 21.404 | 42.982 | 3.03x | 1945/1078 |
| hiva_agnostic | 320 | 855.424 | 468.569 | 2066.788 | 1.83x | 3/181 |
| maternal_health_risk | 84 | 2.358 | 1.209 | 6.348 | 1.95x | 402/1015 |
| splice | 265 | 48.382 | 27.374 | 89.415 | 1.77x | 711/835 |
| students_dropout_and_academic_success | 368 | 24.879 | 11.302 | 23.700 | 2.20x | 396/490 |
| website_phishing | 112 | 8.444 | 4.508 | 13.298 | 1.87x | 738/420 |

CTBoost was faster than the fitted CatBoost default on 11/14 tasks. The geometric CTBoost/CatBoost latency ratio was 0.520. Different stopping points, tree counts, and model quality limit that comparison; this does not establish a general CatBoost speed or accuracy claim. The unchanged CatBoost control drifted by a geometric ratio of 0.997 between the public and candidate passes, with per-task ratios 0.699–1.330. These controls prevent interpreting all observed differences as exact code-only speedups.

## Follow-up disclosed

The [initial candidate](comparison-initial.json) improved 12/14 tasks (geometric speedup 1.316x), but hiva was 47% slower and splice 6.5% slower. We retained those results and profiled hiva before changing the code. Its 1,617 categorical inputs expand to 6,468 outputs, including three CTR class outputs per feature; its native ensemble contains only three root trees. Explicit stage timers identified categorical transformation as the dominant cost, while prediction from a prebuilt Pool took about 2–3 ms. The retained cProfile samples misattribute some extension time to adjacent Python calls, so the diagnosis uses explicit stage timers.

The follow-up reuses each CTR category key and fitted statistics across its class outputs, preserving the arithmetic and feature-selection behavior. Twelve direct-statistics regression cases cover 3/8 classes, missing/unseen categories, combinations, format 3/4, and strengths 0/.2/2. A public-release hiva feature matrix of 37x6,468 values also matches the candidate bit for bit ([receipt](hiva-profile/candidate-transform-check.json)). The final results above are a fresh public/candidate pair labelled `2`, after observing the first pair; they are not an independent validation panel. No models were refitted between timing pairs.

## Models and provenance

All 14 tasks were fixed by metadata before this diagnostic. We reused the CTBoost default models from the [accuracy scout](../accuracy/README.md) and fitted 14 AutoGluon CatBoost defaults once. The source panel caps official outer-training rows at 10,000, then uses seed-47 role folds: train 0–4, stopping 5, development 6. The confirmation role and official outer-test rows were never opened by this runner. Batches cycle or truncate the listed development rows deterministically.

Fits used 300 seconds, two CPUs, 8 GiB, and a 390-second hard limit; all 14 CatBoost preparations succeeded without timeout or memory termination. CatBoost retained AutoGluon's 10,000-tree cap, learning rate .05, and adaptive stopping. Reused CTBoost models used the predeclared 1,000-tree cap, learning rate .05, and patience 50. Fit records preserve exact parameters and model hashes. This latency study adds no TabArena scores, Elo, leaderboard, or admission evidence.

[The plan](plan.json), [source accuracy plan](source-accuracy-plan.json), [final comparison](comparison-final.json), [runtime hashes](runtime-final.json), and [original byte inventory](original-file-inventory.json) bind the source, wheels, role data, models, and measurements. The initial native hash is `611f035a…`; final is `68fe6951432644147e6fdd48a40199dc87d4babd8e5cc624bcbdfce170dd82a8`. Both development wheels report 0.1.60 and are distinguished by these hashes. Exact runner/dependency source copies are retained under `source/`. Models, feature arrays, targets, and prediction arrays remain in the local research cache and are excluded here.

Reproduce with the frozen role/model cache and the matching installed wheels:

```powershell
& BASELINE_PYTHON -I benchmarks/inference_tabarena_scout.py prepare --output RUN --scout SCOUT
& BASELINE_PYTHON -I benchmarks/inference_tabarena_scout.py measure --output RUN --label baseline --repeat 2 --cpus 6 --threads 1
& CANDIDATE_PYTHON -I benchmarks/inference_tabarena_scout.py measure --output RUN --label candidate --repeat 2 --cpus 6 --threads 1
```

A separate [learning-options timing follow-up](learning_options/README.md) measures the existing grouped/joint-feature settings with the same CatBoost controls.
