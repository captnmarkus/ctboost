# Existing learning options: timing-only follow-up

The chosen CTBoost settings were faster than the saved CatBoost defaults on **13/14 tasks**, with a geometric CTBoost/CatBoost latency ratio of **0.518**. Their geometric latency ratio to CTBoost's earlier default models was **0.934**, close to the unchanged control's 0.937 drift; this does not establish an additional speed improvement. All full-development and timed-batch predictions matched the original model receipts exactly under the final optimized wheel.

This separate plan was frozen after inspecting the development quality studies. The mapping is fixed by task family: `grouped8` for all six binary/regression tasks; `grouped8` plus the existing `multiclass_feature_test="joint"` with scalar cut scores for all eight multiclass tasks. The unsuccessful joint-cut prototype is excluded. These observations are exploratory; they do not establish independent validation, a new default, or an official TabArena result.

No fitting or further quality selection occurred here. We loaded the unchanged saved models, verified all 14 complete development prediction arrays and row indices in a separate process, then measured one fresh pass. The eight joint-feature models were produced by the isolated prototype build with its scalar-cut control; the optimized standard build loaded them without editing their state and preserved every prediction. [The plan](plan.json) pins both source runtimes, all model/data/receipt hashes, and the final inference runtime. [Preflight](preflight.json) records the exact comparisons.

All values below are warm medians in milliseconds for the same deterministic 1,000-row development batches, logical CPU6/one thread, normalization and full model preprocessing included. Model loading and row cycling are excluded. All seven raw blocks, first-prediction cold times, physical tree counts, and whole-process RSS are in [measure.json](measure.json). The same 14 saved CatBoost models serve as controls. Larger learned ensembles are part of this comparison: the eight multiclass models have a geometric **1.575x** physical-tree count relative to CTBoost's defaults.

| Dataset | CTBoost ms | CatBoost ms | CTB/CB | CTB/default latency | Trees CTB/CB |
|---|---:|---:|---:|---:|---:|
| blood-transfusion-service-center | 0.522 | 0.822 | 0.635 | 1.08x | 8/37 |
| coil2000_insurance_policies | 6.779 | 7.000 | 0.968 | 0.81x | 66/205 |
| GiveMeSomeCredit | 2.046 | 1.069 | 1.913 | 0.84x | 168/60 |
| QSAR_fish_toxicity | 1.059 | 1.392 | 0.761 | 0.68x | 204/218 |
| wine_quality | 3.144 | 4.143 | 0.759 | 0.77x | 286/1111 |
| diamonds | 6.449 | 12.291 | 0.525 | 0.94x | 199/2333 |
| MIC | 38.808 | 109.070 | 0.356 | 0.87x | 1080/133 |
| SDSS17 | 3.739 | 14.965 | 0.250 | 0.67x | 273/864 |
| anneal | 22.597 | 32.659 | 0.692 | 1.06x | 5000/1078 |
| hiva_agnostic | 415.326 | 2262.463 | 0.184 | 0.89x | 33/181 |
| maternal_health_risk | 1.575 | 6.344 | 0.248 | 1.30x | 483/1015 |
| splice | 24.714 | 72.043 | 0.343 | 0.90x | 543/835 |
| students_dropout_and_academic_success | 14.731 | 21.819 | 0.675 | 1.30x | 930/490 |
| website_phishing | 5.840 | 13.457 | 0.434 | 1.30x | 1323/420 |

For the eight multiclass settings, latency relative to the default models had a geometric ratio of 1.009, while their CatBoost control drifted by 0.899. Maternal health, students, and website phishing were roughly 30% slower than their default CTBoost models, though still faster than these CatBoost models. These tradeoffs remain visible alongside the aggregate.

The unchanged CatBoost control's geometric drift from the preceding candidate pass was **0.937**, with per-task ratios **0.760–1.184**. Active background training and sequential measurements limit code-only speed attribution. Different model quality and tree counts also limit comparisons between libraries. See the [development quality report](../../accuracy/README.md) for those separate measurements and caveats; this timing follow-up does not recompute scores or read confirmation/outer-test data.

Exact frozen source, plan, 14 CTBoost fit receipts, and raw timing records are included. CatBoost fit receipts and the preceding default timings are in the parent inference archive. Models, feature/target arrays, and predictions remain local and are excluded from this archive.

```powershell
& CANDIDATE_PYTHON -I benchmarks/inference_learning_options_scout.py prepare --output NEW_RUN --original ORIGINAL_INFERENCE_RUN --joint JOINT_SCOUT --cpus 6
& CANDIDATE_PYTHON -I benchmarks/inference_learning_options_scout.py preflight --output NEW_RUN --cpus 6
& CANDIDATE_PYTHON -I benchmarks/inference_learning_options_scout.py measure --output NEW_RUN --cpus 6
```
