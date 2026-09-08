# Experimental multiclass joint-cut results - 8 September 2026

The isolated joint-cut candidate **failed its predeclared admission gate** against the paired joint-feature/scalar-cut control. No learner default or release was changed by this study. The original confirmation fold remains sealed.

All eight multiclass tasks from the original 14-task panel were included. Their immutable outer-training roles were reused: training folds 0-4, stopping fold 5, development fold 6. The original 10,000-row cap and seed 47 remain unchanged. This reuses the datasets and development rows inspected in earlier failed grouped and temperature studies; it is staged exploratory work, not independent confirmation or an official TabArena/Elo evaluation. Neither official outer-test rows nor the old fold 7 entered fitting or scoring.

The three arms were fixed before fitting: current default (single feature test, quadratic numeric test, scalar cut), existing joint feature test with grouped-eight numeric tests and scalar cuts, and the same joint-feature control with only multiclass_split_score changed to joint. All arms retained diagonal leaves, one leaf-estimation step, no backtracking, one_output_per_tree storage, no feature-test multiplicity adjustment, 1,000-round cap, learning rate 0.05, depth 6, alpha 0.05, L2 1, Bernoulli subsampling 0.8, ordered CTR, categorical threshold 64, patience 50, and seed 47. The candidate scores cuts only after the conditional feature selection; it does not replace that principle.

All 24 CTBoost models were refitted in the isolated prototype environment. The prototype had already passed 208 focused/relevant tests (six optional-dependency/CUDA skips) and eight small public-default state/prediction equivalence probes. This study separately checks all eight fresh task-level default predictions and iteration counts against the archived public-0.1.60 fits. An additional audit found complete native states equal for all eight fresh/archived defaults after loading both in the prototype. Archived AutoGluon/XGBoost outputs are secondary references using the same immutable development rows. Their versions and artifact hashes are retained.

The runtime wheel SHA256 is `e10c7ca663313fcd92083475741393c8fbe20d9472fa2f8645ec8125298019d7`, native SHA256 `197995abd2ea3ea0679d0711c0d3ed7ee0f9fd897c3444cd833a3b4c021a4e8a`, and source snapshot SHA256 `66215d90e134d835ca7182aedd8df06f7e3426d9e311ab3aeb996dd248a185d9`. Although its package version remains 0.1.60, this is an unpublished experimental build identified by those hashes. The experimental source is isolated from the active Full HPO run.

The prototype uses the public learner source plus the experimental joint-cut changes; it does not include the separate concurrent inference-cache optimizations. This study measures cut-scoring quality and model size, not the speed gain of those optimizations.

One initial preparation was withdrawn before any fit or new score to add requested physical-tree counts and correct an import-format lint issue. Its exact runner, plan, and reason remain in withdrawn_prelaunch; the final parameters, roles, and gate were unchanged. The finalized v2 plan hash is `98eded2dfd0a2cb1db2391e73aa9add25351e72aada41dec0882a448106f168f`.

## Frozen gate

| Requirement | Required | Observed |
| --- | ---: | ---: |
| Median relative log-loss improvement over paired control | >= 1% | -1.062% |
| Dataset wins | >= 5/8 | 2/8 |
| Worst relative regression | <= 5% | 77.696% |
| Geometric mean fit-time ratio | <= 3 | 1.198 |
| 90th percentile fit-time ratio | <= 5 | 2.727 |
| All 24 fits valid | true | true |
| All 8 default prediction/iteration audits exact | true | true |

Errors below are multiclass log loss. Positive gains mean lower error. These comparisons were secondary to the paired-control gate; they were not used to retune any arm.

| Dataset | Current default | Joint feature/scalar cut | Joint feature/joint cut | Archived XGBoost | Gain vs paired control | Gain vs current default |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MIC | 0.389079 | 0.422052 | 0.438367 | 0.418782 | -3.866% | -12.668% |
| SDSS17 | 0.105438 | 0.100832 | 0.09291 | 0.102768 | +7.857% | +11.882% |
| anneal | 0.0373888 | 0.00594981 | 0.0105726 | 0.0241102 | -77.696% | +71.723% |
| hiva_agnostic | 0.170093 | 0.170126 | 0.171047 | 0.170131 | -0.541% | -0.560% |
| maternal_health_risk | 0.715634 | 0.595196 | 0.582746 | 0.607423 | +2.092% | +18.569% |
| splice | 0.12475 | 0.140577 | 0.149272 | 0.137754 | -6.185% | -19.657% |
| students_dropout_and_academic_success | 0.566301 | 0.572022 | 0.581082 | 0.616078 | -1.584% | -2.610% |
| website_phishing | 0.245985 | 0.200075 | 0.200818 | 0.203294 | -0.371% | +18.362% |

| Comparison | Wins | Median relative error gain |
| --- | ---: | ---: |
| Candidate vs paired control | 2/8 | -1.062% |
| Candidate vs current default | 4/8 | +5.661% |
| Paired control vs current default | 4/8 | +2.174% |
| Candidate vs archived XGBoost | 5/8 | +2.640% |
| Current default vs archived XGBoost | 4/8 | -1.288% |
| Paired control vs archived XGBoost | 6/8 | +1.733% |

## Training cost and model sizes

Each cell lists iterations / physical trees / fit seconds. Physical tree counts include one scalar tree per class for each boosting iteration. Fit times were collected under two-worker execution with two logical CPUs, an 8 GiB process-tree memory ceiling, and a 300-second soft / 390-second hard limit per fit. Concurrent workloads make these descriptive development timings, not canonical inference or training benchmarks.

| Dataset | Current default | Joint feature/scalar cut | Joint feature/joint cut |
| --- | ---: | ---: | ---: |
| MIC | 100 / 800 / 30.62 | 135 / 1080 / 53.19 | 115 / 920 / 60.45 |
| SDSS17 | 354 / 1062 / 21.47 | 91 / 273 / 7.70 | 382 / 1146 / 31.52 |
| anneal | 389 / 1945 / 16.43 | 1000 / 5000 / 48.69 | 615 / 3075 / 48.55 |
| hiva_agnostic | 1 / 3 / 93.61 | 11 / 33 / 131.72 | 8 / 24 / 126.38 |
| maternal_health_risk | 134 / 402 / 0.30 | 161 / 483 / 0.64 | 425 / 1275 / 1.36 |
| splice | 237 / 711 / 19.53 | 181 / 543 / 28.39 | 215 / 645 / 37.82 |
| students_dropout_and_academic_success | 132 / 396 / 6.64 | 310 / 930 / 13.20 | 122 / 366 / 8.04 |
| website_phishing | 246 / 738 / 1.22 | 441 / 1323 / 3.65 | 246 / 738 / 2.00 |

The run took 8.46 minutes from first fit start to last completion; successful fit times sum to 793.11 seconds. Peak process-tree RSS was 0.873 GiB, with 0 soft-deadline stops and 0 resource failures. All saved-model reload predictions were exact: true.

[Frozen plan](plan.json), [runner](../../../../accuracy_joint_cut_scout.py), [complete report](development_report.json), [comparison summary](comparison_summary.json), [all 24 fit records](fit_records), [source/wheel manifest](prototype-manifest.json), [pre-launch default probes](prototype-default-equivalence.json), [complete default-state audit](default_native_state_audit.json), [confirmation barrier audit](protocol_barrier_audit.json), [runtime packages](requirements.txt), [withdrawn pre-launch preparation](withdrawn_prelaunch). Models, labels, raw rows, and prediction arrays remain local and are excluded from this report bundle.

The existing joint-feature control is examined separately in the [secondary comparison](existing_joint_feature_secondary.md), including its mixed default-relative quality and larger model sizes. This does not change the failed new-joint-cut decision.

After the failed study, the exact experimental source was preserved locally as commit `6c3fed74ebbbda8350fa521bcca44cc715c2f81a` on `experiment/joint-cut-20260908`; all 188 registered source-file hashes still match. [Preservation receipt](source_preservation_receipt.json). This commit was not merged or pushed.
