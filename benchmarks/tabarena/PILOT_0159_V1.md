# CTBoost 0.1.59 learning-option pilot, version 1

This protocol is frozen before the first pilot fit. The normative machine-readable
record is [`pilot_0159_v1.json`](pilot_0159_v1.json). Its SHA256, the public wheel
and native binary hashes, runner/adapter hashes, hardware and selected CPU layout
must be recorded in the execution manifest before fitting begins. This is a
validation-only development pilot, not a leaderboard evaluation.

The pilot asks whether the new opt-in learning options justify a new limited HPO
run. The statistical feature test still selects a feature before the cut search.
Every arm uses the existing quadratic test, alpha 0.05, Bernoulli sampling and no
external sample/class weights. Grouped testing is excluded: this pilot cannot
overturn its earlier failed runtime qualification or promote a library default.

## Dataset selection and data boundary

Selection uses only the pinned TabArena task metadata. Include all eight eligible
multiclass tasks (3–32 classes). For binary and regression separately, sort by
published training size, then name, and select the smallest, lower median and
largest. The JSON preserves the full 51-task selection input and metadata hash.
No previous validation, test, Elo or timing results inform this selection.

| Family | Pilot datasets |
| --- | --- |
| Binary | blood-transfusion-service-center; coil2000_insurance_policies; GiveMeSomeCredit |
| Regression | QSAR_fish_toxicity; wine_quality; diamonds |
| Multiclass | anneal; hiva_agnostic; maternal_health_risk; MIC; SDSS17; splice; students_dropout_and_academic_success; website_phishing |

Use the official OpenML `r0f0` outer-training indices only. Obtaining the source
table may download the complete table; immediately restrict all preprocessing,
fitting and scoring to outer-training rows. Never predict, inspect scores from,
or serialize labels belonging to the outer test split during this pilot.

Create AutoGluon's native eight-fold `CVSplitter` with seed 0, one repeat, and
classification stratification. Use only inner folds 0 and 1. Model seeds are
respectively 0 and 1 for every arm. All selected metadata tasks are random splits
without group/time constraints; reject metadata drift instead of silently
changing the protocol. Hash the ordered source and inner split indices. Learn
encodings, imputation, CTR state and other preprocessing from each inner-training
fold; inner validation may control early stopping.

## Arms and budget

| Family | Arms |
| --- | --- |
| Binary/regression | baseline; three leaf steps with backtracking |
| Multiclass | baseline; three full-Hessian leaf steps; joint feature test; full three-step leaves plus joint test |

This is **88 fits: 28 baselines and 60 candidates**. Shared parameters include
400 trees maximum, depth 6, learning rate 0.05, L2 1, 256 bins, Bernoulli subsample
0.8, ordered CTR and 50-round validation patience. The JSON gives every override.
Use Logloss/AUC for binary objective/stopping, RMSE for regression and MultiClass
for multiclass. Full and joint options require the built-in CPU multiclass
objective and 3–32 classes. Joint testing requires integer frequency weights;
never round or rescale fractional weights to make a configuration eligible.

Every fit receives the same 300-second setup/training budget. A deadline callback
may retain a valid partial model; record its tree count and stop reason. A
separate watchdog has 30 seconds of grace for an unusually slow tree, after which
the killed fit counts as a failure. Downloads and final validation prediction
are timed separately. No selective retry of failed, slow or unfavorable fits is
allowed. A diagnosed download/environment failure before fitting can be retried
once while preserving both attempt records.

Use the Ryzen 7 5800X3D and its 16 logical CPUs through concurrent CPU workers.
Before the first task fit, compare 1×16, 2×8, 4×4, 8×2, 2×4 and 4×2 worker/thread layouts on a
fixed synthetic workload, then freeze the highest-throughput layout that respects
memory limits. Record the calibration and hold resources constant across arms;
avoid nested BLAS/OpenMP oversubscription. Interleave arms in dataset/fold blocks
with deterministic seed-159 scheduling. Cap each process at 8 GiB RSS and worker
process trees together at 24 GiB, retaining at least 4 GiB free physical memory.
Pause dispatch when memory is tight. No GPU is used because the new full/joint
options being tested are CPU-only.

## Frozen decision gate

Compute each dataset's arithmetic mean validation error over the two inner
folds: binary `1 − ROC AUC`, regression RMSE, multiclass log loss. For each arm,
relative improvement is
`(baseline_error − candidate_error) / max(baseline_error, 1e-12)`; two zero errors
are a tie.
Datasets have equal weight. Runtime ratios compare mean fit seconds across the
two folds on the same dataset; summarize those ratios with a geometric mean
and the 90th percentile using linear interpolation.

A candidate passes separately for a problem-type family only if all conditions
hold:

- Median relative validation error reduction is at least **1%**.
- A strict majority of datasets improve (two of three scalar or five of eight
  multiclass tasks); changes no larger than `1e-12` do not count as wins.
- No dataset's relative error increases by more than **10%**.
- Geometric mean fit-time ratio is at most **3×** and its 90th percentile at most
  **5×**.
- Every predeclared paired fit is eligible, successful and finite, with no OOM
  or hard timeout, and all split, seed, preprocessing and budget checks pass.

Threshold comparisons round to 12 decimals to avoid floating-point boundary
artifacts; dataset wins still use the declared epsilon. Among passing variants
choose the largest median relative improvement, rounded
to 12 decimals for ties. Ties follow the fixed order backtracking, full leaves,
joint test, then combined. If no candidate passes for a family, retain its
baseline. If no family passes, stop after publishing the pilot findings; do not
repeat the old full HPO merely with a new version label.

Publish every arm and failure, including negative results. The small three/eight
dataset panels support a practical development decision, not a statistical
significance or generalization claim. Report secondary binary log loss and
multiclass accuracy, validation prediction time, peak RSS and retained trees,
but do not use them to move the frozen gate.

## Conditional new HPO run

If a family passes, build a separately named `ctboost_0159_learning_hpo25_v1`
portfolio from the unchanged first 25 entries of the old seed-1234 portfolio.
Keep every original numerical/structural value and ordering. Leave the default
and the 12 even-numbered configurations unchanged. The 13 odd-numbered
configurations apply the approved learning variant for their task's family.
An unsupported class count or weight scheme explicitly falls back to baseline
and records why; never modify weights or bootstrap behavior silently.

Freeze all 26 resolved configuration specifications, conditional rules and hash,
the gate report and the full-run CPU/memory/time contract before any new outer
test evaluation. Then run all 51 Lite `r0f0` datasets with eight bag children:
1,326 parents and 10,608 child fits. Keep a new results root and never merge local
timings with historical Kaggle timings. This local pilot's faster per-child
budget does not itself define the full-run per-parent budget.

After validation, publish a separate
`Maiernator/ctboost-tabarena-lite-hpo25-0.1.59` dataset. Preserve 0.1.58 evidence.
Disclose the pilot's reuse of Lite outer-training data, all selected options and
the local hardware. Test rows remain untouched by pilot selection, but the
result is still author-developed, noncanonical Lite evidence; it does not
establish official leaderboard acceptance.

Never edit this protocol after fitting starts. A necessary implementation fix
gets an append-only amendment and a new affected-pairs run, retaining the
original records. Do not revise the admission thresholds after seeing results.
