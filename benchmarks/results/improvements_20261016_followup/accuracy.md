# Conditional-inference accuracy experiments

CTBoost retains its conditional feature-association test before cut selection.
This bounded development study explores existing controls and separately frozen
post-fit policies, without changing production training defaults. It compares
XGBoost **3.4.1** and LightGBM **4.7.0** on twelve cached real tasks. It is not a
full TabArena run, an Elo score, an unbiased estimate, or a universal ranking.
The exact isolated CTBoost runtime is version **0.1.61**, native SHA256
`c5ea7e026942c3f1190ca5511fbf74f353bfeb08e7771f76bc5f6c1ca3e4d197`.
Subsequent production inference changes do not alter this accuracy runtime.

## Design and limits

The twelve tasks were selected from the local 51-task cache using metadata,
before this study's scores: at most 256 features, with four evenly spaced
row-count ranks in each of binary, regression and multiclass families. Original
outer-training samples were capped at 10,000 rows. TRAIN consists of folds 0-4,
STOP is fold 5. The older 14-task cache uses fold 6 as DEV and leaves fold 7
confirmation unopened; remaining-task DEV combines folds 6 and 7. Cached role
identities are verified explicitly. No original confirmation file has been
opened and no official-test model score has been computed.

Primary errors are `1 - ROC-AUC` for binary classification, RMSE for regression,
and log loss for multiclass. Lower is better. Positive relative error gain
favors CTBoost. The gate requires all mandatory fits to succeed, nonnegative
median relative error gain against **each** comparator, and at least six of
twelve dataset wins/ties against each. Six wins alone need not make the median
positive. Results are aggregated over twelve datasets, not seeds multiplied by
datasets. Repeated development access and adaptive hypotheses make these
exploratory comparisons; stopping/CV scores are not unbiased final estimates.

Every wave fixes its controls before its new fits or candidate evaluations.
All selected models are persisted and hash-frozen before DEV. Early phases use
STOP-only selection. Later mandatory refits have no original-versus-refit choice
or DEV fallback. Categorical vocabularies use actual fitting rows; unknown
categories become missing. Ordered target transforms are reconstructed and
state-verified for training-row leaf correction, avoiding inference-transform
leakage. Native fold-in checks preserve all non-leaf topology metadata and
routing, followed by numerical oracles and pickle prediction parity.

Equal candidate counts and common row roles do not equal compute or model
capacity, nor exhaustive HPO. Fit counts refer to trained ensembles, not physical trees. Some declared controls are inert or repeated;
these remain counted as control fits. XGBoost multiclass vector-tree choices
are included. All native learners use one thread. Early waves use two workers;
late bounded refits often run sequentially. Fit seconds are diagnostics, not
isolated speed benchmarks, because builds/tests sometimes overlap. Actual
solver, preprocessing, model-size and resource receipts remain available.

Native early-stopping and external metrics target the same statistical metrics
but need not agree numerically: raw-margin AUC and probability AUC can differ
under ties. Float sigmoid rounding and probability clipping can materially
distort extreme native log losses; the [synthetic metric audit](inference/metric_numerics_audit/README.md)
reproduces this without measuring any actual checkpoint or portfolio impact. CTBoost
prunes training history with its model, so physical patience-round counts are
unavailable. Retained-at-cap is not equivalent to physically reaching a cap.

## Completed development waves

Every row below failed at least one part of the predeclared full-panel gate.
All failures remain evidence; there is no selection of the most favorable wave.

| Wave | New policy | Gain vs XGBoost | Wins | Gain vs LightGBM | Wins |
|---|---|---:|---:|---:|---:|
| 1 | 8 profiles per task/library | -4.772% | 4/12 | -2.691% | 5/12 |
| 2 | 8 further profiles | -0.193% | 6/12 | -0.418% | 6/12 |
| 3 | Categorical/joint-test portfolio | +0.585% | 7/12 | -0.418% | 6/12 |
| 4 | Paired row sampling and regularization | -0.370% | 5/12 | -3.510% | 4/12 |
| 5 | Paired alpha and capacity | -0.636% | 5/12 | -3.641% | 4/12 |
| 6 | Fixed regression leaf correction | -0.433% | 5/12 | -2.757% | 4/12 |
| 7 | Fixed multiclass temperature | +0.459% | 6/12 | -2.098% | 5/12 |
| 8 | 8 binary/regression power profiles | +1.106% | 7/12 | -2.098% | 5/12 |
| 9 | Mandatory TRAIN + STOP refit | -0.600% | 5/12 | -2.179% | 4/12 |
| 10 | Fixed binary leaf correction | -0.600% | 5/12 | -2.179% | 4/12 |
| 11 | Mandatory two-seed ensemble | -0.623% | 5/12 | -0.714% | 6/12 |
| 12 | 4 threshold/LR profiles + refit | +0.073% | 6/12 | -1.685% | 5/12 |
| 13 | Five-fold length selection + refit | -0.539% | 5/12 | -3.158% | 4/12 |
| 14 | Three cached training-data expansions | +0.198% | 6/12 | -1.439% | 6/12 |
| 15b | Houses training-data expansion | -0.143% | 6/12 | -1.439% | 6/12 |
| 16 | Fixed multiclass leaf correction | -0.143% | 6/12 | -1.439% | 6/12 |
| 17b | Certified scalar correction refresh | +0.286% | 7/12 | -1.359% | 5/12 |
| 18b recovery | Full-current-data CV counts + mandatory refit | +0.207% | 7/12 | -0.822% | 5/12 |

### Waves 1-8: bounded controls and fixed topology

Waves 1-5 each add eight fits per task/library (288 fits per wave). Wave 1
includes the missing unrefined learning-rate 0.15 arm and clean learning-rate
0.05/0.15 by one/three leaf-step controls. Later portfolios cover grouped versus
quadratic tests, depth, leaf limits, L2, ordered target representations,
one-hot limits, multiclass joint tests, paired row sampling 0.8/1.0, and paired
alpha values 0.05/0.5. No alpha=1 unconditional arm is used. The exact grids are
in each immutable plan; descriptions do not replace them. Initial caps are
1,000 rounds, then 2,000. The unlaunched original wave-1 plan was withdrawn after
preflight review; `accuracy_wave1_v2` is the actual experiment, not an extra
observed quality trial.

For unconstrained squared loss and fixed leaves, the initial Newton update
already solves the regularized quadratic leaf objective. Repeated scalar leaf
steps are therefore not expected to solve a general regression deficit. Wave 6
instead performs a training-only joint ridge correction of fixed leaf
indicators, with four penalties and two shrinkages plus identity for every
regression task/library. Four solves are reused across eight candidates; STOP
RMSE chooses the actual native-folded result. Identity wins exact ties. All 96
candidates passed in the v2 run. The initial attempt failed on serialization of
a diagnostic NumPy scalar; its failure and source are preserved, and v2 changes
only diagnostic scalar conversion. No tree topology or coefficients were
changed by that compatibility repair.

Wave 7 applies the identical eight-temperature grid
[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 4] to all four multiclass tasks and all three
libraries. STOP log loss selects; identity wins ties. The fixed temperature is
applied to float64 margins exactly once in every scorer. This is not a novel
hypothesis: the September temperature scout had already shown a small median
gain and failed its 1% gate; current models and matched-library budgets differ.
Wave 8 adds eight fits on each binary/regression task/library, using CTBoost
grouped bins [2,4,16,32] by alpha [0.2,0.5] and comparator L1/feature-sampling
controls. Multiclass policies carry unchanged. Grouped numeric/CTR/one-hot tests
and native categorical tests are not the same operation; effects are
configuration evidence, not proof of a particular power mechanism.

### Waves 9-12: refitting, corrections, seed mixtures and thresholds

Wave 9 mandates 36 TRAIN + STOP refits, retaining selected requested rounds,
all tree controls and fixed post-fit controls. There is no early stopping,
DEV selection or old-model fallback. New vocabulary and ordered transforms are
fit on the union. Wave 10 selects binary logistic-offset ridge corrections
using the original wave-8 TRAIN-fitted models and their independent STOP rows;
it never uses the now-in-sample STOP of union models for selection. The grid is
lambda [1,10,100,1000] by shrinkage [0.25,1], plus identity, equally for all
libraries. TRAIN log loss is the optimization objective and STOP AUC the
selection metric. All twelve controls freeze before fixed recomputation on
union models; all 96 candidates pass. Six nonzero controls are selected. These
corrections add no trees and do not change routes.

Wave 11 is a new fixed two-seed ensemble hypothesis after wave 10 failed. It
adds 36 fresh seed-947 models, retains every selected tree/post-fit control,
and equally averages probabilities for classification or predictions for
regression. Stable output transforms and the native LightGBM unit-logit link
are verified. It does not reinterpret any earlier failed seed gate as passed.
Duplicate component predictions are retained (two CTBoost, six XGBoost and two
LightGBM pairs). The predeclared second pair, seeds 137/1543, is not run because
the initial pair fails. All 36 real ensembles and 108 batch cases later pass
exact prediction checks, but those checks do not change the quality outcome.

Wave 12 returns to a separately declared single-model policy. On eight
binary/regression tasks, four additional profiles pair CTBoost alpha 0.5/0.8,
XGBoost minimum child weight base/quarter, or LightGBM minimum leaf rows
base/quarter, with learning rate base/half. Base-LR runs use 2,000 rounds and
50-round patience; half-LR runs use 4,000 and 100 for every library. Seventeen
historical-control repeats remain intentional fits. Early stopping chooses the
uncorrected native prefix; then fixed correction coefficients are recomputed on
TRAIN, and corrected STOP error selects the profile. This does not optimize the
corrected prefix. Fourteen original TRAIN incumbents and ten new profiles win;
24 mandatory union refits follow, while twelve multiclass single models carry.
All 96 search fits and 24 refits pass, with no observed retained-cap hit.
Conditional seed 947 is not run; its prior wave-11 use would preclude calling it
an unseen independent confirmation seed in any case.

### Wave 13: independently fitted cross-validation lengths

Wave 13 fixes shared five-fold assignments on original TRAIN + STOP with split
seed 13013. It independently fits each library at ceil(half), unchanged and
double the wave-12 requested rounds, for 540 fits, then mandates 36 final union
refits. No learner is sliced to simulate a shorter fit. Raw rows are combined
before fitting fold-local normalization/vocabulary. Every classification
training and validation fold retains every class, including anneal's rare
class. Frozen corrections are solved on each fitting fold only, and temperature
is applied once before its held-fold score. Mean fold primary error chooses
length; ties prefer fewer requested rounds then profile index. A duplicate
integer length is retained, not replaced with another trial. Fold AUC is not
pooled across incomparable fold margin scales. Inherited controls had already
used STOP, so this is a tuning procedure, not an unbiased CV estimate.

All 576 fits pass. Each worker is guarded at 1,200 seconds and 8 GiB summed
owned process-tree RSS; Windows launcher descendants are tracked by PID and
creation time and killed child-first on failure. Quick-exit, timeout, child
memory, unrelated-process survival and AccessDenied cleanup tests cover the
watchdog. Unsafe cleanup prevents further dispatch. Two CPU workers are used.
The maximum job is 221.16 seconds and 1.072 GB; summed concurrent job wall time
is 1,247.52 seconds. Every fold retains source/schema/fold IDs, model hash,
round-trip verification, predictions and actual cost; intermediate fold models
are not all retained. The final 36 models are retained.

Half/same/double selections are CTBoost 3/6/3, XGBoost 2/6/4 and LightGBM 3/6/3.
Every final requested count is also its retained native count. All 36 changes
are listed, rather than only favorable tasks:

| Dataset | Library | Anchor | Requested | Retained |
|---|---|---:|---:|---:|
| blood-transfusion-service-center | ctboost | 5 | 3 | 3 |
| blood-transfusion-service-center | xgboost | 28 | 56 | 56 |
| blood-transfusion-service-center | lightgbm | 1 | 2 | 2 |
| churn | ctboost | 61 | 61 | 61 |
| churn | xgboost | 115 | 230 | 230 |
| churn | lightgbm | 6 | 12 | 12 |
| in_vehicle_coupon_recommendation | ctboost | 534 | 534 | 534 |
| in_vehicle_coupon_recommendation | xgboost | 682 | 682 | 682 |
| in_vehicle_coupon_recommendation | lightgbm | 424 | 424 | 424 |
| GiveMeSomeCredit | ctboost | 164 | 328 | 328 |
| GiveMeSomeCredit | xgboost | 64 | 64 | 64 |
| GiveMeSomeCredit | lightgbm | 87 | 87 | 87 |
| QSAR_fish_toxicity | ctboost | 32 | 32 | 32 |
| QSAR_fish_toxicity | xgboost | 117 | 59 | 59 |
| QSAR_fish_toxicity | lightgbm | 38 | 38 | 38 |
| Another-Dataset-on-used-Fiat-500 | ctboost | 58 | 116 | 116 |
| Another-Dataset-on-used-Fiat-500 | xgboost | 28 | 56 | 56 |
| Another-Dataset-on-used-Fiat-500 | lightgbm | 21 | 42 | 42 |
| houses | ctboost | 167 | 167 | 167 |
| houses | xgboost | 1087 | 1087 | 1087 |
| houses | lightgbm | 381 | 191 | 191 |
| diamonds | ctboost | 256 | 512 | 512 |
| diamonds | xgboost | 154 | 154 | 154 |
| diamonds | lightgbm | 1195 | 1195 | 1195 |
| anneal | ctboost | 778 | 389 | 389 |
| anneal | xgboost | 240 | 240 | 240 |
| anneal | lightgbm | 137 | 69 | 69 |
| website_phishing | ctboost | 118 | 118 | 118 |
| website_phishing | xgboost | 243 | 122 | 122 |
| website_phishing | lightgbm | 85 | 43 | 43 |
| splice | ctboost | 169 | 169 | 169 |
| splice | xgboost | 69 | 69 | 69 |
| splice | lightgbm | 157 | 157 | 157 |
| SDSS17 | ctboost | 345 | 173 | 173 |
| SDSS17 | xgboost | 220 | 440 | 440 |
| SDSS17 | lightgbm | 160 | 160 | 160 |

The wave fails both comparators. Conditional seed 137 and official assessment
remain unopened. Counts reach 2,400 benchmark model fits, 800 per library, plus
eight separate diagnostic replay fits.

### Wave 14: three existing training-only caches

Wave 14 expands exactly the three tasks with reviewed existing official-TRAIN-
only caches: GiveMeSomeCredit, diamonds and SDSS17. Availability, not score,
determines eligibility. It mandates nine fits and carries 27 models exactly,
retaining wave-13 requested counts and every post-fit control. Historical caps
and role IDs are reconstructed and TRAIN/STOP values verified. Full official-
TRAIN labels are administratively read for historical stratification and class
IDs, including excluded DEV/confirmation labels; those excluded labels never
enter fitting, normalization, correction or scoring. No confirmation file or
official-test values are read. Rows are original TRAIN, STOP, then ascending
outside-cap official-TRAIN positions.

| Dataset | Original fitting rows | Added | New fitting rows | DEV excluded | Confirmation excluded |
|---|---:|---:|---:|---:|---:|
| GiveMeSomeCredit | 7,500 | 90,000 | 97,500 | 1,250 | 1,250 |
| diamonds | 7,500 | 25,960 | 33,460 | 1,250 | 1,250 |
| SDSS17 | 7,500 | 42,035 | 49,535 | 1,250 | 1,250 |

Houses remains unchanged in this wave because its available parquet would
decode test values. All nine fits pass and retain requested counts. Twenty-nine
synthetic tests pass, including eighteen native toy fits. Maximum job time is
40.81 seconds, peak summed RSS 2.330 GB, total sequential job time 125.59 seconds.
Complete policy payloads are 79.52/73.12/12.23 MB for CTBoost/XGBoost/LightGBM.
Counts reach 2,409 benchmark model fits, 803 per library. The XGBoost gate passes but the
LightGBM median misses, so seed 239 and test remain closed. More data changes
test power and effective regularization jointly; fixed controls chosen on the
smaller sample can be suboptimal after expansion.

### Wave 15: authorized houses rows and preserved compatibility failures

Wave 15 adds **three mandatory houses fits plus 33 exact carries** from wave 14.
All controls, requested counts, corrections and temperatures remain fixed. The
original plan and authorized global-row manifest freeze before body acquisition.
The 1,446,365-byte ARFF matches known MD5 and gets a separate SHA256 receipt before
parsing. The initial parser stops at the header's REAL type label, before any
data instance is parsed or fit is started. That entire attempt and its sources
are preserved in `accuracy_wave15/`.

The separately reviewed `accuracy_wave15b/` retry adds only the numeric REAL
alias alongside NUMERIC. Its new protocol freezes after header-only diagnosis,
but before retry parsing. It binds the same authorized IDs and rehashes the
existing bytes without refetch. Ordered schema, numeric, hash and exclusion
checks remain unchanged. Only 11,260 instances are parsed: original TRAIN
(6,250), STOP (1,250), then 3,760 outside-cap official-TRAIN rows. All original
TRAIN/STOP features and targets match exactly, without rounding tolerance.
The 6,880 official-test and 2,500 DEV instances remain opaque: zero decoded or
tokenized. Raw storage/hashing includes excluded bytes; this proves selected-
row parse integrity, not numeric validity of excluded values. This houses split
has no separate confirmation role. Source SHA256:
`c71f44ba2522f61a8ba1918c220074be4a771e1c8c8c07d031e98368f4f94c6c`.

All three fits and native pickle checks pass, retaining 167/1,087/191 requested
rounds for CTBoost/XGBoost/LightGBM. The compatibility suite passes 52 tests.
The first development scorer finishes scoring but fails to write its report
because its plan omits the descriptive limitations field. A separate reviewed
reporting wrapper supplies that text from the frozen policy in memory, keeps
the original 36 prediction archives untouched and checks prediction/label/ID
dtypes, shapes and bytes against separately regenerated archives before writing
the report. Its focused contract test passes. No model, selection or numeric
scorer changes. Both failures and all original sources remain preserved.

Maximum job time is 17.86 seconds, peak summed RSS 0.796 GB, and sequential job
time 36.01 seconds, with no resource failures. Complete policy payloads are
83.84/73.15/12.21 MB. These are diagnostic costs. At this stage, counts reach **2,412 benchmark model fits**, 804 per library, plus eight separate replays. Parser/report retries
add no fits. Houses RMSE improves for all libraries: CTBoost 2.364%, XGBoost
3.024%, LightGBM 2.029% versus wave 14. Nevertheless the full-panel policy fails
both aggregate medians. Seed 349 and official test remain closed.

### Wave 16: current-control multiclass leaf correction

Twelve fresh multiclass TRAIN anchors use the current fixed tree controls and
requested counts. SDSS17 adds authorized outside-cap rows after original TRAIN,
excluding original STOP; other anchors use original TRAIN. The frozen inherited
temperature is fixed throughout. A training-only joint softmax correction of
existing leaf outputs uses lambda [1,10,100,1000] by shrinkage [0.25,1], plus
clean identity for every task/library. Actual native-folded STOP log loss selects;
exact ties prefer identity, then larger lambda, then smaller shrinkage. All
twelve choices freeze before mandatory recomputation on the parent union or
expanded models in their exact fitting order. No topology or routing changes,
and the other 24 binary/regression models carry exactly.

All 12 anchor fits, 48 selection solves, 96 nonidentity evaluations, 12 identity
evaluations and 12 transfer jobs pass. Eight nonzero controls require eight
transfer solves; four identities preserve the parent artifact. The fixed
temperature is applied once, and scalar class support and vector leaf layouts
are preserved. Independent numeric-gradient/dense-Newton and native-format
tests pass (36 helper tests plus nine experiment integration tests) in the exact
immutable accuracy runtime.

| Dataset | Library | Lambda | Shrinkage | Anchor rows | Requested = retained rounds |
|---|---|---:|---:|---:|---:|
| anneal | ctboost | 1 | 1 | 375 | 389 |
| anneal | xgboost | identity | - | 375 | 240 |
| anneal | lightgbm | identity | - | 375 | 69 |
| website_phishing | ctboost | 100 | 1 | 565 | 118 |
| website_phishing | xgboost | 100 | 1 | 565 | 122 |
| website_phishing | lightgbm | 1000 | 1 | 565 | 43 |
| splice | ctboost | 100 | 0.25 | 1,330 | 169 |
| splice | xgboost | 10 | 0.25 | 1,330 | 69 |
| splice | lightgbm | 100 | 0.25 | 1,330 | 157 |
| SDSS17 | ctboost | 1000 | 1 | 48,285 | 173 |
| SDSS17 | xgboost | identity | - | 48,285 | 440 |
| SDSS17 | lightgbm | identity | - | 48,285 | 160 |

The named-array preflight budget is 2 GiB; the separate owned-process RSS bound
is 8 GiB with 1,200 seconds per complete anchor/selection or transfer job.
Maximum observed named-array estimate is 1.320 GB, peak RSS 2.384 GB and job
time 239.078 seconds, with no resource failures. Summed stage job times are
521.858 seconds for anchor/selection and 37.657 for transfer. These are diagnostic
costs, not isolated timing comparisons. Complete policy payloads occupy
83.849/73.147/12.217 MB for CTBoost/XGBoost/LightGBM.

The multiclass-family median gain becomes +14.753% versus XGBoost and +0.220%
versus LightGBM, with 3/4 and 2/4 wins respectively. Absolute task losses below
show both gains and regressions. The full twelve-task gate still fails: both
aggregate medians and win counts are unchanged from wave 15. Conditional seed
457 and official test remain closed. Counts reach **2,424 benchmark model fits**,
808 per library, plus eight separate diagnostic replay fits. Correction solves
are recorded separately from model-training runs.

### Wave 17: scalar correction refresh stops on a numerical failure

This prospective refresh replaces stale binary/regression correction controls
from clean current-control TRAIN anchors, followed by 24 planned clean union
fits. Both previous scalar correction fields are disabled before fitting; a
new correction cannot be stacked on an already corrected baseline. Credit,
diamonds and houses add authorized outside-cap rows after TRAIN while excluding
STOP from anchor fitting and vocabulary construction. Fixed tree controls and
requested counts are unchanged. The same four penalties and two shrinkages,
plus clean identity, apply to all eight scalar tasks and all three libraries.
The twelve wave-16 multiclass models were intended to carry unchanged.

| Wave | Terminal status | Completed new model-training runs | Candidate evaluations | Union / DEV |
|---|---|---:|---|---|
| 17 | Numerical convergence failure | 24 of 48 planned | 190/192 nonidentity and 24/24 identity pass | 0 union fits; no DEV scoring |

All 24 anchor fits pass. Of 96 attempted coefficient solves, 95 converge under
the frozen gates. The diamonds LightGBM lambda-1 LSQR solve reaches its frozen
2,000-iteration limit with `istop=7`, failing both shrinkage candidates. The
reported residual norm is 30,548.785, normal-residual norm 890.850 and condition
estimate 142,563.346; the latter is a solver estimate, not an independently
measured condition number. The other 23 complete selection jobs pass. No
solver limit, tolerance or numerical acceptance rule is changed. The all-success
barrier prevents every mandatory union fit and all DEV/seed-569/official scoring.
No partial controls are promoted and no quality gain, win count or ranking is
assigned to wave 17. At that point, wave 16 remained the latest complete evaluated policy.

The largest completed worker takes 424.437 seconds and reaches 2.285 GB summed
owned-process RSS; the largest named-array estimate is 2.162 GB. Total selection
job time is 1,277.217 seconds, with no watchdog failure. These are diagnostic
resource receipts, not isolated speed measurements. At that stage, cumulative work was
**2,448 benchmark model fits**, 816 per library, plus eight separate replay
fits. The abandoned 24 final fits are not counted. See the [terminal summary](accuracy_wave17/summary.json),
[failed selection document](accuracy_wave17/scalar_controls.json), and
[independent numerical-solver design review](numerical_ridge_review.md).
Any TRAIN-only numerical diagnostic is separately frozen and does not repair
or overwrite this failed experiment.

### Wave 17b: certified numerical retry and completed scalar refresh

The original wave 17 remains a terminal numerical failure. A separate TRAIN-only
extended diagnostic certified zero-start, unscaled LSQR at 5,541 iterations
(with a prospectively fixed 10,000 cap), relative original-objective gap
1.8320e-14 and normalized original gradient 2.1776e-9. It performed no new
model fit or STOP/DEV score. Its success established numerical feasibility,
not an accuracy gain.

The separately frozen [17b plan](accuracy_wave17b/plan.json) reuses all 24 clean
TRAIN anchors and 12 successful binary grids from the failed wave. All 48 new
regression penalty solves certify, and all 96 new native-folded candidate
evaluations pass. The original 96 binary evaluations remain exact reuses.
All 24 STOP choices freeze before the 24 mandatory clean union refits; 15 select
a nonzero correction and nine select clean identity. Prior scalar corrections
are disabled before every fresh union fit. All 24 recorded nonleaf/schema
checks and final persistence checks pass, followed by 15 fixed correction
solves. Twelve wave-16 multiclass artifacts carry byte-for-byte unchanged.

The [selection audit](validation/wave17b_selection_independent_review.json)
reconstructs every strict choice, including all 108 binary identity/candidate
STOP metrics from saved archives in the exact frozen metric environment.
The [terminal audit](validation/wave17b_terminal_independent_review.json)
verifies sources, closed receipts, model/archive hashes, chronology and all
development aggregates. Regression certificates are checked for recorded
arithmetic/threshold consistency, not rebuilt from the design matrices.

The completed gate still fails: CTBoost's median relative error gain is
**+0.286104% against XGBoost (7/12 wins)** and **-1.358863% against LightGBM
(5/12 wins)**. All errors come from the frozen single seed-47 policies; there
is no DEV-selected fallback. Conditional seed 569 and official test remain
unopened. The same immutable c5 accuracy runtime is retained; production CTR
precision changes are not part of this comparison.

The 24 new union fits bring cumulative work to **2,472 benchmark model fits**,
824 per library, plus eight separate diagnostic replays. Anchor reuse and
coefficient solves are not counted as new model fits. Maximum worker duration
is 445.578 s, peak summed process-tree RSS 2.434 GB, and maximum named-array
estimate 2.264 GB, with no resource failures. Selection and transfer worker
durations sum to 1,327.749 s and 268.909 s. These are diagnostic durations:
a separate production correctness suite overlapped part of the union stage.
Policy payload sizes are CTBoost 83.849 MB, XGBoost 73.149 MB and LightGBM
12.217 MB. Exact controls, costs and all twelve errors are retained in the
[closed summary](accuracy_wave17b/summary.json).

### Wave 18: full-current-data CV stops at the headroom gate

The new fixed protocol applies shared five-fold, raw fold-local count selection
to the complete current fitting populations, including all four authorized
expansions. It inherits the 17b controls and requests half/base/double counts,
with 540 independent fold fits followed by 36 mandatory union fits. Its frozen
resource plan uses one CPU-4 worker, 6 GiB named arrays, 10 GiB process-tree RSS,
4 GiB system reserve and 1,200 s per job. All 216 prospective fold/full-union
allocation profiles passed; the maximum named estimate was 4.205 GiB.

This attempt ends before selection. The first **58 model fits all pass**; before
job index 58, available host memory remains below the required 14 GiB for all
61 samples in the fixed 300 s wait. The largest available sample is
14,901,506,048 bytes and the last is 14,771,314,688, versus the required
15,032,385,536. That worker is never started; all 481 remaining jobs are marked
not started after terminal failure. No owned-worker RSS failure occurs.

No count choice, union fit, DEV, conditional seed-677 or official assessment
is run. This is a resource-admission failure, not a model-quality result or a
failed convergence claim. At that point, the complete 17b policy remained the
latest evaluated policy. Cumulative work reached **2,530 benchmark model fits** plus eight
separate diagnostic replays. The incomplete 58-run allocation is not balanced
across libraries and supports no quality comparison.

The [closed summary](accuracy_wave18/summary.json) and
[independent resource audit](validation/wave18_terminal_resource_retry_review.json)
preserve the failure. A prospectively declared resource-only retry can tighten
the process cap to 8 GiB while retaining 6 GiB named arrays and 4 GiB reserve,
requiring 12 GiB available before dispatch. It must use fresh outputs, restart
all 540 + 36 fits without reusing these 58, and preserve every source, fold,
method, numerical and quality gate. This recommendation alone is not activation
or evidence that the tighter cap will suffice.

### Wave 18b: externally interrupted during fold validation

The resource retry completed 297 fully verified fold jobs. A further native
fit returned for houses/XGBoost, doubled count 2,174, fold 2 (global job 297),
but its worker was then mistakenly stopped by the reviewing agent while that
agent was checking an unrelated archive scan. The exact command and failed
ownership check are preserved in the [incident report](release_preparation/wave18b_external_interruption_incident.json).
This was external agent intervention, not evidence of a library, numerical,
RSS, headroom, timeout or watchdog failure.

The closed attempt records 298 completed native fits, 297 verified jobs and
242 jobs not started. No count choices, union fits, development assessment,
seed677 check or official-test score occurred. Cumulative benchmark model fits
reached 2,828, with eight earlier diagnostic replays separate. The [independent
metadata audit](validation/wave18b_terminal_independent_audit.json) verifies
all 540 identities, source bindings and closed counts. At that point, the latest
completed quality policy remained 17b; the interrupted attempt had no quality result.

The interrupted attempt remains immutable. The separately reviewed recovery
below completes the same fixed experiment. The interruption did not authorize
a numerical/resource fallback or a new accuracy method.

### Wave 18b recovery: completed current-data count selection

The [recovery plan](accuracy_wave18b_recovery/recovery_plan.json) binds the
byte-identical 18b scientific plan, runtime, sources, controls and fold manifest.
It reuses the 297 verified prefix receipts exactly and executes the 243 original
remaining global indices, including a fresh job 297. All **540 unique fold jobs**
pass before the deterministic count choice. All **36 mandatory full-data fits**
then pass before the model freeze and development assessment. There is no
old-model fallback or selection between original and refitted models.

The completed primary seed-47 policy has median relative error gain
**+0.207496% against XGBoost (7/12 wins)** and **-0.821725% against LightGBM
(5/12 wins)**. The combined primary gate therefore **fails**. Conditional seed
677, its mean-seed gate and the official outer assessment are **not run** under
the original gate contract. The original confirmation role remains sealed.
This closes the authorized accuracy experiment without establishing superiority
against both comparators, an Elo gain or an unbiased ranking.

Half/base/double count selections are CTBoost **2/4/6**, XGBoost **4/7/1** and
LightGBM **3/5/4**. The [closed summary](accuracy_wave18b_recovery/summary.json)
retains all 36 requested/native counts, controls and per-task errors. The
[terminal integrity audit](validation/wave18b_continuation_20261022/primary_policy_audit.json)
verifies all fold/union identities, saved model and prediction hashes, source
bindings, preserved original attempts and reported aggregates. Its inspection
uses recorded errors and byte hashes, not new scoring or model selection.

The recovery adds **243 fold fits and 36 union fits**. Cumulative work is
**3,107 benchmark model fits = 2,472 through 17b + 58 in the headroom-failed
attempt + 298 in the externally interrupted attempt + 243 recovery fold fits
+ 36 union fits**. The 297 copied successes are not counted again. The one
interrupted native fit is counted once as additional compute; eight earlier
diagnostic replay fits remain separate. The maximum new recovery worker cost
is 96.250 s and 3,698,835,456 bytes summed process-tree RSS, with no new resource
failures. New fold and union worker durations sum to 2,453.826 s and 354.111 s;
these are diagnostic costs, not isolated training-speed comparisons.

The accuracy runtime remains the immutable c5 version used throughout this
study. Fresh format-5 CTR arithmetic is not evaluated by these frozen quality
models. Under the prospectively frozen [v4 timing policy](inference/final_runner_preparation/selection_policy_v4.json),
the complete 18b recovery primary policy is selected for the final serving-cost
diagnostic because it completed, regardless of its failed quality gate. No
17b fallback is used. Timing verification and results are separate evidence.

## Latest complete task errors and controls

These are primary seed-47 errors of the complete wave-18b recovery policy. There is no
choice of a faster or better seed. Selected configuration, requested-count,
correction and source details are referenced by each dataset/library entry in
`accuracy_wave18b_recovery/frozen_selections.json`; original source-wave records remain
nested for provenance. No new default or universally superior policy is claimed.

| Dataset | Error metric | CTBoost | XGBoost | LightGBM |
|---|---|---:|---:|---:|
| blood-transfusion-service-center | 1-AUC | 0.410638298 | 0.360283688 | 0.332624113 |
| churn | 1-AUC | 0.110596781 | 0.107338698 | 0.109035987 |
| in_vehicle_coupon_recommendation | 1-AUC | 0.179708698 | 0.177468545 | 0.184116977 |
| GiveMeSomeCredit | 1-AUC | 0.133218982 | 0.133688638 | 0.141540064 |
| QSAR_fish_toxicity | RMSE | 0.714212343 | 0.728882176 | 0.750300817 |
| Another-Dataset-on-used-Fiat-500 | RMSE | 740.941613270 | 763.017867725 | 719.360844408 |
| houses | RMSE | 0.215229686 | 0.215366845 | 0.210813032 |
| diamonds | RMSE | 561.521651657 | 558.674001767 | 560.333733860 |
| anneal | Logloss | 0.001231490 | 0.005287589 | 0.000364800 |
| website_phishing | Logloss | 0.145859618 | 0.186879249 | 0.194622971 |
| splice | Logloss | 0.133956690 | 0.133609184 | 0.120646738 |
| SDSS17 | Logloss | 0.087979075 | 0.108846652 | 0.099380412 |

All prior per-task errors, costs, candidate controls and failures remain in each
wave's immutable `development.json`, `plan.json` and fit/selection receipts.
Near-zero denominators must not hide absolute changes: for example wave-3
anneal CTBoost loss 0.007790 versus LightGBM 0.001038 differs by about 0.006752
loss units even though the relative regression is 650.7%. The latest anneal
absolute losses appear in the full table above. Comparator selection variance
also moves aggregates; not every apparent gain is a CTBoost algorithm gain.

## Root stopping diagnosis

Saved wave-3 topology contains many root-only trees (QSAR 163/192, houses
244/318, diamonds 197/256, website phishing 384/502). Topology alone cannot
identify why: p-value rejection, feature eligibility, sample/Hessian limits,
cut gain, regularization, constraints or structural limits may all prevent a
split. Training-row diagnostics that use inference category transforms are
labeled accordingly and never used to fit leaf corrections.

A later exact replay of eight frozen wave-9 CTBoost binary/regression models
logs all 580 root-only trees among 1,061 trees as p > alpha rejections, without
an ambiguous alternative stop reason. This identifies the proximate cause for
those models only; it does not retroactively explain every wave-3 stump or
prove that weaker thresholds improve generalization. Those eight fits are
separate diagnostics and read no DEV/test values. See `root_stopping_replay/`
and the independent source audit in `review.md`.

## Gates, provenance and reproduction

The initial gate requires both comparators simultaneously. A passing policy
would then run its frozen learner-seed robustness procedure and average seed
errors within each of twelve datasets. This is same-split seed sensitivity,
not independent replication or significance. Versioned policies predeclare
947 for early stages, 137 for wave 13, 239 for wave 14, 349 for wave 15,
457 for wave 16, 569 for waves 17/17b, and 677 for the wave-18/18b protocol.
None of those conditional checks is run because the required initial gate has
never passed. Wave 11's mandatory seed-947 component is the explicitly separate
ensemble hypothesis described above. Its second pair is also not run.

Only a passing mean-seed gate permits a one-shot assessment of the original
uncapped official test. Existing trained models are frozen (72 single-model
payloads for two seeds, or 144 components/72 ensembles for the two-pair ensemble
policy), with runtime/code/split/model hashes, before any test scoring. No later
fitting or model adjustment is allowed. The original confirmation role remains
sealed. Metadata-only preflight checked all twelve split identities and
109,654 test-row indices, without scoring test features/labels.

Plans freeze executable sources and dependency identities. Use the recorded
Python 3.12 executable, isolated CTBoost package and task-local LightGBM target;
never overwrite an old output directory. Actual grids and selection policies
are the immutable plans, with independent reviews in `review.md` and primary
research in `research.md`. Large trusted local payloads/predictions are under
`.tmp/improvements-20261016/followup/`; source-controlled receipts retain their
hashes. Exact paths, CLI arguments and versions are in every plan/runner.

Main runners are `fair_accuracy_scout.py` (waves 1/2),
`fair_accuracy_categorical_scout.py` (3), `fair_accuracy_fullsample_scout.py` (4),
`fair_accuracy_alpha_scout.py` (5), `fair_accuracy_leaf_refit_v2.py` (6),
`fair_accuracy_temperature.py` (7), `fair_accuracy_power_scout.py` (8),
`fair_accuracy_union_refit.py` (9), `fair_accuracy_binary_refit.py` (10),
`fair_accuracy_seed_ensemble.py` (11), `fair_accuracy_threshold_scout.py` (12),
`fair_accuracy_round_cv.py` (13), `fair_accuracy_cap_expansion.py` (14),
`fair_accuracy_houses_cap_v2.py` (15), and
`fair_accuracy_multiclass_refit.py` (16), and
`fair_accuracy_scalar_refresh.py` (17), and
`fair_accuracy_scalar_numerical_retry.py` (17b), and
`fair_accuracy_current_round_cv.py` (18/18b). These are under `benchmarks/`.
The dated recovery [controller](validation/wave18b_external_recovery/controller.py)
replays only the original unverified fold suffix after the documented external
interruption; the unchanged main runner then selects, refits and scores.

Ordinary waves use prepare/run/development stages. Wave 10 uses
prepare/select/transfer/development. Wave 12 uses prepare/search/refit/development.
Wave 13 uses prepare/run/refit/development. Wave 15 uses separate
prepare/acquire/materialize/run/development stages; its preserved header-failed
attempt precedes the v2 retry, and `fair_accuracy_houses_report_recovery.py`
performs the report-only recovery after completed fits. Dedicated seed and
outer companions remain bound by each source-frozen versioned protocol.
Wave 16 uses prepare/select/transfer/development; its twelve controls freeze
before any transfer, and all 36 policy artifacts freeze before development.
Wave 17 uses prepare/select/refit/development; selection failed, so refit and
development were never entered. Wave 18b recovery completes run/select/refit/
development with the original scientific plan; its failed primary gate closes
all later quality stages.

This editable narrative was reconstructed from immutable receipts after a
Windows text-encoding write failure. Experimental sources, plans, predictions,
models and numeric results were unaffected; subsequent report writes use UTF-8
and atomic replacement.

Direct evidence: [latest full scores](accuracy_wave18b_recovery/development.json), [latest summary](accuracy_wave18b_recovery/summary.json), [latest terminal audit](validation/wave18b_continuation_20261022/primary_policy_audit.json), [17b scores](accuracy_wave17b/development.json), [wave-16 frozen controls](accuracy_wave16/multiclass_controls.json), [all36 CV counts](accuracy_wave13/summary.json), [fold identities](accuracy_wave13/fold_manifest.json), [preserved header failure](accuracy_wave15/materialization_failed.json), [bit-exact report recovery](accuracy_wave15b/report_recovery_completed.json), [original numerical report audit](validation/accuracy_report_review.json), [wave-16 report update audit](validation/accuracy_report_wave16_review.json),
[wave-17 terminal summary](accuracy_wave17/summary.json), and
[wave-17 report update audit](validation/accuracy_report_wave17_review.json).
