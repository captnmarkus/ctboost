# Independent protocol review

Reviewed 16 October 2026. This review reads source and small protocol artifacts;
it does not execute training, model-load timing, or native builds. Findings were
sent to the relevant implementation owner before measurement/fitting.

## Serialization load scout

Reviewed `benchmarks/serialization_load_scout.py` and
`serialization/{protocol.json,plan.json,preparation_note.json}`.

The protocol compares three formats of three fixed, previously assessed models.
Compact JSON preserves the parsed native model document. Pickle preparation now
uses the public `Booster.save_model(..., model_format="pickle")` API; the
preparation correction is explicitly recorded as preceding all measurements.
No model is refitted or selected using this experiment.

The worker checks artifact and input hashes, loads each artifact in a fresh
process, separates load from first prediction, and checks raw score bits against
the saved native reference. Runtime imports, hash reads and input reads are
outside the timed region. Hashing deliberately warms the filesystem cache. The
isolated package path and native binary hash are checked. These are appropriate
controls for a bounded format/load experiment.

Pre-measurement recommendations sent to the owner:

1. Balance format order by position. The original three orders were
   A-B-C / C-B-A / B-A-C: compact JSON occupied position two twice, and pickle
   position three twice. A-B-C / B-C-A / C-A-B is a balanced three-order design.
2. Freeze the isolated Python package source hashes and NumPy/psutil runtime
   versions as well as the native binary. The Python loader is part of the
   measured implementation.
3. Call `peak_working_set_bytes` the process peak through first prediction. It
   includes imports, hashing, data preparation, loading and prediction; it is
   not a load-only peak. Similarly, `rss_loaded - rss_before` includes retained
   allocator/interpreter memory and is not an exact native model allocation.
4. Three repeats per artifact are a scout. Report individual values and medians;
   do not imply a precise confidence interval or cold-storage latency. There is
   no warm steady-state prediction measurement in this runner.

The proposed production change is two writer calls in
`ctboost/_serialization.py`: remove `indent=2`, add `separators=(",", ":")`,
retain `sort_keys=True`. This leaves fields, schema versions, key encoding,
float encoding, tree order and JSON readers unchanged. Do not combine it with
omission of the currently mandatory 256-element categorical route vectors.
The predictor JSON writer is a different surface and was not included in this
recommendation. A load-time improvement does not establish faster warm traversal.

## Accuracy portfolio scout

Reviewed `benchmarks/fair_accuracy_scout.py` before the first run. The portfolio
contains eight configurations per library/task and a predeclared second wave.
The statistical feature-before-cut procedure remains intact: all CTBoost arms
use existing options. Four first-wave arms provide a clean learning-rate by
leaf-refinement ablation; the other arms test combined configurations.

The selection structure is appropriate: training-only category vocabularies;
distinct cached train/stopping/development files; no official-test or original
failed-gate confirmation loading path; per-configuration stopping evaluation;
all library/task winners frozen before development predictions. XGBoost is
sliced to `best_iteration + 1`; LightGBM is serialized with its selected prefix.
Native early-stopping metric names are AUC/RMSE/multiclass log loss for the
corresponding primary errors. Source and role-file hashes are checked.

Findings sent before freeze:

| Finding | Required handling |
| --- | --- |
| The initial `fit_job` assigned `best` and status `ok` before pickle save/reload and prediction equality checks. Its exception handler then added an error without resetting status. | Mark exceptions failed; validate a temporary candidate artifact before committing the winner. Otherwise a failed serialization/parity check can appear successful. |
| CTBoost's recorded trained-round count used `len(evals_result_.get("validation_0", {}))`, which counts metric keys under the wrong role. | Use the actual evaluation-history length. |
| LightGBM's returned Booster may already be reduced to the best prefix. | Record evaluation history to count actual evaluated rounds; distinguish trained-cap hits from retained-prefix-at-cap. |
| The parent frozen-selection hash was saved but not rechecked before wave-two selection. | Recheck it before reuse. |
| The initial cumulative success gate counted only wave-two fits. | Preserve prior failed-fit status when deciding whether all cumulative fits succeeded. |
| Runtime metadata did not initially assert the native module path, and dependency version metadata can differ from the imported module. | Assert native package containment and record actual imported module paths/versions alongside native hashes. |

Numerical metric caution: binary AUC computed from raw margins is mathematically
equivalent to AUC from probabilities, but floating-point sigmoid saturation can
create ties. CTBoost's native AUC ranks raw margins; XGBoost/LightGBM built-in
metrics can consume transformed probabilities. Multiclass stable log-sum-exp
and native probability clipping can also differ. Record the native selected
stopping metric and independent rescoring, with an explicit tolerance; do not
claim numerical identity solely because the metric names match.

The historical role preparers already encoded classification labels with
`LabelEncoder` and checked class counts. The original 14-task role indices are
positions in the cached outer-training data; the separate 37-task role indices
are source-dataset indices. Disjointness should be checked within each task's
coordinate system, not compared across studies.

Equal trial counts and round caps do not make wall time, realized capacity,
categorical representation, or search-space coverage equal. The vector XGBoost
arms address an important competitor omission, but a vector-versus-scalar
comparison that also changes L2 is not a clean structural ablation. This is a
historically reused, one-seed development panel; task/family-level descriptive
results are appropriate, and leaderboard or independent-confirmation claims
are not. The planned second-wave decision is based on development outcomes and
must remain labeled adaptive development even though its grid was predeclared.

Implementation revisions and final artifact receipts should be read together
with this preflight record. A reported finding here does not imply it remains
unfixed in the final runner.

### Revised accuracy preflight

The owner withdrew the prepare-only `accuracy_wave1` plan and froze
`accuracy_wave1_v2` before any actual fits. Source reinspection confirms:

- Candidate save/reload parity occurs before atomic replacement and winner
  commitment; exceptions explicitly set failed status.
- LightGBM evaluated rounds come from `record_evaluation`; CTBoost's physical
  trained-round count is explicitly unavailable because its history is pruned,
  rather than recording an incorrect zero. The cap flag is named
  `retained_at_round_cap`.
- Prior frozen selections are hash-checked, and the all-fit-success gate carries
  prior-wave failure status forward.
- The native CTBoost path is asserted inside the isolated package, and actual
  CTBoost/XGBoost/LightGBM import paths and versions are recorded.

The frozen script hash matched the current runner during this recheck. The plan
contains 12 tasks, 1,000-round caps and patience 50; its recorded actual imports
are XGBoost 3.4.1, LightGBM 4.7.0, and the CPU CTBoost 0.1.61 MSVC-1950 build.
No remaining launch-blocking logic issue was found. The numerical metric caveat
above remains a disclosed limitation rather than a claim of exact native versus
external metric equivalence.

## Proposed request-local category identity cache

This is design review only; no cache was implemented or measured here. A lazy
cache keyed by object address can reuse normalization of repeated exact immutable
objects within one transform. Strong object references prevent address reuse if
a custom conversion mutates later matrix cells. Use pointer hashing rather than
Python hashing/equality, and keep subclasses, NumPy scalars, arrays and all
custom objects on their original fallback. Preserve call order: eagerly
normalizing a column could change the observable behavior of custom conversions.

Exact integers need care: converting a very large integer to decimal text can
raise according to the interpreter-wide `sys.set_int_max_str_digits` setting.
A custom object's conversion can change that setting during the same transform,
so an earlier cached success would not necessarily preserve a later exception.
Start with exact strings, or exclude large integers from caching. The minimum
configurable digit threshold is exposed as
`sys.int_info.str_digits_check_threshold`.
[Python integer conversion documentation](https://docs.python.org/3/builtins/stdtypes.html#integer-string-conversion-length-limitation)

Bound total cached string bytes as well as entry count, and bypass oversized
entries without changing their output. Keep the encoding version fixed per
cache. For `JoinNormalizedKey`, cache the normalized component but retain the
existing additional escaping of backslash and pipe, separators and legacy
behavior. Cache only successful normalizations.

After the exact-builtin `IsMissing` improvement, string conversion may already
be cheap enough that an extra map lookup loses on small batches or high-cardinality
data. A bounded cache should therefore be evaluated on both high- and
low-cardinality input, including one-row requests, before promotion. A direct
pandas categorical-code path is a larger API/schema undertaking and should not
be mixed into this narrow candidate.

## Wave-three accuracy extension review

Read-only review of `benchmarks/fair_accuracy_categorical_scout.py`, frozen plan
SHA-256 `322e99168bcd1f855ec372d071fe3235c494b08bccac851de00a2b6d21a6db61`.
No new launch blocker was found in the diff from the previously reviewed runner.
Train/stop/development roles, train-only categorical vocabulary, native metric
choices and prefix selection, transactional model persistence and parent selection
hash checks are retained. CTBoost's implicit CTR generation is actually disabled
when `ordered_ctr=False` here because no explicit CTR lists are supplied.

CTBoost profiles 0/1 isolate CTR on/off, 0/2 isolate one-hot representation,
and 4/5 and 6/7 isolate joint multiclass testing (or grouped-8/grouped-16 for
the other families). Profiles 2/3 are not a clean CTR ablation: the categorical
threshold also changes from 64 to 256. Equal eight-candidate extension budgets
remain distinct from equal walltime or statistical capacity. The new grid was
frozen before wave-two development assessment, but was informed by wave-one
development; the resulting cumulative 24-candidate search remains exploratory.

## Compact traversal node review

Read-only review of the 24-byte `PredictionNode` union candidate in
`src/core/booster_predict.cpp`: no valid-model correctness blocker found.
Default construction activates the scalar leaf update; internal-node construction
assigns its trivial `Split` member before use. The existing nonnegative-feature-id
discriminator selects the split arm, while leaves retain scalar updates and vector
payload offsets. The vector copy operations are appropriate for this trivial
union. Root-only handling, multiclass topology fusion, partial prediction prefixes
and leaf-index traversal retain their previous control flow and original node
indices. This layout is private to the prediction cache; training and serialized
node formats are untouched. No independent timing or tests were run for this
read-only review; the owning agent performed its regression and paired benchmarks.

## Fixed-configuration second-seed review

Read-only review of `benchmarks/fair_accuracy_seed_check.py` and
`robustness_protocol.json` found no blocking issue. The runner recovers each
selected configuration and round cap from its original source wave, changes only
the learner seed to 947, keeps early stopping on the original stopping role, and
freezes round-tripped models before development scoring. It checks runtime,
implementation, policy and parent evidence hashes. Aggregation averages the two
primary errors within each dataset before comparator gains and the 12-task median;
seeds are not counted as independent tasks. An optional explicit assertion of
36 unique dataset/library jobs and parent seed 47 would document assumptions
currently guaranteed by the inherited frozen plans. Two seeds on one reused split
measure sensitivity, not independent replication or statistical significance.
No official-test loader or data was accessed in this review.

## Official outer-test assessment runner review

Read-only source review of `benchmarks/fair_accuracy_outer_test.py` found a sound
separation of metadata-only preparation and gated one-shot assessment. Preparation
checks original split identities and freezes 72 model hashes. Assessment writes
its attempt marker before opening the data, validates cached train/stop/development
row mappings and label encoding, and uses only frozen models on uncapped original
outer-test rows. It does not fit or select models; successes and failures are
preserved. The older length-prefixed training-index receipt is handled explicitly.

One missing preflight invariant was sent to the owner: verify the parent
development document's `plan_sha256` against its plan, in addition to its already
checked frozen-selection hash. An explicit unique-12-case assertion was also
recommended (72 unique model keys are already required). No official-test
features, labels or metrics were read as part of this source review.

## Wave-five alpha and possible refit policy

Read-only comparison of `fair_accuracy_alpha_scout.py` with the wave-four runner
found no new launch blocker. Four matched pairs isolate alpha 0.05 versus 0.5 at
fixed full sampling, while depth and L2 vary across pairs. Alpha one remains
excluded so significance stopping is not made vacuous. All three libraries gain
eight candidates; stopping-only cumulative selection and frozen parent hash
checks remain intact. This remains adaptive development research.

A separately frozen TRAIN+STOP refit is a reasonable next policy if needed: one
mandatory refit per selected library/task, with controls, selected iteration
count and seed fixed before development scoring. Refit common vocabulary on the
union and regenerate CTBoost's ordered training CTRs internally; never concatenate
previously prepared training/stopping transforms. Save all models before scoring,
retain original winners, and do not choose original versus refit using development.
Apply the same rule to every library.

This follows the standard distinction between selecting controls and refitting
the selected estimator on the available training partition.
[Scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
More rows can change statistical power, category statistics, leaf estimates and
regularization balance. Compute the increase from actual role sizes; no causal
test-power benefit is established by a development gain. Seed averaging adds
model size and inference work and needs its own equal-budget ensemble policy.

## Compact active-bin scratch review

Read-only review found no valid-model blocker. Activity masks retain original
schema indices, cached nodes receive compact slots after topology fusion, and
full histogram callers retain full-width metadata. Prefix requests skip inactive
later features, including forbidden NaNs; root-only caches need no bins. GPU/FMA
fallbacks and original leaf IDs retain existing paths.

The owner fixed one reported defensive issue: native loaders do not reject every
out-of-schema node feature ID, so remapping must range-check before indexing its
slot vector. A malformed later tree must not break a valid prefix request. The
owner added that bounded regression case; this does not redefine behavior when
an invalid tree is actually traversed.

## Fixed-topology leaf correction review

Read-only review of `fixed_leaf_refit.py`, `fair_accuracy_leaf_refit.py` and the
predeclaration found no launch blocker. The helper reconstructs the original
ordered training transform and requires fitted-state equality before computing
training routes and residuals. It hashes the full native state after excluding
only scalar leaf outputs, compensates CTBoost's per-tree learning rates, and
keeps original models intact. Regression-only scope and inference-only artifacts
are explicit; historical training metadata is not claimed to have been recomputed.

The runner accepts converged LSQR statuses 0/1/2/4/5 and preserves failures. It
checks unchanged TRAIN/STOP leaf IDs, selects on actual folded native STOP RMSE,
and requires the predeclared rtol 5e-5 / atol 1e-5 correction oracle on both roles.
The helper's separate conservative floating-point bound is a diagnostic rather
than a replacement for that strict runner check. Exact ties prefer the unchanged
incumbent, then larger ridge penalty, then smaller shrinkage. Selected native
payloads are round-tripped before commitment.

All 12 regression library/task jobs must produce selection files; the eight
classification tasks remain unchanged and all 36 models feed the existing
12-dataset assessment. Missing jobs cannot silently become successful incumbent
carry-forwards. Parent development-plan identity and unique dataset/library keys
were recommended as additional preflight assertions. No real correction solve,
new labels, or official-test data was accessed during this review.

The subsequent `fair_accuracy_leaf_refit_v2.py` diff was also reviewed. It converts
NumPy scalar diagnostics to Python scalar values so JSON receipts can be written,
including the diagnostic metadata stored beside a model. The original failed
receipt-writing attempt is preserved. No leaf coefficients, native model arrays,
grid, routing, STOP metric or strict independent oracle tolerance changed. The
second-seed runner imports this corrected receipt path and freezes the actual
SciPy version whenever its selected policy includes a correction solve.

## Numeric grouped-test resolution follow-up

The proposed eight binary/regression candidates, groups 2/4/16/32 crossed with
alpha 0.2/0.5 at fixed full sampling, depth 8, minimum leaf 5, L2 1 and learning
rate 0.05, have no feature-before-cut design blocker. In `statistics.cpp`,
grouping pools occupied ordered bins using their cumulative-weight midpoints;
the achieved number of groups can be smaller than requested. Missing values
retain a separate occupied group. The test uses the resulting degrees of
freedom, while cut search retains the original histogram. Native categorical
histograms bypass this grouping in `tree_candidates.cpp`; numeric CTR and
one-hot outputs can still be affected.

This is a concrete power/resolution hypothesis: fewer groups may help broad,
weak effects, but can cancel localized or nonmonotone effects. More groups can
retain those effects while increasing degrees of freedom and small-node
sparsity. Prior coverage was not exclusively eight groups: wave 3 already
paired 8 versus 16 for binary/regression candidates 4/5 and 6/7, under different
sampling and regularization. Groups 2, 4 and 32 are the new range here.

The historical grouped calibration study used eight groups at alpha 0.05 and
found 1.20% null rejection with 24 observations, 3.40% with 96 and 4.50% with
320. Those fixed-node observations motivate checking power; they establish
neither calibration at the proposed thresholds nor adaptive-tree error control.
The older 0.1.55 grouped-eight release decision failed its fit-time gate, so it
is not evidence that all grouped resolutions harm accuracy. No new null
simulation or prediction experiment was run for this review.

Carry all four calibrated multiclass policies unchanged, use STOP-only
selection with the unchanged incumbent winning exact ties, and audit repeated
controls against previous waves. Equal eight L1/feature-sampling comparator
fits per library provide equal trial counts, not equal capacity or compute.
These adaptive development waves cannot establish an external ranking; the
later frozen second-seed and sealed-test stages remain necessary. A mandatory
TRAIN+STOP selected-iteration refit for every library is still a distinct,
reasonable future policy, but the existing grouped-resolution control is the
smaller next hypothesis and requires no production algorithm change.

## Mandatory union-refit and binary-logistic follow-up review

The separate union runner correctly concatenates original TRAIN then STOP,
checks disjoint row identities, fits a fresh common vocabulary on that union,
and refits each library at its original selected parameters and requested round
count. There is no evaluation set, early stopping or development-driven choice
between the old and refitted models. Fixed regression correction controls are
recomputed on the replay-verified ordered union transform; fixed multiclass
temperatures are retained. Native correction oracles and exact pickle prediction
parity remain required before freezing all 36 models. Seed 947 reuses the original
requested rounds, including when LightGBM structurally stopped early at seed 47.
The owner added the recommended unique 36-model identity check and explicit
union seed-protocol/source check before official-test preparation. No new quality
labels or official-test records were opened during this source review.

The new binary helper uses a positive-ridge logistic objective with fixed old
margin offsets and the same physical-tree leaf-indicator basis. Its analytic
gradient is correct; positive ridge resolves the redundant-column coefficient
ambiguity. Success status, finite coefficients/objective, non-increasing objective
and the predeclared maximum mean-gradient threshold jointly gate convergence.
Ordered CTR replay, contribution-unit leaf folding, unchanged non-leaf hashes,
exact routes and both analytical and fixed-tolerance prediction oracles preserve
the conditional feature-before-cut design. XGBoost base probabilities are mapped
to logit magnitude for the rounding bound, including scalar/vector JSON forms.

Two pre-freeze hardening suggestions were sent to the owner: use the signed-margin
`logaddexp` loss to avoid cancellation for confidently correct rows, and reject
non-unit LightGBM binary sigmoid scaling. The real frozen models use the default
scale. The new study must retain the earlier no-weight, no-constraint, no-external-
baseline eligibility and explicit inference-only artifact semantics.

Independent STOP selection must start with the original TRAIN-only trees and
TRAIN-only correction solve. Wave 9 union models have already consumed STOP
labels and cannot supply an independent selection metric. After selecting fixed
penalty/shrinkage controls on the former models, recomputing that fixed policy on
the mandatory union models is coherent. Exact STOP-AUC ties should retain zero
correction. Optimizing logistic loss while selecting by AUC is deliberate and
does not guarantee a ranking improvement; both objectives must remain explicit.

The owner implemented both binary hardening recommendations and added dedicated
regression tests before freezing the helper. The resulting helper hash is
`0ff18de1c16894b121209566e0f6edfc8643223420b2ab16be113ce26dcc0ad4`;
the owner reports 14 passing toy tests across all three libraries in the exact
CP312 experiment runtime. This source review does not substitute for the
separately frozen real-data protocol and its required convergence/oracle gates.

## Wave 10 independent source review, 2026-10-16

No remaining protocol blocker was found in the final binary-correction runner,
seed companion, official-test companion or versioned policy. The reviewed source
identities are:

| File | SHA256 |
| --- | --- |
| `fair_accuracy_binary_refit.py` | `472decf4eeb627e0b45c2ab3cb4e8bcabe66527d4536c5b9af12267f23a1bde7` |
| `fair_accuracy_binary_seed_check.py` | `9e543a84b7fcdd99232ad1c723e410f78bef99811418397a5e76614b6fd8619e` |
| `fair_accuracy_binary_outer_test.py` | `2c5e2a1c88c3645d3ae91ebf6989acf179254fcdf187855cd180b79cec62531c` |
| `robustness_protocol_binary.json` | `6ed5c63f650b83d6096169b0d17f4b126cc8d3ca2052909ba0e3fc3e291a5a80` |

Selection uses the original wave 8 TRAIN-only models, checks disjoint TRAIN/STOP
row identities, replays ordered CT training preprocessing, and verifies the
unchanged STOP metric before solving. Four penalties and two shrinkages produce
eight candidates per library/task, with identity winning exact ties. Every solve,
native fold, route/oracle check and pickle check must pass: a failed candidate
cannot silently disappear from the comparison. All 96 candidate evaluations and
12 unique binary choices must succeed before transfer.

The controls file is frozen and hashed before fixed recomputation on the wave 9
union models. The transfer stage performs no STOP selection; it carries the other
24 models unchanged and freezes all 36 before development scoring. The seed
companion correctly copies selected binary controls into the inherited fixed-round
jobs, including solver settings, while retaining regression corrections and
multiclass temperatures. It checks exact dependencies, runtime/SciPy identities,
parent model hashes and 36 unique model identities. Seed errors are averaged
within datasets before the 12-task comparison.

The official-test companion requires the exact binary seed protocol and source,
successful two-seed development gate, frozen 72-model identities, and official
cache/split identities. Its one-shot marker is written before test data access;
it does not fit, select or replace models. The original 14-task confirmation fold
remains sealed. The owner additionally reports four passing exact-runtime pipeline
tests covering all three libraries, fixed union rounds/correction at seed 947,
ordered CT replay, late categorical vocabulary, pickle parity and overlap rejection.

This is source review, not a new real-data experiment or an accuracy guarantee.
Logistic training loss and STOP AUC intentionally differ. Equal grids do not equal
compute or effective function regularization; transferring a fixed penalty from
TRAIN to the larger union can be suboptimal. Repeated use of development across
the earlier waves remains adaptive, and a 12-task, two-seed panel cannot establish
a universal or official benchmark ranking.

## Fixed two-seed ensemble review, 2026-10-16

No remaining source-level blocker was found in `fair_accuracy_seed_ensemble.py`,
its official-test companion, or `robustness_protocol_ensemble.json`. The main
runner reviewed at SHA256
`f47284a17c0aaf5ce9f0ccf71c6bd8c42ffb4bea94fc0c302d3b9601cef4eefd`
frames 47/947 as a new mandatory hypothesis following the failed wave 10 gate,
not as successful confirmation of that gate. It reuses fixed tree rounds,
parameters, correction controls and temperatures. Every task/library receives
the same two components; no seed choice or weighting is learned from STOP/DEV.
The 137/1543 pair is fixed in advance and runs only after the first pair passes.
Pair errors are averaged within each of the same 12 tasks.

The native component identity keysets, shared encoded labels and exact ordered
categorical schemas are checked. All components and 36 ensemble payloads freeze
before each development stage. Official preparation checks each pair's component
seeds and task/library identities, 144 unique native components and 72 unique
ensemble payloads. The later wording correction correctly calls these two seed
pairs, not two individual seeds. No official test is opened by preparation.

The helper at SHA256
`c28e5b9a8623ac07bc85670dbeeafbfe435d5c067b16927d7a866d8473ad2f83`
uses stable binary log-odds of the probability mean and multiclass log-probability
means after applying each fixed temperature exactly once. Regression predictions
are averaged directly. The one-time native LightGBM objective check handles
model-string reloads with empty Python parameters; the frozen link receipt avoids
dumping models during prediction. Overflowing multiclass log-softmax shifts are
explicitly rejected. XGBoost inplace prediction remains an unpersisted latency
option requiring exact per-batch component and ensemble byte parity. The owner
reports ten exact-runtime integration tests covering all three libraries and
problem families, both components, corrections, temperature, scoring and pickle.

Model provenance and class ordering depend on the runner's frozen-file and
common-label contracts; the helper alone is not a general arbitrary-model
ensemble validator. Averaging has no guarantee against the better component or
against competing libraries, especially for AUC. Two seed pairs do not create
independent datasets. Serving needs two native models and its extra latency and
storage must be reported.

## Root-only attribution correction and verified replay

The wave 3 saved-topology counts do not identify stopping causes. Source audit
found several leaf-return branches, so the earlier phrase "substantial statistical
stopping" was too strong for that evidence alone. The separately predeclared
eight-model wave 9 replay now verifies 580 of 1,061 roots as feature-test
rejections, with exact final native state except logging metadata, pipeline state,
prediction bytes and leaf IDs. None of these root-only cases was a post-test
cut failure or an ambiguous rounded boundary. This conclusion is specific to
the replayed wave 9 union models, not a retroactive classification of wave 3.
See the [cause audit, full quantiles and source limits](root_stopping_replay/README.md).

## Wave 12 independent source review, 2026-10-16

No remaining source-level blocker was found after the prefreeze fixes. The main
runner hashes both new seed/official-test companions before any fit, checks the
exact 24 active TRAIN-selection identities and freezes exactly 36 final jobs and
models. Reviewed identities:

| File | SHA256 |
| --- | --- |
| `fair_accuracy_threshold_scout.py` | `3ce067316e7df229fd3535eece96828f8d05eb889b605a5bf2b96cba24719c91` |
| `fair_accuracy_threshold_seed_check.py` | `e4f8f28e0202159f594192a9b14bb7afd4a971288cc80879cee37c32e01775b6` |
| `fair_accuracy_threshold_outer_test.py` | `d2ee6bbabcda553898830ae2321d15e330029fd8c177bb791790c39121052cb8` |
| `robustness_protocol_threshold.json` | `480bfea189050627010ad0806e1f573c819c16b14dd4219fff651d283609bcb4` |

The four profiles for each binary/regression task/library compare two declared
thresholds at original and half learning rate. Half-rate profiles receive twice
the round cap and patience in every library. These are 96 fresh fits, not 96
unique historical configurations: base comparator profiles and some CT profiles
are explicit repeats, exposed by the canonical-default duplicate audit.

Native early stopping first chooses an uncorrected prefix. The frozen binary or
regression correction is then re-solved on TRAIN alone, and the corrected native
STOP error chooses the profile. The saved regression incumbent is already
corrected and is not corrected twice; the binary incumbent's frozen correction
is recomputed and its prior STOP error checked. Exact STOP ties retain the
incumbent, then new ties prefer fewer rounds and configuration index. This does
not optimize the corrected prefix itself, and the report correctly says so.

All 24 choices freeze before mandatory TRAIN+STOP refits. The 12 multiclass
single-model policies are carried by the declared policy, without selecting
between singles and failed ensembles on individual development scores. Seed947
jobs use the final selected parameters and rounds from `frozen_selections.jobs`,
not the old original jobs. The exact companion source/protocol, model hashes,
two-seed within-task aggregation and one-shot official assessment are gated.

This review ran no real fits or new quality assessments. Equal profile counts
and learning-rate horizons do not imply equal runtime or functional complexity.
Alpha0.8 retains the strict feature-test-before-cut path and a nonvacuous gate,
but does not establish ensemble-wide error control. All results remain an
adaptive, capped, twelve-task development study pending separate confirmation.

## Wave 13 independent source review, 2026-10-16

The final five-fold round-selection implementation has no remaining identified
modeling or selection blocker. Its frozen plan is
`2312ca44784516fa19c5419a3c51715ee5887a5eec47f6979b7da8d75af4c198`.
The reviewed source identities are:

| File | SHA256 |
| --- | --- |
| `fair_accuracy_round_cv.py` | `fa905e7d0de917aad5a8fad71730873f11c36cb0e86a0e01a1da6900f296e521` |
| `fair_accuracy_round_cv_seed_check.py` | `136a65b03fd7f37c006e236d8f054b17628158ec58a2bad72a0895a3c30da226` |
| `fair_accuracy_round_cv_outer_test.py` | `fbae244bd3d8a8bd0466f4e8710cdc5aadda85471cd2fe1763237bb8c5e86ed6` |
| `owned_process_watchdog.py` | `9d95c12c9954a9624b99fadf9da33a634be07ef048ea352ece244f8a06395516` |
| `robustness_protocol_round_cv.json` | `3934f6010a0d85f4ece5f19677261b88bee0a8d617809619c4b13de31f455b88` |

The main runner concatenates raw TRAIN and STOP before splitting, freezes one
five-fold assignment/schema manifest per task, and fits preprocessing from
observed fold-training values only. It checks all classes in both sides of every
classification fold. Requested wave 12 round counts anchor independently fitted
ceil-half, original and double profiles; one duplicate profile index is retained
explicitly. There is no model slicing, public-CV history aggregation or native
early stopping. LightGBM structural termination is recorded separately.

Every fold/profile independently recomputes the inherited correction on fitting
rows, using ordered CT replay, then applies the inherited temperature before
held-fold scoring. The arithmetic mean of five primary errors selects a profile;
exact ties choose fewer requested rounds and then lower profile index. All 540
fits must pass before 36 choices freeze, and all 36 mandatory union refits must
pass before development. There is no old-model or original-round fallback chosen
on development. Fold models are serialized, hashed and prediction-roundtripped
without retaining their large bytes; the prediction arrays and audit receipts
remain available.

The seed137 companion carries the chosen requested rounds and all fixed controls,
checks the actual fold-manifest hash, and runs only after the initial gate. Its
aggregation explicitly uses seeds47/137, not the older947 helper convention.
Errors are averaged within the same twelve datasets. The official companion
requires that exact successful seed protocol, freezes 72 model identities, and
writes its one-shot marker before reading official-test values.

Resource review found and corrected Windows launcher-only accounting, then a
denied-inspection cleanup path. The watchdog sums observed owned-process RSS,
records PID/creation-time identities, kills remembered descendants as well as the
launcher, and fails closed if cleanup cannot be confirmed. A shared dispatch
lock and abort checks prevent further jobs after a cleanup failure. The owner
reports 16 exact-runtime tests passing: nine library/problem fold pipelines,
two rare-class fold tests and five process tests covering child memory, timeout,
unrelated-process survival, vanished processes and denied-memory inspection.
This is an observed-process watchdog, not a general OS job-object containment
facility; shared RSS pages may be counted more than once.

The statistical limitation remains explicit: inherited controls, temperatures
and anchors already used old STOP labels. These folds select round length for
that historically chosen policy and are not an unbiased nested-CV assessment.
No review-time fits, new development scoring or official-test access were done.

## Final single-model timing runner review

`fair_inference_final_scout.py` was reviewed at SHA256
`27a5ea2fa73f8c73d54c68f5f86323a430777e27d1e71e3315099e18b2a241af`.
It requires an immutable check-only reference binding code, runtime, selected
model files, source rows and requested batches. Both full-source and every timed
batch require byte-exact raw/normalized/reference predictions; XGBoost inplace
margins must match DMatrix margins on that exact batch. Raw timing includes one
common vocabulary normalization, one full model call and temperature once.
Normalized-frame timing excludes only the common normalization.

The reported quality belongs to the actually loaded learner seed, while any
two-seed mean is separate metadata. Missing seed errors and conflicting generic
or seed-specific prediction hashes are rejected. This supports the future137
companion's prediction receipts without substituting47 or947 results. Actual
native round counts are reported. Eight owner-reported synthetic tests pass;
this review launched no real timing or accuracy evaluation. Different selected
models can have different quality and capacity, so this diagnostic is not an
equal-quality or official ranking comparison.

## Final production inference audit continuation, 16 October

The inference owner's latest read-only audit reported no unresolved material
correctness issue in native categorical-buffer validation and gathering
(`src/core/feature_pipeline_transform.cpp:102–197`), the conservative public
guard (`ctboost/_categorical_codes.py:21–79`), the one-time old-native ABI
fallback (`ctboost/feature_pipeline.py:294–299`), or state rederivation on restore
(`ctboost/feature_pipeline.py:424–492`). The existing exact-bit, native and Python
interoperability evidence applies; this continuation adds no fresh tests or
timings. The row, width and dictionary eligibility thresholds were established
on the recorded hardware and workloads and are not universal crossover claims.
The root reviewer also visually checked the development figure through wave12
and ran `git diff --check`, which passed with only the repository's existing
CRLF warnings.

## Training-cap expansion review, 16 October

No unresolved material blocker was found in the canonical cap-expansion helper,
main runner, seed 239 companion or official-test companion. Reviewed SHA256s:

| File | SHA256 |
| --- | --- |
| `fair_accuracy_cap_data.py` | `3d2d6d2c1ba4da086a34c1b927f2f6c33a596494885f70a70e73c9dcf80b2844` |
| `fair_accuracy_cap_expansion.py` | `e5ee9835fab07ad51388b2cb548a634d258c6fbd5005412430185697a01e6a91` |
| `fair_accuracy_cap_seed_check.py` | `b200f2617c0c7cec60b1911ac3fc53cbf9870d009c0730b0b3756e1634ea0cb3` |
| `fair_accuracy_cap_outer_test.py` | `565d8ec2c1654acff2eb1e04bb58dd832c4e841eac25fa4c5d06e76ffe60a8ae` |
| `robustness_protocol_cap.json` | `53644683bfd04e2dc77e300e261925edf528c89577277e6696f5e96729bff2e6` |

The helper reconstructs the historical cap, verifies exact original TRAIN/STOP
frames and encoded labels, checks role counts/order/hashes, and derives the
sealed confirmation remainder by count and hash without opening
`confirmation.pkl`. Expanded rows are original TRAIN, original STOP, then sorted
outside-cap positions. The expanded/excluded partition is proved before
normalization or correction. Full official-TRAIN labels are read administratively
to reconstruct historical stratification and class IDs; this includes excluded
DEV/confirmation labels, which never enter new fitting or confirmation scoring.
Source trust comes from the frozen train-only cache bytes and prepared receipts;
the prior outer-training split hash is recorded provenance, not a new raw-data
extraction. Houses remains unchanged.

The main policy performs nine mandatory expanded fits and carries 27 exact models.
It retains requested rounds rather than substituting structurally shortened
LightGBM counts, re-solves fixed corrections on the expanded fitting rows and
applies inherited temperature once. Each eligible task uses one identical
expanded-data receipt/hash for all three libraries. SciPy, native runtime,
helpers, companions and policy are bound before fitting. The complete 36-model
policy is frozen before development scoring; failures cannot substitute old
models. Conditional seed 239 refits all 36 policies with the same rows and controls,
and both primary and seed fits use isolated per-fit process-tree watchdogs.
The companion averages 47/239 errors within tasks. Its passing gate precedes
the 72-model official-test freeze and one-shot assessment.

The owner reports 29 exact-runtime synthetic tests passing, including 18 actual toy
fits across three libraries, three problem families and seeds 47/239, native
correction/pickle checks and tampered-source rejection. This independent review
ran no fits and read no new development scores or real cache values. After
review, the owner reported successful authorized real preparation with expanded
counts 97,500/33,460/49,535 for Credit/diamonds/SDSS17 and plan SHA256
`8f99bc561554b042ea44d37652f6cc6f995c851d12dc077f0611382e80851011`.
This records implementation readiness and provenance, not an accuracy outcome.

## Authorized houses source review, 16 October

The separate wave 15 implementation has no unresolved material review blocker.
The reviewed main, parser, seed 349 companion, outer companion and policy hashes
are, respectively:

- `47237555e2f53dbf326925e201f10494b75926dc56ae6f43a4243a4a07c79b53`
- `05f3e9e7102bf1c301f102ae2b9a8b68900b2eb2f12c7bdce45624210ee4594c`
- `22be215ac373487eb05b750b4a1a9d439d73275a3860c7c5ffb323c24861e0b8`
- `c0f8515b4d0946e2fb965123047a4c6789280b9421998085ccbba8eab78782ee`
- `59f491b9f55865e883d484c4d413a4f3927b38d74f827b2f6fcc4427f4f2f76a`

The protocol freezes authorized global row IDs, split identities, the historical
seed 47 cap, the trusted source MD5 and executable sources before downloading.
Acquisition reads bytes only and persists their SHA256 receipt. A separate
materialization step checks that receipt before the narrow parser decodes the
header and authorized instances. Excluded TEST/DEV lines receive byte-level
structural checks but are never decoded, tokenized, delimiter-counted or converted
to numbers. Parser failures reveal only an error code and optional row ID.
This is a selected-row access guarantee, not numeric validation of excluded data.

Original TRAIN/STOP features and targets must exactly match the trusted fixtures;
only dtype differences are ignored. The fitting order is original TRAIN, STOP,
then ascending outside-cap official TRAIN IDs. Houses has no separate historical
confirmation role. Three mandatory houses fits carry the other 33 models exactly.
Prepared metadata freezes all 36 jobs and all four expanded sources, including
the three inherited wave 14 sources. Conditional seed 349 uses those same jobs
and watchdogs for 36 fresh fits; the official companion requires the two-seed
gate and freezes 72 existing models before assessment. No DEV-based fallback is
provided.

The owner reports 51 synthetic tests passing, including six native fits at seeds
47/349, inherited-source dispatch, exact materialization and excluded invalid
UTF-8 sentinels. This independent source review performed no real acquisition,
row extraction, fit or score read. Wave 14 remains immutable.

## Inactive multiclass leaf-correction math review, 16 October

The scratch helper `fixed_multiclass_leaf_refit.py` at SHA256
`5813f3760f8e0a7a7c4f7ec203f115008b5ddbc2a629802575334ccbdc7acee2`
has no unresolved material issue in its stated scope. The objective is
`(sum CE((F + A delta)/T) + lambda ||delta||^2/2) / n` with fixed positive T and
lambda. Its gradient includes `1/T`. Positive ridge resolves both common-logit
shift and duplicate-indicator nullspaces. Equal lambda grids across scalar and
vector layouts do not imply equal capacity, and this sum-loss penalty weakens
relative to the data term as sample size grows.

Ten independent synthetic math tests passed in 0.72 seconds on CPU 0 with one
thread. They build a separate dense basis, check central finite differences at
three temperatures, verify the positive-ridge Hessian and compare the optimizer
against a damped-Newton solution. No tolerance was relaxed. The inference owner
reports 20 additional native tests covering scalar/vector CTBoost, scalar/vector
XGBoost and LightGBM, ordered categorical replay, schedules, zero identity,
native leaf mappings and persistence; all 30 subsequently passed together.

Read-only review confirms the class support restrictions, CTBoost schedule index
`tree // K` for scalar trees versus `tree` for vector trees, and XGBoost 3.4.1
vector leaf-slot mapping. The route oracle retains physical-tree accumulation
order and assigns zero to unobserved leaves. Analytic rounding bounds and fixed
raw-margin/probability tolerances must both pass. Changed leaf outputs retain
historical training statistics, so these are inference artifacts only.

The draft has no real-data or experiment entry point. Activation still requires
a frozen protocol, resource limits, fresh TRAIN-only anchors for STOP selection,
fixed temperature applied once, and mandatory transfer of selected controls to
the exact parent union/expanded fitting rows. No real model, dataset or score was
read for this review.

## Houses header compatibility continuation, 16 October

The original wave 15 materialization stopped at `unsupported_header`, before
any data-instance parsing or model fit. The acquired 1,446,365 bytes matched the
known MD5; their frozen SHA256 is
`c71f44ba2522f61a8ba1918c220074be4a771e1c8c8c07d031e98368f4f94c6c`.
The owner inspected only the header and found `REAL` declarations. The original
plan, sources, byte receipt and failure receipt remain preserved.

Independent diff review cleared the separate v2 compatibility retry. Its only
parser behavior change is a noncapturing `(?:numeric|real)` alternative in the
header type regex. Ordered schema, raw hashes, row accounting, selected numeric
grammar and excluded-row byte boundary are unchanged. This alias is supported
by the primary [Weka ARFF format documentation](https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/).
The new plan requires an identical authorized manifest, binds the original
failed attempt and source SHA, and rehashes the stored bytes without refetching.
Its seed 349 and official companions follow the new immutable source chain.
The owner reports 52 synthetic tests passing, including an all-REAL header with
opaque invalid-UTF-8 excluded rows and unsupported STRING rejection. No real
source body was opened by this reviewer.

Reviewed v2 SHA256s:

| File | SHA256 |
| --- | --- |
| `authorized_houses_arff_v2.py` | `4f38da8f03249f83e361d744fb662941c9879f6750a75538019676dc44432c16` |
| `fair_accuracy_houses_cap_v2.py` | `37cefdd4263c5413113966bdbd5754b427ac33afc7d31298a99d638fa4dcbde4` |
| `fair_accuracy_houses_seed_check_v2.py` | `82d3f770f86a8c4dda90dcfae79c04bcbb27aa4f12c6f4a12a329de0b544690b` |
| `fair_accuracy_houses_outer_test_v2.py` | `941ef24326e43c59ed1bc1a9863fa660e08798efaad7f47b07c2e1ff03a00171` |
| `robustness_protocol_houses_v2.json` | `7f202c94d86396a490124b8525b986060ea1b863e07713780dbe214bc480eb5d` |

The owner subsequently reported all three wave 15b fits passing, with 33 models
carried unchanged. The original shared scorer then failed constructing its final
report because the plan lacked descriptive `limitations` text. The failed plan,
sources and prediction artifacts remain preserved. Independent review cleared
`fair_accuracy_houses_report_recovery.py` at SHA256
`130fd715fec8d031c60b48e4079380aa1012127a5328f5940d29184ff14b98f1`:
it supplies only the already-frozen policy's limitations text to the in-memory
plan, invokes the unchanged scorer/predictor and records wrapper provenance
before execution. Original 36 prediction archives are retained and hash-checked;
rerun archives go into a separate directory. The final report is withheld until
all prediction, label and row-ID arrays match the original dtype, shape and bytes.
There is no fitting, selection, model fallback or frozen-source modification.
The owner reports the focused report-contract test passing. This continuation
reviews recovery integrity, not the resulting quality scores.

## Wave 16 multiclass protocol review, 16 October

No unresolved material blocker was found in the canonical main, conditional
seed 457 companion, official companion, policy or fixed-leaf helper. Reviewed
SHA256s:

| File | SHA256 |
| --- | --- |
| `fair_accuracy_multiclass_refit.py` | `677f52ab9c2bf3fc16730717aa3aaa5dcc0da7c4571ede1d7329abd09346fe30` |
| `fair_accuracy_multiclass_seed_check.py` | `e3990dffb40d4ac93ac56147b565a7862b35fe2c467447b770c20132eff55358` |
| `fair_accuracy_multiclass_outer_test.py` | `6e60057169819a2353148a464b3af7aeb86dd40e7633c0f435f22340c1a81626` |
| `robustness_protocol_multiclass.json` | `5c26d61422e09c12632ca73e77d9cea49d0a0ecae14e54650aa6c406b7b6379e` |
| `fixed_multiclass_leaf_refit.py` | `03d50d12333581e0c79314b827347484dd2cc22a562394127b9799d3ef6d5cbf` |

Twelve fresh TRAIN anchors use inherited tree controls and requested counts.
SDSS17 removes the original STOP block from its hash-verified expanded source
before normalization; outside-cap rows remain in their inherited ascending
order. Every anchor fits its own category vocabulary. The inherited temperature
is fixed. Four positive-ridge TRAIN solves supply eight shrinkage candidates,
plus identity, to native STOP logloss selection. Exact ties retain identity,
then prefer larger lambda and smaller shrinkage. All 96 candidates and twelve
choices must succeed before the transfer stage begins.

Nonzero selected controls are recomputed on the parent union/expanded model and
its exact fitting rows; identity preserves the parent artifact. The other 24
models carry unchanged. The full 36-model policy freezes before DEV with no
old/new fallback. Conditional seed 457 verifies unchanged inherited job fields,
all four expanded sources and the twelve selected controls, then fits 36 fresh
models without reselection. The official companion remains behind the initial
and mean-seed gates and freezes 72 existing native models.

The canonical helper preserves the previously reviewed objective, native maps,
solver and numerical tolerances. Its added named-array estimate runs before
ordered transformation, native leaf prediction and sparse construction. It
counts one shared CSR for vector trees, class-specific coefficient blocks,
construction/objective arrays and explicit L-BFGS-B work/history. Its 2 GiB
limit excludes input frames, native models, categorical workspace and Python
overhead; the separate 8 GiB process-tree RSS and 1,200-second job watchdogs
remain authoritative. Failures close the gate without relaxed retries.

The recorded immutable accuracy runtime is native
`c5ea7e026942c3f1190ca5511fbf74f353bfeb08e7771f76bc5f6c1ca3e4d197`,
NumPy 2.0.2, SciPy 1.16.3, XGBoost 3.4.1 and LightGBM 4.7.0. Its
[helper validation](inference/multiclass_helper_validation/validation-02.json)
reports 36 passed, zero skipped. Nine owner-run integration tests additionally
cover all five native formats, STOP-only versus outside-cap category handling,
fixed temperature, native/sparse oracles, pickle identity, identity transfer,
all four expanded-source delegations and seed/report contracts. This review
reads code and synthetic receipts only; it does not claim a real-data gain.
Before freezing, the owner added an immediate anchor-persistence receipt so a
later solver timeout cannot erase completed-fit evidence, separated planned and
completed anchor counts and nonidentity/identity evaluations, and corrected
policy wording from seed-model trees to models. Those reporting-only changes
were reviewed, and the nine integration tests passed again. The hashes above
identify that final frozen revision.

The independently restored accuracy report was also audited: all 15 completed
aggregate rows, 36 latest task errors and 36 wave-13 round-count rows match
immutable receipts at displayed precision. Its fold manifest confirms all
classification training and validation folds contain every class. Metric labels
and direct evidence links were added through an explicit UTF-8 atomic rewrite;
the prior nonempty report is preserved. The
[report audit receipt](validation/accuracy_report_review.json) records hashes
and the absence of new fits, scoring or dataset/prediction-array reads.

### Completed wave-16 evidence update, 16 October

The immutable wave-16 summary reports all twelve anchor fits, 48 selection
solves, 96 nonidentity evaluations, twelve identities and twelve transfer jobs
successful; eight nonzero controls required transfer solves. The complete gate
failed again at -0.142578% median versus XGBoost and -1.438692% versus LightGBM,
with six wins against each. The multiclass-family medians improved without
changing full-panel win counts. Conditional seed 457 and official scoring remain
unopened. These are existing recorded outcomes, not new review-time scoring.

The editable accuracy report now contains all sixteen gate rows, the twelve
selected multiclass controls and all 36 latest task errors. All 36 historical
CV-count rows are unchanged. The prior report is preserved, and the
[wave-16 update receipt](validation/accuracy_report_wave16_review.json) binds
the immutable source hashes and atomic UTF-8 report hash. No dataset or prediction
arrays were read, no fit was run and no score was recomputed for this update.

## Wave 17 clean scalar correction refresh review, 16 October

The main runner, conditional seed 569 companion, official companion, policy and
clean-refit helper have no unresolved material source-review blocker. This
review identifies the following revision; real fits still require the owner's
planned experiment integration validation and source freeze:

| File | SHA256 |
| --- | --- |
| `fair_accuracy_scalar_refresh.py` | `6ed3b805717648676a9b5ede4e198c7a4987318ad9cf259240b0237795e26844` |
| `fair_accuracy_scalar_refresh_seed_check.py` | `24251e298e452017de4957c0f845c82930088df3ebebdc29d903aeebdb4ecee8` |
| `fair_accuracy_scalar_refresh_outer_test.py` | `4178fcbda4fb54e33c27097c702ab27f0be7ceac7719ed99d0e1bb617c0f9254` |
| `robustness_protocol_scalar_refresh.json` | `93c060c013870c7f3589945e963a50c4ec892578a7a251e9a18d48f4cedf2db5` |
| `fixed_clean_refit_check.py` | `c9aa686c8496363707d248a82cabda053e0f8d03173e3a9f54bb1a72910a31aa` |

The central safeguard is replacement from clean native fits. Both old scalar
correction fields are disabled before each of 24 TRAIN anchors and each of 24
mandatory union/expanded fits, including identity selections. The runner never
subtracts guessed prior coefficients or uses an already corrected parent as the
new optimization baseline. Standalone anchor and clean-fit receipts record the
disabled controls and their SHA256. A union fit completion receipt is saved
before structural checking, so a failed comparison cannot erase completed work.
Planned and completed native-fit counts are separate.

Credit, diamonds and houses use original TRAIN followed by their inherited
ascending outside-cap rows for selection anchors, excluding original STOP
before vocabulary fitting. Final fit order is original TRAIN, STOP, outside-cap;
other active tasks use their original roles. Four existing TRAIN-only positive
ridge solves yield eight shrinkage candidates per task/library. Actual native
STOP RMSE or 1-AUC chooses, with exact ties favoring clean identity, then larger
lambda, then smaller shrinkage. All 192 nonidentity candidates and 24 choices
must succeed and freeze before any mandatory final fit. No DEV fallback exists.

Before the new correction is folded, the deterministic clean seed-47 union
model must match the corrected parent in full non-leaf native state, ordered
category schema and CTBoost pipeline/data/training metadata. The helper reuses
the frozen scalar topology functions, which strip only leaf outputs. It retains
base scores, feature/cut/category metadata, learning rates and historical
statistics. This comparison cannot itself prove fresh fitting; the runner's
actual fit calls, disabled settings and receipts supply that provenance.

The scalar 4 GiB named-array estimate bounds native leaf copies, resident CSR,
construction/maps/margins and LSQR or explicit L-BFGS-B workspace before design
allocation. It excludes models, frames, category-transform workspace and Python
overhead. The separate owned-process 8 GiB RSS and 1,200-second whole-job limits
remain authoritative. This predeclared scalar budget differs from the smaller
multiclass budget because the fixed regression leaf matrix can be much larger;
it is not an observed-failure relaxation. Solver and native numerical gates are
the unchanged frozen binary/regression helpers.

Twelve wave-16 multiclass artifacts carry exactly. Conditional seed 569 checks
the exact refreshed jobs and all four inherited expanded sources, performs no
STOP reselection, and delegates multiclass correction/temperature to the frozen
wave-16 policy. Its own aggregation uses 47, 569 and their mean within each of
twelve datasets. Cross-seed topology is not required to match the seed-47 model.
The official companion requires that exact seed protocol and passing mean gate,
then freezes 72 existing models before a single assessment.

The [clean-refit validation receipt](inference/clean_refit_validation/validation-01.json)
reports 30 passed, zero skipped in the immutable accuracy runtime. Tests cover
fresh-versus-corrected scalar native models in all three libraries, ordered CTR,
pickle identity, non-leaf/schema/pipeline mutation rejection, objective scope,
and memory rejection before native leaf prediction. No frozen regression or
binary helper was edited. This independent review reads source and synthetic
receipts only and makes no claim about wave-17 real-data outcomes.

### Wave-17 final freeze continuation

The [synthetic validation receipt](accuracy_wave17/synthetic_validation.json)
reports all eleven experiment integration tests passing in 2.37 seconds on the
immutable accuracy runtime. Independent
review read those tests: six library/family pipelines, seed-569 independence
from the seed-47 topology check, a one-correction-only oracle, fresh clean
identity instead of a corrected parent, all eight candidate evaluations,
multiclass correction/temperature and four-source delegation, exact tie order,
and seed/report companion contracts. The frozen plan SHA256 is
`258eae391b666f63cdf074df25e02f722590484d43c1191ceb6a3f9783cc6a71`.
Its main/helper/companions/policy bindings match every reviewed hash above.
No source-review blocker remains; no real fits, dataset reads or score
computations were performed by this reviewer.

## Numerical correctness review, 16 October

The [synthetic metric audit](inference/metric_numerics_audit/README.md) confirms
raw-margin log-loss discrepancies in the immutable accuracy runtime using 21
constructed cases. No real data/model access or fitting was needed. Stable
softplus/log-sum-exp is mathematically appropriate, but the production metric
change is explicitly deferred: ordinary native and Python warm-start loops can
reuse historical clipped scores. A persisted arithmetic version must govern
both loops before safely changing defaults. Recomputing all old checkpoints is
not a general alternative because DART historical weights are unavailable.

The independently reviewed CTR precision design is narrower. New pipelines use
double prior, running and total Mean-CTR statistics and serialize format 5.
Loaded formats 1--4 retain explicit float rounding for sums and priors, with
legacy values first loaded through float before widening. Format 1--3 transform
smoothing stays historical until explicit refit, which still upgrades smoothing
to version 2 and emits format 4; format-4 ordered replay stays format 4. The
serialization dispatch distinguishes smoothing and accumulation versions. New
Mean counts are converted directly to double, while Frequency arithmetic stays
unchanged. Final transformed columns still pass the finite-float range check.
Embedding target-width inference retains the legacy default call and its
feature arithmetic is not modified. No conditional feature-test or split code
is changed.

Initial code and focused-test review found no material implementation blocker.
The focused tests cover all permutations of a cancellation example, three prior
strengths, finite averages with totals above float range, positive integer drift,
legacy rounding/refit, malformed nonfinite state, final output range, Frequency
compatibility and native categorical-code gathering. Adoption still requires
the owner's new-runtime test result plus independent prepatch-runtime state,
ordered-transform and warm-start comparison; candidate self-replay alone is
not evidence of compatibility with the old implementation. No actual quality
impact is claimed by this review.

### CTR candidate validation continuation

The [independent review receipt](validation/ctr_precision_independent_review.json)
binds the reviewed source and evidence. All 110 compiled native source hashes
match the checkout. Candidate native `04adb21b...` passes 103 focused tests
(three optional export-environment skips), 125 broader pipeline/persistence/
embedding/warm-start tests, and 230 native checks. Six initial wider-oracle
one-bit failures came from relabeling new double priors as legacy state but
comparing against their pre-load double values; the revised oracle correctly
uses the legacy loader's rounded priors while deriving category sums directly
from raw synthetic labels. Format-5 coverage was added. Two initial focused
failures were obsolete version expectations. Original failure logs are retained;
no numerical tolerance was relaxed.

An independent old-runtime oracle captured sixteen format/problem cases from
native `94b2b0bf...`: formats 1--4 crossed with regression, binary, multiclass
and Frequency, including category combinations. Candidate loading, ordered
refitting, serialized states and transformed uint32 bits match exactly. A
persisted old model also has exact raw prediction bits and exact continued
native/pipeline state after warm start. New format-5 cancellation cases now
verify their own exact state and bit replay as well. These results remove the
source/focused-validation blocker. Full CP310 validation and existing-model
parity remain separate owner checks before installation, with the immutable
accuracy runtime unchanged.


## Wave-17 terminal result and numerical diagnostic review

The [terminal summary](accuracy_wave17/summary.json) records 24 completed TRAIN
anchors, 96 attempted solves (95 successful), 190/192 successful nonidentity
candidates and 24 identity evaluations. The diamonds/LightGBM lambda-1 solve
reached the frozen 2,000-iteration LSQR limit, status 7; its two shrinkage
candidates failed. All other selection jobs passed. The all-success barrier
prevented every clean-union fit, DEV score, seed-569 fit and official-test
assessment. This is a numerical failure, not a quality-gate result. The latest
complete policy remains wave 16. Cumulative quality-study tree fits are 2,448
(816 per library), plus eight separately counted diagnostic replay fits.

The [report-update audit](validation/accuracy_report_wave17_review.json) verifies
that all sixteen completed development gate rows and the latest complete
36-score section stayed unchanged. The update preserved a backup and used
explicit UTF-8 with atomic replacement. No data or prediction arrays were read
and no scores were computed to make this report.

The separately frozen [ridge diagnostic](ridge_numerical_diagnostic/plan.json)
was independently cleared after source and test review. Its
[review receipt](validation/ridge_numerical_independent_review.json) binds the
plan, source, tests and the 43-pass immutable-runtime synthetic receipt.
The [mathematical specification](numerical_ridge_review.md) keeps the same
positive-ridge objective under augmented right column scaling, with zero
additional solver damping and an original-coordinate gradient dual-gap
certificate. Native status and certificate gates both must pass; status 7
remains a failure. Provenance checks bind the persisted clean anchor, expanded
source and exact TRAIN-plus-outside-cap order. The existing expanded pickle
contains an authorized STOP block that is decoded but excluded from the
residual/design; no STOP predictions or metrics are computed. No DEV,
confirmation or official-test values enter this diagnostic.

The diagnostic predeclares original LSQR and two column-scaled LSQR/LSMR
variants, each limited to 2,000 iterations, with a 1,200-second whole-worker
watchdog, 8 GiB owned-process RSS limit and 4 GiB named-array budget including
an extra linear-vector reserve. Differences in stopping tolerances prevent
direct speed-superiority claims. It cannot revive wave 17 or promote its
partial controls; any later solver policy requires a separate frozen study.


## Prospective current-data CV resource review

The [metadata-only memory table](full_current_cv_memory/README.md) covers all
36 current policies at half/base/double requested counts, both maximum fivefold
training rows and full authorized union rows. It verifies inherited wave-17
controls equal the complete wave-16 job metadata; no failed partial choices are
used. The 216 detailed estimates distinguish shared vector-tree CSR from
scalar class blocks, active versus identity corrections, leaf-cap bounds,
L-BFGS-B allocations, and the new matrix-free ridge vector reserve. No model
payload, dataset or prediction arrays were loaded and no fits/scores ran.

The largest potential named allocation is 4.205 GiB for the full diamonds/
LightGBM double-count refit, versus 3.369 GiB for a fold. Thus the previous
4 GiB named limit cannot cover every prospective final fit. Six GiB would
cover these named bounds, but is not adopted here; the whole-process ceiling
needs a separate quiet-host/headroom assessment because models, native fitting
histograms, CTR/frame work and allocator overhead are excluded. Wave 17's
separate observed maxima (2.128 GiB RSS, 2.014 GiB named) are context, not an
empirical multiplier or a proof of future feasibility. No new policy or
resource limit was frozen, and no existing experiment limit was relaxed.


## Prospective final inference runtime amendment

The [version-2 selection policy](inference/final_runner_preparation/selection_policy_v2.json)
was independently checked against the preserved original policy. All seven
referenced digests match, including the runner, old policy, new native
`04adb21b...` runtime and legacy-model parity provenance. The model-selection
rule, primary seed 47, twelve tasks, batches, orders, prediction checks and
raw/normalized scopes remain identical. The amendment fixes the validated
serving runtime prospectively, before retry accuracy outcomes or final timing;
it does not adopt new training arithmetic in the immutable accuracy runtime.
The [review receipt](validation/final_inference_policy_v2_independent_review.json)
records the comparison. Every eventual selected model must still pass the
runner's exact full-source and batch reference checks. Historical `94b2b0bf...`
timings cannot be relabeled as measurements of this runtime. No timing or
model/data payload read occurred in this review.


### First ridge diagnostic terminal review

All three frozen 2,000-iteration variants reached status 7 and failed both
strict original-coordinate acceptance gates. The [result review](validation/ridge_numerical_results_independent_review.json)
binds the completed receipt and archives; the [mathematical note](numerical_ridge_review.md)
records the three gaps and why no transformed-residual or speed claim follows.
The whole TRAIN-only diagnostic completed in 454.266 seconds with 1,426,464,768
bytes peak RSS, no watchdog failure, no new fit and no held-out score. Wave 17
stays terminal and no numerical successor is open for quality evaluation.


## Inactive full-current-data CV support review

The support module `benchmarks/fair_accuracy_current_round_cv_support.py` and
its synthetic sibling were reviewed as an inactive orchestration prototype.
The one material finding was a silent old-cap fallback when a known expanded
task lacked its source record; the final guard now rejects that omission.
Source SHA256 is `75d92497e14dd5e456d2ba22fe49b2a720094c2ebad0f9990e0616951f2a455e`;
test SHA256 is `05ef46b54714dde1c78790c90a7b9deb4239ac41773f53925b6898f21a92c5c5`.
The owner's log reports 46 synthetic checks passing in 0.26 seconds on CP310,
NumPy 1.26.4 and pandas 2.3.3, with no native fit or real payload access.

Raw source/role hashes, exact original prefix values and labels, authorized
ordered IDs, ascending additional rows, fold-local observed category values,
clean correction/temperature dispatch and all 540 receipt identities have
explicit checks. The reducer uses five equally weighted fold errors and keeps
duplicate count profiles. This clears the support handoff only: the certified
solver/native adapter, completed-parent activation proof, shared fold manifest,
resource policy and exact-c5 all-library native integration remain unbound and
require separate review before any real use.

The separate [fallback research note](ridge_solver_fallback_research.md) favors
unscaled LSMR before a new low-rank preconditioner if the active extended LSQR
diagnostic fails. It records primary sources, objective/certificate compatibility
and rank-64/128 storage/work estimates without implementation or spectral claims.


### Extended ridge diagnostic terminal review

The [independent terminal review](validation/ridge_extended_results_independent_review.json)
confirms the strict unscaled zero-start LSQR diagnostic passed status 2 at
5,541 iterations, relative gap 1.8320e-14 and normalized original gradient
2.1776e-9. Whole-worker cost was 399.640 seconds and 1,426,857,984 bytes RSS;
no resource failure, new tree fit or held-out score occurred. This supports
review of a separately frozen numerical successor, not an accuracy claim.
Every later solve still needs its own original-coordinate certificate and
resource success. The inactive LSMR/Nyström contingencies were not activated.

### 2026-10-16: wave 17b final pre-execution review

The [independent review receipt](validation/scalar_numerical_retry_independent_review.json)
clears the separately frozen numerical successor. All six source/test/policy
digests match the owner's exact-c5 validation receipt (21 synthetic tests passed).
Binary reuse now proves complete indices, prior controls, choice/anchor identity
and unchanged evidence. Missing or uncertified regression jobs close the transfer
barrier. The 24 mandatory union models are freshly fitted without old scalar
corrections; identity is also clean. Multiclass carries and all four expanded
sources remain fixed, with conditional seed 569 and official assessment gated.
The review ran no real fitting or scoring. Every new solve must independently
certify; the successful diagnostic does not establish an accuracy improvement.

### 2026-10-16: inactive native CV adapter

Read-only review of `fair_accuracy_current_round_cv_backend.py` and its synthetic
native integration found no material blocker for the inactive adapter. The
owner's `native-validation.json` records 30 passing checks, including 22 tiny
native fits, on the immutable c5 runtime; no real payload or score was accessed.
Fresh fits clear correction metadata, fold-local processing supplies no
validation labels, fixed temperature is applied once, and correction callbacks
cannot mutate inherited algorithm controls. The 64 MiB test cap is synthetic.
The recorded older 17b source hash predates only binary evidence-reuse hardening
and prose; its numerical callbacks were unchanged. An executable future CV
runner still needs completed-parent admission, final source hashes, shared
fold manifests, actual resource limits and conditional seed/test gates before
any real use. No CV activation follows from this review.

### 2026-10-16: prospective official-assessment access wording

The new current-data CV outer companion preserves the sequential primary/seed
gates, 72-model freeze, source/split hashes, one-shot started marker and no-refit
assessment. Preparation reads metadata, split indices and file hashes only.
If assessment is eventually admitted, its full-source parquet decode and
official-TRAIN label-encoder validation also access values belonging to the
unused original confirmation role. That is distinct from opening
`confirmation.pkl`, selecting a policy on confirmation, or producing a separate
confirmation score; none of those operations is part of the companion. Future
reports must retain this distinction. No official assessment or data decode
was performed in this source review.

### 2026-10-16: current-data CV executable readiness

The [independent executable review](validation/current_round_cv_independent_review.json)
clears the inactive main, resource module, seed-677 companion and outer
companion for source readiness. All 18 final receipt source hashes and the
policy hash match. The owner's final 52 synthetic runner tests pass; the
hash-linked earlier validation also includes 30 native checks and 22 tiny
fits against the final frozen 17b callbacks and immutable c5 runtime.

Admission requires a completed, numerically successful 17b policy followed by
either primary failure or a verified linked-569 mean-gate failure. The primary
47 policy always supplies the anchors. Shared raw authorized rows/folds,
fold-local preprocessing, all 540 independent fits, deterministic choice
reconstruction and 36 mandatory union fits are enforced. Resource checks cover
full-union as well as fold allocations, with positive fixed headroom, bounded
waiting and terminal failure. Conditional 677 and the one-shot outer gate
preserve all current sources and controls.

This is not activation or a quality result. Real preparation still requires
the qualifying parent outcome, a quiet-memory observation and explicit resource
limits frozen in a new plan. The synthetic tests did not open real payloads or
run real scoring; this reviewer performed source and metadata checks only.


### 2026-10-16: wave-17b closed selection audit

The [independent audit](validation/wave17b_selection_independent_review.json)
verified all 24 frozen choices, 48 new certified regression solves and their
96 successful native candidate evaluations, plus 12 unchanged binary grids.
All 43 inspected source/policy hashes and all referenced anchor/artifact hashes
match. Identity wins exact STOP ties, followed by larger penalty and smaller
shrinkage among nonidentity candidates.

With explicit narrow authorization, the audit decoded only the 108 previously
used binary STOP prediction/label/row-ID archives (identity plus eight candidates
for each grid). The exact frozen Python 3.12.11, NumPy 2.0.2, scikit-learn 1.6.1
and hash-bound metric function reproduced every candidate error and the full
identity-inclusive selection exactly. Regression metrics and certificates were
audited from recorded metadata; the audit independently checked certificate
arithmetic and thresholds but did not rebuild their matrices or targets.

No feature frame or model object was decoded; no fitting, model prediction,
DEV, confirmation or official-test access occurred. The largest solve used
5,541 iterations. Selection success does not establish a completed union policy
or an accuracy improvement; those later stages remain separate.


### 2026-10-16: wave-17b terminal policy and report audit

The [closed terminal audit](validation/wave17b_terminal_independent_review.json)
passes all 24 fresh union fit/resource receipts and clean-control provenance,
15 mandatory corrections, 12 byte-identical multiclass carries and all 36
model/archive hash links. Choice freezing precedes every transfer, and the
complete model freeze precedes the development report. Recorded native
nonleaf/schema/routing and scalar numerical evidence remain consistent; this
metadata audit did not decode model objects or recompute native predictions.

Independent aggregation of the twelve reported task errors exactly reproduces
all eight family/comparator summaries. The gate is false: +0.286104% median
relative error gain and 7/12 wins against XGBoost, -1.358863% and 5/12 against
LightGBM. No conditional seed-569 or official assessment follows. The complete
failed-primary policy satisfies the prospective current-data CV admission
contract; activation still requires its own fixed resource/row/source plan.

There are 24 new model-training runs in 17b and 2,472 cumulatively (824 per
library), plus eight separate diagnostic replays. Reused anchors and coefficient
solves are counted separately. The [report update receipt](validation/accuracy_report_wave17b_review.json)
binds the atomic UTF-8 update, all twelve latest errors and the preserved
historical wave-13 count table. No frozen experiment or source was edited.


### 2026-10-16: wave-18 frozen preparation audit

The [preparation audit](validation/wave18_preparation_independent_review.json)
passes the actual frozen plan (`6a01ef38324b7ddc36814733a076c9a46a8f3350b6ca5ec8a0ae28626e16688d`),
which admits the audited complete 17b primary failure and retains all 36 current
jobs and four authorized expanded sources. All 216 prospective count/shape
allocation profiles agree with the frozen memory evidence; the largest active
named estimate is 4,515,168,888 bytes. The fixed one-worker budget is 6 GiB named
arrays, 10 GiB process-tree RSS, 4 GiB reserve and 1,200 s, with bounded waiting.
The preparation snapshot has 16,133,193,728 available bytes, above the 14 GiB
required before dispatch. This is an admission observation, not a guarantee
that every later job will converge or stay within its budget.

All twelve shared manifests retain five folds and positive recorded class
counts in every classification fitting/validation fold. Sixty ordinal
train/validation hash pairs and class-count arithmetic were independently
reconstructed from the frozen assignments/counts. Raw source values, class
labels, row-ID arrays and vocabularies were not independently reloaded; their
correctness remains bound to reviewed preparation code and source receipts.
No running worker receipt was read. The 540 independent fold fits and 36
mandatory final fits remain planned work at this audit, not completed counts
or a quality result.


### 2026-10-16: terminal wave-18 headroom and prospective 18b resource review

The [closed resource audit](validation/wave18_terminal_resource_retry_review.json)
verifies all 540 job identities: 58 completed successfully, one was never
started because the fixed 300 s headroom wait failed, and 481 subsequent jobs
were explicitly not started. No count selection, union, development or later
gate ran; the 58 actual fits bring the cumulative count to 2,530.

A prospective 8 GiB process cap retains the 6 GiB named-array budget and 4 GiB
system reserve, so dispatch requires 12 GiB available. It is feasible to attempt
because all existing named estimates fit, but this does not prove whole-process
memory or convergence. The tighter cap and unchanged time/numerical gates must
remain terminal. A separate amendment/new plan must bind the preserved failure,
restart all 540 + 36 jobs and use no partial fit reuse. No method, source, fold
or quality-policy change is recommended by this resource review.

### 2026-10-16: frozen 18b retry and prospective release timing

The [18b preparation audit](validation/wave18b_preparation_independent_review.json)
confirms that the only resource delta is the 10-to-8 GiB RSS ceiling. Apart from
the creation time and quiet-memory snapshot, all plan fields are identical;
the shared fold manifest is byte-identical. All 36 controls and four expanded
sources remain unchanged. The new plan binds the preserved failure and starts
all 540 + 36 fits afresh. No running result was used in this admission review.

The user subsequently requested release of validated improvements after this
final experiment and passing CI even if the accuracy target remains unmet.
The [independent timing-policy review](validation/final_timing_policy_v3_independent_review.json)
binds draft `423eb67ba4264dce8aa5582e5b0c005a0c5e8e5d69a4b804c2d569c5f8ea41cd`
and independently verifies eleven source/runtime/policy bindings before any
18b development receipt exists. The prospective timing policy selects the
entire complete 18b primary47 policy regardless of its quality result, with
whole17b fallback only on a documented terminal numerical/resource failure.
Original seed677 and official-test quality gates are unchanged. Final branch,
model, check-only parity and three-order timing evidence remain required;
this review does not claim timings or accuracy superiority.

The [release-CI readiness review](release_ci_readiness_review.md) proposes
explicit optional-library checks for native integration cases, a small psutil
test dependency, portable test-local affinity handling, and one pinned
scientific job. Pure protocol/math and core CTBoost checks remain enabled.
No frozen source, test, workflow or environment was changed during this review.
The absolute-leaf numeric/native prototypes remain inactive and unrun.


### 16 October — release source review and external-interruption audit

The [independent release-readiness review](release_preparation/release_readiness_independent_review.md) found no new source-level compatibility blocker. It binds the reviewed release files, distinguishes static sdist checks from actual artifact validation, and requires final installed-package, exact-stack, platform/compiler and publishing checks. The report-only scan of 7,115 proposed public text files found no credential-pattern matches; no values were emitted and no NPZ/model contents were decoded.

During that scan, this reviewing agent mistakenly identified a concurrent accuracy worker as its own scan and issued `Stop-Process -Id 7488 -Force`. Its parent/name check did not establish process ownership. The [incident report](release_preparation/wave18b_external_interruption_incident.json) preserves the exact command, admission of responsibility, process identities and original receipt hashes. This external action caused the 18b interruption; it is not a numerical, resource or library failure. All process-management actions by this agent stopped immediately after discovery.

The [closed-attempt metadata audit](validation/wave18b_terminal_independent_audit.json) independently matches 540 job identities, 297 verified jobs, 298 native fits completed, 242 not started, 298 resource receipts and 53 source/policy/parent hashes. No selection, union or development output exists. Original sources and receipts remain unchanged. The [v4 timing draft review](validation/final_timing_policy_v4_draft_independent_review.json) preserves all scientific and timing gates, but is explicitly conditional on exact recovery-controller/test/prefix-manifest bindings; it does not authorize execution or fallback.

### 22 October — installed release candidate validation

The [aggregate validation receipt](release_preparation/installed_candidate_02/validation_complete_20261022.json)
binds the installed 0.1.62 candidate02 wheel, runtime/source identities and
individual execution receipts. The full Python suite completed with **2,526
passed, 44 optional/CUDA skips and no failures or errors**. The separate pinned
scientific selection completed **447 tests with zero skips**, and the pandas
3.0.6 categorical-compatibility selection completed **117 tests with zero
skips**. These selections overlap the full suite and must not be added together
as a count of unique tests. The same native test executable passed **230
checks with zero failures** after its Python/NumPy embedding environment was
corrected. Earlier failed or deferred validation attempts remain preserved.

The [Java 17/Maven receipt](release_preparation/security/jvm_validation_summary.json)
records **nine JVM tests passed with no failures, errors or skips** and resolves
Jackson databind/core to 2.21.7 and annotations to the upstream 2.21 version.
Live GitHub metadata on 22 October still reports alerts #2–6 open on the
unchanged default branch; 2.21.7 meets all five reported patched-version
requirements. Alert closure is a later default-branch verification, not a
claim from these local tests.

These are local installed-candidate checks. They do not certify the final
rebuilt artifacts, platform/compiler CI, complete wheel/source dry run,
documentation deployment or publication catalogs. The release tag still
requires passing checks for the final release commit. The normal CI job may
skip unavailable optional integrations; the dedicated scientific job must
retain its zero-skip requirement.

### 22 October — completed 18b recovery and final quality gate

The [terminal primary-policy audit](validation/wave18b_continuation_20261022/primary_policy_audit.json)
verifies all 540 fold jobs, 36 mandatory union jobs, 36 saved models and 36 saved
development prediction archives. The frozen source/dependency and original
interrupted evidence remain unchanged. The recovery reused 297 verified prefix
jobs and ran the 243 remaining original indices; selection and final fitting
retained the original plan. The [closed summary](accuracy_wave18b_recovery/summary.json)
separates reused work, new execution and the extra interrupted native fit.

The recorded twelve-task errors reproduce median relative gains of
**+0.207495974% against XGBoost with 7/12 wins** and **-0.821725078% against
LightGBM with 5/12 wins**. The primary gate fails. Conditional seed 677 and the
official outer assessment were not run under their original gates; no further
quality work is authorized by that experiment. This audit checks stored errors
and artifact hashes without a new fit, data decode, score or model selection.

Cumulative benchmark model fitting is **3,107**: 2,472 through 17b, 58 from the
headroom-failed attempt, 298 from the externally interrupted attempt, 243 new
recovery folds and 36 union fits. The reused prefix is not counted twice; eight
older diagnostic replays are separate. The failed and interrupted attempts
remain explicit evidence, not replaced by this successful completion.

The v4 prospective completion rule selects the entire final 18b recovery primary
policy for serving-cost measurement, despite its failed quality gate. It does
not select 17b or mix per-task winners. The wrapper static review verifies the
frozen plan, runner, actual native/Python files and watchdog before launch, and
chains the authorization, wrapper and preceding result identities across the
reference and three passes. The wrapper review ran no timing. Actual parity,
exclusive execution and measured results remain separate receipts.

The accuracy report now includes the completed recovery row, all twelve latest
errors, the failed gate and exact compute accounting. Its writer independently
reconstructed comparator medians/win counts from recorded error scalars and
checked the terminal audit and summary hashes before atomic UTF-8 replacement.
No frozen plan, source, model, prediction archive or result was edited.


### 22 October — final serving-cost and documentation audit

An independent metadata/hash/arithmetic check of the [completed timing study](inference/final_execution_20261022/README.md)
passes all three balanced library orders and all 108 model/task/batch records
per pass. The check verifies reference and execution bindings, unchanged
runtime/model/plan identities, successful cleanup, sequential pass chronology,
all seven-block medians, every per-case three-pass geometric mean, and every
batch/all-case aggregate. It checks recorded parity assertions without rerunning
native prediction or timing. Heavy-job exclusivity is supported by the root
coordination authorization and execution chronology, not independently measured
system-wide utilization.

The [combined observations](inference/final_execution_20261022/comparison.json)
reproduce raw competitor/CTBoost latency ratios of **2.24696900255 for XGBoost**
and **1.91528740309 for LightGBM**, with CTBoost faster in **36/36** and **35/36**
task/batch cases. Corresponding normalized-frame ratios are 7.05131472973 and
4.92834139340. This is warm serving cost for one prospectively selected complete
policy with differing model quality/capacity; it does not establish an
accuracy-matched advantage or repair the failed quality gate. The comparison
SHA256 is `e46058784dcb81cf04b815d655b38f5b7bb9cf5b017c7baeb4ad4635b726b56d`.

The [installed-candidate parity receipt](release_preparation/installed_candidate_02/final_policy_parity_20261022.json)
and its hash-bound reference reproduce all 36 full-development checks and all
108 model/batch records exactly. That candidate uses native build `4e2924…`,
while the measured runtime uses `04adb2…`; this is correctness evidence, not an
additional latency measurement. The parity receipt SHA256 is
`4d323197bef4fd387f8aefea349c2f6392343aa840afb4d24b607079783db005`.

The release-worktree README, index, benchmark page and 0.1.62 notes were checked
against these completed results. They preserve the remaining accuracy gap,
closed seed/official gates, legacy-format behavior, different training/timing
runtimes and local-test scope. No preparation placeholders or unsupported
claims of completed hosted CI/publication remain in those four pages. The
remaining artifact, CI, deployment and publication checks are separate from
this review.
