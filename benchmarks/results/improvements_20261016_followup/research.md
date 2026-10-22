# Follow-up research: quality per tree and deployment cost

## Late follow-up: sampling and generalization limits

After the first three fixed development portfolios, the strongest low-effort
next experiment is **full sampling versus Bernoulli 0.8**, before implementing
another optimizer or changing the feature statistic. This is a hypothesis,
not an explanation established by those results. In
`src/core/booster_sampling.cpp`, CTBoost's Bernoulli branch keeps original row
weights or sets them to zero; it does not multiply selected weights by the
inverse sampling rate. `booster_fit_loop.cpp` passes these weights into tree
fitting. Consequently, subsampling affects local association-test information,
minimum effective leaf data, Hessian sums and relative L2 shrinkage as well as
ensemble randomization.

The conditional-inference framework selects the feature using a weighted
association test before finding its cut. Fewer active observations can reduce
power against a fixed alternative. A heuristic fixed-effect iid calculation
would increase a quadratic test's noncentrality by about 25% when moving from
0.8n to n observations; this is **not** a theorem for an adaptive boosted tree
sequence. Retaining every row can also increase overfitting, so more test power
does not automatically improve prediction.
[Hothorn, Hornik and Zeileis, 2006](https://www.zeileis.org/papers/Hothorn%2BHornik%2BZeileis-2006.pdf)

Friedman's stochastic-gradient-boosting study gives primary evidence that row
subsampling can improve generalization and computation. It does not establish
0.8 as optimal for a learner with a significance gate. A full-sample candidate
therefore belongs beside the sampled baseline, with identical controls and
validation policy, rather than becoming an unsupported new default.
[Friedman, 2002](https://doi.org/10.1016/S0167-9473(01)00065-2)

A small follow-up portfolio can use four fixed capacity/L2 profiles, each paired
at sampling fractions 0.8 and 1.0, for eight candidates per library. Keep learning
rate, feature-test choice, alpha, leaf solver and stopping metric fixed within
each pair. Include comparable sampling alternatives in XGBoost and LightGBM;
their capacity controls need not have identical numeric values. Record root-only
tree proportion, realized depth/leaves, retained rounds and train/stop error gaps.
A selected winner's improvement alone cannot identify which mechanism mattered.
Do not change the frozen earlier waves or choose individual task settings from
their development errors.

The primary power study of CTree/MOB/GUIDE supports examining the behavior of the
test, score representation and feature grouping. It finds that these choices
can materially alter splitting power. It does not justify removing the
feature-before-cut design or importing its model-tree results as a guaranteed
CTBoost gain. The already-implemented grouped and joint options remain a more
bounded experiment than a new uncalibrated statistic.
[Schlosser, Hothorn and Zeileis, 2019](https://www.zeileis.org/news/power_partitioning/)

If a later, separately frozen experiment studies alpha, keep a non-vacuous
significance threshold, for example 0.05, 0.2 and 0.5 at fixed capacity and
sampling. Alpha changes the operating point of the existing test; it does not
change feature-before-cut ordering. Alpha 1 would preserve that ordering but
effectively remove significance-based stopping. It is therefore excluded here
to respect the stronger constraint of retaining statistical stopping, and must
not be presented as an unchanged regularized design or a new production default.
The official partykit controls distinguish the significance level, the split
criterion, and the independent depth/minimum-weight constraints. Those are
different controls, not interchangeable guarantees of prediction quality.
[partykit ctree controls](https://search.r-project.org/CRAN/refmans/partykit/html/ctree_control.html)

The 12-dataset gate has practical limits even if passed:

* Four tasks per family are metadata-selected from a feature-limited population,
  not a random sample of all tabular learning problems. The <=256-feature filter
  excludes wide hiva, whose inference profile motivates several serving changes.
* One reused data split, adaptive development portfolios and selected checkpoints
  make development an exploratory signal. A second learner seed measures
  sensitivity on that split; it does not double the number of independent tasks.
* A nonnegative median and six wins out of twelve are an operational target, not
  a significance test or a universal ranking. Report every family and task.
* Relative error changes are unstable near zero error. For example, tiny
  multiclass logloss differences can produce very large relative regressions;
  preserve absolute errors alongside ratios rather than suppressing outliers.
* Equal trial counts do not imply equal compute, effective capacity or coverage
  of each library's best options. Partial outer-training data and fixed caps
  further limit comparison to full benchmark recipes.

TabArena's primary report emphasizes validation, tuning and ensemble policy as
determinants of rankings. This local single-model portfolio is not an official
TabArena leaderboard run, even with a later one-shot assessment on untouched
outer-test rows.
[TabArena, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)

Research date: 16 October 2026. This is a proposal and source audit, not a new
accuracy result. All proposals retain statistical feature selection before cut
optimization. The multiplication and tensor-routing investigation is already
recorded in [phase one](../improvements_20261016/research.md) and is not repeated
here. No production code, dependencies, or timing measurements were changed by
this research subtask.

## Ranked hypotheses

| Priority | Hypothesis | Concrete next comparison | Effort / principal limitation |
| --- | --- | --- | --- |
| 1 | Tune learning rate and stopping jointly with existing leaf refinement. | A clean 2 x 2 comparison: learning rate 0.05/0.15 and legacy/safeguarded three-step leaves, with identical feature test, depth and L2. Record round-cap hits and selected prefix. | Low: existing options. Faster learning alone can explain gains; a fixed small round cap disadvantages slow learning. |
| 2 | Improve power and regularization without changing feature-before-cut selection. | Existing quadratic/grouped-8 feature tests; bounded alpha, depth, minimum leaf size and L2 portfolio. Keep the full 256-bin cut grid. | Low: existing options. The best setting varies by task; changing alpha changes nominal testing behavior. |
| 3 | Store the already-shared multiclass tree once and separately evaluate the full leaf solver. | Matched scalar/vector representation identity control, then diagonal/full solver at fixed learning rate. Report physical nodes, bytes, rounds and inference. | Low: existing options. CTBoost vector storage is not a new structural regularizer; the solver is limited to CPU and 3-32 classes. |
| 4 | Remove repeated preparation from real serving workloads. | Raw frame, transformed Pool, native bins, and warm traversal scopes; explicit immutable prepared-input ownership. Profile categorical normalization and unused derived features. | Medium. A prepared Pool still requires quantization; correct missing/unseen/category/schema behavior must survive. |
| 5 | Reduce model loading cost without changing the model document. | Compact whitespace JSON versus the identical indented document; compare bytes, parse/load, first prediction and warm prediction separately. | Very low. Whitespace removal does not reduce the number of parsed objects or the native node representation. |
| 6 | Refit existing leaf values across a frozen ensemble instead of adding more trees. | Research-only 1-3 regularized training-only refit sweeps, compared with equal additional boosting work. | Medium/high. Plausible improvement at fixed traversal cost, but no CTBoost evidence yet; can overfit and changes warm-start semantics. |

### 1. Distinguish optimization from learning-rate and budget effects

The [phase-one accuracy scout](../improvements_20261016/accuracy/README.md)
selected smaller, better configurations on its three datasets, but it did not
contain an unrefined learning-rate-0.15 control. Several baseline fits reached
the 300-round cap. Consequently, the combined gain cannot be assigned to the
leaf solver. The clean four-arm ablation above closes that gap before adding a
large search space. A 1,000-round ceiling is more informative, but cap hits must
still be reported as censored optimization, not convergence.

CatBoost's current official documentation supports multiple leaf-estimation
steps and step-halving loss checks; its defaults differ by objective. This is
practical precedent for leaf optimization, not evidence that CatBoost's exact
settings transfer to CTBoost. [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common#leaf_estimation_iterations)

CTBoost already implements these mechanisms in
`src/core/booster_leaf_estimation.cpp` and `src/core/booster_leaf_solver.cpp`.
Its safeguard checks the unshrunk regularized leaf objective before learning-rate
and DART scaling. It does not guarantee better validation loss or AUC. For
unconstrained squared loss, fixed leaf membership and weights, and L2, the first
Newton step already minimizes the quadratic leaf objective in exact arithmetic.
Repeated steps are therefore a weak regression hypothesis; classification and
coupled softmax leaves are more promising. Float rounding and constraints limit
that exact-arithmetic observation.

Small steps and early stopping also act as regularization, so optimizing training
loss faster is not itself a generalization improvement. Rosset, Zhu and Hastie's
2004 analysis relates small-step boosting paths to regularized fitting under its
assumptions; it does not establish optimality of CTBoost's adaptive trees.
[Boosting as a Regularized Path to a Maximum Margin Classifier](https://www.jmlr.org/papers/v5/rosset04a.html)

### 2. Use the available statistical and capacity controls

The repository has a particularly concrete diagnosis: a high-dimensional raw-bin
feature statistic can have poor power, even when useful cuts exist. Grouping only
the numeric *test* bins retains the original cut candidates. The September
[statistical research](../../../docs/split-statistics-research.md) and
[accuracy study](../inference_accuracy_20260908/accuracy/README.md) make grouped-8
a stronger next hypothesis than inventing a different tree learner. Its first
14-task panel failed its gate; the separate 37-task development panel improved
26 tasks, with median error reduction 1.971%, but regression still trailed the
fixed XGBoost comparator in the median. This mixed evidence supports validation
selection, not a universal default change.

Use modest alpha values such as 0.01, 0.05 and 0.2 as predeclared candidates,
together with depth, `min_data_in_leaf` and `lambda_l2`. Do not set alpha to one
to remove the statistical gate. The conditional-inference implementation in
partykit itself exposes nominal alpha and sample-size stopping controls;
choosing these controls preserves the framework while changing its operating
point. [partykit ctree vignette](https://cran.r-project.org/web/packages/partykit/vignettes/ctree.pdf)

LightGBM's official guidance emphasizes that depth alone does not express the
capacity of its leaf-wise learner, and recommends tuning leaf count, minimum
leaf data/Hessian, regularization, learning rate and rounds. That informs fair
comparator search and motivates CTBoost's own existing regularizers; copying
LightGBM's gain-driven feature selection would violate this task's constraint.
[LightGBM tuning guide](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

For CTBoost regression, capacity and test power deserve more budget than extra
RMSE leaf steps. Include root-only-tree frequency, realized depth/leaves,
selected iteration, and train/stop error gap as diagnostics. These describe
realized capacity but do not identify why a node stopped; native search traces
are needed to distinguish test rejection from split infeasibility. Training
gaps alone also do not establish overfitting or insufficient boosting rounds.
Do not use development outcomes to choose a second-wave grid unless that wave
is explicitly reported as adaptive development.

Avoid retesting failed ideas unchanged: reducing global bins to 64 lost the
September aggregate comparison; the joint-cut prototype regressed strongly;
temperature scaling failed its median-improvement gate; fractional CTR-prior
arithmetic correction was not a consistent accuracy gain. A revised experiment
needs a new diagnosis and a separately frozen protocol. Joint feature testing
also retains its documented integer-frequency-weight restriction; no arbitrary
importance-weight calibration claim should be made.

### 3. Separate multiclass representation from multiclass learning

`RunMulticlassIteration` in `src/core/booster_fit_multiclass.cpp` first builds one
shared structure, then materializes scalar class trees or vector leaves. Thus
`multi_output_tree` removes duplicated structure; the phase-one matched control
was prediction-exact. It cannot, by itself, explain a better test score. The
full softmax Hessian solver is a separate change to leaf values.

XGBoost's 25 August 2026 primary implementation report is an important comparator
update. Its scalar option grows separate class trees, whereas its vector option
learns a shared partition. The report found lower held-out loss on 9/10
multiclass datasets and fewer splits on all ten, with some longer training and
round-cap limitations. Those were GPU experiments with one split per dataset;
the result is motivation for an XGBoost vector comparator, not a transferable
CTBoost gain. [XGBoost vector-leaf report](https://xgboost.ai/2026/08/25/introducing-the-xgboost-vector-leaf-model)

Budget fairness requires reporting both rounds and physical trees: one scalar
XGBoost round can add one tree per class. A shared-tree method may need more
rounds even while ending smaller. Search quality and deployment size are
separate axes. XGBoost also documents reduced gradients for structural search
and full gradients for leaf values, but transplanting that structural statistic
would require a new calibration study and is not a recommendation here.
[XGBoost multi-output documentation](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html)

### 4. Make preparation reuse an explicit contract

The source shows two stages before native traversal. The sklearn prediction
mixin transforms raw input through the full fitted feature pipeline and builds
a Pool. `BuildPredictionBins` then quantizes model-active columns against the
stored schema. Existing prepared Pools avoid the first stage but not the
second. The phase-one schema-copy optimization does not eliminate either raw
categorical normalization or repeated binning.

The next API experiment should bind immutable input bins to a quantization-schema
fingerprint, feature order, bin width and row count. Keep prefix-specific active
features and Forbidden-NaN behavior explicit. A model mutation or mismatched
schema must reject or rebuild the prepared object. Never cache arbitrary mutable
arrays by Python object identity. For selective raw transformation, construct a
dependency graph from used transformed features to raw categorical combinations,
CTR inputs, text and embedding columns; simply dropping unused raw columns is
not safe. Preserve whole-input validation where the current public API promises
it.

XGBoost's `inplace_predict` is useful deployment precedent: its documented path
can avoid a separate DMatrix and does not cache prediction results. This supports
measuring wrapper/materialization costs separately; it does not establish that
CTBoost should bypass its quantization semantics.
[XGBoost prediction API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.inplace_predict)

### 5. Persistence audit: whitespace first, schema changes later

`ctboost/_serialization.py` writes both booster and estimator JSON with
`indent=2, sort_keys=True`. Replacing indentation with `separators=(",", ":")`
while retaining key ordering changes only whitespace. Existing JSON loaders,
versions 1/2/3, float spelling, category-key encoding, training state and tree
ordering can remain unchanged. `ctboost/_export.py` has a separate indented
predictor writer; changing native persistence does not silently change that
deployment contract.

This is the smallest backward-compatible experiment. Verify parsed-document
equality and exact reloaded raw scores/leaf IDs, including vector leaves,
categorical keys, nonfinite values where already supported, and baseline/prefix
semantics. Measure serialized bytes, process startup, parse/native restoration,
first prediction and steady-state prediction separately. A fresh process does
not imply cold filesystem caches. Do not describe file-size savings as warm
traversal speedups.

The larger structural waste is visible in `src/bindings/module_state.cpp`:
every node exports a 256-entry `left_categories` vector, including numeric nodes
and leaves. Both tuple and dictionary readers currently require that field and
its exact length. Omitting it would therefore require an explicit format change
or a reader migration, not merely a compact writer. A future version could omit
irrelevant zero routing tables and use typed contiguous arrays, while retaining
legacy readers and a legacy export option. There is already one shared
quantization schema in dictionary model state, so duplicating that optimization
would add no value.

The fixed phase-one digits artifact was 449,316,415 bytes in native JSON;
existing pickle used 62,511,815 bytes with exact reloaded predictions. The
predictor JSON remained 400,719,499 bytes and had small floating-point score
differences. Those are storage measurements for one model, not load-time or
cross-language-compatibility guarantees.
[Recorded export comparison](../improvements_20261016/accuracy/export_sizes/result.json)

XGBoost provides a useful format-design precedent: JSON and UBJSON encode the
same document structure, and stable model persistence is distinguished from
implementation snapshots. This supports considering a future binary container
after the whitespace experiment, without conflating a Python pickle with a
portable model specification.
[XGBoost model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)

### 6. A genuinely new, lower-priority learning experiment

Once the fixed portfolio is evaluated, freeze the learned split features,
thresholds, routing and leaf IDs, then optimize only existing ensemble leaf
values with L2 and loss checks. Use training rows only; the stopping set selects
zero, one or a few sweeps. Retain zero sweeps as a candidate. Never refit on
development/test labels. Keep constraints and multiclass sum-zero handling.
Compare against spending the same additional compute on more boosting rounds.

The May 2023 paper on regularized refitting of a GBDT leaf-indicator model offers
direct motivation, but does not validate CTBoost or all of its objectives.
[Enhancing Robustness of Gradient-Boosted Decision Trees through One-Hot Encoding and Regularization](https://arxiv.org/abs/2304.13761)
LightGBM's `refit` API is implementation precedent for updating existing leaf
outputs with a decay coefficient; its purpose and algorithm are not identical
to joint ensemble optimization.
[LightGBM refit](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster.refit)

The proposed CTBoost implementation would be a bounded experiment, not the
Regularized Greedy Forest algorithm: that method changes forest construction
and its guarantees cannot be imported. Fixed-topology refitting preserves the
original conditional tree decisions but changes scores, calibration, feature
importance interpretation and warm-start state. It must remain opt-in until
those contracts and independent quality checks are addressed.
[Regularized Greedy Forest paper](https://arxiv.org/abs/1109.0887)

## Benchmark review and release provenance

The proposed follow-up uses 12 metadata-selected cached tasks, four per problem
family, with eight fixed configurations per library and stopping-only selection.
This is an equal-trial development comparison. Equal trial counts are not equal
compute, identical capacity, or exhaustive tuning. Keep all configuration
results and failures, report selected models and cumulative search time, and
show a quality/latency/size frontier rather than a single unsupported ranking.

The revised CTBoost grid retains a clean quadratic-test 2 x 2 learning-rate/leaf
ablation in four arms, then four grouped-test/capacity candidates. Only matched
arms support individual causal attributions. XGBoost depth/L2 choices and
LightGBM leaf-count/L2 choices have different meanings; those differences should
be explicit. A scalar-only XGBoost grid also leaves the new vector method
untested. Add it only through a predeclared budget amendment, not by replacing
an underperforming arm after observing development results.

Train-fit all category vocabularies and transformations; use common raw rows,
labels and role indices, with library-native missing/category handling declared.
For binary tasks, align early stopping and configuration selection to AUC if
the primary score is 1-AUC. Explicitly retain the selected XGBoost iteration
range, since the native Booster prediction API otherwise uses the full model.
Preserve the original 14-task confirmation fold and official outer test. The
development panel has been seen in prior studies, so it is not independent
confirmation. Aggregate at dataset level, then report family-level summaries;
random seeds are repeated measurements of the same task, not extra tasks.

TabArena's primary paper emphasizes validation, hyperparameter budgets and
ensembling as material determinants of rankings. A capped single-model local
comparison must not be presented as an official TabArena result.
[TabArena, NeurIPS 2025, revised 3 November 2025](https://arxiv.org/abs/2506.16791)

One environment discrepancy was resolved during this review: the default
Python 3.10 `pip index` result showed XGBoost 3.2.0, while an existing Python 3.12
environment contained 3.4.1. The official PyPI page lists 3.4.1, released
15 August 2026, requiring Python >=3.12. GitHub's 3.4.2 patch, released
15 September, states that it differs only by restoring Python 3.11 support.
Thus compatible-release filtering can explain the older Python's result.
Pin and hash the actual imported package/native library; do not infer it from
the Python wrapper version or call 3.4.1 the absolute newest release.
[PyPI release](https://pypi.org/project/xgboost/3.4.1/),
[XGBoost 3.4.2 release](https://github.com/dmlc/xgboost/releases/tag/v3.4.2)

The raw categorical fast path under review only short-circuits exact Python
str/int/bool values and reads exact Python floats directly. Subclasses, NumPy
floating scalars and arrays retain the previous fallback; normalization still
calls the same string conversion. No correctness issue was found by source
review. Candidate speed and complete validation belong to the separate
inference experiment, not to this research note.

## Fixed-topology post-fit correction: source context

A global regularized update of existing leaf coefficients is distinct from
performing extra Newton steps while fitting each new boosting tree. Johnson and
Zhang's Regularized Greedy Forest work explicitly optimizes coefficients across
the ensemble, and its implementation guide describes leaf indicators as basis
functions. This supports investigating global coefficient correction, but its
greedy forest-growth algorithm must not replace CTBoost's conditional split
selection. Its reported benchmark gains do not transfer automatically to this
fixed-topology CTBoost experiment.
[Johnson and Zhang, revised June 2014](https://arxiv.org/abs/1109.0887),
[official RGF guide](https://github.com/RGF-team/rgf/blob/master/RGF/rgf-guide.rst)

LightGBM exposes refitting and setting existing leaf outputs, providing an
established implementation precedent for changing predictions without growing
new splits. Its documented decay-weighted refit is not the same objective as the
proposed joint ridge correction, so it must not be cited as an implementation of
that exact solver.
[LightGBM 4.7 Booster API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)

The ICML 2018 LeafRefit work also fixes tree structure when updating leaf values,
but does so to approximate training-sample influence. It supports separating
topology from leaf values, not a claim of improved generalization here.
[Sharchilev et al., 2018](https://proceedings.mlr.press/v80/sharchilev18a.html)

For this repository, a TRAIN-only residual ridge solve with STOP-only penalty
selection is a bounded hypothesis: it retains all native routes and can fold
coefficients into existing scalar leaves without increasing inference tree count.
The important CTBoost-specific constraint is replaying the original ordered
training representation before constructing the leaf-indicator matrix. Using
full-statistic inference CTRs on TRAIN would change the optimization problem and
can leak each training label into its own representation. Equal comparator
budgets, an unchanged-model incumbent and frozen development assessment remain
necessary. Positive ridge handles redundant constant/root-only leaf columns,
but equal penalty values across different leaf bases are not equal effective
regularization. Better training residuals alone establish no test improvement.

## Multiclass calibration without changing the trees

After the fixed-leaf regression experiment missed the development gate, the
next bounded hypothesis is positive scalar temperature scaling of multiclass
margins. Guo and colleagues study this single-parameter calibration method for
neural classifiers. Their evidence motivates the method, but does not establish
a CTBoost gain. A separate primary study by Leathart and colleagues evaluates
calibration of boosted trees and shows that probability quality can benefit from
postprocessing; its calibration trees are more complex than the scalar operation
proposed here.
[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html),
[Leathart et al., ACML 2017](https://proceedings.mlr.press/v77/leathart17a.html)

The proposed operation divides every class margin by the same positive
temperature before the existing softmax/log-loss calculation. It preserves
the winning class and all tree routes. A positive scalar also preserves binary
score ordering, so it cannot improve exact-arithmetic binary AUC; the four binary
tasks are deliberately left unchanged. The regression models are unchanged too.

The same eight temperatures, including identity, are offered to all three
libraries on each of the four multiclass tasks. Selection uses only the existing
stopping role, with all selections frozen before development scoring. Because
that role already chose the checkpoint and configuration, this is further
validation tuning, not independent evidence of calibration. A successful
development gate still requires the predeclared second-seed and one-shot
outer-test assessments. Native models remain intact; the selected temperature
is explicit artifact metadata and must also be applied in latency measurements.

## Binary fixed-topology logistic correction

The leaf-indicator literature also motivates a binary follow-up to the squared
loss experiment. The RGF implementation supports binary classification and
jointly updating existing leaf coefficients. That is motivation for the
coefficient step only: its forest-growth algorithm is not adopted here.
[Official RGF guide](https://github.com/RGF-team/rgf/blob/master/RGF/rgf-guide.rst).

The proposed local experiment keeps the original logit vector `f` as an offset
and minimizes `sum(logaddexp(0, f + Z*delta) - y*(f + Z*delta)) +
lambda/2 * ||delta||^2` on fitting rows, where `Z` contains the fixed native leaf
indicators. Positive ridge makes this objective strictly convex even with
duplicate constant-tree columns. The objective is logistic loss, while the
binary selection score remains AUC; improving the former does not guarantee
improving the latter. Correction shrinkage is a separately declared control.

`fixed_binary_leaf_refit.py` implements this as an experimental helper, with
independent finite-difference gradient and dense Newton reference checks. It
replays CTBoost's ordered training transform, folds corrections into native
logit leaves, checks every route and all non-leaf state, and verifies prediction
and pickle oracles for all three libraries. It adds no trees or model topology.
Convergence requires both a successful L-BFGS-B status and a fixed gradient
bound. Original training histories are preserved, so corrected artifacts are
for inference and cannot substantiate resumed-training claims.

The mandatory union refit has absorbed the stopping rows into fitting. They
must therefore not be reused as validation for its leaf correction. If this
hypothesis proceeds, select penalties/shrinkages on original TRAIN-fitted binary
models using the independent original STOP role, then transfer those fixed
controls to the union-trained models without reselection. This is an explicit
training-policy experiment with possible transfer error, not permission to
choose between models after inspecting development results.

## Dated continuation and corrections, 2026-10-16

Wave 10 completed all 96 binary candidate evaluations and all 12 mandatory
transfers successfully, including the native route, prediction and persistence
checks. Its 12-task development gate failed: CTBoost's median relative error gain
was -0.600% against XGBoost (5/12 wins) and -2.179% against LightGBM (4/12 wins).
Thus the logistic correction was implemented and tested, but did not establish
the requested overall accuracy advantage. See the immutable
[wave 10 development receipt](accuracy_wave10/development.json).

Wave 11's mandatory equal-weight 47/947 ensemble also failed the gate: -0.623%
against XGBoost (5/12 wins) and -0.714% against LightGBM (6/12 wins). It did not
open the predeclared second-pair replication or official test. Its two-model
serving cost must be compared with two-model baselines, separately from the
single-model inference optimizations. See the
[wave 11 development receipt](accuracy_wave11/development.json).

Wave 12 completed all 96 fresh threshold/learning-rate fits and 24 mandatory
union refits successfully. It narrowly passed the XGBoost side (+0.0731% median
relative error gain, 6/12 wins) but missed LightGBM (-1.6847%, 5/12 wins), so the
combined gate failed. The conditional seed947 check and official assessment
remained unopened. Increasing alpha while retaining the strict conditional
feature-choice path was therefore a tested hypothesis, not an established
overall solution. See the [wave 12 receipt](accuracy_wave12/development.json).

**Archive erratum:** the first archive incorrectly described the current hiva
profile fixture as three root-only trees. The actual fixture contains three
nontrivial trees, 147 nodes and 22 active transformed features out of 6,468,
derived from 1,617 raw categorical columns. The historical archive remains
unchanged; use the [profile continuation and final guarded results](profile/README.md)
for the corrected interpretation. This fixture's raw-prediction bottleneck was
preparation, not evidence that its trees were all constant.

There is also a distinction between observing root-only topology and identifying
its cause. The separate [eight-model wave 9 replay](root_stopping_replay/README.md)
verified 580 root-only trees among 1,061 roots as feature-test rejections, with
exact final native state except logging metadata, pipeline state, prediction
bytes and leaf IDs. That establishes a cause for these replayed models only;
wave 3 topology counts alone did not establish statistical stopping.

## Next bounded hypothesis: more stable round selection

Cross-validation of boosting's stopping iteration is established practice in
the official `mboost` implementation: it evaluates held-fold risk on an explicit
iteration grid and supports stratified folds. This motivates a round-selection
experiment without changing CTBoost's feature-before-cut rule.
[mboost cvrisk documentation](https://search.r-project.org/CRAN/refmans/mboost/html/cvrisk.html).
Validation variance itself can produce substantial model-selection overfitting;
cross-validation is not immune, and an optimized CV score is not an independent
performance estimate.
[Cawley and Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html).
TabArena uses stratified eight-fold validation and fold ensembles for its usual
models. That supports investigating validation policy, but its gains combine
selection and ensembling and cannot be assigned to a single full-data refit.
[TabArena, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf).

A smaller next hypothesis would freeze each existing tree policy, correction
penalty/shrinkage and temperature, then choose only a round multiplier from
0.5, 1 and 2 using five shared folds within TRAIN+STOP. Fit vocabularies,
quantization and ordered CTRs separately on each fold's training rows. Solve
each fixed correction separately on that fold's training rows at each candidate
round count; apply its frozen temperature before held-fold scoring. Score all
three checkpoints on all five folds, select by a predeclared mean of fold primary
errors, then perform one mandatory union refit. For binary AUC, pooling margins
from different fold models can introduce cross-model scale comparisons; the
mean of within-fold AUC errors is the clearer selection target.

The repository's `ctboost.cv` is not directly suitable for this protocol. It
slices an already-created Pool, so it cannot undo a globally fitted categorical
representation. With early stopping, it aggregates unequal, pruned histories
using NaN padding and `nanmean`; later checkpoints can therefore represent fewer
folds. An explicit raw-fold orchestration would avoid both issues. There is no
public CTBoost model-slicing operation suitable for corrected prefix models.
Independent fixed-round fits are the simpler auditable implementation: 360 fold
fits plus 24 final refits for the eight binary/regression tasks, or 540 plus 36
for all twelve tasks. A train-once/prefix alternative would require separately
validated tree, learning-rate, history and round-count truncation, not just a
prediction iteration limit.

An authorized TRAIN+STOP label-count check found that five stratified folds are
possible for every classification task. The minimum union class counts are 88
(blood), 353 (churn), 2,736 (coupon), 500 (Credit), 5 (anneal), 53 (website), 384
(splice) and 1,345 (SDSS17). Anneal has counts `[5,49,342,34,20]`: each validation
fold would contain one example of its rarest class and each fitting fold four.
Assert class presence in every fold; the rare-class uncertainty remains large.
No development or test values were read for this count check.

This would address round-selection variance, not erase historical selection of
tree controls, corrections or temperatures on the old STOP rows. Those controls
remain conditional on earlier adaptive work. Five-fold fitting uses 80% of the
union, close to the original TRAIN role's 83.3%; three folds would reduce that to
66.7%, a less attractive power shift for the conditional tests. Fixed numerical
ridge penalties also change effective regularization with sample size. This is
a proposal only, with no new CV fits or protocol implemented here.

**Later continuation on 16 October:** after wave 12 failed, the all-twelve-task
version was frozen as [wave 13](accuracy_wave13/plan.json), with 540 independent
fold fits and 36 mandatory union refits. The count grid is ceil-half, original
and double requested rounds, with duplicate indices retained. A fresh learner
seed137 check is conditional on the initial gate. The detailed source and
process-control review is in [review.md](review.md). This records the decision
to run the hypothesis; it is not a claim that CV has improved its outcome.

## Separate data-volume hypothesis: release only unused training rows

A metadata-only audit found unused rows beyond the existing 10,000-row cap in
four selected tasks. Keeping every current DEV row and the original 14-task
sealed confirmation fold excluded gives these possible union sizes:

| Task | Additional outside-cap outer-training rows | Current TRAIN+STOP | Possible expanded fitting union |
| --- | ---: | ---: | ---: |
| GiveMeSomeCredit | 90,000 | 7,500 | 97,500 |
| diamonds | 25,960 | 7,500 | 33,460 |
| SDSS17 | 42,035 | 7,500 | 49,535 |
| houses | 3,760 | 7,500 | 11,260 |

The other eight tasks have no outside-cap training rows. This cannot fix the
small blood/QSAR validation samples. All libraries would receive the same
expanded rows under a mandatory policy; extra conditional-test power would be
one possible mechanism, not a CTBoost-only entitlement or a guaranteed gain.
Vocabulary, quantization and effective regularization would also change, so
this should be separate from a CV experiment.

Verified train-only cache receipts exist for Credit, diamonds and SDSS17.
The current remaining-panel loader used for houses opens the full OpenML cache
before slicing outer training; it should not be reused under a protocol that
forbids reading official-test values. Before any such experiment, freeze an
explicit row-identity manifest and a train-only extraction path that excludes
DEV, original confirmation and official test. Do not equate all outer-training
rows with rows available for fitting. This audit opened metadata only, not the
full feature/label caches, and no cap-release experiment was implemented.

### Training-only access audit and stricter contingent rule, 16 October

The houses cache was inspected through Parquet schema/footer metadata only,
using installed PyArrow 24.0.0. `dataset_46934.pq` contains 20,640 rows in one row
group and nine physical numeric columns. There is no stored row-identity field;
pandas metadata describes a RangeIndex from 0 to 20,640. A single-file Arrow
`Scanner.take` could return only explicitly authorized positional rows, but the
version 24 C++ implementation first scans record batches internally and then
gathers selected rows. Consequently, excluded official-test values may be
decoded internally even when none are returned to the experiment. This is a
logical selection mechanism, not a guarantee that test values are never read.
[Arrow24 TakeRows implementation](https://github.com/apache/arrow/blob/apache-arrow-24.0.0/cpp/src/arrow/dataset/scanner.cc#L463-L542),
[Scanner.take API](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html#pyarrow.dataset.Scanner.take).
No real row extraction, feature/label access, full `read_parquet`, or new score
assessment was performed for this audit.

The existing literal no-test-values boundary therefore takes precedence:
**houses must retain its frozen model in any next cap-release study**. The
contingent policy is based on cache provenance and row availability, not on
quality results: expand a task only when it has both unused outside-cap
outer-training rows and a verified cache containing official training rows
only. Currently that admits GiveMeSomeCredit, diamonds and SDSS17. The other
nine tasks, including houses, retain their frozen policies. All three libraries
receive identical authorized rows and mandatory treatment within each task.

For these three older-panel tasks, reconstruct the original capped subset with
the recorded seed 47 stratified selection, verify historical role/index hashes,
and add only outside-cap rows to the existing TRAIN+STOP union. Exclude every
existing development row and the original sealed confirmation fold; do not
open `confirmation.pkl`. Freeze the manifest and source/runtime identities
before fitting. If authorized after CV failure, a single mandatory refit per
eligible task/library would mean nine new fits and 27 carried models, with the
same twelve-task assessment. Keep selected controls, requested rounds and
postfit controls fixed, and never choose between old and expanded models using
development scores. This is a proposed data-volume policy, not an executed
experiment or a claim that CTBoost will benefit more than its comparators.

The proposed reconstruction has a narrower information boundary than an assertion
that confirmation labels are never accessed: it must read the verified full
official-TRAIN cache's labels to reproduce the historical stratified cap and
class vocabulary. Those labels include historical development and confirmation
rows. Their use is confined to reproducing those already fixed assignments and
label IDs; their rows remain excluded from new fitting, feature normalization,
correction solves and confirmation scoring. `confirmation.pkl` remains unopened.
This does not expose official-test values because the source cache itself is
verified to contain official training rows only. No such reconstruction or
cache-value read has been performed by this read-only audit.

This remains a stronger next hypothesis than further raising alpha to 0.9/0.95:
the exact wave 9 replays established statistical rejection, but their root
p-values were below 0.7 and wave 12 already tested 0.8 on new training paths.
Those old traces cannot establish a remaining benefit from 0.95. Such a threshold
would preserve strict minimum-p feature selection before cut search, yet admit
almost all nondegenerate candidate features. The primary literature instead
supports sensitivity to sample size and the match between association statistic
and signal shape; it does not establish a universal predictive benefit from
near-one stopping thresholds.
[Learning curves and changing model rankings](https://jmlr.org/papers/v4/perlich03a.html),
[Score-based partitioning power study](https://arxiv.org/abs/1906.10179).

### Separate houses ARFF feasibility note, 16 October

This does not change the frozen three-cache wave 14 policy or its houses carry.
The cached OpenML description also identifies raw ARFF file 22125245 and MD5
`5edd572964b0653b789bfc5e5c0293a8`. Its feature metadata specifies nine ordered
numeric columns, each with zero missing values. Unlike the Parquet batch
decoder, a narrow binary ARFF selector could hash all source bytes while decoding
and numerically parsing only predeclared authorized row lines. The primary Weka
format specification describes one instance per line and a separate typed
header. This supports a selective parser, subject to source identity, header,
row-count and original authorized TRAIN/STOP parity checks.
[Weka ARFF specification](https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/).

The proposed contract keeps skipped instances as opaque bytes, with only
predeclared structural checks. It never text-decodes, tokenizes, converts or
includes skipped values in exceptions. It therefore does not claim to validate
numeric values in the entire file. Only the authorized rows would be parsed as
nine finite numbers, after source hashes and row IDs are fixed. The source
SHA256 is currently unknown and would need a byte-only acquisition/hash stage
before parsing; the known MD5 alone is not a new payload verification.

The root agent reports a HEAD-only availability check, with no response body
read: HTTP 200 at `https://www.openml.org/data/v1/download/22125245/houses.arff`,
Content-Length 1,446,365, and no Content-Encoding, ETag or Last-Modified response
headers. This establishes endpoint availability only. No real ARFF payload was
downloaded, opened or parsed in this review; an inactive synthetic parser draft
is separate from the wave 14 experiment.

### Prospective multiclass fixed-leaf correction, 16 October

Following the failed wave 15b gate, a bounded multiclass coefficient experiment
is authorized for preparation. This is a new hypothesis, not a demonstrated
accuracy gain. Regularized Greedy Forest motivates jointly correcting existing
leaf coefficients instead of changing only the newest tree: its primary paper
studies fully corrective forest learning, and its implementation guide explicitly
describes updating all existing leaf weights. Its documented multiclass path is
one-versus-all; the joint fixed-temperature softmax correction proposed here is
our extension, not a claim established by that implementation or a reproduction
of RGF's structure-learning algorithm.
[Johnson and Zhang, revised June 2014](https://arxiv.org/abs/1109.0887),
[official RGF guide](https://github.com/RGF-team/rgf/blob/master/RGF/rgf-guide.rst).

Hold every tree, tested feature, split, missing-value route, physical leaf ID,
class support and requested boosting count fixed. Let F denote native raw margins
and A the existing leaf-indicator operator. At inherited temperature T > 0, solve
`(sum CE((F + A delta)/T) + lambda ||delta||^2/2) / n` for lambda > 0.
This is convex in the leaf corrections, with a unique coefficient optimum from
positive ridge despite redundant leaf indicators or common-logit shifts. The
gradient must include `1/T`; temperature is neither retuned nor applied twice.
This correction cannot discover missing interactions or recover useful splits
that the conditional tests rejected. It preserves the feature-test-before-cut
construction and changes only outputs on the already established partitions.

The proposed equal-count grid is lambda {1, 10, 100, 1000} crossed with correction
shrinkage {0.25, 1}, plus the unchanged identity option, for all three libraries
on all four multiclass tasks. Fresh TRAIN anchors use the inherited controls and
requested rounds. SDSS17 adds ascending outside-cap rows after original TRAIN
and excludes original STOP; the other three tasks use original TRAIN. Build
vocabularies and CTBoost ordered training transforms using only those exact
fitting rows. Select the correction on original STOP, with identity retaining
exact ties. After all twelve choices are frozen, recompute their fixed controls
on the existing wave 15b union/expanded models and their exact fitting order.
The other 24 models are mandatory carries. There is no DEV-based choice between
the original and transferred correction. Conditional seed 457 must preserve all
four expanded sources, including houses, and the official test remains gated.

After folding, serving uses an ordinary native model with the same topology,
tree count and traversal work. Exact route checks, ordered-transform replay,
independent margin/probability oracles and persistence checks are required;
float32 coefficient rounding means corrected predictions are compared against
fixed numerical tolerances as well as an analytic roundoff bound, rather than
claimed identical to an ideal float64 solution. Historical training statistics
remain unchanged, so the folded model is an inference artifact, not a checkpoint
for resumed training.

Equal candidate counts do not equalize parameter count, effective regularization
or training cost. Vector trees expose a coefficient for each supported class at
each leaf; scalar trees retain their existing class support. The stated lambda
penalizes sum loss and is then divided by n, so its effective strength changes
with fitting-set size. Inherited temperatures also change the scale seen by the
loss. Sparse design storage is proportional to rows times physical trees, and
optimizer history grows with coefficient count; preflight estimates and a process
memory/time watchdog must precede real solves. Reused STOP and adaptive DEV
results cannot establish general superiority or statistical significance.

### Prospective binary/regression correction-control refresh, 16 October

Wave 16 completed its fits, coefficient evaluations and transfers but failed the
full-panel gate. The next authorized preparation addresses a specific policy
mismatch: binary and regression ridge/shrinkage choices were selected on older
training models, then transferred through changed round counts and expanded
fitting sets. Current leaf indicators, residuals and the effective strength of
the sum-loss penalty can differ from those used to choose the controls. The
same fully corrective motivation cited above supports testing this mismatch;
it does not predict which library will benefit or establish a new result.

A true replacement needs **48 fresh tree fits**, not a second correction folded
onto the current parent. Source inspection of `fit_expansion` and the round-CV
runner shows that correction precedes persistence; there is no guaranteed saved
uncorrected current union baseline. Fit 24 clean TRAIN anchors and 24 mandatory
clean union models at current fixed parameters, requested rounds and seed 47.
Disable both inherited binary and regression correction fields before fitting.
The identity option means this clean native model. Select the unchanged four
lambda values crossed with two shrinkages on original STOP, then apply only the
newly frozen controls to the clean union model. Compare its schema, ordered
pipeline state and non-leaf topology against the corrected parent where
deterministic; a correction must not silently change the tree structure.

For Credit, diamonds and houses, anchor fitting rows are original TRAIN followed
by the already authorized outside-cap rows, excluding original STOP. Other
binary/regression anchors use original TRAIN. Full fitting order remains TRAIN,
STOP, outside-cap. The 12 wave-16 multiclass models are mandatory exact carries;
any conditional seed 569 policy must preserve their fixed multiclass correction
and temperature, plus all four expanded sources. The same eight candidates and
identity are offered to every library on every active task. This means 96
selection solves, 192 nonidentity evaluations and 24 identities, followed by at
most 24 fixed union correction solves. No per-task DEV-based fallback is allowed.

This isolates stale postfit regularization at smaller scope than another
five-fold, three-round-profile study: 48 new tree fits versus 384 for eight
tasks or 576 for all twelve, including final refits. Fit counts are not wall-time
equivalence. The STOP roles remain reused, and current tree controls already
incorporate prior STOP/CV selection. Binary training optimizes logistic loss
while selection uses AUC; fixed partitions cannot recover missing interactions.
Larger-data round selection remains a distinct later hypothesis, not another
axis to add to this correction experiment. Freeze solver, sparse-memory and
process-resource limits before any real solve, and preserve failures.

### Contingent recommendation after the scalar refresh, 16 October

This is read-only planning while wave 17 runs, not another activated experiment.
The strongest next quality hypothesis, if needed, is **round-count selection
using the current fitting data and current fixed correction policy**. Wave 13
selected lengths before the four data expansions and before the latest leaf
correction choices. The mismatch is identified by protocol history, not by
choosing favorable task outcomes. Training-size changes can change model
comparisons; that general learning-curve observation does not establish the
best CTBoost length or guarantee a gain. See the primary
[Perlich, Provost and Simonoff study](https://jmlr.org/papers/v4/perlich03a.html).

| Priority | Fixed prospective policy, applied equally to all three libraries | New native fits | Main tradeoff |
|---|---|---:|---|
| 1 | All twelve tasks, five-fold CV of ceil(half)/base/double current requested rounds; one mandatory final fit per task/library | 540 fold + 36 final = 576 | Directly tests the stale-count mismatch; much more work on expanded rows, but serves one model |
| 2 | Mandatory half learning rate and double current requested rounds, with fixed correction controls re-solved | 36 | Small experiment count, approximately twice requested trees; mixes optimization path and capacity rather than selecting appropriate length |
| 3 | Fixed equal-weight two-seed average, using the current parent plus one fresh component per task/library | 36 additional | Changes variance, requires two serving components; does not directly repair systematic count/regularization mismatch |

For option 1, raw fitting rows are only the existing authorized union/expanded
sources, excluding every historical DEV, confirmation and official-test row.
Freeze common fold IDs for all libraries and verify classification training
class coverage and binary held-fold class coverage. Each independent fit has
fold-local vocabulary and ordered transforms; re-solve only the inherited fixed
leaf correction on that fold's TRAIN rows, then score the actual corrected
native model with inherited temperature applied once. Do not score an
uncorrected prefix and deploy its corrected counterpart. Select mean fold
primary error, prefer fewer requested rounds on exact ties, keep integer
duplicates explicit, freeze all 36 choices, then perform all 36 mandatory final
fits with no old/new fallback. A conditional fresh-seed check would add 36 fits;
official assessment never adds fits.

This CV score remains a tuning statistic: inherited controls already used the
old STOP/CV roles, which are now distributed among folds. It is not nested CV
or an unbiased estimate. AUC should be averaged by fold rather than pooling
incomparable fold margins. The final model uses a larger training set than each
fold, so even a five-fold optimum has a residual sample-size mismatch. The
previously frozen repeated-DEV caveat and one-shot official boundary remain.

Resources must be settled **before** freezing this study. At the already fixed
diamonds LightGBM count of 1,195 and 33,460 authorized fitting rows, a double-count
final design has 79,969,400 leaf entries. The scalar helper's 56 bytes per entry
across named leaf/CSR/construction budgets alone is 4,478,286,400 bytes, exceeding
4 GiB before maps or solver vectors. The multiclass vector budget also grows
with count and can exceed its present 2 GiB limit. Therefore the old budgets
cannot simply be inherited for this grid. Use one worker and a predeclared,
hardware-feasible larger named-array/RSS budget, or choose a uniformly bounded
upper count multiplier before any outcome is observed. A post-failure budget
increase is not part of this proposal. The five-fold grid costs roughly fourteen
current full-fit row-round units before final fitting, plus 0.5–2 for the chosen
final count, ignoring integer rounding. That is an arithmetic workload proxy,
not a runtime estimate; coefficient-solver convergence and native model sizes
can dominate. Three folds would reduce the same grid to 324 + 36 = 360 fits but
increase the fold/full training-size mismatch. Restricting to the four expanded
tasks by a predeclared data-change rule would cost 180 + 12 = 192 fits and carry
24 artifacts; it would leave the correction/count mismatch on other tasks
untested, so the all-twelve policy is cleaner if resources permit.

Option 2 is plausible because shrinkage and number of boosting stages interact,
as stated in the official [gradient-boosting implementation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).
However, preserving learning-rate times round count is not an identity: changed
residuals can change conditional test acceptance, selected features and cuts.
The earlier half-rate experiment already explored this axis on smaller data;
new expansion makes a fresh test legitimate, not a claim of novelty. Choosing
between base and half-rate on original STOP would instead need 72 fresh TRAIN
fits plus 36 union fits and would reintroduce the STOP-variance problem. Do not
add a rate axis to the round-only CV without explicitly accepting the larger
budget and less focused hypothesis.

Option 3 is lower priority. [Breiman's primary bagging report](https://www.stat.berkeley.edu/~breiman/bagging.pdf)
attributes benefits to predictor instability and documents that stable methods
may not improve. Changing a learner seed on identical rows is not bootstrap
bagging; some current controls can yield identical components. Averages need
fixed weights, common class order, stable probability arithmetic, temperature
once per component, and a predeclared paired-seed validation policy. No selecting
weights or components from DEV, and no claiming a latency-neutral gain. The
completed two-seed study is preserved; any revisit must identify the changed
data/policy rather than silently retrying seeds until success.

Separately, reproducible implementation-correctness findings take precedence
over another expensive quality search. The native float32 target-sum
cancellation finding is a legitimate precision issue, not evidence that it
caused the current twelve-task gap. An objective-aware regression CTR mode and
more accurate accumulation require dedicated synthetic and persistence tests,
an isolated new runtime, and a separately frozen comparison. They must not
silently alter the immutable accuracy runtime or any prior study.

### Numerical correctness audit: CTR accumulation and native log loss, 16 October

The source audit identifies two implementation issues independently of the
development ranking. Float32 target sums can lose representable cancellation
terms: for labels `[1e8, 1, -1e8]`, sequential float accumulation produces zero
instead of one. Wider prior, running and persisted total accumulators address
that precision loss; they do not promise exact summation for arbitrary dynamic
range. A separately versioned pipeline format is necessary because the frozen
ordered-CTR correction helpers replay loaded fitting state exactly. Old formats
must retain their historical accumulation and refit semantics. No measured
accuracy improvement is attributed to this fix.

Native classification metrics consume raw margins, but current binary log loss
first computes a float sigmoid and both binary/multiclass loss clip probabilities
at `1e-12`. This is not merely protection from overflow: binary margin `+17`
against label zero rounds its sigmoid to one and receives loss about 27.63,
whereas the mathematical loss is about 17. A wrong margin of 40 is capped near
27.63. Multiclass target gaps have the same cap. Stable logit-space formulas
avoid both premature probability rounding and the artificial cap:
`max(z,0) - y*z + log1p(exp(-abs(z)))` for binary and
`(max_logit - target_logit) + log(sum(exp(logit-max_logit)))` for multiclass,
with float inputs promoted to double before subtraction. The binary formula is
already used by this repository's leaf-estimation backtracking. The official
[BCE-with-logits documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
explains the numerical advantage over separate sigmoid and cross entropy;
[cross-entropy documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
states the raw-logit loss being evaluated.

Correcting these metrics affects reported histories and potentially early
stopping, not objective gradients or conditional feature selection. The frozen
benchmark's multiclass scorer already uses stable shifted log-sum-exp, so a
selection/scoring alignment issue exists in principle. Its actual checkpoint or
portfolio impact has not been measured. Old saved predictions are unaffected,
but ordinary warm start can retain historical clipped scores; cross-version
identical stopping must not be claimed. This audit does not authorize altering
any frozen experiment or interpreting a metric change as a demonstrated quality
gain.

The [synthetic metric audit](inference/metric_numerics_audit/README.md) records
12 binary and nine multiclass evaluations against the immutable accuracy
runtime, with no real data or model access. The implementation decision is to
**defer** the metric correction, rather than ship mixed old/new histories with
only a warning. A future persisted arithmetic version must flow through both
native and Python warm-start metric evaluation. Full historical rescoring is
not a general solution because DART's old rescaled ensembles are unavailable.
Current fixed-count, externally scored CV does not depend on changing native
early-stopping metrics.

### Prospective CV resource qualification, 16 October

The earlier larger-budget suggestion is not authorization for a 16 GiB worker.
The parent's resource observation was 31.92 GiB physical memory with only
12.56 GiB available while wave 17 and a small native build overlapped. A later
CV protocol must first use a quiet no-build memory baseline and completed
wave-17 peak-RSS evidence to choose its limits. A 6 GiB named-array estimate
with an 8--12 GiB whole-process limit is a candidate range to validate, not a
frozen policy or a demonstrated sufficient bound. Use one worker and a
predeclared system-memory preflight before every dispatch: enough available
memory for the prospective child allowance plus fixed system headroom, with a
bounded recorded wait if temporarily unavailable. Resource failures remain
terminal; do not increase limits or selectively shrink a failed library's grid
after observing the case. No new CV source, fit or freeze follows from this
resource note.

### Inactive final-leaf-weight hypothesis, 16 October

The [separate primary-source and mathematical note](absolute_leaf_ridge_hypothesis.md)
distinguishes the current penalty on changes from native predictions from a
penalty on final leaf contributions. The latter could directly shrink existing
large weights, but requires a common bias convention, effective contribution
extraction, explicit treatment of duplicate/unobserved leaf columns and its own
prospective protocol. Cui et al.'s equation 10 also refits an intercept and their
algorithm removes duplicate features; the suggested fixed-bias physical-leaf
variant would not replicate their method. This remains inactive research after
the nearer full-current-data count-CV proposal, with no new fits or score reads.
