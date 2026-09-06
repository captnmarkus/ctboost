# Safeguarded leaves and joint multiclass tests

Version 0.1.59 adds three optional controls. The existing defaults remain
`leaf_estimation_backtracking=False`, `multiclass_leaf_solver="diagonal"`,
and `multiclass_feature_test="single"`.

| Control | Effect | Supported training |
|---|---|---|
| `leaf_estimation_backtracking=True` | Check scalar leaf updates against their actual regularized loss | Built-in RMSE/SquaredError and LogLoss, CPU |
| `multiclass_leaf_solver="full"` | Fit all class scores together with a coupled softmax Hessian | Built-in multiclass/softmax, CPU, 3–32 classes |
| `multiclass_feature_test="joint"` | Test each feature against all class-gradient coordinates | Built-in multiclass/softmax, CPU, 3–32 classes |

These controls support ordinary CPU training. Scalar backtracking and the full
multiclass leaf solver support fractional sample weights; joint feature testing
requires integer frequency weights. All three controls reject distributed
training, GPU training, and callable objectives explicitly.
Other existing objective and constraint restrictions still apply. Scalar
backtracking and the full multiclass solver are separate choices: the full
solver always performs its own loss checks.

## Scalar leaf backtracking

```python
from ctboost import CTBoostClassifier

binary_model = CTBoostClassifier(
    leaf_estimation_iterations=3,
    leaf_estimation_backtracking=True,
    random_seed=42,
)
binary_model.fit(X_train, y_train)
```

CTBoost first builds the conditional-inference tree as before. It then checks
the initial Newton leaf proposal and each additional refinement against the
weighted training objective plus the existing L2 leaf penalty. Overshooting
steps are repeatedly halved; if no finite candidate with non-increasing loss is found, the
previous values are retained. Monotone projection precedes acceptance, and
the check uses the stored float leaf values.

The loss check evaluates **unshrunk leaf increments**, matching existing leaf
estimation. Learning-rate shrinkage and DART scaling happen afterward. It
does not guarantee improvement in validation loss, a chosen evaluation metric,
or every final ensemble update. `leaf_estimation_iterations=1` also checks the
initial proposal when backtracking is enabled.

## Coupled multiclass leaves

```python
multiclass_model = CTBoostClassifier(
    multi_strategy="multi_output_tree",
    multiclass_leaf_solver="full",
    leaf_estimation_iterations=3,
    random_seed=42,
)
multiclass_model.fit(X_train, y_train)  # 3–32 classes
```

The full solver includes the off-diagonal terms of the softmax Hessian. A
sum-zero representation removes the unidentifiable common offset; damping,
L2 regularization, and backtracking stabilize the solve. If `max_leaf_weight`
is set, the proposed vector respects both the box bound and sum-zero constraint
to floating-point precision.

The solver changes only values on the selected tree structure. It works with
both `one_output_per_tree` and `multi_output_tree` storage. It supports
1–5 leaf estimation steps; the default diagonal multiclass solver supports
one. With identical inputs, both storage layouts fit the same class scores.

## Joint feature selection

```python
multiclass_model = CTBoostClassifier(
    multiclass_feature_test="joint",
    feature_test="grouped",
    feature_test_bins=8,
    feature_test_adjustment="bonferroni",
    multiclass_leaf_solver="full",
    leaf_estimation_iterations=3,
    multi_strategy="multi_output_tree",
)
```

The default shared multiclass tree tests one gradient coordinate. The joint
option uses every coordinate and the effective rank of the weighted response
covariance. For softmax, the redundant all-ones direction is removed without
choosing a reference class. The feature statistic is invariant to class
permutation up to numerical precision. Its covariance decomposition is reused
across features within a node.

The statistical test still precedes cut-point optimization. The selected
feature's cut is scored by the existing scalar structure target; this release
does not introduce a joint multiclass cut objective. Existing constrained or
penalized ranked-feature search rules are retained. Numeric grouped tests
combine adjacent bins by node weight, keep missing values separate, and leave
the original cut candidates available. Categorical bins are not grouped.

Joint training requires finite, nonnegative integer frequency weights, including
the effective product of sample and class weights. Zero weights are allowed.
The check runs before sampling and again on each tree's sampled weights.
Bernoulli and Poisson sampling preserve integer frequencies; Bayesian sampling
with `bagging_temperature > 0` is rejected. Fractional effective weights from
sample weighting, class weighting, or automatic class balancing are rejected.
The default `single` test and the leaf solvers retain their existing weight
support. Combining a leaf solver with the joint test applies the joint test's
integer-weight restriction.

These are asymptotic chi-square tests. Integer weights have a literal
frequency-expansion interpretation, and rescaling weights can change the
p-value. Small, sparse, heavily weighted, or repeatedly selected nodes do not
acquire an exact type-I guarantee. Per-node Bonferroni adjustment does not
establish family-wise error control for an entire fitted boosting ensemble.

The diagnostic statistic API retains the fractional-weight approximation for
auditing. In 1,000 fixed-feature null trials (480 rows, three classes, four
feature bins, seed 159), the unweighted and equivalent integer-frequency
representations both rejected 4.7% at a 5% threshold. Independent fractional
weights drawn uniformly from 0.25 to 2 instead rejected 15.7%. That inflation
is why fractional weights are excluded from joint training. This simulation
does not establish calibration for arbitrary data or adaptive boosted trees.
Reproduce it from the source checkout with
`python -m benchmarks.multivariate_statistics_calibration --repetitions 1000 --seed 159`.

## Persistence and evidence

The options round-trip through model state, Python persistence, scikit-learn
parameters, and snapshots. Older states default to the original behavior.
Prediction formats are unchanged: the new controls affect training, while
exports store the resulting tree structure and leaf values.

Use held-out validation data to compare these options on your task. The
published [0.1.58 TabArena results](../benchmarks.md) remain evidence for that
version and configuration; they do not measure these new algorithms.
