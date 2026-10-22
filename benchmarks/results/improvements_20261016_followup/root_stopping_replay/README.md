# Root-only tree cause audit, 16 October 2026

All eight fixed wave 9 CTBoost diagnostic replays matched their saved models.
Of 1,061 roots, 580 were leaves because the best eligible feature's recorded
p-value exceeded the configured alpha. No root-only tree in this replay set
was attributed to cut infeasibility, nonpositive gain, an ineligible feature,
a structural limit, an empty partition or an ambiguous rounded p-value.

This establishes the stopping cause for these eight **wave 9 union models**.
It does not retrospectively identify causes in the earlier wave 3 models.
The earlier topology-only diagnostic counted `len(tree.nodes) == 1`; that
observation alone cannot distinguish the stopping branches.

## Fixed scope and identity

The [plan](plan.json) was frozen before fitting, SHA256
`b72efe642ea5c02ee16bf00f53b923e3795570b6ae5df2b6dee73430faf947f7`.
The source is [`root_stopping_replay.py`](../../../root_stopping_replay.py).
Its parser and role-access guard passed 12 focused tests before freezing.

Each replay used original TRAIN followed by STOP, the original selected
parameters, seed 47, requested round count, and any already-selected regression
leaf correction. No early stopping, new choice, development data or test data
was used. These eight diagnostic fits are separate from the HPO portfolios.
The exact frozen Python 3.12 runtime and native SHA256
`c5ea7e026942c3f1190ca5511fbf74f353bfeb08e7771f76bc5f6c1ca3e4d197`
were verified. CPU 8 and one-thread settings bound resource use; elapsed times
are not a benchmark.

The sole logging change was `CTBOOST_PROFILE=1`. Constructor resolution stores
that choice as native `verbose` metadata. Every replay required equality of
the entire final native state after removing only that field, equality of the
fitted feature-pipeline state, exact prediction bytes on union/original TRAIN/
original STOP, and identical union leaf indices. Fixed regression corrections
were recomputed with the existing ordered-transform and native-fold checks.
Prediction comparisons use the inference transform, while correction fitting
uses replay-verified ordered training transforms. All eight identity checks
passed; no mismatch was waived.

## Results

Every root-only count in the following table is a clear feature-test rejection.
The quantiles describe **all root** stopping p-values, including roots that split.
Native logs print six significant digits; the table rounds further for reading.
Full quantiles and log/state/prediction hashes are in [summary.json](summary.json).

| Task | Alpha | Root-only / total | Minimum p | Q25 | Median | Q75 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blood-transfusion | .2 | 2 / 15 | 2.40322e-7 | .00175447 | .0253064 | .120709 | .228512 |
| churn | .05 | 59 / 137 | 2.02581e-44 | .00132014 | .0332945 | .0847163 | .242978 |
| coupon | .2 | 1 / 72 | 1.31432e-30 | .0005164405 | .0275658 | .0797480 | .264910 |
| GiveMeSomeCredit | .05 | 96 / 164 | 1.92252e-161 | .00223011 | .0778855 | .160577 | .458899 |
| QSAR fish toxicity | .05 | 167 / 192 | 1.38324e-27 | .0821489 | .158000 | .245651 | .699723 |
| Fiat 500 | .05 | 0 / 58 | 5.37135e-124 | 1.14875e-85 | 3.02276e-41 | 1.94530e-14 | .000107578 |
| houses | .5 | 59 / 167 | 0 | 3.76729e-86 | .0101230 | .518250 | .518250 |
| diamonds | .05 | 196 / 256 | 0 | .0624039 | .133025 | .225338 | .574557 |

Houses uses full sampling and ends with 59 consecutive root-only trees whose
printed p-value is .51825. This is compatible with a persistent significance
gate: a constant regression-tree update shifts all residuals together, and a
centered association statistic is invariant to a common residual shift in
exact arithmetic. The trace does not establish that admitting further splits
would improve generalization. All maximum observed root p-values were below
.7, but a larger alpha would change subsequent trees and residuals; this is
not a counterfactual simulation of an alpha .8 fit.

Raw native logs, parsed per-iteration records, replay models and identity receipts
are retained under `.tmp/improvements-20261016/followup/root-stopping-replay/`.
Their SHA256 identities are recorded in the public summary and per-case receipts.

## What the source permits and the logs resolve

`tree_build_cpu.cpp` can return a leaf before search because of minimum row count,
depth or leaf-count limits. Search can find no eligible feature, reject the best
feature's test, or pass the test but find no admissible positive-gain cut.
`tree_split_scoring.cpp` separately checks positive child weight, minimum leaf
weight, minimum child Hessian, gain after L2 and monotone feasibility. A final
empty-partition guard also exists. The strict default path does not try a
different feature merely because the minimum-p feature has no valid cut.

All eight frozen policies use this strict minimum-p feature-before-cut path:
no random strength, feature weights, first-use penalties, monotone constraints or
interaction constraints activate ranked gain search. Their positive depth,
unlimited default leaf count and ordinary row counts rule out those root limits.
The actual traces additionally rule out the remaining root-only alternatives in
this particular replay set. A successful root may still encounter those other
limits deeper in the tree; this study only classifies roots.

`LogNodeSearch` has no explicit reason enum, but logs the chosen feature,
adjusted stopping p, tested-feature count, split-valid flag and gain. The parser
joins each depth-zero event to its following `tree_build` boundary and verifies
all iteration counts against exact saved topology. Rounded values whose interval
could straddle alpha remain unresolved; none occurred here. A passed test with
an invalid split cannot be decomposed further into minleaf, Hessian and gain
causes from these logs alone. A printed p of one would not by itself distinguish
a null association from degenerate gradient variance.

## Contingent next hypotheses

Changing alpha to .8 or .95 while leaving the strict path intact preserves
feature testing, p-value ranking, the chosen feature, and subsequent cut search
at each fixed node. Both remain below one and can reject a p-value of one.
They materially weaken statistical stopping; they should not be described as
the same regularization or as supported new defaults. In these studies,
`feature_test_adjustment='none'` means the minimum unadjusted feature p is used,
so alpha is not a familywise node-error guarantee.

The original conditional-inference paper separates variable selection from cut
selection and treats alpha as a tree-size control. Its evidence concerns that
framework, not optimal alpha in CTBoost's adaptive boosting sequence.
[Hothorn, Hornik and Zeileis, 2006](https://www.zeileis.org/papers/Hothorn%2BHornik%2BZeileis-2006.pdf)

There is direct implementation precedent for weaker stopping in boosted
conditional trees: official `mboost::blackboost` defaults use quadratic
statistics, raw-statistic selection, `mincriterion=0` and depth 2. This supports
studying the interaction between tree stopping and ensemble regularization.
It does **not** validate CTBoost alpha .8/.95: that raw-statistic mode differs
from CTBoost's p-value ranking and removes the p-value gate altogether, which
is outside the preserved design constraint here.
[Official blackboost documentation](https://search.r-project.org/CRAN/refmans/mboost/html/blackboost.html),
[Model-based Boosting 2.0, 2010](https://www.jmlr.org/papers/v11/hothorn10a.html)

The official CTree controls separately expose significance level, minimum
node/leaf weight and depth. These distinctions support a matched alpha ablation,
not a simultaneous change in every capacity control.
[Official ctree controls](https://search.r-project.org/CRAN/refmans/partykit/html/ctree_control.html)

A defensible contingent experiment is a separately frozen .5/.8 comparison at
fixed sampling, feature statistic, capacity and leaf optimizer, with STOP-only
selection and balanced comparator budgets. Alpha .95 is a more permissive
stress arm, not the first recommendation. Even mathematically non-vacuous
thresholds can offer little practical null protection with many unadjusted
features: under an illustrative independent-uniform fixed-node null, the chance
that at least one of six p-values is below .8 is 1 - .2^6 = .999936. That formula
is not CTBoost calibration evidence; real tests are dependent and boosted
residuals are adaptive. No generalization guarantee follows.

The separately predeclared two-seed ensemble instead keeps every tree learner
unchanged and targets variance. It cannot repair a deterministic inability to
split, and equal averaging can still worsen AUC or lose to the better component.
It needs two native models per prediction, so storage and raw/prepared inference
must be measured. Its two seed pairs remain observations on the same datasets,
not additional independent tasks. Neither hypothesis should be chosen using
the sealed test, and neither is a production default change in this audit.
