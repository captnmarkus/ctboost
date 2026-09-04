# Conditional split statistics

CTBoost keeps feature selection and split-point selection as two separate
steps. At each node it first tests every eligible feature for association with
the current gradient scores, selects by the raw feature-level p-value, applies
the `alpha` stopping rule, and only then searches that selected feature's full
histogram for the gain-optimal cut. The options on this page do not replace
that conditional-inference structure.

The ordinary unconstrained path is strictly minimum raw p-value followed by a
gain search only within that feature. CTBoost's pre-existing advanced search
path is slightly broader: monotone/interaction constraints, nonzero
`random_strength`, feature weights, or first-use penalties iterate the
raw-p-ranked set of significant features and choose the best feasible adjusted
gain. Grouping and multiplicity adjustment change neither search policy.

## Numeric grouped test

The legacy and default setting is `feature_test="quadratic"`. It treats every
active histogram bin as a separate category in the quadratic independence
statistic. This path is unchanged.

`feature_test="grouped"` is an opt-in test for high-resolution numeric
histograms. Before the feature-level independence test, CTBoost combines
adjacent non-missing bins into at most `feature_test_bins` contiguous groups
with approximately equal node weight. A missing-value bin remains a separate
group for both `nan_mode="Min"` and `nan_mode="Max"`. Categorical features
always use the original nominal quadratic test.

```python
from ctboost import CTBoostRegressor

model = CTBoostRegressor(
    feature_test="grouped",
    feature_test_bins=8,  # accepted range: 2..64
)
```

Grouping is used only for the feature-level test. If the feature passes, the
within-feature gain search still sees every original numeric histogram bin, so
the final split is not restricted to a grouped boundary.

## Multiplicity adjustment

Set `feature_test_adjustment="bonferroni"` to compare a raw p-value with
`alpha / m`, where `m` is the number of eligible, non-degenerate features
tested at that node. Raw p-values and their ranking are unchanged. In the
ordinary search this changes the node split/no-split decision. In the advanced
ranked search used with constraints, nonzero random strength, feature weights,
or first-use penalties, it gates the significant candidate set before the
feasible adjusted-gain choice. The default is `"none"`.

With verbose profiling, `node_search` retains the existing `p_value` field for
the raw value and also reports `stopping_p_value`. The latter is
`min(1, m * p_value)` for Bonferroni and equals the raw value for `"none"`;
`tested_features` reports `m`. The raw `p_value` is the one used for ranking.

```python
model = CTBoostRegressor(
    feature_test="grouped",
    feature_test_bins=8,
    feature_test_adjustment="bonferroni",
    alpha=0.05,
)
```

CPU and GPU training use the same candidate selector for quadratic, grouped, and
Bonferroni modes. On GPU, histogram construction and row partitioning stay on device;
the already-required host snapshot is converted losslessly to the CPU selector's
histogram representation. This avoids a second statistical implementation and adds no
extra device-to-host transfer. The selected feature's gain search still uses every raw
bin. Prediction remains device-independent.

This shared-selector route also makes the ordinary unconstrained GPU path obey the same
minimum-raw-p feature choice as CPU. It is a parity-first implementation; a future
device-native statistic may reduce host selection overhead only if it reproduces these
semantics exactly.

Model state, snapshots, warm starts, the scikit-learn estimators, and the CLI
persist all three settings. Snapshot resume rejects configuration drift;
`init_model` can be used when a later stage intentionally changes the test.

## Statistical background

The two-stage design is inspired by the conditional-inference framework of
Hothorn, Hornik, and Zeileis. Production CTBoost uses histogram bins, current
objective gradients, an asymptotic chi-square statistic, missing-as-bin
semantics, and a later gain-optimal cut search. It is not an exact `ctree`,
`partykit`, permutation, or maximally selected-statistic implementation.
Grouping is a power-oriented, lower-degree-of-freedom alternative for ordered
numeric histograms.

- [Unbiased Recursive Partitioning: A Conditional Inference Framework (2006)](https://www.zeileis.org/papers/Hothorn+Hornik+Zeileis-2006.pdf)
- [Model-Based Recursive Partitioning (2008)](https://www.zeileis.org/papers/Zeileis+Hothorn+Hornik-2008.pdf)
