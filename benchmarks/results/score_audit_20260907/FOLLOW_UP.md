# Further audit findings

These synthetic probes identified separate issues. They are not implemented in
this change and were not used to select datasets or parameters in either study.
None requires replacing conditional-inference feature selection.

| Area | Observed behavior in public 0.1.59 | Focused follow-up |
| --- | --- | --- |
| Quantile loss | Sixteen targets equal to 7, median quantile, one tree, learning rate 0.1 and L2 1 move the prediction from its optimum 7 to 6.95294094. Pinball loss increases from 0 to 0.02352953. | Test a zero subgradient at exactly zero residual, including tied targets and several quantiles. See `src/core/objective_regression.cpp`. |
| Huber initialization | Fifteen targets equal to 7 and one finite target of 1e38 initialize at 39,443,051 even when the outlier has zero weight. With outlier weight 0.1 the optimum is about 7.00667, also far from that initializer. | Replace the fixed 100-step bisection stopping rule with floating-point convergence and exclude zero-weight extrema. See `src/core/booster_base_score.cpp`. |
| Zero-weight rows and numeric borders | Adding 8,192 zero-weight numeric outliers to 512 weighted rows changes auto-binning enough to increase the probe's RMSE from about 0.03939 to 0.57848. Explicit borders learned from the weighted rows restore identical predictions. | Audit dense and sparse border learning before excluding zero-weight rows from automatic borders. Preserve explicit-border behavior. |
| Missing categorical values | `pandas.NA` receives the same native category as the literal string `"<NA>"`, while `None` and numeric NaN share another code. | Define and version missing-value normalization before changing saved categorical mappings. |

The two loss findings concern objectives outside the RMSE/AUC/log-loss studies.
The studies do not pass sample weights. Categorical missing-value normalization
is unchanged between the compared arms; these experiments do not isolate or
measure the missing-value issue.
