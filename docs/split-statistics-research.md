# Split-statistics research ledger

## Scope and invariant

The initial experiments on this page are synthetic, reference-only evidence;
by themselves they changed no native C++ code, tree topology, default,
TabArena configuration, or released behavior. The later registered external
panel evaluates the subsequently added opt-in grouped test. CTBoost's default
remains the conditional-inference quadratic test followed by a gain-maximizing
cut search within the selected feature.

The research question is narrower: can alternative *feature tests* improve
power without giving up multiplicity control? Every candidate keeps feature
selection separate from the later cut search.

## Production baseline and missing values

The Python reference reproduces the covariance, weighted variance, `k - 1`
degrees of freedom, and chi-square tail convention in
`src/core/statistics.cpp`. The missing policy is explicit:

- `production_bin` (the reference default) mirrors CTBoost quantization and
  gives missing values their own bin;
- `ctree_omit` follows the original CTree description and omits rows whose
  selected covariate is missing from that feature's test.

At 30% MCAR missingness, 2,000 null repetitions gave type-I error 0.042 for
`production_bin` and 0.040 for `ctree_omit` at alpha 0.05. Thus the policy
correction did not materially alter null calibration in this experiment.
It *does* alter the alternative: when the response shift existed only in the
missing rows, missing-as-bin candidates had power 1.000 while the omission-only
ordered and maxstat tests stayed near their null rates (0.048 and 0.076).

## Primary literature ledger

| Source | Result used here | Research consequence |
|---|---|---|
| [Hothorn, Hornik, and Zeileis (2006), *Unbiased Recursive Partitioning*](https://www.zeileis.org/papers/Hothorn+Hornik+Zeileis-2006.pdf) | Separates a global independence test, variable selection, and cut selection using conditional linear statistics; discusses case weights, multiplicity, edge restrictions, and feature-wise omission of missing values. | Preserve the conditional-testing architecture; make missing and weight semantics explicit. |
| [Strasser and Weber (1999), *On the Asymptotic Theory of Permutation Statistics*](https://research.wu.ac.at/ws/portalfiles/portal/19841038/document.pdf) | Supplies the conditional expectation/covariance framework and asymptotic permutation theory used by CTree. | Use score/covariate transformations within the same inferential framework rather than greedy gain for feature selection. |
| [Hothorn and Zeileis (2008), *Generalized Maximally Selected Statistics*](https://www.zeileis.org/papers/Hothorn+Zeileis-2008.pdf) | Embeds maximally selected statistics in conditional inference and gives efficient asymptotic evaluation over cutpoints. | Maxstat is methodologically valid, but this first implementation uses Monte Carlo permutations as an auditable calibration oracle. |
| [Lausen and Schumacher (1992), *Maximally Selected Rank Statistics*](https://doi.org/10.2307/2532740) | Restricts cut selection to an inner support interval and derives the null law of the maximum standardized process. | Trim edge cuts before taking a maximum; never report an unadjusted best-cut p-value. |
| [Zeileis, Hothorn, and Hornik (2008), *Model-Based Recursive Partitioning*](https://www.zeileis.org/papers/Zeileis+Hothorn+Hornik-2008.pdf) | Tests observation-wise objective scores for parameter instability. | Built-in likelihood gradients are score-like response transformations; arbitrary callable-objective gradients need not retain that inferential interpretation. |
| [Schlosser, Hothorn, and Zeileis (2019), *The Power of Unbiased Recursive Partitioning*](https://arxiv.org/abs/1906.10179) | Compares CTree, MOB, and GUIDE components; score tests avoid power lost by dichotomization, while sum-of-squares and max-selected tests favor different alternative shapes. | Test smooth, abrupt, and U-shaped alternatives rather than declaring one statistic universally best. |
| [Loh (2002), *Regression Trees with Unbiased Variable Selection and Interaction Detection*](https://www3.stat.sinica.edu.tw/statistica/j12n2/j12n21/j12n21.htm) | Uses coarsened residual association tests to avoid variable-selection bias efficiently. | Evaluate selection-only histogram grouping as a cheap GUIDE-like approximation; retain raw bins for the eventual cut. |
| [Fisher (1958), *On Grouping for Maximum Homogeneity*](https://doi.org/10.1080/01621459.1958.10501479) | Studies optimal grouping of ordered values into homogeneous contiguous groups. | Preserve order when coarsening numeric bins; do not hash them into arbitrary groups. |
| [Prokhorenkova et al. (2018), *CatBoost: Unbiased Boosting with Categorical Features*](https://proceedings.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf) | Shows how same-row target statistics create target leakage and uses ordered information to avoid it. | Any target/gradient-derived category ordering must be out-of-fold, ordered, or otherwise exclude the row's own target. |

## Reference methods

The ordered statistic assigns tied feature values a weighted midrank and uses
the one-df standardized cross-product with the objective score. The maxstat
candidate considers only cuts leaving at least 10% on either side. For one
feature test, the same Monte Carlo response permutations are reused across all
eligible cutpoints, so the null statistic is the *maximum* for every draw. Its
p-value is `(1 + count(T_perm >= T_observed)) / (B + 1)`.

Global stopping chooses the smallest feature p-value and uses
`min(1, number_of_features * p_min)`. Note that a Monte Carlo maxstat with `B`
permutations cannot attain p-values below `1 / (B + 1)`. With many features,
far more than 199 permutations or a validated asymptotic/sequential method
would be required before global Bonferroni stopping could reject.

The grouped candidate maps the existing ordered quantization levels into 8 or
16 contiguous, approximately equal-frequency groups and runs the quadratic
test on those groups. This grouping is **selection only**. The later split
search would still see all original bins. Missing values remain a dedicated
test group. The hybrid takes the smaller ordered/grouped p-value and applies a
within-feature Bonferroni factor of two before global feature correction.

For binary categorical orderings at a common prior `p`, a Newton update with
L2 penalty `lambda` and smoothed WoE with pseudo-count
`lambda / (p * (1 - p))` are monotone transforms of the same smoothed category
rate. The reference verifies their rankings under unequal frequencies. This is
an algebraic equivalence for the intercept-only binary score, not a claim for
multiclass or changing per-row Hessians.

## Deterministic experiment

The curated ledger is
`benchmarks/split_research/results/reference_seed_20260810.json`: seed
20260810, n=320, 250 repetitions per main/power scenario, 200 global-null
repetitions, 199 maxstat permutations, alpha 0.05, effect size 0.6. Total
reference runtime was 54.30 seconds on the development machine.

### Null calibration and power

| Scenario | Raw nominal | Ordered midrank | Permutation maxstat |
|---|---:|---:|---:|
| Null, 2 levels | 0.044 | 0.044 | 0.052 |
| Null, 8 levels | 0.044 | 0.068 | 0.044 |
| Null, 32 levels | 0.052 | 0.056 | 0.056 |
| Null, 64 levels | 0.032 | 0.056 | 0.056 |
| Null, 255 levels | 0.008 | 0.040 | 0.064 |
| Null, 30% MCAR missing | 0.024 | 0.056 | 0.048 |
| Null, literal frequency weights | 0.052 | 0.024 | 0.056 |
| Smooth signal | 1.000 | 1.000 | 1.000 |
| Abrupt signal | 1.000 | 1.000 | 1.000 |
| U-shaped signal | 1.000 | 0.096 | 1.000 |

Global Bonferroni family-wise rejection rates across 2/8/32/64/255-level
noise features were 0.015 (nominal), 0.040 (ordered), and 0.035 (maxstat).

### Numeric quantization stress test

For continuous signals, the raw nominal test had power 1.000 at 8, 32, and 64
equal-frequency bins for all three shapes, but power 0.000 at 255 bins. Its
median p-values at 255 bins were 0.221 (smooth), 0.210 (abrupt), and 0.216
(U-shaped). The corresponding 255-bin null rejection rate was also 0.000.
This falsifies the idea that retaining `k - 1` degrees of freedom is harmless
at near-saturated numeric cardinality: the test becomes extremely
conservative and loses all demonstrated power.

### Selection-only grouping at 255 raw bins

| Relationship | Raw 255 | Grouped 8 | Grouped 16 | Ordered | Hybrid 8 | Hybrid 16 | Maxstat |
|---|---:|---:|---:|---:|---:|---:|---:|
| Null | 0.000 | 0.040 | 0.024 | 0.056 | 0.048 | 0.052 | 0.036 |
| Smooth | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Abrupt | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| U-shaped | 0.000 | 1.000 | 1.000 | 0.112 | 1.000 | 1.000 | 1.000 |
| Missingness-only shift | 0.528 | 1.000 | 1.000 | 0.048 | 1.000 | 1.000 | 0.076 |

Mean Python-reference times were 1,458 microseconds for grouped 8, 3,687 for
hybrid 8, and 2,725 for maxstat with only 199 permutations. These are not
native benchmarks: Python factorization dominates some paths. Grouped 8 is a
single histogram reduction and should receive a native timing ablation;
maxstat work scales with the number of permutations.

### Categorical diagnostics

Matched Newton and smoothed-WoE orderings had 1.000 pairwise agreement. On a
null dataset with a unique category per row, in-sample WoE had target
correlation 1.000, while five-fold cross-fitted scores were neutral for all
unseen categories and had correlation 0.000. WoE is therefore not a distinct
binary ordering gain under matched regularization, and leakage control is
non-negotiable if any target-derived ordering is exposed.

## Decision record

### What improved—and what did not

The corrections answer different questions and should not be described as a
single scoring improvement:

- The Hothorn/Hornik/Zeileis conditional-inference framework motivates
  separating variable selection from cut optimization and controlling
  multiplicity. That addresses selection bias and stopping; it does not by
  itself promise lower predictive loss.
- The deterministic experiment observed Bonferroni-adjusted feature-family
  rejection rates of 0.015 nominal, 0.040 ordered, and 0.035 maxstat at alpha
  0.05. In CTBoost the adjustment changes the stopping threshold,
  not raw-p-value feature ranking. The experiments do not establish a
  predictive-score gain caused by Bonferroni.
- Grouping 255 numeric bins into eight selection-only test groups restored
  power across the tested smooth, abrupt, U-shaped, and missingness-only
  alternatives. In the external panel its observed median primary-loss point
  estimate improved by 5.63% (task-bootstrap 95% interval -1.23% to +13.64%),
  but it missed the frozen speed gate; it therefore remains opt-in.
- In the matched binary reference setting, smoothed-WoE and Newton category
  orderings had 1.000 pairwise agreement. Same-row WoE leaked target
  information, while five-fold cross-fitted scores were neutral for unseen
  categories. WoE supplied no separate demonstrated ordering or score gain
  and was not advanced.
- The ordered/grouped hybrid and the 16-group variant showed no power gain
  over grouped-8 in the registered synthetic scenarios, while adding work or
  multiplicity. They were not advanced.

These conclusions are deliberately narrower than the literature: score-test
power depends on the alternative and response transformation, as emphasized
by Schlosser, Hothorn, and Zeileis. None of the reference results establishes
a universal best split statistic.

| Candidate | Decision | Reason |
|---|---|---|
| Grouped quadratic, 8 test groups | **Advance to external/native ablation** | Calibrated null, full power across smooth/abrupt/U-shaped/missingness alternatives, lowest grouped complexity, and no change to the raw-bin gain search. |
| Ordered weighted-midrank | **User-test candidate** | Excellent smooth/abrupt power and one df, but weak for U-shaped and missingness-only alternatives. |
| Permutation maxstat | **User-test candidate** | Broad shape power and correct cut-search calibration, but Monte Carlo resolution conflicts with global correction at large feature counts and needs native/asymptotic performance work. |
| Grouped quadratic, 16 groups | Do not advance now | No synthetic power gain over 8 groups; slightly more degrees of freedom. |
| Ordered + grouped hybrid | Do not advance now | No observed power gain over grouped 8; adds a second test, extra runtime, and another multiplicity adjustment. |
| Raw 255-bin `k - 1` test as a numeric strategy | Do not advance | Null-conservative and zero power in the high-cardinality stress test. The released production default remains unchanged until an external/native ablation justifies migration. |
| Smoothed WoE as a separate binary category ordering | Do not advance | Algebraically ranking-equivalent to the matched Newton score in this setting; same-row target use leaks. |

The required evaluation sequence for grouped 8 is a native, opt-in
implementation followed by the
[pre-registered external panel](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/split_research/EXTERNAL_PANEL.md)
and, only if every panel promotion gate passes, a frozen TabArena ablation.
Synthetic results alone are insufficient to change CTBoost defaults or claim
an Elo improvement.

The final-source CTBoost 0.1.55 external panel is now complete. Its
[sanitized release evidence](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/split_research/results/grouped_external_panel_v2_fb65b685.json)
records 294/294 successful isolated fits and 42/42 exact implicit-control
checks. Grouped-8 won nine of twelve decision datasets and improved median
primary loss by 5.63%, but its 1.1708 median paired fit-time ratio exceeded the
frozen 1.15 ceiling. Because every promotion gate is conjunctive, grouped-8
does not advance; the frozen TabArena scout was not triggered under the
protocol and therefore was not run. This 0.1.55 result supersedes the earlier
0.1.54 positive artifact only for release qualification; the earlier sealed
result remains historical evidence.
