# Accuracy and model-size scout — 16 October 2026

Validation-selected CTBoost configurations improved mean held-out loss on all
three small real datasets in this exploratory study. They do not establish a
leaderboard improvement or justify changing production defaults. The conditional
feature-test-before-cut algorithm was unchanged.

Each value below averages two predeclared split seeds. Loss is log loss for
breast cancer and digits, and RMSE for diabetes; lower is better. The baseline
uses existing default learner controls under fixed experimental hyperparameters,
not every library default.

| Dataset | Baseline CTBoost loss | Validation-selected compact loss | Loss reduction | Fixed XGBoost loss | Mean physical nodes, baseline → compact |
|---|---:|---:|---:|---:|---:|
| Breast cancer | 0.128053 | 0.106565 | 16.78% | 0.097159 | 881 → 825 |
| Diabetes | 66.0231 | 65.2377 | 1.19% | 61.6920 | 389 → 171.5 |
| Digits | 0.180370 | 0.130347 | 27.73% | 0.133430 | 88,680 → 6,874 |

The selected CTBoost policy beat the fixed XGBoost comparator on mean loss for
digits, but remained worse on the other two datasets. Across the six correlated
splits, compact selection improved five and worsened one by 1.06%. These are
three datasets with two splits each, not six independent benchmark tasks.

## What was tested

The [runner](../../../accuracy_frontier_scout.py) first wrote a frozen
[plan](plan.json), then fitted all 26 models and wrote
[frozen selections](frozen_selections.json), then performed a separate test
assessment. The training stage never scores test predictions. Saved data roles,
runner source, native module, Python package sources, fitted models and selection
records are hash checked. All fits, pruning checks and reload checks succeeded.

- Data: bundled scikit-learn breast cancer, diabetes and digits; 60% train,
  20% validation and 20% test. Classification splits are stratified. Seeds are
  20261016 and 20261017, with a separately fixed validation-split seed offset.
- Shared settings: up to 300 rounds, learning rate 0.05, depth 6, alpha 0.05,
  L2 1, 256 bins, Bernoulli subsampling 0.8, CPU execution and validation
  early-stopping patience 40. Numeric feature testing remains quadratic and
  multiclass feature testing remains single-coordinate.
- Scalar refinement: three leaf-estimation steps with the existing loss
  backtracking. Multiclass refinement: existing coupled full-Hessian leaf solver
  with three steps and vector storage. A second refinement arm uses learning
  rate 0.15. Digits also has a vector-storage-only control.
- Quality selection chooses the lowest validation loss among the fixed arms
  and round checkpoints. Compact selection chooses the fewest physical nodes
  among models within 1% of that best validation loss. Checkpoints are
  10, 25, 50, 75, 100, 150, 200, 250, 300 and each arm's retained best round.
  Compact selection targets topology size, not bytes or measured latency.
- XGBoost uses histogram training, the same round cap, learning rate 0.05,
  depth 6, bin cap, L2 and subsampling, with validation patience 40. Its best
  iteration is physically retained for scoring and model-size accounting.

This is a **3- or 4-fit CTBoost search versus one fixed XGBoost fit** per split,
not matched HPO. The candidate combines leaf refinement, learning-rate changes,
round selection and, for digits, vector storage. There is no unrefined
learning-rate-0.15 control, so its gains cannot be attributed solely to the leaf
solver. The baseline frequently reaches the 300-round cap; more baseline
training is another untested alternative.

## Separate component findings

Vector storage reproduced the default digits validation and test margins
**exactly**, using 300 physical trees instead of 3,000 and 90% fewer physical
nodes. Mean serialized JSON size fell from 442.15 MB to 47.87 MB, an 89.17%
reduction. That is evidence for exact storage savings; this scout did not
measure warm inference latency.

At the same learning rate 0.05, leaf refinement improved dataset-mean loss by
8.75% on breast cancer and 0.53% on digits, and left diabetes unchanged. The
unconditionally applied fast-refinement arm worsened one breast-cancer split
by **21.34%**. Validation selection avoided that arm on this split. Its failed
outcome remains in the [complete assessment](assessment.json); no default was
promoted.

The compact policy's diabetes models had 25 and 200 rounds rather than the
baseline's 172 and 292. One test split worsened despite meeting the validation
tolerance. A 1% validation-loss allowance is not a 1% test-loss guarantee.

## Resources, artifacts and limits

The 26 sequential fits used one thread on logical CPU 4. Fit-call durations
summed to 7.9244 seconds: 5.6586 seconds across 20 CTBoost fits and 2.2658 seconds
across six XGBoost fits. These descriptive durations exclude model serialization,
checkpoint scoring, reloads and the rest of the experiment. Compilation was
concurrent, so they are not isolated training-performance benchmarks. Maximum
recorded RSS immediately after a fit was 502,370,304 bytes; this is not a
continuously sampled memory peak. No inference timing is reported here.

CTBoost models use its native pretty-printed JSON `.ctb` format; XGBoost uses
binary UBJSON. Bytes across these formats are deployment artifact sizes, not
an intrinsic measure of model complexity. CTBoost's JSON writer stores all 256
categorical-route entries on every node, including numeric splits and leaves,
which makes these artifacts large. Existing pickle and predictor JSON export
formats are separate deployment options; the follow-up below measures them.

The freshly compiled isolated CTBoost runtime is version 0.1.61, MSVC 19.50,
CPU Release, built from source commit
`9e68e6d5a6fe4a402d7b465092f6690bdf979a94`; see the
[build receipt](baseline_build_receipt.json). Native SHA-256 is
`41ee0e2c10704693f18ae86f77039f49a6a6ddecc6cf77fe310df2121b2495af`.
The plan records Python and dependency versions. An existing editable import
finder is explicitly removed before loading this isolated package, preventing
the checkout's stale native extension from overriding it.

Breast cancer and digits were used in earlier different 40-round development
experiments. These new splits overlap historical data and are **not independent
confirmation** of those studies. The new question is validation-selected
loss-versus-topology cost. No previous reserved TabArena confirmation roles were
opened, and no official benchmark, Elo, broad task ranking or statistical
significance is claimed. A larger untouched panel with matched search budgets
and a learning-rate-only ablation is needed before promoting an accuracy policy.

- [All fit records and checkpoint validation losses](fits.json)
- [Selections fixed before test scoring](frozen_selections.json)
- [Test outcomes, separate component arms, dataset means and split comparisons](assessment.json)
- Local fitted models, role arrays, predictions and physically pruned models:
  `.tmp/improvements-20261016/accuracy/` (excluded from version control).

To reproduce after building an isolated package from the recorded source, run
these stages in order, using fresh output and artifact directories:

```powershell
python -I benchmarks/accuracy_frontier_scout.py prepare --output benchmarks/results/improvements_20261016/accuracy-reproduction --artifacts .tmp/improvements-20261016/accuracy-reproduction --package-root .tmp/improvements-20261016/baseline --cpu 4
python -I benchmarks/accuracy_frontier_scout.py fit --output benchmarks/results/improvements_20261016/accuracy-reproduction --artifacts .tmp/improvements-20261016/accuracy-reproduction --package-root .tmp/improvements-20261016/baseline --cpu 4
python -I benchmarks/accuracy_frontier_scout.py assess --output benchmarks/results/improvements_20261016/accuracy-reproduction --artifacts .tmp/improvements-20261016/accuracy-reproduction --package-root .tmp/improvements-20261016/baseline --cpu 4
```

The protocol refuses to overwrite earlier fit or assessment records. Runtime
hashes are local build identities; a separately reproduced build may have a
different native hash and will freeze its own before training.

## Existing export-format follow-up

After the accuracy assessment, a separately frozen [size/parity
follow-up](export_sizes/plan.json) used the first fixed digits baseline model,
without fitting or choosing models. The original full training artifact is
slightly larger than the pruned assessment artifact because it retains training
metadata. Existing exports produced:

| Format | File size | Reduction from native JSON | Prediction check on 360 test rows |
|---|---:|---:|---|
| Native JSON `.ctb` | 449.32 MB | reference | native raw-score reference |
| Existing pickle `.pkl` | 62.51 MB | 86.09% | raw scores exactly equal |
| Existing predictor JSON | 400.72 MB | 10.82% | max absolute raw-score difference 0.000004975; every predicted class equal |

Predictor JSON passed the predeclared `rtol=2e-5, atol=2e-5` check. Its arithmetic
is not bit-identical to the native runtime. Neither alternative was timed, and
smaller files do not establish faster Python inference. Predictor JSON still
embeds the large node payload; it is not a generally compact representation.
For trusted local model persistence, the existing `save_model("model.pkl")`
option provides the large size reduction demonstrated here without changing
training or serialization contracts.

The [follow-up runner](../../../serialization_size_scout.py) and
[complete results](export_sizes/result.json) retain input/output hashes and
prediction tolerances. Reproduce with fresh destinations:

```powershell
python -I benchmarks/serialization_size_scout.py --accuracy benchmarks/results/improvements_20261016/accuracy --package-root .tmp/improvements-20261016/baseline --output benchmarks/results/improvements_20261016/accuracy-export-reproduction --artifacts .tmp/improvements-20261016/accuracy-export-reproduction
```
