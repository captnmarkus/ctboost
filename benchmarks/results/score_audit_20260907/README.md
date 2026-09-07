# CTBoost development score audit — 2026-09-07

Neither development comparison showed a broad score gain. The existing
**`max_bins=256` default remains unchanged**, and these results do not support
HPO promotion.

Correcting fractional CTR smoothing produced mixed validation results: **6 wins,
4 losses, and a +0.0818% median relative error improvement** across 10 datasets.
The largest regression was **splice: −12.1872%**. This supports correcting the
arithmetic contract, but does not establish a general score improvement.

The audit found two issues: fractional CTR priors were incorrectly scaled down
at cold starts, and native early stopping could retain too many rounds when the
iteration budget expired. The experiment below evaluates CTR smoothing. It uses
callbacks in both wheels, so it does not measure the native early-stopping fix.
The study binary predates the separate DART warm-start follow-up.

## CTR comparison

The [plan](ctr/plan.json) was frozen locally before these fits; it was **not
public preregistration**. The panel contains all 10 datasets with categorical
columns among the original 14 metadata-selected pilot tasks. Only official
outer-training rows were used, with inner folds 2/3 from eight-way CV, seed 0,
stratified for classification. These development data overlap earlier studies;
they are not an independent holdout, HPO admission test, or leaderboard result.

There were 80 fits: 10 datasets × 2 folds × prior strengths 0.2/1.0 × two wheels.
All succeeded, with no deadline stops. Local execution used eight workers with
two CPU threads each. Both wheels used the same 400-tree cap, callback policy,
early-stopping patience 50, and 300-second fit budget. All other parameters are
in the plan. At strength 1, all 20 paired validation prediction arrays were
bitwise identical. Timings are descriptive development measurements.

The following changes compare the two-fold mean error at strength 0.2 using
`(public − development) / public`; positive values mean lower error. Errors are
1−AUC for binary classification, log loss for multiclass, and RMSE for regression.

| Dataset | Relative error improvement |
| --- | ---: |
| coil2000_insurance_policies | +0.1532% |
| wine_quality | −0.3086% |
| diamonds | +0.3984% |
| MIC | +0.0104% |
| SDSS17 | −0.5382% |
| anneal | −5.3345% |
| hiva_agnostic | +0.7532% |
| splice | −12.1872% |
| students_dropout_and_academic_success | +0.7660% |
| website_phishing | +2.6402% |

The [complete summary](ctr/summary.json) and [80 original fit records](ctr/fits)
retain both strengths, fold errors, completed rounds, elapsed time, and peak RSS.

## Provenance and reproduction

The baseline is the public CTBoost 0.1.59 wheel. The development wheel uses the
same package version and is identified separately by its native binary hash:

| Native binary | SHA256 |
| --- | --- |
| Public 0.1.59 | `641bafbb431292a32fea2c3857047bcbd97d849902064e1e1cad2c2f85bc6cfb` |
| Development study wheel | `e795ace998ac5c8031c7bf714edacec4a8139a8a6746b3fdb0dca00213b90639` |

[ctr_original_sha256.json](ctr_original_sha256.json) records the original paths
and hashes for all 82 copied files, plus hashes of the 80 prediction archives.
Copies preserve original bytes and plan references. The summary was reconstructed
from the individual JSON errors. Prediction arrays and raw features are excluded
from this bundle; their exact-prediction checks require the original archives.

The pinned runner is [ctr_prior_scout.py](../../ctr_prior_scout.py), SHA256
`c937c81b5a7089623c6bca0791630b377593ccf75c747fdb1b81268eb987e748`.
An exact refit requires its original prepared training cache and both isolated
study binaries matching the plan hashes. A later rebuilt wheel is a different
experimental artifact even if its package version is unchanged.

To verify the bundled copies from the repository root:

```python
import hashlib, json
from pathlib import Path
root = Path("benchmarks/results/score_audit_20260907")
for name in ["ctr_original_sha256.json", "bins_original_sha256.json"]:
    inventory = json.loads((root / name).read_text())
    for item in inventory["files"] + inventory.get("reused_fit_records", []):
        assert hashlib.sha256((root / item["artifact"]).read_bytes()).hexdigest() == item["sha256"]
```

## Numeric-bin comparison

The separate [plan](bins/plan.json) was frozen locally before inspecting these
scores. It compares 64 versus 256 bins on all 14 original metadata-selected
tasks, using the same inner folds 2/3, development binary, prior strength 1,
400-tree cap, callbacks, and CPU layout. This was also development testing,
not public preregistration or independent confirmation.

The research rationale was a hypothesis: fewer bins might improve the power or
cost of the quadratic feature test. The conditional-inference framework uses an
asymptotic chi-squared distribution whose degrees of freedom depend on covariance
rank ([Hothorn, Hornik and Zeileis, 2006, section 3](https://www.zeileis.org/papers/Hothorn%2BHornik%2BZeileis-2006.pdf)).
CTBoost's [default binned statistic](../../../src/core/statistics.cpp#L338) uses
`active_bins - 1`. Reducing bins can reduce that dimension, while also losing
numeric resolution. The paper does not establish that 64 bins predict better.

All **56 records succeeded: 36 new fits plus 20 reused CTR baselines**, with no
deadline stops. The reused records retain their original provenance and refer
to the copies in [ctr/fits](ctr/fits). Reuse and overlapping development data
prevent a controlled runtime-speedup or independent-validation claim.

The 64-bin arm had **5 wins, 9 losses, and −0.5381% median relative error
improvement**. The largest regression was anneal: mean log loss rose from
**0.0229569 to 0.0601462**, giving **−161.9959%** relative improvement. Positive
values below mean lower error with 64 bins, using `(error256 − error64) / error256`.

| Dataset | Relative error improvement |
| --- | ---: |
| blood-transfusion-service-center | −0.5747% |
| coil2000_insurance_policies | −1.9334% |
| GiveMeSomeCredit | −0.5016% |
| QSAR_fish_toxicity | +0.7784% |
| wine_quality | −0.1250% |
| diamonds | −1.0409% |
| MIC | −0.6318% |
| SDSS17 | −9.7298% |
| anneal | −161.9959% |
| hiva_agnostic | +1.0712% |
| maternal_health_risk | +3.3889% |
| splice | +8.9457% |
| students_dropout_and_academic_success | +0.8531% |
| website_phishing | −1.2055% |

The [summary](bins/summary.json) contains every fold record. The
[original-file inventory](bins_original_sha256.json) covers the master plan,
both arm plans, reuse manifest, summary, and 36 new fit records, with references
to the 20 existing CTR copies. Prediction arrays and raw data are excluded.
The runner is [numeric_bins_scout.py](../../numeric_bins_scout.py); its exact
source and dependency hashes are recorded in the master plan.

## Follow-up and final validation

[Further audit findings](FOLLOW_UP.md) records separate, unimplemented issues;
they were not used to select either study's datasets or parameters.

The final development build corrects fractional CTR smoothing and native
best-model retention at the iteration limit. It also handles early-stopping
resumes from untrimmed DART models by evaluating and retaining the supplied full
ensemble before considering later improvements. That prevents restoring a
prefix whose tree weights no longer match the recorded historical score.
Native and callback paths follow the same resume rule. Conditional-inference
feature selection and split selection are unchanged.

New fitted pipelines use format 4. Existing format-3 pipelines retain their
original smoothing, predictions and export format; runtimes that only understand
format 3 reject new format-4 pipelines. See the [deployment contract](../../../docs/guides/deployment.md).

[Checks against models saved by the public 0.1.59 wheel](legacy_verification.json)
preserved bitwise predictions and transformed features for regression, binary
and multiclass cases, including known, unseen and missing categories. Resaved
model files were byte-identical; raw JSON re-exports changed only producer-build
metadata and retained identical predictions.

The final wheel is a development artifact, not a new official release:

| Final artifact | SHA256 |
| --- | --- |
| Native extension | `bdb79e439ac153067b01f2c3a28a1158e88fd5b46bb646a24891b55faa0396e2` |
| Windows CPython 3.12 wheel | `dc727b109f3917c60d83429457a72547ad6499520b33c273b06a5fa2ca4be03e` |

The full suite passed **1,343 tests**, with 32 skips, on Windows/Python 3.12.11.
Two compiler-dependent C++ export tests then passed separately under MSVC,
bringing the total to **1,345 distinct passing tests and 30 remaining skips**
for CUDA or optional integrations. All 22 early-stopping regression cases passed,
including native/callback path switches, patience and iteration limits, and
changed validation data during an untrimmed DART resume. Documentation builds
with `mkdocs build --strict` and Material 9.7.7.

[Independent metric verification](metric_verification.json) recomputed every
saved error in both studies from prediction archives, with maximum absolute
difference 0.0; split indices, label encodings, hashes and all negative controls
also passed. Both studies retain their original pre-DART-follow-up binary above;
the final DART resume change was regression-tested, not benchmarked in these
Plain-boosting comparisons.

The benchmark helper loader was also repaired to include the adapter's existing
`learning_options` dependency. Its historical protocols, schedules and results
were preserved. No HPO portfolio, official release or TabArena PR was updated.
