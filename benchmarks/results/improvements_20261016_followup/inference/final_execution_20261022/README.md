# Final selected-model inference diagnostic — 22 October 2026

Across all 36 task/batch cases, raw-input CTBoost prediction was **2.247x faster than XGBoost** and **1.915x faster than LightGBM** by geometric mean. CTBoost was faster in 36/36 and 35/36 cases respectively. These are warm serving costs for different selected models and accuracy levels, not an accuracy-matched comparison or an official ranking.

The complete primary-seed-47 wave-18b recovery policy was selected by the prospectively frozen [v4 rule](../final_runner_preparation/selection_policy_v4.json), regardless of its accuracy outcome. All 540 folds and 36 final fits succeeded. The development target failed: median relative error reduction was +0.2075% versus XGBoost (7/12 wins) and -0.8217% versus LightGBM (5/12 wins). Seed 677 and official testing remained closed. No fallback, per-task mixing, model-size filtering or timing-based selection was used.

## Scope and checks

The same 12 tasks and 36 frozen single models were measured at batches of 1, 64 and 1,000 rows. Three passes rotate CTBoost/XGBoost/LightGBM, LightGBM/CTBoost/XGBoost, then XGBoost/LightGBM/CTBoost. Every library occupies each order position once. Each call uses one CPU (logical CPU 6), one native thread, three warmups, five calibration calls and seven blocks targeting 50 ms. Fits, builds and tests were stopped throughout the reference and all three timing passes.

Raw time includes shared frozen vocabulary normalization, one complete model prediction and its fixed temperature once. Normalized-frame time excludes only that shared normalization; each library's own preparation remains included. Batches cycle the development rows. Models differ in retained rounds, corrections, serialized size and accuracy; the full records retain those details.

The unchanged final runner verified byte-identical predictions for every full development input and every batch, including XGBoost inplace versus DMatrix margins. All 108 case records matched the reference in all three passes. The final inference runtime is CPython 3.12/MSVC, native SHA256 `04adb21b0e4cea8046581a23b62585b6c8d631561d96a417ad87bc3c55041613`. The accuracy models were trained by the frozen earlier `c5ea…` runtime and retain legacy CTR arithmetic when loaded. Thus this timing study does not measure fresh format-5 fitting quality.

The separately tested installed 0.1.62 wheel also reproduced all 36 full-development and 108 batch references exactly in a [correctness-only check](../../release_preparation/installed_candidate_02/final_policy_parity_20261022.json). That wheel uses a separate native build (`4e2924…`); its parity check is not an additional timing pass.

## Aggregate speed ratios

Speed ratio is competitor latency / CTBoost latency; greater than one means CTBoost was faster. Per-case pass medians are combined by geometric mean, then aggregated with equal weight over the listed cases.

| Batch rows | Raw vs XGBoost | Raw vs LightGBM | Normalized vs XGBoost | Normalized vs LightGBM | Raw faster cases XGB / LGB |
|---|---:|---:|---:|---:|---:|
| 1 | 2.264x | 1.491x | 15.165x | 6.758x | 12/12 / 12/12 |
| 64 | 2.250x | 1.774x | 7.255x | 4.847x | 12/12 / 12/12 |
| 1000 | 2.227x | 2.656x | 3.187x | 3.655x | 12/12 / 11/12 |
| All | 2.247x | 1.915x | 7.051x | 4.928x | 36/36 / 35/36 |

## All raw-input cases

Values are milliseconds, geometric means of the three pass medians.

| Task | Rows | CTBoost | XGBoost | LightGBM |
|---|---:|---:|---:|---:|
| blood-transfusion-service-center | 1 | 0.5299 | 1.3201 | 0.6993 |
| blood-transfusion-service-center | 64 | 0.6058 | 1.4436 | 0.7198 |
| blood-transfusion-service-center | 1000 | 0.6691 | 1.8084 | 0.9060 |
| churn | 1 | 2.6543 | 5.0539 | 3.8323 |
| churn | 64 | 2.8633 | 5.4504 | 3.8124 |
| churn | 1000 | 5.5963 | 11.7628 | 4.4462 |
| in_vehicle_coupon_recommendation | 1 | 3.7418 | 10.2335 | 6.4153 |
| in_vehicle_coupon_recommendation | 64 | 4.8507 | 13.8622 | 12.8724 |
| in_vehicle_coupon_recommendation | 1000 | 20.1651 | 59.7107 | 107.6172 |
| GiveMeSomeCredit | 1 | 1.3171 | 2.2730 | 1.5661 |
| GiveMeSomeCredit | 64 | 1.4953 | 2.3922 | 1.9464 |
| GiveMeSomeCredit | 1000 | 2.9567 | 3.3248 | 6.6954 |
| QSAR_fish_toxicity | 1 | 0.8061 | 1.6437 | 0.9753 |
| QSAR_fish_toxicity | 64 | 0.9212 | 1.6908 | 1.0206 |
| QSAR_fish_toxicity | 1000 | 1.2056 | 2.0900 | 1.4490 |
| Another-Dataset-on-used-Fiat-500 | 1 | 0.9457 | 2.2364 | 1.7217 |
| Another-Dataset-on-used-Fiat-500 | 64 | 1.0493 | 2.3648 | 1.8642 |
| Another-Dataset-on-used-Fiat-500 | 1000 | 1.9106 | 3.6357 | 3.8008 |
| houses | 1 | 1.0662 | 2.0722 | 1.2976 |
| houses | 64 | 1.4893 | 3.2925 | 4.2492 |
| houses | 1000 | 5.8074 | 16.6142 | 46.5824 |
| diamonds | 1 | 1.3689 | 3.0339 | 2.3989 |
| diamonds | 64 | 1.5934 | 3.3889 | 5.1432 |
| diamonds | 1000 | 3.9074 | 7.7720 | 45.4742 |
| anneal | 1 | 5.5695 | 14.6163 | 9.2784 |
| anneal | 64 | 6.2036 | 14.9436 | 9.6273 |
| anneal | 1000 | 13.6312 | 21.7010 | 14.4830 |
| website_phishing | 1 | 1.4111 | 4.5708 | 2.9011 |
| website_phishing | 64 | 1.5402 | 4.7809 | 3.1444 |
| website_phishing | 1000 | 2.6984 | 8.2378 | 6.7304 |
| splice | 1 | 11.7345 | 25.1856 | 15.7802 |
| splice | 64 | 10.8241 | 25.3145 | 17.0077 |
| splice | 1000 | 14.6246 | 28.9378 | 36.6798 |
| SDSS17 | 1 | 2.8350 | 6.0632 | 4.0871 |
| SDSS17 | 64 | 3.0831 | 7.3470 | 6.0692 |
| SDSS17 | 1000 | 6.5744 | 28.6941 | 41.9370 |

## Evidence

[Combined observations](comparison.json), [reference](results/reference.json), [pass 1](results/pass-1.json), [pass 2](results/pass-2.json), [pass 3](results/pass-3.json), and [root verification and exclusive-run authorization](authorization.json). Execution receipts record clean process exit, whole-process RSS and wall time; these resource totals are not per-prediction latency. All model, input, source and runtime identities are retained. Earlier policy timings remain separate historical observations and are not paired speedups against this policy.
