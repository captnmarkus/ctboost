# Frozen wave11 ensemble serving costs

Final production CPython 3.12 native runtime, 12 datasets, batch sizes 1/64/1000, three balanced library-order passes on one pinned CPU thread. Each library serves its frozen two-seed 47+947 policy. The quality target failed; this is a serving-cost diagnostic of those selected models, whose accuracy and capacities differ.

CTBoost raw end-to-end geometric mean speedup is 2.9047× versus XGBoost 3.4.1 (36/36 dataset-batch cases faster) and 2.4140× versus LightGBM 4.7.0 (35/36). With shared input normalization excluded, speedups are 6.3997× and 4.6897×. These results are scoped to these 36 cases and are not an equal-quality comparison or a general claim about all workloads.

Every timed prediction includes exactly two complete component model calls, each with its fitted preprocessing and temperature applied exactly once, followed by the frozen equal-probability mean (classification) or equal raw-prediction mean (regression). The raw scope also includes one shared train-only vocabulary normalization. The normalized-frame scope excludes only that common normalization. No call is omitted for duplicate component predictions.

The only raw loss is churn at 1000 rows: CTBoost 6.6797 ms, XGBoost 9.2178 ms, LightGBM 5.7634 ms. CTBoost has 274 total retained rounds and 0.11335 development binary error; LightGBM has 12 rounds and 0.11901 error. This illustrates the different quality/capacity operating points.

Each pass independently verifies source/dependency/runtime/model hashes and the full check-only reference. Full development predictions and every timed batch match saved predictions byte for byte. XGBoost inplace raw margins are verified per component and for the final mixture against its standard DMatrix path on the full source and every timed batch. The two timed scopes also match exactly. All 3 passes / 324 batch-library records completed, with no discarded runs.

The three library orders are CTBoost/XGBoost/LightGBM, XGBoost/LightGBM/CTBoost, and LightGBM/CTBoost/XGBoost. Each path has 3 warmups, 5 calibration calls, and 7 blocks targeting 50 ms. Tables use the geometric mean of each order's median. Full block samples and order medians remain in the JSON receipts.

This experiment does not train, select, change quality scores, or access the official test. It does not establish a paired slowdown or improvement relative to historical wave 6 single-model results, which used different models and runtime.

## End-to-end latency

| Dataset | Rows | CTBoost ms | XGBoost ms | LightGBM ms | XGB/CTB | LGB/CTB |
|---|---:|---:|---:|---:|---:|---:|
| blood-transfusion-service-center | 1 | 0.6033 | 2.0619 | 0.9214 | 3.418 | 1.527 |
| blood-transfusion-service-center | 64 | 0.7338 | 2.0967 | 0.9481 | 2.857 | 1.292 |
| blood-transfusion-service-center | 1000 | 0.9923 | 2.6305 | 1.2726 | 2.651 | 1.282 |
| churn | 1 | 2.7107 | 6.9326 | 4.8479 | 2.558 | 1.788 |
| churn | 64 | 3.0293 | 7.1059 | 4.9136 | 2.346 | 1.622 |
| churn | 1000 | 6.6797 | 9.2178 | 5.7634 | 1.380 | 0.863 |
| in_vehicle_coupon_recommendation | 1 | 3.7880 | 16.0027 | 9.0664 | 4.225 | 2.393 |
| in_vehicle_coupon_recommendation | 64 | 4.4083 | 18.4109 | 15.4532 | 4.176 | 3.505 |
| in_vehicle_coupon_recommendation | 1000 | 12.1696 | 48.2098 | 110.3456 | 3.961 | 9.067 |
| GiveMeSomeCredit | 1 | 1.3401 | 3.0656 | 1.7287 | 2.288 | 1.290 |
| GiveMeSomeCredit | 64 | 1.5895 | 3.2059 | 2.1291 | 2.017 | 1.339 |
| GiveMeSomeCredit | 1000 | 3.2467 | 4.8957 | 6.6636 | 1.508 | 2.052 |
| QSAR_fish_toxicity | 1 | 0.8459 | 2.3966 | 1.1714 | 2.833 | 1.385 |
| QSAR_fish_toxicity | 64 | 0.9991 | 2.6734 | 1.3321 | 2.676 | 1.333 |
| QSAR_fish_toxicity | 1000 | 1.4187 | 5.2959 | 2.9505 | 3.733 | 2.080 |
| Another-Dataset-on-used-Fiat-500 | 1 | 1.0204 | 3.4058 | 2.4801 | 3.338 | 2.431 |
| Another-Dataset-on-used-Fiat-500 | 64 | 1.1734 | 3.5383 | 2.5904 | 3.016 | 2.208 |
| Another-Dataset-on-used-Fiat-500 | 1000 | 2.7105 | 4.8896 | 4.0527 | 1.804 | 1.495 |
| houses | 1 | 1.0978 | 2.9286 | 1.5489 | 2.668 | 1.411 |
| houses | 64 | 1.8463 | 5.2591 | 7.6424 | 2.848 | 4.139 |
| houses | 1000 | 9.9987 | 30.4841 | 95.9407 | 3.049 | 9.595 |
| diamonds | 1 | 1.4703 | 4.5620 | 3.4269 | 3.103 | 2.331 |
| diamonds | 64 | 1.8035 | 5.2254 | 9.0875 | 2.897 | 5.039 |
| diamonds | 1000 | 5.6093 | 13.6279 | 91.9250 | 2.430 | 16.388 |
| anneal | 1 | 5.8622 | 22.7686 | 12.7710 | 3.884 | 2.179 |
| anneal | 64 | 7.7282 | 24.2014 | 16.1968 | 3.132 | 2.096 |
| anneal | 1000 | 28.1983 | 47.1784 | 64.5660 | 1.673 | 2.290 |
| website_phishing | 1 | 1.6123 | 7.7145 | 4.4441 | 4.785 | 2.756 |
| website_phishing | 64 | 1.8702 | 9.4852 | 5.6346 | 5.072 | 3.013 |
| website_phishing | 1000 | 4.2596 | 35.9895 | 23.4333 | 8.449 | 5.501 |
| splice | 1 | 11.7447 | 39.6071 | 21.6620 | 3.372 | 1.844 |
| splice | 64 | 11.9586 | 39.6371 | 23.9606 | 3.315 | 2.004 |
| splice | 1000 | 19.1555 | 52.0133 | 61.9442 | 2.715 | 3.234 |
| SDSS17 | 1 | 3.6756 | 8.6269 | 5.8960 | 2.347 | 1.604 |
| SDSS17 | 64 | 4.1972 | 9.9389 | 9.0907 | 2.368 | 2.166 |
| SDSS17 | 1000 | 11.2312 | 27.6461 | 59.8899 | 2.462 | 5.332 |

## Frozen quality, capacity, and storage

Each row describes one two-component ensemble. Error is 1−AUC (binary), multiclass log loss, or RMSE (regression); comparisons of error across datasets are not meaningful. Serialized pickle bytes include both embedded components and schema; they are not runtime RSS. Individual component IDs, hashes, file sizes, temperatures, and retained rounds are in every timing receipt.

| Dataset | Library | Error | Component rounds | Total rounds | Serialized bytes |
|---|---|---:|---|---:|---:|
| blood-transfusion-service-center | ctboost | 0.4070922 | 15+15 | 30 | 159886 |
| blood-transfusion-service-center | xgboost | 0.36737589 | 28+28 | 56 | 85267 |
| blood-transfusion-service-center | lightgbm | 0.36028369 | 1+1 | 2 | 10659 |
| churn | ctboost | 0.11335042 | 137+137 | 274 | 2147515 |
| churn | xgboost | 0.11523762 | 52+52 | 104 | 241196 |
| churn | lightgbm | 0.11901201 | 6+6 | 12 | 51899 |
| in_vehicle_coupon_recommendation | ctboost | 0.19099426 | 72+72 | 144 | 8712898 |
| in_vehicle_coupon_recommendation | xgboost | 0.17682759 | 235+235 | 470 | 7551492 |
| in_vehicle_coupon_recommendation | lightgbm | 0.18004604 | 424+424 | 848 | 4669036 |
| GiveMeSomeCredit | ctboost | 0.14417422 | 164+164 | 328 | 1293885 |
| GiveMeSomeCredit | xgboost | 0.14256616 | 64+64 | 128 | 168366 |
| GiveMeSomeCredit | lightgbm | 0.14134097 | 95+95 | 190 | 182957 |
| QSAR_fish_toxicity | ctboost | 0.72644383 | 192+192 | 384 | 683343 |
| QSAR_fish_toxicity | xgboost | 0.71906522 | 117+117 | 234 | 716028 |
| QSAR_fish_toxicity | lightgbm | 0.74870668 | 38+38 | 76 | 71037 |
| Another-Dataset-on-used-Fiat-500 | ctboost | 743.51957 | 58+58 | 116 | 1657191 |
| Another-Dataset-on-used-Fiat-500 | xgboost | 759.61821 | 32+32 | 64 | 201244 |
| Another-Dataset-on-used-Fiat-500 | lightgbm | 726.20375 | 21+21 | 42 | 67603 |
| houses | ctboost | 0.22243137 | 167+167 | 334 | 45338939 |
| houses | xgboost | 0.22097125 | 1087+1087 | 2174 | 9320864 |
| houses | lightgbm | 0.21627087 | 380+380 | 760 | 9322653 |
| diamonds | ctboost | 619.11576 | 256+256 | 512 | 7091811 |
| diamonds | xgboost | 615.51438 | 154+154 | 308 | 1644802 |
| diamonds | lightgbm | 622.69999 | 635+635 | 1270 | 3597324 |
| anneal | ctboost | 0.00012566794 | 778+778 | 1556 | 12762710 |
| anneal | xgboost | 0.0024279284 | 240+240 | 480 | 2522828 |
| anneal | lightgbm | 0.00010654156 | 137+137 | 274 | 1114949 |
| website_phishing | ctboost | 0.15323677 | 118+118 | 236 | 2730588 |
| website_phishing | xgboost | 0.17311348 | 243+243 | 486 | 2428343 |
| website_phishing | lightgbm | 0.1698592 | 85+85 | 170 | 809135 |
| splice | ctboost | 0.12119698 | 169+169 | 338 | 7622037 |
| splice | xgboost | 0.11112098 | 69+69 | 138 | 742045 |
| splice | lightgbm | 0.12222517 | 157+157 | 314 | 811505 |
| SDSS17 | ctboost | 0.099698491 | 345+345 | 690 | 7581235 |
| SDSS17 | xgboost | 0.16340454 | 220+220 | 440 | 21320687 |
| SDSS17 | lightgbm | 0.10530843 | 160+160 | 320 | 3060061 |

## Reproduction and evidence

- `latency-plan.json` freezes the original source/runtime/model selection and check-only reference. `run-ensemble-latency.ps1` checks those hashes before launching each fresh process. The frozen absolute paths describe this workspace; reruns need the retained model/data artifacts and dependencies at those paths, or a new explicitly refrozen relocation plan.
- `wave11-final-check-v2.json` is the pre-timing reference; `wave11-ensemble-order-1/2/3.json` are all raw receipts. `summary.json` contains all absolute raw/normalized timings, capacities, component provenance, and comparison aggregates.
- `source/` contains the actual runner, math/preprocessing helpers, summarizer, and six passing synthetic runner tests. `accuracy/` preserves the exact failed-quality plan, selections and development report. The full fitted model/data files stay in the local artifact cache; this archive is not independently executable without them.
- `build_receipt.json` identifies the final native build; each run also captures the actual imported package, native binary SHA256, source hashes, Python and library versions.
- `manifest.json` hashes every archived file except itself. No frozen historical runner or result was changed.
