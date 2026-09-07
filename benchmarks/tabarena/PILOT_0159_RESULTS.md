# CTBoost 0.1.59 validation-only pilot

The [complete audited artifact bundle](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-pilot-0.1.59) contains all original fit records, prediction archives and provenance. Artifact paths below refer to that dataset. The 88 successful fits took 11.94 minutes from the first fit start to the last fit completion; no training deadline was reached.

No task family met the frozen gate; this pilot does not justify the proposed full HPO rerun.

All **88 predeclared fits** are accounted for across 14 datasets. Status counts: ok: 88.

Only official outer-training rows were used. Two fixed inner folds supplied validation scores and early stopping. Outer-test scores and Elo were neither computed nor used for selection. This is an author-run development pilot on noncanonical hardware, not a TabArena leaderboard result, an HPO25 run, or evidence of a library-default promotion.

## Decisions by task family

| Family | Approved mode | Candidate | Passed | Median error reduction | Dataset wins | Geometric mean fit ratio | P90 fit ratio | Reasons |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| binary | baseline | backtracking_3 | False | 0.10% | 2/3 | 1.20 | 1.37 | median_quality_gain_below_threshold |
| regression | baseline | backtracking_3 | False | 0.00% | 3/3 | 0.93 | 0.94 | median_quality_gain_below_threshold |
| multiclass | baseline | full_3 | False | 0.60% | 5/8 | 1.09 | 1.69 | median_quality_gain_below_threshold |
| multiclass | baseline | joint | False | 0.10% | 4/8 | 1.64 | 2.81 | median_quality_gain_below_threshold; no_strict_majority_dataset_wins |
| multiclass | baseline | full_3_joint | False | -0.59% | 4/8 | 1.97 | 3.05 | median_quality_gain_below_threshold; no_strict_majority_dataset_wins |

Admission required at least 1% median relative validation-error reduction, a strict majority of dataset wins, no dataset regression above 10%, geometric mean fit ratio ≤3× and 90th percentile ≤5×, with complete successful paired fits and intact provenance. Failures remain in the report and disqualify their affected comparison. These are pragmatic development thresholds, not statistical significance tests.

## Every dataset and arm

Errors are 1 − ROC AUC for binary, RMSE for regression and log loss for multiclass. Each mean requires both successful inner folds. A dash preserves an undefined comparison after failure; no task is imputed or omitted. Positive error reduction favors the candidate.

| Dataset | Arm | Successful folds | Mean error | Error reduction | Mean fit seconds | Peak RSS GiB | Deadline stops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blood-transfusion-service-center | baseline | 2/2 | 0.256250 | 0.00% | 0.04 | 0.182 | 0 |
| blood-transfusion-service-center | backtracking_3 | 2/2 | 0.260417 | -1.63% | 0.04 | 0.182 | 0 |
| coil2000_insurance_policies | baseline | 2/2 | 0.228492 | 0.00% | 1.32 | 0.192 | 0 |
| coil2000_insurance_policies | backtracking_3 | 2/2 | 0.225583 | 1.27% | 1.51 | 0.189 | 0 |
| GiveMeSomeCredit | baseline | 2/2 | 0.131559 | 0.00% | 62.08 | 0.222 | 0 |
| GiveMeSomeCredit | backtracking_3 | 2/2 | 0.131427 | 0.10% | 88.47 | 0.218 | 0 |
| QSAR_fish_toxicity | baseline | 2/2 | 0.865473 | 0.00% | 0.35 | 0.187 | 0 |
| QSAR_fish_toxicity | backtracking_3 | 2/2 | 0.865473 | 0.00% | 0.32 | 0.189 | 0 |
| wine_quality | baseline | 2/2 | 0.682152 | 0.00% | 4.89 | 0.204 | 0 |
| wine_quality | backtracking_3 | 2/2 | 0.682152 | 0.00% | 4.58 | 0.203 | 0 |
| diamonds | baseline | 2/2 | 551.764989 | 0.00% | 80.90 | 0.248 | 0 |
| diamonds | backtracking_3 | 2/2 | 551.764988 | 0.00% | 75.86 | 0.242 | 0 |
| MIC | baseline | 2/2 | 0.475983 | 0.00% | 17.00 | 0.420 | 0 |
| MIC | full_3 | 2/2 | 0.484171 | -1.72% | 14.08 | 0.391 | 0 |
| MIC | joint | 2/2 | 0.485176 | -1.93% | 48.16 | 0.358 | 0 |
| MIC | full_3_joint | 2/2 | 0.491263 | -3.21% | 51.39 | 0.388 | 0 |
| SDSS17 | baseline | 2/2 | 0.081146 | 0.00% | 112.65 | 0.277 | 0 |
| SDSS17 | full_3 | 2/2 | 0.080259 | 1.09% | 152.60 | 0.295 | 0 |
| SDSS17 | joint | 2/2 | 0.081595 | -0.55% | 131.93 | 0.314 | 0 |
| SDSS17 | full_3_joint | 2/2 | 0.082161 | -1.25% | 134.95 | 0.313 | 0 |
| anneal | baseline | 2/2 | 0.080871 | 0.00% | 4.53 | 0.315 | 0 |
| anneal | full_3 | 2/2 | 0.084363 | -4.32% | 3.69 | 0.291 | 0 |
| anneal | joint | 2/2 | 0.078213 | 3.29% | 12.64 | 0.331 | 0 |
| anneal | full_3_joint | 2/2 | 0.086752 | -7.27% | 11.85 | 0.354 | 0 |
| hiva_agnostic | baseline | 2/2 | 0.176860 | 0.00% | 125.46 | 1.079 | 0 |
| hiva_agnostic | full_3 | 2/2 | 0.176899 | -0.02% | 72.25 | 1.018 | 0 |
| hiva_agnostic | joint | 2/2 | 0.175457 | 0.79% | 142.89 | 1.030 | 0 |
| hiva_agnostic | full_3_joint | 2/2 | 0.174804 | 1.16% | 148.34 | 0.985 | 0 |
| maternal_health_risk | baseline | 2/2 | 0.708501 | 0.00% | 0.38 | 0.194 | 0 |
| maternal_health_risk | full_3 | 2/2 | 0.684378 | 3.40% | 0.85 | 0.197 | 0 |
| maternal_health_risk | joint | 2/2 | 0.603731 | 14.79% | 0.87 | 0.207 | 0 |
| maternal_health_risk | full_3_joint | 2/2 | 0.604139 | 14.73% | 1.19 | 0.212 | 0 |
| splice | baseline | 2/2 | 0.140401 | 0.00% | 19.79 | 0.375 | 0 |
| splice | full_3 | 2/2 | 0.132973 | 5.29% | 23.01 | 0.396 | 0 |
| splice | joint | 2/2 | 0.142925 | -1.80% | 35.80 | 0.430 | 0 |
| splice | full_3_joint | 2/2 | 0.146579 | -4.40% | 46.87 | 0.435 | 0 |
| students_dropout_and_academic_success | baseline | 2/2 | 0.562899 | 0.00% | 5.92 | 0.255 | 0 |
| students_dropout_and_academic_success | full_3 | 2/2 | 0.562336 | 0.10% | 5.99 | 0.267 | 0 |
| students_dropout_and_academic_success | joint | 2/2 | 0.558687 | 0.75% | 7.19 | 0.255 | 0 |
| students_dropout_and_academic_success | full_3_joint | 2/2 | 0.562496 | 0.07% | 10.86 | 0.284 | 0 |
| website_phishing | baseline | 2/2 | 0.285492 | 0.00% | 1.85 | 0.228 | 0 |
| website_phishing | full_3 | 2/2 | 0.278225 | 2.55% | 2.71 | 0.229 | 0 |
| website_phishing | joint | 2/2 | 0.288201 | -0.95% | 1.81 | 0.234 | 0 |
| website_phishing | full_3_joint | 2/2 | 0.279588 | 2.07% | 2.81 | 0.242 | 0 |

## Runtime and reproducibility

Recorded fit time totals 3345.35 seconds across concurrent workers; this sum is not elapsed wall time. Prediction times, secondary binary log loss / multiclass accuracy, retained rounds, per-fit status and all recorded failures are in the JSON/CSV tables and original fit records.

Hardware: AMD Ryzen 7 5800X3D; 8 physical / 16 logical CPUs; 32 GB RAM. Detected platform: Windows-11-10.0.26200-SP0; physical/logical CPUs: 8/16. Selected layout: 8 workers × 2 histogram threads, with 1 BLAS/OpenMP thread per process. Each fit receives 300 seconds plus 30 seconds of watchdog grace. Worker memory cap: 1.79 GiB; aggregate budget: 14.33 GiB. GPU use is disabled.

| Evidence | SHA256 |
| --- | --- |
| Frozen protocol | `1fa7e1b482a9048a52f75d2b554b015ee027da5eddad78bb43908c213c55eff5` |
| Installed native extension | `641bafbb431292a32fea2c3857047bcbd97d849902064e1e1cad2c2f85bc6cfb` |
| Public wheel | `5faca874e14d4c2bf5a41b0b80116a709fbe476bdd99a9d75d4c66203154c55b` |
| Resource contract | `1a621b664b957ae0ce873bf7d449315e37d8e2fadc0701e0575cc684255a03a7` |
| Package environment | `64875522b0d102104030f7b9543e0109aa6fd490becaee4aff6cdac94880e329` |

The complete package versions, CPU affinity, public-wheel identity, preparation hashes and frozen source files are under `provenance/`. `SHA256SUMS` covers every staged artifact. Original JSON records are copied byte-for-byte. Inner-validation prediction archives contain only predictions, encoded labels and indices relative to the outer-training table; raw feature tables and outer-test labels are excluded.

## Limits and next action

A pragmatic development gate on 3 binary, 3 regression and 8 multiclass datasets, not a significance test or proof of generalization. Any final Lite comparison must disclose pilot dataset reuse; held-out test rows remain unused during selection.

Any new 25-configuration portfolio must follow the frozen admission mapping and be committed with its exact configurations and resource contract before final outer-test evaluation. The historical 0.1.58 results remain separate. This report does not claim official leaderboard acceptance.

Underlying datasets and validation labels retain their original terms; no blanket license is asserted over them. Benchmark code follows the [CTBoost repository license](https://github.com/captnmarkus/ctboost/blob/v0.1.59/LICENSE). This package is evaluation evidence and does not redistribute raw feature tables.

## Source tasks

| Dataset | OpenML task | OpenML dataset |
| --- | --- | --- |
| blood-transfusion-service-center | [363621](https://www.openml.org/t/363621) | [46913](https://www.openml.org/d/46913) |
| coil2000_insurance_policies | [363624](https://www.openml.org/t/363624) | [46916](https://www.openml.org/d/46916) |
| GiveMeSomeCredit | [363673](https://www.openml.org/t/363673) | [46929](https://www.openml.org/d/46929) |
| QSAR_fish_toxicity | [363698](https://www.openml.org/t/363698) | [46954](https://www.openml.org/d/46954) |
| wine_quality | [363708](https://www.openml.org/t/363708) | [46964](https://www.openml.org/d/46964) |
| diamonds | [363631](https://www.openml.org/t/363631) | [46923](https://www.openml.org/d/46923) |
| MIC | [363711](https://www.openml.org/t/363711) | [46980](https://www.openml.org/d/46980) |
| SDSS17 | [363699](https://www.openml.org/t/363699) | [46955](https://www.openml.org/d/46955) |
| anneal | [363614](https://www.openml.org/t/363614) | [46906](https://www.openml.org/d/46906) |
| hiva_agnostic | [363677](https://www.openml.org/t/363677) | [46933](https://www.openml.org/d/46933) |
| maternal_health_risk | [363685](https://www.openml.org/t/363685) | [46941](https://www.openml.org/d/46941) |
| splice | [363702](https://www.openml.org/t/363702) | [46958](https://www.openml.org/d/46958) |
| students_dropout_and_academic_success | [363704](https://www.openml.org/t/363704) | [46960](https://www.openml.org/d/46960) |
| website_phishing | [363707](https://www.openml.org/t/363707) | [46963](https://www.openml.org/d/46963) |

The [public preregistration](https://github.com/captnmarkus/ctboost/blob/35a95b77535dc8abebcb1459016764ccf2aba4ee/benchmarks/tabarena/pilot_0159_v1.json) preceded the first pilot fit. `provenance/preregistration.json` records both the exact frozen Windows JSON hash and the public Git blob hash; their only difference is CRLF versus LF line endings. The frozen protocol is copied byte-for-byte here.
