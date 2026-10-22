# Matched-library latency diagnostic, frozen wave6

This diagnostic compares the completed wave6 selected models using the previously validated composed CTBoost runtime, XGBoost 3.4.1 and LightGBM 4.7.0. It is not the final model/runtime selection and does not claim equal quality or a leaderboard win.

All three libraries receive the same raw development rows and identical train-only vocabulary normalization. Raw latency includes that shared normalization plus each library's fastest checked standard DataFrame margin prediction: CTBoost predict, XGBoost inplace_predict, and LightGBM predict with one thread. Normalized-frame latency excludes only shared normalization. CTBoost's fitted feature pipeline still runs in both paths.

Three fresh-process passes rotate the library order so each library occupies each position once. One fixed CPU/thread, three warmups, five calibration calls, and seven blocks targeting 50 ms are used. All team fitting/build/test work was paused. Every selected model reproduces its frozen development predictions exactly, including XGBoost inplace margins. Models and inputs are hash checked; runtime binaries, Python sources, versions, capacities and development quality are recorded.

Speedup below is competitor latency divided by CTBoost latency; values above 1 mean CTBoost is faster. Aggregates are geometric means over 12 datasets (36 dataset/batch points for all rows). They combine models with different quality and capacity; per-model development error and retained rounds accompany every result.

| Batch rows | CTBoost vs XGBoost raw | CTBoost vs LightGBM raw | CTBoost vs XGBoost normalized | CTBoost vs LightGBM normalized |
|---|---:|---:|---:|---:|
| all | 2.266x | 2.010x | 7.427x | 5.480x |
| 1 | 2.300x | 1.507x | 15.448x | 6.908x |
| 64 | 2.281x | 1.824x | 8.275x | 5.601x |
| 1000 | 2.216x | 2.953x | 3.206x | 4.252x |

| Dataset | Rows | CTBoost raw, ms | XGBoost raw, ms | LightGBM raw, ms | CTBoost / XGBoost / LightGBM development error |
|---|---:|---:|---:|---:|---|
| blood-transfusion-service-center | 1 | 0.5127 | 1.2925 | 0.6717 | 0.39504 / 0.37163 / 0.33759 |
| blood-transfusion-service-center | 64 | 0.5747 | 1.3142 | 0.6845 | 0.39504 / 0.37163 / 0.33759 |
| blood-transfusion-service-center | 1000 | 0.6675 | 1.5593 | 0.8206 | 0.39504 / 0.37163 / 0.33759 |
| churn | 1 | 2.5454 | 4.8490 | 3.6739 | 0.11975 / 0.12942 / 0.12528 |
| churn | 64 | 2.7490 | 4.9982 | 3.7269 | 0.11975 / 0.12942 / 0.12528 |
| churn | 1000 | 4.4772 | 6.5183 | 4.1311 | 0.11975 / 0.12942 / 0.12528 |
| in_vehicle_coupon_recommendation | 1 | 3.4045 | 9.6770 | 6.0461 | 0.20413 / 0.1926 / 0.19244 |
| in_vehicle_coupon_recommendation | 64 | 3.7467 | 10.8637 | 9.1458 | 0.20413 / 0.1926 / 0.19244 |
| in_vehicle_coupon_recommendation | 1000 | 7.5389 | 25.7409 | 54.4927 | 0.20413 / 0.1926 / 0.19244 |
| GiveMeSomeCredit | 1 | 1.2635 | 2.2269 | 1.4799 | 0.14687 / 0.14603 / 0.14 |
| GiveMeSomeCredit | 64 | 1.3982 | 2.3183 | 1.6812 | 0.14687 / 0.14603 / 0.14 |
| GiveMeSomeCredit | 1000 | 2.2550 | 3.1277 | 4.0391 | 0.14687 / 0.14603 / 0.14 |
| QSAR_fish_toxicity | 1 | 0.7635 | 1.6103 | 0.9409 | 0.78036 / 0.69789 / 0.75531 |
| QSAR_fish_toxicity | 64 | 0.8457 | 1.7449 | 1.0175 | 0.78036 / 0.69789 / 0.75531 |
| QSAR_fish_toxicity | 1000 | 1.0679 | 3.0892 | 1.8367 | 0.78036 / 0.69789 / 0.75531 |
| Another-Dataset-on-used-Fiat-500 | 1 | 0.9201 | 2.1836 | 1.6738 | 758.81 / 768.05 / 743.94 |
| Another-Dataset-on-used-Fiat-500 | 64 | 1.0013 | 2.2409 | 1.7422 | 758.81 / 768.05 / 743.94 |
| Another-Dataset-on-used-Fiat-500 | 1000 | 1.7810 | 2.8801 | 2.4234 | 758.81 / 768.05 / 743.94 |
| houses | 1 | 1.0260 | 2.0318 | 1.2679 | 0.22567 / 0.22503 / 0.22082 |
| houses | 64 | 1.4158 | 3.2321 | 4.3578 | 0.22567 / 0.22503 / 0.22082 |
| houses | 1000 | 5.3779 | 15.9343 | 49.0528 | 0.22567 / 0.22503 / 0.22082 |
| diamonds | 1 | 1.3249 | 2.9632 | 2.3337 | 637.05 / 626.67 / 654.26 |
| diamonds | 64 | 1.4902 | 3.0912 | 5.2488 | 637.05 / 626.67 / 654.26 |
| diamonds | 1000 | 3.3342 | 4.7586 | 48.0177 | 637.05 / 626.67 / 654.26 |
| anneal | 1 | 5.2791 | 14.0977 | 8.8813 | 0.0077902 / 0.053728 / 0.0010378 |
| anneal | 64 | 6.2204 | 14.8606 | 10.5780 | 0.0077902 / 0.053728 / 0.0010378 |
| anneal | 1000 | 16.3516 | 26.2648 | 34.4101 | 0.0077902 / 0.053728 / 0.0010378 |
| website_phishing | 1 | 1.3216 | 4.4822 | 2.7844 | 0.18281 / 0.16206 / 0.17443 |
| website_phishing | 64 | 1.4354 | 5.3147 | 3.3935 | 0.18281 / 0.16206 / 0.17443 |
| website_phishing | 1000 | 2.5160 | 18.2875 | 12.2234 | 0.18281 / 0.16206 / 0.17443 |
| splice | 1 | 11.1848 | 24.3682 | 15.2306 | 0.11179 / 0.11883 / 0.12127 |
| splice | 64 | 10.2352 | 24.5374 | 16.3000 | 0.11179 / 0.11883 / 0.12127 |
| splice | 1000 | 20.3564 | 30.6718 | 35.4575 | 0.11179 / 0.11883 / 0.12127 |
| SDSS17 | 1 | 2.3427 | 4.8770 | 3.4797 | 0.10232 / 0.17822 / 0.1034 |
| SDSS17 | 64 | 2.5781 | 5.5261 | 5.0692 | 0.10232 / 0.17822 / 0.1034 |
| SDSS17 | 1000 | 5.4631 | 13.6843 | 29.8511 | 0.10232 / 0.17822 / 0.1034 |

Development errors use the frozen accuracy study metric for each dataset; comparisons are meaningful within a dataset, not across different task types. Serialized model_bytes refers to the saved payload file, not resident memory. Large model/input fixtures are retained locally and identified by hashes, rather than bundled in this evidence archive.
