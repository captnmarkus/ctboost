# Accuracy development results - 8 September 2026

The original 14-task grouped-eight and temperature candidates failed their admission gates. A separate 37-task grouped-eight extension passed its predeclared quality thresholds after one explicitly recorded XGBoost input-dtype compatibility correction. Library accuracy defaults remain unchanged. These studies establish no Elo gain or result on the official TabArena leaderboard.

The 42-fit comparison used the same 14 metadata-selected tasks as earlier CTBoost development studies. Each task uses only cached official outer-training rows, capped at 10,000 before splitting (seed 47, stratified for classification). Eight folds separate training (folds 0-4), early stopping (5), development scoring (6), and reserved confirmation (7). The models never trained or stopped on the development scoring rows. Historical dataset reuse limits generalization claims; confirmation fold 7 remains unused for fitting or scoring because both gates failed.

All 42 fits succeeded without soft deadlines, hard timeouts, or memory failures. Prediction hashes, labels, role indices, and all metrics were audited; reloading all 42 saved models reproduced predictions exactly. The run took about 10 minutes with one worker, 2 logical CPUs per fit, an 8 GiB process-memory ceiling, and a 300-second training budget. Fit times sum to 463.39 seconds; peak process-tree RSS was 0.877 GiB. Other workloads ran concurrently, so these are descriptive development timings.

The comparator is AutoGluon's manual-default XGBoostModel with seed 47 and matching resource limits: AutoGluon 1.6.2b20260821, XGBoost 3.4.1, CTBoost 0.1.60. It retains its default adaptive early stopping and tree cap. This is a single-model comparison on capped training data, without TabArena's eight-child bagging; it may differ from the package versions used in Lennart's results. Native thread allocation was 2 even though the recorded AG configuration retains its pre-fit n_jobs=-1 default. XGBoost `rounds` records physical training rounds, including patience trees; prediction uses its best-iteration limit.

## Grouped-eight candidate

The candidate changes only the existing numeric feature-test mode to grouped with 8 groups and no multiplicity adjustment. It preserves full quantization/cut candidates and the conditional feature-then-cut principle. All other parameters match CTBoost's TabArena default: 1,000-round cap, learning rate 0.05, depth 6, alpha 0.05, L2 1, Bernoulli subsampling 0.8, ordered CTR, categorical threshold 64, and patience 50. Ordered CTR adds numeric columns even to raw categorical datasets.

Grouped-eight won 8/14 task comparisons, with 0.723% median relative error reduction and 8.134% worst regression. Geometric mean fit-time ratio was 0.898 and the 90th percentile 2.554. The predeclared gate required at least 1% median improvement, at least 8 wins, no task regression above 10%, and fit-time ratios within 3/5; the median improvement requirement failed.

Against default XGBoost, default CTBoost won 7/14 tasks with 1.288% worse median relative error; grouped-eight also won 7/14, with 0.239% better median relative error. These task-level comparisons are not Elo. Default CTBoost won 2/3 binary, 1/3 regression, and 4/8 multiclass comparisons.

Errors are 1 - AUC for binary, RMSE for regression, and log loss for multiclass. Positive grouped gain means lower error than default CTBoost.

| Dataset | Problem | CTBoost default error | XGBoost default error | Grouped error | Grouped gain |
| --- | --- | ---: | ---: | ---: | ---: |
| blood-transfusion-service-center | binary | 0.385106 | 0.387234 | 0.370213 | +3.867% |
| coil2000_insurance_policies | binary | 0.33221 | 0.312943 | 0.323505 | +2.620% |
| GiveMeSomeCredit | binary | 0.146267 | 0.147707 | 0.146977 | -0.485% |
| QSAR_fish_toxicity | regression | 0.760572 | 0.689084 | 0.75753 | +0.400% |
| wine_quality | regression | 0.670388 | 0.629841 | 0.663295 | +1.058% |
| diamonds | regression | 614.78 | 682.544 | 624.198 | -1.532% |
| MIC | multiclass | 0.389079 | 0.418782 | 0.385006 | +1.047% |
| SDSS17 | multiclass | 0.105438 | 0.102768 | 0.102785 | +2.516% |
| anneal | multiclass | 0.0373888 | 0.0241102 | 0.0103917 | +72.206% |
| hiva_agnostic | multiclass | 0.170093 | 0.170131 | 0.170775 | -0.401% |
| maternal_health_risk | multiclass | 0.715634 | 0.607423 | 0.730819 | -2.122% |
| splice | multiclass | 0.12475 | 0.137754 | 0.134897 | -8.134% |
| students_dropout_and_academic_success | multiclass | 0.566301 | 0.616078 | 0.584772 | -3.262% |
| website_phishing | multiclass | 0.245985 | 0.203294 | 0.23003 | +6.486% |

## Temperature candidate

After the grouped study, a separately frozen hypothesis fit one positive temperature per default multiclass model, using only stopping fold 5. Temperatures were optimized in log space within [0.25, 4], retaining T=1 if it was no worse; all eight values were frozen before new development predictions. This reuses the early-stopping/model-selection rows for calibration and the already inspected development panel, so it is explicitly staged exploratory work.

Temperature scaling preserved every predicted class and left all trees untouched. It improved 6/8 tasks, but median log-loss reduction was only 0.195% (required: at least 1%); worst regression was 0.730% (limit: 5%). Maternal-health and website-phishing improved 17.359% and 13.919%, respectively, but those two gains do not replace the failed overall gate. No calibration default or library feature was promoted, and confirmation remained sealed.

| Dataset | Temperature | Default log loss | Calibrated log loss | Gain |
| --- | ---: | ---: | ---: | ---: |
| MIC | 1.04981 | 0.389079 | 0.389937 | -0.221% |
| SDSS17 | 0.866134 | 0.105438 | 0.104581 | +0.812% |
| anneal | 0.852901 | 0.0373888 | 0.037344 | +0.120% |
| hiva_agnostic | 0.989658 | 0.170093 | 0.170065 | +0.017% |
| maternal_health_risk | 0.401379 | 0.715634 | 0.591408 | +17.359% |
| splice | 0.926154 | 0.12475 | 0.125661 | -0.730% |
| students_dropout_and_academic_success | 0.865941 | 0.566301 | 0.564775 | +0.269% |
| website_phishing | 0.764099 | 0.245985 | 0.211747 | +13.919% |

## Reproduction and audit

- [Grouped runner](../../../accuracy_grouped_scout.py), [frozen plan](grouped/plan.json), [all 42 fit records](grouped/fit_records), [complete report](grouped/development_report.json), [artifact audit](grouped/artifact_audit.json).
- [Temperature runner](../../../accuracy_temperature_scout.py), [frozen plan](temperature/plan.json), [frozen temperatures](temperature/frozen_temperatures.json), [complete report](temperature/development_report.json), [artifact audit](temperature/artifact_audit.json).
- [Pinned environment](requirements.txt); SHA-256 inventory in `manifest.json`. Original plan hashes and local predecessor paths are preserved for audit. Models and prediction/target arrays remain local and are intentionally excluded from this report bundle.
- Nine synthetic fits exercised all three model arms across binary, regression, and multiclass data with numeric and categorical columns; saved-model reload predictions were exact. Failed-gate confirmation barriers were tested for both runners.

## Separate 37-task extension

The [initial proposal](next_panel_proposal.json) was subsequently authorized as a separate [frozen 37-task protocol](remaining37/plan.json). It includes every remaining metadata task: 27 binary and 10 regression, with no multiclass tasks or outcome-based filtering. Official task/split and dataset files were snapshotted from the current cache with hashes; train/test partitions are disjoint. The loader reads full cached datasets before retaining official outer-training rows. Only those training rows enter fitted, stopping, or scored roles. Preparation used the existing OpenML-capable environment read-only; execution used the separate public-0.1.60 baseline environment. Both version inventories are recorded in the plan.

The new assessment combines inner folds 6 and 7 of these 37 tasks, separately from training folds 0-4 and stopping fold 5. It does not use the original 14-task panel's reserved fold 7 or revise its failed gates. All three arms, 10,000-row cap, seed, and per-fit limits remain unchanged. Two workers ran the 111 original fits. This is a new development panel following the failed pilot, not an official outer-test evaluation or independent confirmation of the original 14-task gate. It cannot validate multiclass temperature calibration.

### Original run and comparator amendment

The original 111-fit run completed with 110 successes and one failure; its all-success gate remains failed. AutoGluon's direct XGBoost preprocessing passed a mixed bool/numeric DataFrame as an object array to SciPy sparse conversion on online_shoppers_intention. The failure was before learning and is preserved in the original report and fit record.

A separate protocol enumerated all 37 tasks by input dtypes alone before any amended fit. Exactly one task contained a plain boolean column: online_shoppers_intention, Weekend. The frozen rule converts every such column to uint8 with exact 0/1 values for XGBoost fitting, stopping, and prediction. The correction retained Weekend as a feature; other feature values/dtypes, column order, targets, and row IDs were verified unchanged. This changes comparator input compatibility, not CTBoost quality. It did not alter package code, resources, or the original records.

The single corrected XGBoost fit succeeded in 4.14 seconds. The amended 111-record analysis explicitly reuses the other 110 original records by model, prediction, plan, and record hashes. The original failed report remains available separately. All 111 amended saved-model reloads reproduced predictions exactly; all metrics were recomputed and role indices verified disjoint. Model and prediction arrays remain local, outside this report bundle.

### Results on the new assessment

Grouped-eight won 26/37 comparisons against default CTBoost, lost 10, and tied one. Median relative error reduction was 1.971%, and worst regression was 8.602%. Geometric mean fit-time ratio was 0.817 and its 90th percentile 2.451. The amended analysis passes every frozen gate: at least 1% median improvement, at least 19 wins, no regression above 10%, fit-time ratios within 3/5, and all 111 valid records. The original run's gate remains failed.

Against default XGBoost, default CTBoost won 15/37 tasks with 2.482% worse median relative error. Grouped-eight won 21/37 with 0.858% better median relative error. Binary grouped-eight improved over CTBoost on 20/27 tasks (1.971% median) and beat XGBoost on 17/27 (2.556% median relative advantage). Regression improved over CTBoost on 6/10 tasks, with one tie (1.903% median); it still beat XGBoost on only 4/10 and had 1.556% worse median error. These are descriptive task-level comparisons, not Elo or a claim that CTBoost universally matches XGBoost.

The results support evaluating the existing grouped-eight option more broadly while preserving the failed 14-task evidence. They do not establish an accuracy improvement from the separate prediction-preserving inference optimization, a multiclass improvement, or a reason to relabel old evaluations. No library default was changed by this study.

| Dataset | Problem | CTBoost default error | XGBoost default error | Grouped error | Grouped gain |
| --- | --- | ---: | ---: | ---: | ---: |
| APSFailure | binary | 0.0136246 | 0.0109626 | 0.0112416 | +17.490% |
| Amazon_employee_access | binary | 0.255629 | 0.288266 | 0.26323 | -2.974% |
| Another-Dataset-on-used-Fiat-500 | regression | 743.063 | 833.147 | 752.268 | -1.239% |
| Bank_Customer_Churn | binary | 0.125814 | 0.124337 | 0.123802 | +1.599% |
| Bioresponse | binary | 0.192494 | 0.192463 | 0.186339 | +3.198% |
| Diabetes130US | binary | 0.39312 | 0.407372 | 0.389947 | +0.807% |
| E-CommereShippingData | binary | 0.238252 | 0.262795 | 0.242414 | -1.747% |
| Fitness_Club | binary | 0.218712 | 0.238252 | 0.220209 | -0.685% |
| Food_Delivery_Time | regression | 7.66809 | 7.72555 | 7.85608 | -2.452% |
| HR_Analytics_Job_Change_of_Data_Scientists | binary | 0.179454 | 0.184332 | 0.179256 | +0.111% |
| Is-this-a-good-customer | binary | 0.308009 | 0.325418 | 0.266609 | +13.441% |
| Marketing_Campaign | binary | 0.195072 | 0.179702 | 0.188291 | +3.476% |
| NATICUSdroid | binary | 0.0198757 | 0.0182902 | 0.0194839 | +1.971% |
| QSAR-TID-11 | regression | 0.822098 | 0.842619 | 0.822098 | +0.000% |
| airfoil_self_noise | regression | 3.84325 | 2.22248 | 3.25518 | +15.301% |
| bank-marketing | binary | 0.226087 | 0.25057 | 0.226174 | -0.038% |
| churn | binary | 0.110775 | 0.105659 | 0.114003 | -2.914% |
| concrete_compressive_strength | regression | 5.41944 | 5.47991 | 5.28185 | +2.539% |
| credit-g | binary | 0.201207 | 0.214828 | 0.197586 | +1.799% |
| credit_card_clients_default | binary | 0.25424 | 0.241074 | 0.240024 | +5.592% |
| customer_satisfaction_in_airline | binary | 0.0153235 | 0.0130727 | 0.0144096 | +5.964% |
| diabetes | binary | 0.243912 | 0.181142 | 0.185065 | +24.126% |
| hazelnut-spread-contaminant-detection | binary | 0.0804875 | 0.04335 | 0.050375 | +37.413% |
| healthcare_insurance_expenses | regression | 5393.22 | 5085.23 | 4697.59 | +12.898% |
| heloc | binary | 0.194277 | 0.194817 | 0.193144 | +0.583% |
| houses | regression | 0.233315 | 0.226186 | 0.238071 | -2.039% |
| in_vehicle_coupon_recommendation | binary | 0.216971 | 0.273431 | 0.222397 | -2.501% |
| jm1 | binary | 0.284017 | 0.277539 | 0.283843 | +0.061% |
| kddcup09_appetency | binary | 0.218348 | 0.213059 | 0.181771 | +16.752% |
| miami_housing | regression | 107107 | 95783.2 | 100705 | +5.977% |
| online_shoppers_intention | binary | 0.066967 | 0.0671391 | 0.0637662 | +4.780% |
| physiochemical_protein | regression | 4.66738 | 4.33447 | 4.58249 | +1.819% |
| polish_companies_bankruptcy | binary | 0.0737221 | 0.0561585 | 0.0800636 | -8.602% |
| qsar-biodeg | binary | 0.0918549 | 0.0786266 | 0.0846463 | +7.848% |
| seismic-bumps | binary | 0.225213 | 0.213042 | 0.204513 | +9.191% |
| superconductivity | regression | 11.6631 | 11.2711 | 11.4313 | +1.987% |
| taiwanese_bankruptcy_prediction | binary | 0.0703096 | 0.0644566 | 0.0628089 | +10.668% |

The original run used 13.02 minutes from first fit start to last completion. Successful training times sum to 1083.16 seconds; peak process-tree RSS was 0.325 GiB. There were 0 soft-deadline stops and 0 resource failures. Concurrent unrelated workloads make these descriptive development timings.

- Original extension: [runner](../../../accuracy_remaining_scout.py), [frozen plan](remaining37/plan.json), [cache provenance](remaining37/cache_provenance.json), [original failed report](remaining37/assessment_report.json), [all 111 original fit records](remaining37/fit_records), [resource summary](remaining37/resources_summary.json).
- Comparator amendment: [runner](../../../accuracy_xgb_bool_amendment.py), [frozen dtype inventory and rule](remaining37_xgb_bool/plan.json), [amended report](remaining37_xgb_bool/assessment_report.json), [one corrected fit](remaining37_xgb_bool/fit_records), [artifact reuse ledger](remaining37_xgb_bool/reuse_ledger.json), [conversion audit](remaining37_xgb_bool/conversion_audit.json), [111-model artifact audit](remaining37_xgb_bool/artifact_audit.json), [problem-type summary](remaining37_xgb_bool/problem_type_summary.json).

## Isolated multiclass joint-cut experiment

The separately preregistered joint-cut experiment **failed its admission gate** on the eight reused multiclass development tasks: 2/8 wins versus its paired joint-feature/scalar-cut control, -1.062% median relative log-loss improvement, and 77.696% worst regression. All three arms were refitted in an isolated experimental wheel; no default or release was changed. The original confirmation fold remains sealed. [Complete protocol, task-level results, default-equivalence audit, fit times, and physical-tree counts](joint_cut/README.md).

A [secondary comparison of the existing joint-feature/grouped-eight/scalar-cut option](joint_cut/existing_joint_feature_secondary.md) found 4/8 wins over current CTBoost (2.174% median gain; 12.687% worst regression), and 6/8 wins over archived XGBoost. Its larger model sizes and reused development panel prevent treating this as a validated new default or an inference improvement.
