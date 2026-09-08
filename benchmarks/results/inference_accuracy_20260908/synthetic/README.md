# Synthetic inference diagnostic

Four deterministic synthetic cases use 3,000 training rows and 1,000 prediction rows. CTBoost and CatBoost were fitted once with 400 boosting rounds and maximum depth 6; tree topologies and prediction quality are not matched. Every CTBoost revision loads the same saved model and input. This is a CPU inference diagnostic, not an accuracy or Elo evaluation.

The final development wheel is compared below with public 0.1.60. One logical CPU (6), one native thread, three warmups, and seven timing blocks targeting 100 ms per block were used. Prediction includes input preprocessing; prepared-Pool timing is a separate diagnostic. Other training jobs remained active. The public and final runs occurred at different times; the unchanged CatBoost control shows the resulting timing variation. These results should not be read as an exact code-only speedup.

| Case | Rows | Public CTBoost ms | Development CTBoost ms | CatBoost ms in final run | CTBoost speedup | CatBoost control final/public |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numeric_binary | 1 | 0.1662 | 0.1466 | 0.4185 | 1.134x | 1.083x |
| numeric_binary | 128 | 0.5213 | 0.3830 | 0.4716 | 1.361x | 1.141x |
| numeric_binary | 1000 | 2.9488 | 1.6746 | 1.0786 | 1.761x | 1.136x |
| categorical_100 | 1 | 0.4749 | 0.3360 | 1.1425 | 1.413x | 1.486x |
| categorical_100 | 128 | 0.9739 | 0.7229 | 1.1724 | 1.347x | 1.021x |
| categorical_100 | 1000 | 4.0842 | 3.3587 | 2.5086 | 1.216x | 0.888x |
| categorical_2000 | 1 | 2.1812 | 0.3308 | 7.6143 | 6.593x | 1.025x |
| categorical_2000 | 128 | 2.8399 | 0.7796 | 6.0186 | 3.643x | 0.735x |
| categorical_2000 | 1000 | 7.2864 | 3.6123 | 7.0434 | 2.017x | 0.841x |
| numeric_multiclass | 1 | 0.2689 | 0.1481 | 0.3948 | 1.816x | 0.977x |
| numeric_multiclass | 128 | 2.1099 | 0.7608 | 0.6945 | 2.773x | 0.889x |
| numeric_multiclass | 1000 | 15.6449 | 3.6791 | 2.2259 | 4.252x | 0.753x |

All 12 final raw-input and prepared-Pool predictions were checked exactly against slices of the saved public-wheel predictions. Each final row records input/model/prediction file hashes. The final runtime records both native and Python pipeline hashes and the exact runner hash. Earlier records checked the complete 1,000-row prediction and do not separately record a Python pipeline hash.

Cold prediction time excludes model loading and covers the full 1,000-row input. The recorded process RSS delta is allocator- and workload-dependent; it is not a precise cache-size measurement. It is repeated across each case's three batch-size records because only one cold call was made per loaded model.

## Preserved iterations

- `timings-baseline-1.json` and `timings-candidate-1.json`: earlier two-thread runs; retained but not used for the table because of severe timing noise.
- `timings-baseline-serial-1.json`: public 0.1.60 with one CPU/thread.
- `timings-candidate-fusion-serial-1.json`: compact cache and scalar-multiclass traversal sharing.
- `timings-candidate-packet-serial-1.json`: an eight-row traversal experiment that was slower and was removed.
- `timings-candidate-root-serial-1.json`: root-only update optimization while the traversal experiment was still present.
- `timings-candidate-nonpacket-serial-1.json`: traversal experiment removed, before the final tiny-frame/CTR preprocessing revisions.
- `timings-candidate-final-serial-2.json`: final tiny-frame and CTR preprocessing, floating-point fallback, and direct-C++ bounds correction.

The earlier phases share `pre-final-inference_latency.py`. The final `final-inference_latency.py` adds per-batch/raw/prepared parity checks and explicit runner/Python/model hashes; timing loops, models and resource settings are unchanged. The native hash identifies each wheel revision. Earlier experimental binaries remain local; the discarded variants are not part of the final library source.

## Reproduction

Use separate environments with the pinned [requirements](../accuracy/requirements.txt), public CTBoost 0.1.60 in one and the development wheel in the other. Run the [source runner](../../../inference_latency.py) with `python -I benchmarks/inference_latency.py prepare --output <new-directory> --cpus 6 --iterations 400`, then `measure --output <same-directory> --label <unique-label> --cpus 6 --concurrent-hpo` once per environment. Omit `--concurrent-hpo` when no background training is active. Preparation refuses an existing directory, and measurement refuses to overwrite an existing label.

`models.json` records preparation settings. Raw timing blocks, cold calls and memory deltas remain in every timing file; `comparison.json` contains the table values. Model/data/prediction arrays remain local. `manifest.json` inventories this archive by SHA-256.
