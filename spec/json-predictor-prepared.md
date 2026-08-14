# CTBoost prepared-feature JSON predictor profile

Status: inference-only interoperability profile for JSON predictor format versions 1 and 2.

This profile lets non-Python runtimes score a fitted CTBoost tree ensemble from
already-prepared numeric features. It does not train trees and therefore cannot
change CTBoost's conditional-inference feature selection or split selection.

## Required envelope

An artifact must be a JSON object with:

- `format` equal to `ctboost-json-predictor`;
- `format_version` equal to `1` or `2`;
- `expects_prepared_features` equal to the JSON boolean `true`;
- a positive `prediction_dimension` and a non-negative `num_features`;
- `base_score` with exactly `prediction_dimension` finite numbers;
- a quantization schema and at least one tree.

Binary objectives require `prediction_dimension == 1`; multiclass objectives
require `prediction_dimension >= 2`.

The R and JVM runtimes in this repository deliberately fail closed when
`expects_prepared_features` is false. A `feature_pipeline_state` may be present
for metadata compatibility, but these runtimes never execute it.

Producers should omit `feature_pipeline_state` from prepared artifacts. It is
unused here and may contain unnecessarily sensitive fitted preprocessing state.

All arrays use zero-based model indices, even in the R implementation. JSON
numbers used as indices must be exact integers. Non-finite JSON numbers are not
part of this profile. Prepared input values are converted to IEEE-754 binary32
before binning; accumulation and link functions use binary64.

## Quantization

For feature `f`, cuts occupy
`cut_values[cut_offsets[f]:cut_offsets[f + 1]]`.

- Missing input (`NaN`) maps to the last bin for nan mode `2` (`Max`) and bin
  zero otherwise.
- Numeric input maps to the number of cut values less than or equal to it
  (`upper_bound`), plus one when a `Min` missing bin precedes non-missing bins.
- Categorical input maps to the first cut greater than or equal to it
  (`lower_bound`), clamped to the final non-missing bin, with the same missing
  offset.

`nan_modes`, when non-empty, has one entry per feature. Otherwise `nan_mode` is
used for every feature.

## Tree evaluation

Each tree starts at node zero. A numeric node goes left when the feature bin is
less than or equal to `split_bin_index`. A categorical node indexes
`left_categories` by feature bin and goes left for a non-zero entry. Child
indices are relative to that tree's node array.

Trees are interleaved by output dimension. Tree `t` contributes to output
`t % prediction_dimension`; its iteration is
`floor(t / prediction_dimension)`. The scale is the corresponding entry in
`tree_learning_rates`, when present, otherwise `learning_rate`.

Binary objectives (`Logloss`, `binary_logloss`, `binary:logistic`) use a stable
sigmoid and return probabilities in negative/positive order. Multiclass
objectives (`MultiClass`, `softmax`, `softmaxloss`) use a stable softmax.

## Validation requirements

Readers must validate before scoring:

- all parallel quantization arrays and cut offsets;
- all feature, bin, child, and category-route indices;
- exact base-score dimensionality and tree-count divisibility;
- finite model parameters;
- that every tree is rooted at zero, acyclic, and all nodes are reachable.

Readers must cap traversal by the validated node count. Unknown objectives may
be scored as raw margins, but probability/class methods must reject them.

Readers must reject duplicate JSON object keys. The format and optional
manifest fingerprints are not a signature or authentication mechanism; load
artifacts only from trusted sources and apply a bounded artifact-size limit.

The machine-readable prepared profile is
[`json-predictor-prepared.schema.json`](json-predictor-prepared.schema.json).
Cross-language golden cases live in `tests/export_conformance`.
