# CTBoost vector-leaf JSON predictor format

JSON predictor format `3` adds opt-in multiclass vector leaves to the format `2`
input and preprocessing contract. CTBoost's Python JSON predictor supports it.
The prepared-feature R and JVM readers support formats `1` and `2` and reject
format `3`; their scalar-tree scoring rules must not be applied to vector trees.

## Tree representation

The document declares `multi_strategy: "multi_output_tree"`, a multiclass
objective, and `prediction_dimension >= 2`. Each physical tree represents one
boosting iteration and shares its split topology across every class. Every node
contains a finite `leaf_weights` array of length `prediction_dimension`.
`leaf_weight` remains a finite scalar for persistence compatibility but does not
contribute to vector predictions.

For physical tree `t`, traverse its nodes once, starting at node zero, using the
same numeric and categorical routing rules as the scalar profile. For the reached
leaf, add `tree_learning_rates[t] * leaf_weights[k]` to each output score `k`.
When the learning-rate array is absent or empty, use `learning_rate`. A non-empty
array must have one value per physical tree. Initialize each output from its
`base_score` entry and apply a stable softmax for class probabilities.

Readers must retain the scalar profile's checks for finite values, quantization
dimensions, child indices, cycles, unreachable nodes, and input contracts. They
must also validate the vector strategy, objective, leaf dimensions and rate count.
Format `3` without the vector strategy and vector nodes in formats `1` or `2` are
invalid.

## Inputs and manifests

Prepared feature arrays follow the format `2` contract. Python JSON predictors
also support raw features when `expects_prepared_features` is false, embedding a
validated feature pipeline of format `3` with categorical key encoding `2` and
an inference manifest. This raw-feature mode requires the CTBoost runtime.

Vector inference manifests use schema version `2`; scalar manifests retain
schema version `1`. The model section declares one tree per iteration and
`tree_representation: "shared_topology_vector_leaves"`. Model fingerprints
include `multi_strategy`, complete node vectors and, as in predictor format `2`,
`expects_prepared_features`. The separate preprocessing fingerprint covers any
fitted pipeline.

JSON and standalone Python exports preserve compact physical trees. C++ and ONNX
exports expand vector trees into scalar trees at the export boundary. Their
manifests identify `artifact.tree_representation: "expanded_scalar_trees"` and
record the expanded `artifact.exported_tree_count`; `model.tree_count` still
describes the original physical ensemble.

Native model documents use schema version `3` for vector models and version `2`
for scalar models. The native vector booster state has `format_version: 2`.
