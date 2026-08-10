# Why CTBoost

Most gradient-boosting libraries greedily search feature and cut-point pairs using
the same training response. CTBoost's defining choice is different: each node first
uses a conditional statistical test to select a feature, then chooses that feature's
split point. The goal is to reduce variable-selection bias while retaining the
iterative error-correction behavior of gradient boosting.

## What CTBoost deliberately keeps

- conditional-inference feature selection at every tree node;
- honest missing-value and categorical routing;
- objective gradients and Hessians feeding the same tree builder;
- reproducible CPU execution and explicit stochastic seeds.

Adding objectives, exports, adapters, tuning, or GPU acceleration must not silently
replace that split mechanism with symmetric trees or a conventional greedy split scan.

## What is already available

- regression, classification, ranking, survival, Gamma/Poisson/Tweedie, AFT,
  callable objectives, multi-output, and multilabel convenience estimators;
- ordered categorical target statistics, text n-grams/TF-IDF, and embedding statistics;
- exact empirical interventional TreeSHAP and SHAP interactions;
- snapshots, warm starts, staged prediction, callbacks, CV and model-selection helpers;
- standalone JSON/Python/C++ predictors, ONNX export, a CLI, and inference manifests;
- columnar, streaming, Dask, Ray, and Spark integration surfaces.

## What is not claimed

CTBoost does not yet match CatBoost/XGBoost's production history, language bindings,
cluster integrations, or full objective catalog. Object influence is a shared-leaf
approximation rather than a deletion/refit counterfactual. Spark training collects to
the driver. Standalone pipeline-backed exports currently expect prepared numeric input.
These limits are tracked explicitly rather than hidden behind broad parity claims.
