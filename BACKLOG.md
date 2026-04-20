# CTBoost Generic Feature Backlog

Last updated: 2026-04-20

## Goal

Prioritize the remaining generic library gaps versus CatBoost and XGBoost without changing CTBoost's core conditional-inference-tree split logic.

This means:

- Prefer additions in Python API plumbing, data ingestion, serialization, training orchestration, packaging, and preprocessing.
- Avoid work that requires rewriting the core split-selection principle or replacing conditional-tree growth.
- If a feature would necessarily change tree-growth semantics, keep it out of the active backlog and mark it as deferred.

## Current Baseline

CTBoost now covers the core tabular baseline and several generic-library layers that used to be open:

- classifier / regressor / ranker
- pandas support
- SciPy sparse support without dense conversion
- categorical, text, and embedding preprocessing through the native feature pipeline
- weights / class imbalance controls
- missing values via `nan_mode`
- multi-metric evaluation, callable eval metrics, and early stopping
- per-iteration callbacks, learning-rate schedules, callback-driven learning-rate control, and built-in logging or checkpoint helpers
- staged prediction
- warm start
- stable JSON and pickle persistence
- snapshot-path resume convenience on top of warm start, with saved config and schema validation
- grouped ranking with `group_id`, `group_weight`, `subgroup_id`, `pairs`, `pairs_weight`, and `baseline`
- richer schema metadata in `Pool`, boosters, and sklearn estimators
- standalone Python export plus a lightweight JSON predictor export
- Python CV helper
- external-memory pool staging

Relevant local references:

- [README.md](README.md)
- [ctboost/core.py](ctboost/core.py)
- [ctboost/training.py](ctboost/training.py)
- [ctboost/sklearn.py](ctboost/sklearn.py)
- [ctboost/_export.py](ctboost/_export.py)

## Priority 0: Keep Out Of Scope

Keep the following out of the active backlog unless the project explicitly allows native learner changes:

- any rewrite of the conditional-inference-tree split-selection principle
- native custom objectives inside the optimization loop
- multi-output objectives or losses
- new survival or ranking objective families that require deeper native optimization changes
- true distributed algorithm redesign beyond the current orchestration layer

The repository already includes `monotone_constraints`, `interaction_constraints`, and survival objectives. Any future redesign of those areas that would require deeper split-logic changes should also stay deferred.

## Priority 1: Exact Snapshot / Resume Guarantees

Why first:

- users now have `snapshot_path`, `resume_from_snapshot`, and `checkpoint_callback(...)`
- the remaining gap is tightening the guarantee surface, not inventing the API from scratch
- this is orchestration work, but it still has real practical value for long-running jobs

Current state:

- `checkpoint_callback(...)` writes model checkpoints from the Python callback surface
- `snapshot_path` writes a resumable checkpoint and `resume_from_snapshot=True` wraps the existing warm-start flow
- `resume_from_snapshot=True` now rejects config or data-schema drift instead of silently reusing an incompatible checkpoint
- resumed training still follows warm-start semantics; exact parity with one uninterrupted run is not promised yet

Scope:

- document and tighten the contract around snapshot equivalence
- improve native periodic snapshot behavior without overselling guarantees
- extend the convenience flow where exactness can be proven

Acceptance:

- resume semantics are explicit and tested
- periodic checkpoint behavior is documented precisely
- any exact-equivalence claim is limited to cases the implementation really satisfies

## Priority 2: Richer Export / Deployment Surface

Why second:

- JSON predictor export is landed, but deployment ergonomics can still improve
- this is high leverage for downstream consumers without changing learner behavior

Scope:

- documented export manifest and prediction contract
- richer metadata bundle for exported predictors
- ONNX export only if the fitted ensemble maps cleanly and can be verified

Caution:

- do not promise CoreML or PMML parity until conversion fidelity is proven

Acceptance:

- one clearer deployment artifact contract beyond local JSON, pickle, Python export, and the current JSON predictor
- round-trip verification tests for any new export surface

## Priority 3: Platform Integration Helpers

Why third:

- useful for adoption
- lower priority than single-node training and deployment ergonomics

Scope:

- thin helpers for partitioned preprocessing or dataset handoff
- wrappers or utilities for Dask-style preprocessing flows
- Spark or Kaggle-oriented integration helpers where the learner itself stays unchanged

Important note:

- thin orchestration wrappers are in scope
- true distributed training redesign is deferred

## Recently Closed

- native sparse ingestion and sparse-preserving pool plumbing
- callback-based training control, including learning-rate schedules and callback-driven learning-rate changes
- callable evaluation metrics with explicit early-stopping safety flags
- schema metadata round-tripping for boosters and sklearn estimators
- lightweight JSON predictor export via `export_format="json_predictor"`

## Suggested Execution Order

1. exact snapshot / resume guarantees
2. richer export / deployment surface
3. platform integration helpers

## Minimal PR Sequence

1. PR 1: tighten snapshot semantics and documented guarantees
2. PR 2: export manifest and deployment helpers
3. PR 3: platform integration utilities

## External Reference Matrix

Official comparison references used for this backlog:

- CatBoost text features:
  - https://catboost.ai/docs/en/features/text-features
- CatBoost embeddings features:
  - https://catboost.ai/docs/en/features/embeddings-features
- CatBoost snapshots:
  - https://catboost.ai/docs/en/features/snapshots.html
- CatBoost Spark:
  - https://catboost.ai/docs/en/concepts/spark-overview
- XGBoost custom metric / objective:
  - https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html
- XGBoost callbacks:
  - https://xgboost.readthedocs.io/en/stable/python/callbacks.html
- XGBoost Dask:
  - https://xgboost.readthedocs.io/en/stable/tutorials/dask.html
- XGBoost Spark:
  - https://xgboost.readthedocs.io/en/release_2.0.0/tutorials/spark_estimator.html
- XGBoost external memory:
  - https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html
- XGBoost multi-output:
  - https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html
- XGBoost survival:
  - https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html
