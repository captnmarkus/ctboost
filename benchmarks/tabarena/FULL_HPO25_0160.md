# CTBoost 0.1.60: full TabArena evaluation with 25 HPO configurations

This run evaluates the published CTBoost 0.1.60 CPU wheel on **every outer split
of TabArena-v0.1**, using the default configuration and the first 25 entries of
the existing frozen HPO portfolio. It replaces the unfinished Lite HPO200 run.
The old run's artifacts remain separate; no old result is silently relabelled
or included in this run.

## Coverage and learning protocol

- 51 datasets, with all 816 official outer splits: 34 datasets have nine splits
  each, and 17 have thirty splits each.
- 26 configurations per split: one default and 25 HPO configurations.
- Eight sequential bagging folds per configuration, without extra bagging sets.
- **21,216 parent results and 169,728 child fits** in a complete evaluation.
- The official train/test indices, metrics and preprocessing are retained.
  Exact split indices and row counts are checked; averaged CSV row counts are
  descriptive metadata, not substitutes for the actual split sizes.
- Child seeds are `config_index * 8 + inner_fold`, following the pinned
  TabArena experiment definitions. No extra outer-split seed offset is added.
- The original conditional-inference feature and split selection rules remain
  unchanged. Optional learning modes are not promoted by benchmark outcomes.

The configuration order and full task/split population are fixed before fits.
Only out-of-fold validation predictions may determine tuning and ensemble
weights. Test predictions are final evaluation evidence, not selection inputs.

## Resources and execution

The user selected the combined author-run allocation: **two logical CPUs and
8 GiB per parent**, zero GPUs, a 3,600-second training limit and a 4,500-second
parent wall limit. These differ from TabArena's standard eight-CPU/32-GB preset.
Timing results must be described as author-run timings on heterogeneous hardware;
they are not canonical TabArena timings.

The local controller admits up to eight parents with disjoint CPU affinities,
subject to observed memory needs and a 4 GiB free-memory reserve. Remote workers
admit up to two parents per four-CPU instance, with the same per-parent limits.
The fixed partition assigns four of every nine parents locally and five remotely:
9,431 local parents and 11,785 remote parents. The first remote bundle contains
one parent as a small transport check; subsequent bundles contain twelve parents
each. Other bundles may start concurrently when account capacity permits, as
implemented by the frozen controller. Remote collection runs every twenty minutes.

The complete plan is published before fitting. Workers verify its public
registration, source digests, pinned TabArena checkout, package environment and
public wheel bytes. Every result is bound to its configuration, task, repeat,
fold, child seeds and actual runtime. Cache paths include both repeat and fold.

Started failures are terminal and remain visible. Missing or failed results are
not replaced by baselines or counted as completed. Resuming a controller only
dispatches never-started parents; changing this policy requires a deliberate,
separately recorded decision.

## Completion and publication

Publication is gated on exact membership and validation of all 21,216 parents,
all 816 outer splits and all eight children per parent. The finalizer uses
TabArena's official result-processing API. Public artifacts contain canonical
result tables, metadata, completeness and resource records, and checksums.
Raw prediction/target pickles and intermediate processed artifacts remain local.

After validation, publish a new dataset at
`Maiernator/ctboost-tabarena-full-hpo25-0.1.60`, verify its public files, and write
a concise reply for the contributor to post. No comment or update is sent to
the TabArena PR automatically. An incomplete run must never be presented as a
completed Full evaluation.
