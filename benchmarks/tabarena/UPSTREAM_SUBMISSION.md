# CTBoost TabArena upstream submission

This is the hand-off checklist for a TabArena-v0.1 CPU submission. CTBoost's
conditional-inference split selection remains unchanged; the wrapper only supplies
TabArena data, resources, validation, seeds, and result metadata.

## Required upstream patch

Base the contribution on the current TabArena `main` after (or rebased over)
[PR 468](https://github.com/autogluon/tabarena/pull/468), because AutoGluon 1.6
rejects the former override-method model API. Add:

1. `packages/tabarena/src/tabarena/models/ctboost/__init__.py`
2. `packages/tabarena/src/tabarena/models/ctboost/model.py`
3. `packages/tabarena/src/tabarena/models/ctboost/hpo.py`
4. `packages/tabarena/src/tabarena/models/ctboost/info.py`
5. Lazy `CTBoostModel` export in `packages/tabarena/src/tabarena/models/__init__.py`
6. `ctboost = ["ctboost>=0.1.58"]` and `tabarena[ctboost]` in the `extended`
   extra in `packages/tabarena/pyproject.toml`
7. `CTB` in the tree-model family mapping in
   `packages/tabarena/src/tabarena/website/website_format.py`. Do not add it to
   the raw display-prefix rename map: `ag_name="CTBoost"` already supplies the
   display name, and a raw `CTB -> CTBoost` replacement corrupts `CTBoost` into
   `CTBoostoost`.
8. A small-iteration `CTBoost` entry in `tests/tabarena/models/smoke_configs.py`

After a complete run is processed and hosted, maintainers add the verified method
metadata to `packages/tabarena/src/tabarena/contexts/tabarena/methods.py` and record
the cluster invocation in `packages/tabflow_slurm/BENCHMARK_LOG.md`.

The wrapper contract is:

- `ag_key = "CTB"`, `ag_name = "CTBoost"`, `ag_priority = 65`
- `_supported_problem_types = ["binary", "multiclass", "regression"]`
- `_default_auxiliary_params_extra` for bool/int/float/category/object inputs
- native pandas category/object/string, missing-value, boolean, and unseen-value
  semantics preserved without sentinel replacement or string coercion
- `default_resources_physical_cores_only = True`, CPU compute
- AutoGluon validation data used for CTBoost early stopping
- `num_cpus` applied through `CTBOOST_HIST_THREADS`
- `random_seed` supplied by AutoGluon's fold/config seed contract
- finite `time_limit` enforced by the between-tree callback
- a conservative static memory estimate for fold-parallel scheduling
- no explicit warm-up: after AutoGluon, NumPy, and pandas are loaded, CTBoost's
  import is small and it has no JIT or separate warm-up API

The HPO generator is the frozen deterministic portfolio in `ctboost_model.py`:
one manual default plus a progressively ordered 200-point Latin-hypercube
design, depth 3-8, learning rate 0.02-0.2, conditional leaf/CTR parameters,
learning-rate-scaled 400-1,600 tree caps, and 30-80 round early-stopping
patience. TabArena's `CustomAGConfigGenerator` accepts this deterministic
callable directly; ConfigSpace sampling is not required by the upstream API.

Some ordered-CTR configurations request a two- or four-pair categorical budget.
The wrapper resolves it from training-fold cardinalities only, considers at
most 16 non-constant low-cardinality columns, rejects products above 4,096, and
passes explicit pairs rather than enabling the quadratic all-pairs option. The
portfolio contains no `random_seed`: TabArena-v0.1's fold-config-wise seed
contract injects disjoint seeds through `seed_name = "random_seed"`.

These choices are generated from fixed seed `1234` and frozen independently of
TabArena dataset metadata, validation outcomes, and test metrics. They must not
be changed after inspecting TabArena-Full test performance; a future revision
requires a newly identified portfolio and fresh artifact directory. Adaptive
tree caps never replace or extend the official 3,600-second per-fit deadline.

## Required evidence before cluster time

- `FitHelper.verify_model` passes binary, multiclass, and regression under the
  exact AutoGluon/TabArena dependency versions.
- All 201 configs fit a tiny binary, multiclass, and regression dataset.
- The generated configs are deterministic, unique, and prefix-stable;
  DepthWise has no partial leaf cap, LeafWise caps stay below
  `2**max_depth`, CTR smoothing/pair budgets are emitted only when ordered CTR
  is enabled, and no config hard-codes a model seed.
- Pair-budget preflights cover no-category, constant/high-cardinality, binary,
  multiclass, and regression inputs and never produce more than four explicit
  pairs.
- A default-only TabArena-Full run has 816/816 raw outer-split artifacts with no
  failures, non-finite metrics, or resource-limit violations.
- The full default + 200 run has 164,016/164,016 raw outer-split artifacts. Each
  outer split contains eight AutoGluon bag folds (1,312,128 child fits total).
- Raw-to-results processing produces default, tuned (`n_iterations=1`), and tuned
  plus ensemble (`n_iterations=40`) rows. No CTBoost-specific ensemble code is
  required.

The published [0.1.56 Lite artifacts](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-0.1.56)
cover all 51 datasets at `r0f0`, with one default configuration and no imputed
tasks. They do not satisfy the full-run gates or establish a 0.1.58 score.
The older smoke JSON files remain historical integration evidence.

## Pull request text

Suggested title: `[New Model] CTBoost`

The body should state that CTBoost is a standalone CPU conditional-inference-tree
gradient booster, link the repository/PyPI/docs/license, list the three supported
task types, and document native categorical/missing handling. Include the exact
fairness contract above, the 201-config preflight counts, default-full artifact
count, hardware/resources, and the contribution license statement used by other
TabArena model PRs. Explicitly say that the search space was frozen without using
TabArena-Full test results.

The integration is under review in [PR #479](https://github.com/autogluon/tabarena/pull/479).
Lennart requested a limited HPO check of about 25 configurations and offered to
review full-suite results for leaderboard inclusion. Complete that check using
the frozen portfolio, then report the version, coverage, failures, and artifact
link briefly in the existing PR. Ask which full-suite artifact set and transfer
channel the maintainers need for 0.1.58. Benchmark execution is the contributor's
responsibility unless maintainers explicitly offer compute.

No Hugging Face token is needed to download public TabArena data or run the CPU
benchmark. Publication to TabArena's artifact storage is a maintainer step and may
require their R2 credentials.
