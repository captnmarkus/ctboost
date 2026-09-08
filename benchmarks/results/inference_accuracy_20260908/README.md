# Inference and accuracy development - 8 September 2026

This work improves CTBoost's CPU prediction path and tests accuracy candidates
while retaining conditional feature selection before cut optimization. It is
development work based on public 0.1.60, not a new published release. Existing
0.1.60 evaluations retain their original source, runtime and settings.

The prediction changes remove repeated metadata serialization, add typed numeric
preprocessing and reuse multiclass CTR category lookups. A compact native cache
stores traversal data separately from training nodes. Matching scalar multiclass
trees share traversal; root-only trees add updates directly. Model formats and
physical leaf-index columns remain unchanged. The cache adds first-call work and
memory and is invalidated when model state changes. Fast-math builds and GCC
targets with native FMA retain the original score-accumulation path to preserve
their rounding behavior.

## Evidence

The final real-data pair measured a geometric 1.7568x CTBoost speedup across
14 identical saved models, with exact prediction parity and improvements on
all 14 tasks. The unchanged CatBoost control's geometric timing ratio was
0.997, with individual ratios from 0.699 to 1.330. Development CTBoost was
faster than the saved CatBoost models on 11/14 tasks, with different model
sizes and prediction quality; this is not an accuracy-matched comparison.

- [Real-data inference](inference/README.md): identical saved CTBoost models,
  complete input preprocessing, public and development wheels, and CatBoost
  comparisons with model sizes and timing variation disclosed.
- [Synthetic inference](synthetic/README.md): four fixed cases and three batch
  sizes, including previous experiments and the discarded traversal variant.
- [Accuracy studies](accuracy/README.md): fixed parameters and row roles,
  complete per-task results, failed candidates, and artifact hashes.
- [Floating-point probe](fp_contract/README.md): the GCC contraction mismatch,
  its target-specific fallback, and reproduction commands.
- [Direct C++ bounds regression](tree_bounds/README.md): six previously missing
  exceptions now work; all 22 checks pass, including valid-input controls.

The 37-task grouped-test extension reduced median error by 1.971% versus CTBoost
and beat default XGBoost on 21/37 tasks, with 0.858% better median error. Its
separate original 14-task gate failed, and regression still trailed XGBoost on
the extension. Grouped tests are an existing opt-in control; these results do
not justify a global default change. Temperature calibration also failed its
predeclared gate.

Existing joint multiclass feature tests combined with grouped-eight improved
median multiclass log loss by 2.174% versus CTBoost and beat default XGBoost on
6/8 development tasks (1.733% median relative advantage). They also regressed
on four CTBoost comparisons and produced 1.575x as many physical trees by
geometric mean. This is an exploratory option comparison, not a promoted
default or an independent confirmation.

A separate [timing-only comparison of these existing options](inference/learning_options/README.md)
reused grouped-eight models for six binary/regression tasks and joint-feature,
grouped-eight models for eight multiclass tasks. All predictions remained exact
under the development wheel. These models were faster than the saved CatBoost
models on 13/14 tasks (geometric latency ratio 0.518); quality and model sizes
are not matched between libraries. Multiclass latency was 1.009x that of the
CTBoost defaults while the CatBoost control was 0.899x, so the larger models
are not a free accuracy gain. No extra fits or confirmation scores were used.

An isolated new joint-cut implementation passed correctness tests but failed
its quality gate: 2/8 wins and 1.062% worse median log loss versus its matched
control. It is excluded from the library changes. Its source is retained on
the local `experiment/joint-cut-20260908` branch at
`6c3fed74ebbbda8350fa521bcca44cc715c2f81a`; the full results remain in the
accuracy archive. The [prototype patch](joint-cut-prototype.patch) against the
recorded base commit permits reproduction without merging it into this branch.
Reserved confirmation rows remain unscored.

The final installed Windows CPU wheel passed 1,776 Python tests, including
generated C++ DLL exports; 35 tests were skipped for unavailable optional
dependencies, CUDA build support, or checkout-only source checks. The separate
direct-C++ bounds probe passed 22 checks.
GCC FMA behavior was checked with a standalone compiled probe, not a complete
Linux or ARM wheel test. The final wheel, native extension and Python pipeline
hashes are recorded in the inference and synthetic archives.
The [validation receipt](validation.json) records the implementation commit and
source hashes; [the complete installed-test log](windows-installed-tests.log)
preserves individual skip reasons. Archive file hashes are in `manifest.json`.

All quality comparisons use capped outer-training data and inner assessment
splits. They are single-model development studies, without TabArena's full
bagging and official evaluation protocol. They establish no Elo increase or
universal match to XGBoost. Latency measurements were made with other training
jobs active and cannot establish canonical TabArena timing or universal parity
with CatBoost. Exact prediction parity validates the inference optimizations;
it does not imply equal prediction quality between different libraries.
