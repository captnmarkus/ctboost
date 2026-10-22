# Prediction preparation profile and single-row categorical prototype

Date: 16 October 2026. Diagnostic profiling only, using the already-fitted
`hiva_agnostic` and `splice` models and the same development feature rows as the
existing inference benchmark. Profiling did not fit models, inspect labels, score
accuracy, or change production behavior or dependencies. A subsequent narrowly
guarded Python candidate is described below.

**The main one-row cost is pandas categorical materialization, not tree traversal
or repeated pipeline-state export.** On hiva, pandas expands 1,617 categorical
columns separately even for one row. For 1,000 rows, native feature transformation
becomes the largest component. Splice has many more trees, so its native prediction
cost is also material for larger batches.

## Scope and identity

The isolated `combined` runtime is
`.tmp/improvements-20261016/followup-native/combined`, with native SHA-256
`565f490d1e9ae4fe2ee7333c9a3bd41ab21375705c294442ab0ae7d925a961fb`.
Python 3.10.14, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.3.2; native version
0.1.61, MSVC 19.50.35737.0, `/O2 /Ob2 /DNDEBUG`. Exact source, model and input
hashes, build receipt, thread settings and package paths are in
[profile_summary.json](profile_summary.json). The script removes the editable
import finder and checks actual Python/native module paths and native hash.

Each profiled scope runs once to warm it, then three times under cProfile.
The process is pinned to CPU 6 with thread environment variables set to one.
Other accuracy fits may run concurrently. **These are attribution diagnostics,
not benchmark latency or optimization speedup claims.** cProfile exaggerates
Python-heavy paths; native calls remain opaque. Independently measured scopes
must not be added or subtracted as exact accounting identities.

Inputs are batches of 1, 64 and 1,000 rows from cached development `X`, repeating
rows only where needed to reach 1,000. The script never accesses the label field.
Float32 transformed matrices match saved reference bits; raw/prepared probability
predictions match saved reference bytes; cached and uncached native raw prediction
bits match for all six cases.

## Observed costs

Milliseconds per request under cProfile, rounded. `Raw library` receives an
already-normalized DataFrame. `Adapter` is the benchmark's
`normalize_tabarena_frame` alone. `Native transform` receives an already-created
object matrix. `Prepared sklearn` receives a reusable transformed Pool. `Native
predict` calls the booster handle on that same Pool.

| Model / rows | Raw library | Adapter | pandas extraction | Native transform | Pool construction | Prepared sklearn | Native predict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hiva / 1 | 49.754 | 7.556 | 47.431 | 0.723 | 0.792 | 0.224 | 0.012 |
| hiva / 64 | 63.122 | 7.625 | 49.298 | 12.247 | 0.811 | 0.237 | 0.037 |
| hiva / 1,000 | 252.920 | 8.153 | 60.859 | 185.817 | 0.849 | 1.449 | 1.229 |
| splice / 1 | 2.114 | 0.428 | 1.814 | 0.024 | 0.107 | 0.068 | 0.017 |
| splice / 64 | 3.189 | 0.413 | 1.886 | 0.451 | 0.105 | 0.673 | 0.635 |
| splice / 1,000 | 15.170 | 0.412 | 2.249 | 6.751 | 0.115 | 5.962 | 5.827 |

Hiva has 3 nontrivial trees, 147 nodes and 22 active transformed features out of
6,468. Its input has 1,617 categorical columns, with 1,617 categorical states
and 1,617 CTR states. Its 1,000-row transformed matrix is 25,872,000 bytes.
Splice has 711 trees, 32,367 nodes and 237 active features out of 240, from 60
categorical columns and 60 CTR states. Its corresponding matrix is 960,000 bytes.
All produced matrices are Fortran contiguous.

### What the stacks establish

* Hiva one-row `_extract_frame -> DataFrame.to_numpy -> BlockManager.as_array ->
  _interleave` calls categorical `astype` 1,617 times per request. Under the
  three-call profile, those 4,851 conversions consume 125 ms cumulatively out of
  142 ms for extraction. pandas dtype handling, categorical `take`, missing-value
  checks and warning-filter setup are repeated per column. Splice shows the same
  structure with 60 conversions per request.
* The benchmark adapter's `frame.copy()` traverses all categorical blocks. In the
  one-row hiva profile, copying takes about 6 ms of the adapter's 7.56 ms. This is
  benchmark-adapter work and must remain separately labeled from CTBoost's public
  prediction API.
* No `to_state` or `_refresh_metadata` call occurs in the raw prediction profiles.
  The estimator and booster share the same pipeline object in both models;
  `_feature_pipelines_equivalent` therefore takes its identity fast path. A
  repeated full metadata export is not supported by this evidence.
* Pool construction is below 0.9 ms for hiva at every batch size. Its Python
  feature-name conversion and categorical-index checks are visible, but the
  measurements do not support blaming a row-scaled dense copy in this scope.
  The prepared sklearn wrapper still validates schema names; it is not identical
  to the native handle scope.
* The isolated native transform dominates large-batch hiva. cProfile cannot
  distinguish normalization, string lookup, CTR calculation, allocation and
  native output metadata conversion within that call. Reading
  `feature_pipeline_transform.cpp` shows that it computes all transformed output
  columns. This supports investigating unused derived-feature work, but does not
  quantify any single C++ loop's share.

## Public pandas one-row prototype

[categorical_row_scout.py](categorical_row_scout.py) tests the public operation
`frame.iloc[0].to_numpy(dtype=object, copy=False).reshape(1, -1)`.
Using it for every all-categorical frame is unsafe:

* An integer category `2**62 + 1` beside a floating category is coerced to float,
  losing integer precision.
* A missing value in an integer-category column beside an integer-category column
  can make `iloc[0]` raise `ValueError: cannot convert float NaN to integer`, while
  the established DataFrame conversion succeeds.
* Integer/string and boolean/integer categorical mixtures can change Python scalar
  types into NumPy scalar types even when values appear equal.

The bounded candidate therefore requires exactly one row and, for every column,
`isinstance(dtype, pd.CategoricalDtype)`, object-typed categories, and
`dtype.categories.inferred_type == "string"`. Other frames retain the original
conversion. Fourteen adversarial frames cover missing strings, identical/different
vocabularies, empty vocabularies, integer and float32 categories, large integers,
numeric and categorical mixtures, and mixed object category vocabularies.
The guarded path preserves exact cell types, values and missingness in these
fixtures. Hiva and splice pass the guard and preserve full transformed and
prediction bits against both the original method and saved references.

Five-call unprofiled diagnostics, with possible concurrent accuracy work:

| One-row case | Original extraction | Guarded extraction | Original raw library | Guarded raw library |
| --- | ---: | ---: | ---: | ---: |
| hiva | 30.849 ms | 18.600 ms | 34.435 ms | 21.488 ms |
| splice | 1.118 ms | 0.791 ms | 1.323 ms | 1.003 ms |

The object-only guard had nearly the same observed cost as the string guard;
there is no measured reason here to broaden the accepted types. Full sample
times and parity outcomes are in
[categorical_row_results.json](categorical_row_results.json). These sequential,
concurrent diagnostics justify an isolated paired benchmark, not a production
speedup claim. The prototype changes the class method only in its own process
and restores it in `finally`; no production source was changed.

### Candidate prepared after the prototype

The subsequent production candidate in `ctboost/feature_pipeline.py` uses a
stricter guard than the diagnostic prototype: exact pandas DataFrame, exactly
one row, ordinary object-typed pandas Index category vocabularies, 1-32 categories
per column, and every category's type exactly builtin `str`. This excludes custom
equality/conversion hooks in string subclasses and bounds vocabulary scanning.
Larger vocabularies, bytes, NumPy strings, numeric categories, nullable text and
DataFrame subclasses retain the existing conversion. No fitted state changes.

The isolated `row-fast` package copies the accepted `combined` runtime and changes
only `feature_pipeline.py`; its receipt retains the same native hash and records
Python SHA-256
`a9cb34f98dfb2f84d1e01126a0b0387451802b0b6e7e921503fd68ccd56da475`.
All 23 cases in `tests/test_feature_pipeline_categorical_row.py` passed with this
isolated package in Python 3.10/pandas 2.3.3. They include native transform parity
for unseen/missing categories and no mutation of fitted pipeline state.
The prototype times above are **not** measurements of this stricter production
guard. Separate paired real-data timing is owned by the inference experiment.

The first paired timing rejected that unconditional one-row guard: it helped
hiva/splice but added roughly 50-60 microseconds to ineligible small mixed/numeric
requests. `row-fast-v2` therefore opts in only for fitted pipelines whose native
metadata has exactly one categorical source state for every raw feature, with no
duplicates/gaps or one-hot/combination/text/embedding states. The flag is derived
during the existing fit/from-state metadata refresh and is not serialized.
Default static extraction and fitting never opt in; ineligible transforms avoid
the dtype scan entirely. The dtype check also requires exact CategoricalDtype.

The revised Python SHA-256 is
`f210e51c916089f0ff62e98d22854b24881c60f9474bb88474c51350dddc9df6`.
All 28 focused cases passed on Python 3.10/NumPy 1.26.4 and
Python 3.12.11/NumPy 2.0.2, both with pandas 2.3.3. The new tests prohibit dtype
inspection on ineligible fitted transforms and verify reconstruction from native
state. The Python 3.12 receipt records the actual installed native binary used
for correctness; it is not represented as the final rebuilt native candidate.
See [row_fast_v2_receipt.json](row_fast_v2_receipt.json) and
[row_fast_v2_cp312_validation.json](row_fast_v2_cp312_validation.json).

After the composed candidate was retained, the final isolated Python 3.12 native
binary `81c25d5316ce3930d94cb7f1916f8b91d4e14e69974f84171fe85d74781c50b2`
passed all 146 requested interoperability cases: row extraction, categorical-key
codec, feature-pipeline persistence, cached prediction and training/prediction
arithmetic. There were no failures or skips, with NumPy 2.0.2 and pandas 2.3.3.
[Final Python 3.12 receipt](final_cp312_validation.json) records exact imported
paths and source/test hashes. The separate real-data paired benchmark validated
the complete composed candidate; its gains must not be attributed only to the
Python shortcut because it also includes native string normalization changes.

## Prioritized next steps

This list records the priorities at the initial profiling stage. The dated
continuation below records completed work and rejected candidates; these are
not all outstanding tasks.

1. Independently benchmark and regression-test the conservative one-row string
   categorical shortcut, including schema names, empty shapes, missing/unseen
   categories and current supported pandas environments. It changes preparation,
   not the conditional tree design.
2. For large hiva batches, profile native transformation more finely before
   assigning cost to string normalization. The separate exact-string native
   candidate is compatible with the pandas optimization. A dependency-aware
   transform could avoid many unused outputs, but would require a carefully
   specified serving path: blindly skipping conversion can change errors or
   custom-object side effects on unused columns.
3. Keep reusable transformed Pools as an explicit serving option when inputs
   repeat. This avoids both pandas expansion and native feature generation;
   prediction still performs its active-feature quantization and traversal.
4. Consider benchmark adapter copying separately. Its public contract includes
   normalized names and categorical membership checks; reducing copies must
   preserve those and input ownership. Report library and adapter scopes apart.
5. Compact traversal storage is more relevant to large-batch splice than one-row
   hiva. Evaluate it separately so improvements cannot be misattributed to the
   categorical preparation path.

Reproduction uses `profile_prediction.py` with `--package` pointing to the
isolated combined runtime, `--scout` to
`.tmp/inference-accuracy-20260908/accuracy-grouped-scout-v1`, and `--reference` to
`.tmp/improvements-20261016/followup-native/real/reference`. Complete cProfile
binary/text outputs and detailed function rows are retained locally in
`.tmp/improvements-20261016-followup/profile/`; the compact committed receipt
contains the source identity and relevant stacks. The single-row prototype uses
the same arguments and writes its own JSON via `--output`.

## Continuation: 16 October 2026, after guarded native validation

The initial priorities led to the following completed experiments. Earlier
profiles, prototypes and negative results above remain historical evidence.

* **Completed and retained:** fitted-metadata opt-in for the conservative
  one-row categorical extraction, exact-string native normalization, and the
  existing bins-only prediction preparation. The first unconditional row guard
  was rejected because of mixed/numeric small-request overhead; only its fitted
  opt-in replacement was retained. The old cProfile and concurrent prototype
  numbers above are not the final latency measurements.
* **Completed and rejected:** broad Python dictionary/`np.take` dispatch. It
  improved some 1,000-row requests but roughly doubled hiva's 64-row cost. The
  native compact-bin and compact traversal-storage candidates also remain
  experimental rather than being silently included in the accepted preparation
  result. Their raw versus prepared/native effects must remain distinct.
* **Completed and retained after separate validation:** native categorical-code
  gather using the unchanged encoder/CTR formulas, guarded by at least 32 fitted
  input columns and 1,000 request rows. The fitted flag is derived once; ordinary
  string-named categorical inputs receive a public dtype precheck before Series
  creation. Unsupported inputs retain normal prediction. The initial native
  broad-dispatch fallback regressions are preserved, followed by a separately
  frozen corrective guard experiment.
* **Still an explicit serving option:** reusable prepared Pools avoid repeated
  feature generation when inputs repeat. This is a different API scope from raw
  prediction. Dependency-aware feature generation and benchmark-adapter copy
  changes were not implemented; the side-effect/error-order caveats above still
  apply to any future work.

The [dictionary/native report](categorical_codes_feasibility.md) traces those
decisions. A [predeclared synthetic crossover](native_crossover/plan.json) selected
1,000 rows independently of the added real batch timings. Its
[original results](native_crossover/comparison.json) also exposed unsupported-frame
overhead, which prevented retaining that broad candidate. The separate
[final guard protocol](native_guard_replication/plan.json) then tested all 96
remaining eligible synthetic cases, the six original fallback controls and 35
real raw-prediction cases in actual-package ABBA order.

The [final comparison](native_guard_replication/comparison.json) passed its frozen
criteria with exact transforms and predictions throughout. At 1,000 rows, hiva
improved from 212.05 to 51.32 ms (4.13x) and splice from 13.69 to 7.84 ms (1.75x).
At 2,048 rows the respective speedups were 6.34x and 1.98x. Small requests use the
existing path; at 64 rows hiva/splice measured 0.995x/1.009x. The aggregate across
all 35 raw cases was 1.136x. These are fixed saved-model local measurements,
including the existing normalization adapter, not a global inference ranking.

The fallback gate was specifically **no regression above 3% in both orders**.
It does not claim that every fallback pooled median improved: the 32-column
late-33-category control measured 0.965x pooled, with order ratios 0.995x and
0.943x. This residual uncertainty is visible in the retained raw records.

The clean final CP312 integration passed 198 interoperability tests with no
failures or skips under NumPy 2.0.2/pandas 2.3.3, including native buffers,
guard boundaries, older-binary fallback, categorical codecs, persistence,
prediction caches and arithmetic. The
[final validation receipt](final_codes_cp312_validation.json) identifies the
actual native binary and Python/test hashes. Production integration extracts
runtime code only from the validated prototype; benchmark builders remain
outside the package.
