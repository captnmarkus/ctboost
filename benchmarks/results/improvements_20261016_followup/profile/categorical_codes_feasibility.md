# Prediction-only categorical dictionary experiment

This experiment targets repeated preprocessing work without changing fitted trees,
conditional feature selection, ordered training CTRs, or persistence. Source:
`benchmarks/categorical_codes_prototype.py`; focused tests:
`tests/test_categorical_codes_prototype.py`. The builder copies an isolated runtime
and patches only that copy's Python pipeline. The checkout package stays unchanged.

Native `TransformInternal` normalizes and hashes every categorical cell, then
repeats normalization in `JoinNormalizedKey` and computes each inference CTR.
Singleton CTR outputs depend only on the current category and fitted statistics,
so a small category table can compute these values once and integer codes can
gather them. This is broader than assuming string conversion alone is the culprit:
cProfile cannot separate normalization, hashing and arithmetic inside native code.

Hiva has 1,617 categorical sources and 6,468 transformed outputs. At 1,000 rows
the 25,872,000-byte float32 output is unavoidable in this prototype, while string
work can shrink from row-count times feature-count to used-category-count times
feature-count. Compact prediction bins optimize a later stage. No quantitative
speed claim follows from this complexity observation.

Pandas exposes dictionaries and codes publicly; missing is code -1 and input
dictionary ordering need not match fitted categorical encoding.
[Pandas 2.3 categorical guide](https://pandas.pydata.org/pandas-docs/version/2.3/user_guide/categorical.html)

The prototype derives output-to-source correspondence during the existing fitted
metadata refresh. It admits only exact pandas DataFrames with ordinary object
string categoricals, at most 32 vocabulary entries per feature, and more than one
row. Fitted states must cover every raw feature exactly once with categorical
encoders and use only singleton-source CTRs. One-hot, combinations, numeric/mixed,
text, embedding, extension storage and custom string cases fall back.

Each dictionary contains only codes actually present, in first-encounter order.
Padding repeats an actually used value, so it neither converts unused malformed
Unicode nor invents a missing value that could change errors. The existing native
inference transform runs on that table. NumPy then gathers each output column
into a Fortran-contiguous float32 matrix. Mean/Frequency formulas, float/double
rounding, legacy codecs and old smoothing all use the original implementation.
Even singleton `JoinNormalizedKey` escaping remains intact.

The public one-row shortcut and all fitting paths are unchanged. There is no
cross-request dictionary cache, persisted state migration, custom-object retry,
or new native binding. Using public per-column extraction and NumPy calls has a
fixed cost, which might outweigh savings for small or high-cardinality requests;
isolated paired measurements and fallback controls must decide retention.

If this experiment wins, a later private native code-gather interface could avoid
the Python loop and temporary index arrays. That would require explicit code
dtype/shape/stride/bounds validation, owner retention, shared CTR helpers and the
same equivalence tests. It is unnecessary before demonstrating useful upside.

## Initial correctness result

All 34 focused cases passed on Python 3.10/NumPy 1.26.4 and Python 3.12/NumPy 2.0.2,
with pandas 2.3.3. They cover legacy/current codecs, fractional/zero priors,
regression/multiclass CTR outputs, missing/unseen categories, other buckets,
reordered dictionaries, empty vocabularies, sliced arrays, unused malformed
Unicode, matching errors for used malformed Unicode, conservative fallbacks,
state reconstruction, feature names and final prediction bits.

The isolated `category-codes-cp310` variant additionally passed the existing saved
real-model parity runner on all five cases at 1, 64 and 1,000 rows: transformed
float32 arrays and final predictions match the original references bit-for-bit.
Its native binary is unchanged from `composed-cp310`, SHA-256
`d069a028a65f35656d977dfc808e5f43f113cba59c81cb19e16b26f05dedab24`.
The local real-data receipt is
`.tmp/improvements-20261016/followup-native/real/category-codes-parity.json`.
An initial receipt write failed because the variant builder had not copied the
native build receipt; all parity checks had passed. Copying that unchanged receipt
and rerunning completed successfully. No parity failure was suppressed.

The inference experiment owns subsequent isolated timing. These correctness runs
overlapped other work and provide no throughput claims.

## Paired timing decision: broad dispatch rejected

The inference owner's six-pass composed/dictionary/sparse-identity experiment
finished with exact parity on every case. The dictionary arm changes only Python
preparation and uses the same native binary as composed. Its broad `rows > 1`
dispatch is **not retained**: aggregate raw speedup was 0.9563x, with large repeated
64-row regressions despite useful 1,000-row gains.

| Saved model | Rows | Composed raw ms | Dictionary raw ms | Speedup |
|---|---:|---:|---:|---:|
| hiva | 64 | 48.549 | 100.567 | 0.483x |
| hiva | 1,000 | 212.115 | 143.565 | 1.477x |
| splice | 64 | 2.637 | 4.550 | 0.580x |
| splice | 1,000 | 13.642 | 10.862 | 1.256x |

Raw scope includes the existing benchmark normalization adapter and prediction;
prepared and native controls were approximately unchanged. Thus the improvement
is attributable to preparation, while the exact distribution of Python setup
cost requires further profiling. The candidate's fixed per-column dictionary,
unique-code and gather work is not amortized at 64 rows. A separately measured
bulk eligibility threshold or a native gather implementation would be needed
before production integration. The current checkout remains unchanged.

Paired receipts and ordering results are archived under
`inference/sparse/identity/real/`, including `dictionary-comparison.json` and the
individual raw passes. These are fixed saved-model local measurements, not new
accuracy results or a general inference ranking.

## Native gather and separately frozen dispatch studies

The next isolated native prototype retains the same dictionary-only call to the
existing encoder, but validates signed code buffers and performs the final gather
in C++. The metadata helper caches each exact object Index and its public
`values` array, and collects the code/dictionary arguments without repeated
property access. Its unchanged native binary is
`6df3e210c2086004519e7d2977e69db72d465e2e007799b9ba4288b5099d7479`.
The metadata variant passed 59 focused cases and all 15 original real bit-parity
references before any crossover timing.

The [native crossover plan](native_crossover/plan.json) froze 360 eligible cases:
widths 1/32/512, vocabularies 2/8/32, uniform or 85%-first-category draws, four
output profiles with 1/2/3/4 columns per input, and batches 128/256/512/1000/2048.
Six additional fallback cases use object columns or an unsupported 33-category
last column. Identical synthetic fitted states were restored by the actual
composed and native packages in four ABBA processes. Public transform plus a
fresh shallow DataFrame copy was timed with three blocks targeting 10 ms; no
tree prediction or real labels were used for threshold selection.

All 366 transforms matched exact float32 bits and metadata in every pass. The
predeclared universal threshold rule selected 1,000 rows: every eligible case at
1,000 and 2,048 rows exceeded 1.10x pooled speedup and improved in both orderings.
Their minimum pooled speedup was 1.312x. The alternative metadata buckets were
predeclared but were not selected because the universal rule qualified.

That row guard alone is insufficient. Four of six fallback controls repeatedly
regressed: the late unsupported column at widths 1/32/512 yielded 0.830x/0.868x/
0.879x, and the one-column object frame yielded 0.793x. These negative results
remain in [the original comparison](native_crossover/comparison.json). They are
not removed or relabeled as an accepted optimization.

A separate candidate disables the code plan at fit/state-refresh time below
32 input columns, returns before metadata inspection below 1,000 rows, and checks
public `data.dtypes` before creating per-column Series. This preserves the wide
eligible opportunity while detecting a late unsupported column early. Width 32
is the smallest nontrivial width studied, not a claim of a universally optimal
cutoff. All narrower inputs deliberately use the established path.

Read-only review also found a real custom-name side effect: a column name's
`__str__` can mutate a categorical code after the fast path captures its live
view, whereas established extraction already copied the object cells. A saved
counterexample demonstrates this difference. The guarded candidate therefore
requires an exact pandas.Index with object dtype and exact builtin-string names
before column iteration; RangeIndex, MultiIndex and custom names fall back.
Thirteen adversarial/boundary tests passed, including 31/32 columns, 999/1000
rows, custom Index hooks and rejection before Series iteration. All 35 real
transform/prediction cases at 1/64/128/256/512/1000/2048 rows also match exactly.

The [guard replication protocol](native_guard_replication/plan.json) is a distinct
frozen experiment, covering all 96 remaining eligible synthetic cases, the six
original fallback controls, and all 35 real raw prediction cases. It retains
the original fallback failure evidence and forbids tuning from its real-case
outcome. Production integration remains conditional on that result.

The completed replication **passed** its frozen criteria. Every eligible case
improved in both orders, all 96 eligible synthetic cases retained at least 1.10x
pooled speedup, and no fallback case regressed more than 3% in both orders. Every
transform and prediction remained exact in all four passes. Across the 35 real
raw cases the geometric speedup was 1.136x; this aggregate includes 31 fallback
cases and four eligible bulk cases.

| Saved model | Rows | Composed raw ms | Guarded native raw ms | Speedup |
|---|---:|---:|---:|---:|
| hiva | 64 | 47.784 | 48.037 | 0.995x |
| hiva | 1,000 | 212.052 | 51.317 | 4.132x |
| hiva | 2,048 | 394.921 | 62.299 | 6.339x |
| splice | 64 | 2.652 | 2.629 | 1.009x |
| splice | 1,000 | 13.692 | 7.843 | 1.746x |
| splice | 2,048 | 24.877 | 12.578 | 1.978x |

Passing the fallback criterion does not imply that every fallback median
improved. For example, the 32-column late-33-category synthetic control has
0.965x pooled speedup, with separate-order ratios 0.995x and 0.943x; its slowdown
was not repeated above the frozen 3% threshold. These full ordering records are
retained in [the final comparison](native_guard_replication/comparison.json).

The final isolated guarded Python source has SHA-256
`294b642b84206b556e2c846624107f783bf47ee30947e9aae3dfd85edd237de0`.
Its Python 3.12/NumPy 2.0.2/pandas 2.3.3 variant passed all 72 native/metadata and
guard tests in 1.32 seconds, plus all 35 real parity references. An initial test
collection failed because the isolated launcher did not expose the local
benchmark helper package; adding the repository explicitly after the isolated
native package resolved collection without changing implementation code.
The unchanged CP312 native hash is
`94b2b0bf3f0b4863b8c27eb308a41354a92025b6b81f2c7f539c6a4923a7e233`.
The first CP312 real receipt inherited the original helper's build metadata;
the corrected final receipt preserves the actual guarded Python hash, and the
earlier receipt remains available. NumPy deprecation warnings come from the
existing archived-pickle compatibility loader.

After extracting only runtime code into the clean production module and adding
a one-time native-method availability check for older binary compatibility, the
final CP312 package passed 198 focused interoperability tests in 10.37 seconds,
with no failures or skips. This combines the 52 new public/native/legacy-ABI
cases with the earlier 146 prediction, arithmetic, categorical-codec, row-fast
and persistence cases. Exact final module, native and test hashes are recorded
in [the final CP312 validation receipt](final_codes_cp312_validation.json).
