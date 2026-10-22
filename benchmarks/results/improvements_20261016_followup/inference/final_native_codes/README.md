# Accepted categorical batching integration

The production fast path preserves the fitted categorical codecs, CTR arithmetic, output order, tree structure, and prediction bits. It transforms each **used category once**, then gathers the existing float32 outputs by pandas category codes inside C++. It does not change fitting or split selection.

Dispatch requires an existing compatible native method, at least **32 fitted input columns** and **1,000 request rows**, complete independent categorical coverage, ordinary string column names, and at most 32 builtin-string categories per column. A public dtype preflight rejects unsupported frames before constructing their Series. Existing paths handle smaller batches, unsupported frames, and older native extensions.

The width bound is deliberately conservative; it is not claimed to be optimal. The initial broad dispatch and sparse-bin/compact-node experiments remain archived with their regressions.

## Independent acceptance evidence

[The frozen guarded replication](../../profile/native_guard_replication/comparison.json) tested 102 synthetic cases and 35 real case/batch pairs in balanced ABBA order. Every eligible real case improved in both paired orders, with no repeated fallback regression above 3%. Across all 35 real pairs, the geometric mean raw prediction speedup was 1.136×.

| Existing real model | Rows | Previous composed runtime | Guarded batching | Speedup |
|---|---:|---:|---:|---:|
| hiva_agnostic | 1,000 | 212.05ms | 51.32ms | 4.13× |
| hiva_agnostic | 2,048 | 394.92ms | 62.30ms | 6.34× |
| splice | 1,000 | 13.69ms | 7.84ms | 1.75× |
| splice | 2,048 | 24.88ms | 12.58ms | 1.98× |

![All 35 real case/batch pairs](../../figures/categorical_batching.png)

These timings include the existing shared DataFrame normalization and full prediction call. They reuse frozen development fixtures, cycle rows where necessary, and do not imply an accuracy or equal-quality leaderboard result. The clean production helper's runtime functions/constants are AST-identical to the accepted prototype; builder/CLI imports were removed. Final package validation is recorded separately in [validation.json](validation.json).

## Reproduction and provenance

[production.patch](production.patch) is incremental from the previously validated composed runtime. Both native binaries were built with MSVC 19.50, `/EHsc /O2 /Ob2 /DNDEBUG`; all 109 native source/header/root-CMake files match the isolated build sources byte-for-byte. The exact native and Python hashes are retained in the CP310/CP312 receipts. The isolated package directories remain under `.tmp/improvements-20261016/followup-native/final-codes-cp310` and `final-codes-cp312`.

Validation passed: **2,192 tests** in the full CP310 suite (46 optional-dependency/environment skips), **198 focused CP312 tests**, **230 native checks per ABI**, all **35 real transform/prediction parity cases**, and all **36 wave9 selected-model parity checks**. The final binary-leaf research helper also passed its 14 tests under CP312 with all three libraries available.

After that production validation, the explicit source-distribution allowlist was corrected to include `ctboost/_categorical_codes.py`. This packaging-only correction did not change runtime or native code. An installed-backend, sdist-only build with scikit-build-core 0.12.2 succeeded on CPU 0; no native or wheel build ran. Its 564-member archive contains the helper byte-for-byte, SHA256 `54bd751794046107f1ffd0d29152146c4b90351e24fe4a489fd2b453e4d6470d`, and excludes scratch directories and compiled extensions. The 1,777,224-byte artifact remains at `.tmp/improvements-20261016-followup/packaging/sdist/ctboost-0.1.61.tar.gz`, SHA256 `fc798d20e3872963d0e680ae34365796899546a5640a21c0cc110c5b923d885f`. The [membership/hash receipt](../packaging/sdist_membership.json), [complete member list](../packaging/sdist_members.txt), and [verification script](../packaging/verify_sdist.py) preserve the check.

Large input/model fixtures remain in the local cache and are identified by hashes; this archive alone is not a portable copy of those datasets/models. Latest matched-library timing must use the final selected models and the dependency-aware frozen runner, including temperature calibration in both timed paths.
