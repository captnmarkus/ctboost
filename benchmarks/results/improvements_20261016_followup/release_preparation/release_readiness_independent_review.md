# Independent release-readiness review — 16 October 2026

No new source-level correctness blocker was found in the reviewed release checkout. This is a read-only review, not a successful release-validation receipt. The exact source hashes are in [the review receipt](release_readiness_independent_review.json); [the static audit](release_readiness_independent_static.json) records packaging and publication checks. The candidate checkout and original scientific inputs were not modified. No tests, builds, installs, fits or scoring jobs were run.

## Compatibility

- Compact JSON changes whitespace, not the parsed document. File bytes and hashes change; callers must not expect the prior indentation. Pickle and parsed schema contracts remain separate.
- New CTR pipelines persist format 5 with double sums and priors and checked float32 output. Formats 1–4 load through the legacy float-rounding path; explicit replay keeps that accumulation mode. Existing smoothing upgrades for refitted formats 1–3 remain documented. New pipelines may change training features and models; unchanged tree-growth/hyperparameter defaults do not imply identical newly trained models. Double accumulation is not exact summation.
- Old runtimes reject format 5. New Python export guards accept it; prepared-feature R/JVM and vector/raw restrictions remain explicit. The changed native private state layout requires native consumers to rebuild.
- Categorical acceleration is conservatively dispatched and uses the existing native transform for dictionary values; derived plans are reconstructed on fit/load and old native modules without the new binding fall back. The fixed-model bit-parity and legacy replay evidence remains applicable to the reviewed implementation, but final installed artifacts still require validation. Hardware/workload-specific guard thresholds do not establish universal speedups.

## CI and source packaging

The normal 13-entry CMake matrix adds psutil, retains platform/compiler coverage and tests an installed package outside the checkout. Optional comparison-library tests can skip in that matrix. The separate Windows/Python3.12 job pins the scientific stack, asserts package location/native identities, runs 23 synthetic test files and rejects every skip. Its copied test layout includes required headers/spec/release helpers; pure protocol/math tests are not hidden behind the native-library fixture. These source checks do not establish that hosted dependency installation or execution has passed.

The static sdist allowlist matched 595 source files, including the private runtime helper, the top-level CV policy and the three precise report-contract policy fixtures. No compiled extension, model, prediction/data array or unrelated FullHPO0161 file matched. Full research evidence remains excluded from wheels. Actual backend-produced sdist/wheel membership and a clean sdist rebuild are still required; a static glob review is not an artifact audit. Historical experiment replay intentionally also needs its hash-bound local datasets, models and receipts.

**Release sequencing condition:** the tag-triggered Publish workflow validates its artifacts but does not depend on the separate CMake/scientific workflows. The publishing tag must therefore be created only after those workflows succeed for the final release commit. A successful older commit or candidate02 build alone is insufficient. The pinned science workflow runs on PR/manual dispatch; a push solely to the current `codex/` branch does not trigger its branch-filtered push event.

## Remaining execution plan

1. Finish the authorized current experiment and the prospective whole-policy timing selection; retain failures and accuracy limitations. No further method search is needed for release.
2. Complete candidate02 as a build-only/toolchain check. After final documentation/source freeze, build final artifacts and verify version/build/source/native identities. Run native CTest with the correct CP312 Python home/DLL search environment.
3. Run the installed-package full Python suite without the historical nineteen-node exclusion plugin; report optional skips by reason. Run the exact-stack scientific selection with zero skips. Preserve compatibility, new-format precision, categorical routing, raw export and warm-start coverage; do not substitute historical source-tree results for the final wheel.
4. Run platform/compiler CI, publishing wheel smoke, artifact validation/twine checks, actual sdist membership plus clean rebuild, and the documentation build. Retain R/JVM checks and the resolved Jackson dependency evidence. Finalize pending prose before publishing; GitHub alert closure is a post-default-branch check.

The final timing runtime and immutable scientific runtime have different roles. A legacy-policy latency measurement is not evidence that fresh format-5 fitting improves accuracy. Reports must bind the measured package and preserve that distinction.

## Public archive hygiene

The requested report-only pattern scan covered 7,115 textual files (87,428,889 bytes) below the original follow-up archive. It found no matches for private-key headers, supported token shapes, bearer literals or password-bearing HTTP URLs. No matched values were emitted, and NPZ/model/binary contents were not decoded. This is bounded pattern evidence, not a comprehensive secret guarantee.

The 83 approved ignored public files previously checked all matched their sizes/hashes. Publication must copy only final `manifest.files` plus `manifest.json`, preserving nested frozen receipts. The 418 NPZ archives remain local and are represented by path/size/hash in `local_only_artifacts`; a recursive copy of the whole archive would violate that contract. The established first-stage exclusions remain unchanged. The final manifest and copy/staging hashes must be checked after the remaining evidence is complete.
