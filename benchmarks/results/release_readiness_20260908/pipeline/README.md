# Feature-pipeline release checks

**Phase 2 evidence; not final release approval.** The Windows results below remain tied to wheel `9413d08...`. Subsequent ARM CI found 12 float16 warning/callback mismatches. A further correction keeps float16 on the original object path; that newer build still requires platform validation. Original logs, receipts, runners, and timing samples below are unchanged. Here, "final wheel" means the last wheel measured in this phase.

The final development wheel passed **323 pipeline tests on each of two native NumPy environments**. The checks cover exact numeric conversion, missing values, categorical/CTR inference, schemas, persistence, text/embeddings, optional frames, and NumPy signaling-NaN policies. No learner or training-default changes were involved.

| Python | NumPy | pandas | Arrow / Polars | Result | Scope |
|---|---|---|---|---|---|
| 3.12.11 | 1.26.4 | 2.1.4 | 15.0.2 / 0.20.31 | 323 passed | Final 0.1.61 native wheel |
| 3.12.11 | 2.5.3 | 3.0.5 | 25.0.1 / 1.44.1 | 323 passed | Final 0.1.61 native wheel |
| 3.8.20 | 1.24.4 | 1.5.3 | 12.0.1 / 0.20.31 | 323 passed | Current Python wrapper over public 0.1.60 native |

**The Python3.8 row does not validate a newly compiled Python3.8 wheel.** Older optional-library behavior is compared with the established object-conversion path, including its errors. Complete dependency inventories are in [dependencies/](dependencies/); original logs and runtime hashes are in [matrix/](matrix/).

## Signaling-NaN regression and correction

The first wheel passed 251 existing checks but a separate policy probe exposed an additional NumPy2 warning, callback, or log event when float32 signaling NaNs were widened. Nine of 27 NumPy2 probe cases differed; all prediction bits matched. The corresponding 27 NumPy1 cases matched completely. The original probe outputs are preserved in [policy/](policy/).

The final implementation examines float32 bits while copying ordinary values and routes signaling NaNs through the original object conversion. Uniform float32 DataFrames retain their dtype; mixed numeric DataFrames containing float32 use the object path. It does not silence callbacks or alter the caller's error policy. The successful final suites include 60 explicit warning/callback comparisons per environment, exception-policy checks, nonnative byte order, reversed/unaligned views, late signaling NaNs, quiet-NaN payloads, and subnormals. This is the green verification; the standalone first-wheel probe was not rerun. See [policy/summary.json](policy/summary.json).

## Bounded preprocessing timing check

These are medians from seven short blocks on one logical CPU (5), one thread, with concurrent background workloads. They time `FeaturePipeline.transform_array`, including frame extraction, on finite synthetic inputs. They are a regression check, not end-to-end inference, accuracy, or leaderboard evidence. The runs were sequential, without randomized pairing.

For 1,000 rows and 32 columns, milliseconds per call:

| Input | First development wheel | Final development wheel | Public 0.1.60 |
|---|---:|---:|---:|
| float32 array | 0.083 | 0.074 | 1.059 |
| float64 array | 0.058 | 0.056 | 1.143 |
| float32 DataFrame | 0.148 | 0.164 | 1.115 |
| float64 DataFrame | 0.125 | 0.136 | 1.083 |
| Mixed float32/integer DataFrame | 0.173 | 1.306 | 1.509 |
| Mixed float64/integer DataFrame | 0.148 | 0.181 | 1.488 |

The mixed float32 case loses the first development wheel's acceleration because preserving NumPy policy requires the compatible fallback. The public comparison used **NumPy2.0.2/pandas2.3.3**, whereas both development runs used **NumPy2.5.3/pandas3.0.5**; dependency and load differences limit those comparisons. Raw blocks, 16-row cases, hashes, and computed ratios are in [timing/](timing/).

## Provenance and reproduction

The first wheel came from source `9fc74087ba6cd5f4296b719ae1ac5dddcd6e68fc`; the final implementation from `aaf50d7a6d40449efc9171a6b3f634156398fad0`. Their SHA256 digests are respectively `eabd3197d9bed5eb0267f436200268b847cebc0d6e6edad2eeff43be1ee79bb5` and `9413d08ac1380f0222de639fc73967b233aa9ee67879755d8fb401af43a6dd43`.

[provenance.json](provenance.json) binds native and wrapper hashes. The archived installed wrapper matches the final wheel byte-for-byte; a later two-line source comment clarification was outside that wheel. Two final quiet-NaN/subnormal tests were added after the implementation commit and passed against the unchanged wheel. Their executed source is included in [tests/](tests/).

[runners/](runners/) contains the original verification programs, which retain their original workspace paths. Matrix runs used isolated Python (`-I`), preloaded the installed package, and then loaded repository tests with pytest's importlib mode. No benchmarks or fits were added while preparing this archive. Wheels, models, and datasets are not included.

Run `python verify_archive.py` from this directory to verify its inventory. Original UTF-16 PowerShell logs/probe files retain their exact bytes; JSON summaries and this README use UTF-8. Local `.gitattributes` prevents newline conversion. The manifest excludes itself.
