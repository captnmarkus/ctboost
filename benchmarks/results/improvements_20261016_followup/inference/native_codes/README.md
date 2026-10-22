# Native categorical dictionary gather prototype

The isolated candidate receives signed integer category codes and exact-string
dictionaries, discovers only used entries in first-encounter order, applies the
existing native transform to a dictionary table of at most 33 rows, and gathers
all transformed float columns in C++. This removes full object-matrix creation,
Python unique/remap loops, and per-output-column NumPy gather calls. The original
codec and CTR arithmetic perform every conversion. Training, fitted state,
trees, split selection, leaf values, and accumulation are unchanged.

The prototype is **not enabled in the shared checkout**. Its current broad
batch dispatch still regresses the 64-row hiva case; eligibility/overhead needs
more work before retention.

In a frozen five-case ABBA with 1, 64, and 1000 rows, full raw geometric speedup
is 1.1182×. Prepared and native controls remain neutral (0.9981× and 1.0038×).

| Case | Rows | Composed baseline, ms | Native codes, ms | Raw speedup |
|---|---:|---:|---:|---:|
| hiva | 1 | 28.848 | 28.831 | 1.001× |
| hiva | 64 | 48.210 | 54.643 | 0.882× |
| hiva | 1000 | 214.822 | 67.094 | 3.202× |
| splice | 1 | 1.385 | 1.391 | 0.996× |
| splice | 64 | 2.644 | 2.453 | 1.078× |
| splice | 1000 | 13.806 | 7.836 | 1.762× |

All full transformed matrices and predictions are bit-exact against saved
references. The native candidate passed 59 focused tests, including legacy
codecs/smoothing, multiclass CTRs, missing/unseen values, unused invalid Unicode,
feature-name validation, four signed integer widths, negative/unaligned strides,
empty input, 32 categories plus missing, bounds and malformed input checks.
The unchanged native prediction smoke suite passed all 230 checks. Independent
source review found no blocker. Source and runtime hashes are in the receipts.

Every pass uses the same frozen inputs/models and one fixed CPU/thread; other
fitting, builds, and tests were paused. Raw timings include the existing frame
normalization. The copied runner, protocol, source patch, prototype wrapper,
test source/logs, receipts and timing blocks are preserved. Large local fixture
files are not bundled, but their hashes are in the raw receipts. These results
diagnose inference and do not establish a model-quality or library ranking.
