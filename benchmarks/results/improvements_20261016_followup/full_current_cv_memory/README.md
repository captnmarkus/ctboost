# Prospective full-current-data round CV: memory

16 October 2026. Metadata-only resource review; **no policy freeze, fit, score or data/model payload read**. Counts and controls come from the latest complete wave-16 policy, cross-checked against the inherited wave-17 job manifest. Wave 17 failed numerically and supplies no replacement selected controls.

## Decision

A single worker is the appropriate starting scope, but 4 GiB of named refit arrays is insufficient for every potential final union refit. The worst case is diamonds/LightGBM at 2,390 requested rounds. Its bound is **4.205 GiB** on all 33,460 fitting rows, versus **3.369 GiB** on a fivefold training partition. A 6 GiB named-array budget would cover the calculated cases; this is a feasibility observation, not an adopted limit. An 8 GiB whole-process ceiling cannot be guaranteed from named arrays alone. Do not copy the old limits blindly or relax them after failure.

## All 36 policies

Every triple is ordered **0.5 / 1 / 2** times the current requested count; half uses ceiling and minimum one. GiB means 2^30 bytes. Values sum conservative named-allocation budgets, not simultaneously measured RSS. `S` is scalar; `V(K)` shares one CSR across K class blocks; `S×K` uses K class-specific CSRs and K physical trees per round. `I` marks an inherited multiclass identity: its displayed bound is hypothetical and no correction design is needed under fixed identity. Scalar bounds assume a prospective refresh chooses a nonzero correction, even where the current policy is identity.

| Task | Library / layout | Full N; fold train N | Requested rounds (½/1/2) | Full named GiB (½/1/2) | Fold named GiB (½/1/2) |
|---|---|---:|---:|---:|---:|
| blood | ctboost S | 374; 299–300 | 2 / 3 / 6 | <0.001 / <0.001 / <0.001 | <0.001 / <0.001 / <0.001 |
| blood | xgboost S | 374; 299–300 | 28 / 56 / 112 | 0.001 / 0.002 / 0.003 | 0.001 / 0.001 / 0.003 |
| blood | lightgbm S | 374; 299–300 | 1 / 2 / 4 | <0.001 / <0.001 / <0.001 | <0.001 / <0.001 / <0.001 |
| churn | ctboost S | 2501; 2000–2001 | 31 / 61 / 122 | 0.005 / 0.009 / 0.017 | 0.004 / 0.007 / 0.014 |
| churn | xgboost S | 2501; 2000–2001 | 115 / 230 / 460 | 0.019 / 0.037 / 0.074 | 0.016 / 0.031 / 0.061 |
| churn | lightgbm S | 2501; 2000–2001 | 6 / 12 / 24 | 0.001 / 0.002 / 0.004 | 0.001 / 0.002 / 0.003 |
| coupons | ctboost S | 6342; 5073–5074 | 267 / 534 / 1068 | 0.120 / 0.239 / 0.476 | 0.102 / 0.203 / 0.406 |
| coupons | xgboost S | 6342; 5073–5074 | 341 / 682 / 1364 | 0.153 / 0.305 / 0.608 | 0.130 / 0.259 / 0.518 |
| coupons | lightgbm S | 6342; 5073–5074 | 212 / 424 / 848 | 0.083 / 0.165 / 0.330 | 0.069 / 0.137 / 0.273 |
| Credit | ctboost S | 97500; 78000 | 164 / 328 / 656 | 0.923 / 1.832 / 3.650 | 0.753 / 1.496 / 2.980 |
| Credit | xgboost S | 97500; 78000 | 32 / 64 / 128 | 0.177 / 0.340 / 0.665 | 0.141 / 0.272 / 0.532 |
| Credit | lightgbm S | 97500; 78000 | 44 / 87 / 174 | 0.238 / 0.457 / 0.899 | 0.190 / 0.365 / 0.719 |
| QSAR | ctboost S | 454; 363–364 | 16 / 32 / 64 | 0.002 / 0.003 / 0.007 | 0.002 / 0.003 / 0.006 |
| QSAR | xgboost S | 454; 363–364 | 30 / 59 / 118 | 0.001 / 0.003 / 0.005 | 0.001 / 0.002 / 0.005 |
| QSAR | lightgbm S | 454; 363–364 | 19 / 38 / 76 | 0.001 / 0.001 / 0.002 | 0.001 / 0.001 / 0.002 |
| Fiat500 | ctboost S | 769; 615–616 | 58 / 116 / 232 | 0.007 / 0.014 / 0.028 | 0.007 / 0.013 / 0.026 |
| Fiat500 | xgboost S | 769; 615–616 | 28 / 56 / 112 | 0.004 / 0.007 / 0.014 | 0.003 / 0.006 / 0.013 |
| Fiat500 | lightgbm S | 769; 615–616 | 21 / 42 / 84 | 0.001 / 0.002 / 0.004 | 0.001 / 0.002 / 0.004 |
| houses | ctboost S | 11260; 9008 | 84 / 167 / 334 | 0.060 / 0.115 / 0.226 | 0.049 / 0.095 / 0.186 |
| houses | xgboost S | 11260; 9008 | 544 / 1087 / 2174 | 0.334 / 0.664 / 1.323 | 0.269 / 0.535 / 1.067 |
| houses | lightgbm S | 11260; 9008 | 96 / 191 / 382 | 0.064 / 0.124 / 0.243 | 0.052 / 0.100 / 0.197 |
| diamonds | ctboost S | 33460; 26768 | 256 / 512 / 1024 | 0.478 / 0.945 / 1.879 | 0.387 / 0.764 / 1.519 |
| diamonds | xgboost S | 33460; 26768 | 77 / 154 / 308 | 0.148 / 0.283 / 0.555 | 0.118 / 0.227 / 0.445 |
| diamonds | lightgbm S | 33460; 26768 | 598 / 1195 / 2390 | 1.061 / 2.108 / 4.205 | 0.850 / 1.689 / 3.369 |
| anneal | ctboost V(5) | 450; 360 | 195 / 389 / 778 | 0.032 / 0.064 / 0.128 | 0.031 / 0.062 / 0.125 |
| anneal | xgboost S×5 I | 450; 360 | 120 / 240 / 480 | 0.083 / 0.166 / 0.331 | 0.080 / 0.160 / 0.320 |
| anneal | lightgbm S×5 I | 450; 360 | 35 / 69 / 138 | 0.005 / 0.009 / 0.019 | 0.004 / 0.008 / 0.015 |
| website_phishing | ctboost V(3) | 678; 542–543 | 59 / 118 / 236 | 0.007 / 0.014 / 0.029 | 0.007 / 0.014 / 0.027 |
| website_phishing | xgboost S×3 | 678; 542–543 | 61 / 122 / 244 | 0.008 / 0.016 / 0.031 | 0.007 / 0.013 / 0.026 |
| website_phishing | lightgbm S×3 | 678; 542–543 | 22 / 43 / 86 | 0.003 / 0.006 / 0.011 | 0.003 / 0.005 / 0.009 |
| splice | ctboost V(3) | 1596; 1276–1277 | 85 / 169 / 338 | 0.009 / 0.018 / 0.036 | 0.008 / 0.015 / 0.030 |
| splice | xgboost S×3 | 1596; 1276–1277 | 35 / 69 / 138 | 0.010 / 0.019 / 0.038 | 0.008 / 0.016 / 0.031 |
| splice | lightgbm S×3 | 1596; 1276–1277 | 79 / 157 / 314 | 0.021 / 0.041 / 0.082 | 0.017 / 0.033 / 0.066 |
| SDSS17 | ctboost V(3) | 49535; 39628 | 87 / 173 / 346 | 0.241 / 0.465 / 0.916 | 0.193 / 0.373 / 0.734 |
| SDSS17 | xgboost V(3) I | 49535; 39628 | 220 / 440 / 880 | 0.657 / 1.301 / 2.587 | 0.541 / 1.070 / 2.129 |
| SDSS17 | lightgbm S×3 I | 49535; 39628 | 80 / 160 / 320 | 0.639 / 1.262 / 2.509 | 0.511 / 1.011 / 2.010 |

All current retained counts equal their requested counts in the saved refit receipts. Future folds may structurally stop earlier, especially LightGBM, so the table budgets the requested count. Old nested correction receipts can contain historical tree counts and were deliberately not used as current model-size evidence.

## Accounting and uncertainty

Let n be fitting rows, T physical trees, q=nT, L the sum of observed leaf columns, and P the coefficient count. For prospective bounds, each tree has at most min(n, 2^depth) leaves for CTBoost/XGBoost, or min(n, num_leaves) for LightGBM; any explicit additional depth/leaf cap is intersected. No minimum-leaf heuristic is treated as a hard bound. The manifest has no parallel-tree multiplicity above one. Actual topology and observed leaves remain unknown until fitting.

The common leading budgets are 16q bytes for native leaf IDs/copy, 16q for resident CSR data/indices, and 24q for design construction: **56nT bytes combined**. Maps add 8L. CSR row pointers add 8(n+1) per distinct matrix; construction adds 32n. Native margins/copy add 16n for scalar or 16nK for multiclass. For scalar models P=L; multiclass scalar class blocks partition the trees and also have P=L; vector leaves share one CSR but have P=KL.

Regression includes the existing 64(n+P) LSQR vector budget **plus 256(n+P)** for the prospective augmented operator/certificate. Binary L-BFGS-B includes 96n objective workspace, 8(45P+4,560)+12P optimizer work arrays (history 20), and 100P vector reserve. Multiclass uses the same optimizer terms and 8n(8K+4) objective workspace. The generator executes only the pure arithmetic function ASTs from the frozen helpers and adds the explicitly named new regression reserve; it does not import native modules or instantiate payloads. [The JSON table](estimates.json) contains all 216 stage/profile rows, term-by-term bytes and source hashes.

For diamonds/LightGBM at double rounds/full union, q=79,969,400: the three leading terms alone consume 4.171 GiB. Right column preconditioning is matrix-free and must not create another full CSR or a dense Gram matrix. Bounds are deliberately loose when trees are shallow/root-only or leaves are unobserved. They do not cover native fitting histograms, model objects, input frames, raw-fold vocabulary/CTR workspaces, test predictions, Python containers, BLAS caches, serialization copies or allocator retention. Named totals therefore are neither an RSS bound nor a native-training memory estimate.

## Host and empirical context

Wave 17's largest completed worker used **2.128 GiB RSS**; its largest named estimate was **2.014 GiB**, and its longest job took 424.4 seconds. These are separate maxima, not a paired overhead factor, and only TRAIN anchors ran. The numerical failure occurred despite staying within resource limits; additional memory cannot establish convergence.

A contemporaneous OS metadata snapshot reported 31.925 GiB visible and 14.100 GiB free physical memory. Other diagnostic/build activity can change availability; this is not a quiet baseline or a reserved capacity guarantee. Before any new freeze, obtain a no-build/no-timing baseline and predeclare a one-worker RSS limit plus independent system-headroom preflight. A prospective 6 GiB named / 8–12 GiB owned-process envelope is arithmetically plausible on this host, but the higher RSS end requires sufficient available RAM and a fixed additional headroom reserve before dispatch. A bounded wait or terminal insufficient-memory outcome must be decided before running. Keep the owned-process-tree watchdog; do not claim that this table validates an 8 GiB ceiling.

Fivefold × three independent counts × 36 policies means **540 fold fits plus 36 mandatory union fits**, before any conditional seed replication. Do not multiply a one-job memory estimate by the fit count: workers must execute serially and release models/designs between jobs. No early stopping or slicing is assumed. Every fold needs raw fold-local normalization and TRAIN-fold-only correction solving at its actual count; fixed multiclass temperature is applied once. Preserve the four expanded-source exclusions and full class vocabulary checks. Numerical solver adoption and resource limits require separate prospective decisions; this report makes neither.
