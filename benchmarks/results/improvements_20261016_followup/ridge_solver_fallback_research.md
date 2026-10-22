# Contingent ridge solver fallbacks

16 October 2026. Preparation only: no implementation, data access, fit, score,
new frozen protocol or change to the running 10,000-iteration LSQR diagnostic.

**Try unscaled LSMR before building a new preconditioner if the current LSQR
diagnostic fails.** Fong and Saunders (2011) show that LSMR decreases the
least-squares normal residual in exact arithmetic. For the augmented ridge
system `[Z; sqrt(lambda) I]`, that residual is precisely the original gradient
used by our certificate. Unlike the previous scaled LSMR run, there is no
coordinate scaling between its mathematical target and that gradient. This
motivates a bounded diagnostic; it does not guarantee finite-precision
convergence. [Primary paper](https://arxiv.org/abs/1006.0758)

The simplest prospective call is unscaled `lsmr(Z, r, damp=sqrt(lambda))`,
starting from zero, with the same strict tolerances and fixed work budget as
the extended LSQR comparison. SciPy supports the damped problem directly.
Retain the explicit original-coordinate gap and normalized-gradient checks,
finite values, accepted status and whole-worker watchdog. This proposal is
not permission to ignore iteration-limit status or accept the internal norm
alone. [SciPy 1.16 API](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.sparse.linalg.lsmr.html)

## Low-rank preconditioning is the second option

Frangella, Tropp and Udell (2023) construct an SPD Nyström preconditioner from
matrix-vector access, then use conjugate gradients on the original system.
Their guarantees require rank comparable to effective dimension; neither rank
64 nor 128 is justified here by existing spectral evidence. Use their stable
shifted QR/Cholesky/SVD construction, not a naive pseudoinverse formula.
Setup costs rank-many matrix-vector products plus small dense work; memory
is linear in dimension times rank. [Primary paper, Algorithm 2.1 and Theorem 5.1](https://tropp.caltech.edu/papers/FTU23-Randomized-Nystrom-SIMAX.pdf)

Applied here, define `H(v)=Z.T@(Z@v)` and solve
`(H + lambda I) delta = Z.T@r`. Never form the dense Gram matrix. For a
sketch eigensystem `U diag(sigma) U.T`, the paper's inverse preconditioner is

`M(v) = v + U @ (((sigma_min + lambda)/(sigma + lambda) - 1) * (U.T @ v))`.

It changes solver coordinates, not the objective: every outer matrix product
still uses the exact original `H + lambda I`. Solving only the low-rank proxy
would change the answer. Positive lambda and orthonormal U make this operator
SPD. Tiny independent dense, adjoint, symmetry/positivity and original-gap
tests would be required before any real numerical use.

## Size calculation from the existing design receipt

These are our arithmetic estimates using only the already recorded dimensions:
32,210 rows, 37,045 coefficients and 38,490,950 nonzeros. They are not timings
or spectral measurements.

| Sketch rank | One float64 coefficient basis | One data-space block | Six bases plus one data block | H products to form sketch |
|---:|---:|---:|---:|---:|
| 64 | 18.09 MiB | 15.73 MiB | 124.26 MiB | 64 |
| 128 | 36.18 MiB | 31.46 MiB | 248.52 MiB | 128 |

The six-basis figure is an illustrative named-workspace reserve, not a proven
peak: factorization scratch, allocator copies and the existing CSR must be
budgeted separately. A dense Gram alone would be 10.225 GiB and is excluded.
Sketch sparse work is approximately `4*nnz*rank` multiply/add operations:
9.85 or 19.71 billion, plus QR/SVD work of order `p*rank^2`. Each inverse
preconditioner application adds about `4*p*rank` operations. Blocking can
reduce transient data-space storage, but does not remove the work.

No spectrum or effective dimension has been measured. At lambda 1, many
directions may remain important; a small sketch can be affordable yet useless.
Any rank/seed choice must be declared from TRAIN-only numerical evidence,
with unchanged original certificates and no STOP/DEV selection. The preferred
sequence is therefore: finish current LSQR; if needed, try unscaled LSMR;
consider the more complex Nyström route only after those results warrant it.


Later status: the extended unscaled LSQR diagnostic passed its frozen
certificate at 5,541 iterations. The fallback options above remain inactive;
no LSMR or Nyström implementation or experiment was triggered. See the
[numerical result review](numerical_ridge_review.md#extended-diagnostic-outcome-certified-numerical-success).
