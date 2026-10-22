# Prospective exact-ridge numerical diagnostic

16 October 2026. This is a solver design review, not a new accuracy policy.
Wave 17's diamonds/LightGBM selection solve at lambda 1 reached the frozen
LSQR limit of 2,000 iterations with status 7. That failed study, its solver
settings and its artifacts remain unchanged. A separately frozen diagnostic
may reuse that clean TRAIN anchor and its exact TRAIN residuals without any
tree fit, STOP score, DEV access or test access. Parent 16 remains the latest
complete policy. No solver result is a license to promote partial wave-17
selections.

## Same objective in different coordinates

Let `Z` be the existing sparse leaf-indicator matrix, `r = y - baseline`, and
`lambda > 0`. Use the half-sum objective for certificates:

\[
P(\delta)=\tfrac12\|r-Z\delta\|_2^2+
           \tfrac\lambda2\|\delta\|_2^2.
\]

This has exactly the same minimizer as the unhalved sum-residual ridge
objective used in the frozen experiment. Define
`s[j] = 1/sqrt(sum_i Z[i,j]^2 + lambda)` and `delta = s * theta`.
Apply a matrix-free augmented operator:

\[
B\theta=\begin{bmatrix}Z(s\odot\theta)\\
                  \sqrt\lambda(s\odot\theta)\end{bmatrix},\qquad
B^T\begin{bmatrix}u\\v\end{bmatrix}
 =s\odot(Z^Tu+\sqrt\lambda v),\qquad
b=\begin{bmatrix}r\\0\end{bmatrix}.
\]

Call LSQR or LSMR on `(B,b)` with **damp=0**, then recover `delta=s*theta`.
Every column of B has unit Euclidean norm. Applying an isotropic damping
parameter to `Z*diag(s)` alone would instead penalize theta, changing the
original ridge problem. Do not row-scale the data loss, alter lambda, remove
dependent columns, or materialize a dense Gram matrix.

For the frozen indicator builder every stored entry is one, so column norms
squared are column counts. Compute them from sparse column sums without
forming `Z.multiply(Z)` or copying all nonzeros. Establish or validate that
indicator invariant before using this shortcut. The operator requires only
the existing CSR and O(n+p) vectors; do not construct an augmented sparse
matrix or a second full matrix scaled by columns.

## Independent certificate in original coordinates

After recovering delta, recompute `u=r-Z*delta`, `w=Z.T*u`, and
`g=w-lambda*delta` from the original Z. The unconstrained dual function is

\[
D(u)=r^Tu-\tfrac12\|u\|_2^2-\frac{\|Z^Tu\|_2^2}{2\lambda}.
\]

For this residual choice, direct algebra gives

\[
\operatorname{gap}=P(\delta)-D(u)
 =\frac{\|Z^T(r-Z\delta)-\lambda\delta\|_2^2}{2\lambda}.
\]

Use the squared-gradient expression as the authoritative certificate; direct
subtraction `P-D` can lose precision. The latter is an independent tiny-test
identity checked with a scale-aware floating-point tolerance. For the
unhalved objective both objective and gap are twice these values.

In exact arithmetic the certificate implies
`P(delta)-P(optimum) <= gap`,
`||delta-optimum|| <= sqrt(2*gap/lambda)`, and
`||Z*(delta-optimum)||/sqrt(n) <= sqrt(2*gap/n)`.
These are mathematical bounds evaluated numerically, not interval-arithmetic
proofs. Report finite arithmetic and independent dense-oracle checks. They
apply before shrinkage and native float rounding; after fixed shrinkage q,
the corresponding prediction bound is multiplied by `abs(q)`. Existing
native-fold rounding/oracle checks remain a separate requirement.

Proposed diagnostic acceptance, frozen before a real solve:

- Native solver status is in `{0,1,2,4,5}` and all outputs/certificates are finite.
- With `P0=0.5*||r||^2`, require
  `gap/max(P, float64_epsilon*P0) <= 1e-10`.
- Require `||g||/(||Z.T*r|| + lambda*||delta||) <= 1e-8`.
- Handle an exactly zero RHS directly with zero coefficients and zero gap.
  A zero denominator in the normal-residual ratio is acceptable only when
  its numerator is zero; never mask a nonzero residual by adding a large
  arbitrary denominator constant.
- Status 7, condition-limit failures, certificate failure, time/RSS failure
  or nonfinite output remain diagnostic failures. Do not reinterpret them
  as convergence or silently rerun with looser thresholds.

The proposed bounded comparison keeps the old unscaled LSQR at its original
`atol=btol=1e-8, iter_lim=2000`. The two new variants use augmented column-
scaled LSQR and LSMR, `atol=btol=1e-12`, 2,000 iterations, `conlim=1e8`,
and `damp=0`. These differing tolerances prevent a direct speed-superiority
claim. All three are numerical diagnostics on the same residual and lambda;
no choice depends on STOP or DEV. A later larger iteration budget would be
a new predeclared diagnostic, not a change to wave 17.

## Tests and limits

Before real execution, require tiny independent dense solves of
`(Z.T*Z + lambda*I)*delta = Z.T*r`, explicit dense B/operator equivalence,
and adjoint inner-product checks. Include duplicate tree blocks, rank-
deficient indicator designs, root-only columns, imbalanced/rare leaves,
all four frozen lambdas, zero RHS, and residual scales `1e-6`, `1`, `1e6`.
Verify the dual identity and solution/prediction error bounds; demonstrate
that the incorrect scaled-matrix-plus-isotropic-damp formulation disagrees
with the dense ridge oracle. Include rejected invalid/nonfinite inputs and
forced iteration/certificate failures. Do not weaken tolerances merely to
pass a synthetic case.

LSMR is the preferred prospective solver because it controls the least-squares
normal residual, but diagonal scaling does not eliminate correlations among
many trees' leaf columns or guarantee convergence. Moreover, its normal
residual in transformed coordinates is `s*g`; monotonicity there is not
monotonicity of the original-coordinate gap. The independent certificate
therefore remains necessary. The authors' [LSMR documentation](https://www-leland.stanford.edu/group/SOL/software/lsmr/)
describes right diagonal preconditioning and normal-residual behavior; their
[LSQR documentation](https://stanford.edu/group/SOL/software/lsqr)
states the damped least-squares problem and uniqueness for positive damping.
[SciPy's LSMR contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html)
documents LinearOperator support, damping, stopping criteria and status codes.

The existing expanded diamonds pickle physically contains the authorized
TRAIN, STOP and outside-cap blocks. If the diagnostic loads that pickle,
its access statement must say so: only TRAIN plus outside-cap rows enter
the residual/design, no STOP role file or STOP prediction/metric is used,
and no DEV/confirmation/official-test file is accessed. Hash the anchor,
source, exact selected-row order, helper, diagnostic source, runtime and
solver plan before execution. Keep the whole-process watchdog; a small
extra solver-vector estimate does not replace its memory limit.


## Frozen implementation review

The separate diagnostic plan is frozen at SHA256
`36e6a9370b2fd786387beb893da49039db24035fe019dce0583302d3660e9f21`.
Its reviewed implementation and 43 synthetic cases match the
[validation receipt](ridge_numerical_diagnostic/synthetic_validation.json);
[the independent review](validation/ridge_numerical_independent_review.json)
records exact source/test hashes and scope. No real result was used for this
clearance. The unchanged wave-17 failure and its partial controls stay terminal.


## First diagnostic outcome: all three variants fail

The frozen [first diagnostic](ridge_numerical_diagnostic/numerics/receipt.json)
completed without a resource failure, but all three solvers reached status 7
at 2,000 iterations and failed the unchanged original-coordinate certificates.
[Independent receipt review](validation/ridge_numerical_results_independent_review.json)
checked the result/source hashes and coefficient-file digests without decoding
coefficient arrays or any dataset/model payload.

| Variant | Original gap / P | Normalized original gradient | Status |
|---|---:|---:|---:|
| Original unscaled LSQR | 0.000746434 | 0.000439547 | 7, failed |
| Augmented scaled LSQR | 0.0463601 | 0.00346481 | 7, failed |
| Augmented scaled LSMR | 0.00152142 | 0.000627764 | 7, failed |

The acceptance limits were 1e-10 and 1e-8 respectively, plus an accepted native
status. In this problem diagonal scaling did not improve the original-coordinate
certificate at the fixed cap. A smaller transformed normal residual is not an
original-objective guarantee, and these different-tolerance runs do not support
a speed-superiority claim. The whole diagnostic used 454.266 seconds and peaked
at 1,426,464,768 bytes of owned-process RSS; no watchdog failure occurred. There
were zero new tree fits and zero STOP/DEV/test scores. No solver is adopted and
the contemplated wave-17 numerical successor remains closed.

The next authorized preparation is a separate, prospectively frozen diagnostic
using original unscaled LSQR from zero, the same residual/indicator matrix and
lambda 1, atol=btol=1e-12, conlim=1e8 and a fixed 10,000-iteration cap. It must
retain the same certificate/status gates and 1,200-second/8-GiB watchdog. This
is more bounded numerical work on the same objective, not a relaxation of wave
17 or the first diagnostic. No warm-start-centered damping, new native fit or
held-out selection is part of that proposal. If it fails, record failure.


### Extended diagnostic freeze

The [separate extended plan](ridge_numerical_diagnostic_extended/plan.json) was
frozen at SHA256 `4b8ec2e10e6d53793d3285417a7f5dec4ce00ea6376e2cb0b8182638ef848ed6`
after independent source/test review and 14 passing synthetic tests. The
[review receipt](validation/ridge_extended_independent_review.json) records the
exact hashes. It uses only the zero-start unscaled LSQR variant described
above and binds the completed first diagnostic's evidence. This clearance
precedes the numerical result and is not solver adoption or permission to
score an accuracy successor without a successful certificate.


## Extended diagnostic outcome: certified numerical success

The [extended result](ridge_numerical_diagnostic_extended/numerics/unscaled_lsqr_10000.json)
converged from zero at 5,541 iterations with accepted status 2. Its explicit
original-coordinate relative gap is 1.8320312356806292e-14 and normalized
gradient is 2.177588593392716e-9, both below the unchanged frozen limits.
The fitting-prediction RMS error bound is 2.4591201088527176e-5 before shrinkage
and native rounding. The whole worker took 399.640 seconds and peaked at
1,426,857,984 bytes RSS, with no watchdog failure. There were no new tree fits
or STOP/DEV/test scores.

The [independent result review](validation/ridge_extended_results_independent_review.json)
checks exact plan/source/receipt hashes, coefficient archive digest, terminal
status and scalar certificate identities. It relies on the already reviewed
certificate implementation and synthetic dense oracles; it does not reload or
recompute the real residual/design. The direct subtraction P-D loses visible
relative accuracy at this tiny gap, as anticipated; the authoritative value
remains the squared original-gradient formula.

This is numerical success on one fixed anchor at lambda 1, **not evidence of
a quality gain** or universal convergence. A separately frozen accuracy
successor may use the single strict unscaled LSQR policy, while each new
regression penalty and mandatory clean-union transfer must pass its own
unchanged certificate/status/resource gates. The prior wave-17 failure and
first diagnostic remain intact. No scaled solver or fallback rank is selected.
