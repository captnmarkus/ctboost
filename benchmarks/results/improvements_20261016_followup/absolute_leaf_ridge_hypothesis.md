# Inactive hypothesis: regularize final leaf contributions

16 October 2026. Research and source inspection only; no implementation, model
loading, fitting, score access or policy activation. Full-current-data count CV
remains the nearer prospective experiment.

## Primary-source distinction

Cui, Sudjianto, Zhang and Li (11 May 2023, version 3) express a fitted tree
ensemble through leaf indicator features. Their equation 10 refits final leaf
coefficients with squared-weight regularization and an unpenalized fitted
intercept. Their algorithm also removes duplicate leaf features. This motivates
testing final-weight regularization, but does not establish that it improves
CTBoost or the present portfolio. A fixed-bias, unchanged-physical-leaf version
would differ from their procedure and must not be called a replication.
[Paper, equation 10 and algorithm 1](https://arxiv.org/html/2304.13761v3).

RGF also performs fully corrective updates of all existing leaf weights during
training. Its separate structure-growth algorithm is outside this proposal.
[Official RGF guide, section 2.2](https://github.com/RGF-team/rgf/blob/master/RGF/rgf-guide.rst).

## Independent mathematical comparison

Let `Z` be fixed leaf indicators, `b` a fixed scalar bias, and `theta0` the
effective native leaf contributions, so `F = b + Z theta0` up to the documented
native accumulation roundoff. The current offset objective is

`0.5 ||y - F - Z delta||² + 0.5 lambda ||delta||²`.

Writing `w = theta0 + delta` shows that its penalty is
`0.5 lambda ||w - theta0||²`. Large lambda returns the original model. It can
reduce an oversized original leaf only when the residual improvement pays for
moving away from that leaf; it does not directly penalize its existing size.

The distinct fixed-bias objective would be

`0.5 ||y - b - Z w||² + 0.5 lambda ||w||²`.

Its normal equations have the same positive-definite matrix
`Z.T Z + lambda I`, with right-hand side `Z.T (y-b)`. The reviewed matrix-free
solver and original-coordinate certificate apply after this explicit change of
unknown and residual. No dense Gram matrix is needed. The final candidate can
use `w_q = theta0 + q*(w-theta0)`, with exact native identity at `q=0`.
For large lambda and `q=1`, predictions tend toward `b`, rather than `F`.

This changes the hypothesis, not just the solver. Any future comparison must
freeze its own grid for every library, fit coefficients only on TRAIN, select
on STOP, and mandatorily recompute the chosen controls on the authorized union.
No per-task DEV fallback is justified. The existing offset experiment and its
failed or passed numerical evidence remain intact.

## Native extraction and fairness prerequisites

- Extract **effective prediction contributions**, not raw training statistics.
  Current CT fold code uses each tree's float learning-rate/schedule multiplier;
  multiclass scalar trees map rates by round while vector trees map by tree.
  XGBoost scalar predictions use leaf `split_conditions`; vector predictions use
  the `leaf_weights` slot mapped by each leaf's `right_children`, not training
  `base_weights`. LightGBM exposes effective outputs through `get_leaf_output`.
  The existing helpers demonstrate folding, not a complete absolute-weight
  extractor. DART weights and unsupported mixtures should initially be rejected.
- Freeze a comparable bias convention. LightGBM may embed its initial constant
  in the first tree, unlike an explicit separate base score. Simply penalizing
  every serialized leaf would then penalize that constant differently across
  libraries. Require a validated `b + Z theta0` native oracle and a deliberate
  common bias convention before choosing a penalty. Intercept refitting is an
  additional hypothesis, not a silent extension.
- Ridge depends on the representation: duplicate indicator columns, duplicated
  trees and constant partitions change the effective penalty. Preserve and
  disclose the physical-leaf basis initially; do not silently deduplicate it.
  Every native leaf needs an explicit rule. A leaf absent from fitting rows has
  optimum zero under absolute ridge, whereas the current offset helper leaves
  its original value unchanged.
- Keep tree topology, categorical routes, ordered training transform, schema,
  class support and temperature fixed. Require native reconstruction, unchanged
  routes, finite coefficients, solver certificates, roundoff-aware folded
  prediction checks and exact persistence parity. Binary/multiclass analogues
  require separate loss/intercept contracts; a regression result would not
  establish them.

Source inspected: `benchmarks/fixed_leaf_refit.py` and
`benchmarks/fixed_multiclass_leaf_refit.py`. These requirements are prospective;
neither source was modified.

## Prospective refinement: eliminate a free intercept analytically

The parent's later proposal avoids choosing a library-specific fixed bias.
Independently deriving the joint objective
`0.5 ||y - b*1 - Z w||² + 0.5 lambda ||w||²` gives
`b = mean(y) - p.T w`, where `p = Z.T 1 / n`.
Let `P = I - 11.T/n`. Solve the same positive-ridge problem with
`A = P Z` and `c = P y`, using only matrix-free products:

`A w = Z w - mean(Z w)`

`A.T v = Z.T v - p * sum(v)`.

Neither a dense centered design nor a Gram matrix is necessary. Positive
lambda makes the coefficient minimizer unique despite constant-partition
dependencies. This removes dependence on where a native model stores its
initial constant. It is not invariance to changing the indicator basis:
duplicate columns and redundant trees still alter the ridge penalty.

For a joint-objective certificate, compute the original residual
`e = y - b*1 - Z w` and a centered dual vector `u = P e`. In exact arithmetic,

`primal - dual = ||Z.T u - lambda*w||²/(2*lambda) + n*mean(e)²/2`.

The second term checks intercept stationarity; it is zero for the analytic
optimum. In floating point, report the residual mean and the two nonnegative
terms rather than assume centering is exact. For a finite-precision centered
operator, retain explicit adjoint tests, dense-oracle checks and scale-aware
certificate gates in the original variables. No stopping thresholds or solver
budget are selected by this derivation.

A future shrinkage candidate must interpolate both components:
`b_q = (1-q)*b0 + q*b` and `theta_q = (1-q)*theta0 + q*w`.
Native identity at `q=0` should bypass serialization edits. At nonzero q, folding
must represent the intercept change in the native base or as a constant shift
of every leaf in a designated existing tree. Library layouts, CT learning-rate
rounding, all unobserved leaves and the permitted bias-field change require a
new reconstruction proof; the current offset-only fold checker is insufficient.

Free intercept fitting is consistent with the intercept in Cui et al.'s
equation 10, but retaining duplicate physical leaves still differs from their
algorithm. The derivation above is our independent analysis. No implementation,
fit, score access or activation followed from this refinement.


## Numerical feasibility after the 17b audit (inactive)

The centered solve keeps the sparse indicator matrix and adds a column-mean
vector plus a few row/parameter work vectors. Each matrix-vector pair still
costs O(nnz(Z) + n + p); no dense n-by-p centered matrix or p-by-p Gram matrix
is needed. For the largest current scalar shape, diamonds/LightGBM at 1,195
rounds has n=33,460, at most p=37,045 and 39,984,700 indicator nonzeros. The
current named budget is about 2.264 GB including its certificate reserve;
centering vectors are small relative to that design. These are metadata
bounds, not a new allocation measurement. Double-count CV profiles can be
roughly twice as large, so any successor must budget its actual selected
counts before launch.

The centered Hessian is Z.T Z - n*p*p.T + lambda*I (here p denotes the
column-mean vector). Positive lambda supplies strict convexity and a unique
coefficient solution, with an analytic intercept. Removing the constant mode
does not guarantee fewer Krylov iterations: the failed 2,000-iteration ridge
variants and the 5,541-iteration successful unscaled solve illustrate why a
fixed original-objective certificate and terminal iteration/resource limits
are still needed. Reuse the certificate concept, not an untested promise that
the existing 10,000 cap will suffice. Independent adjoint/dense oracle tests,
large-offset/cancellation cases and explicit intercept stationarity are
prerequisites.

A fair bounded successor would cover all four regression tasks and all three
libraries with the same penalty/shrinkage grid plus an exact identity, select
only on original STOP, then mandate one fixed final policy per task/library.
Fresh clean TRAIN and union baselines would cost 12 + 12 model fits, at most
48 selection solves and 12 selected final solves. That count is a prospective
clean-baseline design, not an activated protocol. Alternatively, an absolute
replacement can start from existing corrected models if every effective leaf
and the intercept are replaced and the current-policy shrinkage reference is
explicitly frozen; the two identity conventions must not be mixed. A separate
current-control TRAIN anchor is still necessary for independent STOP choice.

Representation dependence and extrapolation are material limits: duplicated
physical leaves change the penalty, and an unseen fitting leaf goes to zero
at full absolute refit. Intercept fitting addresses storage of the constant,
not those other effects. Native-format feasibility, all-leaf replacement and
exact identity are separate from convergence; no quality benefit follows from
this derivation or from the completed offset-ridge experiments.


Native extraction/folding feasibility is separately reviewed in
[absolute_leaf_native_format_audit.md](absolute_leaf_native_format_audit.md).
A standalone numerical prototype and synthetic tests were prepared, then
[archived unrun](inactive_absolute_leaf_prototypes/README.md), together with
the native-format prototype. The archive preserves original paths and hashes
in its [manifest](inactive_absolute_leaf_prototypes/manifest.json). No experiment
was activated, and no numerical result or native folding validation follows
from this source checkpoint. On 16 October the user asked to finish the current
18b experiment and release validated improvements after CI, even if an accuracy
gap remains; this hypothesis is therefore retained only as inactive research.
