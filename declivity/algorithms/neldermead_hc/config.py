from dataclasses import dataclass

from declivity.algorithms.neldermead.config import NelderMeadConfig

__all__ = [
    "NelderMeadHCConfig",
]


@dataclass
class NelderMeadHCConfig(NelderMeadConfig):
    """Configuration for Hessian-completed Nelder-Mead.

    Inherits every classic Nelder-Mead field (``xatol`` / ``fatol`` /
    ``adaptive`` and the reflect/expand/contract/shrink coefficients) and adds
    the knobs of the model step.  With ``model_step=False`` the optimizer *is*
    classic Nelder-Mead, which is what makes the two an exact ablation.
    """

    model_step: bool = True
    """Attempt the donated-Hessian model step.  ``False`` reduces the optimizer
    to classic Nelder-Mead, so the two arms differ in exactly one block."""

    fit_scale: bool = True
    """Treat the donated curvature as *shape only* and re-fit its magnitude at
    run time from points Nelder-Mead has already evaluated and discarded.

    A learned geometry usually pins down anisotropy far better than overall
    size -- a CMA-ES covariance fixes the shape of ``H^-1`` but its magnitude is
    absorbed into ``sigma`` -- and the model step's Newton length depends on
    that magnitude.  Off, the donated magnitude is taken at face value."""

    shape_initial_simplex: bool = False
    """Also shape the *initial simplex* from the geometry's principal axes (what
    :class:`~declivity.algorithms.neldermead.NelderMeadOptimizer` does with an
    ``initial_geometry=``).

    Off by default, and deliberately so: because every classic Nelder-Mead move
    is a fixed affine combination of the vertices, an anisotropic starting
    simplex is only a change of variables, and on an ill-conditioned quadratic
    it measurably accelerates simplex degeneration.  Leaving this off means the
    curvature reaches the run *only* through the model step."""

    max_stride: int = 128
    """Ceiling on the geometric back-off between model-step attempts.  The
    stride doubles after every attempt that was not predictive and resets to 1
    after one that was, so on a landscape the donated curvature cannot describe
    the overhead decays toward ``1 / max_stride`` of the iterations."""

    condition_limit: float = 1e12
    """Skip the model step when ``cond(D)`` of the simplex edge matrix exceeds
    this: past it the fitted gradient is numerical noise."""

    trust_low: float = 1.0 / 32.0
    trust_high: float = 32.0
    """Band for the dimensionless trust-region factor.  The radius is
    ``factor * (the simplex's own extent in the H-metric)``, so it rides the
    simplex instead of floating free -- a free radius collapses to zero and
    never recovers."""

    ratio_threshold: float = 0.25
    """Trust-region ratio (actual / predicted decrease) above which a model step
    counts as predictive, both for growing the radius and for resetting the
    attempt schedule."""

    pivot_floor: float = 4.0
    """How far a model step must reach, relative to the vertex it displaces,
    to be admitted into the simplex.

    Writing the step in the edge basis as ``sum_j lambda_j d_j``, dropping
    vertex ``j`` multiplies the simplex volume by exactly ``|lambda_j|``, so this
    is a floor on that factor: at ``1.0`` an accepted step cannot shrink the
    simplex, and above it the step must actively grow it.  The rule matters far
    more than it looks.  Without a floor the pivot picks the least damaging swap
    but never declines one, and a run of short, marginally-improving insertions
    grinds the simplex away: measured with ``B_0 = I`` on the
    10^6-conditioned Ellipsoid at d=10, the simplex volume was already 500x
    below plain Nelder-Mead's by evaluation 200 and the run finished 13 orders
    of magnitude worse -- while consuming only 3 % of the budget.  The damage
    was geometric, not budgetary, and Nelder-Mead has no move that restores it.

    Sweeping the floor over the same grid gives the default.  Medians at d=10,
    5 seeds, 3000 evaluations, against plain Nelder-Mead's own result:

    ============ ========= ========= ========= =========
    pivot_floor  I         C_20      C_320     H^-1
    ============ ========= ========= ========= =========
    0.1          1.2e+03   9.2e-01   4.5e-20   7.3e-23
    1.0          3.1e-04   1.7e-01   3.6e-17   8.8e-18
    4.0          3.8e-09   2.5e-08   3.1e-15   5.1e-17
    8.0          3.8e-09   2.5e-08   9.6e-14   3.0e-16
    plain NM     3.2e-10   9.6e-10   1.1e-09   2.4e-06
    ============ ========= ========= ========= =========

    The reading is that *short model steps are the harmful ones*: they are the
    moves Nelder-Mead's own reflection already makes better, and they cost
    geometry to take.  A floor of 4 keeps the long jumps -- which is where a
    donated Hessian actually pays, six to eleven orders on a well-learned
    conditioner -- while leaving plain Nelder-Mead intact as the floor when the
    conditioner is poor."""

    gain_floor: float = 0.5
    """How productive a model step must be, per evaluation, relative to the
    classic Nelder-Mead moves around it, to keep its place in the schedule.

    The trust-region ratio alone is not enough to protect the run: it asks
    whether the model predicted its *own* decrease, and a model fitted through
    ``fit_scale`` on a badly conditioned donated matrix is perfectly
    self-consistent while pointing somewhere useless.  On the 10^6-conditioned
    Ellipsoid with ``B_0 = I`` that combination cost 13 orders of magnitude,
    because self-consistent-but-useless steps crowded the classic moves out of
    the budget.  Comparing decrease-per-evaluation against a decaying average of
    what the classic moves are achieving asks the question that actually matters
    -- is this evaluation better spent here? -- and restores plain Nelder-Mead
    as the floor."""

    gain_decay: float = 0.8
    """Decay of the exponential moving averages behind ``gain_floor``.  Both
    averages track the same recent window, which is what makes their ratio
    meaningful as the run's absolute decreases shrink."""

    history_factor: int = 4
    """Size of the discarded-point buffer used by ``fit_scale``, in multiples of
    the dimension."""

    def validate(self) -> None:
        super().validate()
        if self.max_stride < 1:
            raise ValueError("max_stride must be at least 1.")
        if self.condition_limit <= 1.0:
            raise ValueError("condition_limit must exceed 1.")
        if not 0.0 < self.trust_low <= self.trust_high:
            raise ValueError("Require 0 < trust_low <= trust_high.")
        if self.history_factor < 1:
            raise ValueError("history_factor must be at least 1.")
        # Values above 1 are the useful range, not an error: 1 means "must not
        # shrink the simplex" and the default of 4 means "must grow it".
        if self.pivot_floor <= 0.0:
            raise ValueError("pivot_floor must be positive.")
        if self.gain_floor < 0.0:
            raise ValueError("gain_floor must be non-negative.")
        if not 0.0 <= self.gain_decay < 1.0:
            raise ValueError("gain_decay must lie in [0, 1).")

    def __str__(self) -> str:
        return (
            f"NelderMeadHCConfig(dimensions={self.dimensions}, "
            f"model_step={self.model_step}, fit_scale={self.fit_scale}, "
            f"shape_initial_simplex={self.shape_initial_simplex}, "
            f"xatol={self.xatol:.1e}, fatol={self.fatol:.1e})"
        )
