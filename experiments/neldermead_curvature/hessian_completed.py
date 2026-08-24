"""Hessian-completed Nelder-Mead (NM-HC): a proof of concept.

Motivation
----------
Every classic Nelder-Mead move -- reflect, expand, contract, shrink -- is a
*fixed affine combination of the current vertices*::

    x_r = (1 + rho) * xbar - rho * v_worst        (coefficients independent of f, H)

so NM is exactly affine-invariant: NM on ``f`` from simplex ``S`` and NM on
``f(A y)`` from ``A^-1 S`` trace out exact affine images of one another.  The
corollary is the whole reason seeding the *initial simplex* from a learned
geometry underdelivers:

    Classic Nelder-Mead is structurally incapable of consuming curvature.
    The only place a Hessian can enter is the initial simplex, and that is
    merely a change of variables -- consumed once, then forgotten.

The mechanism: Hessian completion
---------------------------------
A quadratic model over ``n`` dimensions has ``1 + n + n(n+1)/2`` free
parameters, hopelessly underdetermined by the simplex's ``n + 1`` known
values.  But if the Hessian is *given*, only the constant and the gradient
remain unknown: ``1 + n`` unknowns against ``n + 1`` equations.  **The system
is square.**  With ``v_0`` the best vertex and ``d_i = v_i - v_0``:

    m(x) = f_0 + g' (x - v_0) + 1/2 (x - v_0)' H (x - v_0)
    m(v_i) = f_i   <=>   g' d_i = f_i - f_0 - 1/2 d_i' H d_i =: b_i
    D g = b,    D = [d_1; ...; d_n]  (n x n)

``g`` is a *curvature-corrected simplex gradient*: the plain simplex gradient
NM implicitly follows, minus the second-order term the donated Hessian
supplies.  Solving costs one ``n x n`` solve and **zero function
evaluations** -- the model is fitted entirely from points NM already paid for.

The model's minimiser gives a Newton step from the best vertex::

    x+ = v_0 - H^-1 g

evaluated once per attempt.  On an exact quadratic with the exact Hessian this
lands on the optimum in a single step, from any non-degenerate simplex.

Three safeguards, each fixing a concrete failure mode
----------------------------------------------------
1. **Geometry-preserving pivot.**  The naive move -- replace the *worst*
   vertex -- destroys the simplex: a short model step drops the far vertex and
   pulls everything toward ``v_0``, so the simplex collapses onto a hyperplane
   and NM's own moves starve.  (Measured: log10 simplex volume falls to -38 vs
   -28 for plain NM, and the run ends two orders of magnitude worse.)  Instead
   write the step in the edge basis, ``x+ - v_0 = sum_j lambda_j d_j``; dropping
   vertex ``j`` multiplies the simplex volume by exactly ``|lambda_j|``, so pick
   the largest ``|lambda_j|`` among vertices worse than ``x+``.  This is the
   Lagrange-pivot rule from model-based DFO, and it costs one ``n x n`` solve.

2. **A trust region that rides the simplex.**  The radius is held at
   ``tr_factor * (H-norm extent of the current simplex)`` with ``tr_factor``
   adapted by the classic actual/predicted ratio and clipped to a bounded
   band.  A free-floating radius collapses to zero and never recovers; anchoring
   it to the simplex keeps the model step commensurate with NM's own scale, at
   every stage of the run, with no problem-dependent constant.

3. **Success-driven schedule.**  Attempts start every iteration and back off
   geometrically after failures, so on a function where the donated curvature
   is useless the overhead decays toward zero and the run degrades to plain NM.

Unknown scale (``fit_scale``)
-----------------------------
A donated curvature is often known only up to a scalar -- a CMA-ES covariance
``C`` fixes the *shape* of ``H^-1`` but its magnitude is absorbed into
``sigma``.  Writing ``H = alpha * H_1`` with ``H_1`` normalised to unit
geometric-mean eigenvalue, the interpolation conditions stay **linear in
(g, alpha)**::

    g' d + alpha * (1/2 d' H_1 d) = f(x) - f_0

so one extra evaluated point makes the system square again in ``n + 1``
unknowns.  Nelder-Mead discards a point every iteration; ``fit_scale=True``
recycles those discards as the extra rows and solves for ``(g, alpha)`` by
least squares.  The donated matrix then supplies only the shape, and the run
calibrates its own magnitude.

Ablation discipline
-------------------
This module re-implements the Nelder-Mead loop standalone rather than
subclassing :class:`~declivity.algorithms.neldermead.NelderMeadOptimizer`
(which is ``@final`` and bit-identical to SciPy) so that ``model_step=False``
and ``model_step=True`` run *the same code* except for one clearly marked
block.  With ``model_step=False`` the loop is a line-for-line copy of the
framework optimiser; ``validate.py`` asserts the two agree exactly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Curvature",
    "HessianCompletedNelderMead",
    "NMResult",
]

_FLOOR = 1e-30


@dataclass(frozen=True)
class Curvature:
    """A donated curvature, split into a unit-scale shape and a magnitude.

    ``H = scale * shape``, where ``shape`` is normalised to unit geometric-mean
    eigenvalue.  The split exists because a learned geometry usually pins down
    the shape far better than the magnitude, and only the magnitude has to be
    re-fitted at run time (see ``fit_scale``).
    """

    shape: NDArray[np.float64]
    shape_inverse: NDArray[np.float64]
    scale: float

    def norm(self, v: NDArray[np.float64]) -> float:
        """``sqrt(v' H_1 v)`` -- the scale-free shape metric the TR measures in."""
        return float(np.sqrt(max(float(v @ self.shape @ v), 0.0)))

    @classmethod
    def from_hessian(cls, hessian: NDArray[np.float64]) -> "Curvature":
        """Split an SPD matrix into (unit-geometric-mean shape, magnitude)."""
        matrix = np.asarray(hessian, dtype=float)
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues, basis = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, _FLOOR)
        scale = float(np.exp(np.mean(np.log(eigenvalues))))
        unit = eigenvalues / scale
        return cls(
            shape=(basis * unit) @ basis.T,
            shape_inverse=(basis * (1.0 / unit)) @ basis.T,
            scale=scale,
        )

    @classmethod
    def from_covariance(
        cls,
        eigenvectors: NDArray[np.float64],
        eigenvalues_sqrt: NDArray[np.float64],
        sigma: float = 1.0,
    ) -> "Curvature":
        """``H = (sigma^2 C)^-1`` from a CMA-ES eigendecomposition of ``C``."""
        variances = np.maximum(
            np.asarray(eigenvalues_sqrt, float) ** 2 * float(sigma) ** 2, _FLOOR
        )
        basis = np.asarray(eigenvectors, float)
        return cls.from_hessian((basis * (1.0 / variances)) @ basis.T)


@dataclass
class NMResult:
    """One run: convergence trace plus the diagnostics the PoC plots."""

    best_solution: NDArray[np.float64]
    best_fitness: float
    evaluations: int
    message: str

    trace_evaluations: list[int] = field(default_factory=list)
    trace_best: list[float] = field(default_factory=list)

    log_volume: list[float] = field(default_factory=list)
    """log10 raw simplex volume.  Confounded by progress: a simplex that has
    *reached* the optimum is legitimately tiny.  Use ``log_shape_quality`` to
    ask about degeneracy."""
    log_shape_quality: list[float] = field(default_factory=list)
    """log10 of the Hadamard ratio ``|det D| / prod_i ||d_i||``, in ``(0, 1]``:
    ``0`` (i.e. ratio 1) means mutually orthogonal edges, large negative means
    the simplex has flattened onto a hyperplane.  Scale-invariant, so it
    separates *degeneracy* from *convergence*."""
    log_edge_condition: list[float] = field(default_factory=list)
    """log10 cond(D) of the edge matrix -- how ill-posed the model fit is."""

    attempt_iteration: list[int] = field(default_factory=list)
    attempt_evaluation: list[int] = field(default_factory=list)
    attempt_accepted: list[bool] = field(default_factory=list)
    attempt_improved_best: list[bool] = field(default_factory=list)
    attempt_rho: list[float] = field(default_factory=list)
    attempt_pivot: list[float] = field(default_factory=list)
    """``|lambda_j|`` of the dropped vertex: the factor the simplex volume was
    multiplied by.  Values near 1 mean the swap preserved the geometry."""
    attempt_scale: list[float] = field(default_factory=list)
    skipped: int = 0
    """Attempts abandoned before any evaluation (ill-conditioned / no descent)."""

    simplex_history: list[NDArray[np.float64]] = field(default_factory=list)
    model_points: list[NDArray[np.float64]] = field(default_factory=list)

    @property
    def attempts(self) -> int:
        return len(self.attempt_accepted)

    @property
    def accepts(self) -> int:
        return int(np.sum(self.attempt_accepted)) if self.attempt_accepted else 0

    @property
    def improvements(self) -> int:
        return (
            int(np.sum(self.attempt_improved_best)) if self.attempt_improved_best else 0
        )


class HessianCompletedNelderMead:
    """Nelder-Mead with an optional donated-Hessian model step.

    ``model_step=False`` reproduces the framework / SciPy Nelder-Mead exactly.
    ``model_step=True`` adds the Hessian-completion block; ``curvature`` must
    then be supplied.
    """

    def __init__(
        self,
        func,
        x0: NDArray[np.float64],
        *,
        lower_bounds: NDArray[np.float64] | float = -np.inf,
        upper_bounds: NDArray[np.float64] | float = np.inf,
        max_evaluations: int = 10_000,
        initial_simplex: NDArray[np.float64] | None = None,
        curvature: Curvature | None = None,
        model_step: bool = False,
        fit_scale: bool = False,
        xatol: float = 1e-8,
        fatol: float = 1e-8,
        adaptive: bool = False,
        max_stride: int = 128,
        condition_limit: float = 1e12,
        trust_band: tuple[float, float] = (1.0 / 32.0, 32.0),
        history_size: int | None = None,
        record_simplices: bool = False,
    ) -> None:
        self.func = func
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.n = self.x0.size
        self.lower = np.broadcast_to(np.asarray(lower_bounds, float), (self.n,)).copy()
        self.upper = np.broadcast_to(np.asarray(upper_bounds, float), (self.n,)).copy()
        self.max_evaluations = int(max_evaluations)
        self.initial_simplex = initial_simplex
        self.curvature = curvature
        self.model_step = bool(model_step)
        self.fit_scale = bool(fit_scale)
        self.xatol = float(xatol)
        self.fatol = float(fatol)
        self.max_stride = int(max_stride)
        self.condition_limit = float(condition_limit)
        self.trust_band = trust_band
        self.history_size = 4 * self.n if history_size is None else int(history_size)
        self.record_simplices = bool(record_simplices)

        if self.model_step and curvature is None:
            raise ValueError("model_step=True requires a curvature.")

        n = self.n
        self.rho = 1.0
        self.chi = 1.0 + 2.0 / n if adaptive else 2.0
        self.psi = 0.75 - 1.0 / (2.0 * n) if adaptive else 0.5
        self.shrink = 1.0 - 1.0 / n if adaptive else 0.5

        self.evaluations = 0
        self._history: deque[tuple[NDArray[np.float64], float]] = deque(
            maxlen=max(self.history_size, 1)
        )
        self._scale = curvature.scale if curvature is not None else 1.0

    # Objective plumbing

    def _clip(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(x, self.lower, self.upper)

    def _evaluate(self, x: NDArray[np.float64]) -> float:
        self.evaluations += 1
        value = float(self.func(x))
        value = value if np.isfinite(value) else np.inf
        if self.fit_scale:
            self._history.append((np.asarray(x, float).copy(), value))
        return value

    def _default_simplex(self) -> NDArray[np.float64]:
        """SciPy's 5 %-per-coordinate starting simplex."""
        nonzdelt, zdelt = 0.05, 0.00025
        sim = np.empty((self.n + 1, self.n), dtype=float)
        sim[0] = self.x0
        for k in range(self.n):
            vertex = self.x0.copy()
            vertex[k] = (1 + nonzdelt) * vertex[k] if vertex[k] != 0 else zdelt
            sim[k + 1] = vertex
        return sim

    # ---- the Hessian-completion block ------------------------------------

    def _fit_model(
        self,
        sim: NDArray[np.float64],
        fsim: NDArray[np.float64],
        edges: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float] | None:
        """Fit ``(g, alpha)`` of ``m(x) = f_0 + g'd + alpha/2 d' H_1 d``.

        With ``fit_scale=False`` the scale is the donated one and the system is
        the square interpolation ``D g = b``.  With ``fit_scale=True`` the scale
        joins the unknowns and the extra rows come from points Nelder-Mead has
        already evaluated and discarded.
        """
        assert self.curvature is not None
        shape = self.curvature.shape
        f0 = fsim[0]

        # 1/2 d' H_1 d for each edge.
        quadratic = 0.5 * np.einsum("ij,ij->i", edges @ shape, edges)

        if not self.fit_scale:
            rhs = fsim[1:] - f0 - self._scale * quadratic
            if not np.all(np.isfinite(rhs)):
                return None
            try:
                gradient = np.linalg.solve(edges, rhs)
            except np.linalg.LinAlgError:
                return None
            return (gradient, self._scale) if np.all(np.isfinite(gradient)) else None

        # Least squares over the simplex plus recent discards.
        v0 = sim[0]
        radius = float(np.max(np.linalg.norm(edges, axis=1))) or 1.0
        rows = [edges]
        quads = [quadratic]
        values = [fsim[1:] - f0]
        extra_points: list[NDArray[np.float64]] = []
        extra_values: list[float] = []
        for point, value in self._history:
            offset = point - v0
            distance = float(np.linalg.norm(offset))
            if not np.isfinite(value) or distance > 4.0 * radius or distance < 1e-14:
                continue
            if np.min(np.linalg.norm(sim - point, axis=1)) < 1e-13 * max(radius, 1.0):
                continue
            extra_points.append(offset)
            extra_values.append(value - f0)
        if extra_points:
            extra = np.vstack(extra_points)
            rows.append(extra)
            quads.append(0.5 * np.einsum("ij,ij->i", extra @ shape, extra))
            values.append(np.asarray(extra_values, dtype=float))

        design = np.hstack([np.vstack(rows), np.concatenate(quads)[:, None]])
        target = np.concatenate(values)
        if not (np.all(np.isfinite(design)) and np.all(np.isfinite(target))):
            return None
        # Column-equilibrate: the gradient block and the quadratic column can
        # differ by many orders of magnitude on an ill-conditioned problem.
        norms = np.linalg.norm(design, axis=0)
        norms[norms == 0.0] = 1.0
        solution, *_ = np.linalg.lstsq(design / norms, target, rcond=None)
        solution = solution / norms
        if not np.all(np.isfinite(solution)):
            return None
        gradient, alpha = solution[:-1], float(solution[-1])
        if not np.isfinite(alpha) or alpha <= 0.0:
            # The data say the objective is not convex in the donated metric;
            # keep the previous magnitude rather than stepping uphill.
            alpha = self._scale
        self._scale = alpha
        return gradient, alpha

    def _propose(
        self,
        sim: NDArray[np.float64],
        fsim: NDArray[np.float64],
        trust_factor: float,
    ) -> tuple[NDArray[np.float64], float, bool, float] | None:
        """Return ``(x_plus, predicted_reduction, hit_boundary, scale)``.

        ``None`` when the edge matrix is too ill-conditioned to trust, the model
        cannot be fitted, or it predicts no descent.
        """
        assert self.curvature is not None
        shape, shape_inverse = self.curvature.shape, self.curvature.shape_inverse
        v0 = sim[0]
        edges = sim[1:] - v0

        if not np.all(np.isfinite(edges)):
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            condition = float(np.linalg.cond(edges))
        if not np.isfinite(condition) or condition > self.condition_limit:
            return None

        fitted = self._fit_model(sim, fsim, edges)
        if fitted is None:
            return None
        gradient, alpha = fitted

        # Newton step for m: s = -(alpha H_1)^-1 g.
        step = -(shape_inverse @ gradient) / alpha
        length = self.curvature.norm(step)
        if not np.isfinite(length) or length == 0.0:
            return None

        # Trust region anchored to the simplex's own H-metric extent, so the
        # model step is always commensurate with NM's current scale.
        extent = max(
            float(np.max([self.curvature.norm(edge) for edge in edges])), _FLOOR
        )
        radius = trust_factor * extent
        hit_boundary = length > radius
        if hit_boundary:
            step = step * (radius / length)

        predicted = -(float(gradient @ step) + 0.5 * alpha * float(step @ shape @ step))
        if not np.isfinite(predicted) or predicted <= 0.0:
            return None
        return self._clip(v0 + step), predicted, hit_boundary, alpha

    @staticmethod
    def _pivot(
        edges: NDArray[np.float64],
        step: NDArray[np.float64],
        fsim: NDArray[np.float64],
        f_candidate: float,
    ) -> tuple[int, float] | None:
        """Which vertex to drop, by the Lagrange-pivot rule.

        Writing ``step = sum_j lambda_j d_j``, dropping vertex ``j`` multiplies
        the simplex volume by ``|lambda_j|``.  Choose the largest among vertices
        the candidate actually beats, so the swap is both an improvement and the
        least damaging one available.  ``None`` when the candidate beats nothing.
        """
        try:
            coefficients = np.linalg.solve(edges.T, step)
        except np.linalg.LinAlgError:
            return None
        eligible = np.flatnonzero(fsim[1:] > f_candidate)
        if eligible.size == 0:
            return None
        magnitudes = np.abs(coefficients[eligible])
        if not np.all(np.isfinite(magnitudes)):
            return None
        best = int(eligible[int(np.argmax(magnitudes))])
        return best + 1, float(magnitudes.max())

    # Main loop

    def optimize(self) -> NMResult:
        n = self.n
        self.evaluations = 0
        self._history.clear()
        if self.curvature is not None:
            self._scale = self.curvature.scale

        sim = (
            self._default_simplex()
            if self.initial_simplex is None
            else np.asarray(self.initial_simplex, dtype=float).copy()
        )
        sim = self._clip(sim)
        fsim = np.array([self._evaluate(vertex) for vertex in sim])

        order = np.argsort(fsim)
        sim, fsim = sim[order], fsim[order]

        result = NMResult(
            best_solution=sim[0].copy(),
            best_fitness=float(fsim[0]),
            evaluations=self.evaluations,
            message="",
        )
        self._log(result, sim, fsim)

        iteration = 0
        stride = 1
        next_attempt = 0
        trust_factor = 1.0
        low, high = self.trust_band
        message = f"Budget exhausted: {self.max_evaluations} evaluations"

        while self.evaluations < self.max_evaluations:
            diameter = float(np.max(np.abs(sim[1:] - sim[0])))
            spread = float(np.max(np.abs(fsim[0] - fsim[1:])))
            if diameter <= self.xatol and spread <= self.fatol:
                message = (
                    f"Converged: simplex extent {diameter:.2e} <= xatol and "
                    f"fitness spread {spread:.2e} <= fatol"
                )
                break

            iteration += 1

            # ---- Hessian-completion model step -------------------------
            # The only block that differs from classic Nelder-Mead.  Try the
            # informed move first; fall through to the classic move if the
            # model is unusable or its point buys nothing.
            if self.model_step and iteration >= next_attempt:
                next_attempt = iteration + stride
                proposal = self._propose(sim, fsim, trust_factor)
                if proposal is None:
                    result.skipped += 1
                    stride = min(2 * stride, self.max_stride)
                else:
                    candidate, predicted, hit_boundary, alpha = proposal
                    f_candidate = self._evaluate(candidate)
                    ratio = float(fsim[0] - f_candidate) / predicted

                    if ratio < 0.25:
                        trust_factor = max(0.5 * trust_factor, low)
                    elif ratio > 0.75 and hit_boundary:
                        trust_factor = min(2.0 * trust_factor, high)

                    pivot = self._pivot(
                        sim[1:] - sim[0], candidate - sim[0], fsim, f_candidate
                    )
                    # Reset the schedule only when the model both *predicted*
                    # the decrease it delivered and improved the incumbent.
                    # Gating on the trust-region ratio (not on improvement
                    # alone) is what makes the overhead vanish on landscapes
                    # the donated curvature does not describe: there the model
                    # step descends a little but predicts badly, so the stride
                    # keeps doubling and NM gets its budget back.
                    improved_best = bool(f_candidate < fsim[0])
                    predictive = improved_best and ratio > 0.25
                    stride = 1 if predictive else min(2 * stride, self.max_stride)

                    result.attempt_iteration.append(iteration)
                    result.attempt_evaluation.append(self.evaluations)
                    result.attempt_accepted.append(pivot is not None)
                    result.attempt_improved_best.append(improved_best)
                    result.attempt_rho.append(ratio)
                    result.attempt_pivot.append(0.0 if pivot is None else pivot[1])
                    result.attempt_scale.append(alpha)
                    if self.record_simplices:
                        result.model_points.append(candidate.copy())

                    if pivot is not None:
                        index, _ = pivot
                        sim[index], fsim[index] = candidate, f_candidate
                        order = np.argsort(fsim)
                        sim, fsim = sim[order], fsim[order]
                        self._log(result, sim, fsim)
                        continue  # this *was* the iteration's move, for 1 eval
            # ---- end of the model step ---------------------------------

            # Classic Nelder-Mead move (identical to the framework optimizer).
            xbar = np.add.reduce(sim[:-1], 0) / n
            xr = self._clip((1 + self.rho) * xbar - self.rho * sim[-1])
            fxr = self._evaluate(xr)
            doshrink = False

            if fxr < fsim[0]:
                xe = self._clip(
                    (1 + self.rho * self.chi) * xbar - self.rho * self.chi * sim[-1]
                )
                fxe = self._evaluate(xe)
                if fxe < fxr:
                    sim[-1], fsim[-1] = xe, fxe
                else:
                    sim[-1], fsim[-1] = xr, fxr
            else:
                if fxr < fsim[-2]:
                    sim[-1], fsim[-1] = xr, fxr
                else:
                    if fxr < fsim[-1]:
                        xc = self._clip(
                            (1 + self.psi * self.rho) * xbar
                            - self.psi * self.rho * sim[-1]
                        )
                        fxc = self._evaluate(xc)
                        if fxc <= fxr:
                            sim[-1], fsim[-1] = xc, fxc
                        else:
                            doshrink = True
                    else:
                        xcc = self._clip((1 - self.psi) * xbar + self.psi * sim[-1])
                        fxcc = self._evaluate(xcc)
                        if fxcc < fsim[-1]:
                            sim[-1], fsim[-1] = xcc, fxcc
                        else:
                            doshrink = True

                    if doshrink:
                        sim[1:] = self._clip(sim[0] + self.shrink * (sim[1:] - sim[0]))
                        for k in range(1, n + 1):
                            fsim[k] = self._evaluate(sim[k])

            order = np.argsort(fsim)
            sim, fsim = sim[order], fsim[order]
            self._log(result, sim, fsim)

        result.best_solution = sim[0].copy()
        result.best_fitness = float(fsim[0])
        result.evaluations = self.evaluations
        result.message = message
        return result

    # Diagnostics

    def _log(
        self,
        result: NMResult,
        sim: NDArray[np.float64],
        fsim: NDArray[np.float64],
    ) -> None:
        best = float(fsim[0])
        if result.trace_best and best > result.trace_best[-1]:
            best = result.trace_best[-1]
        result.trace_evaluations.append(self.evaluations)
        result.trace_best.append(best)

        edges = sim[1:] - sim[0]
        sign, logdet = np.linalg.slogdet(edges)
        log_factorial = float(np.sum(np.log(np.arange(1, self.n + 1))))
        degenerate = sign == 0 or not np.isfinite(logdet)
        result.log_volume.append(
            -np.inf if degenerate else float((logdet - log_factorial) / np.log(10.0))
        )
        # Hadamard ratio in log space, so the product of edge lengths cannot
        # under/overflow on an ill-conditioned simplex.
        lengths = np.linalg.norm(edges, axis=1)
        if degenerate or np.any(lengths <= 0.0):
            result.log_shape_quality.append(-np.inf)
        else:
            result.log_shape_quality.append(
                float(logdet / np.log(10.0) - np.sum(np.log10(lengths)))
            )
        with np.errstate(over="ignore", invalid="ignore"):
            condition = float(np.linalg.cond(edges))
        result.log_edge_condition.append(
            float(np.log10(condition)) if np.isfinite(condition) else np.inf
        )

        if self.record_simplices:
            result.simplex_history.append(sim.copy())
