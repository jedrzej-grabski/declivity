"""Hessian-completed Nelder-Mead: the simplex plus a donated Hessian is a
square quadratic model.

Why classic Nelder-Mead cannot use curvature
--------------------------------------------
Every Nelder-Mead move -- reflect, expand, contract, shrink -- is a *fixed
affine combination of the current vertices*, with coefficients that depend on
neither the objective nor any Hessian::

    x_r = (1 + rho) * xbar - rho * v_worst          rho, chi, psi, sigma constant

so the method is exactly affine-invariant: Nelder-Mead on ``f`` from simplex
``S`` and on ``f(A y)`` from ``A^-1 S`` trace out exact affine images of one
another.  The corollary is sharp, and it is why seeding the *initial simplex*
from a learned geometry underdelivers: the only place a Hessian can enter
classic Nelder-Mead is that initial simplex, and that is merely a change of
variables, consumed once and then forgotten.  Measured on the 10^6-conditioned
Ellipsoid at d=10, an anisotropically shaped starting simplex ends ~60x *worse*
than the isotropic default, with half the runs stalling early on a collapsed
simplex -- starting near-degenerate in a method with no mechanism to restore
geometry only reaches stagnation sooner.

Hessian completion
------------------
A quadratic model over ``n`` dimensions has ``1 + n + n(n+1)/2`` free
parameters, hopelessly underdetermined by the simplex's ``n + 1`` known values.
But if the Hessian is *given* -- which is exactly what
:class:`~declivity.utils.initial_geometry.InitialGeometry` carries -- then only
the constant and the gradient are unknown: ``1 + n`` unknowns against ``n + 1``
equations.  **The system is square.**  With ``v_0`` the best vertex and
``d_i = v_i - v_0``::

    m(x) = f_0 + g' (x - v_0) + 1/2 (x - v_0)' H (x - v_0)
    m(v_i) = f_i   <=>   g' d_i = f_i - f_0 - 1/2 d_i' H d_i =: b_i
    D g = b,       D = [d_1; ...; d_n]   (n x n)

``g`` is a *curvature-corrected simplex gradient*: the plain simplex gradient
Nelder-Mead implicitly follows, minus the second-order term the donated Hessian
supplies.  It costs one ``n x n`` solve and **zero function evaluations** -- the
model is fitted entirely from points the simplex already holds.  Its minimiser
is a Newton step from the best vertex, evaluated once per attempt::

    x+ = v_0 - H^-1 g

When the curvature came from a CMA-ES covariance this needs no inversion at all:
``H^-1`` is ``sigma^2 C``, and ``InitialGeometry.solve`` already applies it.

Three safeguards, each for a measured failure mode
--------------------------------------------------
1. **Geometry-preserving pivot, with a floor.**  Replacing the *worst* vertex
   destroys the simplex: a short model step drops the far vertex and pulls
   everything toward ``v_0``, so the simplex flattens onto a hyperplane and
   Nelder-Mead's own moves starve -- and nothing in Nelder-Mead restores it.
   Write the step in the edge basis, ``x+ - v_0 = sum_j lambda_j d_j``: dropping
   vertex ``j`` multiplies the simplex volume by exactly ``|lambda_j|``, so drop
   the largest one among the vertices the candidate beats (the Lagrange-pivot
   rule from model-based derivative-free optimisation), *and decline the step
   entirely* when even that best option falls below ``config.pivot_floor``.
   The floor is what makes this safe rather than merely tidy: choosing the least
   damaging swap still grinds the simplex away over dozens of insertions.  Its
   default of 4 encodes the empirical finding that short model steps are the
   harmful ones -- they are moves Nelder-Mead's reflection already makes better,
   bought at the price of geometry -- while long jumps, which is where a donated
   Hessian actually pays, sail past it.

2. **A trust region riding the simplex.**  The radius is
   ``trust_factor * (simplex extent in the H-metric)``, with the factor adapted
   by the classic actual/predicted ratio inside ``[trust_low, trust_high]``.
   Anchoring to the simplex keeps the model step commensurate with
   Nelder-Mead's current scale at every stage, with no problem-dependent
   constant, and makes the trust ball inherit the donated ellipsoid's shape.

3. **Ratio-gated schedule.**  The interval between attempts doubles unless an
   attempt both improved the incumbent *and* predicted the improvement.  Gating
   on predictiveness rather than improvement alone is what makes the overhead
   decay toward zero on landscapes the curvature does not describe: the run
   hands its budget back to plain Nelder-Mead, then wakes up if the landscape
   turns quadratic again.

Unknown magnitude
-----------------
The interpolation conditions stay **linear in (g, alpha)** when the donated
matrix is split as ``H = alpha * H_1`` with ``H_1`` of unit geometric-mean
eigenvalue::

    g' d + alpha * (1/2 d' H_1 d) = f(x) - f_0

so extra evaluated points make the system square again in ``n + 1`` unknowns.
Nelder-Mead discards a point every iteration; ``config.fit_scale`` recycles
those discards as the extra rows and solves for ``(g, alpha)`` by least squares.
The donated matrix then supplies only the shape and the run calibrates its own
magnitude -- which is what a CMA-ES covariance actually gives you.
"""

from collections import deque
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Union, final

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.neldermead_hc.config import NelderMeadHCConfig
from declivity.core.algorithm_factory import register_optimizer
from declivity.core.base_optimizer import OptimizationResult
from declivity.core.population_optimizer import PopulationOptimizer
from declivity.utils.constraint_handlers import ConstraintHandler
from declivity.utils.initial_geometry import InitialGeometry
from declivity.utils.population_initializers import (
    CovarianceSimplexInitializer,
    PopulationInitializer,
    SimplexPopulationInitializer,
)
from declivity.utils.repair_strategies import LamarckianRepair, RepairStrategy
from declivity.utils.stopping_conditions import StoppingCondition

if TYPE_CHECKING:
    from declivity.logging.neldermead_hc_logger import NelderMeadHCLogData

_FLOOR = 1e-30


class HCSimplexOperation(IntEnum):
    """Which move an iteration performed.

    Codes 0-4 match
    :class:`~declivity.algorithms.neldermead.neldermead_optimizer.SimplexOperation`
    so an operation timeline is directly comparable between the two optimizers;
    ``MODEL_STEP`` is the one this optimizer adds.
    """

    REFLECT = 0
    EXPAND = 1
    CONTRACT_OUTSIDE = 2
    CONTRACT_INSIDE = 3
    SHRINK = 4
    MODEL_STEP = 5


@final
@register_optimizer(AlgorithmChoice.NELDERMEAD_HC, NelderMeadHCConfig)
class NelderMeadHCOptimizer(
    PopulationOptimizer["NelderMeadHCLogData", NelderMeadHCConfig]
):
    """Nelder-Mead with a donated-Hessian model step.

    Consumes the shared learned-geometry seam: pass ``initial_geometry=`` and
    the curvature reaches the *model step* every iteration rather than only the
    initial simplex.  Without one, the model step uses ``B_0 = I`` -- a
    trust-region gradient step on the simplex gradient, which is the natural
    no-curvature control.
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: NelderMeadHCConfig | None = None,
        repair_strategy: RepairStrategy | None = None,
        population_initializer: PopulationInitializer | None = None,
        constraint_handler: ConstraintHandler | None = None,
        stopping_condition: StoppingCondition | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        initial_geometry: InitialGeometry | None = None,
        simplex_base_size: float | None = None,
    ) -> None:
        if config is None:
            config = NelderMeadHCConfig(dimensions=len(initial_point))

        dimensions = len(initial_point)
        geometry = initial_geometry or InitialGeometry.identity(dimensions)

        if population_initializer is not None and initial_geometry is not None:
            raise ValueError(
                "Pass either population_initializer or initial_geometry, not both."
            )
        if population_initializer is None:
            # Isotropic by default: the curvature is meant to reach the model
            # step, not the simplex (see config.shape_initial_simplex).  When a
            # base size is given, an identity-geometry initializer honours it,
            # so a shaped and an unshaped simplex are the *same size* and differ
            # only in shape -- which is what makes the two comparable.
            shaping = (
                geometry
                if config.shape_initial_simplex
                else (InitialGeometry.identity(dimensions))
            )
            if config.shape_initial_simplex or simplex_base_size is not None:
                population_initializer = CovarianceSimplexInitializer(
                    shaping,
                    base_size=simplex_base_size,
                    min_step=100.0 * config.xatol,
                )
            else:
                population_initializer = SimplexPopulationInitializer()

        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            repair_strategy=repair_strategy or LamarckianRepair(),
            population_initializer=population_initializer,
            algorithm=AlgorithmChoice.NELDERMEAD_HC,
            constraint_handler=constraint_handler,
            stopping_condition=stopping_condition,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )

        # Split the donated curvature into a unit-geometric-mean shape H_1 and a
        # magnitude, once.  Only the magnitude is re-fitted at run time, so the
        # eigendecomposition never repeats.
        dense = 0.5 * (geometry.dense() + geometry.dense().T)
        eigenvalues, basis = np.linalg.eigh(dense)
        eigenvalues = np.maximum(eigenvalues, _FLOOR)
        donated_scale = float(np.exp(np.mean(np.log(eigenvalues))))
        unit = eigenvalues / donated_scale
        self._shape: NDArray[np.float64] = (basis * unit) @ basis.T
        self._shape_inverse: NDArray[np.float64] = (basis * (1.0 / unit)) @ basis.T
        self._donated_scale = donated_scale
        self._scale = donated_scale

        self._history: deque[tuple[NDArray[np.float64], float]] = deque(
            maxlen=max(config.history_factor * dimensions, 1)
        )
        self._final_simplex: NDArray[np.float64] | None = None

        # Decaying averages of decrease-per-evaluation, one per move kind, so
        # the schedule can ask whether a model step is out-earning the classic
        # moves rather than only whether it predicted itself (see
        # ``config.gain_floor``).
        self._model_gain: float | None = None
        self._classic_gain: float | None = None

    def _blend(self, average: float | None, sample: float) -> float:
        decay = self.config.gain_decay
        return sample if average is None else decay * average + (1.0 - decay) * sample

    def _competitive(self) -> bool:
        """Is the model step still earning its evaluations?

        True until both averages exist, so a fresh run always gives the model a
        chance; afterwards the model must deliver at least ``gain_floor`` times
        the classic moves' decrease per evaluation.
        """
        if self._model_gain is None or self._classic_gain is None:
            return True
        return self._model_gain >= self.config.gain_floor * self._classic_gain

    @property
    def final_simplex(self) -> NDArray[np.float64] | None:
        """The ``(n+1, n)`` simplex at the end of the last ``optimize()`` call
        (defensive copy); ``None`` before any run."""
        if self._final_simplex is None:
            return None
        return self._final_simplex.copy()

    @property
    def curvature_scale(self) -> float:
        """The magnitude currently attributed to the donated curvature shape.

        Equal to the donated one unless ``config.fit_scale`` re-fitted it; the
        ratio to :attr:`donated_scale` is a direct, free measure of how well the
        donated geometry matches the local curvature."""
        return self._scale

    @property
    def donated_scale(self) -> float:
        """Geometric-mean eigenvalue of the curvature as donated."""
        return self._donated_scale

    # Objective plumbing

    def _shape_norm(self, v: NDArray[np.float64]) -> float:
        """``sqrt(v' H_1 v)`` -- the scale-free metric the trust region uses."""
        return float(np.sqrt(max(float(v @ self._shape @ v), 0.0)))

    def _repair_point(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Repair one trial vertex through the injected :class:`RepairStrategy`."""
        return self.repair_strategy.repair_population(
            x[np.newaxis, :], self.constraint_handler
        )[0]

    def _evaluate_trial(self, x: NDArray[np.float64]) -> float:
        """Evaluate and, when ``fit_scale`` is on, remember the point.

        Every evaluated point is a datum for the magnitude fit, including the
        ones Nelder-Mead is about to throw away.
        """
        value = self.evaluate(x)
        if self.config.fit_scale:
            self._history.append((np.asarray(x, dtype=float).copy(), float(value)))
        return value

    # The Hessian-completion block

    def _fit_model(
        self,
        sim: NDArray[np.float64],
        fsim: NDArray[np.float64],
        edges: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float] | None:
        """Fit ``(g, alpha)`` of ``m = f_0 + g'd + alpha/2 d' H_1 d``.

        Without ``fit_scale`` this is the square interpolation ``D g = b`` at the
        donated magnitude.  With it, the magnitude joins the unknowns and the
        extra rows come from already-evaluated, already-discarded points.
        """
        f0 = float(fsim[0])
        quadratic = 0.5 * np.einsum("ij,ij->i", edges @ self._shape, edges)

        if not self.config.fit_scale:
            rhs = fsim[1:] - f0 - self._scale * quadratic
            if not np.all(np.isfinite(rhs)):
                return None
            try:
                gradient = np.linalg.solve(edges, rhs)
            except np.linalg.LinAlgError:
                return None
            return (gradient, self._scale) if np.all(np.isfinite(gradient)) else None

        v0 = sim[0]
        radius = float(np.max(np.linalg.norm(edges, axis=1))) or 1.0
        offsets: list[NDArray[np.float64]] = []
        gaps: list[float] = []
        for point, value in self._history:
            offset = point - v0
            distance = float(np.linalg.norm(offset))
            # Keep the model local, and skip anything that duplicates a vertex.
            if not np.isfinite(value) or distance > 4.0 * radius or distance < 1e-14:
                continue
            if float(np.min(np.linalg.norm(sim - point, axis=1))) < 1e-13 * max(
                radius, 1.0
            ):
                continue
            offsets.append(offset)
            gaps.append(value - f0)

        rows = [edges]
        quads = [quadratic]
        values = [fsim[1:] - f0]
        if offsets:
            extra = np.vstack(offsets)
            rows.append(extra)
            quads.append(0.5 * np.einsum("ij,ij->i", extra @ self._shape, extra))
            values.append(np.asarray(gaps, dtype=float))

        design = np.hstack([np.vstack(rows), np.concatenate(quads)[:, None]])
        target = np.concatenate(values)
        if not (np.all(np.isfinite(design)) and np.all(np.isfinite(target))):
            return None
        # Column-equilibrate: on an ill-conditioned problem the gradient block
        # and the magnitude column differ by many orders of magnitude.
        norms = np.linalg.norm(design, axis=0)
        norms[norms == 0.0] = 1.0
        solution, *_ = np.linalg.lstsq(design / norms, target, rcond=None)
        solution = solution / norms
        if not np.all(np.isfinite(solution)):
            return None
        gradient, alpha = solution[:-1], float(solution[-1])
        if not np.isfinite(alpha) or alpha <= 0.0:
            # The data say the objective is not convex in the donated metric.
            # Keep the previous magnitude rather than stepping uphill.
            alpha = self._scale
        self._scale = alpha
        return gradient, alpha

    def _propose(
        self,
        sim: NDArray[np.float64],
        fsim: NDArray[np.float64],
        trust_factor: float,
    ) -> tuple[NDArray[np.float64], float, bool] | None:
        """The trust-region minimiser of the completed model.

        Returns ``(x_plus, predicted_reduction, hit_boundary)``, or ``None`` when
        the edge matrix is too ill-conditioned, the fit fails, or the model
        predicts no descent.
        """
        v0 = sim[0]
        edges = sim[1:] - v0
        if not np.all(np.isfinite(edges)):
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            condition = float(np.linalg.cond(edges))
        if not np.isfinite(condition) or condition > self.config.condition_limit:
            return None

        fitted = self._fit_model(sim, fsim, edges)
        if fitted is None:
            return None
        gradient, alpha = fitted

        step = -(self._shape_inverse @ gradient) / alpha
        length = self._shape_norm(step)
        if not np.isfinite(length) or length == 0.0:
            return None

        extent = max(float(np.max([self._shape_norm(edge) for edge in edges])), _FLOOR)
        radius = trust_factor * extent
        hit_boundary = length > radius
        if hit_boundary:
            step = step * (radius / length)

        predicted = -(
            float(gradient @ step) + 0.5 * alpha * float(step @ self._shape @ step)
        )
        if not np.isfinite(predicted) or predicted <= 0.0:
            return None
        return self._repair_point(v0 + step), predicted, hit_boundary

    def _pivot(
        self,
        edges: NDArray[np.float64],
        step: NDArray[np.float64],
        fsim: NDArray[np.float64],
        f_candidate: float,
    ) -> tuple[int, float] | None:
        """Which vertex to drop, by the Lagrange-pivot rule.

        Writing ``step = sum_j lambda_j d_j``, dropping vertex ``j`` multiplies
        the simplex volume by exactly ``|lambda_j|``.  Take the largest among the
        vertices the candidate beats, so the swap is both an improvement and the
        least damaging one available -- and decline it altogether when even that
        best option falls below ``config.pivot_floor``, which is what stops a run
        of marginal insertions from collapsing the simplex.  The evaluation is
        not wasted when we decline: it still moves the trust region and still
        joins the history the magnitude fit reads.
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
        best = float(magnitudes.max())
        if best < self.config.pivot_floor:
            return None
        return int(eligible[int(np.argmax(magnitudes))]) + 1, best

    # Main loop

    def optimize(self) -> OptimizationResult["NelderMeadHCLogData"]:
        config = self.config
        n = self.dimensions
        rho, chi, psi = config.rho, config.chi, config.psi
        shrink = config.sigma_shrink

        self.evaluations = 0
        self._history.clear()
        self._scale = self._donated_scale
        self._model_gain = None
        self._classic_gain = None
        self._begin_run()

        sim = self.population_initializer.generate_population(
            rng=self.rng,
            x0=self.initial_point,
            pop_size=n + 1,
            constraint_handler=self.constraint_handler,
        )
        sim = self.repair_strategy.repair_population(sim, self.constraint_handler)
        fsim = self.evaluate_population(sim)

        order = np.argsort(fsim)
        sim = np.take(sim, order, 0)
        fsim = np.take(fsim, order, 0)

        best_fitness = float(fsim[0])
        best_solution = sim[0].copy()
        iteration = 0
        stride = 1
        next_attempt = 0
        trust_factor = 1.0
        attempts = 0
        accepted = 0
        improvements = 0
        termination_message = None

        while not self.should_stop(iteration, best_fitness):
            simplex_diameter = float(np.max(np.abs(sim[1:] - sim[0])))
            fitness_spread = float(np.max(np.abs(fsim[0] - fsim[1:])))
            if simplex_diameter <= config.xatol and fitness_spread <= config.fatol:
                termination_message = (
                    f"Converged: simplex extent {simplex_diameter:.2e} <= "
                    f"xatol and fitness spread {fitness_spread:.2e} <= fatol"
                )
                break

            iteration += 1
            operation: HCSimplexOperation | None = None
            ratio = 0.0
            fitness_before = best_fitness
            evaluations_before = self.evaluations

            # ---- Hessian-completion model step -------------------------
            # The only block that differs from classic Nelder-Mead.  Try the
            # informed move first; fall through to the classic move when the
            # model is unusable or its point buys nothing.
            if config.model_step and iteration >= next_attempt:
                next_attempt = iteration + stride
                proposal = self._propose(sim, fsim, trust_factor)
                if proposal is None:
                    stride = min(2 * stride, config.max_stride)
                else:
                    candidate, predicted, hit_boundary = proposal
                    f_candidate = self._evaluate_trial(candidate)
                    attempts += 1
                    ratio = float(fsim[0] - f_candidate) / predicted

                    if ratio < config.ratio_threshold:
                        trust_factor = max(0.5 * trust_factor, config.trust_low)
                    elif ratio > 1.0 - config.ratio_threshold and hit_boundary:
                        trust_factor = min(2.0 * trust_factor, config.trust_high)

                    improved_best = bool(f_candidate < fsim[0])
                    improvements += improved_best
                    # A model step costs exactly one evaluation, so its decrease
                    # *is* its decrease-per-evaluation; the classic branch below
                    # divides by however many its move consumed.
                    self._model_gain = self._blend(
                        self._model_gain, max(float(fsim[0] - f_candidate), 0.0) / 1.0
                    )
                    predictive = (
                        improved_best
                        and ratio > config.ratio_threshold
                        and self._competitive()
                    )
                    stride = 1 if predictive else min(2 * stride, config.max_stride)

                    pivot = self._pivot(
                        sim[1:] - sim[0], candidate - sim[0], fsim, f_candidate
                    )
                    if pivot is not None:
                        index, _ = pivot
                        sim[index], fsim[index] = candidate, f_candidate
                        accepted += 1
                        operation = HCSimplexOperation.MODEL_STEP
            # ---- end of the model step ---------------------------------

            if operation is None:
                # Classic Nelder-Mead move.
                xbar = np.add.reduce(sim[:-1], 0) / n
                xr = self._repair_point((1 + rho) * xbar - rho * sim[-1])
                fxr = self._evaluate_trial(xr)
                operation = HCSimplexOperation.REFLECT
                doshrink = False

                if fxr < fsim[0]:
                    xe = self._repair_point(
                        (1 + rho * chi) * xbar - rho * chi * sim[-1]
                    )
                    fxe = self._evaluate_trial(xe)
                    if fxe < fxr:
                        sim[-1], fsim[-1] = xe, fxe
                        operation = HCSimplexOperation.EXPAND
                    else:
                        sim[-1], fsim[-1] = xr, fxr
                else:
                    if fxr < fsim[-2]:
                        sim[-1], fsim[-1] = xr, fxr
                    else:
                        if fxr < fsim[-1]:
                            xc = self._repair_point(
                                (1 + psi * rho) * xbar - psi * rho * sim[-1]
                            )
                            fxc = self._evaluate_trial(xc)
                            if fxc <= fxr:
                                sim[-1], fsim[-1] = xc, fxc
                                operation = HCSimplexOperation.CONTRACT_OUTSIDE
                            else:
                                doshrink = True
                        else:
                            xcc = self._repair_point((1 - psi) * xbar + psi * sim[-1])
                            fxcc = self._evaluate_trial(xcc)
                            if fxcc < fsim[-1]:
                                sim[-1], fsim[-1] = xcc, fxcc
                                operation = HCSimplexOperation.CONTRACT_INSIDE
                            else:
                                doshrink = True

                        if doshrink:
                            sim[1:] = sim[0] + shrink * (sim[1:] - sim[0])
                            sim[1:] = self.repair_strategy.repair_population(
                                sim[1:], self.constraint_handler
                            )
                            fsim[1:] = self.evaluate_population(sim[1:])
                            operation = HCSimplexOperation.SHRINK

            order = np.argsort(fsim)
            sim = np.take(sim, order, 0)
            fsim = np.take(fsim, order, 0)

            if operation is not HCSimplexOperation.MODEL_STEP:
                spent = max(self.evaluations - evaluations_before, 1)
                self._classic_gain = self._blend(
                    self._classic_gain,
                    max(fitness_before - float(fsim[0]), 0.0) / spent,
                )

            best_fitness = float(fsim[0])
            best_solution = sim[0].copy()

            eigenvalues = self._simplex_eigenvalues(sim) if config.diag_eigen else None
            volume, quality = (
                self._simplex_geometry(sim) if config.diag_volume else (0.0, 0.0)
            )

            self.logger.log_iteration(
                iteration=iteration,
                evaluations=self.evaluations,
                best_fitness=best_fitness,
                worst_fitness=float(fsim[-1]),
                mean_fitness=float(np.mean(fsim)),
                fitness=fsim,
                population=sim,
                best_solution=best_solution,
                simplex_diameter=float(np.max(np.abs(sim[1:] - sim[0]))),
                fitness_spread=float(np.max(np.abs(fsim[0] - fsim[1:]))),
                operation=int(operation),
                simplex_volume=volume,
                simplex_shape_quality=quality,
                eigenvalues=eigenvalues,
                model_attempts=attempts,
                model_accepted=accepted,
                model_improvements=improvements,
                model_ratio=ratio,
                trust_factor=trust_factor,
                curvature_scale=self._scale,
            )

        if termination_message is None:
            termination_message = self.stop_message

        self._final_simplex = sim
        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            message=termination_message,
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.NELDERMEAD_HC,
        )

    # Simplex geometry diagnostics

    @staticmethod
    def _simplex_eigenvalues(sim: NDArray[np.float64]) -> NDArray[np.float64]:
        """Eigenvalues (descending) of the vertex covariance."""
        covariance = np.atleast_2d(np.cov(sim.T))
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues, np.finfo(float).tiny)
        return eigenvalues[::-1]

    @staticmethod
    def _simplex_geometry(sim: NDArray[np.float64]) -> tuple[float, float]:
        """``(volume, shape_quality)`` from one determinant.

        ``shape_quality`` is the Hadamard ratio ``|det D| / prod ||d_i||`` in
        ``(0, 1]``: 1 means mutually orthogonal edges, near 0 means the simplex
        has flattened.  Unlike the raw volume it is scale-invariant, so it
        separates *degeneracy* from *convergence* -- a simplex sitting on the
        optimum is legitimately tiny but need not be degenerate.
        """
        n = sim.shape[1]
        edges = sim[1:] - sim[0]
        sign, logdet = np.linalg.slogdet(edges)
        if sign == 0 or not np.isfinite(logdet):
            return 0.0, 0.0
        log_factorial = float(np.sum(np.log(np.arange(1, n + 1))))
        volume = float(np.exp(logdet - log_factorial))
        lengths = np.linalg.norm(edges, axis=1)
        if np.any(lengths <= 0.0):
            return volume, 0.0
        return volume, float(np.exp(logdet - np.sum(np.log(lengths))))
