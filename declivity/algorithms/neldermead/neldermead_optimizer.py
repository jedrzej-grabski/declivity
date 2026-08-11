from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Union, final

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.neldermead.config import NelderMeadConfig
from declivity.core.algorithm_factory import register_optimizer
from declivity.core.base_optimizer import OptimizationResult
from declivity.core.population_optimizer import PopulationOptimizer
from declivity.utils.constraint_handlers import ConstraintHandler
from declivity.utils.population_initializers import (
    CovarianceSimplexInitializer,
    PopulationInitializer,
    SimplexPopulationInitializer,
)
from declivity.utils.repair_strategies import LamarckianRepair, RepairStrategy
from declivity.utils.stopping_conditions import StoppingCondition

if TYPE_CHECKING:
    from declivity.logging.neldermead_logger import NelderMeadLogData
    from declivity.utils.initial_geometry import InitialGeometry


class SimplexOperation(IntEnum):
    """Which simplex move an iteration performed.

    Logged as integer codes so the operation timeline can be plotted as
    an ordinary numeric series.
    """

    REFLECT = 0
    EXPAND = 1
    CONTRACT_OUTSIDE = 2
    CONTRACT_INSIDE = 3
    SHRINK = 4


@final
@register_optimizer(AlgorithmChoice.NELDERMEAD, NelderMeadConfig)
class NelderMeadOptimizer(PopulationOptimizer["NelderMeadLogData", NelderMeadConfig]):
    """Nelder-Mead simplex optimizer."""

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: NelderMeadConfig | None = None,
        repair_strategy: RepairStrategy | None = None,
        population_initializer: PopulationInitializer | None = None,
        constraint_handler: ConstraintHandler | None = None,
        stopping_condition: StoppingCondition | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        initial_geometry: "InitialGeometry | None" = None,
        simplex_base_size: float | None = None,
    ) -> None:
        if config is None:
            config = NelderMeadConfig(dimensions=len(initial_point))

        if initial_geometry is not None:
            if population_initializer is not None:
                raise ValueError(
                    "Pass either population_initializer or initial_geometry, not both."
                )
            # Shape the initial simplex from the geometry's principal axes.
            # ``min_step`` keeps a collapsed covariance from producing a
            # simplex that satisfies xatol/fatol immediately.
            population_initializer = CovarianceSimplexInitializer(
                initial_geometry,
                base_size=simplex_base_size,
                min_step=100.0 * config.xatol,
            )

        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            repair_strategy=repair_strategy or LamarckianRepair(),
            population_initializer=population_initializer
            or SimplexPopulationInitializer(),
            algorithm=AlgorithmChoice.NELDERMEAD,
            constraint_handler=constraint_handler,
            stopping_condition=stopping_condition,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )

        self._final_simplex: NDArray[np.float64] | None = None

    @property
    def final_simplex(self) -> NDArray[np.float64] | None:
        """The ``(n+1, n)`` simplex at the end of the last ``optimize()`` call
        (defensive copy); ``None`` before any run."""
        if self._final_simplex is None:
            return None
        return self._final_simplex.copy()

    def _repair_point(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Repair one trial vertex through the injected :class:`RepairStrategy`.

        A reflected / expanded / contracted vertex goes through the same
        policy layer as the initial simplex and the shrink step.  Under the
        default :class:`~declivity.utils.repair_strategies.LamarckianRepair`
        this is ``ConstraintHandler.repair(x)``.
        """
        return self.repair_strategy.repair_population(
            x[np.newaxis, :], self.constraint_handler
        )[0]

    # Main loop

    def optimize(self) -> OptimizationResult["NelderMeadLogData"]:
        config = self.config
        n = self.dimensions
        rho = config.rho
        chi = config.chi
        psi = config.psi
        sigma = config.sigma_shrink

        self.evaluations = 0
        self._begin_run()

        # Initial simplex: (n+1, n) through the population seams.
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

            # Reflect the worst vertex through the centroid of the rest.
            xbar = np.add.reduce(sim[:-1], 0) / n
            xr = self._repair_point((1 + rho) * xbar - rho * sim[-1])
            fxr = self.evaluate(xr)
            operation = SimplexOperation.REFLECT
            doshrink = False

            if fxr < fsim[0]:
                # Best so far — try expanding further.
                xe = self._repair_point((1 + rho * chi) * xbar - rho * chi * sim[-1])
                fxe = self.evaluate(xe)
                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                    operation = SimplexOperation.EXPAND
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:  # fxr >= fsim[-2]: contract
                    if fxr < fsim[-1]:
                        xc = self._repair_point(
                            (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        )
                        fxc = self.evaluate(xc)
                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                            operation = SimplexOperation.CONTRACT_OUTSIDE
                        else:
                            doshrink = True
                    else:
                        xcc = self._repair_point((1 - psi) * xbar + psi * sim[-1])
                        fxcc = self.evaluate(xcc)
                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                            operation = SimplexOperation.CONTRACT_INSIDE
                        else:
                            doshrink = True

                    if doshrink:
                        sim[1:] = sim[0] + sigma * (sim[1:] - sim[0])
                        sim[1:] = self.repair_strategy.repair_population(
                            sim[1:], self.constraint_handler
                        )
                        fsim[1:] = self.evaluate_population(sim[1:])
                        operation = SimplexOperation.SHRINK

            order = np.argsort(fsim)
            sim = np.take(sim, order, 0)
            fsim = np.take(fsim, order, 0)

            best_fitness = float(fsim[0])
            best_solution = sim[0].copy()

            eigenvalues = self._simplex_eigenvalues(sim) if config.diag_eigen else None

            simplex_volume = self._simplex_volume(sim) if config.diag_volume else 0.0

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
                simplex_volume=simplex_volume,
                eigenvalues=eigenvalues,
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
            algorithm=AlgorithmChoice.NELDERMEAD,
        )

    # Simplex geometry diagnostics

    @staticmethod
    def _simplex_eigenvalues(sim: NDArray[np.float64]) -> NDArray[np.float64]:
        """Eigenvalues (descending) of the vertex covariance.

        Its condition number tracks how anisotropic the simplex has become.
        """
        covariance = np.cov(sim.T)
        covariance = np.atleast_2d(covariance)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues, np.finfo(float).tiny)
        return eigenvalues[::-1]

    @staticmethod
    def _simplex_volume(sim: NDArray[np.float64]) -> float:
        """Volume of the simplex: |det(edge matrix)| / n!."""
        n = sim.shape[1]
        edges = sim[1:] - sim[0]
        _, logdet = np.linalg.slogdet(edges)
        if not np.isfinite(logdet):
            return 0.0
        log_factorial = float(np.sum(np.log(np.arange(1, n + 1))))
        return float(np.exp(logdet - log_factorial))
