from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.logging.base_logger import BaseLogger
from declivity.logging.logger_factory import register_logger
from declivity.logging.neldermead_logger import NelderMeadLogData


@dataclass
class NelderMeadHCLogData(NelderMeadLogData):
    """Nelder-Mead log data plus the model step's own diagnostics.

    Every added field is a scalar per iteration, so
    :class:`~declivity.benchmarking.BenchmarkAlgorithm` captures them onto
    ``RunTrace.series`` automatically and a benchmark can be banded on them by
    the same panel that drew them for a single run.
    """

    simplex_shape_quality: list[float] = field(default_factory=list)
    """Hadamard ratio ``|det D| / prod ||d_i||`` of the simplex edges, in
    ``(0, 1]``: 1 = mutually orthogonal, near 0 = flattened onto a hyperplane.
    Scale-invariant, so unlike ``simplex_volume`` it distinguishes degeneracy
    from convergence.  Logged with ``diag_volume`` (same determinant)."""

    model_attempts: list[int] = field(default_factory=list)
    """Cumulative model-step attempts (each costs one evaluation)."""

    model_accepted: list[int] = field(default_factory=list)
    """Cumulative model steps that entered the simplex."""

    model_improvements: list[int] = field(default_factory=list)
    """Cumulative model steps that improved the incumbent."""

    model_ratio: list[float] = field(default_factory=list)
    """Trust-region ratio (actual / predicted decrease) of this iteration's
    attempt; ``0`` on iterations with no attempt."""

    trust_factor: list[float] = field(default_factory=list)
    """Dimensionless trust-region factor.  The radius is this times the
    simplex's own extent in the H-metric, so the series reads as "how far beyond
    the simplex the model is currently trusted"."""

    curvature_scale: list[float] = field(default_factory=list)
    """Magnitude currently attributed to the donated curvature shape.  With
    ``fit_scale`` on, its drift away from the donated value measures how badly
    the donated geometry's magnitude was calibrated."""

    def clear(self) -> None:
        super().clear()
        self.simplex_shape_quality.clear()
        self.model_attempts.clear()
        self.model_accepted.clear()
        self.model_improvements.clear()
        self.model_ratio.clear()
        self.trust_factor.clear()
        self.curvature_scale.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        return super().to_dict() | {
            "simplex_shape_quality": self.simplex_shape_quality,
            "model_attempts": self.model_attempts,
            "model_accepted": self.model_accepted,
            "model_improvements": self.model_improvements,
            "model_ratio": self.model_ratio,
            "trust_factor": self.trust_factor,
            "curvature_scale": self.curvature_scale,
        }


@register_logger(AlgorithmChoice.NELDERMEAD_HC)
class NelderMeadHCLogger(BaseLogger[NelderMeadHCLogData]):
    """Logger for Hessian-completed Nelder-Mead."""

    def __init__(self, config):
        super().__init__(config, AlgorithmChoice.NELDERMEAD_HC)

    def _create_log_data(self) -> NelderMeadHCLogData:
        return NelderMeadHCLogData()

    def log_iteration(
        self,
        iteration: int,
        evaluations: int,
        best_fitness: float = float("inf"),
        worst_fitness: float = float("inf"),
        mean_fitness: float = 0.0,
        fitness: NDArray[np.float64] | None = None,
        population: NDArray[np.float64] | None = None,
        best_solution: NDArray[np.float64] | None = None,
        simplex_diameter: float = 0.0,
        fitness_spread: float = 0.0,
        operation: int = 0,
        simplex_volume: float = 0.0,
        simplex_shape_quality: float = 0.0,
        eigenvalues: NDArray[np.float64] | None = None,
        model_attempts: int = 0,
        model_accepted: int = 0,
        model_improvements: int = 0,
        model_ratio: float = 0.0,
        trust_factor: float = 0.0,
        curvature_scale: float = 0.0,
        **kwargs,
    ) -> None:
        self.logs.iteration.append(iteration)
        self.logs.evaluations.append(evaluations)
        self.logs.best_fitness.append(best_fitness)
        self.logs.worst_fitness.append(worst_fitness)
        self.logs.mean_fitness.append(mean_fitness)
        self.logs.std_fitness.append(
            float(np.std(fitness)) if fitness is not None else 0.0
        )

        if population is not None and self.config.diag_pop:
            self.logs.population.append(population.copy())

        if best_solution is not None:
            self.logs.best_solution.append(best_solution.copy())

        if eigenvalues is not None and self.config.diag_eigen:
            self.logs.eigenvalues.append(eigenvalues.copy())
            if len(eigenvalues) > 0:
                self.logs.condition_number.append(
                    float(eigenvalues[0] / eigenvalues[-1])
                )

        self.logs.simplex_diameter.append(simplex_diameter)
        self.logs.fitness_spread.append(fitness_spread)

        if self.config.diag_operations:
            self.logs.operation.append(operation)

        if self.config.diag_volume:
            self.logs.simplex_volume.append(simplex_volume)
            self.logs.simplex_shape_quality.append(simplex_shape_quality)

        self.logs.model_attempts.append(model_attempts)
        self.logs.model_accepted.append(model_accepted)
        self.logs.model_improvements.append(model_improvements)
        self.logs.model_ratio.append(model_ratio)
        self.logs.trust_factor.append(trust_factor)
        self.logs.curvature_scale.append(curvature_scale)
