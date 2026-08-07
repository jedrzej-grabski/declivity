import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.core.config_base import BaseConfig
from declivity.logging.base_logger import BaseLogData, BaseLogger
from declivity.logging.logger_factory import LoggerFactory
from declivity.utils.constraint_handlers import (
    BoxConstraintHandler,
    BoxStrategy,
    ConstraintHandler,
)
from declivity.utils.stopping_conditions import (
    OptimizationState,
    StoppingCondition,
    default_stopping_condition,
)

LogDataType = TypeVar("LogDataType", bound=BaseLogData)
ConfigType = TypeVar("ConfigType", bound=BaseConfig)


@dataclass
class OptimizationResult(Generic[LogDataType]):
    """Result of an optimization run with proper typing."""

    best_solution: NDArray[np.float64]
    best_fitness: float
    evaluations: int
    message: str
    diagnostic: LogDataType
    algorithm: AlgorithmChoice = AlgorithmChoice.Unknown


class BaseOptimizer(ABC, Generic[LogDataType, ConfigType]):
    """Abstract base class for optimization algorithms.

    Boundary handling goes through the injected
    :class:`~declivity.utils.constraint_handlers.ConstraintHandler`:
    ``max_feasible_step``, ``project_direction``, ``projected_gradient``,
    ``feasible_step_interval``, ``is_feasible``, ``repair``, ``penalty``.

    :attr:`lower_bounds` / :attr:`upper_bounds` cache the handler's
    :meth:`~declivity.utils.constraint_handlers.ConstraintHandler.bounding_box`
    intersected with the box the caller asked for, for the box-based
    algorithms that need the arrays in hot loops.  Mutating them does not
    change the feasible region.
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: ConfigType,
        algorithm: AlgorithmChoice = AlgorithmChoice.Unknown,
        constraint_handler: ConstraintHandler | None = None,
        stopping_condition: StoppingCondition | None = None,
        lower_bounds: float | NDArray[np.float64] | list[float] = -100.0,
        upper_bounds: float | NDArray[np.float64] | list[float] = 100.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the base optimizer."""
        self.func = func
        self.initial_point = np.array(initial_point, dtype=float)
        self.dimensions = len(initial_point)
        self.config: ConfigType = config
        self.algorithm = algorithm
        self.evaluations = 0

        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)

        requested_lower = self._process_bounds(lower_bounds, self.dimensions)
        requested_upper = self._process_bounds(upper_bounds, self.dimensions)
        self.lower_bounds = requested_lower
        self.upper_bounds = requested_upper
        self._validate_bounds()

        self.constraint_handler: ConstraintHandler = (
            constraint_handler
            if constraint_handler is not None
            else BoxConstraintHandler(
                BoxStrategy.CLAMP, requested_lower, requested_upper
            )
        )

        # Intersect the caller's search box with the handler's own box; a
        # point has to satisfy both.  A handler that declares no box (the
        # unbounded default) leaves the caller's box untouched.
        handler_lower, handler_upper = self.constraint_handler.bounding_box(
            self.dimensions
        )
        self.lower_bounds = np.maximum(requested_lower, handler_lower)
        self.upper_bounds = np.minimum(requested_upper, handler_upper)
        self._validate_bounds()

        # The default is a ``10_000 * dimensions`` evaluation budget.
        self.stopping_condition: StoppingCondition = (
            stopping_condition
            if stopping_condition is not None
            else default_stopping_condition(self.dimensions)
        )

        # Per-run bookkeeping consumed by ``should_stop`` / the stopping
        # condition.  Populated afresh by ``_begin_run`` at the top of each
        # ``optimize()`` call.
        self._run_start_time: float = 0.0
        self._iterations: int = 0
        self._best_fitness: float = float("inf")
        self._stop_message: str | None = None

        self.logger: BaseLogger[LogDataType] = LoggerFactory.create_logger(
            algorithm, config
        )

    @staticmethod
    def _process_bounds(
        bounds: float | NDArray[np.float64] | list[float], dimensions: int
    ) -> NDArray[np.float64]:
        """Process bounds input into numpy array format."""
        if isinstance(bounds, (int, float)):
            return np.full(dimensions, bounds, dtype=float)
        else:
            return np.array(bounds, dtype=float)

    def _validate_bounds(self) -> None:
        """Validate that bounds are compatible."""
        if self.lower_bounds.shape != self.upper_bounds.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("Lower bounds must be less than or equal to upper bounds.")

    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate a single solution and increment the evaluation counter."""
        self.evaluations += 1
        return self.constraint_handler.penalty(x, self.func(x))

    def evaluate_population(
        self, population: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate a population of solutions.

        If the stopping condition imposes an evaluation cap
        (:meth:`StoppingCondition.remaining_evaluations`), the generation is
        trimmed to it and the members beyond are marked ``inf``.  Conditions
        that do not bound evaluations return ``None``, and the whole
        generation is evaluated.
        """
        count = population.shape[0]
        fitness = np.zeros(count)

        remaining = self.stopping_condition.remaining_evaluations(self.evaluations)
        n_eval = count if remaining is None else max(0, min(remaining, count))

        for i in range(n_eval):
            fitness[i] = self.evaluate(population[i])
        if n_eval < count:
            fitness[n_eval:] = float("inf")

        return fitness

    def _begin_run(self) -> None:
        """Reset per-run stopping-condition bookkeeping.

        Every ``optimize()`` implementation calls this once, before the main
        loop and before any pre-loop ``evaluate_population``.
        """
        self._run_start_time = time.monotonic()
        self._iterations = 0
        self._best_fitness = float("inf")
        self._stop_message = None
        self.stopping_condition.reset()

    def should_stop(self, iterations: int, best_fitness: float) -> bool:
        """Test the injected stopping condition against the current state.

        Used as the main-loop guard: ``while not self.should_stop(...)``.
        When the condition fires, its message is stashed for
        :attr:`stop_message`.
        """
        self._iterations = iterations
        self._best_fitness = best_fitness
        state = OptimizationState(
            evaluations=self.evaluations,
            iterations=iterations,
            best_fitness=best_fitness,
            elapsed_seconds=time.monotonic() - self._run_start_time,
        )
        if self.stopping_condition.should_stop(state):
            self._stop_message = self.stopping_condition.message
            return True
        return False

    @property
    def stop_message(self) -> str:
        """Termination message from the stopping condition that last fired.

        Falls back to a generic string if the loop exited without the
        condition firing.
        """
        return self._stop_message or "Stopping condition met."

    def get_logs(self) -> LogDataType:
        """Get all logged data with proper typing."""
        return self.logger.get_logs()

    @abstractmethod
    def optimize(self) -> OptimizationResult[LogDataType]:
        """Run the optimization algorithm."""
