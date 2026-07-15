"""
PopulationOptimizer — abstract base class for population-based algorithms.

Sits between :class:`BaseOptimizer` and the three evolutionary algorithms
(DES, CMA-ES, MF-CMA-ES).  Adds a **required** ``repair_strategy``
attribute that is structurally enforced at the constructor level:
pyright/mypy reject a subclass whose ``__init__`` does not forward a
concrete :class:`~src.utils.repair_strategies.RepairStrategy` instance
to ``super().__init__``.

Relationship to the logging layer
----------------------------------
This mirrors the existing :class:`~src.logging.base_logger.BaseLogData` /
:class:`~src.logging.base_logger.PopulationLogData` split: ``BaseOptimizer``
is paired with ``BaseLogData`` (single-point algorithms such as L-BFGS-B),
and ``PopulationOptimizer`` is paired with ``PopulationLogData``
(evolutionary algorithms).
"""

from abc import ABC
from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.core.base_optimizer import BaseOptimizer, LogDataType, ConfigType
from declivity.utils.constraint_handlers import ConstraintHandler
from declivity.utils.repair_strategies import RepairStrategy
from declivity.utils.population_initializers import PopulationInitializer
from declivity.utils.stopping_conditions import StoppingCondition


class PopulationOptimizer(BaseOptimizer[LogDataType, ConfigType], ABC):
    """Abstract base class for population-based optimisation algorithms.

    Extends :class:`BaseOptimizer` with two *mandatory* attributes:

    - ``repair_strategy`` — how infeasible individuals are repaired.
    - ``population_initializer`` — how the initial population is seeded.

    Every evolutionary optimiser (DES, CMA-ES, MF-CMA-ES) must supply
    concrete instances for both; subclasses typically provide sensible
    per-algorithm defaults and allow callers to override.

    The structural enforcement means the following fails type-checking::

        class BadEvolutionary(PopulationOptimizer[LogDataType, ConfigType]):
            def __init__(self, func, initial_point, config):
                # Missing repair_strategy / population_initializer — pyright error.
                super().__init__(func, initial_point, config)

    Parameters common to all evolutionary algorithms are validated and
    stored once here rather than being duplicated across subclasses.
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: ConfigType,
        repair_strategy: RepairStrategy,
        population_initializer: PopulationInitializer,
        algorithm: AlgorithmChoice = AlgorithmChoice.Unknown,
        constraint_handler: ConstraintHandler | None = None,
        stopping_condition: StoppingCondition | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """Initialise the population optimiser.

        Parameters
        ----------
        func:
            Objective function to minimise.
        initial_point:
            Starting point for the algorithm (length determines
            dimensionality).
        config:
            Algorithm-specific configuration dataclass.
        repair_strategy:
            **Required — no default.**  Strategy used to repair infeasible
            population members.  Callers must pass an explicit concrete
            instance; subclasses typically default to ``LamarckianRepair``
            (the canonical "apply ``ConstraintHandler.repair_batch`` to
            every individual" policy) and allow callers to override.
        population_initializer:
            **Required — no default.**  Strategy used to seed the initial
            population.  Subclasses pass a per-algorithm default
            (``NormalPopulationInitializer`` for DES,
            ``MeanSigmaPopulationInitializer`` for CMA-ES and MF-CMA-ES)
            and allow callers to override.
        algorithm:
            :class:`~src.algorithms.choices.AlgorithmChoice` enum value
            forwarded to :class:`BaseOptimizer` for logging and result
            labelling.
        constraint_handler:
            Feasibility / penalty handler.  Defaults to
            ``BoxConstraintHandler(BoxStrategy.CLAMP, ...)`` if ``None``.
        lower_bounds / upper_bounds:
            Search-space bounds (scalar broadcast or per-dimension array).
        seed:
            Random seed or ``numpy.random.Generator`` instance.
        """
        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            algorithm=algorithm,
            constraint_handler=constraint_handler,
            stopping_condition=stopping_condition,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )
        self.repair_strategy: RepairStrategy = repair_strategy
        self.population_initializer: PopulationInitializer = population_initializer
