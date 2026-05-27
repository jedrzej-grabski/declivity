from __future__ import annotations

from typing import Callable, Type, TypeVar, Union, overload, Literal, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from src.algorithms.choices import AlgorithmChoice
from src.core.base_optimizer import BaseOptimizer

from src.core.config_base import BaseConfig
from src.utils.constraint_handlers import ConstraintHandler

if TYPE_CHECKING:
    from src.algorithms.des.des_optimizer import DESOptimizer
    from src.algorithms.des.config import DESConfig
    from src.algorithms.mfcmaes.mfcmaes_optimizer import MFCMAESOptimizer
    from src.algorithms.mfcmaes.mfcmaes_config import MFCMAESConfig
    from src.algorithms.cmaes.cmaes_optimizer import CMAESOptimizer
    from src.algorithms.cmaes.config import CMAESConfig
    from src.algorithms.lbfgsb.lbfgsb_optimizer import LBFGSBOptimizer
    from src.algorithms.lbfgsb.config import LBFGSBConfig


class AlgorithmFactory:
    """Factory for creating optimization algorithm instances."""

    _algorithms: dict[AlgorithmChoice, Type[BaseOptimizer]] = {}
    _configs: dict[AlgorithmChoice, Type[BaseConfig]] = {}

    @classmethod
    def register_algorithm(
        cls,
        name: AlgorithmChoice,
        optimizer_class: Type[BaseOptimizer],
        config_class: Type[BaseConfig],
    ) -> None:
        """Register a new optimization algorithm."""
        if name in cls._algorithms:
            raise ImportError(f"Duplicate optimizer registration for {name}")
        cls._algorithms[name] = optimizer_class
        cls._configs[name] = config_class

    @overload
    @classmethod
    def create_optimizer(
        cls,
        algorithm: Literal[AlgorithmChoice.DES],
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: "DESConfig | None" = None,
        constraint_handler: ConstraintHandler | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        **kwargs,
    ) -> DESOptimizer: ...

    # Generic fallback
    @overload
    @classmethod
    def create_optimizer(
        cls,
        algorithm: AlgorithmChoice,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: BaseConfig | None = None,
        constraint_handler: ConstraintHandler | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        **kwargs,
    ) -> BaseOptimizer: ...

    # Optional typed overload for MFCMAES
    @overload
    @classmethod
    def create_optimizer(
        cls,
        algorithm: Literal[AlgorithmChoice.MFCMAES],
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: "MFCMAESConfig | None" = None,
        constraint_handler: ConstraintHandler | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        **kwargs,
    ) -> "MFCMAESOptimizer": ...

    @overload
    @classmethod
    def create_optimizer(
        cls,
        algorithm: Literal[AlgorithmChoice.CMAES],
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: "CMAESConfig | None" = None,
        constraint_handler: ConstraintHandler | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        **kwargs,
    ) -> "CMAESOptimizer": ...

    @overload
    @classmethod
    def create_optimizer(
        cls,
        algorithm: Literal[AlgorithmChoice.LBFGSB],
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: "LBFGSBConfig | None" = None,
        constraint_handler: ConstraintHandler | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        **kwargs,
    ) -> "LBFGSBOptimizer": ...

    @classmethod
    def create_optimizer(
        cls,
        algorithm: AlgorithmChoice,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: BaseConfig | None = None,
        constraint_handler: ConstraintHandler | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        **kwargs,
    ) -> BaseOptimizer:
        """Create an optimizer instance with proper typing."""
        if algorithm not in cls._algorithms:
            available = ", ".join(str(k) for k in cls._algorithms.keys())
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")

        optimizer_class = cls._algorithms[algorithm]

        if config is None:
            config_class = cls._configs[algorithm]
            config = config_class(dimensions=len(initial_point))

        return optimizer_class(
            func=func,
            initial_point=initial_point,
            config=config,
            constraint_handler=constraint_handler,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
            **kwargs,
        )

    @classmethod
    def get_available_algorithms(cls) -> list[AlgorithmChoice]:
        """Get list of available algorithm names."""
        return list(cls._algorithms.keys())

    @classmethod
    def create_config(
        cls, algorithm: AlgorithmChoice, dimensions: int, **kwargs
    ) -> BaseConfig:
        """Create a configuration object for the specified algorithm."""
        if algorithm not in cls._configs:
            available = ", ".join(str(k) for k in cls._configs.keys())
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")

        config_class = cls._configs[algorithm]
        return config_class(dimensions=dimensions, **kwargs)


_OptimizerT = TypeVar("_OptimizerT", bound=BaseOptimizer)


def register_optimizer(
    choice: AlgorithmChoice, config_class: type[BaseConfig]
) -> Callable[[type[_OptimizerT]], type[_OptimizerT]]:
    """Class decorator that registers an optimizer with the AlgorithmFactory.

    Usage::

        @register_optimizer(AlgorithmChoice.DES, DESConfig)
        class DESOptimizer(BaseOptimizer[DESLogData, DESConfig]):
            ...

    The TypeVar preserves the decorated class's specific type, so
    ``CMAESOptimizer`` and friends retain their full constructor
    signature (including ``repair_strategy`` / ``population_initializer``)
    rather than being narrowed to ``type[BaseOptimizer]``.

    Raises:
        ImportError: if the same AlgorithmChoice is registered twice.
    """

    def decorator(cls: type[_OptimizerT]) -> type[_OptimizerT]:
        AlgorithmFactory.register_algorithm(choice, cls, config_class)
        return cls

    return decorator
