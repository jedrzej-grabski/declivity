"""
Python Evolutionary Optimization Package
"""

from src.algorithms.choices import AlgorithmChoice
from src.core.algorithm_factory import AlgorithmFactory, register_optimizer
from src.core.base_optimizer import BaseOptimizer, OptimizationResult
from src.core.config_base import BaseConfig
from src.logging.logger_factory import register_logger

# Import optimizer and logger modules to trigger @register_optimizer and
# @register_logger decorators.  These must come after the factory imports so
# the factories are already defined when the decorators execute.
import src.algorithms.des.des_optimizer  # noqa: F401
import src.algorithms.cmaes.cmaes_optimizer  # noqa: F401
import src.algorithms.mfcmaes.mfcmaes_optimizer  # noqa: F401
import src.algorithms.lbfgsb.lbfgsb_optimizer  # noqa: F401
import src.logging.des_logger  # noqa: F401
import src.logging.cmaes_logger  # noqa: F401
import src.logging.mfcmaes_logger  # noqa: F401
import src.logging.lbfgsb_logger  # noqa: F401

__all__ = [
    "AlgorithmFactory",
    "BaseOptimizer",
    "OptimizationResult",
    "BaseConfig",
    "register_optimizer",
    "register_logger",
]
