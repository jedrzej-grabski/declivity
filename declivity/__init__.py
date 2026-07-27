"""
Python Evolutionary Optimization Package
"""

from declivity.algorithms.choices import AlgorithmChoice
from declivity.core.algorithm_factory import AlgorithmFactory, register_optimizer
from declivity.core.base_optimizer import BaseOptimizer, OptimizationResult
from declivity.core.config_base import BaseConfig
from declivity.logging.logger_factory import register_logger

# Import optimizer and logger modules to trigger @register_optimizer and
# @register_logger decorators.  These must come after the factory imports so
# the factories are already defined when the decorators execute.
import declivity.algorithms.des.des_optimizer  # noqa: F401
import declivity.algorithms.cmaes.cmaes_optimizer  # noqa: F401
import declivity.algorithms.mfcmaes.mfcmaes_optimizer  # noqa: F401
import declivity.algorithms.lbfgsb.lbfgsb_optimizer  # noqa: F401
import declivity.algorithms.powell.powell_optimizer  # noqa: F401
import declivity.algorithms.neldermead.neldermead_optimizer  # noqa: F401
import declivity.logging.des_logger  # noqa: F401
import declivity.logging.cmaes_logger  # noqa: F401
import declivity.logging.mfcmaes_logger  # noqa: F401
import declivity.logging.lbfgsb_logger  # noqa: F401
import declivity.logging.powell_logger  # noqa: F401
import declivity.logging.neldermead_logger  # noqa: F401

__all__ = [
    "AlgorithmFactory",
    "BaseOptimizer",
    "OptimizationResult",
    "BaseConfig",
    "register_optimizer",
    "register_logger",
]
