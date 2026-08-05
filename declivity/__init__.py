"""
Python Evolutionary Optimization Package
"""

# Import optimizer and logger modules to trigger @register_optimizer and
# @register_logger decorators.
import declivity.algorithms.cmaes.cmaes_optimizer
import declivity.algorithms.des.des_optimizer
import declivity.algorithms.lbfgsb.lbfgsb_optimizer
import declivity.algorithms.mfcmaes.mfcmaes_optimizer
import declivity.algorithms.powell.powell_optimizer
import declivity.logging.cmaes_logger
import declivity.logging.des_logger
import declivity.logging.lbfgsb_logger
import declivity.logging.mfcmaes_logger
import declivity.logging.powell_logger  # noqa: F401
from declivity.algorithms.choices import AlgorithmChoice
from declivity.core.algorithm_factory import AlgorithmFactory, register_optimizer
from declivity.core.base_optimizer import BaseOptimizer, OptimizationResult
from declivity.core.config_base import BaseConfig
from declivity.logging.logger_factory import register_logger

__all__ = [
    "AlgorithmChoice",
    "AlgorithmFactory",
    "BaseConfig",
    "BaseOptimizer",
    "OptimizationResult",
    "register_logger",
    "register_optimizer",
]
