"""
Python Evolutionary Optimization Package
"""

# Import optimizer and logger modules to trigger @register_optimizer and
# @register_logger decorators.  These must come after the factory imports so
# the factories are already defined when the decorators execute.
import declivity.algorithms.bfgs.bfgs_optimizer
import declivity.algorithms.cmaes.cmaes_optimizer
import declivity.algorithms.des.des_optimizer
import declivity.algorithms.lbfgsb.lbfgsb_optimizer
import declivity.algorithms.mfcmaes.mfcmaes_optimizer
import declivity.algorithms.neldermead.neldermead_optimizer
import declivity.algorithms.neldermead_hc.neldermead_hc_optimizer
import declivity.algorithms.powell.powell_optimizer
import declivity.logging.bfgs_logger
import declivity.logging.cmaes_logger
import declivity.logging.des_logger
import declivity.logging.lbfgsb_logger
import declivity.logging.mfcmaes_logger
import declivity.logging.neldermead_hc_logger
import declivity.logging.neldermead_logger
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
