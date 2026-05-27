"""
Utility modules for the optimization package.
"""

# Import core utilities without circular dependencies
from src.utils.constraint_handlers import (
    BoxStrategy,
    ConstraintHandler,
    BoxConstraintHandler,
    ConstraintHandlerType,
)
from src.utils.initial_point_generator import (
    InitialPointGenerator,
    InitialPointGeneratorType,
)
from src.utils.benchmark_functions import (
    Sphere,
    Rastrigin,
    Rosenbrock,
    CEC17Function,
)
from src.utils.repair_strategies import (
    RepairStrategy,
    IdentityRepair,
    LamarckianRepair,
    ClampRepair,
    RepairStrategyType,
)

__all__ = [
    # Constraint handler surface
    "BoxStrategy",
    "ConstraintHandler",
    "BoxConstraintHandler",
    "ConstraintHandlerType",
    # Initial point generation
    "InitialPointGenerator",
    "InitialPointGeneratorType",
    # Benchmark functions
    "Sphere",
    "Rastrigin",
    "Rosenbrock",
    "CEC17Function",
    # Repair strategy surface
    "RepairStrategy",
    "IdentityRepair",
    "LamarckianRepair",
    "ClampRepair",
    "RepairStrategyType",
]
