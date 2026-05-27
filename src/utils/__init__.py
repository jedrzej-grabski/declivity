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
    UniformInitialPointGenerator,
    FixedInitialPointGenerator,
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
from src.utils.population_initializers import (
    PopulationInitializer,
    NormalPopulationInitializer,
    MeanSigmaPopulationInitializer,
    IdentityPopulationInitializer,
    PopulationInitializerType,
)

__all__ = [
    # Constraint handler surface
    "BoxStrategy",
    "ConstraintHandler",
    "BoxConstraintHandler",
    "ConstraintHandlerType",
    # Initial point generation
    "InitialPointGenerator",
    "UniformInitialPointGenerator",
    "FixedInitialPointGenerator",
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
    # Population initializer surface
    "PopulationInitializer",
    "NormalPopulationInitializer",
    "MeanSigmaPopulationInitializer",
    "IdentityPopulationInitializer",
    "PopulationInitializerType",
]
