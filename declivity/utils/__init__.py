"""
Utility modules for the optimization package.
"""

# Import core utilities without circular dependencies
from declivity.utils.constraint_handlers import (
    BoxStrategy,
    ConstraintHandler,
    BoxConstraintHandler,
    ConstraintHandlerType,
)
from declivity.utils.initial_point_generator import (
    InitialPointGenerator,
    UniformInitialPointGenerator,
    FixedInitialPointGenerator,
    InitialPointGeneratorType,
)
from declivity.utils.benchmark_functions import (
    Sphere,
    Rastrigin,
    Rosenbrock,
    CEC17Function,
)
from declivity.utils.repair_strategies import (
    RepairStrategy,
    IdentityRepair,
    LamarckianRepair,
    RepairStrategyType,
)
from declivity.utils.population_initializers import (
    PopulationInitializer,
    NormalPopulationInitializer,
    MeanSigmaPopulationInitializer,
    IdentityPopulationInitializer,
    PopulationInitializerType,
)
from declivity.utils.line_search import (
    GradientLineSearch,
    LineSearchStrategy,
    LineSearchResult,
    MoreThuenteLineSearch,
    ArmijoBacktracking,
)
from declivity.utils.stopping_conditions import (
    OptimizationState,
    StoppingCondition,
    MaxEvaluations,
    MaxIterations,
    MaxTime,
    TargetFitness,
    Stagnation,
    StagnationUnit,
    AnyStoppingCondition,
    AllStoppingCondition,
    StoppingConditionType,
    default_stopping_condition,
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
    "RepairStrategyType",
    # Population initializer surface
    "PopulationInitializer",
    "NormalPopulationInitializer",
    "MeanSigmaPopulationInitializer",
    "IdentityPopulationInitializer",
    "PopulationInitializerType",
    # Line-search surface
    "GradientLineSearch",
    "LineSearchStrategy",
    "LineSearchResult",
    "MoreThuenteLineSearch",
    "ArmijoBacktracking",
    # Stopping-condition surface
    "OptimizationState",
    "StoppingCondition",
    "MaxEvaluations",
    "MaxIterations",
    "MaxTime",
    "TargetFitness",
    "Stagnation",
    "StagnationUnit",
    "AnyStoppingCondition",
    "AllStoppingCondition",
    "StoppingConditionType",
    "default_stopping_condition",
]
