"""
Utility modules for the optimization package.
"""

# Import core utilities without circular dependencies
from declivity.utils.benchmark_functions import (
    CEC17Function,
    Rastrigin,
    Rosenbrock,
    Sphere,
)
from declivity.utils.constraint_handlers import (
    BoxConstraintHandler,
    BoxStrategy,
    ConstraintHandler,
    ConstraintHandlerType,
)
from declivity.utils.initial_geometry import (
    GeometryMode,
    HandoffTransform,
    InitialGeometry,
    InitialHessian,
    InitialHessianMode,
    covariance_to_hessian_matrix,
)
from declivity.utils.initial_point_generator import (
    FixedInitialPointGenerator,
    InitialPointGenerator,
    InitialPointGeneratorType,
    UniformInitialPointGenerator,
)
from declivity.utils.line_search import (
    ArmijoBacktracking,
    BrentLineSearch,
    DerivativeFreeLineSearch,
    GoldenSectionLineSearch,
    GradientLineSearch,
    LineSearchResult,
    LineSearchStrategy,
    MoreThuenteLineSearch,
    ScalarSearchResult,
)
from declivity.utils.population_initializers import (
    IdentityPopulationInitializer,
    MeanSigmaPopulationInitializer,
    NormalPopulationInitializer,
    PopulationInitializer,
    PopulationInitializerType,
)
from declivity.utils.repair_strategies import (
    IdentityRepair,
    LamarckianRepair,
    RepairStrategy,
    RepairStrategyType,
)
from declivity.utils.stopping_conditions import (
    AllStoppingCondition,
    AnyStoppingCondition,
    MaxEvaluations,
    MaxIterations,
    MaxTime,
    OptimizationState,
    Stagnation,
    StagnationUnit,
    StoppingCondition,
    StoppingConditionType,
    TargetFitness,
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
    "DerivativeFreeLineSearch",
    "ScalarSearchResult",
    "BrentLineSearch",
    "GoldenSectionLineSearch",
    # Initial geometry surface (shared learned-curvature object)
    "InitialGeometry",
    "InitialHessian",
    "InitialHessianMode",
    "GeometryMode",
    "HandoffTransform",
    "covariance_to_hessian_matrix",
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
