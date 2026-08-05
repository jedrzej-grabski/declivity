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
    "AllStoppingCondition",
    "AnyStoppingCondition",
    "ArmijoBacktracking",
    "BoxConstraintHandler",
    # Constraint handler surface
    "BoxStrategy",
    "BrentLineSearch",
    "CEC17Function",
    "ConstraintHandler",
    "ConstraintHandlerType",
    "DerivativeFreeLineSearch",
    "FixedInitialPointGenerator",
    "GeometryMode",
    "GoldenSectionLineSearch",
    # Line-search surface
    "GradientLineSearch",
    "HandoffTransform",
    "IdentityPopulationInitializer",
    "IdentityRepair",
    # Initial geometry surface (shared learned-curvature object)
    "InitialGeometry",
    "InitialHessian",
    "InitialHessianMode",
    # Initial point generation
    "InitialPointGenerator",
    "InitialPointGeneratorType",
    "LamarckianRepair",
    "LineSearchResult",
    "LineSearchStrategy",
    "MaxEvaluations",
    "MaxIterations",
    "MaxTime",
    "MeanSigmaPopulationInitializer",
    "MoreThuenteLineSearch",
    "NormalPopulationInitializer",
    # Stopping-condition surface
    "OptimizationState",
    # Population initializer surface
    "PopulationInitializer",
    "PopulationInitializerType",
    "Rastrigin",
    # Repair strategy surface
    "RepairStrategy",
    "RepairStrategyType",
    "Rosenbrock",
    "ScalarSearchResult",
    # Benchmark functions
    "Sphere",
    "Stagnation",
    "StagnationUnit",
    "StoppingCondition",
    "StoppingConditionType",
    "TargetFitness",
    "UniformInitialPointGenerator",
    "covariance_to_hessian_matrix",
    "default_stopping_condition",
]
