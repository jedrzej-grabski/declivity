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
    SimplexPopulationInitializer,
    CovarianceSimplexInitializer,
    IdentityPopulationInitializer,
    PopulationInitializerType,
)
from declivity.utils.line_search import (
    GradientLineSearch,
    LineSearchStrategy,
    LineSearchResult,
    MoreThuenteLineSearch,
    ArmijoBacktracking,
    DerivativeFreeLineSearch,
    ScalarSearchResult,
    BrentLineSearch,
    GoldenSectionLineSearch,
)
from declivity.utils.initial_geometry import (
    InitialGeometry,
    InitialHessian,
    InitialHessianMode,
    GeometryMode,
    HandoffTransform,
    covariance_to_hessian_matrix,
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
    "SimplexPopulationInitializer",
    "CovarianceSimplexInitializer",
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
