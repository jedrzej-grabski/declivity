"""
Utility modules for the optimization package.
"""

# Import core utilities without circular dependencies
from declivity.utils.benchmark_functions import (
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
    HessianScaling,
    InitialGeometry,
    InitialHessian,
    InitialHessianMode,
    covariance_to_hessian_matrix,
    scaling_factor,
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
    DerivativeFreeLineSearchType,
    GoldenSectionLineSearch,
    GradientLineSearch,
    LineSearchResult,
    LineSearchStrategy,
    MoreThuenteLineSearch,
    ScalarSearchResult,
)
from declivity.utils.population_initializers import (
    CovarianceSimplexInitializer,
    IdentityPopulationInitializer,
    MeanSigmaPopulationInitializer,
    NormalPopulationInitializer,
    PopulationInitializer,
    PopulationInitializerType,
    SimplexPopulationInitializer,
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
    "ConstraintHandler",
    "ConstraintHandlerType",
    "CovarianceSimplexInitializer",
    "DerivativeFreeLineSearch",
    "DerivativeFreeLineSearchType",
    "FixedInitialPointGenerator",
    "GeometryMode",
    "GoldenSectionLineSearch",
    # Line-search surface
    "GradientLineSearch",
    "HandoffTransform",
    "HessianScaling",
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
    "SimplexPopulationInitializer",
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
    "scaling_factor",
]
