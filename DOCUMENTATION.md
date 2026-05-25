# Python-Evo: Evolutionary Algorithms Framework

A comprehensive Python framework for evolutionary optimization algorithms with implementations of DES (Differential Evolution Strategy), CMA-ES (Covariance Matrix Adaptation Evolution Strategy), and MF-CMA-ES (Matrix-Free CMA-ES).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Available Algorithms](#available-algorithms)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Benchmark Functions](#benchmark-functions)
- [Boundary Handling](#boundary-handling)
- [Logging and Diagnostics](#logging-and-diagnostics)
- [Visualization](#visualization)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [License](#license)

## Overview

Python-Evo is an object-oriented framework designed for evolutionary optimization research and applications. It provides:

- **Multiple Algorithm Implementations**: DES, CMA-ES, and MF-CMA-ES
- **Factory Pattern Design**: Easy algorithm selection and instantiation
- **Comprehensive Configuration**: Flexible parameter control with sensible defaults
- **Rich Diagnostics**: Built-in logging for algorithm behavior analysis
- **Boundary Handling**: Multiple constraint handling strategies * ConstraintHandling
- **Benchmark Functions**: Collection of standard test functions including CEC2017
- **Visualization Tools**: Built-in plotting for convergence analysis

### Architecture

The framework follows a modular architecture with clear separation of concerns:

```
python-evo/
├── src/
│   ├── core/              # Base classes and factory
│   ├── algorithms/        # Algorithm implementations
│   │   ├── des/          # Differential Evolution Strategy
│   │   ├── cmaes/        # CMA-ES
│   │   └── mfcmaes/      # Matrix-Free CMA-ES
│   ├── utils/            # Utilities and helpers
│   ├── logging/          # Diagnostic logging
│   └── plotting/         # Visualization tools
└── examples/             # Usage examples
```

## Installation

### Using pip (editable mode)

```bash
pip install -e .
```

### Using PDM

```bash
pdm install
```

### Dependencies

- Python >= 3.12
- NumPy >= 2.2.5
- SciPy >= 1.15.3
- Matplotlib >= 3.10.3
- opfunu >= 1.0.1 (for CEC benchmark functions)
- seaborn >= 0.13.2

## Core Concepts

### Algorithm Factory

The `AlgorithmFactory` provides a unified interface for creating optimizer instances:

```python
from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice

# Create an optimizer using the factory
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.DES,
    func=objective_function,
    initial_point=starting_point,
    config=config,
    lower_bounds=-100,
    upper_bounds=100
)
```

### Base Optimizer

All algorithms inherit from `BaseOptimizer`, which provides:

- Evaluation tracking
- Boundary constraint handling
- Logging infrastructure
- Population evaluation methods

### Configuration System

Each algorithm has a corresponding configuration class that extends `BaseConfig`:

- `DESConfig` for DES
- `CMAESConfig` for CMA-ES
- `MFCMAESConfig` for MF-CMA-ES

Configurations support:
- Dimension-dependent default parameters
- Diagnostic logging control
- Validation of parameter values

### Optimization Result

The `OptimizationResult` dataclass contains:

- `best_solution`: Best found solution vector
- `best_fitness`: Fitness value of best solution
- `evaluations`: Total function evaluations used
- `message`: Termination message
- `diagnostic`: Algorithm-specific diagnostic data
- `algorithm`: Algorithm identifier

## Available Algorithms

### 1. DES (Differential Evolution Strategy)

DES combines differential evolution with adaptive parameter control based on successful mutation history.

**Key Features:**
- Adaptive scaling factor (Ft) based on historical success
- Evolution path tracking
- Optional Lamarckian evolution
- Recombination using weighted parent selection

**Parameters:**
- `Ft`: Scaling factor of difference vectors (default: 1.0)
- `initFt`: Initial scaling factor (default: 1.0)
- `pathLength`: Size of evolution path (default: 6)
- `c_Ft`: Control parameter for Ft adaptation (default: 1)
- `Lamarckism`: Whether to use Lamarckian evolution (default: False)

**Default Settings:**
- Population size: `4 * dimensions`
- Budget: `10000 * dimensions`
- Parents (mu): `floor(population_size / 2)`

### 2. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

Classic CMA-ES algorithm with full covariance matrix adaptation.

**Key Features:**
- Covariance matrix adaptation for search distribution
- Step-size control via cumulative path
- Rank-one and rank-mu updates
- Multiple termination criteria

**Parameters:**
- `sigma`: Initial step size (0 for auto-calculation)
- `population_size`: Population size (0 for default)
- `tolfun`: Tolerance for function value changes (default: 1e-12)
- `tolx`: Tolerance for changes in x
- `tolxup`: Upper tolerance for step size (default: 1e4)
- `tolconditioncov`: Tolerance for covariance condition number (default: 1e14)

**Default Settings:**
- Population size: `4 + floor(3 * log(dimensions))`
- Budget: `10000 * dimensions`
- Parents (mu): `floor(population_size / 2)`

### 3. MF-CMA-ES (Matrix-Free CMA-ES)

Memory-efficient variant of CMA-ES that avoids storing the full covariance matrix.

**Key Features:**
- Matrix-free implementation for large-scale problems
- Optional PPMF (Precision-based Parameter Modification Framework)
- Reduced memory footprint
- Comparable performance to full CMA-ES

**Parameters:**
- Similar to CMA-ES with additional:
- `use_ppmf`: Enable/disable PPMF mechanism

## Quick Start

### Basic Usage

```python
import numpy as np
from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.utils.benchmark_functions import Sphere

# Define problem
dimensions = 10
func = Sphere(dimensions=dimensions)

# Create initial point
initial_point = np.random.uniform(-50, 50, dimensions)

# Create optimizer
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.DES,
    func=func,
    initial_point=initial_point,
    lower_bounds=-100,
    upper_bounds=100
)

# Run optimization
result = optimizer.optimize()

# Print results
print(f"Best fitness: {result.best_fitness}")
print(f"Best solution: {result.best_solution}")
print(f"Evaluations: {result.evaluations}")
print(f"Message: {result.message}")
```

### Using Custom Configuration

```python
from src.algorithms.des.config import DESConfig

# Create custom configuration
config = DESConfig(dimensions=10)
config.budget = 50000
config.population_size = 100
config.Ft = 0.8
config.pathLength = 10

# Enable diagnostics
config.enable_all_diagnostics()

# Create optimizer with custom config
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.DES,
    func=objective_function,
    initial_point=initial_point,
    config=config,
    lower_bounds=-100,
    upper_bounds=100
)

result = optimizer.optimize()
```

### Using Initial Point Generator

```python
from src.utils.initial_point_generator import (
    InitialPointGenerator,
    InitialPointGeneratorType
)

# Generate initial point
generator = InitialPointGenerator(
    strategy=InitialPointGeneratorType.UNIFORM,
    dimensions=10,
    lower_bounds=-50,
    upper_bounds=50
)

initial_point = generator.generate()
```

### Boundary Handling Strategies

```python
from src.utils.boundary_handlers import BoundaryHandlerType

# Create optimizer with specific boundary strategy
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.CMAES,
    func=objective_function,
    initial_point=initial_point,
    lower_bounds=-100,
    upper_bounds=100,
    boundary_strategy=BoundaryHandlerType.BOUNCE_BACK
)
```

## Configuration

### Base Configuration Options

All configurations inherit these common options:

```python
class BaseConfig:
    dimensions: int           # Problem dimensionality
    budget: int              # Max function evaluations
    population_size: int     # Population size

    # Diagnostic logging
    diag_enabled: bool       # Enable all diagnostics
    diag_value: bool         # Log population fitness values
    diag_mean: bool          # Log mean fitness
    diag_meanCords: bool     # Log mean coordinates
    diag_pop: bool           # Log populations
    diag_bestVal: bool       # Log best fitness (default: True)
    diag_worstVal: bool      # Log worst fitness
    diag_eigen: bool         # Log eigenvalues
```

### DES Configuration

```python
config = DESConfig(dimensions=10)

# Algorithm parameters
config.Ft = 1.0              # Scaling factor
config.initFt = 1.0          # Initial scaling factor
config.pathLength = 6        # Evolution path length
config.c_Ft = 1              # Ft adaptation control
config.Lamarckism = False    # Lamarckian evolution

# Diagnostics
config.diag_Ft = True        # Log Ft values
```

### CMA-ES Configuration

```python
config = CMAESConfig(dimensions=10)

# Algorithm parameters
config.sigma = 0.0           # Initial step size (0=auto)
config.population_size = 0   # Population size (0=default)
config.cm = 1.0              # Step size multiplier

# Termination criteria
config.tolfun = 1e-12        # Function value tolerance
config.tolx = 1e-12 * config.sigma  # x change tolerance
config.tolxup = 1e4          # Upper step size tolerance
config.tolconditioncov = 1e14  # Covariance condition tolerance

# Diagnostics
config.diag_sigma = True     # Log sigma values
config.diag_cond = True      # Log condition number
```

### MF-CMA-ES Configuration

```python
config = MFCMAESConfig(dimensions=10)

# Same as CMA-ES plus:
config.use_ppmf = True       # Enable PPMF mechanism
```

### Configuration Helpers

```python
# Enable all diagnostics
config.enable_all_diagnostics()

# Disable all diagnostics
config.disable_all_diagnostics()

# Enable only convergence diagnostics
config.with_convergence_diagnostics()

# Create config via factory
config = AlgorithmFactory.create_config(
    algorithm=AlgorithmChoice.DES,
    dimensions=10,
    budget=100000
)
```

## Benchmark Functions

The framework includes several standard benchmark functions:

### Built-in Functions

```python
from src.utils.benchmark_functions import (
    Sphere, Rosenbrock, Rastrigin, Ackley, Schwefel
)

# Sphere function: f(x) = sum(x_i^2)
sphere = Sphere(dimensions=10)

# Rosenbrock function (Valley function)
rosenbrock = Rosenbrock(dimensions=10)

# Rastrigin function (highly multimodal)
rastrigin = Rastrigin(dimensions=10)

# Ackley function
ackley = Ackley(dimensions=10)

# Schwefel function
schwefel = Schwefel(dimensions=10)

# Evaluate
fitness = sphere(solution_vector)

# Get bounds
lower, upper = sphere.bounds

# Get global minimum
optimal_solution, optimal_value = sphere.global_minimum
```

### CEC2017 Functions

```python
from src.utils.benchmark_functions import CEC17Function

# Create CEC2017 function
# Function IDs: 1-30
func = CEC17Function(dimensions=10, function_id=1)

# Evaluate
fitness = func(solution_vector)

# Get properties
lower, upper = func.bounds
opt_solution, opt_value = func.global_minimum
```

### Custom Functions

```python
import numpy as np
from numpy.typing import NDArray

def custom_objective(x: NDArray[np.float64]) -> float:
    """Custom objective function."""
    # Example: weighted sum of squares
    weights = np.arange(1, len(x) + 1)
    return np.sum(weights * x**2)

# Use with optimizer
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.DES,
    func=custom_objective,
    initial_point=initial_point,
    lower_bounds=-10,
    upper_bounds=10
)
```

## Boundary Handling

The framework provides two boundary handling strategies:

### Clamp Strategy (Default)

Violations are clamped to boundary values:

```python
from src.utils.boundary_handlers import BoundaryHandlerType

optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.DES,
    func=objective_function,
    initial_point=initial_point,
    lower_bounds=-100,
    upper_bounds=100,
    boundary_strategy=BoundaryHandlerType.CLAMP
)
```

### Bounce-Back Strategy

Solutions "bounce back" from boundaries into the feasible region:

```python
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.DES,
    func=objective_function,
    initial_point=initial_point,
    lower_bounds=-100,
    upper_bounds=100,
    boundary_strategy=BoundaryHandlerType.BOUNCE_BACK
)
```

### Custom Boundary Handler

```python
from src.utils.boundary_handlers import BoundaryHandler
import numpy as np

class CustomBoundaryHandler(BoundaryHandler):
    def repair(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Custom repair strategy."""
        # Implement your repair logic
        return repaired_x

# Use custom handler
handler = CustomBoundaryHandler(lower_bounds, upper_bounds)
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.DES,
    func=objective_function,
    initial_point=initial_point,
    boundary_handler=handler
)
```

## Logging and Diagnostics

### Accessing Diagnostic Data

```python
# Run optimization
result = optimizer.optimize()

# Access diagnostic data
diagnostics = result.diagnostic

# Common diagnostic fields (if enabled):
# - best_values: Best fitness per generation
# - mean_values: Mean fitness per generation
# - worst_values: Worst fitness per generation
# - mean_coordinates: Mean position per generation
# - populations: Population history
# - eigenvalues: Eigenvalue history

# Algorithm-specific diagnostics:
# DES: Ft_values (scaling factor history)
# CMA-ES/MF-CMA-ES: sigma_values, condition_numbers
```

### Diagnostic Configuration

```python
# Enable specific diagnostics
config = DESConfig(dimensions=10)
config.diag_bestVal = True   # Best fitness (always useful)
config.diag_mean = True      # Mean fitness
config.diag_Ft = True        # DES-specific

# Enable all diagnostics
config.enable_all_diagnostics()

# Minimal diagnostics (convergence only)
config.with_convergence_diagnostics()
```

### Accessing Logs During Optimization

```python
# Create optimizer
optimizer = AlgorithmFactory.create_optimizer(...)

# Run optimization
result = optimizer.optimize()

# Get logged data
log_data = optimizer.get_logs()

# Or from result
log_data = result.diagnostic
```

## Visualization

### Multi-Algorithm Plotter

```python
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter
from pathlib import Path

# Create plotter
plotter = MultiAlgorithmPlotter()

# Plot algorithm-specific metrics
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

metrics_path = output_dir / "des_metrics.png"
fig = plotter.plot_algorithm_specific_metrics(
    result=result,
    algorithm=AlgorithmChoice.DES,
    save_path=metrics_path
)
```

### Creating Custom Plots

```python
import matplotlib.pyplot as plt

# Get diagnostic data
diagnostics = result.diagnostic

# Plot convergence
plt.figure(figsize=(10, 6))
plt.semilogy(diagnostics.best_values)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Convergence Plot')
plt.grid(True)
plt.savefig('convergence.png')
```

## Advanced Usage

### Running Multiple Algorithms

```python
from src.algorithms.choices import AlgorithmChoice
from src import AlgorithmFactory

algorithms = [
    AlgorithmChoice.DES,
    AlgorithmChoice.CMAES,
    AlgorithmChoice.MFCMAES
]

results = {}
for algorithm in algorithms:
    config = AlgorithmFactory.create_config(
        algorithm=algorithm,
        dimensions=10
    )

    optimizer = AlgorithmFactory.create_optimizer(
        algorithm=algorithm,
        func=objective_function,
        initial_point=initial_point,
        config=config,
        lower_bounds=-100,
        upper_bounds=100
    )

    results[algorithm] = optimizer.optimize()

# Compare results
for algo, result in results.items():
    print(f"{algo.value}: {result.best_fitness:.6e}")
```

### Parameter Sweeps

```python
import numpy as np

# Sweep over population sizes
pop_sizes = [20, 40, 80, 160]
results = []

for pop_size in pop_sizes:
    config = DESConfig(dimensions=10)
    config.population_size = pop_size

    optimizer = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.DES,
        func=objective_function,
        initial_point=initial_point.copy(),
        config=config,
        lower_bounds=-100,
        upper_bounds=100
    )

    result = optimizer.optimize()
    results.append({
        'pop_size': pop_size,
        'best_fitness': result.best_fitness,
        'evaluations': result.evaluations
    })
```

### Checkpointing and Resume

```python
import pickle

# Run partial optimization
config = DESConfig(dimensions=10)
config.budget = 10000  # Partial budget

optimizer = AlgorithmFactory.create_optimizer(...)
result = optimizer.optimize()

# Save state
with open('checkpoint.pkl', 'wb') as f:
    pickle.dump({
        'result': result,
        'optimizer_state': optimizer.__dict__
    }, f)

# Later: Load and continue
with open('checkpoint.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# Create new optimizer with larger budget
config.budget = 50000
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.DES,
    func=objective_function,
    initial_point=checkpoint['result'].best_solution,  # Warm start
    config=config,
    lower_bounds=-100,
    upper_bounds=100
)
```

### Constrained Optimization

```python
def constrained_objective(x):
    """Objective with penalty for constraint violations."""
    # Original objective
    fitness = np.sum(x**2)

    # Add penalty for constraints
    # Example: sum(x) <= 0
    constraint_violation = max(0, np.sum(x))
    penalty = 1e6 * constraint_violation

    return fitness + penalty

# Use with optimizer
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.DES,
    func=constrained_objective,
    initial_point=initial_point,
    lower_bounds=-100,
    upper_bounds=100
)
```

### Parallel Evaluations (External)

```python
from multiprocessing import Pool

def evaluate_population(population):
    """Evaluate population in parallel."""
    with Pool() as pool:
        fitness = pool.map(objective_function, population)
    return np.array(fitness)

# Note: Current implementation evaluates sequentially
# To implement parallel evaluation, extend BaseOptimizer
# and override evaluate_population method
```

## API Reference

### Core Classes

#### AlgorithmFactory

```python
class AlgorithmFactory:
    @classmethod
    def create_optimizer(
        cls,
        algorithm: AlgorithmChoice,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: BaseConfig | None = None,
        boundary_handler: BoundaryHandler | None = None,
        boundary_strategy: BoundaryHandlerType | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        **kwargs
    ) -> BaseOptimizer

    @classmethod
    def create_config(
        cls,
        algorithm: AlgorithmChoice,
        dimensions: int,
        **kwargs
    ) -> BaseConfig

    @classmethod
    def get_available_algorithms(cls) -> list[AlgorithmChoice]

    @classmethod
    def register_algorithm(
        cls,
        name: AlgorithmChoice,
        optimizer_class: Type[BaseOptimizer],
        config_class: Type[BaseConfig]
    ) -> None
```

#### BaseOptimizer

```python
class BaseOptimizer(ABC, Generic[LogDataType, ConfigType]):
    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: ConfigType,
        algorithm: AlgorithmChoice = AlgorithmChoice.Unknown,
        boundary_handler: BoundaryHandler | None = None,
        boundary_strategy: BoundaryHandlerType | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0
    ) -> None

    def evaluate(self, x: NDArray[np.float64]) -> float
    def evaluate_population(self, population: NDArray[np.float64]) -> NDArray[np.float64]
    def get_logs(self) -> LogDataType

    @abstractmethod
    def optimize(self) -> OptimizationResult[LogDataType]
```

#### OptimizationResult

```python
@dataclass
class OptimizationResult(Generic[LogDataType]):
    best_solution: NDArray[np.float64]
    best_fitness: float
    evaluations: int
    message: str
    diagnostic: LogDataType
    algorithm: AlgorithmChoice = AlgorithmChoice.Unknown
```

#### BaseConfig

```python
@dataclass
class BaseConfig:
    dimensions: int
    budget: int = 0
    population_size: int = 0

    # Diagnostic flags
    diag_enabled: bool = False
    diag_value: bool = False
    diag_mean: bool = False
    diag_meanCords: bool = False
    diag_pop: bool = False
    diag_bestVal: bool = True
    diag_worstVal: bool = False
    diag_eigen: bool = False

    def validate(self) -> None
    def to_dict(self) -> dict[str, Any]
    def enable_all_diagnostics(self) -> None
    def disable_all_diagnostics(self) -> None
    def with_convergence_diagnostics(self) -> None
```

### Enumerations

#### AlgorithmChoice

```python
class AlgorithmChoice(Enum):
    Unknown = "Unknown"
    DES = "DES"
    MFCMAES = "MFCMAES"
    CMAES = "CMAES"
```

#### BoundaryHandlerType

```python
class BoundaryHandlerType(Enum):
    BOUNCE_BACK = "bounce_back"
    CLAMP = "clamp"
```

#### InitialPointGeneratorType

```python
class InitialPointGeneratorType(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    # Add other strategies as implemented
```

## Examples

### Complete Example: DES Optimization

```python
from typing import cast
import numpy as np
from pathlib import Path

from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.des.config import DESConfig
from src.utils.benchmark_functions import Rastrigin
from src.utils.boundary_handlers import BoundaryHandlerType
from src.utils.initial_point_generator import (
    InitialPointGenerator,
    InitialPointGeneratorType
)
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter

# Setup problem
dimensions = 10
func = Rastrigin(dimensions=dimensions)
lower_bounds, upper_bounds = func.bounds

# Generate initial point
generator = InitialPointGenerator(
    strategy=InitialPointGeneratorType.UNIFORM,
    dimensions=dimensions,
    lower_bounds=lower_bounds[0],
    upper_bounds=upper_bounds[0]
)
initial_point = generator.generate()

# Configure algorithm
config = DESConfig(dimensions=dimensions)
config.budget = 100000
config.population_size = 40
config.Ft = 0.9
config.pathLength = 8
config.enable_all_diagnostics()

# Create optimizer
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.DES,
    func=func,
    initial_point=initial_point,
    config=config,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    boundary_strategy=BoundaryHandlerType.CLAMP
)

# Run optimization
print("Starting optimization...")
result = optimizer.optimize()

# Print results
print(f"\nResults:")
print(f"Best fitness: {result.best_fitness:.10e}")
print(f"Evaluations: {result.evaluations}")
print(f"Status: {result.message}")
print(f"Distance to optimum: {np.linalg.norm(result.best_solution):.10e}")

# Save plots
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

plotter = MultiAlgorithmPlotter()
plotter.plot_algorithm_specific_metrics(
    result=result,
    algorithm=AlgorithmChoice.DES,
    save_path=output_dir / "des_metrics.png"
)

print(f"\nPlots saved to: {output_dir.absolute()}")
```

### Running PDM Examples

```bash
# Simple optimization example
pdm run run-example

# R comparison benchmark
pdm run run-r
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contributing

This is a research framework. For contributions or issues, please contact the repository maintainer.

## References

1. Hansen, N., & Ostermeier, A. (2001). Completely Derandomized Self-Adaptation in Evolution Strategies. Evolutionary Computation, 9(2), 159-195.

2. Loshchilov, I. (2014). A Computationally Efficient Limited Memory CMA-ES for Large Scale Optimization. GECCO 2014.

3. CEC2017: Congress on Evolutionary Computation benchmark suite.

## Contact

Author: Jedrzej Grabski
Email: grabski.dev@gmail.com
