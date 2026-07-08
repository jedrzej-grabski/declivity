import numpy as np
from numpy.typing import NDArray
from typing import Any
import math


def norm(vector: NDArray[np.float64]) -> float:
    """
    Calculate the Euclidean norm (L2 norm) of a vector.

    Args:
        vector: Input vector

    Returns:
        Euclidean norm of the vector
    """
    return np.sqrt(np.sum(vector**2))


def success_probability(
    benchmark_fitness: float, population_fitness: NDArray[np.float64]
) -> float:
    """
    Calculate what proportion of the population has a better fitness than the benchmark.

    Args:
        benchmark_fitness: Fitness value to compare against
        population_fitness: Array of fitness values for the population

    Returns:
        Proportion of population with better fitness
    """
    return np.sum(population_fitness < benchmark_fitness) / len(population_fitness)


def calculate_ft(
    steps_buffer: NDArray[np.float64],
    n_dim: int,
    lambda_: int,
    path_length: int,
    current_ft: float,
    c_ft: float,
    path_ratio: float,
    chi_n: float,
    mu_eff: float,
) -> float:
    """
    Calculate new scaling factor Ft (step size).

    Args:
        steps_buffer: Buffer containing recent steps
        n_dim: Number of dimensions
        lambda_: Population size
        path_length: Size of evolution path
        current_ft: Current Ft value
        c_ft: Learning factor for Ft adaptation
        path_ratio: Path length control reference value
        chi_n: Expected length of a standard normal random vector
        mu_eff: Variance effectiveness factor

    Returns:
        New Ft value
    """
    # Reshape buffer to get individual steps
    steps = steps_buffer.reshape(-1, n_dim)
    steps = steps[-path_length:]  # Take last path_length steps

    # Calculate direct path
    direct_path = np.sum(steps, axis=0)
    direct_path_norm = norm(direct_path)

    # Calculate total path
    total_path = np.sum([norm(step) for step in steps])

    # Calculate new Ft
    return current_ft * math.exp(
        1
        / (math.sqrt(n_dim) + 1)
        * (c_ft * (chi_n / (total_path / direct_path_norm) - 1))
    )


def delete_inf_nan(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Replace any NaN or Inf values with a large finite value.

    Args:
        x: Array that may contain NaN or Inf values

    Returns:
        Array with NaN and Inf values replaced
    """
    result = x.copy()
    result[np.isnan(result)] = np.finfo(float).max
    result[np.isinf(result)] = np.finfo(float).max
    return result


def sample_from_history(
    history: list[NDArray[np.float64]],
    history_sample: NDArray[np.float64],
    lambda_: int,
) -> NDArray[np.float64]:
    """
    Sample indices from history entries.

    Args:
        history: list of history entries
        history_sample: Indices of history entries to sample from
        lambda_: Number of samples to generate

    Returns:
        Array of sampled indices
    """
    result = np.zeros(lambda_, dtype=int)
    for i in range(lambda_):
        history_idx = history_sample[i]
        result[i] = np.random.randint(0, history[history_idx].shape[1])
    return result
