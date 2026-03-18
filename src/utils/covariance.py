"""Empirical covariance matrix estimation for populations."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


_EIGENVALUE_FLOOR = 1e-30


@dataclass(frozen=True)
class CovarianceMatrix:
    """Result of an empirical covariance estimation.

    Provides the covariance matrix along with its precomputed
    eigendecomposition, avoiding redundant recomputation.

    When pop_size <= dimensions, the empirical covariance is rank-deficient
    (at most rank pop_size - 1 for unweighted, pop_size for weighted).
    The `effective_rank` field tracks how many eigenvalues are meaningful,
    and `condition_number` only considers those.
    """

    matrix: NDArray[np.float64]
    """Covariance matrix of shape (d, d)."""

    eigenvalues: NDArray[np.float64]
    """Eigenvalues sorted in descending order, shape (d,)."""

    eigenvectors: NDArray[np.float64]
    """Eigenvectors as columns, ordered to match eigenvalues, shape (d, d)."""

    mean: NDArray[np.float64]
    """The mean vector used for centering, shape (d,)."""

    effective_rank: int
    """Number of meaningful eigenvalues (non-degenerate directions)."""

    @property
    def significant_eigenvalues(self) -> NDArray[np.float64]:
        """Only the eigenvalues corresponding to non-degenerate directions."""
        return self.eigenvalues[: self.effective_rank]

    @property
    def condition_number(self) -> float:
        """Ratio of largest to smallest *significant* eigenvalue."""
        sig = self.significant_eigenvalues
        if len(sig) == 0 or sig[-1] <= 0:
            return float("inf")
        return float(sig[0] / sig[-1])

    @property
    def dimensions(self) -> int:
        return self.matrix.shape[0]

    def sqrt_matrix(self) -> NDArray[np.float64]:
        """C^(1/2) = V @ diag(sqrt(eigenvalues)) @ V^T."""
        D_sqrt = np.sqrt(np.maximum(self.eigenvalues, 0))
        return self.eigenvectors @ np.diag(D_sqrt) @ self.eigenvectors.T

    def inv_sqrt_matrix(self) -> NDArray[np.float64]:
        """C^(-1/2) = V @ diag(1/sqrt(eigenvalues)) @ V^T."""
        D_inv_sqrt = np.where(
            self.eigenvalues > 0,
            1.0 / np.sqrt(self.eigenvalues),
            0.0,
        )
        return self.eigenvectors @ np.diag(D_inv_sqrt) @ self.eigenvectors.T


def empirical_covariance(
    population: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
) -> CovarianceMatrix:
    """Compute the sample covariance matrix of a population.

    Args:
        population: Array of shape (pop_size, dimensions), one individual per row.
        mean: Center for the covariance computation. If None, uses the sample mean.

    Returns:
        CovarianceMatrix with the covariance and its eigendecomposition.
    """
    if population.ndim != 2:
        raise ValueError(
            f"population must be 2-D (pop_size, dimensions), got shape {population.shape}"
        )

    if mean is None:
        mean = np.mean(population, axis=0)

    centered = population - mean
    n = population.shape[0]
    cov = (centered.T @ centered) / (n - 1)

    # Effective rank: at most n-1 (due to centering) or d, whichever is smaller
    rank = min(n - 1, population.shape[1])

    return _decompose(cov, mean, rank)


def weighted_covariance(
    population: NDArray[np.float64],
    weights: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
) -> CovarianceMatrix:
    """Compute a weighted empirical covariance matrix.
    Args:
        population: Array of shape (pop_size, dimensions), one individual per row.
            Typically only the selected (mu best) individuals.
        weights: Non-negative weights of shape (pop_size,), summing to 1.
        mean: Center for the covariance computation.
            If None, uses the weighted mean: m = sum_i w_i * x_i.

    Returns:
        CovarianceMatrix with the weighted covariance and its eigendecomposition.
    """
    if population.ndim != 2:
        raise ValueError(
            f"population must be 2-D (pop_size, dimensions), got shape {population.shape}"
        )
    if weights.shape[0] != population.shape[0]:
        raise ValueError(
            f"weights length ({weights.shape[0]}) must match "
            f"population rows ({population.shape[0]})"
        )

    if mean is None:
        mean = population.T @ weights

    centered = population - mean
    cov = (centered.T * weights) @ centered

    # Weighted case: rank is at most min(n, d) since mean may be external
    n = population.shape[0]
    rank = min(n, population.shape[1])

    return _decompose(cov, mean, rank)


def _decompose(
    cov: NDArray[np.float64],
    mean: NDArray[np.float64],
    max_rank: int,
) -> CovarianceMatrix:
    """Eigendecompose a symmetric matrix and wrap in CovarianceMatrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    significant = np.sum(eigenvalues > _EIGENVALUE_FLOOR)
    effective_rank = min(int(significant), max_rank)

    return CovarianceMatrix(
        matrix=cov,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        mean=mean,
        effective_rank=effective_rank,
    )
