"""Result of a single (problem, algorithm, seed) run.

Stores the convergence trace as two parallel lists. ``evaluations`` is the
cumulative function-evaluation count at each logged step; ``best_fitness``
is the best objective value seen so far. Both are monotone (non-decreasing
and non-increasing respectively).

For chained algorithms (e.g. CMA-ES -> L-BFGS-B handoff) the lists are
concatenated and ``handoff_eval`` marks the boundary.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunTrace:
    """A single run's convergence record."""

    algorithm: str
    """Algorithm name (matches the AlgorithmRun that produced this)."""

    problem: str
    """Problem name."""

    seed: int

    evaluations: list[int] = field(default_factory=list)
    """Cumulative evaluations at each logged point."""

    best_fitness: list[float] = field(default_factory=list)
    """Best fitness so far at each logged point."""

    final_evaluations: int = 0
    """Total evaluations consumed by the run."""

    final_fitness: float = float("inf")
    """Final best fitness."""

    handoff_eval: Optional[int] = None
    """Evaluation count at which an algorithm handoff occurred, if any."""

    handoff_iter: Optional[int] = None
    """CMA-ES generation count at which the handoff occurred. Same event as
    ``handoff_eval`` but expressed in iterations of the warmup algorithm.
    Useful when plots want to annotate handoffs in a population-size-agnostic
    way."""
