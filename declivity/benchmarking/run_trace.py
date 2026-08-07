"""Result of a single (problem, algorithm, seed) run.

A ``RunTrace`` is a trimmed ``LogData`` plus run identity and summary
metadata.  It stores the convergence trace as two parallel lists:
``evaluations`` is the cumulative function-evaluation count at each logged
step, ``best_fitness`` the best objective value so far.  Both are monotone.

``series`` holds any additional cheap scalar-per-step diagnostics retained
from the full ``LogData`` at run time (``sigma``, ``condition_number``,
``mean_fitness``, ...).  The persisted record and the in-memory ``LogData``
expose the same named series via :meth:`get_series`, so one plotter serves
both.  Heavy per-iteration fields (population matrices, eigenvalue vectors,
the per-step ``best_solution``) are not retained.

For chained algorithms (e.g. CMA-ES -> L-BFGS-B handoff) the lists are
concatenated and ``handoff_eval`` marks the boundary.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Per-iteration fields too heavy to persist per seed, and not aggregatable
# into a cross-seed band. ``capture_scalar_series`` always skips these.
HEAVY_LOGDATA_FIELDS = frozenset({"best_solution", "population", "eigenvalues"})

# Fields that live as first-class attributes on RunTrace, so they are never
# duplicated into the ``series`` dict.
_TOP_LEVEL_SERIES = frozenset({"evaluations", "best_fitness"})


@dataclass
class RunTrace:
    """A single run's convergence record: a trimmed ``LogData`` + metadata."""

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

    handoff_eval: int | None = None
    """Evaluation count at which an algorithm handoff occurred, if any."""

    handoff_iter: int | None = None
    """CMA-ES generation count at which the handoff occurred. Same event as
    ``handoff_eval`` but expressed in iterations of the warmup algorithm.
    Useful when plots want to annotate handoffs in a population-size-agnostic
    way."""

    series: dict[str, list[float]] = field(default_factory=dict)
    """Additional retained scalar-per-step diagnostics, keyed by field name
    (e.g. ``"sigma"``, ``"condition_number"``, ``"iteration"``). Populated by
    :func:`capture_scalar_series` when a run is trimmed for benchmarking;
    empty for a bare trace. Exposed uniformly via :meth:`get_series` so the
    panel plotter treats a persisted trace and a live ``LogData`` alike."""

    def get_series(self, name: str) -> list[float] | None:
        """Return the per-step array for ``name``, or ``None`` if not present.

        The :class:`~declivity.plotting.unified.RunSeries` contract: the
        plotter asks for a panel's ``field`` / ``x_field`` by name whether
        the source is a full ``LogData`` or a trimmed ``RunTrace``.
        ``evaluations`` and ``best_fitness`` resolve to the first-class
        lists; everything else comes from :attr:`series`, so an un-retained
        field returns ``None`` and the plotter draws an empty series.
        """
        if name == "evaluations":
            return [float(v) for v in self.evaluations]
        if name == "best_fitness":
            return self.best_fitness
        return self.series.get(name)


def capture_scalar_series(
    log_data: Any,
    *,
    retain: tuple[str, ...] | None = None,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, list[float]]:
    """Trim a ``LogData`` to its cheap, aggregatable scalar-per-step fields.

    Returns ``{field_name: [values]}`` for every list-of-scalars field whose
    length matches ``best_fitness``: the per-step scalar diagnostics
    (``sigma``, ``mean_fitness``, ``condition_number``, ``iteration``, ...).
    ``best_fitness`` and ``evaluations`` are skipped, being first-class on
    :class:`RunTrace`, as are the heavy vector/matrix fields in
    :data:`HEAVY_LOGDATA_FIELDS`.

    Args:
        log_data: Any ``LogData`` instance (a dataclass; introspected via
            ``vars``).
        retain: If given, restrict capture to these field names, still
            subject to the scalar/length filter.  ``None`` (default)
            captures every qualifying scalar field.
        exclude: Field names to skip on top of the built-in heavy set.

    The length filter doubles as a diag-flag gate: a field only logged when
    a ``diag_*`` flag is on has length 0 when the flag is off, so it is
    skipped.
    """
    base = getattr(log_data, "best_fitness", None)
    base_len = len(base) if base is not None else 0
    if base_len == 0:
        return {}

    captured: dict[str, list[float]] = {}
    for name, value in vars(log_data).items():
        if name in _TOP_LEVEL_SERIES or name in HEAVY_LOGDATA_FIELDS or name in exclude:
            continue
        if retain is not None and name not in retain:
            continue
        if not isinstance(value, list) or len(value) != base_len:
            continue
        first = value[0]
        if not isinstance(first, (int, float, np.floating, np.integer)):
            continue
        captured[name] = [float(v) for v in value]
    return captured
