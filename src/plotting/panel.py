"""Declarative panel specification for diagnostic plots.

A ``Panel`` describes one diagnostic plot — what to read off a LogData,
how to label it, what scale to use. Panels are registered per algorithm
under semantic keys (e.g. ``"step_size"``), so the same key resolves to
``sigma`` on CMA-ES, ``Ft`` on DES, and ``step_length`` on L-BFGS-B. This
makes cross-algorithm comparison declarative: ask the registry which
keys are common to a set of algorithms, then plot them overlaid.

Example:

    from src.algorithms.choices import AlgorithmChoice
    from src.plotting.panel import Panel, PanelRegistry

    PanelRegistry.register(
        AlgorithmChoice.CMAES,
        Panel("step_size", "Step Size", "sigma", field="sigma"),
    )

    PanelRegistry.common([AlgorithmChoice.CMAES, AlgorithmChoice.DES])
    # -> ["convergence", "step_size", ...] (intersection)
"""

from dataclasses import dataclass
from typing import Literal, Sequence

from src.algorithms.choices import AlgorithmChoice


@dataclass(frozen=True)
class Panel:
    """Declarative specification for a single diagnostic plot.

    Attributes:
        key: Semantic identifier used for registration and cross-algorithm
            matching. Two algorithms can register different ``field``s under
            the same ``key`` — that's how ``PanelRegistry.common`` finds
            comparable metrics.
        title: Plot title.
        ylabel: Y-axis label.
        field: Attribute name on the LogData object to read as the y-series.
        x_field: Attribute name on the LogData object to read as the x-series.
            Defaults to ``"evaluations"`` since that's the budget unit shared
            across algorithms.
        yscale: ``"log"`` or ``"linear"``.
        floor: If set, clip y-values below this threshold (useful with
            ``yscale="log"`` to avoid ``log(0)`` blowups on runs that
            converge essentially to zero).
    """

    key: str
    title: str
    ylabel: str
    field: str
    x_field: str = "evaluations"
    yscale: Literal["linear", "log"] = "log"
    floor: float | None = None


class PanelRegistry:
    """Per-algorithm registry of ``Panel`` specs, keyed by semantic key.

    Algorithms register panels by calling :py:meth:`register`. The plotter
    looks them up by ``(algorithm, key)``. :py:meth:`common` returns the
    semantic keys available for every algorithm in a list, which is what
    powers the "5 panels comparable across both" workflow.
    """

    _panels: dict[AlgorithmChoice, dict[str, Panel]] = {}

    @classmethod
    def register(cls, algorithm: AlgorithmChoice, *panels: Panel) -> None:
        """Register one or more panels under an algorithm.

        Re-registering the same key overrides the previous panel. This is
        intentional so studies can swap in custom panels without needing
        to deregister first.
        """
        bucket = cls._panels.setdefault(algorithm, {})
        for panel in panels:
            bucket[panel.key] = panel

    @classmethod
    def get(cls, algorithm: AlgorithmChoice, key: str) -> Panel:
        """Look up a panel by ``(algorithm, key)``. Raises ``KeyError`` with a
        listing of what *is* available if the key is missing."""
        bucket = cls._panels.get(algorithm)
        if bucket is None:
            raise KeyError(
                f"No panels registered for {algorithm}. "
                f"Registered algorithms: {[a.value for a in cls._panels]}."
            )
        if key not in bucket:
            available = ", ".join(sorted(bucket.keys()))
            raise KeyError(
                f"Panel '{key}' not registered for {algorithm}. "
                f"Available: {available}."
            )
        return bucket[key]

    @classmethod
    def available(cls, algorithm: AlgorithmChoice) -> list[str]:
        """All semantic keys registered for one algorithm, sorted."""
        return sorted(cls._panels.get(algorithm, {}).keys())

    @classmethod
    def common(cls, algorithms: Sequence[AlgorithmChoice]) -> list[str]:
        """Keys registered for *every* algorithm in the list.

        With an empty input, returns ``[]``. Order is deterministic (sorted).
        """
        algos = list(algorithms)
        if not algos:
            return []
        keys: set[str] = set(cls._panels.get(algos[0], {}).keys())
        for algo in algos[1:]:
            keys &= set(cls._panels.get(algo, {}).keys())
        return sorted(keys)

    @classmethod
    def all_registered(cls) -> dict[AlgorithmChoice, list[str]]:
        """Snapshot of every (algorithm -> keys) mapping. Useful for ``--help``-style
        listings or debugging missing registrations."""
        return {
            algo: sorted(panels.keys())
            for algo, panels in cls._panels.items()
        }
