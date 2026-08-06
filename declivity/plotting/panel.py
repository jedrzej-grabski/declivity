"""Declarative panel specification for diagnostic plots.

A ``Panel`` describes one diagnostic plot — what to read off a LogData,
how to label it, what scale to use. Panels are registered per algorithm
under semantic keys (e.g. ``"step_size"``), so the same key resolves to
``sigma`` on CMA-ES, ``Ft`` on DES, and ``step_length`` on L-BFGS-B. This
makes cross-algorithm comparison declarative: ask the registry which
keys are common to a set of algorithms, then plot them overlaid.

A panel can be **single-series** (one ``field``) or **multi-series**
(several :class:`Series` overlaid on the same axes — useful for
convergence-style views with best/mean/median together). Multi-series
is honoured by :py:func:`plot_metrics`; :py:func:`plot_comparison` only
uses the first series of each panel because its overlay axis is the
algorithm, not the field.

Example:

    from declivity.algorithms.choices import AlgorithmChoice
    from declivity.plotting.panel import Panel, Series, PanelRegistry

    PanelRegistry.register(
        AlgorithmChoice.CMAES,
        Panel("step_size", "Step Size", "sigma", field="sigma"),
        Panel(
            "convergence_overlay",
            "Convergence",
            "Fitness",
            series=(
                Series("best_fitness",   "Best"),
                Series("mean_fitness",   "Mean f(m)", linestyle="--"),
                Series("median_fitness", "Median",    linestyle=":"),
            ),
        ),
    )
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from declivity.algorithms.choices import AlgorithmChoice
from declivity.plotting.types import LineStyle, PanelKey, YScale


@dataclass(frozen=True)
class Series:
    """One line within a :class:`Panel`.

    Multi-series panels exist so that "convergence" can show
    best/mean/median together (one panel, three lines) instead of
    splitting them into three separate panels.

    Attributes:
        field: Attribute name on the LogData object for the y-values.
        label: Legend label. Empty string defaults to a humanized
            version of ``field``.
        linestyle: A :class:`~declivity.plotting.types.LineStyle` enum (or any
            raw matplotlib linestyle string).
        color: Matplotlib color string. ``None`` falls back to the
            default color cycle. Per-series colors are mainly useful in
            single-algorithm views; in :py:func:`plot_comparison` the
            color is taken from the per-algorithm palette instead.
    """

    field: str
    label: str = ""
    linestyle: LineStyle | str = LineStyle.SOLID
    color: str | None = None

    @property
    def display_label(self) -> str:
        return self.label or self.field.replace("_", " ").title()


@dataclass(frozen=True)
class Panel:
    """Declarative specification for a single diagnostic plot.

    Construct with either ``field=...`` (single-series convenience) or
    ``series=(...)`` (one or more :class:`Series`). Specifying both is
    an error; specifying neither is an error.

    Attributes:
        key: Semantic identifier used for registration and cross-algorithm
            matching. Two algorithms can register different ``field``s
            under the same ``key`` — that is what makes
            :py:meth:`PanelRegistry.common` produce comparable metrics.
        title: Plot title.
        ylabel: Y-axis label.
        field: Single-series convenience. Internally normalized to
            ``series=(Series(field),)``.
        series: Tuple of :class:`Series` for multi-line panels.
        x_field: Attribute name on the LogData object for the x-series.
            Defaults to ``"evaluations"`` (the budget unit shared across
            algorithms).
        yscale: ``"log"`` or ``"linear"``.
        floor: If set, clip y-values below this threshold. Useful with
            ``yscale="log"`` to keep runs that converge to zero finite.
        default: If ``True``, included when :py:func:`plot_metrics` is
            called without an explicit ``panels=`` list. Set ``False``
            for noisy or rarely-useful panels (still available via
            ``panels=[key]`` or ``panels="all"``).
    """

    key: PanelKey | str
    title: str
    ylabel: str
    field: str | None = None
    series: tuple[Series, ...] | None = None
    x_field: str = "evaluations"
    yscale: YScale | str = YScale.LOG
    floor: float | None = None
    default: bool = True

    def __post_init__(self) -> None:
        field_set = self.field is not None
        series_set = self.series is not None and len(self.series) > 0
        if field_set and series_set:
            raise ValueError(
                f"Panel {self.key!r} has both 'field' and 'series' set — pick one."
            )
        if not field_set and not series_set:
            raise ValueError(
                f"Panel {self.key!r} needs either 'field=' or 'series=(Series(...), ...)'."
            )

    @property
    def series_list(self) -> tuple[Series, ...]:
        """Normalized tuple of :class:`Series` — always at least one entry."""
        if self.series is not None and len(self.series) > 0:
            return self.series
        assert self.field is not None  # enforced by __post_init__
        return (Series(self.field),)


class PanelRegistry:
    """Per-algorithm registry of ``Panel`` specs, keyed by semantic key.

    Algorithms register panels by calling :py:meth:`register`. The plotter
    looks them up by ``(algorithm, key)``. :py:meth:`common` returns the
    semantic keys available for every algorithm in a list, which powers
    the "panels comparable across N algorithms" workflow.
    """

    _panels: ClassVar[dict[AlgorithmChoice, dict[str, Panel]]] = {}

    @classmethod
    def register(cls, algorithm: AlgorithmChoice, *panels: Panel) -> None:
        """Register one or more panels under an algorithm.

        Re-registering the same key overrides the previous panel. This is
        intentional so studies can swap in custom panels without needing
        to deregister first.
        """
        bucket = cls._panels.setdefault(algorithm, {})
        for panel in panels:
            # Normalize to plain str so lookups via either PanelKey or
            # raw string land on the same entry.
            bucket[str(panel.key)] = panel

    @classmethod
    def get(cls, algorithm: AlgorithmChoice, key: PanelKey | str) -> Panel:
        """Look up a panel by ``(algorithm, key)``. Raises ``KeyError`` with a
        listing of what *is* available if the key is missing."""
        bucket = cls._panels.get(algorithm)
        if bucket is None:
            raise KeyError(
                f"No panels registered for {algorithm}. "
                f"Registered algorithms: {[a.value for a in cls._panels]}."
            )
        key_str = str(key)
        if key_str not in bucket:
            available = ", ".join(sorted(bucket.keys()))
            raise KeyError(
                f"Panel '{key_str}' not registered for {algorithm}. "
                f"Available: {available}."
            )
        return bucket[key_str]

    @classmethod
    def available(cls, algorithm: AlgorithmChoice) -> list[str]:
        """All semantic keys registered for one algorithm, sorted."""
        return sorted(cls._panels.get(algorithm, {}).keys())

    @classmethod
    def default(cls, algorithm: AlgorithmChoice) -> list[str]:
        """Semantic keys flagged ``default=True`` for one algorithm, sorted.

        ``plot_metrics(result)`` with no explicit ``panels=`` uses this
        subset, keeping the headline plot focused on the panels that
        actually carry signal for that algorithm. Pass ``panels="all"``
        to bypass.
        """
        bucket = cls._panels.get(algorithm, {})
        return sorted(key for key, panel in bucket.items() if panel.default)

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
        return {algo: sorted(panels.keys()) for algo, panels in cls._panels.items()}
