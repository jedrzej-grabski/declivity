"""
Modular stopping-condition abstractions.

A :class:`StoppingCondition` decides *when an optimizer's main loop should
halt*, independently of the algorithm's own internal convergence tests
(CMA-ES ``tolfun`` / ``tolx``, L-BFGS-B ``pgtol`` / ``factr``, ...).  It is
the shared, cross-algorithm termination signal that used to live as a bare
``budget: int`` field on the configuration dataclass.

Like :class:`~declivity.utils.constraint_handlers.ConstraintHandler` and
:class:`~declivity.utils.repair_strategies.RepairStrategy`, a
``StoppingCondition`` is *injected into the optimizer at construction time*
and defaults to a sensible built-in — an evaluation budget of
``10_000 * dimensions`` (see :func:`default_stopping_condition`), which
reproduces the framework's historical default exactly.

Contract
--------
The optimizer feeds each condition an :class:`OptimizationState` snapshot
(evaluations, iterations, best fitness, elapsed wall-clock) every time the
main loop tests for termination; the condition answers :meth:`should_stop`
and exposes a :attr:`message` describing why it fired.  *Stateful*
conditions (time, stagnation) additionally override :meth:`reset`, which
the optimizer calls once at the start of every ``optimize()`` so a single
instance can be reused across runs.

Composition
-----------
Conditions compose with ``|`` (OR — :class:`AnyStoppingCondition`) and
``&`` (AND — :class:`AllStoppingCondition`)::

    # stop after 5000 evals, OR 30 seconds, OR reaching f <= 1e-8
    condition = MaxEvaluations(5000) | MaxTime(30.0) | TargetFitness(1e-8)

Evaluation cap
--------------
Evaluation-budget conditions additionally expose
:meth:`~StoppingCondition.remaining_evaluations`, which lets
``BaseOptimizer.evaluate_population`` trim a generation that would otherwise
overshoot a hard evaluation cap.  A condition that does not bound
evaluations (time, target fitness, stagnation) returns ``None`` — evaluate
the whole generation, then stop at the next loop test.

Discoverability
---------------
:class:`StoppingConditionType` lists the built-ins with a ``.build(...)``
factory, mirroring ``ConstraintHandlerType`` / ``RepairStrategyType``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import final, override

__all__ = [
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
    "DEFAULT_EVALUATIONS_PER_DIMENSION",
]


DEFAULT_EVALUATIONS_PER_DIMENSION = 10_000
"""Evaluations-per-dimension used by :func:`default_stopping_condition`;
reproduces the framework's historical ``10_000 * d`` budget default."""


class OptimizationState:
    """Immutable snapshot of optimizer progress handed to a condition.

    The optimizer builds one of these each time it tests for termination.
    All four fields are cheap scalars every algorithm already tracks, so a
    condition can key off any of them without reaching into
    algorithm-specific internals.
    """

    __slots__ = ("evaluations", "iterations", "best_fitness", "elapsed_seconds")

    def __init__(
        self,
        evaluations: int,
        iterations: int,
        best_fitness: float,
        elapsed_seconds: float,
    ) -> None:
        self.evaluations = evaluations
        """Function evaluations consumed so far."""
        self.iterations = iterations
        """Main-loop iterations (generations) completed so far."""
        self.best_fitness = best_fitness
        """Best objective value found so far (``inf`` before the first eval)."""
        self.elapsed_seconds = elapsed_seconds
        """Wall-clock seconds since the run started (``time.monotonic``)."""

    def __repr__(self) -> str:
        return (
            f"OptimizationState(evaluations={self.evaluations}, "
            f"iterations={self.iterations}, best_fitness={self.best_fitness!r}, "
            f"elapsed_seconds={self.elapsed_seconds!r})"
        )


class StoppingCondition(ABC):
    """Abstract base class for optimizer termination conditions.

    A ``StoppingCondition`` is a predicate over :class:`OptimizationState`.
    Concrete conditions override :meth:`should_stop` and :attr:`message`;
    *stateful* conditions (time, stagnation) also override :meth:`reset`
    to clear per-run bookkeeping — the optimizer calls :meth:`reset` once
    at the start of every ``optimize()`` so one instance can be reused
    across runs.

    Conditions compose with ``|`` (:class:`AnyStoppingCondition` — stop
    when *any* fires) and ``&`` (:class:`AllStoppingCondition` — stop only
    when *all* fire).
    """

    def reset(self) -> None:
        """Clear any per-run internal state.

        Called once at the start of each ``optimize()`` run.  The default
        is a no-op; stateless conditions (:class:`MaxEvaluations`,
        :class:`MaxIterations`, :class:`TargetFitness`) need not override.
        """

    @abstractmethod
    def should_stop(self, state: OptimizationState) -> bool:
        """Return ``True`` iff the optimizer should halt given *state*."""
        ...

    @property
    @abstractmethod
    def message(self) -> str:
        """Human-readable reason, used as ``OptimizationResult.message``
        when this condition ends the run."""
        ...

    def remaining_evaluations(self, evaluations: int) -> int | None:
        """Evaluations still permitted, or ``None`` for no evaluation cap.

        Only conditions that bound *evaluations* return a number;
        everything else returns ``None``.  Consumed by
        :meth:`BaseOptimizer.evaluate_population` to trim a generation that
        would otherwise overshoot a hard budget.  Deliberately takes only
        the evaluation count (not a full :class:`OptimizationState`) so it
        cannot accidentally depend on the iteration / time fields, which
        are stale at population-evaluation time.
        """
        return None

    def __or__(self, other: StoppingCondition) -> StoppingCondition:
        return AnyStoppingCondition(self, other)

    def __and__(self, other: StoppingCondition) -> StoppingCondition:
        return AllStoppingCondition(self, other)


@final
class MaxEvaluations(StoppingCondition):
    """Stop after a fixed number of function evaluations.

    The framework default and historical behaviour (the old
    ``config.budget``).  The message is kept verbatim —
    ``"Maximum function evaluations reached."`` — because the interleaved
    handoff distinguishes a budget stop from genuine convergence by
    matching on that prefix.
    """

    def __init__(self, max_evaluations: int) -> None:
        if max_evaluations <= 0:
            raise ValueError("max_evaluations must be a positive integer.")
        self.max_evaluations = int(max_evaluations)

    @override
    def should_stop(self, state: OptimizationState) -> bool:
        return state.evaluations >= self.max_evaluations

    @override
    def remaining_evaluations(self, evaluations: int) -> int | None:
        return max(0, self.max_evaluations - evaluations)

    @property
    @override
    def message(self) -> str:
        return "Maximum function evaluations reached."

    def __repr__(self) -> str:
        return f"MaxEvaluations({self.max_evaluations})"


@final
class MaxIterations(StoppingCondition):
    """Stop after a fixed number of main-loop iterations (generations)."""

    def __init__(self, max_iterations: int) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        self.max_iterations = int(max_iterations)

    @override
    def should_stop(self, state: OptimizationState) -> bool:
        return state.iterations >= self.max_iterations

    @property
    @override
    def message(self) -> str:
        return "Maximum iterations reached."

    def __repr__(self) -> str:
        return f"MaxIterations({self.max_iterations})"


@final
class MaxTime(StoppingCondition):
    """Stop once wall-clock elapsed time reaches *max_seconds*.

    Reads the ``elapsed_seconds`` the optimizer measures with
    ``time.monotonic()``; the condition itself stays clock-free and is
    therefore deterministically testable.
    """

    def __init__(self, max_seconds: float) -> None:
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive.")
        self.max_seconds = float(max_seconds)

    @override
    def should_stop(self, state: OptimizationState) -> bool:
        return state.elapsed_seconds >= self.max_seconds

    @property
    @override
    def message(self) -> str:
        return f"Time limit reached ({self.max_seconds:g}s)."

    def __repr__(self) -> str:
        return f"MaxTime({self.max_seconds!r})"


@final
class TargetFitness(StoppingCondition):
    """Stop once the best fitness reaches a target quality.

    Fires when ``best_fitness <= target + atol``.  For minimisation the
    target is an upper bound on the acceptable objective value — e.g. the
    known global optimum plus a tolerance.
    """

    def __init__(self, target: float, atol: float = 0.0) -> None:
        self.target = float(target)
        self.atol = float(atol)

    @override
    def should_stop(self, state: OptimizationState) -> bool:
        return state.best_fitness <= self.target + self.atol

    @property
    @override
    def message(self) -> str:
        return f"Target fitness reached (<= {self.target + self.atol:g})."

    def __repr__(self) -> str:
        return f"TargetFitness(target={self.target!r}, atol={self.atol!r})"


class StagnationUnit(Enum):
    """Unit in which :class:`Stagnation` measures its patience window."""

    ITERATIONS = "iterations"
    EVALUATIONS = "evaluations"


@final
class Stagnation(StoppingCondition):
    """Stop when the best fitness has not improved for a patience window.

    "Improvement" means a strict decrease of more than *tol* below the
    best value recorded at the previous improvement.  *patience* is
    measured in either iterations or evaluations (*unit*).

    Stateful: the first observation establishes the baseline (it never
    fires immediately), and :meth:`reset` clears the history between runs.
    """

    def __init__(
        self,
        patience: int,
        tol: float = 1e-12,
        unit: StagnationUnit = StagnationUnit.ITERATIONS,
    ) -> None:
        if patience <= 0:
            raise ValueError("patience must be a positive integer.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")
        self.patience = int(patience)
        self.tol = float(tol)
        self.unit = unit
        self._best: float = math.inf
        self._last_improvement_index: int | None = None

    @override
    def reset(self) -> None:
        self._best = math.inf
        self._last_improvement_index = None

    def _index(self, state: OptimizationState) -> int:
        if self.unit is StagnationUnit.ITERATIONS:
            return state.iterations
        return state.evaluations

    @override
    def should_stop(self, state: OptimizationState) -> bool:
        index = self._index(state)
        if state.best_fitness < self._best - self.tol:
            self._best = state.best_fitness
            self._last_improvement_index = index
            return False
        if self._last_improvement_index is None:
            # First observation with no prior baseline — start the clock.
            self._last_improvement_index = index
            return False
        return (index - self._last_improvement_index) >= self.patience

    @property
    @override
    def message(self) -> str:
        return f"No improvement for {self.patience} {self.unit.value}."

    def __repr__(self) -> str:
        return (
            f"Stagnation(patience={self.patience}, tol={self.tol!r}, "
            f"unit={self.unit})"
        )


@final
class AnyStoppingCondition(StoppingCondition):
    """Composite that stops as soon as *any* child condition fires (OR).

    Built by the ``|`` operator, or directly from two-or-more children.
    :attr:`message` reports the child that fired; the evaluation cap is
    the tightest (smallest) among children that impose one.  Every child's
    :meth:`should_stop` is called on each test so stateful children stay
    current regardless of ordering.
    """

    def __init__(self, *conditions: StoppingCondition) -> None:
        flat: list[StoppingCondition] = []
        for condition in conditions:
            # Flatten nested OR-composites so ``a | b | c`` is one node.
            if isinstance(condition, AnyStoppingCondition):
                flat.extend(condition.conditions)
            else:
                flat.append(condition)
        if len(flat) < 2:
            raise ValueError("AnyStoppingCondition needs at least two conditions.")
        self.conditions: tuple[StoppingCondition, ...] = tuple(flat)
        self._fired: StoppingCondition | None = None

    @override
    def reset(self) -> None:
        self._fired = None
        for condition in self.conditions:
            condition.reset()

    @override
    def should_stop(self, state: OptimizationState) -> bool:
        fired: StoppingCondition | None = None
        for condition in self.conditions:
            # Evaluate every child (stateful ones must observe each state);
            # remember the first that fired for the message.
            if condition.should_stop(state) and fired is None:
                fired = condition
        self._fired = fired
        return fired is not None

    @override
    def remaining_evaluations(self, evaluations: int) -> int | None:
        caps = [
            cap
            for condition in self.conditions
            if (cap := condition.remaining_evaluations(evaluations)) is not None
        ]
        return min(caps) if caps else None

    @property
    @override
    def message(self) -> str:
        if self._fired is not None:
            return self._fired.message
        return "Stopping condition met."

    def __repr__(self) -> str:
        return " | ".join(repr(condition) for condition in self.conditions)


@final
class AllStoppingCondition(StoppingCondition):
    """Composite that stops only when *all* child conditions fire (AND).

    Built by the ``&`` operator.  Imposes no per-generation evaluation cap
    (:meth:`remaining_evaluations` is ``None``): with AND semantics the run
    continues until every child agrees, so trimming a generation to one
    child's budget would end the run early.
    """

    def __init__(self, *conditions: StoppingCondition) -> None:
        flat: list[StoppingCondition] = []
        for condition in conditions:
            if isinstance(condition, AllStoppingCondition):
                flat.extend(condition.conditions)
            else:
                flat.append(condition)
        if len(flat) < 2:
            raise ValueError("AllStoppingCondition needs at least two conditions.")
        self.conditions: tuple[StoppingCondition, ...] = tuple(flat)

    @override
    def reset(self) -> None:
        for condition in self.conditions:
            condition.reset()

    @override
    def should_stop(self, state: OptimizationState) -> bool:
        # Evaluate every child so stateful ones observe each state.
        results = [condition.should_stop(state) for condition in self.conditions]
        return all(results)

    @property
    @override
    def message(self) -> str:
        return "All conditions met: " + "; ".join(
            condition.message for condition in self.conditions
        )

    def __repr__(self) -> str:
        return " & ".join(repr(condition) for condition in self.conditions)


def default_stopping_condition(dimensions: int) -> StoppingCondition:
    """The framework default: ``MaxEvaluations(10_000 * dimensions)``.

    Reproduces the historical ``default_budget(d)`` every optimizer used
    before stopping conditions were extracted from the config.
    """
    return MaxEvaluations(DEFAULT_EVALUATIONS_PER_DIMENSION * dimensions)


class StoppingConditionType(Enum):
    """Discoverability enum for built-in stopping conditions.

    Call ``.build(**kwargs)`` to obtain an instance without importing the
    concrete classes directly::

        cond = StoppingConditionType.EVALUATIONS.build(max_evaluations=5000)
        cond = StoppingConditionType.TIME.build(max_seconds=30.0)
        cond = StoppingConditionType.TARGET_FITNESS.build(target=1e-8)
        cond = StoppingConditionType.STAGNATION.build(patience=50)
    """

    EVALUATIONS = "evaluations"
    ITERATIONS = "iterations"
    TIME = "time"
    TARGET_FITNESS = "target_fitness"
    STAGNATION = "stagnation"

    def build(self, **kwargs: object) -> StoppingCondition:
        """Construct the matching :class:`StoppingCondition`.

        Keyword arguments are forwarded to the concrete constructor:

        - ``EVALUATIONS`` → ``max_evaluations``
        - ``ITERATIONS`` → ``max_iterations``
        - ``TIME`` → ``max_seconds``
        - ``TARGET_FITNESS`` → ``target``, optional ``atol``
        - ``STAGNATION`` → ``patience``, optional ``tol``, ``unit``
        """
        if self is StoppingConditionType.EVALUATIONS:
            return MaxEvaluations(**kwargs)  # type: ignore[arg-type]
        if self is StoppingConditionType.ITERATIONS:
            return MaxIterations(**kwargs)  # type: ignore[arg-type]
        if self is StoppingConditionType.TIME:
            return MaxTime(**kwargs)  # type: ignore[arg-type]
        if self is StoppingConditionType.TARGET_FITNESS:
            return TargetFitness(**kwargs)  # type: ignore[arg-type]
        if self is StoppingConditionType.STAGNATION:
            return Stagnation(**kwargs)  # type: ignore[arg-type]
        # Exhaustive match — new members must extend this method.
        raise NotImplementedError(f"No build() implementation for {self!r}")
