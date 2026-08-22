"""Round-trip tests for RunTrace persistence via Parquet.

``save_traces_parquet``/``load_traces_parquet`` replaced the old JSON-based
persistence entirely (no backward compatibility). These tests pin down the
round-trip contract: a batch of RunTraces saved and reloaded must reproduce
the same values, types, and grouping the callers rely on.
"""

from pathlib import Path

import pytest

from declivity.benchmarking.persistence import (
    load_traces_parquet,
    save_traces_parquet,
)
from declivity.benchmarking.run_trace import RunTrace


def _traces_dict(
    traces: list[RunTrace],
) -> dict[tuple[str, str], list[RunTrace]]:
    grouped: dict[tuple[str, str], list[RunTrace]] = {}
    for trace in traces:
        grouped.setdefault((trace.problem, trace.algorithm), []).append(trace)
    return grouped


def _assert_traces_equal(a: RunTrace, b: RunTrace) -> None:
    assert a.algorithm == b.algorithm
    assert a.problem == b.problem
    assert a.seed == b.seed
    assert a.evaluations == b.evaluations
    assert a.best_fitness == b.best_fitness
    assert a.final_evaluations == b.final_evaluations
    assert a.final_fitness == b.final_fitness
    assert a.handoff_eval == b.handoff_eval
    assert a.handoff_iter == b.handoff_iter
    assert a.series.keys() == b.series.keys()
    for key in a.series:
        assert a.series[key] == b.series[key]

    # Python-native types, not numpy scalars: downstream code JSON-serializes
    # and does exact-equality checks against these values.
    assert isinstance(b.seed, int)
    assert all(isinstance(v, int) for v in b.evaluations)
    assert all(isinstance(v, float) for v in b.best_fitness)
    assert isinstance(b.final_evaluations, int)
    assert isinstance(b.final_fitness, float)
    if b.handoff_eval is not None:
        assert isinstance(b.handoff_eval, int)
    if b.handoff_iter is not None:
        assert isinstance(b.handoff_iter, int)
    for values in b.series.values():
        assert all(isinstance(v, float) for v in values)


def _round_trip(
    traces: list[RunTrace], path: Path
) -> dict[tuple[str, str], list[RunTrace]]:
    save_traces_parquet(_traces_dict(traces), path)
    load_traces_parquet.cache_clear()
    return load_traces_parquet(path)


def test_round_trip_overlapping_and_nonoverlapping_series_keys(tmp_path: Path):
    trace_a = RunTrace(
        algorithm="CMA-ES",
        problem="Rastrigin-10D",
        seed=0,
        evaluations=[100, 200, 300],
        best_fitness=[10.0, 5.0, 1.0],
        final_evaluations=300,
        final_fitness=1.0,
        series={"sigma": [1.0, 0.9, 0.8], "condition_number": [2.0, 2.1, 2.2]},
    )
    trace_b = RunTrace(
        algorithm="L-BFGS-B",
        problem="Rastrigin-10D",
        seed=0,
        evaluations=[50, 100, 150],
        best_fitness=[8.0, 4.0, 0.5],
        final_evaluations=150,
        final_fitness=0.5,
        series={"function_value": [8.0, 4.0, 0.5]},
    )

    path = tmp_path / "traces.parquet"
    reloaded = _round_trip([trace_a, trace_b], path)

    assert set(reloaded.keys()) == {
        ("Rastrigin-10D", "CMA-ES"),
        ("Rastrigin-10D", "L-BFGS-B"),
    }
    (reloaded_a,) = reloaded[("Rastrigin-10D", "CMA-ES")]
    (reloaded_b,) = reloaded[("Rastrigin-10D", "L-BFGS-B")]
    _assert_traces_equal(trace_a, reloaded_a)
    _assert_traces_equal(trace_b, reloaded_b)
    # Neither run's series dict picked up the other's keys.
    assert set(reloaded_a.series) == {"sigma", "condition_number"}
    assert set(reloaded_b.series) == {"function_value"}


def test_round_trip_empty_series(tmp_path: Path):
    trace = RunTrace(
        algorithm="Powell",
        problem="Rosenbrock-5D",
        seed=7,
        evaluations=[10, 20],
        best_fitness=[3.0, 1.0],
        final_evaluations=20,
        final_fitness=1.0,
        series={},
    )
    path = tmp_path / "traces.parquet"
    reloaded = _round_trip([trace], path)
    (reloaded_trace,) = reloaded[("Rosenbrock-5D", "Powell")]
    _assert_traces_equal(trace, reloaded_trace)
    assert reloaded_trace.series == {}


def test_round_trip_handoff_none_and_set_in_same_batch(tmp_path: Path):
    trace_no_handoff = RunTrace(
        algorithm="CMA-ES",
        problem="Ellipsoid-20D",
        seed=1,
        evaluations=[100],
        best_fitness=[1.0],
        final_evaluations=100,
        final_fitness=1.0,
        handoff_eval=None,
        handoff_iter=None,
    )
    trace_with_handoff = RunTrace(
        algorithm="CMA-ES -> L-BFGS-B",
        problem="Ellipsoid-20D",
        seed=1,
        evaluations=[100, 200],
        best_fitness=[1.0, 0.1],
        final_evaluations=200,
        final_fitness=0.1,
        handoff_eval=100,
        handoff_iter=15,
    )
    path = tmp_path / "traces.parquet"
    reloaded = _round_trip([trace_no_handoff, trace_with_handoff], path)

    (reloaded_no_handoff,) = reloaded[("Ellipsoid-20D", "CMA-ES")]
    (reloaded_with_handoff,) = reloaded[("Ellipsoid-20D", "CMA-ES -> L-BFGS-B")]
    assert reloaded_no_handoff.handoff_eval is None
    assert reloaded_no_handoff.handoff_iter is None
    assert reloaded_with_handoff.handoff_eval == 100
    assert reloaded_with_handoff.handoff_iter == 15
    _assert_traces_equal(trace_no_handoff, reloaded_no_handoff)
    _assert_traces_equal(trace_with_handoff, reloaded_with_handoff)


def test_round_trip_ragged_lengths_across_runs(tmp_path: Path):
    long_trace = RunTrace(
        algorithm="CMA-ES",
        problem="Ellipsoid-30D",
        seed=2,
        evaluations=list(range(0, 5000, 100)),
        best_fitness=[1.0 / (i + 1) for i in range(50)],
        final_evaluations=4900,
        final_fitness=1.0 / 50,
        series={"sigma": [1.0 - i * 0.01 for i in range(50)]},
    )
    short_trace = RunTrace(
        algorithm="BFGS",
        problem="Ellipsoid-30D",
        seed=2,
        evaluations=[10, 20, 30],
        best_fitness=[5.0, 2.0, 0.5],
        final_evaluations=30,
        final_fitness=0.5,
    )
    path = tmp_path / "traces.parquet"
    reloaded = _round_trip([long_trace, short_trace], path)

    (reloaded_long,) = reloaded[("Ellipsoid-30D", "CMA-ES")]
    (reloaded_short,) = reloaded[("Ellipsoid-30D", "BFGS")]
    assert len(reloaded_long.evaluations) == 50
    assert len(reloaded_short.evaluations) == 3
    _assert_traces_equal(long_trace, reloaded_long)
    _assert_traces_equal(short_trace, reloaded_short)
    # The short run's step rows must be null (not e.g. 0.0) in series_sigma,
    # so it correctly comes back with an empty series dict.
    assert reloaded_short.series == {}


def test_round_trip_latex_ish_algorithm_names(tmp_path: Path):
    trace = RunTrace(
        algorithm="L-BFGS-B | $C_{20}$",
        problem="CEC17-F1-d20",
        seed=3,
        evaluations=[1, 2, 3],
        best_fitness=[3.0, 2.0, 1.0],
        final_evaluations=3,
        final_fitness=1.0,
    )
    path = tmp_path / "traces.parquet"
    reloaded = _round_trip([trace], path)
    keys = list(reloaded.keys())
    assert keys == [("CEC17-F1-d20", "L-BFGS-B | $C_{20}$")]
    (reloaded_trace,) = reloaded[keys[0]]
    assert reloaded_trace.algorithm == "L-BFGS-B | $C_{20}$"
    _assert_traces_equal(trace, reloaded_trace)


def test_load_traces_parquet_is_cached(tmp_path: Path):
    trace = RunTrace(
        algorithm="CMA-ES",
        problem="P",
        seed=0,
        evaluations=[1],
        best_fitness=[1.0],
        final_evaluations=1,
        final_fitness=1.0,
    )
    path = tmp_path / "traces.parquet"
    save_traces_parquet(_traces_dict([trace]), path)
    load_traces_parquet.cache_clear()
    first = load_traces_parquet(path)
    second = load_traces_parquet(path)
    assert first is second


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
