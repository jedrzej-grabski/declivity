"""Save and load RunTraces.

After a Benchmark.run() the traces live in memory only. Persisting them
to disk lets a re-plot or post-hoc statistical analysis skip re-running
the optimizers.

- ``traces.parquet``: long/tidy columnar dump, one row per (run, step).
                     Floats are stored as real floats (not ASCII text) and
                     zstd-compressed, which matters at scale: an
                     uncompressed JSON dump of a d=30 aggregate can run
                     ~500MB+ against a tight cluster home-directory quota.
- ``summary.csv``  : one row per (problem, algorithm) with median/best/etc.
- ``runs.csv``     : one row per run with final fitness, eval count,
                     handoff eval, seed. Easier for spreadsheets.
"""

import csv
import functools
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from declivity.benchmarking.run_trace import RunTrace


def save_traces_parquet(
    traces: dict[tuple[str, str], list[RunTrace]],
    path: str | Path,
) -> Path:
    """Write the full trace dictionary as a single long/tidy Parquet file.

    One row per (run, step). Per-run scalars (``final_fitness``,
    ``handoff_eval``, ...) are repeated on every step row -- parquet
    RLE-encodes the repetition cheaply. ``series`` diagnostics get one
    ``series_<key>`` column per distinct key encountered across every trace
    in this call; a run missing a given key has that column null throughout
    its rows.
    """
    path = Path(path)
    run_list = [trace for runs in traces.values() for trace in runs]

    series_keys: list[str] = []
    seen = set()
    for trace in run_list:
        for key in trace.series:
            if key not in seen:
                seen.add(key)
                series_keys.append(key)

    columns: dict[str, list] = {
        "problem": [],
        "algorithm": [],
        "seed": [],
        "step": [],
        "evaluations": [],
        "best_fitness": [],
        "final_evaluations": [],
        "final_fitness": [],
        "handoff_eval": [],
        "handoff_iter": [],
        **{f"series_{key}": [] for key in series_keys},
    }

    for trace in run_list:
        n = len(trace.evaluations)
        columns["problem"].extend([trace.problem] * n)
        columns["algorithm"].extend([trace.algorithm] * n)
        columns["seed"].extend([trace.seed] * n)
        columns["step"].extend(range(n))
        columns["evaluations"].extend(int(v) for v in trace.evaluations)
        columns["best_fitness"].extend(float(v) for v in trace.best_fitness)
        columns["final_evaluations"].extend([trace.final_evaluations] * n)
        columns["final_fitness"].extend([trace.final_fitness] * n)
        columns["handoff_eval"].extend([trace.handoff_eval] * n)
        columns["handoff_iter"].extend([trace.handoff_iter] * n)
        for key in series_keys:
            values = trace.series.get(key)
            columns[f"series_{key}"].extend(
                [None] * n if values is None else [float(v) for v in values]
            )

    schema_fields = [
        pa.field("problem", pa.string()),
        pa.field("algorithm", pa.string()),
        pa.field("seed", pa.int64()),
        pa.field("step", pa.int64()),
        pa.field("evaluations", pa.int64()),
        pa.field("best_fitness", pa.float64()),
        pa.field("final_evaluations", pa.int64()),
        pa.field("final_fitness", pa.float64()),
        pa.field("handoff_eval", pa.int64()),
        pa.field("handoff_iter", pa.int64()),
        *(pa.field(f"series_{key}", pa.float64()) for key in series_keys),
    ]
    table = pa.table(columns, schema=pa.schema(schema_fields))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    pq.write_table(table, tmp_path, compression="zstd")
    os.replace(tmp_path, path)
    return path


@functools.cache
def load_traces_parquet(path: str | Path) -> dict[tuple[str, str], list[RunTrace]]:
    """Reconstruct the trace dictionary from a previously saved Parquet file.

    Cached on ``path``: callers that re-read the same file (e.g. the
    marimo visualizer re-rendering on every UI interaction) skip the disk
    round trip on repeat calls. Pass a consistent path representation
    (``str`` or ``Path``, not a mix) so cache hits land.
    """
    table = pq.read_table(path)
    series_columns = [name for name in table.column_names if name.startswith("series_")]

    groups: dict[tuple[str, str, int], list[int]] = {}
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    for row_index, (problem, algorithm, seed) in enumerate(
        zip(columns["problem"], columns["algorithm"], columns["seed"], strict=True)
    ):
        groups.setdefault((problem, algorithm, seed), []).append(row_index)

    traces: dict[tuple[str, str], list[RunTrace]] = {}
    for (problem, algorithm, seed), row_indices in groups.items():
        row_indices.sort(key=lambda i: columns["step"][i])
        first = row_indices[0]

        handoff_eval = columns["handoff_eval"][first]
        handoff_iter = columns["handoff_iter"][first]

        series: dict[str, list[float]] = {}
        for series_column in series_columns:
            values = [columns[series_column][i] for i in row_indices]
            if any(v is not None for v in values):
                series[series_column[len("series_") :]] = [float(v) for v in values]

        trace = RunTrace(
            algorithm=algorithm,
            problem=problem,
            seed=int(seed),
            evaluations=[int(columns["evaluations"][i]) for i in row_indices],
            best_fitness=[float(columns["best_fitness"][i]) for i in row_indices],
            final_evaluations=int(columns["final_evaluations"][first]),
            final_fitness=float(columns["final_fitness"][first]),
            handoff_eval=None if handoff_eval is None else int(handoff_eval),
            handoff_iter=None if handoff_iter is None else int(handoff_iter),
            series=series,
        )
        traces.setdefault((trace.problem, trace.algorithm), []).append(trace)
    return traces


def save_runs_csv(
    traces: dict[tuple[str, str], list[RunTrace]],
    path: str | Path,
) -> Path:
    """Write one CSV row per run (good for spreadsheet inspection)."""
    path = Path(path)
    rows = [
        {
            "problem": trace.problem,
            "algorithm": trace.algorithm,
            "seed": trace.seed,
            "final_fitness": trace.final_fitness,
            "final_evaluations": trace.final_evaluations,
            "handoff_eval": "" if trace.handoff_eval is None else trace.handoff_eval,
        }
        for run_list in traces.values()
        for trace in run_list
    ]
    with path.open("w", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            fh.write("")
    return path


def _nested_pa_array(values: list) -> pa.Array:
    """Convert a Python list of per-row values (each a scalar, 1D, or 2D
    array) into a Parquet column, nesting as ``list``/``list<list>`` as
    needed. Every row must share the same rank."""
    sample = np.asarray(values[0])
    if sample.ndim == 0:
        return pa.array([float(v) for v in values], type=pa.float64())
    if sample.ndim == 1:
        return pa.array(
            [np.asarray(v, dtype=float).tolist() for v in values],
            type=pa.list_(pa.float64()),
        )
    if sample.ndim == 2:
        return pa.array(
            [np.asarray(v, dtype=float).tolist() for v in values],
            type=pa.list_(pa.list_(pa.float64())),
        )
    raise ValueError(f"unsupported array rank {sample.ndim} (expected 0, 1, or 2)")


def save_arrays_parquet(path: str | Path, columns: Mapping[str, list]) -> Path:
    """Write named columns -- each a list of per-row scalars/1D/2D arrays --
    as a single Parquet file, atomically.

    Used as a zstd-compressed replacement for ``np.savez_compressed`` where
    the payload is a handful of named arrays rather than tidy per-step rows:
    a matrix-valued field becomes a nested ``list<list<float64>>`` column.
    """
    path = Path(path)
    table = pa.table(
        {name: _nested_pa_array(values) for name, values in columns.items()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    pq.write_table(table, tmp_path, compression="zstd")
    os.replace(tmp_path, path)
    return path


def load_arrays_parquet(path: str | Path) -> dict[str, list[NDArray[np.float64]]]:
    """Reconstruct the column dict written by :func:`save_arrays_parquet`.

    Each column comes back as a list (one entry per row) of ``float64``
    arrays, with rank (0D/1D/2D) restored from the column's nesting depth.
    """
    table = pq.read_table(path)
    return {
        name: [np.asarray(v, dtype=np.float64) for v in table.column(name).to_pylist()]
        for name in table.column_names
    }


def save_summary_csv(rows: Iterable[dict], path: str | Path) -> Path:
    """Write the aggregate summary table (one row per problem x algorithm)."""
    path = Path(path)
    rows = list(rows)
    with path.open("w", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            fh.write("")
    return path
