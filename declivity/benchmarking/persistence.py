"""Save and load RunTraces.

After a Benchmark.run() the traces live in memory only. Persisting them
to disk lets a re-plot or post-hoc statistical analysis skip re-running
the optimizers.

- ``traces.json``  : structured dump of every trace (lists in, lists out).
- ``summary.csv``  : one row per (problem, algorithm) with median/best/etc.
- ``runs.csv``     : one row per run with final fitness, eval count,
                     handoff eval, seed. Easier for spreadsheets.
"""

import csv
import json
from pathlib import Path
from typing import Iterable, Union

from declivity.benchmarking.run_trace import RunTrace


def _trace_to_dict(trace: RunTrace) -> dict:
    payload = {
        "algorithm": trace.algorithm,
        "problem": trace.problem,
        "seed": trace.seed,
        "evaluations": list(trace.evaluations),
        "best_fitness": list(trace.best_fitness),
        "final_evaluations": trace.final_evaluations,
        "final_fitness": trace.final_fitness,
        "handoff_eval": trace.handoff_eval,
        "handoff_iter": trace.handoff_iter,
    }
    # Retained scalar-per-step diagnostics (sigma, condition_number, ...).
    # Omitted from the JSON when empty so bare traces stay compact and old
    # readers are unaffected.
    if trace.series:
        payload["series"] = {
            name: [float(v) for v in values] for name, values in trace.series.items()
        }
    return payload


def _trace_from_dict(payload: dict) -> RunTrace:
    return RunTrace(
        algorithm=payload["algorithm"],
        problem=payload["problem"],
        seed=int(payload["seed"]),
        evaluations=[int(x) for x in payload["evaluations"]],
        best_fitness=[float(x) for x in payload["best_fitness"]],
        final_evaluations=int(payload["final_evaluations"]),
        final_fitness=float(payload["final_fitness"]),
        handoff_eval=(
            int(payload["handoff_eval"]) if payload["handoff_eval"] is not None else None
        ),
        handoff_iter=(
            int(payload["handoff_iter"]) if payload.get("handoff_iter") is not None else None
        ),
        # Backward-compatible: traces.json written before retained series
        # existed simply have no "series" key.
        series={
            name: [float(v) for v in values]
            for name, values in payload.get("series", {}).items()
        },
    )


def save_traces_json(
    traces: dict[tuple[str, str], list[RunTrace]],
    path: Union[str, Path],
) -> Path:
    """Write the full trace dictionary as a single JSON file."""
    path = Path(path)
    payload = {
        "runs": [
            _trace_to_dict(trace)
            for run_list in traces.values()
            for trace in run_list
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_traces_json(path: Union[str, Path]) -> dict[tuple[str, str], list[RunTrace]]:
    """Reconstruct the trace dictionary from a previously saved JSON file."""
    payload = json.loads(Path(path).read_text())
    traces: dict[tuple[str, str], list[RunTrace]] = {}
    for run in payload["runs"]:
        trace = _trace_from_dict(run)
        traces.setdefault((trace.problem, trace.algorithm), []).append(trace)
    return traces


def save_runs_csv(
    traces: dict[tuple[str, str], list[RunTrace]],
    path: Union[str, Path],
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


def save_summary_csv(rows: Iterable[dict], path: Union[str, Path]) -> Path:
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
