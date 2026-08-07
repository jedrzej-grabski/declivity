"""Benchmark orchestrator.

Given a list of problems, a list of algorithm specs, and a list of seeds,
runs every (problem, algorithm, seed) combination and stores the
RunTraces. Produces convergence plots (median + IQR across seeds) and
final-fitness distribution summaries.

Set ``num_workers > 1`` to parallelize across (problem, algorithm, seed)
triples via joblib (which uses cloudpickle and so handles lambdas).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from declivity.benchmarking.algorithm_run import AlgorithmRun
from declivity.benchmarking.persistence import (
    save_runs_csv,
    save_summary_csv,
    save_traces_json,
)
from declivity.benchmarking.problem import Problem
from declivity.benchmarking.run_trace import RunTrace


def _execute_one(
    algorithm: AlgorithmRun,
    problem: Problem,
    x0,
    seed: int,
) -> tuple[str, str, RunTrace]:
    """Worker entry point; returned tuple is keyed for re-assembly."""
    trace = algorithm.run(problem, x0, seed)
    return problem.name, algorithm.name, trace


@dataclass
class Benchmark:
    """Run a grid of (problem x algorithm x seed) optimization runs."""

    problems: list[Problem]
    algorithms: list[AlgorithmRun]
    seeds: list[int]
    output_dir: Path

    num_workers: int = 1
    """Process count. 1 (default) keeps it serial; >1 uses joblib's loky backend."""

    save_artifacts: bool = True
    """If True, dump traces.json + runs.csv + summary.csv into ``output_dir`` after run()."""

    traces: dict[tuple[str, str], list[RunTrace]] = field(default_factory=dict)
    """Keyed by (problem.name, algorithm.name) -> list of traces (one per seed)."""

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_jobs(
        self,
    ) -> list[tuple[AlgorithmRun, Problem, NDArray[np.float64], int]]:
        jobs = []
        for problem in self.problems:
            for seed in self.seeds:
                x0 = problem.starting_point(seed)
                for algorithm in self.algorithms:
                    jobs.append((algorithm, problem, x0, seed))
        return jobs

    def run(self, verbose: bool = True) -> dict[tuple[str, str], list[RunTrace]]:
        """Execute all runs and store traces. Returns the trace dictionary."""
        jobs = self._build_jobs()
        total_runs = len(jobs)
        start = time.time()

        if verbose:
            mode = (
                "serial"
                if self.num_workers <= 1
                else f"parallel (n_workers={self.num_workers})"
            )
            print(f"Running {total_runs} jobs in {mode} mode.")

        if self.num_workers <= 1:
            results = self._run_serial(jobs, verbose, start)
        else:
            results = self._run_parallel(jobs, verbose, start)

        for problem_name, algorithm_name, trace in results:
            key = (problem_name, algorithm_name)
            self.traces.setdefault(key, []).append(trace)

        # Sort each (problem, algorithm) bucket by seed so the trace order is
        # stable regardless of parallel completion order.
        for traces in self.traces.values():
            traces.sort(key=lambda trace: trace.seed)

        if verbose:
            print(f"\nAll runs done in {time.time() - start:.1f}s.")

        if self.save_artifacts:
            self._save_artifacts(verbose=verbose)

        return self.traces

    def _run_serial(
        self,
        jobs: list,
        verbose: bool,
        start: float,
    ) -> list[tuple[str, str, RunTrace]]:
        results: list[tuple[str, str, RunTrace]] = []
        prev_problem: str | None = None
        prev_seed: int | None = None
        for idx, (algorithm, problem, x0, seed) in enumerate(jobs, start=1):
            if verbose and problem.name != prev_problem:
                print(f"\n{'=' * 70}")
                print(
                    f"Problem: {problem.name} "
                    f"(d={problem.dimensions}, "
                    f"bounds=[{problem.lower_bound}, {problem.upper_bound}])"
                )
                print(f"{'=' * 70}")
                prev_problem = problem.name
                prev_seed = None
            if verbose and seed != prev_seed:
                f_x0 = float(problem.function(x0))
                print(f"\n  seed={seed} | f(x0) = {f_x0:.4e}")
                prev_seed = seed

            trace = algorithm.run(problem, x0, seed)
            results.append((problem.name, algorithm.name, trace))

            if verbose:
                self._print_progress(idx, len(jobs), algorithm.name, trace, start)
        return results

    def _run_parallel(
        self,
        jobs: list,
        verbose: bool,
        start: float,
    ) -> list[tuple[str, str, RunTrace]]:
        # joblib's loky backend ships closures via cloudpickle, so lambdas in
        # config_factory work.
        completed = 0
        total = len(jobs)
        results: list[tuple[str, str, RunTrace]] = []

        def callback(result: tuple[str, str, RunTrace]) -> tuple[str, str, RunTrace]:
            nonlocal completed
            completed += 1
            if verbose:
                _, algorithm_name, trace = result
                self._print_progress(completed, total, algorithm_name, trace, start)
            return result

        raw = Parallel(n_jobs=self.num_workers, backend="loky")(
            delayed(_execute_one)(algorithm, problem, x0, seed)
            for algorithm, problem, x0, seed in jobs
        )
        for result in raw or []:
            results.append(callback(result))  # type: ignore[arg-type]
        return results

    def _print_progress(
        self,
        idx: int,
        total: int,
        algorithm_name: str,
        trace: RunTrace,
        start: float,
    ) -> None:
        handoff_str = (
            f" handoff@{trace.handoff_eval}" if trace.handoff_eval is not None else ""
        )
        elapsed = time.time() - start
        print(
            f"    [{idx:>3d}/{total}] "
            f"{algorithm_name:32s} "
            f"problem={trace.problem:12s} "
            f"seed={trace.seed:>3d}  "
            f"f={trace.final_fitness:.4e}  "
            f"evals={trace.final_evaluations:>6d}"
            f"{handoff_str}  "
            f"({elapsed:.1f}s)"
        )

    def _save_artifacts(self, verbose: bool) -> None:
        traces_path = save_traces_json(self.traces, self.output_dir / "traces.json")
        runs_path = save_runs_csv(self.traces, self.output_dir / "runs.csv")
        summary_path = save_summary_csv(
            self.summary_table(), self.output_dir / "summary.csv"
        )
        if verbose:
            print("\nArtifacts written:")
            print(f"  - {traces_path}")
            print(f"  - {runs_path}")
            print(f"  - {summary_path}")

    def traces_for(self, problem_name: str, algorithm_name: str) -> list[RunTrace]:
        return self.traces.get((problem_name, algorithm_name), [])

    def summary_table(self) -> list[dict]:
        """One row per (problem, algorithm) with aggregate statistics."""
        import numpy as np

        rows: list[dict] = []
        for problem in self.problems:
            for algorithm in self.algorithms:
                traces = self.traces_for(problem.name, algorithm.name)
                if not traces:
                    continue
                final_fitnesses = np.array([t.final_fitness for t in traces])
                final_evaluations = np.array([t.final_evaluations for t in traces])
                rows.append(
                    {
                        "problem": problem.name,
                        "algorithm": algorithm.name,
                        "n_runs": len(traces),
                        "median_fitness": float(np.median(final_fitnesses)),
                        "mean_fitness": float(np.mean(final_fitnesses)),
                        "best_fitness": float(np.min(final_fitnesses)),
                        "worst_fitness": float(np.max(final_fitnesses)),
                        "median_evaluations": float(np.median(final_evaluations)),
                    }
                )
        return rows

    def print_summary(self) -> None:
        rows = self.summary_table()
        if not rows:
            print("No runs to summarise.")
            return

        header = (
            f"{'Problem':<15s} {'Algorithm':<32s} "
            f"{'n':>3s} {'median f':>12s} {'best f':>12s} "
            f"{'worst f':>12s} {'median evals':>13s}"
        )
        print()
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['problem']:<15s} {row['algorithm']:<32s} "
                f"{row['n_runs']:>3d} "
                f"{row['median_fitness']:>12.4e} "
                f"{row['best_fitness']:>12.4e} "
                f"{row['worst_fitness']:>12.4e} "
                f"{int(row['median_evaluations']):>13d}"
            )
