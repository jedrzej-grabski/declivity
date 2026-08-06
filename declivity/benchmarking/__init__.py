"""Benchmarking framework for fair, repeatable algorithm comparisons.

Quick example:

    from declivity.benchmarking import (
        Benchmark, Problem, SingleAlgorithm, CMAESLBFGSBHandoff,
    )

    problem = Problem.from_benchmark("Rastrigin-10D", Rastrigin(10))
    algorithms = [
        SingleAlgorithm(name="CMA-ES",   color="#e74c3c", ...),
        SingleAlgorithm(name="L-BFGS-B", color="#3498db", ...),
        CMAESLBFGSBHandoff(name="CMA-ES + L-BFGS-B", color="#2ecc71", ...),
    ]
    Benchmark([problem], algorithms, seeds=range(25), output_dir="plots/").run()
"""

from declivity.benchmarking.algorithm_run import (
    AlgorithmRun,
    BenchmarkAlgorithm,
    CMAESLBFGSBHandoff,
    CMAESLocalHandoff,
    HandoffAlgorithm,
    HandoffTransform,
    InterleavedCMAESLBFGSB,
    InterleaveResult,
    SingleAlgorithm,
    initial_hessian_from_cmaes,
)
from declivity.benchmarking.benchmark import Benchmark
from declivity.benchmarking.ecdf import (
    DEFAULT_THRESHOLD_FLOOR,
    aggregate_ecdf,
    ecdf_auc,
    run_ecdf,
    threshold_grid,
)
from declivity.benchmarking.persistence import (
    load_traces_json,
    save_runs_csv,
    save_summary_csv,
    save_traces_json,
)
from declivity.benchmarking.problem import Problem
from declivity.benchmarking.run_trace import RunTrace

__all__ = [
    "DEFAULT_THRESHOLD_FLOOR",
    "AlgorithmRun",
    "Benchmark",
    "BenchmarkAlgorithm",
    "CMAESLBFGSBHandoff",
    "CMAESLocalHandoff",
    "HandoffAlgorithm",
    "HandoffTransform",
    "InterleaveResult",
    "InterleavedCMAESLBFGSB",
    "Problem",
    "RunTrace",
    "SingleAlgorithm",
    "aggregate_ecdf",
    "ecdf_auc",
    "initial_hessian_from_cmaes",
    "load_traces_json",
    "run_ecdf",
    "save_runs_csv",
    "save_summary_csv",
    "save_traces_json",
    "threshold_grid",
]
