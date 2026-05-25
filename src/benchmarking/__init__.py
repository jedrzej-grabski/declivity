"""Benchmarking framework for fair, repeatable algorithm comparisons.

Quick example:

    from src.benchmarking import (
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

from src.benchmarking.algorithm_run import (
    AlgorithmRun,
    CMAESLBFGSBHandoff,
    SingleAlgorithm,
)
from src.benchmarking.benchmark import Benchmark
from src.benchmarking.persistence import (
    load_traces_json,
    save_runs_csv,
    save_summary_csv,
    save_traces_json,
)
from src.benchmarking.plotter import BenchmarkPlotter
from src.benchmarking.problem import Problem
from src.benchmarking.run_trace import RunTrace

__all__ = [
    "AlgorithmRun",
    "Benchmark",
    "BenchmarkPlotter",
    "CMAESLBFGSBHandoff",
    "Problem",
    "RunTrace",
    "SingleAlgorithm",
    "load_traces_json",
    "save_runs_csv",
    "save_summary_csv",
    "save_traces_json",
]
