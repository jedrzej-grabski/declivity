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
    HessianScaling,
    InterleavedCMAESLBFGSB,
    InterleaveResult,
    SingleAlgorithm,
    initial_hessian_from_cmaes,
)
from declivity.benchmarking.benchmark import Benchmark
from declivity.benchmarking.cmaes_path import (
    CMAESPath,
    CMAESSnapshot,
    load_cmaes_path,
    record_cmaes_path,
    save_cmaes_path,
)
from declivity.benchmarking.conditioning import (
    LOCAL_ALGORITHMS,
    ConditionedLocalAlgorithm,
    compose_switch_trace,
    local_seeding_kwargs,
    probe_trace,
    retag_trace,
    run_conditioned_local,
    snapshot_geometry,
)
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
from declivity.benchmarking.problem import Problem, ProblemFamily
from declivity.benchmarking.run_trace import RunTrace

__all__ = [
    "DEFAULT_THRESHOLD_FLOOR",
    "LOCAL_ALGORITHMS",
    "AlgorithmRun",
    "Benchmark",
    "BenchmarkAlgorithm",
    "CMAESLBFGSBHandoff",
    "CMAESLocalHandoff",
    "CMAESPath",
    "CMAESSnapshot",
    "ConditionedLocalAlgorithm",
    "HandoffAlgorithm",
    "HandoffTransform",
    "HessianScaling",
    "InterleaveResult",
    "InterleavedCMAESLBFGSB",
    "Problem",
    "ProblemFamily",
    "RunTrace",
    "SingleAlgorithm",
    "aggregate_ecdf",
    "compose_switch_trace",
    "ecdf_auc",
    "initial_hessian_from_cmaes",
    "load_cmaes_path",
    "load_traces_json",
    "local_seeding_kwargs",
    "probe_trace",
    "record_cmaes_path",
    "retag_trace",
    "run_conditioned_local",
    "run_ecdf",
    "save_cmaes_path",
    "save_runs_csv",
    "save_summary_csv",
    "save_traces_json",
    "snapshot_geometry",
    "threshold_grid",
]
