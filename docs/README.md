# Documentation index

Algorithm references, design notes, and study writeups. Organized
roughly chronologically and by topic.

## Algorithm references

- [`lbfgsb_lecture.md`](lbfgsb_lecture.md) — Full algorithm lecture for
  L-BFGS-B v3.0 (11 sections: representation, Cauchy point, subspace
  minimization, Woodbury identity, More-Thuente line search, etc.).
- [`lbfgsb_initial_hessian_design.md`](lbfgsb_initial_hessian_design.md) —
  Cost analysis and dispatch strategy for the configurable initial
  Hessian extension (None / scalar / diagonal / dense).
- [`cmaes_framework_integration.md`](cmaes_framework_integration.md) —
  How the framework-native CMA-ES wires `RepairStrategy` and
  `PopulationInitializer`, what changed vs the historical reference
  port, and the convergence-equivalence evidence.

## Study writeups

- [`covariance_handoff_when_it_matters.md`](covariance_handoff_when_it_matters.md) —
  When does passing the CMA-ES covariance as `B₀` to L-BFGS-B actually
  help? Walks through the `RippledEllipsoid` study that isolates the
  decisive regime.
- [`supervisor_report_initial_hessian.md`](supervisor_report_initial_hessian.md) —
  English-language supervisor report on the initial-Hessian study, with
  graphs from the rotation experiments.
- [`cmaes_diagnostic_plots.md`](cmaes_diagnostic_plots.md) — Legend for
  the CMA-ES 8-panel diagnostic figure (convergence, σ, condition,
  determinant, eigenvalues, evolution paths, mean norm).

## Polish-language thesis materials

- [`wnioski.md`](wnioski.md) — Conclusions for the supervisor, sections
  1–5 (Polish).
- [`wnioski.pdf`](wnioski.pdf) — Rendered PDF.
