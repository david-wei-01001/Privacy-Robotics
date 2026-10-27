# 0008: Reporting event recall at the ceiling
- Status: accepted (2026-10-27).
- Context: in Night 1 (`2026-10-27_n1`), three encoders (`clap`, `ast`, `vggish`) reached 97% to 100% event recall
  at FMR 1% on the home track, and `clap` detected all 67 validation events (run `2026-10-27_n1-0023`). A bootstrap
  CI over events then collapses to one point (100.0 to 100.0), which made CI-overlap statements in `comparison.md`
  and the CI width in the go / no-go table look more certain than 67 events allow. Comparisons where both options
  reach 100% at FMR 1% also carry no information.
- Options: 1) keep the bootstrap CI everywhere; 2) where every event or none is detected, report the exact
  Clopper-Pearson interval instead, and compare options that both reach 100% at FMR 1% at the 0.1% target.
- Decisions:
  1) Option 2. `evaluate.py` still stores the bootstrap CI for every run; `report.derive()` adds the exact interval
     for runs whose bootstrap CI collapsed (`derived.exact_recall_ci` in `facts.json`), and the report marks it
     "exact". For 67 of 67 events it is 94.6 to 100.0.
  2) When two options both detect every event at FMR 1%, the bullet compares them at 0.1% and cites both targets.
  3) "Best LSH setting" is chosen among runs whose measured FMR stays within 1.5 times the target (PLAN §7.3,
     item 1), and the report says that choosing it on the validation events makes its recall slightly optimistic.
- Evidence: `reports/2026-10-27_n1/comparison.md` and `feasibility.md`, re-rendered with `--report-only` from the
  run's `facts.json` (the report code is kept in `runs/2026-10-27_n1/code/report_rerender.py`); the lint passes.
- Consequences: no run is recomputed and `autoresearch/results.tsv` is unchanged. PLAN §7.3 (the decision rule,
  which changes only with Elise and David agreeing) is not touched.
