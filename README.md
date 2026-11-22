# Incremental cascaded rescheduling — Python implementation

Implementation of *"Incremental Cascaded Rescheduling of Dynamic University
Timetables with Protected Soft Constraints and Lesson Locks"* (M. Saratikyan),
using OR-Tools CP-SAT as the exact solver.

## Setup

All code lives in the solution folder, next to `Article implementation.slnx`.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The `.venv` is registered in `Article implementation.pyproj`, so Visual Studio uses it.
**F5** runs `Article_implementation.py`, which replays the worked examples of
Section 6 in English and Armenian.

```powershell
.venv\Scripts\python.exe Article_implementation.py --lang both
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m experiments.runner --quick
.venv\Scripts\python.exe -m experiments.analysis experiments\results\quick.csv --out experiments\results\quick.md
```

What each test checks, and why, is described in [TESTS.md](TESTS.md).

## Performance measurement

Every measured request (test case) runs in a **fresh process**, so Windows' per-process peak
counters belong to that request alone. Each row of the result CSVs records:

| Group | Columns |
|---|---|
| Time | `elapsed_s` (real wall-clock time of the request), `cpu_s`, `startup_s` (process start and imports, excluded from `elapsed_s`), `process_wall_s`, phases `t_search` / `t_optimise` / `t_extra` / `t_explain`, `solver_seconds`, `solver_calls` |
| Memory (MB) | `peak_virtual_mb` (reserved address space), `peak_commit_mb` (committed private bytes, Task Manager's "Commit size"), `peak_wset_mb` (physical RAM), `base_commit_mb` and `request_commit_mb` (growth during the request) |
| Quality | `P_k`, `P_0`, `G`, `dG_plus` (lecturers / groups), `Q_O`, `dV_pref` / `dV_load` / `dV_room`, `changed`, `hours_moved`, `day_changes`, `room_only_moves`, `lecturers_affected`, `groups_affected`, violation counts |
| Distance from optimal | `P_k / P_k*` and `P_k − P_k*` against a B4 reference run with a 300 s limit (heavy workload, single-request scenarios) |
| Certification | `pstar_proven`, `lexmin_proven`, `lexmin_stages`, `opt_gap_max` |
| Explanations | `mus_size`, `mus_minimal`, `mcs_count`, `mcs_verified` |
| Effort | `iterations`, `F_ratio`, `max_model_vars`, `max_model_constraints`, `final_model_*` |

A parent-side guard kills a request process whose committed memory passes `--mem-limit-mb`
(default 3072). The row then records `memory_limit`, or `killed_timeout` for a hung process.

Three test tiers:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest -m perf -q
.venv\Scripts\python.exe -m experiments.perf
```

1. **Unit and paper tests** (`pytest`): about 3 minutes, and it prints the elapsed time of every test case.
2. **Performance smoke tests** (`pytest -m perf`): about 20–30 minutes, with budgets and invariants on S and M.
3. **Heavy workload** (`experiments.perf`): about 9–10 hours and resumable. It runs scenarios H1–H8
   on S, M, L and XL (60 / 200 / 500 / 1000 lessons) with all methods plus the reference, and writes
   `experiments/results/perf.csv` and `experiments/results/perf_report/report.md` with four charts:
   elapsed time vs size, peak memory vs size, P_k/P_k*, and the outcome mix.

The L and XL starting timetables x^(0) come from a greedy construction followed by windowed
CP-SAT repair: at most 60 lessons of one day are re-optimised at a time, which keeps memory bounded.
S and M use one full CP-SAT solve.

## Where each part of the paper lives

| Paper | Code |
|---|---|
| Sets E, T, R, L, K, Π^b, δ_e, joint and linked lessons, special resources | `cascade/model.py` |
| Ω_e^(k+1) (independent of other lessons' occupancy), D_0, D, P_k (3), P^0 | `cascade/evaluate.py`, `cascade/context.py` |
| Hard constraints (1)–(2), protected soft constraints (4), locks (5), budget (13) | `cascade/cpsat_model.py` |
| Exact gap formulation (6)–(9), ΔG⁺ per day | `cascade/cpsat_model.py` (`_gap_expr`), checked against `evaluate.gaps` (Claim 5) |
| Statuses S^O / S^P0 / S^P≤; P0 already violated → P≤ and sent for review | `cascade/context.py` |
| Dependency graph: C^H, C^alt, C^P, W (10)–(11); expansion rule (12) with cl(·), C_j, Top_m | `cascade/dependency.py` |
| Table 1 steps 2–5: lock conflict, (F_j, B_j) growth, P* and frozen P̄ = P* + δ_k, lexmin (14), quality-triggered extra expansion, outcome classification | `cascade/rescheduler.py` |
| MUS [8], MCS [11] verified by re-solving (at most μ elements) | `cascade/explain.py` |
| Step 6 optimistic version check, step 7 approval and notification | `cascade/procedure.py` |
| Plain-language explanations (en / hy) | `cascade/messages.py` |
| Baselines B1–B6 of Section 7 | `cascade/baselines.py`, `cascade/rescheduler.py` (`STRATEGIES`) |
| Section 7 protocol (generator, runner, Wilcoxon + Holm) | `experiments/` |

Outcomes: `solved`, `physical_infeasibility`, `regulatory_infeasibility`,
`lock_conflict`, `algorithmic_failure` (plus `rejected_local_check` for B1 and
`rejected_by_approver` for step 7). An infeasibility label is given only after
CP-SAT proves X_J = ∅ with F_J = E∖Λ^A and B_J ≥ Σ w^T (Claim 2). Stopping
earlier is reported as an algorithmic failure, with a hard-only candidate and the last core.

## Implementation choices not fixed by the paper

- **Solver core C_j.** Fixings of the `core_pool` (default 30) highest-ranked
  neighbours are posted as CP-SAT assumption literals, so the core can name them.
  Lessons further away are constants. CP-SAT cores are not minimal. When the
  budget literal appears with other items, one extra check without it decides
  between "raise B_j" and "expand F_j".
- **Budget ladder B.** 2w^T, 4w^T, 8w^T, … and finally Σ_e w^T.
- **Lexicographic objective (14)** is solved as four successive CP-SAT optimisations,
  each warm-started from the previous solution and bounded by its value.
- **Q_O** uses integer coefficients round(1000·π_s / den_s).
- **Soft constraints.** Lecturer time preference (per lecturer), daily workload
  (per lecturer and day, with an auxiliary variable bounded only from below) and
  room stability (per course). All are linear, so (4) is exact.
- **B6** is an *approximation* of Akkan et al. [4]: simulated annealing with a
  perturbation limit on the same model, not their exact algorithm.
- **ITC 2019** loading is not implemented. The instance interface (`cascade/io.py`) is
  where a loader would plug in. Parameters were not tuned on the ITC "early" group.
- **CP-SAT settings.** `find_big_linear_overlap` is off (on these models this presolve step
  takes most of the run time). Models with more than 5000 x-variables skip probing and
  symmetry detection. Models with fewer than 3000 use a single worker.

All defaults are in `cascade/params.py`.

## Scenario format

See `examples/*.json`. Days and hours are 1-based. The fields are:

- `instance`: rooms, lecturers, groups, lessons, links and soft constraints.
- `timetable`: x^(k), with an optional `initial` x^(0).
- `locks`: `{lesson: "T" | "R" | "A"}`.
- `protected`: `{soft id: "P0" | "PLE"}`.
- `gap_protected`: U^P.
- `params`: overrides of `cascade/params.py`.
- `disruptions`: `room_unavailable`, `participant_unavailable`, `time_unavailable`,
  `capacity_change` or `move_request`.
