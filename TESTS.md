# Test cases

Three tiers. Every run prints a table with the **elapsed (wall-clock) time of each test case** at the end.

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest -m perf -q
.venv\Scripts\python.exe -m experiments.perf
```

The first runs the 43 unit and paper tests (about 40 s), the second the performance smoke tests
(about 20–30 min), and the third the full heavy workload (about 9–10 h).

To run every pytest test in one go (tiers 1 and 2, 71 test cases):

```powershell
.venv\Scripts\python.exe -m pytest -m "perf or not perf" -v
```

Times below were measured on the development laptop (4-core Ryzen 3, 7.3 GB RAM).
Every test function also has a docstring, which `pytest -v` and Visual Studio's Test Explorer show.

## Scenarios used by the paper tests

**Chain example** (`examples/chain_example.json`, §6). One day of 5 hours, rooms R0 and R1, and
lessons A, B and C with their own lecturers (LA, LB, LC) and groups.
- A sits at hour 1 in R0. Its lecturer is free only at hours 1–2, and R1 is booked at hour 1.
  Once R0 closes, A's only admissible assignment is hour 2 in R1, where B is.
- B can use hours 2–4 in R1, but hour 3 violates its lecturer's protected preference s_B (status P0).
- C sits at hour 4 and can also use hour 5, which is free.

**Claim 3 counterexample** (`examples/claim3_counterexample.json`, §6). Group G1 has X, Y and A at hours 1–3.
- A's room closes, leaving Ω_A = {1, 6}. Ω_X = {1, 3}.
- No gap protection applies.

## Tier 1: unit and paper tests (`pytest`)

### `tests/test_examples.py`: the §6 worked examples

| Test | Paper | Setup | Expected result and why | Time |
|---|---|---|---|---|
| `test_chain_repair_found_through_cascade` | §6, Table 1, rule (12) | Chain example, R0 closes, method B7 | **Solved** with A→2, B→4, C→5. F_0={A} is infeasible and the core names B's fixing. F_1={A,B} is infeasible: B→3 breaks s_B, B→4 hits C, and the core names C. F_2={A,B,C} succeeds after one budget increase. P_k = 3·w^T, no protected violation. | 0.11 s |
| `test_local_check_rejects_the_move` | Baseline B1 ([7]) | Same | **Rejected by the local check**: only A may move, and its only slot is taken. | 0.01 s |
| `test_hard_only_chain_violates_protected_preference` | Baseline B2 | Same | **Solved but wrong**: without protection the shorter chain A→2, B→3 wins and breaks s_B, so protected violations = 1. This is why protection levels (4) exist. | 0.06 s |
| `test_regulation_limited_full_reoptimisation_agrees` | B4 vs B7 | Same | Both solved with the same P_k: the incremental method loses nothing on this case. | 0.15 s |
| `test_locked_c_gives_regulatory_infeasibility_with_mus_and_mcs` | Step 5, MUS [8], MCS [11] | Chain example with C fully locked | **Regulatory infeasibility**: the hard constraints alone are satisfiable. MUS = {lock C, s_B} (proven minimal). Two corrections are verified by re-solving: lift C's lock, or relax s_B. The Armenian text reads «C դասի ամրագրումը պետք է հանվի». | 0.09 s |
| `test_lock_conflict_without_search` | Table 1, step 2 | Request to move fully locked C | **Lock conflict** with no search iteration: C ∈ D_0 ∩ Λ^A. | 0.01 s |
| `test_physical_infeasibility` | Step 5, Claim 2 | LB loses hours 3–4, then R0 closes | **Physical infeasibility**: A and B both need R1 at hour 2. This is proven, not a time-out. | 0.03 s |
| `test_claim3_counterexample_extra_expansion` | §6, Claim 3, quality trigger | Claim 3 example, δ_k = w^T | **Solved** with A→1, X→3. A→6 costs 1 change but creates 3 gaps. ΔG⁺ = 3 > θ_G triggers the extra expansion, and the 2-change, no-gap repair is accepted. | 0.06 s |
| `test_claim3_with_zero_slack_keeps_single_move` | Frozen threshold P̄ | Same, δ_k = 0 | **Solved** with A→6. P̄ = P* forbids a second change, so the extra expansion is rejected and the 3 gaps remain. | 0.04 s |

### `tests/test_procedure.py`: Table 1, steps 6–7

| Test | Setup | Expected result and why | Time |
|---|---|---|---|
| `test_voluntary_chain_requires_approval` | LA voluntarily asks for A → hour 2 in R1 | Solved, δ_k = 0. The chain moves lessons of LB and LC, so the approver is asked with exactly [LB, LC]. The chain is committed (store version 1) and all six participants are notified. | 0.09 s |
| `test_rejected_chain_is_not_committed` | Same, approver says no | **Rejected by approver**. Timetable and version unchanged. The Armenian text mentions «իրավասու անձի». | 0.10 s |
| `test_version_conflict_restarts_from_step_1` | Another client writes during solving | The first commit fails the version check, the procedure restarts at step 1, and succeeds on attempt 2 (store version 2). | 0.22 s |
| `test_external_failure_persists_instance_change` | R0 closes, C locked (infeasible) | Regulatory infeasibility. The closed room is still recorded, because the event happened, and x^(k) is kept. | 0.13 s |

### `tests/test_gaps.py`: Claim 5

| Test | Setup | Expected result and why | Time |
|---|---|---|---|
| `test_gap_formulation_is_exact[0..24]` | 25 random fixed timetables: 1–6 lessons of one group over 2 days, some with a lunch break | Minimising G over (6)–(7) gives exactly the true number of gaps, and G ≤ true − 1 is infeasible. So the b/a/g formulation is exact without integrality. | 0.01–0.03 s each |

### `tests/test_claims.py`: claims on a generated XS instance

The instance has 20 lessons, ρ=0.3 protected and λ=0.15 locked, with random disruptions.

| Test | Paper | Expected result and why | Time |
|---|---|---|---|
| `test_claim1_frozen_threshold_expansion_never_worsens` | Claim 1 | With P̄ frozen, each expansion F_j ⊆ F_{j+1} gives a lexicographic objective (14) that is ≤ the previous one. | 7.6 s |
| `test_claim2_full_expansion_equals_regulation_limited_reoptimisation` | Claim 2 | F = E∖Λ^A with B = Σw^T has the same feasibility and the same min P_k as the full model with locks as constraints. | 7.7 s |
| `test_proposed_method_agrees_with_b4_on_outcome` | Claims 1–2 | B7 solves a request exactly when B4 does, never violates hard, protected or lock constraints, and has P*_B7 ≥ P*_B4. | 18.5 s |
| `test_b6_heuristic_produces_valid_or_failure` | Baseline B6 | B6 either returns a hard-feasible timetable or reports an algorithmic failure. | 3.2 s |
| `test_io_roundtrip` | — | The instance and x^(0) survive JSON save and load, and the result stays hard-feasible. | <0.01 s |

## Tier 2: performance smoke tests (`pytest -m perf`)

These run scenarios H1, H5, H7 and H9 on sizes S (60 lessons) and M (200 lessons) with seed 1. The methods are
B7 and B4, plus the reference REF (B4 with a 120 s limit) for H1, H7 and H9. **Every request runs in a fresh
process**, exactly as in the heavy workload, and the tests check the measured rows:

| Test | Expected result and why |
|---|---|
| `test_requests_finish_within_time_and_memory_budget[method-size-scenario]` | No crash or memory-guard kill. Elapsed time ≤ 30 s limit + 10 s explanations + 15 s model building. Peak committed memory < 3 GB. |
| `test_proposed_method_never_breaks_regulation[size-scenario]` | Every B7 solution has zero hard, protected, lock and gap-protection violations, and every proposed MCS is verified. |
| `test_feasible_scenario_measures_distance_to_optimum[size]` | On H9 the reference proves the optimum P_k* (required on S, on M if it finishes within 120 s). B7 solves the request without regulation violations, and its P_k / P_k* (≥ 1) goes into the report. |
| `test_changes_never_below_the_proven_optimum[size-scenario]` | P_k / P_k* ≥ 1 for B7 and B4 against the proven optimum of REF. B4 attains P_k* when both are proven. |
| `test_outcome_classes_agree_between_b7_and_b4[size-scenario]` | Same verdict (solved or not) for single-request scenarios unless one hit the time limit (Claim 2). |

When the tier finishes, every measured row is saved to `experiments/results/perf_tests.csv`, and
`experiments/results/perf_tests_report/` receives `report.md` and the four charts: `elapsed_vs_size.png`,
`memory_vs_size.png`, `pk_ratio.png` and `outcomes.png`. The pytest summary prints their paths.

## Tier 3: heavy workload (`python -m experiments.perf`)

| Scenario | Disruption | What it stresses |
|---|---|---|
| H1 | The busiest room that holds no fully locked lesson is closed for the whole week | Large F_0, long chains |
| H2 | The lecturer with the most hours is absent on their 2 busiest days | Chains across groups |
| H3 | Hour 3 is closed in every room on two days | Simultaneous displacement in all rooms |
| H4 | The two largest rooms that can spare it lose 50% of their capacity | Room-only moves (w^R) |
| H5 | 10 voluntary move requests in sequence | Approval path, exact P* with δ_k = 0 |
| H6 | 20 mixed disruptions in sequence | P⁰ drift, lack of monotonicity between steps (Claim 4) |
| H7 | H1 under dense regulation (ρ = 0.5 protected, λ = 0.3 locked) | Regulatory infeasibility, MUS/MCS |
| H8 | One group keeps only h−1 of the h hours it needs | Physical-infeasibility proof (counting argument + CP-SAT) |
| H9 | The busiest room is closed on its busiest day, with gap protection off | A request that is usually repairable, so P_k / P_k* against the proven optimum can be measured |

Each scenario runs on sizes S, M, L and XL (60 / 200 / 500 / 1000 lessons) with 3, 3, 1 and 1 seeds,
all 7 methods, a 30 s limit plus 10 s for explanations, and the 300 s B4 reference for the
single-request scenarios. The run is resumable. The results go to `experiments/results/perf.csv`, and
`experiments/results/perf_report/report.md` holds the tables and the four charts.

**H9 was added after the first perf run.** Under the default regulation every single-request scenario
(H1, H7) came out regulatorily infeasible: all groups are protected against new gaps, and x^(0) has none.
So no solved request had a proven reference, and P_k / P_k* stayed unmeasured.

Deviations from the original plan, made after the first smoke runs:
- **H2** uses a 2-day absence instead of a whole week, because with no substitute lecturers in the model
  a week-long absence makes the lecturer's lessons impossible by definition.
- **H1, H7 and H4** pick rooms whose change still leaves every lesson some admissible option, and that
  hold no fully locked lesson. Otherwise every method stops at the same trivial physical infeasibility
  or step-2 lock conflict.
