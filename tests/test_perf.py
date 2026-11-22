"""Performance tier (``pytest -m perf``; with the 1-hour safety cap it can take
from about an hour to several hours on a laptop).

Scenarios H1 (room closed for a week), H5 (10 voluntary requests), H7 (H1
under dense regulation) and H9 (room closed for one day, gap protection off -
usually repairable, so the distance P_k / P_k* to the proven optimum is
measured) on sizes S and M, and H1/H5/H9 on the two smallest ITC 2019(-style)
instances in experiments/itc.  Methods: B7 (proposed) and B4 (full
re-optimisation under the same regulation); B4's own proven P* is the optimum
P_k*.  Every request runs in a fresh process with a 1-hour cap, so in practice
each ends with a proven outcome (solution, or proven infeasibility).

When the tier finishes, the measured rows are merged into
``experiments/results/perf_tests.csv`` (a partial run such as ``-k H9`` replaces
only the sequences it re-measured) and the report with the four charts is
rebuilt in ``experiments/results/perf_tests_report/``.
"""
import csv
import os
from pathlib import Path

import pytest

from experiments.generator import ITC_DIR, itc_sizes
from experiments.perf import CAP, FIELDS, REFERENCE_METHOD, TimeBudget, run_sequence
from experiments.scenarios import steps_of

pytestmark = pytest.mark.perf

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
TIME_LIMIT, EXPLAIN, GUARD_MB, MARGIN = CAP, CAP, 3072.0, 60.0
# ITC 2019(-style) instances in experiments/itc: the two smallest files
ITC = sorted(itc_sizes(), key=lambda s: (ITC_DIR / f"{s[4:]}.xml").stat().st_size)[:2]
CASES = ([(size, sid) for size in ("S", "M") for sid in ("H1", "H5", "H7", "H9")]
         + [(size, sid) for size in ITC for sid in ("H1", "H5", "H9")])
SINGLE_CASES = [c for c in CASES if c[1] != "H5"]
TIMEOUTS = {"algorithmic_failure", "killed_timeout"}
_cache: dict = {}


@pytest.fixture(scope="module", autouse=True)
def save_results_and_charts():
    """After the last perf test: merge every measured row into perf_tests.csv
    and build report.md with the elapsed / memory / P_k/P_k* / outcome charts."""
    yield
    rows = [r for rs in _cache.values() for r in rs]
    if not rows:
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "perf_tests.csv"
    # merge: a partial run (e.g. -k H9) replaces only the sequences it re-measured
    key = ("size", "seed", "scenario", "method")
    fresh = {tuple(str(r[k]) for k in key) for r in rows}
    kept = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            if rd.fieldnames == FIELDS:
                kept = [r for r in rd if tuple(r[k] for k in key) not in fresh and r["method"] != "REF"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
        w.writerows(rows)
    from experiments.perf_report import build_report
    report = build_report(path, RESULTS / "perf_tests_report")
    print(f"\nperf results: {path}\nperf report and charts: {report}")


_budget: list = []


def budget():
    """With PERF_BUDGET_MINUTES set (CI), all requests of the tier share that
    wall-clock budget (see experiments.perf.TimeBudget); otherwise each request
    simply gets the 1-hour caps."""
    if not _budget:
        minutes = float(os.environ.get("PERF_BUDGET_MINUTES") or 0)
        planned = sum(2 * steps_of(sid) for _, sid in CASES)      # B7 and B4
        _budget.append(TimeBudget(minutes * 60, planned) if minutes > 0 else None)
    return _budget[0]


def rows_for(size: str, sid: str, method: str) -> list[dict]:
    key = (size, sid, method)
    if key not in _cache:
        _cache[key] = run_sequence(dict(
            size=size, seed=1, scenario=sid, method=method, time_limit=TIME_LIMIT,
            explain_time=EXPLAIN, mem_limit_mb=GUARD_MB, solver_workers=2,
            cache=str(ROOT / "experiments" / "instances")), budget())
    return _cache[key]


def proven_optimum(size: str, sid: str):
    """P_k* = B4's P* when B4 solved the request and proved P* optimal."""
    ref = rows_for(size, sid, REFERENCE_METHOD)[0]
    if ref["outcome"] == "solved" and ref.get("pstar_proven"):
        return float(ref["pstar"]), ref
    return None, ref


@pytest.mark.parametrize("size,sid", CASES)
@pytest.mark.parametrize("method", ["B7", "B4"])
def test_requests_finish_within_time_and_memory_budget(size, sid, method):
    """Every request ends with a real outcome (no crash, no memory-guard kill),
    its elapsed wall-clock time stays within the 1-hour search cap + the 1-hour
    explanation cap + 60 s of model building, and its peak committed memory
    stays under 3 GB."""
    for r in rows_for(size, sid, method):
        assert r["outcome"] not in ("error", "crashed", "memory_limit", "killed_timeout"), r
        assert float(r["elapsed_s"]) <= TIME_LIMIT + EXPLAIN + MARGIN, r
        assert float(r["peak_commit_mb"]) < GUARD_MB, r
        assert float(r["peak_wset_mb"]) <= float(r["peak_commit_mb"]) + 512, r


@pytest.mark.parametrize("size,sid", CASES)
def test_proposed_method_never_breaks_regulation(size, sid):
    """When B7 solves a request, the new timetable has no hard, protected,
    lock or gap-protection violation; every proposed correction set of a
    regulatory infeasibility is verified by re-solving."""
    for r in rows_for(size, sid, "B7"):
        if r["outcome"] == "solved":
            for f in ("hard_violations", "protected_violations", "lock_violations",
                      "gap_protection_violations"):
                assert int(r[f]) == 0, (f, r)
        if r["outcome"] == "regulatory_infeasibility":
            assert int(r["mcs_verified"]) == int(r["mcs_count"]), r


@pytest.mark.parametrize("size", ["S", "M"] + ITC)
def test_feasible_scenario_measures_distance_to_optimum(size):
    """H9 (one room closed for one day, gap protection off) is repairable: B4
    proves the optimum P_k* (required on S; elsewhere skipped only if B4 still
    hit the 1-hour cap), B7 solves the request without regulation violations,
    and its P_k / P_k* (>= 1) is recorded for the report."""
    star, ref = proven_optimum(size, "H9")
    b7 = rows_for(size, "H9", "B7")[0]
    if size != "S" and star is None:
        pytest.skip(f"{size}: B4 did not prove P* within the cap: {ref['outcome']}")
    assert star is not None, ref["outcome"]
    assert b7["outcome"] == "solved", b7["outcome"]
    assert float(b7["P_k"]) >= star
    assert int(b7["protected_violations"]) == int(b7["lock_violations"]) == 0


@pytest.mark.parametrize("size,sid", SINGLE_CASES)
def test_changes_never_below_the_proven_optimum(size, sid):
    """P_k of B7's regulation-respecting solution is >= the proven optimum P_k*
    (B4's proven P*), i.e. P_k / P_k* >= 1."""
    star, ref = proven_optimum(size, sid)
    if star is None:
        pytest.skip(f"no proven optimum: B4 outcome {ref['outcome']}")
    r = rows_for(size, sid, "B7")[0]
    if r["outcome"] == "solved":
        assert float(r["P_k"]) >= star, (r["P_k"], star)


@pytest.mark.parametrize("size,sid", SINGLE_CASES)
def test_outcome_classes_agree_between_b7_and_b4(size, sid):
    """Both exact methods reach the same verdict - solved, or proven infeasible
    (Claim 2: full expansion = B4) - unless one still hit the 1-hour cap."""
    a, b = rows_for(size, sid, "B7")[0], rows_for(size, sid, "B4")[0]
    if a["outcome"] in TIMEOUTS or b["outcome"] in TIMEOUTS:
        pytest.skip(f"1-hour cap reached: B7={a['outcome']}, B4={b['outcome']}")
    assert (a["outcome"] == "solved") == (b["outcome"] == "solved"), (a["outcome"], b["outcome"])
