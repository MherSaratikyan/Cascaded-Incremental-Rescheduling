"""Performance smoke tier (``pytest -m perf``, about 20-30 minutes).

Scenarios H1 (room closed for a week), H5 (10 voluntary requests), H7 (H1
under dense regulation) and H9 (room closed for one day, gap protection off -
usually repairable, so the distance P_k / P_k* to the proven optimum is
measured) on sizes S and M, methods B7 (proposed) and B4
(regulation-limited full re-optimisation), plus the reference run REF (B4 with
a long limit) for the single-request scenarios.  Every request runs in a fresh
process, exactly as in ``python -m experiments.perf``; the tests assert budgets
and invariants on the measured rows.

When the tier finishes, the measured rows are merged into
``experiments/results/perf_tests.csv`` (a partial run such as ``-k H9`` replaces
only the sequences it re-measured) and the report with the four charts is
rebuilt in ``experiments/results/perf_tests_report/``.
"""
import csv
from pathlib import Path

import pytest

from experiments.perf import FIELDS, REF, run_sequence

pytestmark = pytest.mark.perf

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
TIME_LIMIT, EXPLAIN, REF_TIME, GUARD_MB, MARGIN = 30.0, 10.0, 120.0, 3072.0, 15.0
CASES = [(size, sid) for size in ("S", "M") for sid in ("H1", "H5", "H7", "H9")]
SINGLE_CASES = [c for c in CASES if c[1] != "H5"]
_cache: dict = {}


@pytest.fixture(scope="module", autouse=True)
def save_results_and_charts():
    """After the last perf test: write every measured row to perf_tests.csv
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
                kept = [r for r in rd if tuple(r[k] for k in key) not in fresh]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
        w.writerows(rows)
    from experiments.perf_report import build_report
    report = build_report(path, RESULTS / "perf_tests_report")
    print(f"\nperf results: {path}\nperf report and charts: {report}")


def rows_for(size: str, sid: str, method: str) -> list[dict]:
    key = (size, sid, method)
    if key not in _cache:
        _cache[key] = run_sequence(dict(
            size=size, seed=1, scenario=sid, method=method, time_limit=TIME_LIMIT,
            ref_time=REF_TIME, explain_time=EXPLAIN, mem_limit_mb=GUARD_MB, solver_workers=2,
            cache=str(ROOT / "experiments" / "instances")))
    return _cache[key]


@pytest.mark.parametrize("size,sid", CASES)
@pytest.mark.parametrize("method", ["B7", "B4"])
def test_requests_finish_within_time_and_memory_budget(size, sid, method):
    """Every request ends with a real outcome (no crash, no memory-guard kill),
    its elapsed wall-clock time stays within time limit + explanation limit +
    15 s of model building, and its peak committed memory stays under 3 GB."""
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


@pytest.mark.parametrize("size", ["S", "M"])
def test_feasible_scenario_measures_distance_to_optimum(size):
    """H9 (one room closed for one day, gap protection off) is repairable: the
    reference proves the optimum P_k* (required on S; on M only if it finishes
    within its limit), B7 solves the request without regulation violations, and
    its P_k / P_k* (>= 1) is recorded for the report."""
    ref = rows_for(size, "H9", REF)[0]
    b7 = rows_for(size, "H9", "B7")[0]
    if size == "M" and not (ref["outcome"] == "solved" and ref.get("pstar_proven")):
        pytest.skip(f"M reference not proven within {REF_TIME:g} s: {ref['outcome']}")
    assert ref["outcome"] == "solved" and ref.get("pstar_proven"), ref["outcome"]
    assert b7["outcome"] == "solved", b7["outcome"]
    star, pk = float(ref["pstar"]), float(b7["P_k"])
    assert pk >= star
    assert int(b7["protected_violations"]) == int(b7["lock_violations"]) == 0


@pytest.mark.parametrize("size,sid", [c for c in SINGLE_CASES])
def test_changes_never_below_the_proven_optimum(size, sid):
    """P_k of any regulation-respecting solution is >= the proven optimum
    P_k* of the reference run (P_k / P_k* >= 1), and B4 attains it when both
    are proven optimal."""
    ref = rows_for(size, sid, REF)[0]
    if ref["outcome"] != "solved" or not ref.get("pstar_proven"):
        pytest.skip(f"reference not proven: {ref['outcome']}")
    star = float(ref["pstar"])
    for method in ("B7", "B4"):
        r = rows_for(size, sid, method)[0]
        if r["outcome"] == "solved":
            assert float(r["P_k"]) >= star, (method, r["P_k"], star)
    b4 = rows_for(size, sid, "B4")[0]
    if b4["outcome"] == "solved" and b4.get("pstar_proven"):
        assert float(b4["pstar"]) == star


@pytest.mark.parametrize("size,sid", CASES)
def test_outcome_classes_agree_between_b7_and_b4(size, sid):
    """For single-request scenarios both exact methods reach the same verdict
    whenever neither ran out of time (Claim 2: full expansion = B4)."""
    if sid == "H5":
        pytest.skip("sequences diverge after the first step")
    a, b = rows_for(size, sid, "B7")[0], rows_for(size, sid, "B4")[0]
    timeouts = {"algorithmic_failure", "killed_timeout"}
    if a["outcome"] in timeouts or b["outcome"] in timeouts:
        pytest.skip(f"time limit reached: B7={a['outcome']}, B4={b['outcome']}")
    assert (a["outcome"] == "solved") == (b["outcome"] == "solved"), (a["outcome"], b["outcome"])
