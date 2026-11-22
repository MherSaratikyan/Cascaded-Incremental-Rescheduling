"""Shared fixtures, the ``perf`` tier switch, and the per-test elapsed-time table.

* ``pytest``          unit and paper tests (perf tests are collected but skipped)
* ``pytest -m perf``  performance smoke tests (each request in a fresh process)

At the end of every run a table lists the real (wall-clock) elapsed time of
each test case (setup + call + teardown).
"""
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade import TimetableStore, load_scenario  # noqa: E402

_ELAPSED: dict[str, float] = {}
_OUTCOME: dict[str, str] = {}


_START = time.time()


def pytest_configure(config):
    config.addinivalue_line("markers", "perf: heavy performance tests (run with: pytest -m perf)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("markexpr"):      # an explicit -m expression selects by itself
        return
    skip = pytest.mark.skip(reason="performance tier - run with: pytest -m perf")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip)


def pytest_runtest_logreport(report):
    _ELAPSED[report.nodeid] = _ELAPSED.get(report.nodeid, 0.0) + report.duration
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _OUTCOME[report.nodeid] = report.outcome


def pytest_terminal_summary(terminalreporter):
    ran = [(n, t) for n, t in _ELAPSED.items() if _OUTCOME.get(n) in ("passed", "failed")]
    if not ran:
        return
    tr = terminalreporter
    tr.section("elapsed (wall-clock) time per test case")
    width = min(100, max(len(n) for n, _ in ran))
    for n, t in sorted(ran, key=lambda x: -x[1]):
        tr.write_line(f"{t:9.2f} s  {_OUTCOME[n]:<6}  {n[:width]}")
    tr.write_line(f"{sum(t for _, t in ran):9.2f} s  total over {len(ran)} test cases")
    report = ROOT / "experiments" / "results" / "perf_tests_report" / "report.md"
    if report.exists() and report.stat().st_mtime >= _START:
        tr.section("performance report and charts")
        tr.write_line(f"report : {report}")
        for png in sorted(report.parent.glob("*.png")):
            tr.write_line(f"chart  : {png}")


@pytest.fixture
def examples() -> Path:
    return ROOT / "examples"


def make_store(sc, **over):
    return TimetableStore(sc.instance, sc.timetable, sc.initial,
                          over.get("locks", sc.locks), over.get("protected", sc.protected),
                          over.get("gap_protected", sc.gap_protected))


@pytest.fixture
def chain(examples):
    """Section 6 chain example (rooms R0, R1; lessons A, B, C; protected s_B)."""
    return load_scenario(examples / "chain_example.json")


@pytest.fixture
def claim3(examples):
    """Section 6 counterexample to Claim 3 (lessons X, Y, A of group G1)."""
    return load_scenario(examples / "claim3_counterexample.json")
