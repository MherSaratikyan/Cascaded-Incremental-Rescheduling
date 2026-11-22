"""TimeBudget: a CI job's requests share a wall-clock budget so the job can
never pass GitHub's 6-hour limit, while fast requests leave time to hard ones."""
import experiments.perf as perf
from experiments.perf import TimeBudget


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_budget_splits_remaining_time_over_remaining_requests(monkeypatch):
    """300 min for 420 requests (the largest CI job): each request's share is
    about 43 s; search + explanations + the process reserve add up to it, with
    search about three times the explanation limit."""
    clock = Clock()
    monkeypatch.setattr(perf.time, "monotonic", clock)
    b = TimeBudget(300 * 60, 420)
    s, e = b.limits()
    share = 300 * 60 / 420
    assert abs(s + e + b.kill_margin - share) < 0.3
    assert abs(s - 3 * e) < 0.5 and s > 20


def test_time_saved_by_fast_requests_goes_to_later_ones(monkeypatch):
    """If the first request finishes after 1 s, the next one gets more time."""
    clock = Clock()
    monkeypatch.setattr(perf.time, "monotonic", clock)
    b = TimeBudget(3600, 10)
    s1, _ = b.limits()
    clock.t += 1
    b.done()
    s2, _ = b.limits()
    assert s2 > s1


def test_worst_case_never_exceeds_the_budget(monkeypatch):
    """Even if every request uses its full search + explanation limit + the
    overhead allowance, the job ends within the budget."""
    clock = Clock()
    monkeypatch.setattr(perf.time, "monotonic", clock)
    budget_s, n = 300 * 60, 420
    b = TimeBudget(budget_s, n)
    start = clock.t
    for _ in range(n):
        s, e = b.limits()
        clock.t += s + e + b.kill_margin
        b.done()
    assert clock.t - start <= budget_s + 1


def test_limits_are_capped_at_one_hour(monkeypatch):
    """A job with few requests still never gives a request more than the caps
    (search time beyond the cap goes to the explanations, also capped)."""
    monkeypatch.setattr(perf.time, "monotonic", Clock())
    s, e = TimeBudget(300 * 60, 2).limits()
    assert s == perf.CAP and e == perf.CAP
