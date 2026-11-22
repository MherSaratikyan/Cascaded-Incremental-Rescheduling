"""Heavy-workload performance run: scenarios H1-H9 on sizes S, M, L, XL and on
every ITC 2019(-style) instance in experiments/itc (as ``itc-<problem>``).

Every request runs in a fresh process (exact peak memory, memory guard) with a
1-hour safety cap, so in practice it ends with a proven outcome: a solution, or
proven physical / regulatory infeasibility.  The optimum P_k* used for
P_k / P_k* is B4's own P*, whenever B4 proved it optimal (B4 re-optimises the
whole timetable under the same regulation).

    python -m experiments.perf                         # everything (long: hours to days, resumable)
    python -m experiments.perf --sizes S itc-synth-s   # a smaller selection
    python -m experiments.perf --report-only           # rebuild the report from the CSV
    python -m experiments.perf --sizes S --scenarios H6 --budget-minutes 290   # one CI job

Output: experiments/results/perf.csv and experiments/results/perf_report/ (report.md + charts).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade.baselines import METHODS  # noqa: E402
from cascade.params import Params  # noqa: E402
from experiments import scenarios as sc  # noqa: E402
from experiments.generator import itc_sizes, load_or_generate, sample_regulation  # noqa: E402
from experiments.measure import ALL_METRICS, State, advance_state, run_isolated  # noqa: E402
from experiments.runner import _done_keys, open_csv  # noqa: E402

KEY = ("size", "seed", "scenario", "method")
FIELDS = list(KEY) + ["step", "time_limit", "lessons"] + ALL_METRICS
SEEDS = {"S": 3, "M": 3, "L": 1, "XL": 1}
CAP = 3600.0          # per-request safety cap (search) and explanation cap, seconds
REF = "REF"           # legacy reference rows (older CSVs); no longer produced
REFERENCE_METHOD = "B4"


class TimeBudget:
    """Shares a wall-clock budget over a known number of requests (e.g. one CI
    job, which must stay under GitHub's 6-hour job limit).

    Before each request its share = remaining budget / remaining requests.  A
    quarter of the share (at most ``overhead`` seconds) is reserved for process
    start, instance load and model building - the request process is killed
    after search + explanations + that reserve - and the rest is split 75 % search
    / 25 % explanations, each capped at ``cap`` (time the search cannot use goes
    to the explanations).  So a request never takes more than its share, the job
    never more than the budget (as long as shares stay above ~2 x ``floor``), and
    requests that finish early leave their time to later ones."""

    def __init__(self, seconds: float, requests: int, cap: float = CAP,
                 overhead: float = 60.0, explain_share: float = 0.25, floor: float = 2.0):
        self.end = time.monotonic() + seconds
        self.left = max(1, requests)
        self.cap, self.overhead, self.share, self.floor = cap, overhead, explain_share, floor
        self._reserve = overhead

    def limits(self) -> tuple[float, float]:
        share = max(0.0, self.end - time.monotonic()) / max(1, self.left)
        self._reserve = min(self.overhead, max(1.0, 0.25 * share))
        usable = max(2 * self.floor, share - self._reserve)
        search = min(self.cap, max(self.floor, usable * (1 - self.share)))
        explain = min(self.cap, max(self.floor, usable - search))
        return round(search, 1), round(explain, 1)

    def done(self) -> None:
        self.left = max(1, self.left - 1)

    @property
    def kill_margin(self) -> float:
        """Reserve of the last :meth:`limits` call (process start, load, build)."""
        return self._reserve


def requests_of(task: dict) -> int:
    return sc.steps_of(task["scenario"])


def run_sequence(task: dict, budget: "TimeBudget | None" = None) -> list[dict]:
    inst, x0 = load_or_generate(task["size"], task["seed"], Path(task["cache"]))
    rho, lam = (0.5, 0.3) if task["scenario"] == "H7" else (0.2, 0.1)
    protected, locks = sample_regulation(inst, task["seed"], rho, lam)
    hs = sc.build(task["scenario"], inst, x0, task["seed"], locks)
    base = Params(time_limit=task["time_limit"], explain_time_limit=task["explain_time"],
                  workers=task["solver_workers"], seed=task["seed"], gap_protect=hs.gap_protect)
    state = State(inst, x0, x0, locks, protected, None)
    rows = []
    for k, dis in enumerate(hs.disruptions):
        params, kill_after = base, None
        if budget is not None:
            search, explain = budget.limits()
            params = base.with_(time_limit=min(base.time_limit, search),
                                explain_time_limit=min(base.explain_time_limit, explain))
            kill_after = params.time_limit + params.explain_time_limit + budget.kill_margin
        row, new_state = run_isolated(state, dis, params, task["method"],
                                      mem_limit_mb=task["mem_limit_mb"], kill_after=kill_after)
        if budget is not None:
            budget.done()
        state = advance_state(state, dis, new_state)
        row.update({f: task[f] for f in KEY}, step=k, time_limit=params.time_limit,
                   lessons=len(inst.lessons))
        rows.append(row)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=["S", "M", "L", "XL"] + itc_sizes(),
                    help="synthetic sizes and itc-<problem> names (default: S M L XL + every "
                         "ITC 2019(-style) XML in experiments/itc)")
    ap.add_argument("--scenarios", nargs="+", default=list(sc.ALL))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--seeds", type=int, default=None,
                    help="seeds per size (default: S/M 3, L/XL and ITC 1)")
    ap.add_argument("--time-limit", type=float, default=CAP,
                    help="per-request safety cap for the search, seconds (default 1 h)")
    ap.add_argument("--explain-time", type=float, default=CAP,
                    help="cap for MUS / MCS / physical check, seconds (default 1 h)")
    ap.add_argument("--budget-minutes", type=float, default=None,
                    help="total wall-clock budget for all requests of this run (e.g. a CI job): "
                         "each request gets remaining budget / remaining requests, capped by "
                         "--time-limit and --explain-time")
    # accepted for older scripts / CI workflows; no effect: the reference optimum
    # is now B4's own proven P* (no separate REF run)
    ap.add_argument("--ref-time", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--no-ref", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mem-limit-mb", type=float, default=3072)
    ap.add_argument("--solver-workers", type=int, default=2)
    ap.add_argument("--cache", default=str(ROOT / "experiments" / "instances"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "perf.csv"))
    ap.add_argument("--report-dir", default=str(ROOT / "experiments" / "results" / "perf_report"))
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not a.report_only:
        tasks = []
        for size in a.sizes:
            for seed in range(1, (a.seeds or SEEDS.get(size, 1)) + 1):
                t = time.perf_counter()
                inst, _ = load_or_generate(size, seed, Path(a.cache))
                print(f"instance {size}-{seed}: {len(inst.lessons)} lessons "
                      f"({time.perf_counter() - t:.1f}s)", flush=True)
                for sid in a.scenarios:
                    for m in a.methods:
                        tasks.append(dict(size=size, seed=seed, scenario=sid, method=m,
                                          time_limit=a.time_limit, explain_time=a.explain_time,
                                          mem_limit_mb=a.mem_limit_mb,
                                          solver_workers=a.solver_workers, cache=a.cache))
        done = _done_keys(out, FIELDS, KEY)
        todo = [t for t in tasks if tuple(str(t[k]) for k in KEY) not in done]
        print(f"{len(todo)} of {len(tasks)} sequences to run -> {out}", flush=True)
        budget = None
        if a.budget_minutes:
            n_req = sum(requests_of(t) for t in todo)
            budget = TimeBudget(a.budget_minutes * 60, n_req)
            s, e = budget.limits()
            print(f"time budget {a.budget_minutes:g} min for {n_req} requests: "
                  f"first request {s:g} s search + {e:g} s explanations", flush=True)
        with open_csv(out, FIELDS) as (fh, w):
            for i, t in enumerate(todo, 1):
                t0 = time.perf_counter()
                rows = run_sequence(t, budget)
                w.writerows(rows)
                fh.flush()
                outs = ", ".join(sorted({r["outcome"] for r in rows}))
                el = sum(float(r.get("elapsed_s") or 0) for r in rows)
                pk = max(float(r.get("peak_commit_mb") or 0) for r in rows)
                print(f"[{i}/{len(todo)}] {t['size']}-{t['seed']} {t['scenario']} {t['method']}: "
                      f"{len(rows)} req, {outs}, elapsed {el:.1f}s, peak commit {pk:.0f} MB "
                      f"(wall {time.perf_counter() - t0:.0f}s)", flush=True)

    from experiments.perf_report import build_report
    path = build_report(out, Path(a.report_dir))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
