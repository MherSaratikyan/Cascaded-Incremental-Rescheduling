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


def run_sequence(task: dict) -> list[dict]:
    inst, x0 = load_or_generate(task["size"], task["seed"], Path(task["cache"]))
    rho, lam = (0.5, 0.3) if task["scenario"] == "H7" else (0.2, 0.1)
    protected, locks = sample_regulation(inst, task["seed"], rho, lam)
    hs = sc.build(task["scenario"], inst, x0, task["seed"], locks)
    tl = task["time_limit"]
    params = Params(time_limit=tl, explain_time_limit=task["explain_time"],
                    workers=task["solver_workers"], seed=task["seed"], gap_protect=hs.gap_protect)
    state = State(inst, x0, x0, locks, protected, None)
    rows = []
    for k, dis in enumerate(hs.disruptions):
        row, new_state = run_isolated(state, dis, params, task["method"],
                                      mem_limit_mb=task["mem_limit_mb"])
        state = advance_state(state, dis, new_state)
        row.update({f: task[f] for f in KEY}, step=k, time_limit=tl, lessons=len(inst.lessons))
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
        with open_csv(out, FIELDS) as (fh, w):
            for i, t in enumerate(todo, 1):
                t0 = time.perf_counter()
                rows = run_sequence(t)
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
