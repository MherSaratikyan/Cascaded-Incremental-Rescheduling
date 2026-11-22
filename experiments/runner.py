"""Experiment protocol of Section 7 on synthetic instances.

Every (size, seed, method, time limit, U^P variant) runs one disruption
sequence of 5-20 steps on its own trajectory (each method commits its own
x^(k+1)); one CSV row is written per request.

    python -m experiments.runner                      # S and M, 20 seeds, all methods, 30 s
    python -m experiments.runner --quick              # smoke run (S, 2 seeds, 5 steps, 10 s)
    python -m experiments.runner --sizes L --time-limits 10 30 60 --up groups lecturers both

Runs resume: rows already in --out are skipped.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade.baselines import METHODS  # noqa: E402
from cascade.params import Params  # noqa: E402
from experiments.generator import (generate_sequence, load_or_generate,  # noqa: E402
                                   sample_regulation)
from experiments.measure import (ALL_METRICS, State, advance_state, run_isolated,  # noqa: E402
                                 run_request)

KEY = ("size", "seed", "method", "time_limit", "U_P")
FIELDS = list(KEY) + ["step"] + ALL_METRICS


def run_task(task: dict) -> list[dict]:
    """One disruption sequence on its own trajectory; each request runs in a
    fresh process (``isolated``) unless ``task["in_process"]``."""
    inst, x0 = load_or_generate(task["size"], task["seed"], Path(task["cache"]))
    seq = generate_sequence(inst, task["seed"], steps=task.get("steps"))
    protected, locks = sample_regulation(inst, task["seed"], task["rho"], task["lam"])
    tl = task["time_limit"]
    params = Params(time_limit=tl, explain_time_limit=min(10.0, tl), workers=task["solver_workers"],
                    gap_protect=task["U_P"], seed=task["seed"])
    state = State(inst, x0, x0, locks, protected, None)
    rows = []
    for k, dis in enumerate(seq):
        if task.get("in_process"):
            row, new_state = run_request(state, dis, params, task["method"])
        else:
            row, new_state = run_isolated(state, dis, params, task["method"],
                                          mem_limit_mb=task.get("mem_limit_mb", 3072))
        state = advance_state(state, dis, new_state)
        row.update({f: task.get(f) for f in KEY}, step=k)
        rows.append(row)
    return rows


def _done_keys(path: Path, fields: list[str], key=KEY) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames != fields:          # old format: will be set aside, nothing done
            return set()
        return {tuple(str(row[k]) for k in key) for row in rd}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=["S", "M"])
    ap.add_argument("--seeds", type=int, default=20, help="seeds 1..N")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--time-limits", nargs="+", type=float, default=[30.0])
    ap.add_argument("--up", nargs="+", default=["groups"], choices=["groups", "lecturers", "both"])
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=None, help="fixed sequence length (default 5-20)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4),
                    help="parallel sequences (default: logical CPUs / 4)")
    ap.add_argument("--solver-workers", type=int, default=2)
    ap.add_argument("--cache", default=str(ROOT / "experiments" / "instances"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "run.csv"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--in-process", action="store_true",
                    help="run requests in this process (faster, but peak memory is not per request)")
    ap.add_argument("--mem-limit-mb", type=float, default=3072,
                    help="memory guard on committed private bytes of a request process")
    a = ap.parse_args(argv)
    if a.quick:
        a.sizes, a.seeds, a.steps, a.time_limits = ["S"], 2, 5, [10.0]
        a.out = str(ROOT / "experiments" / "results" / "quick.csv")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, a.seeds + 1))
    print(f"generating / loading {len(a.sizes) * len(seeds)} instances ...", flush=True)
    for size in a.sizes:
        for seed in seeds:
            t = time.perf_counter()
            inst, _ = load_or_generate(size, seed, Path(a.cache))
            print(f"  {size}-{seed}: {len(inst.lessons)} lessons ({time.perf_counter() - t:.1f}s)", flush=True)

    done = _done_keys(out, FIELDS)
    tasks = []
    for size in a.sizes:
        for seed in seeds:
            for tl in a.time_limits:
                for up in a.up:
                    for method in a.methods:
                        task = dict(size=size, seed=seed, method=method, time_limit=tl, U_P=up,
                                    rho=a.rho, lam=a.lam, steps=a.steps, cache=a.cache,
                                    solver_workers=a.solver_workers, in_process=a.in_process,
                                    mem_limit_mb=a.mem_limit_mb)
                        if tuple(str(task[k]) for k in KEY) not in done:
                            tasks.append(task)
    print(f"{len(tasks)} sequences to run on {a.jobs} processes -> {out}", flush=True)
    with open_csv(out, FIELDS) as (fh, w):
        def report(i, t, rows):
            w.writerows(rows)
            fh.flush()
            solved = sum(r["outcome"] == "solved" for r in rows)
            print(f"[{i}/{len(tasks)}] {t['size']}-{t['seed']} {t['method']} tl={t['time_limit']} "
                  f"U_P={t['U_P']}: {solved}/{len(rows)} solved", flush=True)

        if a.jobs <= 1:
            for i, t in enumerate(tasks, 1):
                report(i, t, run_task(t))
            return
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            futs = {pool.submit(run_task, t): t for t in tasks}
            for i, f in enumerate(as_completed(futs), 1):
                t = futs[f]
                try:
                    rows = f.result()
                except Exception as exc:          # keep going; report the failed sequence
                    print(f"[{i}/{len(tasks)}] FAILED {t['size']}-{t['seed']} {t['method']}: {exc!r}",
                          flush=True)
                    continue
                report(i, t, rows)


class open_csv:
    """Append to a results CSV; an older file with a different header is
    renamed to ``*.old.csv`` first so columns never get mixed."""

    def __init__(self, path: Path, fields: list[str]):
        self.path, self.fields = path, fields

    def __enter__(self):
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), [])
            if header != self.fields:
                self.path.replace(self.path.with_suffix(".old.csv"))
        new = not self.path.exists()
        self.fh = self.path.open("a", newline="", encoding="utf-8")
        w = csv.DictWriter(self.fh, fieldnames=self.fields, extrasaction="ignore")
        if new:
            w.writeheader()
        return self.fh, w

    def __exit__(self, *exc):
        self.fh.close()


if __name__ == "__main__":
    main()
