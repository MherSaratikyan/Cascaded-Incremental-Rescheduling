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

from cascade import TimetableStore, handle_request  # noqa: E402
from cascade import evaluate as ev  # noqa: E402
from cascade.baselines import METHODS, run_method  # noqa: E402
from cascade.params import Params  # noqa: E402
from experiments.generator import (generate_sequence, load_or_generate,  # noqa: E402
                                   sample_regulation)

FIELDS = ["size", "seed", "method", "time_limit", "U_P", "step", "kind", "voluntary", "outcome",
          "seconds", "iterations", "F_ratio", "P_k", "P_0", "G", "dG_plus", "Q_O", "Q_O_k",
          "protected_violations", "lock_violations", "gap_protection_violations",
          "hard_violations", "changed", "lecturers_changed", "groups_changed",
          "dG_lecturers", "dG_groups", "extra_expansion", "requires_approval",
          "mus_size", "mcs_count", "mcs_verified", "attempts"]
KEY = ("size", "seed", "method", "time_limit", "U_P")


def split_new_gaps(ctx, tt):
    g = ev.gaps(ctx.inst, tt)
    out = {"lecturer": 0.0, "group": 0.0}
    for (u, d), v in g.items():
        part = ctx.inst.participants[u]
        out[part.kind] += part.weight * max(0, v - ctx.gaps_k.get((u, d), 0))
    return out


def run_task(task: dict) -> list[dict]:
    inst, x0 = load_or_generate(task["size"], task["seed"], Path(task["cache"]))
    seq = generate_sequence(inst, task["seed"], steps=task.get("steps"))
    protected, locks = sample_regulation(inst, task["seed"], task["rho"], task["lam"])
    tl = task["time_limit"]
    params = Params(time_limit=tl, explain_time_limit=min(10.0, tl), workers=task["solver_workers"],
                    gap_protect=task["U_P"], seed=task["seed"])
    store = TimetableStore(inst, x0, x0, locks, protected, None)
    method = task["method"]
    rows = []
    for k, dis in enumerate(seq):
        h = handle_request(store, dis, params, solver=lambda c: run_method(method, c))
        r, m = h.result, h.result.metrics
        row = {f: task.get(f) for f in KEY}
        row.update(step=k, kind=dis.kind, voluntary=dis.kind == "move_request",
                   outcome=r.outcome.value, seconds=round(r.seconds, 4),
                   iterations=len(r.trace), extra_expansion=r.extra_expansion,
                   requires_approval=r.requires_approval, mus_size=len(r.mus or []),
                   mcs_count=len(r.mcs), mcs_verified=sum(c.verified for c in r.mcs),
                   attempts=h.attempts)
        for f in ("F_ratio", "P_k", "P_0", "G", "dG_plus", "Q_O", "Q_O_k", "protected_violations",
                  "lock_violations", "gap_protection_violations", "hard_violations"):
            v = m.get(f)
            row[f] = round(v, 6) if isinstance(v, float) else v
        changed = m.get("changed", [])
        row["changed"] = len(changed)
        lessons = h.ctx.inst.lessons
        row["lecturers_changed"] = len({l for e in changed for l in lessons[e].lecturers})
        row["groups_changed"] = len({g for e in changed for g in lessons[e].groups})
        if r.timetable is not None:
            sg = split_new_gaps(h.ctx, r.timetable)
            row["dG_lecturers"], row["dG_groups"] = sg["lecturer"], sg["group"]
        rows.append(row)
    return rows


def _done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {tuple(str(row[k]) for k in KEY) for row in csv.DictReader(fh)}


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

    done = _done_keys(out)
    tasks = []
    for size in a.sizes:
        for seed in seeds:
            for tl in a.time_limits:
                for up in a.up:
                    for method in a.methods:
                        task = dict(size=size, seed=seed, method=method, time_limit=tl, U_P=up,
                                    rho=a.rho, lam=a.lam, steps=a.steps, cache=a.cache,
                                    solver_workers=a.solver_workers)
                        if tuple(str(task[k]) for k in KEY) not in done:
                            tasks.append(task)
    print(f"{len(tasks)} sequences to run on {a.jobs} processes -> {out}", flush=True)
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
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
                w.writerows(rows)
                fh.flush()
                solved = sum(r["outcome"] == "solved" for r in rows)
                print(f"[{i}/{len(tasks)}] {t['size']}-{t['seed']} {t['method']} tl={t['time_limit']} "
                      f"U_P={t['U_P']}: {solved}/{len(rows)} solved", flush=True)


if __name__ == "__main__":
    main()
