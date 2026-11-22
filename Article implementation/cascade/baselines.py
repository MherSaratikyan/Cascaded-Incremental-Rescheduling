"""Baselines of Section 7 and a single dispatcher ``run_method``.

B1 local check [7]       -> Strategy "B1"  (only F_0, no chain)
B2 hard-only chain       -> Strategy "B2"
B3 full re-optimisation  -> Strategy "B3"  (minimal perturbation, hard constraints only)
B4 regulation-limited full re-optimisation -> Strategy "B4"
B5 unweighted expansion F_{j+1} = cl(F_j u N_j), no core -> Strategy "B5"
B6 perturbation-limited quality heuristic [4]  -> :func:`b6_local_search`
   NOTE: an *approximation* of Akkan, Guelcue & Kus (2021) -- simulated
   annealing with a perturbation limit on the same model, not their algorithm.
B7 the proposed method   -> Strategy "B7"
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from typing import Optional

from . import evaluate as ev
from .context import StepContext
from .explain import Deadline
from .rescheduler import STRATEGIES, compute_metrics, reschedule
from .result import Outcome, Result

METHODS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7")


def run_method(name: str, ctx: StepContext) -> Result:
    if name == "B6":
        return b6_local_search(ctx)
    return reschedule(ctx, STRATEGIES[name])


class _State:
    """Incrementally evaluated timetable for the local search."""

    def __init__(self, ctx: StepContext, tt):
        self.ctx = ctx
        inst = ctx.inst
        self.inst = inst
        self.tt = dict(tt)
        self.room: Counter = Counter()
        self.part: Counter = Counter()
        self.res: Counter = Counter()
        self.valid = {e: set(ctx.omega[e]) for e in inst.lessons}
        self.links_of: dict[str, list] = {}
        for a, b in inst.links:
            self.links_of.setdefault(a, []).append((a, b))
            self.links_of.setdefault(b, []).append((a, b))
        self.soft_of: dict[str, list[str]] = {e: [] for e in inst.lessons}
        for sid in inst.soft:
            for e in ctx.gamma(sid):
                self.soft_of[e].append(sid)
        for e, a in self.tt.items():
            self._cells(e, a, +1)
        self.hard = ev.hard_violations(inst, self.tt, ctx.omega).units
        self.num = {s: ev.soft_num(inst, self.tt, c) for s, c in inst.soft.items()}
        self.gap = ev.gaps(inst, self.tt)
        self.coef = {s: ev.q_coef(inst, c, ctx.params.q_scale) for s, c in inst.soft.items()}
        p = ctx.params
        self.pk = ev.perturbation(self.tt, ctx.xk, p.w_time, p.w_room)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _bump(counter: Counter, key, sign: int, cap: int) -> int:
        """Change a cell count; return the exact change of overload units."""
        if sign < 0:
            d = -1 if counter[key] > cap else 0
            counter[key] -= 1
        else:
            counter[key] += 1
            d = 1 if counter[key] > cap else 0
        return d

    def _cells(self, e, a, sign) -> int:
        l = self.inst.lessons[e]
        t, r = a
        d = 0
        for tau in range(t, t + l.duration):
            d += self._bump(self.room, (r, tau), sign, 1)
            for u in l.participants:
                d += self._bump(self.part, (u, tau), sign, 1)
            for q in l.resources:
                d += self._bump(self.res, (q, tau), sign, self.inst.resources[q].capacity)
        return d

    def _hard_local(self, e) -> int:
        """Validity of e's assignment and its link mismatches."""
        h = 0 if self.tt[e] in self.valid[e] else 1
        for a, b in self.links_of.get(e, ()):
            h += self.tt[a][0] != self.tt[b][0]
        return h

    def _day_gaps(self, u, d) -> int:
        H = self.inst.hours
        busy = [self.part[(u, (d - 1) * H + p - 1)] > 0 for p in range(1, H + 1)]
        if not any(busy):
            return 0
        f = busy.index(True)
        l = H - 1 - busy[::-1].index(True)
        return sum(1 for i in range(f + 1, l) if not busy[i] and not self.inst.is_break(d, i + 1))

    def cost(self) -> float:
        ctx = self.ctx
        prot = sum(max(0, self.num[s] - ctx.beta[s]) for s in ctx.status)
        gp = sum(max(0, self.gap[(u, d)] - ctx.gaps_k.get((u, d), 0))
                 for (u, d) in self.gap if u in ctx.gap_protected)
        dg = sum(self.inst.participants[u].weight * max(0, v - ctx.gaps_k.get((u, d), 0))
                 for (u, d), v in self.gap.items())
        q = sum(self.coef[s] * self.num[s] for s in ctx.ordinary)
        return 1e6 * self.hard + 1e4 * (prot + gp) + 100.0 * dg + q / 10.0 + self.pk

    def move(self, e, new):
        """Apply a move and return the old assignment (for undo)."""
        old = self.tt[e]
        if old == new:
            return old
        inst, p = self.inst, self.ctx.params
        # hard: exact incremental delta (cell overloads, validity, links)
        before = self._hard_local(e)
        dh = self._cells(e, old, -1)
        self.tt[e] = new
        dh += self._cells(e, new, +1)
        self.hard += dh + self._hard_local(e) - before
        for s in self.soft_of[e]:
            self.num[s] = ev.soft_num(inst, self.tt, inst.soft[s])
        days = {old[0] // inst.hours + 1, new[0] // inst.hours + 1}
        for u in inst.lessons[e].participants:
            for d in days:
                self.gap[(u, d)] = self._day_gaps(u, d)
        ref = self.ctx.xk.get(e)
        if ref is not None:
            def pe(a):
                return p.w_time if a[0] != ref[0] else (p.w_room if a[1] != ref[1] else 0)
            self.pk += pe(new) - pe(old)
        return old

    def exact_hard(self) -> int:
        self.hard = ev.hard_violations(self.inst, self.tt, self.ctx.omega).units
        return self.hard


def b6_local_search(ctx: StepContext) -> Result:
    p = ctx.params
    t_start = time.perf_counter()
    dl = Deadline(p.time_limit)
    rng = random.Random(p.seed)
    inst = ctx.inst
    res = Result(outcome=Outcome.ALGORITHMIC_FAILURE, method="B6", review=list(ctx.review))
    res.notes.append("B6 is an approximation of [4] (simulated annealing with a perturbation limit)")

    opts: dict[str, list] = {}
    for e in inst.lessons:
        lk = ctx.locks.get(e)
        tk, rk = ctx.xk[e]
        o = ctx.omega[e]
        if lk == "A":
            continue
        if lk == "T":
            o = [a for a in o if a[0] == tk]
        elif lk == "R":
            o = [a for a in o if a[1] == rk]
        if o:
            opts[e] = o
    movable = sorted(opts)
    limit = p.w_time * (len(ctx.f0) + p.b6_extra_moves) + ctx.delta

    s = _State(ctx, ctx.xk)
    must = [e for e in sorted(ctx.f0) if e in opts]
    # greedy start: put displaced lessons on their least-cost option
    for e in must:
        best, best_c = s.tt[e], s.cost()
        for a in opts[e]:
            old = s.move(e, a)
            c = s.cost()
            if c < best_c and s.pk <= limit:
                best, best_c = a, c
            s.move(e, old)
        s.move(e, best)
    cur = s.cost()
    best_tt, best_cost, best_hard = dict(s.tt), cur, s.hard
    temp, it, since = 50.0, 0, 0
    while movable and not dl.expired:
        it += 1
        if since > 3000 and best_hard == 0:
            break
        pool = must if (must and rng.random() < 0.5) else movable
        e = rng.choice(pool)
        a = rng.choice(opts[e])
        undo = [(e, s.move(e, a))]
        if rng.random() < 0.5:                 # ejection: move one lesson now clashing in the room
            l = inst.lessons[e]
            clash = [j for j in movable if j != e and s.tt[j][1] == a[1]
                     and set(range(s.tt[j][0], s.tt[j][0] + inst.lessons[j].duration))
                     & set(range(a[0], a[0] + l.duration))]
            if clash:
                j = rng.choice(clash)
                undo.append((j, s.move(j, rng.choice(opts[j]))))
        new = s.cost()
        ok = s.pk <= limit and (new <= cur or rng.random() < math.exp(-(new - cur) / max(temp, 1e-6)))
        if ok:
            cur = new
            if new < best_cost - 1e-9:
                best_tt, best_cost, best_hard = dict(s.tt), new, s.hard
                since = 0
            else:
                since += 1
        else:
            for j, old in reversed(undo):
                s.move(j, old)
            since += 1
        temp = max(0.5, temp * 0.999)

    hard = ev.hard_violations(inst, best_tt, ctx.omega).units
    res.seconds = time.perf_counter() - t_start
    if hard == 0:
        res.outcome = Outcome.SOLVED
        res.timetable = best_tt
    res.metrics = compute_metrics(ctx, res.timetable, len(movable))
    res.metrics["iterations"] = it
    return res
