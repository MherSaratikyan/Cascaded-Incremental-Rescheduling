"""Explanations of regulatory infeasibility: MUS [8] and verified MCS [11].

Relaxable items are the regulatory ones only -- locks, protected soft
constraints and gap protections; hard constraints (1)-(2) are physical.
"""
from __future__ import annotations

import time
from typing import Optional

from .context import StepContext
from .cpsat_model import Lin, SubModel
from .result import Correction


class Deadline:
    def __init__(self, seconds: float):
        self.end = time.perf_counter() + seconds

    def remaining(self) -> float:
        return max(0.0, self.end - time.perf_counter())

    @property
    def expired(self) -> bool:
        return self.remaining() <= 0.01


def minimize_core(sub: SubModel, core: list, dl: Deadline, cap: float = 2.0) -> tuple[list, bool]:
    """Deletion-based minimisation of an assumption core.  Returns (core, proven_minimal)."""
    keep = list(core)
    minimal = True
    i = 0
    while i < len(keep):
        if dl.expired:
            return keep, False
        trial = keep[:i] + keep[i + 1:]
        r = sub.check(min(cap, dl.remaining()), assume=trial)
        if r.infeasible:
            sub_core = set(r.core) if r.core is not None else set(trial)
            keep = [c for c in trial if c in sub_core]
        elif r.feasible:
            i += 1
        else:
            minimal = False
            i += 1
    return keep, minimal


def regulation_model(ctx: StepContext, relaxed=(), assumptions=True) -> SubModel:
    """F = E (fully locked lessons included as variables with lock literals),
    budget dropped (B >= sum_e w_e^T makes it redundant, Claim 2)."""
    return SubModel(ctx, ctx.inst.lessons, budget=None, relaxed=relaxed, assumptions=assumptions)


def counting_proof(ctx: StepContext) -> Optional[str]:
    """Exact sufficient condition for physical infeasibility: a participant
    whose lessons need more hours than the slots any of them can cover.
    (Participants attend one lesson at a time, so the lessons need disjoint hours.)"""
    inst = ctx.inst
    lessons_of: dict[str, list[str]] = {}
    for e, l in inst.lessons.items():
        for u in l.participants:
            lessons_of.setdefault(u, []).append(e)
    for u, es in sorted(lessons_of.items()):
        need = sum(inst.lessons[e].duration for e in es)
        avail: set[int] = set()
        for e in es:
            d = inst.lessons[e].duration
            for t in {t for t, _ in ctx.omega[e]}:
                avail.update(range(t, t + d))
        if need > len(avail):
            return f"{u} needs {need} hours but its lessons can only use {len(avail)}"
    return None


def physical_check(ctx: StepContext, seconds: float):
    sub = SubModel(ctx, ctx.inst.lessons, protected=False, locks=False, gap_protection=False,
                   assumptions=False)
    return sub.check(seconds, hint=ctx.xk)


def find_mus(ctx: StepContext, dl: Deadline) -> tuple[Optional[list], bool]:
    sub = regulation_model(ctx)
    r = sub.check(dl.remaining(), hint=ctx.xk)
    if not r.infeasible:
        return None, False
    core = r.core or []
    return minimize_core(sub, core, dl, cap=max(1.0, dl.remaining() / 4))


def find_mcs(ctx: StepContext, dl: Deadline) -> list[Correction]:
    """Minimum-cardinality correction sets, enumerated by blocking, each one
    verified by an independent re-solve of the relaxed regulation model."""
    p = ctx.params
    sub = regulation_model(ctx)          # literals stay free (no assumptions posted)
    lits = sub.lits
    if not lits:
        return []
    items = list(lits)
    relaxed_count = Lin([lits[i] for i in items], [-1] * len(items), len(items))
    out: list[Correction] = []
    for _ in range(p.mcs_max_count):
        if dl.expired:
            break
        r = sub.minimize(relaxed_count, dl.remaining(), hint=ctx.xk)
        if not r.feasible:
            break
        relaxed = [i for i in items if not r.lit_values.get(i, True)]
        if not relaxed or len(relaxed) > p.mcs_max_size:
            break
        if r.status != "OPTIMAL" and out and len(relaxed) > len(out[-1].items) + p.mcs_max_size:
            break
        check = regulation_model(ctx, relaxed=relaxed, assumptions=False)
        v = check.check(dl.remaining(), hint=r.timetable)
        out.append(Correction(items=relaxed, verified=v.feasible,
                              candidate=v.timetable if v.feasible else r.timetable))
        sub.add_clause(relaxed)                         # block this set and its supersets
    return out
