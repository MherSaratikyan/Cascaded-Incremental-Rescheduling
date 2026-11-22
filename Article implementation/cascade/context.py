"""Everything that is fixed during step k (assumption 'E' of Section 5).

A :class:`StepContext` is computed once per request in step 1 of Table 1:
Omega^(k+1), the statuses S^O / S^P0 / S^P<= with their bounds beta_s^k,
locks Lambda^T / Lambda^R / Lambda^A closed over linked components,
the baselines G_ud(x^(k)), D_0^(k+1), D^(k+1), F_0 and the budget ladder B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import evaluate as ev
from .model import Instance, Timetable, link_components
from .params import Params

LOCK_TYPES = ("T", "R", "A")


@dataclass
class MoveRequest:
    """Request to move lesson e to start slot t* (optionally to a given room)."""
    lesson: str
    start: int
    room: Optional[str] = None
    voluntary: bool = True


def _merge_locks(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a is None:
        return b
    if b is None or a == b:
        return a
    return "A"          # a time lock and a room lock together fix both


@dataclass
class StepContext:
    inst: Instance                       # instance after the disruption (Omega^(k+1))
    xk: Timetable                        # x^(k)
    x0: Timetable                        # x^(0)
    params: Params
    locks: dict[str, str]                # Lambda^T/R/A, closed over components
    status: dict[str, str]               # s -> "P0" | "PLE"   (S^P0 / S^P<=)
    beta: dict[str, int]                 # numerator bound of (4)
    ordinary: list[str]                  # S^O
    review: list[str]                    # P0 constraints demoted to P<= (sent for review)
    gap_protected: set[str]              # U^P
    omega: dict[str, list[tuple[int, str]]]
    comp: dict[str, frozenset[str]]
    num_k: dict[str, int]                # num_s(x^(k)) after the disruption (v_s^+)
    gaps_k: dict[tuple[str, int], int]   # G_ud(x^(k))
    d0: set[str]
    d: set[str]
    request: Optional[MoveRequest]
    delta: int
    ladder: list[int]
    f0: set[str] = field(default_factory=set)

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, inst_before: Instance, inst_after: Instance, xk: Timetable, x0: Timetable,
              params: Params, locks: Optional[dict[str, str]] = None,
              protected: Optional[dict[str, str]] = None,
              gap_protected: Optional[set[str]] = None,
              request: Optional[MoveRequest] = None) -> "StepContext":
        locks = dict(locks or {})
        protected = dict(protected or {})
        inst = inst_after
        comp = link_components(inst.lessons, inst.links)

        # locks over whole linked components
        closed: dict[str, str] = {}
        for e, lk in locks.items():
            if lk not in LOCK_TYPES:
                raise ValueError(f"lock of {e}: {lk!r} not in {LOCK_TYPES}")
            for e2 in comp[e]:
                closed[e2] = _merge_locks(closed.get(e2), lk)

        # statuses and beta (4): beta = 0 for P0, v_s(x^(k)) otherwise
        status: dict[str, str] = {}
        beta: dict[str, int] = {}
        review: list[str] = []
        for s, st in protected.items():
            if s not in inst.soft:
                raise ValueError(f"protected constraint {s} unknown")
            if st not in ("P0", "PLE"):
                raise ValueError(f"status of {s}: {st!r} not in ('P0','PLE')")
            v = ev.soft_num(inst_before, xk, inst_before.soft[s])
            if st == "P0" and v > 0:
                st = "PLE"              # already violated: would give a false infeasibility
                review.append(s)
            status[s] = st
            beta[s] = 0 if st == "P0" else v
        ordinary = [s for s in inst.soft if s not in status]

        if gap_protected is None:
            mode = params.gap_protect
            gap_protected = set()
            if mode in ("groups", "both"):
                gap_protected |= set(inst.groups)
            if mode in ("lecturers", "both"):
                gap_protected |= set(inst.lecturers)

        req = (request.lesson, request.start, request.room) if request else None
        om = ev.all_omegas(inst, req)
        d0 = {e for e in inst.lessons if e not in xk or xk[e] not in set(om[e])}
        hard = ev.hard_violations(inst, xk, om)
        d = d0 | hard.lessons

        delta = 0 if (request is not None and request.voluntary) else params.delta_external

        total = params.w_time * len(inst.lessons)
        ladder: list[int] = []
        b = max(1, params.first_budget_factor * params.w_time)
        while b < total:
            ladder.append(b)
            b *= 2
        ladder.append(total)

        ctx = cls(inst=inst, xk=dict(xk), x0=dict(x0), params=params, locks=closed,
                  status=status, beta=beta, ordinary=ordinary, review=review,
                  gap_protected=set(gap_protected), omega=om, comp=comp,
                  num_k={s: ev.soft_num(inst, xk, c) for s, c in inst.soft.items()},
                  gaps_k=ev.gaps(inst, xk), d0=d0, d=d, request=request, delta=delta,
                  ladder=ladder)
        seeds = set(d)
        if request is not None:
            seeds.add(request.lesson)
        ctx.f0 = ctx.closure(seeds) - ctx.full_locked
        return ctx

    # -------------------------------------------------------------- helpers
    def gamma(self, sid: str) -> list[str]:
        """Gamma(s), cached."""
        cache = self.__dict__.setdefault("_gamma_cache", {})
        if sid not in cache:
            cache[sid] = ev.gamma(self.inst, self.inst.soft[sid])
        return cache[sid]

    @property
    def full_locked(self) -> set[str]:
        return {e for e, lk in self.locks.items() if lk == "A"}

    @property
    def all_free(self) -> set[str]:
        """E \\ Lambda^A."""
        return set(self.inst.lessons) - self.full_locked

    def closure(self, lessons) -> set[str]:
        out: set[str] = set()
        for e in lessons:
            out |= self.comp[e]
        return out

    @property
    def lock_conflicts(self) -> set[str]:
        """D_0^(k+1) intersected with Lambda^A (step 2)."""
        return self.d0 & self.full_locked

    @property
    def theta_q(self) -> float:
        return self.params.theta_q_per_soft * max(1, len(self.ordinary))

    def protected_violations(self, tt: Timetable) -> list[str]:
        return sorted(s for s in self.status
                      if ev.soft_num(self.inst, tt, self.inst.soft[s]) > self.beta[s])

    def lock_violations(self, tt: Timetable) -> list[str]:
        bad = []
        for e, lk in self.locks.items():
            (t, r), (tk, rk) = tt[e], self.xk[e]
            if (lk == "T" and t != tk) or (lk == "R" and r != rk) or (lk == "A" and (t, r) != (tk, rk)):
                bad.append(e)
        return sorted(bad)

    def gap_protection_violations(self, tt: Timetable) -> list[str]:
        g = ev.gaps(self.inst, tt, self.gap_protected)
        return sorted({u for (u, d), v in g.items() if v > self.gaps_k.get((u, d), 0)})
