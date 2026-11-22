"""Dependency graph H = (E, A, W) and the active-set expansion rule (10)-(12).

C^H_ij   share of i's admissible assignments that clash with a(j; x^(k))
C^alt_ij overlap of resource-time cells Theta_i, Theta_j, times scarcity
C^P_ij   shared protected constraints, damped by tension theta_s
W_ij  = alpha_H C^H + alpha_A C^alt + alpha_P C^P
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from . import evaluate as ev
from .context import StepContext


class DependencyGraph:
    def __init__(self, ctx: StepContext):
        self.ctx = ctx
        inst = ctx.inst
        T = inst.num_slots
        self.T = T
        # resource index for the cell bitsets of Theta_e: rooms, then participants
        names = [("room", r) for r in sorted(inst.rooms)] + [("part", u) for u in sorted(inst.participants)]
        self.res_index = {n: i for i, n in enumerate(names)}
        # occupancy of x^(k): cell -> lessons
        self.occ: dict[tuple, set[str]] = defaultdict(set)
        for e, (t, r) in ctx.xk.items():
            l = inst.lessons[e]
            for tau in inst.cells(l, t):
                self.occ[("room", r, tau)].add(e)
                for u in l.participants:
                    self.occ[("part", u, tau)].add(e)
                for q in l.resources:
                    if inst.resources[q].capacity == 1:
                        self.occ[("res", q, tau)].add(e)
        self._theta: dict[str, int] = {}
        self._theta_n: dict[str, int] = {}
        self._ch: dict[str, dict[str, float]] = {}
        # protected-constraint contributions (11)
        self.prot_of: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for sid in ctx.status:
            s = inst.soft[sid]
            gam = ctx.gamma(sid)
            if len(gam) < 2:
                continue
            den = ev.soft_den(inst, s)
            slack = max(0.0, ctx.beta[sid] / den - ctx.num_k[sid] / den)
            theta = 1.0 / (1.0 + slack)
            factor = theta * s.weight / (len(gam) - 1)
            for e in gam:
                self.prot_of[e].append((sid, factor))

    # ------------------------------------------------------------- C^H (10)
    def blocking(self, i: str) -> dict[str, float]:
        if i in self._ch:
            return self._ch[i]
        inst, om = self.ctx.inst, self.ctx.omega[i]
        l = inst.lessons[i]
        counts: dict[str, int] = defaultdict(int)
        for t, r in om:
            blockers: set[str] = set()
            for tau in inst.cells(l, t):
                blockers |= self.occ.get(("room", r, tau), set())
                for u in l.participants:
                    blockers |= self.occ.get(("part", u, tau), set())
                for q in l.resources:
                    blockers |= self.occ.get(("res", q, tau), set())
            blockers.discard(i)
            for j in blockers:
                counts[j] += 1
        n = max(1, len(om))
        self._ch[i] = {j: c / n for j, c in counts.items()}
        return self._ch[i]

    # ----------------------------------------------------------- C^alt (10)
    def theta(self, e: str) -> int:
        """Theta_e as a Python-int bitset over (resource, slot) cells."""
        if e not in self._theta:
            inst = self.ctx.inst
            l = inst.lessons[e]
            bits = 0
            T = self.T
            parts = [self.res_index[("part", u)] for u in l.participants]
            seen_t: set[int] = set()
            for t, r in self.ctx.omega[e]:
                ri = self.res_index[("room", r)]
                cs = inst.cells(l, t)
                for tau in cs:
                    bits |= 1 << (ri * T + tau)
                if t not in seen_t:
                    seen_t.add(t)
                    for pi in parts:
                        for tau in cs:
                            bits |= 1 << (pi * T + tau)
            self._theta[e] = bits
            self._theta_n[e] = bits.bit_count()
        return self._theta[e]

    def c_alt(self, i: str, j: str) -> float:
        ti, tj = self.theta(i), self.theta(j)
        ni, nj = self._theta_n[i], self._theta_n[j]
        if not ni or not nj:
            return 0.0
        inter = (ti & tj).bit_count()
        if not inter:
            return 0.0
        oi, oj = len(self.ctx.omega[i]), len(self.ctx.omega[j])
        return inter / min(ni, nj) / max(1, min(oi, oj))

    # ------------------------------------------------------------ C^P (11)
    def c_p(self, i: str, j: str) -> float:
        gj = {sid for sid, _ in self.prot_of.get(j, ())}
        return sum(f for sid, f in self.prot_of.get(i, ()) if sid in gj)

    # ---------------------------------------------------------- neighbours
    def neighbours(self, F: Iterable[str], exclude: Optional[set[str]] = None
                   ) -> dict[str, tuple[float, float, int]]:
        """N_j with the ranking key sigma_j(h) = (max_i C^H_ih, sum_i W_ih, -|Omega_h|)."""
        ctx, p = self.ctx, self.ctx.params
        F = set(F)
        excl = F | ctx.full_locked | (exclude or set())
        max_ch: dict[str, float] = defaultdict(float)
        sum_w: dict[str, float] = defaultdict(float)
        candidates = [h for h in ctx.inst.lessons if h not in excl]
        for i in F:
            ch = self.blocking(i)
            for h in candidates:
                w = 0.0
                c = ch.get(h, 0.0)
                if c:
                    w += p.alpha_h * c
                    if c > max_ch[h]:
                        max_ch[h] = c
                if p.alpha_alt:
                    w += p.alpha_alt * self.c_alt(i, h)
                if p.alpha_p and self.prot_of.get(h):
                    w += p.alpha_p * self.c_p(i, h)
                if w > 0:
                    sum_w[h] += w
        return {h: (max_ch[h], sum_w[h], -len(ctx.omega[h])) for h in sum_w}

    @staticmethod
    def ranked(nb: dict[str, tuple[float, float, int]]) -> list[str]:
        return sorted(nb, key=lambda h: (-nb[h][0], -nb[h][1], -nb[h][2], h))

    def top(self, F: Iterable[str], m: int) -> list[str]:
        return self.ranked(self.neighbours(F))[:m]

    def expand(self, F: set[str], core_fixed: Iterable[str], m: int,
               weighted: bool = True) -> tuple[set[str], list[str]]:
        """Rule (12). Returns (F_{j+1}, newly ranked neighbours used)."""
        ctx = self.ctx
        lam_a = ctx.full_locked
        cj = set(core_fixed) - lam_a
        nb = self.neighbours(F)
        if not nb and not cj:
            return ctx.all_free, []
        chosen = self.ranked(nb)[:m] if weighted else list(nb)
        new = ctx.closure(F | cj | set(chosen)) - lam_a
        if new == F:                         # nothing new reachable: take everything
            return ctx.all_free, chosen
        return new, chosen
