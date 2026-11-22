"""CP-SAT encoding of the sub-problem X_j (constraints (1), (2), (4), (5), (9), (13))
and of the objectives P_k, P^0, G, Delta G^+_k, Q_O used in (14).

Lessons are split into
* variable lessons  -- the active set F_j (plus, optionally, a pool of neighbours
  whose fixing a(e;x)=a(e;x^(k)) is posted as an *assumption literal*, so that the
  solver core C_j can point at them), and
* constant lessons  -- everything else, kept at x^(k) exactly as in (13).

Every regulatory element that may be blamed for an infeasibility is guarded by a
literal (``item``):

    ("fix", e)    fixing of a neighbour e to x^(k)         (core C_j)
    ("budget",)   P_k(x) <= B_j                            (13)
    ("prot", s)   v_s(x) <= beta_s for s in S^P            (4)
    ("lock", e)   time / room / full lock of e             (5)
    ("gap", u)    G_ud(x) <= G_ud(x^(k)) for u in U^P       (9)

With ``assumptions=True`` those literals are passed to CP-SAT as assumptions and
``sufficient_assumptions_for_infeasibility`` gives the core; otherwise they are
fixed to 1 (plain hard constraints) or left free (for MCS search).

Linear constraints are written straight into the model proto through the small
:class:`Lin` expression type (much faster than building ``LinearExpr`` trees).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

from ortools.sat.python import cp_model

from . import evaluate as ev
from .context import StepContext
from .model import Timetable

Item = tuple
BIG = 1 << 40

STATUS = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
          cp_model.INFEASIBLE: "INFEASIBLE", cp_model.UNKNOWN: "UNKNOWN",
          cp_model.MODEL_INVALID: "MODEL_INVALID"}


class Lin:
    """Integer linear expression sum(coef_i * var_i) + const over variable indices."""
    __slots__ = ("idx", "coef", "const")

    def __init__(self, idx: Iterable[int] = (), coef: Optional[Iterable[int]] = None, const: int = 0):
        self.idx = list(idx)
        self.coef = list(coef) if coef is not None else [1] * len(self.idx)
        self.const = const

    @property
    def is_const(self) -> bool:
        return not self.idx

    def __add__(self, o: Union["Lin", int]) -> "Lin":
        if isinstance(o, Lin):
            return Lin(self.idx + o.idx, self.coef + o.coef, self.const + o.const)
        return Lin(self.idx, self.coef, self.const + o)

    __radd__ = __add__

    def __mul__(self, k: int) -> "Lin":
        return Lin(self.idx, [c * k for c in self.coef], self.const * k)

    __rmul__ = __mul__

    def __sub__(self, o: Union["Lin", int]) -> "Lin":
        return self + (o * -1 if isinstance(o, Lin) else -o)

    @staticmethod
    def sum(items: Iterable[Union["Lin", int]]) -> "Lin":
        idx, coef, const = [], [], 0
        for it in items:
            if isinstance(it, Lin):
                idx.extend(it.idx)
                coef.extend(it.coef)
                const += it.const
            else:
                const += it
        return Lin(idx, coef, const)


@dataclass
class SolveResult:
    status: str                         # OPTIMAL | FEASIBLE | INFEASIBLE | UNKNOWN
    timetable: Optional[Timetable] = None
    core: Optional[list[Item]] = None   # only for INFEASIBLE under assumptions
    value: Optional[int] = None         # objective value, if any
    bound: Optional[float] = None       # best objective bound (minimisation)
    wall: float = 0.0
    lit_values: dict = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.status in ("OPTIMAL", "FEASIBLE")

    @property
    def infeasible(self) -> bool:
        return self.status == "INFEASIBLE"


class SubModel:
    def __init__(self, ctx: StepContext, var_lessons: Iterable[str], *,
                 assumed_fixed: Iterable[str] = (), budget: Optional[int] = None,
                 cap: Optional[int] = None, protected: bool = True, locks: bool = True,
                 gap_protection: bool = True, relaxed: Iterable[Item] = (),
                 assumptions: bool = True):
        self.ctx = ctx
        inst = ctx.inst
        prm = ctx.params
        self.m = m = cp_model.CpModel()
        self.P = m.proto
        self.assumptions = assumptions
        self.relaxed = set(relaxed)
        self.lits: dict[Item, int] = {}
        self._false_idx = m.new_constant(0).index
        self.var_lessons = set(var_lessons)
        self.const_lessons = [e for e in inst.lessons if e not in self.var_lessons]
        for e in self.const_lessons:
            if e not in ctx.xk:
                raise ValueError(f"constant lesson {e} has no assignment in x^(k)")
        H, D = inst.hours, inst.days

        # ---------------------------------------------------- x_etr and (1)
        self.x: dict[str, list[tuple[int, str, int]]] = {}
        for e in sorted(self.var_lessons):
            vs = []
            for (t, r) in ctx.omega[e]:
                vs.append((t, r, m.new_bool_var("").index))
            self.x[e] = vs
            ct = self.P.constraints.add()
            if vs:
                ct.exactly_one.literals.extend(v for *_, v in vs)
            else:
                ct.bool_or.literals.append(self._false_idx)   # Omega_e empty: infeasible

        # -------------------------------------- cells: rooms, participants, q
        room_cells: dict[tuple, list] = defaultdict(list)
        part_cells: dict[tuple, list] = defaultdict(list)
        res_cells: dict[tuple, list] = defaultdict(list)
        room_c: Counter = Counter()
        self.part_c: Counter = Counter()
        res_c: Counter = Counter()
        for e, vs in self.x.items():
            l = inst.lessons[e]
            parts, ress = l.participants, l.resources
            for t, r, v in vs:
                for tau in inst.cells(l, t):
                    room_cells[(r, tau)].append(v)
                    for u in parts:
                        part_cells[(u, tau)].append(v)
                    for q in ress:
                        res_cells[(q, tau)].append(v)
        for e in self.const_lessons:
            l = inst.lessons[e]
            t, r = ctx.xk[e]
            for tau in inst.cells(l, t):
                room_c[(r, tau)] += 1
                for u in l.participants:
                    self.part_c[(u, tau)] += 1
                for q in l.resources:
                    res_c[(q, tau)] += 1
        self.part_cells = part_cells
        for cell, vs in room_cells.items():
            self._atmost(vs, 1 - room_c[cell])
        for cell, vs in part_cells.items():
            self._atmost(vs, 1 - self.part_c[cell])
        for (q, tau), vs in res_cells.items():
            self._atmost(vs, inst.resources[q].capacity - res_c[(q, tau)])

        # ------------------------------------------------- linked lessons (2)
        for a, b in inst.links:
            ina, inb = a in self.x, b in self.x
            if ina and inb:
                sa, sb = defaultdict(list), defaultdict(list)
                for t, _, v in self.x[a]:
                    sa[t].append(v)
                for t, _, v in self.x[b]:
                    sb[t].append(v)
                for t in set(sa) | set(sb):
                    lin = Lin(sa[t]) - Lin(sb[t])
                    self._le(lin, 0)
                    self._ge(lin, 0)
            elif ina or inb:
                ve, ce = (a, b) if ina else (b, a)
                t0 = ctx.xk[ce][0]
                self._eq(Lin([v for t, _, v in self.x[ve] if t == t0]), 1)

        # ----------------------------------------- fixings of the core pool
        for e in assumed_fixed:
            if e not in self.x:
                continue
            tk, rk = ctx.xk[e]
            self._eq(Lin([v for t, r, v in self.x[e] if (t, r) == (tk, rk)]), 1, ("fix", e))

        # ----------------------------------------------------------- locks (5)
        if locks:
            for e, lk in ctx.locks.items():
                if e not in self.x or ("lock", e) in self.relaxed:
                    continue
                tk, rk = ctx.xk[e]
                if lk == "T":
                    vs = [v for t, _, v in self.x[e] if t == tk]
                elif lk == "R":
                    vs = [v for _, r, v in self.x[e] if r == rk]
                else:
                    vs = [v for t, r, v in self.x[e] if (t, r) == (tk, rk)]
                self._eq(Lin(vs), 1, ("lock", e))

        # -------------------------------------------- perturbation (3), P^0
        wT, wR = prm.w_time, prm.w_room
        self.Pk = self._perturbation(ctx.xk, wT, wR)
        self.P0 = self._perturbation(ctx.x0, wT, wR) + ev.perturbation(
            {e: ctx.xk[e] for e in self.const_lessons}, ctx.x0, wT, wR, self.const_lessons)
        if budget is not None:
            self._le(self.Pk, budget, ("budget",))
        if cap is not None:
            self._le(self.Pk, cap)

        # ----------------------------------------------- soft constraints
        self.soft_expr: dict[str, Lin] = {}
        q_parts: list[Lin] = []
        for sid, s in inst.soft.items():
            expr = self._soft_expr(s)
            self.soft_expr[sid] = expr
            if sid in ctx.status:
                if protected and ("prot", sid) not in self.relaxed:
                    self._le(expr, ctx.beta[sid], ("prot", sid))
            else:
                q_parts.append(expr * ev.q_coef(inst, s, prm.q_scale))
        self.QO = Lin.sum(q_parts)

        # ------------------------------------------------------- gaps (6)-(9)
        days_of: dict[str, set[int]] = defaultdict(set)
        for e, vs in self.x.items():
            ds = set().union(*(inst.days_of(e, t) for t, _, _ in vs)) if vs else set()
            for u in inst.lessons[e].participants:
                days_of[u] |= ds
        const_occ: dict[str, set[int]] = defaultdict(set)
        for (u, tau), c in self.part_c.items():
            if c:
                const_occ[u].add(tau)
        G_parts, dG_parts = [], []
        for u, part in inst.participants.items():
            wu = int(round(part.weight))
            protect = gap_protection and u in ctx.gap_protected and ("gap", u) not in self.relaxed
            for d in range(1, D + 1):
                gk = ctx.gaps_k.get((u, d), 0)
                if d in days_of.get(u, ()):
                    Gud = self._gap_expr(u, d)
                    h = m.new_int_var(0, H, "").index
                    self._ge(Lin([h]) - Gud, -gk)              # h >= G_ud - G_ud^k   (8)
                    G_parts.append(Gud * wu)
                    dG_parts.append(Lin([h], [wu]))
                    if protect:
                        self._le(Gud, gk, ("gap", u))           # (9)
                else:
                    g = _const_day_gaps(inst, const_occ[u], d) if u in days_of else gk
                    G_parts.append(wu * g)
                    dG_parts.append(wu * max(0, g - gk))
                    if protect and g > gk:
                        self._false_under(("gap", u))
        self.G = Lin.sum(G_parts)
        self.dG = Lin.sum(dG_parts)

        if not assumptions:
            for lit in self.lits.values():
                ct = self.P.constraints.add()
                ct.bool_and.literals.append(lit)

    # ================================================================ writers
    def _lit(self, item: Item) -> int:
        if item not in self.lits:
            self.lits[item] = self.m.new_bool_var(repr(item)).index
        return self.lits[item]

    def _false_under(self, item: Optional[Item]) -> None:
        ct = self.P.constraints.add()
        if item is None:
            ct.bool_or.literals.append(self._false_idx)
        else:
            ct.bool_or.literals.append(-self._lit(item) - 1)    # negated literal

    def _linear(self, lin: Lin, lb: int, ub: int, item: Optional[Item]) -> None:
        if lin.is_const:
            if not (lb <= lin.const <= ub):
                self._false_under(item)
            return
        ct = self.P.constraints.add()
        if item is not None:
            ct.enforcement_literal.append(self._lit(item))
        ct.linear.vars.extend(lin.idx)
        ct.linear.coeffs.extend(lin.coef)
        ct.linear.domain.extend([max(-BIG, lb - lin.const), min(BIG, ub - lin.const)])

    def _le(self, lin: Lin, ub: int, item: Optional[Item] = None) -> None:
        self._linear(lin, -BIG, ub, item)

    def _ge(self, lin: Lin, lb: int, item: Optional[Item] = None) -> None:
        self._linear(lin, lb, BIG, item)

    def _eq(self, lin: Lin, v: int, item: Optional[Item] = None) -> None:
        self._linear(lin, v, v, item)

    def _atmost(self, vs: list[int], free: int) -> None:
        if len(vs) > free:
            if free == 1:
                ct = self.P.constraints.add()
                ct.at_most_one.literals.extend(vs)
            else:
                self._le(Lin(vs), free)

    # ============================================================ expressions
    def _perturbation(self, ref: Timetable, wT: int, wR: int) -> Lin:
        idx, coef = [], []
        for e, opts in self.x.items():
            if e not in ref:
                continue
            t0, r0 = ref[e]
            for t, r, v in opts:
                c = wT if t != t0 else (wR if r != r0 else 0)
                if c:
                    idx.append(v)
                    coef.append(c)
        return Lin(idx, coef)

    def _soft_expr(self, s) -> Lin:
        """Integer numerator num_s(x) as a linear expression."""
        inst, ctx = self.ctx.inst, self.ctx
        gam = ctx.gamma(s.id)
        var = [e for e in gam if e in self.x]
        const = [e for e in gam if e not in self.x]
        if not var:
            return Lin(const=ev.soft_num(inst, {e: ctx.xk[e] for e in const}, s))
        idx, coef = [], []
        if s.kind == "pref":
            un = s.unwanted
            c0 = 0
            for e in const:
                c0 += sum(1 for tau in inst.cells(e, ctx.xk[e][0]) if tau in un)
            for e in var:
                for t, _, v in self.x[e]:
                    c = sum(1 for tau in inst.cells(e, t) if tau in un)
                    if c:
                        idx.append(v)
                        coef.append(c)
            return Lin(idx, coef, c0)
        if s.kind == "room":
            c0 = sum(1 for e in const if ctx.xk[e][1] != s.home)
            for e in var:
                idx.extend(v for _, r, v in self.x[e] if r != s.home)
            return Lin(idx, None, c0)
        # load: aux >= hours - cap, aux >= 0 (bounded from below only -> exact)
        c0 = sum(inst.hours_on_day(e, ctx.xk[e][0], s.day) for e in const)
        for e in var:
            for t, _, v in self.x[e]:
                h = inst.hours_on_day(e, t, s.day)
                if h:
                    idx.append(v)
                    coef.append(h)
        aux = self.m.new_int_var(0, inst.hours * len(gam) + 1, "").index
        self._ge(Lin([aux]) - Lin(idx, coef, c0), -s.cap)
        return Lin([aux])

    def _z(self, u: str, tau: int) -> Lin:
        """z_udp as a 0/1 literal (or a constant)."""
        c = self.part_c.get((u, tau), 0)
        vs = self.part_cells.get((u, tau))
        if c or not vs:
            return Lin(const=min(1, c))
        if len(vs) == 1:
            return Lin(vs)
        z = self.m.new_bool_var("").index
        self._eq(Lin([z]) - Lin(vs), 0)
        return Lin([z])

    def _gap_expr(self, u: str, d: int) -> Lin:
        """(6)-(7): b (lesson before), a (lesson after), g (gap) for one u, d."""
        m, inst = self.m, self.ctx.inst
        H = inst.hours
        base = (d - 1) * H
        z = [Lin()] + [self._z(u, base + p - 1) for p in range(1, H + 1)] + [Lin()]
        b = [None] * (H + 2)
        a = [None] * (H + 2)
        for p in range(1, H + 1):
            b[p] = Lin([m.new_bool_var("").index])
            if p > 1:
                self._ge(b[p] - z[p - 1], 0)
                self._ge(b[p] - b[p - 1], 0)
        for p in range(H, 0, -1):
            a[p] = Lin([m.new_bool_var("").index])
            if p < H:
                self._ge(a[p] - z[p + 1], 0)
                self._ge(a[p] - a[p + 1], 0)
        gs = []
        for p in range(1, H + 1):
            if inst.is_break(d, p):
                continue
            g = m.new_bool_var("").index
            # g >= b + a - 1 - z
            self._ge(Lin([g]) - b[p] - a[p] + z[p], -1)
            gs.append(g)
        return Lin(gs)

    # ================================================================ solving
    def _solver(self, time_limit: float) -> cp_model.CpSolver:
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = max(0.05, time_limit)
        # many workers only pay off on large models (start-up cost dominates otherwise)
        small = self.size() < self.ctx.params.single_worker_below
        s.parameters.num_workers = 1 if small else self.ctx.params.workers
        s.parameters.random_seed = self.ctx.params.seed
        # this presolve step is quadratic on timetabling models and dominated run time
        s.parameters.find_big_linear_overlap = False
        if self.size() > self.ctx.params.light_presolve_above:
            s.parameters.cp_model_probing_level = 0
            s.parameters.symmetry_level = 0
        return s

    def _timetable(self, s: cp_model.CpSolver) -> Timetable:
        tt = {e: self.ctx.xk[e] for e in self.const_lessons}
        vals = s.response_proto.solution
        for e, vs in self.x.items():
            for t, r, v in vs:
                if vals[v]:
                    tt[e] = (t, r)
                    break
        return tt

    def hint(self, tt: Optional[Timetable]) -> None:
        self.m.clear_hints()
        if not tt:
            return
        idx, val = [], []
        for e, vs in self.x.items():
            a = tt.get(e)
            for t, r, v in vs:
                idx.append(v)
                val.append(1 if (t, r) == a else 0)
        self.P.solution_hint.vars.extend(idx)
        self.P.solution_hint.values.extend(val)

    def set_objective(self, lin: Lin) -> None:
        self.m.clear_objective()
        if lin.is_const:
            return
        merged: dict[int, int] = defaultdict(int)
        for i, c in zip(lin.idx, lin.coef):
            merged[i] += c
        o = self.P.objective
        keys = [i for i, c in merged.items() if c]
        o.vars.extend(keys)
        o.coeffs.extend(merged[i] for i in keys)
        o.offset = float(lin.const)

    def check(self, time_limit: float, assume: Optional[Iterable[Item]] = None,
              hint: Optional[Timetable] = None) -> SolveResult:
        """Feasibility check.  Under assumptions, an INFEASIBLE answer carries
        the core.  ``assume`` restricts the assumed items (others are relaxed)."""
        m = self.m
        m.clear_objective()
        m.clear_assumptions()
        index: dict[int, Item] = {}
        if self.assumptions:
            items = list(self.lits) if assume is None else [i for i in assume if i in self.lits]
            self.P.assumptions.extend(self.lits[i] for i in items)
            index = {self.lits[i]: i for i in items}
        self.hint(hint)
        s = self._solver(time_limit)
        st = STATUS[s.solve(m)]
        res = SolveResult(status=st, wall=s.wall_time)
        self._record(s)
        if res.feasible:
            res.timetable = self._timetable(s)
        elif st == "INFEASIBLE" and self.assumptions:
            res.core = [index[i] for i in s.sufficient_assumptions_for_infeasibility() if i in index]
        return res

    def minimize(self, lin: Lin, time_limit: float, hint: Optional[Timetable] = None) -> SolveResult:
        m = self.m
        m.clear_assumptions()
        self.set_objective(lin)
        self.hint(hint)
        s = self._solver(time_limit)
        st = STATUS[s.solve(m)]
        res = SolveResult(status=st, wall=s.wall_time)
        self._record(s)
        if res.feasible:
            res.timetable = self._timetable(s)
            res.value = lin.const if lin.is_const else int(round(s.objective_value))
            res.bound = float(lin.const) if lin.is_const else float(s.best_objective_bound)
            sol = s.response_proto.solution
            res.lit_values = {i: bool(sol[l]) for i, l in self.lits.items()}
        return res

    def _record(self, s: cp_model.CpSolver) -> None:
        stats = getattr(self.ctx, "stats", None)
        if stats is not None:
            stats.record(len(self.P.variables), len(self.P.constraints), s.wall_time)

    def bound(self, lin: Lin, value: int) -> None:
        self._le(lin, value)

    def add_clause(self, items: Iterable[Item]) -> None:
        """At least one of the given items' literals must hold."""
        ct = self.P.constraints.add()
        ct.bool_or.literals.extend(self.lits[i] for i in items)

    def size(self) -> int:
        return sum(len(v) for v in self.x.values())


def _const_day_gaps(inst, occ: set[int], d: int) -> int:
    H = inst.hours
    busy = [((d - 1) * H + p - 1) in occ for p in range(1, H + 1)]
    if not any(busy):
        return 0
    first = busy.index(True)
    last = H - 1 - busy[::-1].index(True)
    return sum(1 for i in range(first + 1, last) if not busy[i] and not inst.is_break(d, i + 1))
