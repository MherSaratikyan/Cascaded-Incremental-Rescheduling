"""Steps 2-5 of Table 1: the incremental cascaded rescheduling procedure.

The same driver also runs the CP-SAT baselines of Section 7 through a
:class:`Strategy` (B1 local check, B2 hard-only chain, B3/B4 full
re-optimisation, B5 unweighted expansion, B7 the proposed method).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Optional

from . import evaluate as ev
from .context import SolveStats, StepContext
from .cpsat_model import SubModel
from .dependency import DependencyGraph
from .explain import Deadline, counting_proof, find_mcs, find_mus, physical_check
from .model import Timetable
from .result import Iteration, Outcome, Result


@dataclass(frozen=True)
class Strategy:
    name: str = "B7"
    expansion: str = "weighted"      # weighted | unweighted | none | full
    protected: bool = True           # enforce (4)
    locks: bool = True               # enforce (5)
    gap_protection: bool = True      # enforce (9)
    quality_trigger: bool = True     # extra expansion on unacceptable quality
    use_core: bool = True            # use the solver core C_j and the budget literal


STRATEGIES: dict[str, Strategy] = {
    "B1": Strategy("B1", expansion="none", quality_trigger=False),
    "B2": Strategy("B2", protected=False, locks=False, gap_protection=False),
    "B3": Strategy("B3", expansion="full", protected=False, locks=False,
                   gap_protection=False, quality_trigger=False),
    "B4": Strategy("B4", expansion="full", quality_trigger=False),
    "B5": Strategy("B5", expansion="unweighted", use_core=False),
    "B7": Strategy("B7"),
}


def effective_context(ctx: StepContext, st: Strategy) -> StepContext:
    """Context as seen by a strategy that ignores some regulatory elements.
    Ignored protected constraints fall back to ordinary ones (they are still
    penalised in Q_O); metrics are always computed on the original context."""
    if st.protected and st.locks and st.gap_protection:
        return ctx
    c = replace(ctx)
    if not st.locks:
        c.locks = {}
    if not st.protected:
        c.status, c.beta = {}, {}
        c.ordinary = list(ctx.inst.soft)
    if not st.gap_protection:
        c.gap_protected = set()
    seeds = set(ctx.d) | ({ctx.request.lesson} if ctx.request else set())
    c.f0 = c.closure(seeds) - c.full_locked
    return c


def objective_tuple(c: StepContext, tt: Timetable) -> tuple:
    """(Delta G^+_k, P^0, G, Q_O) of (14), evaluated exactly."""
    p = c.params
    g = ev.gaps(c.inst, tt)
    return (round(ev.new_gaps(c.inst, g, c.gaps_k), 6),
            ev.perturbation(tt, c.x0, p.w_time, p.w_room),
            round(ev.total_gaps(c.inst, g), 6),
            round(ev.q_ordinary(c.inst, tt, c.ordinary), 6))


def soft_by_kind(ctx: StepContext, tt: Timetable) -> dict[str, float]:
    """Change of sum pi_s v_s per soft-constraint kind (all constraints, protected included)."""
    out = {"pref": 0.0, "load": 0.0, "room": 0.0}
    for s in ctx.inst.soft.values():
        out[s.kind] += s.weight * (ev.soft_value(ctx.inst, tt, s) - ev.soft_value(ctx.inst, ctx.xk, s))
    return out


def disruption_to_people(ctx: StepContext, tt: Timetable, changed: list[str]) -> dict:
    inst, H = ctx.inst, ctx.inst.hours
    hours_moved = day_changes = room_only = 0
    for e in changed:
        (t, r), (tk, rk) = tt[e], ctx.xk[e]
        if t != tk:
            hours_moved += inst.lessons[e].duration
            day_changes += (t // H) != (tk // H)
        else:
            room_only += 1
    return {"lecturers_affected": len({l for e in changed for l in inst.lessons[e].lecturers}),
            "groups_affected": len({g for e in changed for g in inst.lessons[e].groups}),
            "hours_moved": hours_moved, "day_changes": day_changes, "room_only_moves": room_only}


def compute_metrics(ctx: StepContext, tt: Optional[Timetable], f_size: int) -> dict:
    p, inst = ctx.params, ctx.inst
    out = {"F_ratio": f_size / max(1, len(inst.lessons)), "F_size": f_size,
           "Q_O_k": ev.q_ordinary(inst, ctx.xk, ctx.ordinary)}
    if tt is None:
        out["changed"] = []
        return out
    g = ev.gaps(inst, tt)
    changed = ev.changed_lessons(tt, ctx.xk)
    kinds = soft_by_kind(ctx, tt)
    out.update(disruption_to_people(ctx, tt, changed))
    out.update({f"dV_{k}": round(v, 6) for k, v in kinds.items()})
    out.update({
        "changed": changed,
        "P_k": ev.perturbation(tt, ctx.xk, p.w_time, p.w_room),
        "P_0": ev.perturbation(tt, ctx.x0, p.w_time, p.w_room),
        "G": ev.total_gaps(inst, g),
        "dG_plus": ev.new_gaps(inst, g, ctx.gaps_k),
        "Q_O": ev.q_ordinary(inst, tt, ctx.ordinary),
        "protected_violations": len(ctx.protected_violations(tt)),
        "lock_violations": len(ctx.lock_violations(tt)),
        "gap_protection_violations": len(ctx.gap_protection_violations(tt)),
        "hard_violations": ev.hard_violations(inst, tt, ctx.omega).units,
    })
    return out


def lexmin(c: StepContext, F: set[str], cap: int, dl: Deadline, hint: Timetable,
           notes: list[str], stages: Optional[list] = None) -> tuple[Timetable, tuple]:
    """(14): lexicographic minimisation of (Delta G^+, P^0, G, Q_O) over X_j
    with the frozen threshold P_k(x) <= P-bar, warm-started from ``hint``.
    ``stages`` collects (name, status, value, best bound) per stage."""
    sub = SubModel(c, F, cap=cap, assumptions=False)
    best = hint
    for name in ("dG", "P0", "G", "QO"):
        if dl.expired:
            notes.append(f"lexmin stopped before {name}: time limit")
            if stages is not None:
                stages.append((name, "SKIPPED", None, None))
            break
        expr = getattr(sub, name)
        r = sub.minimize(expr, dl.remaining(), hint=best)
        if stages is not None:
            stages.append((name, r.status, r.value, r.bound))
        if not r.feasible:
            notes.append(f"lexmin {name}: {r.status}, kept previous solution")
            break
        best = r.timetable
        sub.bound(expr, r.value)
        if r.status != "OPTIMAL":
            notes.append(f"lexmin {name}: not proven optimal")
    return best, objective_tuple(c, best)


def _opt_gap(value, bound) -> Optional[float]:
    if value is None or bound is None:
        return None
    return max(0.0, (value - bound) / max(1.0, abs(value)))


def solver_metrics(res: Result, ctx: StepContext) -> dict:
    """Effort, certification and explanation quality of one request."""
    st = ctx.stats
    stages = res.stages
    lex = [s for s in stages if s[0] != "P*"]
    gaps = [g for g in (_opt_gap(s[2], s[3]) for s in stages) if g is not None]
    out = {
        "iterations": len(res.trace),
        "solver_calls": st.calls,
        "solver_seconds": round(st.seconds, 4),
        "max_model_vars": st.max_vars,
        "max_model_constraints": st.max_constraints,
        "final_model_vars": st.last_vars,
        "final_model_constraints": st.last_constraints,
        "pstar_proven": any(s[0] == "P*" and s[1] == "OPTIMAL" for s in stages) if stages else None,
        "lexmin_proven": (all(s[1] == "OPTIMAL" for s in lex) if lex else None),
        "lexmin_stages": "/".join(f"{s[0]}:{s[1][:3]}" for s in stages),
        "opt_gap_max": round(max(gaps), 6) if gaps else None,
        "mus_size": len(res.mus or []),
        "mus_minimal": res.mus_minimal if res.mus else None,
        "mcs_count": len(res.mcs),
        "mcs_verified": sum(c.verified for c in res.mcs),
    }
    out.update({f"t_{k}": round(v, 4) for k, v in res.phases.items()})
    return out


def quality_unacceptable(c: StepContext, tt: Timetable) -> bool:
    g = ev.gaps(c.inst, tt)
    dg = ev.new_gaps(c.inst, g, c.gaps_k)
    dq = ev.q_ordinary(c.inst, tt, c.ordinary) - ev.q_ordinary(c.inst, c.xk, c.ordinary)
    return dg > c.params.theta_g + 1e-9 or dq > c.theta_q + 1e-9


def reschedule(ctx: StepContext, strategy: Optional[Strategy] = None) -> Result:
    st = strategy or STRATEGIES["B7"]
    p = ctx.params
    t_start = time.perf_counter()
    dl = Deadline(p.time_limit)
    ctx.stats = SolveStats()
    c = effective_context(ctx, st)
    res = Result(outcome=Outcome.ALGORITHMIC_FAILURE, method=st.name, review=list(ctx.review))
    mark = [time.perf_counter(), "search"]       # running phase timer

    def phase(name: str) -> None:
        now = time.perf_counter()
        res.phases[mark[1]] = res.phases.get(mark[1], 0.0) + now - mark[0]
        mark[0], mark[1] = now, name

    res.phase = phase

    def finish(outcome: Outcome, tt: Optional[Timetable] = None, f_size: int = 0) -> Result:
        phase("done")
        res.phases.pop("done", None)
        res.outcome = outcome
        res.timetable = tt if outcome == Outcome.SOLVED else None
        res.seconds = time.perf_counter() - t_start
        res.metrics = compute_metrics(ctx, res.timetable, f_size)
        res.metrics.update(solver_metrics(res, ctx))
        res.__dict__.pop("phase", None)            # drop the closure (keeps Result picklable)
        if outcome == Outcome.SOLVED and ctx.request is not None and ctx.request.voluntary:
            requester = set(ctx.inst.lessons[ctx.request.lesson].lecturers)
            moved = res.metrics["changed"]
            others = sorted({l for e in moved for l in ctx.inst.lessons[e].lecturers} - requester)
            res.affected_lecturers = others
            res.requires_approval = bool(others)
        return res

    # ---- trivial physical infeasibility: some lesson has no admissible option
    empty = sorted(e for e in c.inst.lessons if not c.omega[e])
    proof = None if empty else counting_proof(c)
    if proof:
        res.notes.append(proof)
        return finish(Outcome.PHYSICAL_INFEASIBLE)
    if empty:
        res.notes.append(f"Omega empty for {', '.join(empty)}")
        return finish(Outcome.PHYSICAL_INFEASIBLE)

    # ---- step 2: D_0 ∩ Λ^A ≠ ∅ -> lock conflict without search
    if st.locks and c.lock_conflicts:
        res.lock_conflicts = sorted(c.lock_conflicts)
        phase("explain")
        r = physical_check(c, p.explain_time_limit)
        if r.infeasible:
            return finish(Outcome.PHYSICAL_INFEASIBLE)
        if r.feasible:
            res.candidate = r.timetable
        else:
            res.notes.append("hard-only check timed out")
        return finish(Outcome.LOCK_CONFLICT)

    if not c.f0:
        return finish(Outcome.SOLVED, dict(c.xk), 0)

    # ---- step 3: grow (F_j, B_j) until X_j ≠ ∅
    graph = DependencyGraph(c)
    all_free = c.all_free
    last = len(c.ladder) - 1
    if st.expansion == "full":
        F, bi = set(all_free), last
    else:
        F, bi = set(c.f0), 0
    j, no_core_failures = 0, 0
    feasible = None
    while True:
        budget = c.ladder[bi]
        pool: list[str] = []
        if st.use_core and st.expansion == "weighted" and p.core_pool > 0 and not F >= all_free:
            pool = graph.ranked(graph.neighbours(F))[:p.core_pool]
        sub = SubModel(c, F | set(pool), assumed_fixed=pool, budget=budget)
        t_it = time.perf_counter()
        r = sub.check(dl.remaining(), hint=c.xk)
        it = Iteration(j=j, f_size=len(F), budget=budget, status=r.status,
                       variables=sub.size(), lessons=sorted(F))
        res.trace.append(it)
        if r.feasible:
            it.action = "feasible"
            it.seconds = time.perf_counter() - t_it
            feasible = r
            break
        if not r.infeasible:                       # time limit before a decision
            it.action = "time limit"
            it.seconds = time.perf_counter() - t_it
            return _failure(res, c, finish, F)
        core = r.core
        if st.use_core and core and p.minimize_core and ("budget",) in core and len(core) > 1:
            # CP-SAT cores are not minimal; the only decision that depends on
            # minimality is "raise B_j vs expand F_j", so test the budget alone
            rest = [i for i in core if i != ("budget",)]
            r2 = sub.check(max(0.2, min(2.0, dl.remaining() / 4)), assume=rest)
            if r2.infeasible:
                core = r2.core if r2.core is not None else rest
        it.core = core
        res.core = core
        it.seconds = time.perf_counter() - t_it
        full = F >= all_free
        bmax = bi == last
        has_budget = bool(core) and ("budget",) in core
        if full and (bmax or (st.use_core and not has_budget)):
            it.action = "X_J proven empty"
            return _classify(res, c, st, finish, F)
        if st.expansion == "none":
            it.action = "rejected"
            return finish(Outcome.REJECTED_LOCAL, f_size=len(F))
        if st.use_core and has_budget and not bmax:
            bi += 1
            it.action = "raise B"
            j += 1
            continue
        if full:
            bi += 1
            it.action = "raise B (F complete)"
            j += 1
            continue
        cj = [i[1] for i in core if i[0] == "fix"] if (st.use_core and core) else []
        F, _ = graph.expand(F, cj, p.top_m, weighted=(st.expansion == "weighted"))
        it.action = "expand F" + (f" (core: {', '.join(sorted(cj))})" if cj else "")
        if not (st.use_core and core):
            no_core_failures += 1
            if no_core_failures % 2 == 0 and not bmax:
                bi += 1
                it.action += " + raise B"
        j += 1

    # ---- first feasible X_j0: P* and frozen threshold P-bar = P* + delta_k
    phase("optimise")
    opt = SubModel(c, F, budget=c.ladder[bi], assumptions=False)
    r1 = opt.minimize(opt.Pk, dl.remaining(), hint=feasible.timetable)
    res.stages.append(("P*", r1.status, r1.value, r1.bound))
    if r1.feasible:
        base = r1.timetable
        p_star = r1.value
        if r1.status != "OPTIMAL":
            res.notes.append("P* not proven optimal")
    else:
        base = feasible.timetable
        p_star = ev.perturbation(base, c.xk, p.w_time, p.w_room)
        res.notes.append("P* taken from the first feasible solution")
    p_bar = p_star + c.delta
    res.p_star, res.p_bar = p_star, p_bar

    # ---- step 4: lexmin (14) with warm start
    tt, obj = lexmin(c, F, p_bar, dl, base, res.notes, res.stages)
    if st.quality_trigger and not F >= all_free and quality_unacceptable(c, tt):
        phase("extra")
        F2, _ = graph.expand(F, [], p.top_m, weighted=(st.expansion != "unweighted"))
        extra_stages: list = []
        tt2, obj2 = lexmin(c, F2, p_bar, dl, tt, res.notes, extra_stages)
        res.trace.append(Iteration(j=j + 1, f_size=len(F2), budget=p_bar, status="extra",
                                   action="extra expansion", lessons=sorted(F2)))
        if obj2 < obj:
            tt, obj, F = tt2, obj2, F2
            res.extra_expansion = "accepted"
            res.stages = [s for s in res.stages if s[0] == "P*"] + extra_stages
        else:
            res.extra_expansion = "rejected"
    res.objective = obj
    res.final_f = sorted(F)
    return finish(Outcome.SOLVED, tt, len(F))


def _classify(res: Result, c: StepContext, st: Strategy, finish, F) -> Result:
    """Step 5 after X_J = ∅ was proven (Claim 2)."""
    p = c.params
    if not (st.protected or st.locks or st.gap_protection):
        return finish(Outcome.PHYSICAL_INFEASIBLE, f_size=len(F))
    res.phase("explain")
    edl = Deadline(p.explain_time_limit)
    r = physical_check(c, edl.remaining())
    if r.infeasible:
        return finish(Outcome.PHYSICAL_INFEASIBLE, f_size=len(F))
    if not r.feasible:
        res.notes.append("regulation-constrained problem proven infeasible; hard-only check timed out")
        return finish(Outcome.ALGORITHMIC_FAILURE, f_size=len(F))
    res.candidate = r.timetable
    res.mus, res.mus_minimal = find_mus(c, edl)
    res.mcs = find_mcs(c, edl)
    return finish(Outcome.REGULATORY_INFEASIBLE, f_size=len(F))


def _failure(res: Result, c: StepContext, finish, F) -> Result:
    """Time limit before a decision on X_J: algorithmic failure, with a relaxed
    (hard-only) candidate and the last core."""
    res.phase("explain")
    r = physical_check(c, min(5.0, c.params.explain_time_limit))
    if r.feasible:
        res.candidate = r.timetable
    return finish(Outcome.ALGORITHMIC_FAILURE, f_size=len(F))
