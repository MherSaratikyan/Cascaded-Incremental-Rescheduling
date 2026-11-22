"""Exact (solver-independent) evaluation of timetables.

Used to compute D_0/D, the baselines beta_s and G_ud, post-hoc metrics of every
method, and to cross-check the CP-SAT model (e.g. Claim 5 on gaps).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .model import Assignment, Instance, SoftConstraint, Timetable


# --------------------------------------------------------------------------- Omega
def omega(inst: Instance, e: str, start: Optional[int] = None,
          room: Optional[str] = None) -> list[Assignment]:
    """Omega_e: admissible (t, r) for lesson e.  Depends only on the instance
    (same day, available participants, room large enough and open), never on
    other lessons' occupancy.  ``start``/``room`` restrict it for a move request."""
    l = inst.lessons[e]
    size = inst.lesson_size(l)
    rooms = list(l.rooms) if l.rooms is not None else sorted(inst.rooms)
    rooms = [r for r in rooms if inst.rooms[r].capacity >= size]
    if room is not None:
        rooms = [r for r in rooms if r == room]
    blocked: set[int] = set(inst.closed)
    for u in l.participants:
        blocked |= inst.participants[u].unavailable
    out: list[Assignment] = []
    ids = inst.time_ids() if l.starts is None else sorted(l.starts)
    for t in ids:
        if start is not None and t != start:
            continue
        c = inst.cells(l, t)
        if not c or any(tau in blocked for tau in c):
            continue
        for r in rooms:
            un = inst.rooms[r].unavailable
            if any(tau in un for tau in c):
                continue
            out.append((t, r))
    return out


def all_omegas(inst: Instance, request: Optional[tuple[str, int, Optional[str]]] = None
               ) -> dict[str, list[Assignment]]:
    om = {e: omega(inst, e) for e in inst.lessons}
    if request is not None:
        e, t, r = request
        om[e] = omega(inst, e, start=t, room=r)
    return om


# ------------------------------------------------------------------- hard check
@dataclass
class HardReport:
    units: int = 0
    lessons: set[str] = field(default_factory=set)
    details: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.units == 0


def hard_violations(inst: Instance, tt: Timetable,
                    om: Optional[dict[str, list[Assignment]]] = None) -> HardReport:
    """Violations of (1)-(2) plus the special-resource capacities."""
    rep = HardReport()
    om = om if om is not None else all_omegas(inst)
    room_cells: dict[tuple[str, int], list[str]] = defaultdict(list)
    part_cells: dict[tuple[str, int], list[str]] = defaultdict(list)
    res_cells: dict[tuple[str, int], list[str]] = defaultdict(list)
    for e, l in inst.lessons.items():
        if e not in tt:
            rep.units += 1
            rep.lessons.add(e)
            rep.details.append(f"{e} unassigned")
            continue
        a = tt[e]
        if a not in set(om[e]):
            rep.units += 1
            rep.lessons.add(e)
            rep.details.append(f"{e} at inadmissible {a}")
        c = inst.cells(l, a[0]) or range(a[0], a[0] + 1)
        for tau in c:
            room_cells[(a[1], tau)].append(e)
            for u in l.participants:
                part_cells[(u, tau)].append(e)
            for q in l.resources:
                res_cells[(q, tau)].append(e)
    for cells, label in ((room_cells, "room"), (part_cells, "participant")):
        for cell, es in cells.items():
            if len(es) > 1:
                rep.units += len(es) - 1
                rep.lessons.update(es)
                rep.details.append(f"{label} clash {cell}: {sorted(es)}")
    for (q, tau), es in res_cells.items():
        k = inst.resources[q].capacity
        if len(es) > k:
            rep.units += len(es) - k
            rep.lessons.update(es)
            rep.details.append(f"resource {q} over capacity at {tau}: {sorted(es)}")
    for a, b in inst.links:
        if a in tt and b in tt and tt[a][0] != tt[b][0]:
            rep.units += 1
            rep.lessons.update((a, b))
            rep.details.append(f"linked {a},{b} not simultaneous")
    return rep


# ------------------------------------------------------------- soft constraints
def gamma(inst: Instance, s: SoftConstraint) -> list[str]:
    """Gamma(s): lessons whose assignment influences v_s."""
    if s.kind in ("pref", "load"):
        return [e for e, l in inst.lessons.items() if s.lecturer in l.lecturers]
    return inst.lessons_of_course(s.course)


def soft_den(inst: Instance, s: SoftConstraint) -> int:
    if s.kind == "pref":
        return max(1, sum(inst.min_hours(e) for e in gamma(inst, s)))
    if s.kind == "load":
        return max(1, inst.hours - s.cap)
    return max(1, len(gamma(inst, s)))


def soft_num(inst: Instance, tt: Timetable, s: SoftConstraint) -> int:
    """Integer numerator of v_s(x) (den_s is constant for a fixed instance)."""
    if s.kind == "pref":
        n = 0
        for e in gamma(inst, s):
            if e in tt:
                n += sum(1 for tau in inst.cells(e, tt[e][0]) if tau in s.unwanted)
        return n
    if s.kind == "load":
        hours = sum(inst.hours_on_day(e, tt[e][0], s.day) for e in gamma(inst, s) if e in tt)
        return max(0, hours - s.cap)
    return sum(1 for e in gamma(inst, s) if e in tt and tt[e][1] != s.home)


def soft_value(inst: Instance, tt: Timetable, s: SoftConstraint) -> float:
    return soft_num(inst, tt, s) / soft_den(inst, s)


def q_coef(inst: Instance, s: SoftConstraint, scale: int) -> int:
    """Integer coefficient of num_s in the scaled Q_O used by CP-SAT."""
    if s.weight <= 0:
        return 0
    return max(1, round(scale * s.weight / soft_den(inst, s)))


def q_ordinary(inst: Instance, tt: Timetable, ordinary: Iterable[str]) -> float:
    """Q_O(x) = sum_{s in S^O} pi_s v_s(x)."""
    return sum(inst.soft[s].weight * soft_value(inst, tt, inst.soft[s]) for s in ordinary)


# --------------------------------------------------------------------- idle gaps
def gaps(inst: Instance, tt: Timetable, participants: Optional[Iterable[str]] = None
         ) -> dict[tuple[str, int], int]:
    """G_ud: hours p that are free, not a break, with a lesson of u before and
    after p on the same day d."""
    us = list(participants) if participants is not None else list(inst.participants)
    occ: dict[str, set[int]] = {u: set() for u in us}
    for e, l in inst.lessons.items():
        if e not in tt:
            continue
        c = inst.cells(l, tt[e][0])
        for u in l.participants:
            if u in occ:
                occ[u].update(c)
    out: dict[tuple[str, int], int] = {}
    H = inst.hours
    for u in us:
        o = occ[u]
        for d in range(1, inst.days + 1):
            busy = [((d - 1) * H + p - 1) in o for p in range(1, H + 1)]
            if not any(busy):
                out[(u, d)] = 0
                continue
            first = busy.index(True)
            last = H - 1 - busy[::-1].index(True)
            out[(u, d)] = sum(1 for i in range(first + 1, last)
                              if not busy[i] and not inst.is_break(d, i + 1))
    return out


def total_gaps(inst: Instance, g: dict[tuple[str, int], int]) -> float:
    return sum(inst.participants[u].weight * v for (u, _), v in g.items())


def new_gaps(inst: Instance, g: dict[tuple[str, int], int], g_ref: dict[tuple[str, int], int]) -> float:
    """Delta G^+_k(x) = sum_{u,d} omega_u * max(0, G_ud(x) - G_ud(x^(k)))."""
    return sum(inst.participants[u].weight * max(0, v - g_ref.get((u, d), 0))
               for (u, d), v in g.items())


# ------------------------------------------------------------------ perturbation
def perturbation(tt: Timetable, ref: Timetable, w_time: int, w_room: int,
                 lessons: Optional[Iterable[str]] = None) -> int:
    """P_k(x) of (3) with respect to the reference timetable ``ref``."""
    es = lessons if lessons is not None else ref.keys()
    p = 0
    for e in es:
        if e not in ref or e not in tt:
            continue
        (t, r), (t0, r0) = tt[e], ref[e]
        if t != t0:
            p += w_time
        elif r != r0:
            p += w_room
    return p


def changed_lessons(tt: Timetable, ref: Timetable) -> list[str]:
    return sorted(e for e in tt if e in ref and tt[e] != ref[e])
