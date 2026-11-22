"""Seeded synthetic instances, initial timetables x^(0), disruption sequences
and regulation samples (protected share rho, locked share lambda) for Section 7.

    python -m experiments.generator --size S --seed 1 --out experiments/instances
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from cascade import evaluate as ev
from cascade import io
from cascade.context import StepContext
from cascade.cpsat_model import SubModel
from cascade.disruptions import Disruption
from cascade.model import Instance, Lesson, Participant, Resource, Room, SoftConstraint, Timetable
from cascade.params import Params

DAYS, HOURS, BREAK_HOUR = 5, 8, 5

SIZES = {
    "XS": dict(lessons=20, rooms=3, lecturers=5, groups=4),
    "S": dict(lessons=60, rooms=6, lecturers=12, groups=8),
    "M": dict(lessons=200, rooms=15, lecturers=35, groups=25),
    # L and XL keep the room occupancy of M (~60 % of room-hours), as denser
    # instances are mostly infeasible already for x^(0)
    "L": dict(lessons=500, rooms=38, lecturers=80, groups=60),
    "XL": dict(lessons=1000, rooms=75, lecturers=160, groups=120),
}
# sizes whose x^(0) comes from greedy construction + windowed CP-SAT repair
WINDOWED = {"L", "XL"}


def _rng(*parts) -> random.Random:
    return random.Random("|".join(str(p) for p in parts))


def generate_instance(size: str, seed: int) -> Instance:
    cfg = SIZES[size]
    rng = _rng("instance", size, seed)
    T = DAYS * HOURS
    caps = [30, 30, 40, 40, 60, 100]
    rooms = {}
    for i in range(cfg["rooms"]):
        cap = caps[i % len(caps)] if i < len(caps) else rng.choice(caps)
        rooms[f"R{i + 1}"] = Room(f"R{i + 1}", cap)
    labs = [r for i, r in enumerate(sorted(rooms, key=lambda x: int(x[1:]))) if i % 3 == 2]
    parts: dict[str, Participant] = {}
    for i in range(cfg["lecturers"]):
        un = {t for t in range(T) if rng.random() < 0.10}
        parts[f"L{i + 1}"] = Participant(f"L{i + 1}", "lecturer", 0, un)
    for i in range(cfg["groups"]):
        parts[f"G{i + 1}"] = Participant(f"G{i + 1}", "group", rng.randint(15, 30))
    lecturers = [p for p in parts if p.startswith("L")]
    groups = [p for p in parts if p.startswith("G")]
    lect_hours = {l: 0 for l in lecturers}
    group_hours = {g: 0 for g in groups}
    lessons: dict[str, Lesson] = {}
    soft: dict[str, SoftConstraint] = {}
    resources = {"kit": Resource("kit", 1)}
    course_no = 0
    attempts = 0
    while len(lessons) < cfg["lessons"] and attempts < 10 * cfg["lessons"]:
        attempts += 1
        course_no += 1
        cid = f"C{course_no}"
        lec = rng.choice(lecturers)
        k = rng.choice([2, 3]) if rng.random() < 0.15 else 1
        gs = tuple(sorted(rng.sample(groups, min(k, len(groups)))))
        size_needed = sum(parts[g].size for g in gs)
        n = rng.choice([1, 2, 2, 3])
        durs = [rng.choices([1, 2, 3], weights=[0.4, 0.45, 0.15])[0] for _ in range(n)]
        hours = sum(durs)
        if lect_hours[lec] + hours > 18 or any(group_hours[g] + hours > 24 for g in gs):
            continue
        is_lab = rng.random() < 0.10 and k == 1
        allowed = None
        res: tuple[str, ...] = ()
        if is_lab:
            allowed = tuple(r for r in labs if rooms[r].capacity >= size_needed)
            if not allowed:
                continue
            if rng.random() < 0.3:
                res = ("kit",)
        fits = [r for r in (allowed or rooms) if rooms[r].capacity >= size_needed]
        if not fits:
            continue
        lect_hours[lec] += hours
        for g in gs:
            group_hours[g] += hours
        for i, d in enumerate(durs):
            if len(lessons) >= cfg["lessons"]:
                break
            eid = f"{cid}.{i + 1}"
            lessons[eid] = Lesson(eid, d, (lec,), gs, cid, allowed, None, res)
        if len([e for e in lessons if lessons[e].course == cid]) >= 2:
            home = rng.choice(fits)
            soft[f"room:{cid}"] = SoftConstraint(f"room:{cid}", "room", 1.0, course=cid, home=home)

    # linked (simultaneous) pairs: ~5% of lessons, disjoint participants, equal duration
    links: list[tuple[str, str]] = []
    used: set[str] = set()
    ids = sorted(lessons)
    rng.shuffle(ids)
    want = max(1, len(lessons) // 40)
    for a in ids:
        if len(links) >= want:
            break
        if a in used:
            continue
        for b in ids:
            if b == a or b in used or lessons[a].course == lessons[b].course:
                continue
            la, lb = lessons[a], lessons[b]
            if la.duration == lb.duration and not set(la.participants) & set(lb.participants) \
                    and not la.resources and not lb.resources:
                links.append((a, b))
                used |= {a, b}
                break

    for l in lecturers:
        if lect_hours[l] == 0:
            continue
        unwanted = set()
        for d in rng.sample(range(DAYS), 2):
            unwanted |= {d * HOURS, d * HOURS + HOURS - 1}
        unwanted |= {t for t in range(T) if rng.random() < 0.05}
        soft[f"pref:{l}"] = SoftConstraint(f"pref:{l}", "pref", 1.0, lecturer=l,
                                           unwanted=frozenset(unwanted))
        for d in range(1, DAYS + 1):
            soft[f"load:{l}:{d}"] = SoftConstraint(f"load:{l}:{d}", "load", 1.0, lecturer=l,
                                                   day=d, cap=4)
    inst = Instance(name=f"synthetic-{size}-{seed}", days=DAYS, hours=HOURS, rooms=rooms,
                    participants=parts, lessons=lessons,
                    breaks={d: frozenset({BREAK_HOUR}) for d in range(1, DAYS + 1)},
                    resources=resources, links=links, soft=soft)
    inst.validate()
    return inst


def greedy_timetable(inst: Instance, seed: int = 0) -> Timetable:
    """First-fit construction (most constrained lesson first, packed next to the
    participants' existing lessons).  May leave lessons unplaced; used as hint."""
    rng = _rng("greedy", inst.name, seed)
    om = ev.all_omegas(inst)
    partner: dict[str, str] = {}
    for a, b in inst.links:
        partner[a], partner[b] = b, a
    order = sorted(inst.lessons, key=lambda e: (len(om[e]), -inst.lessons[e].duration, e))
    room_busy: set[tuple] = set()
    part_busy: set[tuple] = set()
    res_busy: Counter = Counter()
    tt: Timetable = {}

    def free(e, t, r) -> bool:
        l = inst.lessons[e]
        for tau in range(t, t + l.duration):
            if (r, tau) in room_busy or any((u, tau) in part_busy for u in l.participants):
                return False
            if any(res_busy[(q, tau)] >= inst.resources[q].capacity for q in l.resources):
                return False
        return True

    def place(e, t, r):
        l = inst.lessons[e]
        tt[e] = (t, r)
        for tau in range(t, t + l.duration):
            room_busy.add((r, tau))
            for u in l.participants:
                part_busy.add((u, tau))
            for q in l.resources:
                res_busy[(q, tau)] += 1

    def score(e, t) -> float:
        l = inst.lessons[e]
        adj = sum(1 for u in l.participants for tau in (t - 1, t + l.duration)
                  if (u, tau) in part_busy and tau // inst.hours == t // inst.hours)
        return -adj + rng.random() * 0.5

    for e in order:
        if e in tt:
            continue
        cands = [(t, r) for t, r in om[e] if free(e, t, r)]
        if e in partner and partner[e] not in tt:
            f = partner[e]
            pairs = []
            for t, r in cands:
                place(e, t, r)
                ok = [(t2, r2) for t2, r2 in om[f] if t2 == t and free(f, t2, r2)]
                tt.pop(e)
                _unplace(inst, e, t, r, room_busy, part_busy, res_busy)
                if ok:
                    pairs.append(((t, r), ok[0]))
            if pairs:
                (a, b) = min(pairs, key=lambda pr: score(e, pr[0][0]))
                place(e, *a)
                place(f, *b)
            continue
        if cands:
            place(e, *min(cands, key=lambda a: score(e, a[0])))
    return tt


def _unplace(inst, e, t, r, room_busy, part_busy, res_busy):
    l = inst.lessons[e]
    for tau in range(t, t + l.duration):
        room_busy.discard((r, tau))
        for u in l.participants:
            part_busy.discard((u, tau))
        for q in l.resources:
            res_busy[(q, tau)] -= 1


def initial_timetable(inst: Instance, time_limit: float = 60.0, workers: int = 8,
                      seed: int = 0) -> Optional[Timetable]:
    """x^(0): CP-SAT solve warm-started from the greedy construction, first for
    feasibility, then minimising 200*G + Q_O (good starting quality)."""
    params = Params(workers=workers, seed=seed)
    ctx = StepContext.build(inst, inst, {}, {}, params, gap_protected=set())
    sub = SubModel(ctx, inst.lessons, protected=False, locks=False, gap_protection=False,
                   assumptions=False)
    hint = greedy_timetable(inst, seed)
    r = sub.check(time_limit / 2, hint=hint)
    if not r.feasible:
        return None
    r2 = sub.minimize(200 * sub.G + sub.QO, time_limit / 2, hint=r.timetable)
    return r2.timetable if r2.feasible else r.timetable


def windowed_timetable(inst: Instance, time_budget: float = 240.0, window: int = 60,
                       workers: int = 4, seed: int = 0) -> Optional[Timetable]:
    """x^(0) for L / XL: greedy construction, then CP-SAT repair of one window
    at a time (<= ``window`` lessons of one day, kept on that day, everything
    else fixed), minimising 200*G + Q_O.  Memory per solve stays bounded."""
    import time as _time
    from dataclasses import replace as _replace
    tt = greedy_timetable(inst, seed)
    if len(tt) < len(inst.lessons) or not ev.hard_violations(inst, tt).ok:
        return None
    params = Params(workers=workers, seed=seed)
    base = StepContext.build(inst, inst, tt, tt, params, gap_protected=set())
    cache = base.__dict__.setdefault("_gamma_cache", {})
    H = inst.hours
    windows: list[list[str]] = []
    for d in range(1, inst.days + 1):
        on_day = sorted((e for e in tt if tt[e][0] // H + 1 == d),
                        key=lambda e: (inst.lessons[e].groups, e))
        windows += [on_day[i:i + window] for i in range(0, len(on_day), window)]
    end = _time.perf_counter() + time_budget
    for i, win in enumerate(windows):
        left = end - _time.perf_counter()
        if left <= 1:
            break
        om = dict(base.omega)
        for e in win:
            d = tt[e][0] // H
            om[e] = [a for a in base.omega[e] if a[0] // H == d]
        c = _replace(base, xk=dict(tt), x0=dict(tt), omega=om)
        c.__dict__["_gamma_cache"] = cache
        sub = SubModel(c, win, protected=False, locks=False, gap_protection=False, assumptions=False)
        r = sub.minimize(200 * sub.G + sub.QO, min(20.0, left / (len(windows) - i)), hint=tt)
        if r.feasible:
            tt = r.timetable
    return tt


def load_or_generate(size: str, seed: int, cache_dir: Path, time_limit: float = 60.0,
                     workers: int = 8) -> tuple[Instance, Timetable]:
    path = Path(cache_dir) / f"{size}-{seed}.json"
    if path.exists():
        d = io.load_json(path)
        inst = io.instance_from_dict(d["instance"])
        return inst, io.timetable_from_dict(inst, d["timetable"])
    for sub_seed in range(10):
        inst = generate_instance(size, seed * 100 + sub_seed)
        if size in WINDOWED:
            x0 = windowed_timetable(inst, time_budget=240.0 if size == "L" else 360.0,
                                    workers=min(workers, 4), seed=seed)
        else:
            x0 = initial_timetable(inst, time_limit, workers, seed)
        if x0 is not None:
            io.save_json({"instance": io.instance_to_dict(inst),
                          "timetable": io.timetable_to_dict(inst, x0)}, path)
            return inst, x0
    raise RuntimeError(f"could not generate a feasible {size} instance for seed {seed}")


def generate_sequence(inst: Instance, seed: int, steps: Optional[int] = None,
                      min_steps: int = 5, max_steps: int = 20) -> list[Disruption]:
    rng = _rng("sequence", inst.name, seed)
    K = steps if steps is not None else rng.randint(min_steps, max_steps)
    kinds = ["room_unavailable", "participant_unavailable", "time_unavailable",
             "capacity_change", "move_request"]
    weights = [0.25, 0.25, 0.10, 0.15, 0.25]
    rooms = sorted(inst.rooms)
    lecturers = sorted(inst.lecturers)
    lessons = sorted(inst.lessons)
    out = []
    for _ in range(K):
        kind = rng.choices(kinds, weights)[0]
        d = rng.randint(1, inst.days)
        if kind in ("room_unavailable", "participant_unavailable"):
            if rng.random() < 0.3:
                slots = [(d, h) for h in range(1, inst.hours + 1)]
            else:
                h0 = rng.randint(1, inst.hours - 1)
                slots = [(d, h) for h in range(h0, min(inst.hours, h0 + rng.randint(2, 4)) + 1)]
            if kind == "room_unavailable":
                out.append(Disruption(kind, room=rng.choice(rooms), slots=slots))
            else:
                out.append(Disruption(kind, participant=rng.choice(lecturers), slots=slots))
        elif kind == "time_unavailable":
            h0 = rng.randint(1, inst.hours)
            out.append(Disruption(kind, slots=[(d, h) for h in range(h0, min(inst.hours, h0 + 1) + 1)]))
        elif kind == "capacity_change":
            r = rng.choice(rooms)
            out.append(Disruption(kind, room=r,
                                  capacity=int(inst.rooms[r].capacity * rng.uniform(0.5, 0.8))))
        else:
            e = rng.choice(lessons)
            dur = inst.lessons[e].duration
            out.append(Disruption(kind, lesson=e, day=d, hour=rng.randint(1, inst.hours - dur + 1),
                                  voluntary=True))
    return out


def sample_regulation(inst: Instance, seed: int, rho: float = 0.2, lam: float = 0.1
                      ) -> tuple[dict[str, str], dict[str, str]]:
    """Random protected constraints (share rho, half P0 / half P<=) and locks
    (share lambda, time / room / full equally)."""
    rng = _rng("regulation", inst.name, seed)
    soft = sorted(inst.soft)
    prot = rng.sample(soft, int(round(rho * len(soft))))
    protected = {s: ("P0" if i % 2 == 0 else "PLE") for i, s in enumerate(prot)}
    lessons = sorted(inst.lessons)
    locked = rng.sample(lessons, int(round(lam * len(lessons))))
    locks = {e: "TRA"[i % 3] for i, e in enumerate(locked)}
    return protected, locks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="S", choices=sorted(SIZES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="experiments/instances")
    ap.add_argument("--time-limit", type=float, default=60.0)
    a = ap.parse_args()
    inst, x0 = load_or_generate(a.size, a.seed, Path(a.out), a.time_limit)
    print(f"{inst.name}: {len(inst.lessons)} lessons, {len(inst.rooms)} rooms, "
          f"{len(inst.participants)} participants, {len(inst.soft)} soft constraints")


if __name__ == "__main__":
    main()
