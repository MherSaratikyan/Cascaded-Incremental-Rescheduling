"""Heavy-workload scenarios H1-H9 (built deterministically from an instance,
its x^(0) and a seed)."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from cascade.disruptions import Disruption
from cascade.model import Instance, Timetable
from experiments.generator import generate_sequence

# scenarios consisting of one request: every method starts from the same state,
# so the reference optimum P_k* (B4 with a long limit) is comparable
SINGLE = ("H1", "H2", "H3", "H4", "H7", "H8", "H9")
ALL = ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9")

TITLES = {
    "H1": "Room closed for a week (busiest room)",
    "H2": "Lecturer with most hours absent on their 2 busiest days",
    "H3": "One hour closed in every room on two days",
    "H4": "Two largest rooms (that can spare it) lose 50% of their capacity",
    "H5": "Burst of 10 voluntary move requests",
    "H6": "Long mixed sequence of 20 disruptions",
    "H7": "H1 under dense regulation (rho=0.5, lambda=0.3)",
    "H8": "Physically impossible: a group gets fewer hours than it needs",
    "H9": "Busiest room closed on its busiest day, gap protection off (feasible; measures P_k/P_k*)",
}

# who is protected against new gaps by (9), per scenario (default: groups)
GAP_PROTECT = {"H9": "none"}


@dataclass
class HeavyScenario:
    id: str
    title: str
    disruptions: list[Disruption]
    rho: float = 0.2
    lam: float = 0.1
    notes: dict = field(default_factory=dict)
    gap_protect: str = "groups"


def _hours_by(inst: Instance, x0: Timetable, key) -> Counter:
    c: Counter = Counter()
    for e, (t, r) in x0.items():
        for k in key(e, r):
            c[k] += len(inst.cells(e, t))
    return c


def _still_possible(inst: Instance, d: Disruption) -> bool:
    """True when, after the disruption, every lesson keeps an admissible option."""
    from cascade import evaluate as ev
    from cascade.disruptions import apply
    after, _ = apply(inst, d)
    return all(ev.omega(after, e) for e in after.lessons)


def build(sid: str, inst: Instance, x0: Timetable, seed: int,
          locks: Optional[dict] = None) -> HeavyScenario:
    """``locks`` are the sampled locks of this run: H1/H7 avoid closing a room
    that holds a fully locked lesson (that is a lock conflict, decided in step 2
    without any search, so it would not load the solver)."""
    rng = random.Random(f"heavy|{sid}|{inst.name}|{seed}")
    H, D = inst.hours, inst.days
    locks = locks or {}
    if sid in ("H1", "H7"):
        # busiest room whose closure keeps every lesson possible and hits no full lock
        ranked = sorted(_hours_by(inst, x0, lambda e, r: [r]).items(), key=lambda kv: (-kv[1], kv[0]))
        locked_rooms = {x0[e][1] for e, lk in locks.items() if lk == "A"}
        room = next((r for r, _ in ranked if r not in locked_rooms
                     and _still_possible(inst, Disruption("room_unavailable", room=r))), ranked[0][0])
        dis = [Disruption("room_unavailable", room=room, note=f"room {room} closed all week")]
        if sid == "H7":
            return HeavyScenario(sid, TITLES[sid], dis, rho=0.5, lam=0.3, notes={"room": room})
        return HeavyScenario(sid, TITLES[sid], dis, notes={"room": room})
    if sid == "H2":
        # a whole-week absence makes the lecturer's lessons impossible by definition
        # (no substitutes in the model), so the absence covers the 2 busiest days
        ranked = sorted(_hours_by(inst, x0, lambda e, r: inst.lessons[e].lecturers).items(),
                        key=lambda kv: (-kv[1], kv[0]))
        for lec, _ in ranked:
            per_day = Counter()
            for e, (t, r) in x0.items():
                if lec in inst.lessons[e].lecturers:
                    for d in inst.days_of(e, t):
                        per_day[d] += inst.hours_on_day(e, t, d)
            days = sorted(d for d, _ in sorted(per_day.items(), key=lambda kv: (-kv[1], kv[0]))[:2])
            dis = Disruption("participant_unavailable", participant=lec,
                             slots=[(d, p) for d in days for p in range(1, H + 1)],
                             note=f"lecturer {lec} absent on days {days}")
            if _still_possible(inst, dis):
                return HeavyScenario(sid, TITLES[sid], [dis], notes={"lecturer": lec, "days": days})
        raise RuntimeError("no lecturer can be absent for two days without an impossible lesson")
    if sid == "H3":
        hour = 3
        days = sorted(rng.sample(range(1, D + 1), 2))
        return HeavyScenario(sid, TITLES[sid],
                             [Disruption("time_unavailable", slots=[(d, hour) for d in days],
                                         note=f"hour {hour} closed on days {days}")],
                             notes={"days": days, "hour": hour})
    if sid == "H4":
        # the two largest rooms whose halving still leaves every lesson a room
        caps: dict[str, int] = {}
        for r in sorted(inst.rooms.values(), key=lambda r: (-r.capacity, r.id)):
            trial = {**caps, r.id: max(1, r.capacity // 2)}
            if _still_possible(inst, Disruption("capacity_change", capacities=trial)):
                caps = trial
            if len(caps) == 2:
                break
        return HeavyScenario(sid, TITLES[sid],
                             [Disruption("capacity_change", capacities=caps,
                                         note=f"capacities halved: {caps}")], notes={"capacities": caps})
    if sid == "H5":
        lessons = sorted(inst.lessons)
        dis = []
        for _ in range(10):
            e = rng.choice(lessons)
            if inst.patterns is not None:        # ITC: request another of its time patterns
                opts = sorted(inst.lessons[e].starts - {x0[e][0]}) or sorted(inst.lessons[e].starts)
                dis.append(Disruption("move_request", lesson=e, start=rng.choice(opts),
                                      voluntary=True))
                continue
            dur = inst.lessons[e].duration
            dis.append(Disruption("move_request", lesson=e, day=rng.randint(1, D),
                                  hour=rng.randint(1, H - dur + 1), voluntary=True))
        return HeavyScenario(sid, TITLES[sid], dis)
    if sid == "H6":
        return HeavyScenario(sid, TITLES[sid], generate_sequence(inst, seed * 1000 + 6, steps=20))
    if sid == "H8":
        hours = _hours_by(inst, x0, lambda e, r: inst.lessons[e].groups)
        cands = [(h, g) for g, h in hours.items()
                 if h >= 3 and max(inst.min_hours(e) for e in inst.lessons
                                   if g in inst.lessons[e].groups) <= min(H, h - 1)]
        h, g = max(cands) if cands else max((h, g) for g, h in hours.items())
        # keep exactly h-1 hours available, in blocks from hour 1 of each day
        keep, left = set(), h - 1
        for d in range(1, D + 1):
            take = min(H, left)
            keep |= {(d, p) for p in range(1, take + 1)}
            left -= take
            if left <= 0:
                break
        closed = [(d, p) for d in range(1, D + 1) for p in range(1, H + 1) if (d, p) not in keep]
        return HeavyScenario(sid, TITLES[sid],
                             [Disruption("participant_unavailable", participant=g, slots=closed,
                                         note=f"group {g} needs {h} h, only {h - 1} h left")],
                             notes={"group": g, "needs": h, "available": h - 1})
    if sid == "H9":
        # busiest (room, day) whose closure keeps every lesson possible and hits no full lock
        locked = {(x0[e][1], d) for e, lk in locks.items() if lk == "A"
                  for d in inst.days_of(e, x0[e][0])}
        load: Counter = Counter()
        for e, (t, r) in x0.items():
            for d in inst.days_of(e, t):
                load[(r, d)] += inst.hours_on_day(e, t, d)
        for (room, day), _ in sorted(load.items(), key=lambda kv: (-kv[1], kv[0])):
            if (room, day) in locked:
                continue
            dis = Disruption("room_unavailable", room=room, slots=[(day, p) for p in range(1, H + 1)],
                             note=f"room {room} closed on day {day}")
            if _still_possible(inst, dis):
                return HeavyScenario(sid, TITLES[sid], [dis], notes={"room": room, "day": day},
                                     gap_protect=GAP_PROTECT[sid])
        raise RuntimeError("no room-day can be closed without an impossible lesson or a lock conflict")
    raise ValueError(f"unknown scenario {sid}")
