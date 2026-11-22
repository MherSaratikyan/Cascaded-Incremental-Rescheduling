"""JSON load/save.  Days and hours are 1-based in files, slots are internal."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .disruptions import Disruption
from .model import (Instance, Lesson, Participant, Resource, Room, SoftConstraint, TimePattern,
                    Timetable)
from .params import Params


def _slots_in(inst_days: int, inst_hours: int, pairs) -> set[int]:
    return {(d - 1) * inst_hours + (h - 1) for d, h in (pairs or [])}


def _slots_out(inst: Instance, slots) -> list[list[int]]:
    return [list(inst.day_hour(t)) for t in sorted(slots)]


def instance_from_dict(d: dict[str, Any]) -> Instance:
    D, H = int(d["days"]), int(d["hours"])
    br = d.get("breaks", [])
    if isinstance(br, dict):
        breaks = {int(k): frozenset(v) for k, v in br.items()}
    else:
        breaks = {day: frozenset(br) for day in range(1, D + 1)}
    rooms = {r["id"]: Room(r["id"], int(r["capacity"]), _slots_in(D, H, r.get("unavailable")))
             for r in d["rooms"]}
    parts: dict[str, Participant] = {}
    for kind, key in (("lecturer", "lecturers"), ("group", "groups")):
        for p in d.get(key, []):
            if p["id"] in parts:
                raise ValueError(f"duplicate participant id {p['id']}")
            parts[p["id"]] = Participant(p["id"], kind, int(p.get("size", 0)),
                                         _slots_in(D, H, p.get("unavailable")),
                                         float(p.get("weight", 1.0)))
    resources = {q["id"]: Resource(q["id"], int(q["capacity"])) for q in d.get("resources", [])}
    patterns = None
    if d.get("patterns") is not None:
        patterns = [TimePattern(tuple(sorted(_slots_in(D, H, p["cells"]))), p.get("label", ""))
                    for p in d["patterns"]]
    lessons = {}
    for l in d["lessons"]:
        starts = l.get("starts")
        if starts is None:
            st = None
        elif patterns is not None:
            st = frozenset(int(x) for x in starts)          # pattern ids
        else:
            st = frozenset(_slots_in(D, H, starts))         # (day, hour) pairs
        lessons[l["id"]] = Lesson(
            id=l["id"], duration=int(l.get("duration", 1)),
            lecturers=tuple(l.get("lecturers", [])), groups=tuple(l.get("groups", [])),
            course=l.get("course"),
            rooms=tuple(l["rooms"]) if l.get("rooms") is not None else None,
            starts=st, resources=tuple(l.get("resources", [])))
    soft = {}
    for s in d.get("soft", []):
        soft[s["id"]] = SoftConstraint(
            id=s["id"], kind=s["kind"], weight=float(s.get("weight", 1.0)),
            lecturer=s.get("lecturer"), unwanted=frozenset(_slots_in(D, H, s.get("unwanted"))),
            day=s.get("day"), cap=s.get("cap"), course=s.get("course"), home=s.get("home"))
    inst = Instance(name=d.get("name", "instance"), days=D, hours=H, rooms=rooms,
                    participants=parts, lessons=lessons, breaks=breaks, resources=resources,
                    links=[tuple(x) for x in d.get("links", [])], soft=soft,
                    closed=_slots_in(D, H, d.get("closed")), patterns=patterns,
                    start_hour=int(d.get("start_hour", 1)),
                    period_minutes=int(d.get("period_minutes", 60)), meta=d.get("meta", {}))
    inst.validate()
    return inst


def instance_to_dict(inst: Instance) -> dict[str, Any]:
    br_sets = {d: sorted(v) for d, v in inst.breaks.items() if v}
    if br_sets and len({tuple(v) for v in br_sets.values()}) == 1 and len(br_sets) == inst.days:
        breaks: Any = next(iter(br_sets.values()))
    else:
        breaks = {str(k): v for k, v in br_sets.items()}

    def part(p: Participant) -> dict:
        out: dict[str, Any] = {"id": p.id}
        if p.kind == "group":
            out["size"] = p.size
        if p.unavailable:
            out["unavailable"] = _slots_out(inst, p.unavailable)
        if p.weight != 1.0:
            out["weight"] = p.weight
        return out

    lessons = []
    for l in inst.lessons.values():
        o: dict[str, Any] = {"id": l.id, "duration": l.duration, "lecturers": list(l.lecturers),
                             "groups": list(l.groups)}
        if l.course is not None:
            o["course"] = l.course
        if l.rooms is not None:
            o["rooms"] = list(l.rooms)
        if l.starts is not None:
            o["starts"] = (sorted(l.starts) if inst.patterns is not None
                           else _slots_out(inst, l.starts))
        if l.resources:
            o["resources"] = list(l.resources)
        lessons.append(o)
    soft = []
    for s in inst.soft.values():
        o = {"id": s.id, "kind": s.kind}
        if s.weight != 1.0:
            o["weight"] = s.weight
        if s.kind == "pref":
            o.update(lecturer=s.lecturer, unwanted=_slots_out(inst, s.unwanted))
        elif s.kind == "load":
            o.update(lecturer=s.lecturer, day=s.day, cap=s.cap)
        else:
            o.update(course=s.course, home=s.home)
        soft.append(o)
    out = {
        "name": inst.name, "days": inst.days, "hours": inst.hours, "breaks": breaks,
        "rooms": [{"id": r.id, "capacity": r.capacity,
                   **({"unavailable": _slots_out(inst, r.unavailable)} if r.unavailable else {})}
                  for r in inst.rooms.values()],
        "lecturers": [part(p) for p in inst.participants.values() if p.kind == "lecturer"],
        "groups": [part(p) for p in inst.participants.values() if p.kind == "group"],
        "lessons": lessons,
    }
    if inst.resources:
        out["resources"] = [{"id": q.id, "capacity": q.capacity} for q in inst.resources.values()]
    if inst.links:
        out["links"] = [list(x) for x in inst.links]
    if soft:
        out["soft"] = soft
    if inst.closed:
        out["closed"] = _slots_out(inst, inst.closed)
    if inst.patterns is not None:
        out["patterns"] = [{"cells": _slots_out(inst, p.cells), "label": p.label}
                           for p in inst.patterns]
        out["start_hour"] = inst.start_hour
        out["period_minutes"] = inst.period_minutes
    if inst.meta:
        out["meta"] = inst.meta
    return out


def timetable_from_dict(inst: Instance, d: dict[str, Any]) -> Timetable:
    """[day, hour, room] for slot instances, [time id, room] for pattern instances."""
    return {e: ((int(v[0]), v[1]) if len(v) == 2 else (inst.slot(int(v[0]), int(v[1])), v[2]))
            for e, v in d.items()}


def timetable_to_dict(inst: Instance, tt: Timetable) -> dict[str, list]:
    if inst.patterns is not None:
        return {e: [t, r] for e, (t, r) in sorted(tt.items())}
    return {e: [*inst.day_hour(t), r] for e, (t, r) in sorted(tt.items())}


@dataclass
class Scenario:
    description: str
    instance: Instance
    timetable: Timetable
    initial: Timetable
    locks: dict[str, str] = field(default_factory=dict)
    protected: dict[str, str] = field(default_factory=dict)
    gap_protected: Optional[set[str]] = None
    params: Params = field(default_factory=Params)
    disruptions: list[Disruption] = field(default_factory=list)


def load_scenario(path: str | Path) -> Scenario:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    inst = instance_from_dict(d["instance"])
    tt = timetable_from_dict(inst, d["timetable"])
    x0 = timetable_from_dict(inst, d["initial"]) if "initial" in d else dict(tt)
    gp = set(d["gap_protected"]) if "gap_protected" in d else None
    return Scenario(description=d.get("description", ""), instance=inst, timetable=tt,
                    initial=x0, locks=d.get("locks", {}), protected=d.get("protected", {}),
                    gap_protected=gp, params=Params(**d.get("params", {})),
                    disruptions=[Disruption.from_dict(x) for x in d.get("disruptions", [])])


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
