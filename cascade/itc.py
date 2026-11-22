"""ITC 2019 problem XML -> the paper's model (adaptation "A").

ITC 2019 (https://www.itc2019.org/format) differs from the paper's model in
time, attendance and constraints, so instances are *mapped*, not loaded 1:1:

time      Each <time days start length weeks> option becomes a TimePattern on a
          weekly grid of ``period_minutes`` periods (default 30 min = 6 ITC
          slots of 5 min), covering every listed day.  Weeks are merged (a
          class meeting only in some weeks is treated as meeting every week:
          stricter than ITC).  Options that collapse to the same cells merge.
rooms     Rooms keep id and capacity; an <unavailable> entry is kept when it
          covers every week, otherwise dropped (counted).  Travel times are
          ignored (counted).  A class with room="false" gets its own virtual
          room.
attendees Every *required* SameAttendees distribution becomes an attendee
          (kind "lecturer"), every required NotOverlap one an attendee of kind
          "group": their classes may not overlap.  Students and sectioning are
          not modelled (counted).
other     All other distribution types, and all soft distributions, are dropped
          and counted per type in the fidelity report ``inst.meta["itc"]``.
soft      For each attendee, a lecturer-preference soft constraint marks the
          cells that only penalised time options would use.

The fidelity report says exactly what was kept and what was dropped.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .model import Instance, Lesson, Participant, Room, SoftConstraint, TimePattern

DAY_NAMES = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
KEPT_TYPES = ("SameAttendees", "NotOverlap")


def _bits(s: str) -> list[int]:
    return [i for i, ch in enumerate(s) if ch == "1"]


def _clock(slot: int, slot_minutes: int) -> str:
    m = slot * slot_minutes
    return f"{m // 60:02d}:{m % 60:02d}"


def _base_type(t: str) -> str:
    return t.split("(", 1)[0]


def load_itc(path: str | Path, period_minutes: int = 30) -> Instance:
    """Parse an ITC 2019 problem XML into an :class:`Instance` with time patterns.
    The fidelity report is stored in ``inst.meta["itc"]``."""
    root = ET.parse(str(path)).getroot()
    if root.tag != "problem":
        raise ValueError(f"{path}: root element is <{root.tag}>, expected <problem>")
    name = root.get("name", Path(path).stem)
    n_days = int(root.get("nrDays"))
    n_weeks = int(root.get("nrWeeks"))
    spd = int(root.get("slotsPerDay"))
    slot_minutes = 1440 // spd
    q = max(1, period_minutes // slot_minutes)           # ITC slots per period
    all_weeks = "1" * n_weeks
    rep: Counter = Counter()

    classes = []                                         # (class elem, course id)
    for course in root.iter("course"):
        for cls in course.iter("class"):
            classes.append((cls, course.get("id")))
    times = [t for cls, _ in classes for t in cls.findall("time")]
    if not times:
        raise ValueError(f"{path}: no class time options")
    t_min = min(int(t.get("start")) for t in times)
    t_max = max(int(t.get("start")) + int(t.get("length")) for t in times)
    day_start = (t_min // q) * q
    H = max(1, math.ceil((t_max - day_start) / q))

    def cells_of(days: str, start: int, length: int) -> tuple[int, ...]:
        p0 = (start - day_start) // q
        p1 = max(p0 + 1, math.ceil((start + length - day_start) / q))
        return tuple(sorted(d * H + p for d in _bits(days) for p in range(p0, min(p1, H))))

    # ------------------------------------------------------------- rooms
    rooms: dict[str, Room] = {}
    for r in root.find("rooms").findall("room"):
        un: set[int] = set()
        for u in r.findall("unavailable"):
            if u.get("weeks", all_weeks).count("1") == n_weeks:
                un.update(cells_of(u.get("days"), int(u.get("start")), int(u.get("length"))))
            else:
                rep["room_unavailable_partial_weeks_dropped"] += 1
        rep["travel_entries_ignored"] += len(r.findall("travel"))
        rooms[f"R{r.get('id')}"] = Room(f"R{r.get('id')}", int(r.get("capacity")), un)

    # ------------------------------------------------- time patterns
    patterns: list[TimePattern] = []
    pid: dict[tuple[int, ...], int] = {}
    class_starts: dict[str, set[int]] = {}
    pattern_penalty: dict[tuple[str, int], int] = {}
    for cls, _ in classes:
        cid = f"C{cls.get('id')}"
        starts: set[int] = set()
        for t in cls.findall("time"):
            rep["time_options"] += 1
            if t.get("weeks", all_weeks).count("1") != n_weeks:
                rep["time_options_partial_weeks_merged"] += 1
            start, length = int(t.get("start")), int(t.get("length"))
            cs = cells_of(t.get("days"), start, length)
            if not cs:
                rep["time_options_outside_grid"] += 1
                continue
            if cs not in pid:
                days = "".join(DAY_NAMES[d] if d < 7 else f"D{d + 1}" for d in _bits(t.get("days")))
                label = f"{days} {_clock(start, slot_minutes)}-{_clock(start + length, slot_minutes)}"
                pid[cs] = len(patterns)
                patterns.append(TimePattern(cs, label))
            k = pid[cs]
            if k in starts:
                rep["time_options_merged"] += 1
            starts.add(k)
            pen = int(t.get("penalty", 0))
            pattern_penalty[(cid, k)] = min(pen, pattern_penalty.get((cid, k), pen))
        class_starts[cid] = starts

    # ------------------------------------------------ attendees (distributions)
    members: dict[str, list[str]] = defaultdict(list)       # lesson -> attendee ids
    attendees: dict[str, Participant] = {}
    attendee_classes: dict[str, list[str]] = {}
    dist_counts: Counter = Counter()
    dists = root.find("distributions")
    for k, dist in enumerate(dists.findall("distribution") if dists is not None else []):
        typ = _base_type(dist.get("type"))
        required = dist.get("required", "false") == "true"
        ids = [f"C{c.get('id')}" for c in dist.findall("class")]
        key = f"{typ} ({'required' if required else 'soft'})"
        if required and typ in KEPT_TYPES and len(ids) >= 2:
            dist_counts[f"kept: {key}"] += 1
            aid = f"{'A' if typ == 'SameAttendees' else 'N'}{k}"
            attendees[aid] = Participant(aid, "lecturer" if typ == "SameAttendees" else "group")
            attendee_classes[aid] = ids
            for e in ids:
                members[e].append(aid)
        else:
            dist_counts[f"dropped: {key}"] += 1

    # ---------------------------------------------------------- lessons
    lessons: dict[str, Lesson] = {}
    for cls, course in classes:
        cid = f"C{cls.get('id')}"
        if not class_starts[cid]:
            rep["classes_without_time_dropped"] += 1
            continue
        room_ids = [f"R{r.get('id')}" for r in cls.findall("room")]
        if cls.get("room") == "false" or not room_ids:
            vr = f"V{cls.get('id')}"
            rooms[vr] = Room(vr, 10 ** 6)
            room_ids = [vr]
            rep["classes_without_room"] += 1
        per_day = max(len(patterns[t].cells) // max(1, len({c // H for c in patterns[t].cells}))
                      for t in class_starts[cid])
        lec = tuple(a for a in members.get(cid, ()) if attendees[a].kind == "lecturer")
        grp = tuple(a for a in members.get(cid, ()) if attendees[a].kind == "group")
        lessons[cid] = Lesson(cid, max(1, min(H, per_day)), lec, grp, f"K{course}",
                              tuple(room_ids), frozenset(class_starts[cid]))
    for aid in list(attendees):                    # drop attendees left with < 2 lessons
        if sum(1 for e in attendee_classes[aid] if e in lessons) < 2:
            del attendees[aid]
    for e, l in list(lessons.items()):
        lessons[e] = Lesson(l.id, l.duration, tuple(a for a in l.lecturers if a in attendees),
                            tuple(a for a in l.groups if a in attendees), l.course, l.rooms, l.starts)

    # ---------------------------------- soft: penalised-time preference per attendee
    soft: dict[str, SoftConstraint] = {}
    for aid, p in attendees.items():
        if p.kind != "lecturer":
            continue
        good: set[int] = set()
        bad: set[int] = set()
        for e in attendee_classes[aid]:
            for t in class_starts.get(e, ()):
                (bad if pattern_penalty.get((e, t), 0) > 0 else good).update(patterns[t].cells)
        unwanted = frozenset(bad - good)
        if unwanted:
            soft[f"pref:{aid}"] = SoftConstraint(f"pref:{aid}", "pref", 1.0, lecturer=aid,
                                                 unwanted=unwanted)

    students = root.find("students")
    n_students = len(students.findall("student")) if students is not None else 0
    inst = Instance(name=f"itc-{name}", days=n_days, hours=H, rooms=rooms, participants=attendees,
                    lessons=lessons, soft=soft, patterns=patterns,
                    start_hour=day_start * slot_minutes // 60, period_minutes=q * slot_minutes)
    inst.meta["itc"] = {
        "source": Path(path).name, "problem": name, "nrDays": n_days, "nrWeeks": n_weeks,
        "slotsPerDay": spd, "period_minutes": q * slot_minutes, "periods_per_day": H,
        "day_window": f"{_clock(day_start, slot_minutes)}-{_clock(day_start + H * q, slot_minutes)}",
        "classes": len(classes), "lessons": len(lessons), "rooms": len(rooms),
        "patterns": len(patterns), "attendees": len(attendees),
        "students_ignored": n_students, **dict(rep),
        "distributions": dict(sorted(dist_counts.items())),
    }
    inst.validate()
    return inst


def fidelity_text(inst: Instance) -> str:
    m = inst.meta.get("itc", {})
    if not m:
        return "not an ITC 2019 instance"
    lines = [f"{m['problem']} ({m['source']}): {m['classes']} classes -> {m['lessons']} lessons, "
             f"{m['rooms']} rooms, {m['patterns']} time patterns, {m['attendees']} attendees",
             f"  grid: {m['nrDays']} days x {m['periods_per_day']} periods of {m['period_minutes']} min "
             f"({m['day_window']}); weeks merged ({m['nrWeeks']} weeks)"]
    for k in ("time_options", "time_options_merged", "time_options_partial_weeks_merged",
              "time_options_outside_grid", "room_unavailable_partial_weeks_dropped",
              "travel_entries_ignored", "classes_without_room", "classes_without_time_dropped",
              "students_ignored"):
        if m.get(k):
            lines.append(f"  {k.replace('_', ' ')}: {m[k]}")
    for k, v in m.get("distributions", {}).items():
        lines.append(f"  distributions {k}: {v}")
    return "\n".join(lines)
