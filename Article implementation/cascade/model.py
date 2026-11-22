"""Instance data model (Section 3 of the paper).

Time is discretised into slots t = (d, p) with day d in 1..D and hour p in 1..H.
Internally a slot is the integer ``t = (d - 1) * H + (p - 1)``.

Sets of the paper and their counterparts here:

* E  -> ``Instance.lessons``        (a joint lesson of several groups is ONE lesson)
* T  -> ``range(Instance.num_slots)``
* R  -> ``Instance.rooms``
* L  -> ``Instance.lecturers``
* K  -> ``Instance.groups``
* Pi_d^b (break hours) -> ``Instance.breaks``
* linked lessons (simultaneous pairs, the set "L" of (2)) -> ``Instance.links``
* special resources q with capacity kappa_q -> ``Instance.resources``
* soft constraints S -> ``Instance.soft``
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, Optional

Assignment = tuple[int, str]          # (start slot t, room id)
Timetable = dict[str, Assignment]      # lesson id -> assignment a(e; x)


@dataclass
class Room:
    id: str
    capacity: int
    unavailable: set[int] = field(default_factory=set)


@dataclass
class Participant:
    """A lecturer or a student group (the set U = K u L)."""
    id: str
    kind: str                      # "lecturer" | "group"
    size: int = 0                  # number of students (groups only)
    unavailable: set[int] = field(default_factory=set)
    weight: float = 1.0            # omega_u in G(x) and Delta G^+


@dataclass
class Resource:
    id: str
    capacity: int                  # kappa_q


@dataclass
class Lesson:
    id: str
    duration: int                  # delta_e (hours)
    lecturers: tuple[str, ...]     # L_e
    groups: tuple[str, ...]        # K_e
    course: Optional[str] = None
    rooms: Optional[tuple[str, ...]] = None    # allowed rooms (None = any)
    starts: Optional[frozenset[int]] = None    # allowed start slots (None = any)
    resources: tuple[str, ...] = ()

    @property
    def participants(self) -> tuple[str, ...]:
        return self.lecturers + self.groups


@dataclass
class SoftConstraint:
    """A soft constraint s with normalised violation v_s(x) = num_s(x) / den_s.

    kinds
    -----
    ``pref``  lecturer time preference:  unwanted hours taught / hours taught
    ``load``  daily workload:            max(0, hours on day - cap) / (H - cap)
    ``room``  room stability of course:  lessons outside home room / lessons
    """
    id: str
    kind: str
    weight: float = 1.0                       # pi_s
    lecturer: Optional[str] = None            # pref, load
    unwanted: frozenset[int] = frozenset()    # pref
    day: Optional[int] = None                 # load (1-based)
    cap: Optional[int] = None                 # load
    course: Optional[str] = None              # room
    home: Optional[str] = None                # room


@dataclass
class Instance:
    name: str
    days: int
    hours: int
    rooms: dict[str, Room]
    participants: dict[str, Participant]
    lessons: dict[str, Lesson]
    breaks: dict[int, frozenset[int]] = field(default_factory=dict)  # day -> break hours
    resources: dict[str, Resource] = field(default_factory=dict)
    links: list[tuple[str, str]] = field(default_factory=list)
    soft: dict[str, SoftConstraint] = field(default_factory=dict)
    closed: set[int] = field(default_factory=set)   # slots closed for every lesson

    # ------------------------------------------------------------------ time
    @property
    def num_slots(self) -> int:
        return self.days * self.hours

    def slot(self, day: int, hour: int) -> int:
        if not (1 <= day <= self.days and 1 <= hour <= self.hours):
            raise ValueError(f"slot ({day},{hour}) outside {self.days}x{self.hours}")
        return (day - 1) * self.hours + (hour - 1)

    def day_hour(self, t: int) -> tuple[int, int]:
        return t // self.hours + 1, t % self.hours + 1

    def cov(self, t: int, duration: int) -> Optional[range]:
        """cov(t, delta): the delta consecutive slots starting at t, or None when
        the lesson would run past the end of the day."""
        if (t % self.hours) + duration > self.hours:
            return None
        return range(t, t + duration)

    def is_break(self, day: int, hour: int) -> bool:
        return hour in self.breaks.get(day, frozenset())

    # ---------------------------------------------------------- participants
    @property
    def lecturers(self) -> dict[str, Participant]:
        return {k: v for k, v in self.participants.items() if v.kind == "lecturer"}

    @property
    def groups(self) -> dict[str, Participant]:
        return {k: v for k, v in self.participants.items() if v.kind == "group"}

    def lesson_size(self, lesson: Lesson) -> int:
        return sum(self.participants[g].size for g in lesson.groups)

    def lessons_of(self, participant: str) -> list[str]:
        return [e for e, l in self.lessons.items() if participant in l.participants]

    def lessons_of_course(self, course: str) -> list[str]:
        return [e for e, l in self.lessons.items() if l.course == course]

    def copy(self) -> "Instance":
        return copy.deepcopy(self)

    # --------------------------------------------------------------- checks
    def validate(self) -> None:
        for e, l in self.lessons.items():
            if l.duration < 1 or l.duration > self.hours:
                raise ValueError(f"lesson {e}: bad duration {l.duration}")
            for u in l.participants:
                if u not in self.participants:
                    raise ValueError(f"lesson {e}: unknown participant {u}")
            for r in l.rooms or ():
                if r not in self.rooms:
                    raise ValueError(f"lesson {e}: unknown room {r}")
            for q in l.resources:
                if q not in self.resources:
                    raise ValueError(f"lesson {e}: unknown resource {q}")
        for a, b in self.links:
            if a not in self.lessons or b not in self.lessons:
                raise ValueError(f"link ({a},{b}) references unknown lesson")
        for s in self.soft.values():
            if s.kind not in ("pref", "load", "room"):
                raise ValueError(f"soft constraint {s.id}: unknown kind {s.kind}")
            if s.kind == "load" and (s.cap is None or s.cap >= self.hours):
                raise ValueError(f"soft constraint {s.id}: cap must be < H")


def link_components(lessons: Iterable[str], links: list[tuple[str, str]]) -> dict[str, frozenset[str]]:
    """Connected components of the 'linked lessons' relation: cl(.) of (12)."""
    parent = {e: e for e in lessons}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[str, set[str]] = {}
    for e in parent:
        groups.setdefault(find(e), set()).add(e)
    out: dict[str, frozenset[str]] = {}
    for comp in groups.values():
        fc = frozenset(comp)
        for e in comp:
            out[e] = fc
    return out
