"""Disruptions Delta^(k+1) (Section 7): room / lecturer / group / time-slot
unavailability, capacity change, and move requests."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from .context import MoveRequest
from .model import Instance

KINDS = ("room_unavailable", "participant_unavailable", "time_unavailable",
         "capacity_change", "move_request")


@dataclass
class Disruption:
    kind: str
    room: Optional[str] = None
    participant: Optional[str] = None
    lesson: Optional[str] = None
    slots: Optional[list[tuple[int, int]]] = None    # (day, hour), 1-based; None = whole week
    capacity: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    voluntary: bool = True
    note: str = ""

    @property
    def external(self) -> bool:
        return self.kind != "move_request" or not self.voluntary

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if v not in (None, "")}
        if self.kind != "move_request":
            d.pop("voluntary", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Disruption":
        d = dict(d)
        kind = d.pop("type", None) or d.pop("kind")
        if kind not in KINDS:
            raise ValueError(f"unknown disruption kind {kind!r}")
        if d.get("slots") is not None:
            d["slots"] = [tuple(s) for s in d["slots"]]
        return cls(kind=kind, **d)


def _slots(inst: Instance, slots) -> set[int]:
    if slots is None:
        return set(range(inst.num_slots))
    return {inst.slot(d, h) for d, h in slots}


def apply(inst: Instance, d: Disruption) -> tuple[Instance, Optional[MoveRequest]]:
    """Returns the instance after the disruption and, for a move request, the
    request (which restricts Omega_e to the start t*)."""
    new = inst.copy()
    if d.kind == "room_unavailable":
        new.rooms[d.room].unavailable |= _slots(inst, d.slots)
    elif d.kind == "participant_unavailable":
        new.participants[d.participant].unavailable |= _slots(inst, d.slots)
    elif d.kind == "time_unavailable":
        new.closed |= _slots(inst, d.slots)
    elif d.kind == "capacity_change":
        new.rooms[d.room].capacity = d.capacity
    elif d.kind == "move_request":
        return new, MoveRequest(d.lesson, inst.slot(d.day, d.hour), d.room, d.voluntary)
    else:
        raise ValueError(d.kind)
    return new, None
