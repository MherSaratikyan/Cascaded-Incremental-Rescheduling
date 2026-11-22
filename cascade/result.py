"""Outcome classification and result records."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .model import Timetable


class Outcome(str, Enum):
    SOLVED = "solved"
    PHYSICAL_INFEASIBLE = "physical_infeasibility"
    REGULATORY_INFEASIBLE = "regulatory_infeasibility"
    LOCK_CONFLICT = "lock_conflict"
    ALGORITHMIC_FAILURE = "algorithmic_failure"
    REJECTED_LOCAL = "rejected_local_check"          # baseline B1 only
    REJECTED_BY_APPROVER = "rejected_by_approver"    # step 7


@dataclass
class Iteration:
    j: int
    f_size: int
    budget: Optional[int]
    status: str
    core: Optional[list] = None
    action: str = ""
    variables: int = 0
    seconds: float = 0.0
    lessons: list[str] = field(default_factory=list)


@dataclass
class Correction:
    """A minimal correction set (MCS), verified by re-solving."""
    items: list[tuple]
    verified: bool
    candidate: Optional[Timetable] = None


@dataclass
class Result:
    outcome: Outcome
    method: str
    timetable: Optional[Timetable] = None        # x^(k+1) when SOLVED
    trace: list[Iteration] = field(default_factory=list)
    p_star: Optional[int] = None
    p_bar: Optional[int] = None
    objective: Optional[tuple] = None            # (dG+, P0, G, Q_O) of (14)
    extra_expansion: str = "not_triggered"       # not_triggered | accepted | rejected
    final_f: list[str] = field(default_factory=list)
    mus: Optional[list[tuple]] = None
    mus_minimal: bool = True
    mcs: list[Correction] = field(default_factory=list)
    core: Optional[list] = None                  # last core (algorithmic failure)
    candidate: Optional[Timetable] = None        # relaxed candidate (failure)
    lock_conflicts: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    requires_approval: bool = False
    affected_lecturers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0
    metrics: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)   # elapsed seconds per phase
    stages: list = field(default_factory=list)   # (stage, status, value, bound) of P* and (14)

    def phase(self, name: str) -> None:          # replaced by the driver's timer
        pass

    @property
    def solved(self) -> bool:
        return self.outcome == Outcome.SOLVED
