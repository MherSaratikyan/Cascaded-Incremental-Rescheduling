"""Incremental cascaded rescheduling of dynamic university timetables with
protected soft constraints and lesson locks (M. Saratikyan).

Quick use::

    from cascade import load_scenario, TimetableStore, handle_request
    sc = load_scenario("examples/chain_example.json")
    store = TimetableStore(sc.instance, sc.timetable, sc.initial, sc.locks,
                           sc.protected, sc.gap_protected)
    h = handle_request(store, sc.disruptions[0], sc.params)
    print(h.explanation("en")); print(h.explanation("hy"))
"""
from .context import MoveRequest, StepContext
from .disruptions import Disruption, apply
from .io import Scenario, load_scenario
from .messages import explain
from .model import Instance, Lesson, Participant, Room, SoftConstraint, Timetable
from .params import Params
from .procedure import Handled, TimetableStore, VersionConflict, handle_request
from .rescheduler import STRATEGIES, Strategy, reschedule
from .result import Correction, Iteration, Outcome, Result

__all__ = [
    "MoveRequest", "StepContext", "Disruption", "apply", "Scenario", "load_scenario",
    "explain", "Instance", "Lesson", "Participant", "Room", "SoftConstraint", "Timetable",
    "Params", "Handled", "TimetableStore", "VersionConflict", "handle_request",
    "STRATEGIES", "Strategy", "reschedule", "Correction", "Iteration", "Outcome", "Result",
]
