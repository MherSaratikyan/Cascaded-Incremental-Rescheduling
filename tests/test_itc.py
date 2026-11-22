"""ITC 2019 loader (adaptation A) on the hand-written sample examples/itc_sample.xml."""
import pytest

from cascade import Disruption, Outcome, StepContext, io, reschedule
from cascade import evaluate as ev
from cascade.baselines import run_method
from cascade.disruptions import apply
from cascade.itc import fidelity_text, load_itc
from cascade.params import Params
from conftest import ROOT
from experiments.generator import initial_timetable

SAMPLE = ROOT / "examples" / "itc_sample.xml"
PARAMS = Params(time_limit=20, explain_time_limit=10, workers=2)


@pytest.fixture(scope="module")
def itc():
    inst = load_itc(SAMPLE)
    x0 = initial_timetable(inst, 20, 2)
    return inst, x0


def labels(inst, e):
    return sorted(inst.time_label(t) for t in inst.lessons[e].starts)


def test_mapping_keeps_and_counts_what_it_drops(itc):
    """Classes become lessons, required SameAttendees / NotOverlap become
    attendees, every other distribution is dropped and counted; partial-week
    room unavailability, travel times and students are reported as ignored."""
    inst, _ = itc
    m = inst.meta["itc"]
    assert m["lessons"] == 6 and m["attendees"] == 4
    assert m["distributions"] == {
        "dropped: MaxDays (required)": 1, "dropped: Precedence (required)": 1,
        "dropped: SameRoom (soft)": 1, "kept: NotOverlap (required)": 1,
        "kept: SameAttendees (required)": 3}
    assert m["room_unavailable_partial_weeks_dropped"] == 1
    assert m["travel_entries_ignored"] == 2 and m["students_ignored"] == 2
    assert m["time_options_partial_weeks_merged"] == 2
    assert "weeks merged" in fidelity_text(inst)


def test_multi_day_time_options_become_patterns(itc):
    """'MWF 8:00-8:50' occupies 2 half-hour periods on each of Mon, Wed, Fri;
    a room-less class gets a private virtual room; attendees follow the
    distributions (class 1 is taught by A0 and is the parent in A1)."""
    inst, _ = itc
    assert labels(inst, "C1") == ["MoWeFr 08:00-08:50", "TuTh 09:00-10:15"]
    mwf = next(t for t in inst.lessons["C1"].starts if inst.time_label(t).startswith("MoWeFr"))
    assert inst.days_of("C1", mwf) == {1, 3, 5}
    assert len(inst.cells("C1", mwf)) == 6
    assert inst.lessons["C4"].rooms == ("V4",)
    assert set(inst.lessons["C1"].lecturers) == {"A0", "A1"}
    assert inst.lessons["C5"].groups == ("N3",) == inst.lessons["C6"].groups


def test_room_unavailability_covering_all_weeks_is_kept(itc):
    """Room 3 is unavailable Mon+Tue 8:30-10:30 in every week -> those cells are
    closed; its odd-week Friday unavailability was dropped."""
    inst, _ = itc
    un = inst.rooms["R3"].unavailable
    assert un and {c // inst.hours + 1 for c in un} == {1, 2}


def test_json_roundtrip_of_pattern_instance(itc):
    """Pattern instances (and [time id, room] timetables) survive JSON."""
    inst, x0 = itc
    d = io.instance_to_dict(inst)
    inst2 = io.instance_from_dict(d)
    assert io.instance_to_dict(inst2) == d
    assert io.timetable_from_dict(inst2, io.timetable_to_dict(inst, x0)) == x0


def test_initial_timetable_is_hard_feasible(itc):
    """x^(0) for the mapped instance places every class without a clash."""
    inst, x0 = itc
    assert x0 is not None and len(x0) == 6
    assert ev.hard_violations(inst, x0).ok


def test_cascade_repairs_a_room_closure_on_itc_data(itc):
    """Closing room 1 for the week displaces the classes in it; B7 repairs the
    timetable through a chain, without breaking any kept ITC constraint, and
    B4 (full re-optimisation) agrees that a repair exists."""
    inst, x0 = itc
    after, req = apply(inst, Disruption("room_unavailable", room="R1"))
    c = StepContext.build(inst, after, x0, x0, PARAMS, {}, {}, None, req)
    r7, r4 = reschedule(c), run_method("B4", c)
    assert r7.outcome == r4.outcome == Outcome.SOLVED
    assert r7.metrics["hard_violations"] == 0
    assert all(r7.timetable[e][1] != "R1" for e in r7.timetable)
    assert r7.metrics["P_k"] >= r4.p_star


def test_voluntary_request_for_another_time_pattern(itc):
    """A move request names one of the class's own ITC time patterns."""
    inst, x0 = itc
    e = "C2"
    target = next(t for t in sorted(inst.lessons[e].starts) if t != x0[e][0])
    after, req = apply(inst, Disruption("move_request", lesson=e, start=target, voluntary=True))
    c = StepContext.build(inst, after, x0, x0, PARAMS, {}, {}, None, req)
    r = reschedule(c)
    assert r.outcome == Outcome.SOLVED
    assert r.timetable[e][0] == target
