"""Steps 6-7 of Table 1: version check, approval, notifications."""
from cascade import Disruption, Outcome, handle_request

from conftest import make_store

VOLUNTARY = Disruption("move_request", lesson="A", day=1, hour=2, room="R1", voluntary=True)


def test_voluntary_chain_requires_approval(chain):
    store = make_store(chain)
    seen = []

    def approver(result, ctx):
        seen.append(result.affected_lecturers)
        return True

    h = handle_request(store, VOLUNTARY, chain.params, approver=approver)
    assert h.result.outcome == Outcome.SOLVED
    assert h.result.requires_approval
    assert seen == [["LB", "LC"]]
    assert h.ctx.delta == 0
    assert store.version == 1
    assert set(h.notifications) >= {"LA", "LB", "LC", "GA", "GB", "GC"}


def test_rejected_chain_is_not_committed(chain):
    store = make_store(chain)
    before = dict(store.timetable)
    h = handle_request(store, VOLUNTARY, chain.params, approver=lambda r, c: False)
    assert h.result.outcome == Outcome.REJECTED_BY_APPROVER
    assert store.timetable == before
    assert store.version == 0
    assert "իրավասու անձի" in h.explanation("hy")


def test_version_conflict_restarts_from_step_1(chain):
    store = make_store(chain)

    def concurrent_writer(attempt):
        if attempt == 0:
            store.force_write(note="someone else edited the timetable")

    h = handle_request(store, chain.disruptions[0], chain.params, before_commit=concurrent_writer)
    assert h.result.outcome == Outcome.SOLVED
    assert h.attempts == 2
    assert store.version == 2


def test_external_failure_persists_instance_change(chain):
    store = make_store(chain, locks={"C": "A"})
    h = handle_request(store, chain.disruptions[0], chain.params)
    assert h.result.outcome == Outcome.REGULATORY_INFEASIBLE
    assert store.version == 1
    assert store.instance.rooms["R0"].unavailable        # the room is closed anyway
    assert store.timetable == chain.timetable            # x^(k) kept
