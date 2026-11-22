"""Section 6 worked examples, reproduced exactly."""
from cascade import Disruption, Outcome, StepContext, handle_request, reschedule
from cascade.baselines import run_method
from cascade.disruptions import apply
from cascade.rescheduler import STRATEGIES

from conftest import make_store


def slot(inst, a):
    d, h = inst.day_hour(a[0])
    return (h, a[1])


def ctx_for(sc, disruption=None, **over):
    d = disruption or sc.disruptions[0]
    after, req = apply(sc.instance, d)
    return StepContext.build(sc.instance, after, sc.timetable, sc.initial, sc.params,
                             over.get("locks", sc.locks), over.get("protected", sc.protected),
                             over.get("gap_protected", sc.gap_protected), req)


def test_chain_repair_found_through_cascade(chain):
    h = handle_request(make_store(chain), chain.disruptions[0], chain.params)
    r = h.result
    assert r.outcome == Outcome.SOLVED
    inst = h.ctx.inst
    assert {e: slot(inst, a) for e, a in r.timetable.items()} == \
        {"A": (2, "R1"), "B": (4, "R1"), "C": (5, "R1")}
    # F_0 = {A} and F_1 = {A, B} infeasible; the core names C; F_2 = {A, B, C}
    assert [it.lessons for it in r.trace[:3]] == [["A"], ["A", "B"], ["A", "B", "C"]]
    assert ("fix", "B") in r.trace[0].core
    assert ("fix", "C") in r.trace[1].core and ("prot", "sB") in r.trace[1].core
    assert r.metrics["protected_violations"] == 0
    assert r.metrics["P_k"] == 3 * chain.params.w_time


def test_local_check_rejects_the_move(chain):
    r = run_method("B1", ctx_for(chain))
    assert r.outcome == Outcome.REJECTED_LOCAL


def test_hard_only_chain_violates_protected_preference(chain):
    c = ctx_for(chain)
    r = run_method("B2", c)
    assert r.outcome == Outcome.SOLVED
    assert slot(c.inst, r.timetable["B"]) == (3, "R1")
    assert r.metrics["protected_violations"] == 1


def test_regulation_limited_full_reoptimisation_agrees(chain):
    c = ctx_for(chain)
    r4, r7 = run_method("B4", c), run_method("B7", c)
    assert r4.outcome == r7.outcome == Outcome.SOLVED
    assert r4.metrics["P_k"] == r7.metrics["P_k"]


def test_locked_c_gives_regulatory_infeasibility_with_mus_and_mcs(chain):
    h = handle_request(make_store(chain, locks={"C": "A"}), chain.disruptions[0], chain.params)
    r = h.result
    assert r.outcome == Outcome.REGULATORY_INFEASIBLE
    assert set(r.mus) == {("lock", "C"), ("prot", "sB")}
    assert r.mus_minimal
    assert {tuple(c.items) for c in r.mcs} == {(("lock", "C"),), (("prot", "sB"),)}
    assert all(c.verified for c in r.mcs)
    hy = h.explanation("hy")
    assert "C դասի ամրագրումը պետք է հանվի" in hy
    assert "lock on lesson C must be lifted" in h.explanation("en")


def test_lock_conflict_without_search(chain):
    d = Disruption("move_request", lesson="C", day=1, hour=5, voluntary=True)
    c = ctx_for(chain, d, locks={"C": "A"})
    r = reschedule(c)
    assert r.outcome == Outcome.LOCK_CONFLICT
    assert r.lock_conflicts == ["C"]
    assert r.trace == []


def test_physical_infeasibility(chain):
    store = make_store(chain)
    first = Disruption("participant_unavailable", participant="LB", slots=[(1, 3), (1, 4)])
    assert handle_request(store, first, chain.params).result.outcome == Outcome.SOLVED
    r = handle_request(store, chain.disruptions[0], chain.params).result
    assert r.outcome == Outcome.PHYSICAL_INFEASIBLE


def test_claim3_counterexample_extra_expansion(claim3):
    h = handle_request(make_store(claim3), claim3.disruptions[0], claim3.params)
    r = h.result
    inst = h.ctx.inst
    assert r.outcome == Outcome.SOLVED
    assert r.extra_expansion == "accepted"
    assert slot(inst, r.timetable["A"])[0] == 1 and slot(inst, r.timetable["X"])[0] == 3
    assert r.metrics["dG_plus"] == 0


def test_claim3_with_zero_slack_keeps_single_move(claim3):
    params = claim3.params.with_(delta_external=0)
    h = handle_request(make_store(claim3), claim3.disruptions[0], params)
    r = h.result
    assert r.outcome == Outcome.SOLVED
    assert slot(h.ctx.inst, r.timetable["A"])[0] == 6
    assert r.extra_expansion == "rejected"
    assert r.metrics["dG_plus"] == 3
