"""Claims 1 and 2 on small generated instances, plus IO round-trips."""
from dataclasses import replace

import pytest

from cascade import Outcome, StepContext, io
from cascade import evaluate as ev
from cascade.baselines import run_method
from cascade.cpsat_model import SubModel
from cascade.dependency import DependencyGraph
from cascade.disruptions import apply
from cascade.explain import Deadline
from cascade.params import Params
from cascade.rescheduler import lexmin
from experiments.generator import generate_sequence, load_or_generate, sample_regulation

PARAMS = Params(workers=4, time_limit=30, explain_time_limit=10, gap_protect="groups")


@pytest.fixture(scope="session")
def xs(tmp_path_factory):
    cache = tmp_path_factory.mktemp("instances")
    inst, x0 = load_or_generate("XS", 3, cache, time_limit=20, workers=4)
    protected, locks = sample_regulation(inst, 3, rho=0.3, lam=0.15)
    return inst, x0, protected, locks


def contexts(xs, n=6):
    inst, x0, protected, locks = xs
    for d in generate_sequence(inst, 7, steps=n):
        after, req = apply(inst, d)
        yield d, StepContext.build(inst, after, x0, x0, PARAMS, locks, protected, None, req)


def test_claim1_frozen_threshold_expansion_never_worsens(xs):
    checked = 0
    for _, c in contexts(xs, 10):
        if not c.f0 or c.lock_conflicts:
            continue
        F = set(c.f0)
        top = c.ladder[-1]
        first = SubModel(c, F, budget=top, assumptions=False)
        r = first.minimize(first.Pk, 10)
        if not r.feasible:
            continue
        p_bar = r.value + c.delta
        graph = DependencyGraph(c)
        tt, obj = lexmin(c, F, p_bar, Deadline(30), r.timetable, [])
        for _ in range(3):
            F2, _ = graph.expand(F, [], 2)
            if F2 == F:
                break
            tt2, obj2 = lexmin(c, F2, p_bar, Deadline(30), tt, [])
            assert obj2 <= obj, (obj, obj2)
            F, tt, obj = F2, tt2, obj2
        checked += 1
    assert checked >= 2


def test_claim2_full_expansion_equals_regulation_limited_reoptimisation(xs):
    compared = 0
    for _, c in contexts(xs, 8):
        if c.lock_conflicts or not c.f0:
            continue
        full = SubModel(c, c.all_free, budget=c.ladder[-1], assumptions=False)
        reg = SubModel(c, c.inst.lessons, assumptions=False)     # fully locked as variables + locks
        a, b = full.minimize(full.Pk, 20), reg.minimize(reg.Pk, 20)
        assert a.feasible == b.feasible
        if a.feasible:
            assert a.value == b.value
        compared += 1
    assert compared >= 3


def test_proposed_method_agrees_with_b4_on_outcome(xs):
    for d, c in contexts(xs, 6):
        r7, r4 = run_method("B7", c), run_method("B4", c)
        solved7, solved4 = r7.outcome == Outcome.SOLVED, r4.outcome == Outcome.SOLVED
        assert solved7 == solved4, (d, r7.outcome, r4.outcome)
        if solved7:
            assert r7.metrics["hard_violations"] == 0
            assert r7.metrics["protected_violations"] == 0
            assert r7.metrics["lock_violations"] == 0
            # B4 is the global minimum of P_k over the regulation-feasible set
            if c.f0:
                assert r4.p_star <= r7.p_star
            else:                         # nothing displaced: x^(k) is kept by both
                assert r7.metrics["P_k"] == r4.metrics["P_k"] == 0


def test_b6_heuristic_produces_valid_or_failure(xs):
    for _, c in contexts(xs, 3):
        r = run_method("B6", replace(c, params=c.params.with_(time_limit=3)))
        assert r.outcome in (Outcome.SOLVED, Outcome.ALGORITHMIC_FAILURE)
        if r.outcome == Outcome.SOLVED:
            assert r.metrics["hard_violations"] == 0


def test_io_roundtrip(xs):
    inst, x0, _, _ = xs
    d = io.instance_to_dict(inst)
    inst2 = io.instance_from_dict(d)
    assert io.instance_to_dict(inst2) == d
    tt = io.timetable_from_dict(inst2, io.timetable_to_dict(inst, x0))
    assert tt == x0
    assert ev.hard_violations(inst2, tt).ok
