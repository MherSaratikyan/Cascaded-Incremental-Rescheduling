"""Claim 5: for a fixed integral x, the b/a/g formulation (6)-(7) never
under-estimates the gaps and attains them exactly at the optimum."""
import random

import pytest

from cascade import evaluate as ev
from cascade.context import StepContext
from cascade.cpsat_model import SubModel
from cascade.model import Instance, Lesson, Participant, Room
from cascade.params import Params


def random_fixed_instance(seed: int):
    rng = random.Random(seed)
    H, D = 8, 2
    inst = Instance(name=f"gaps-{seed}", days=D, hours=H,
                    rooms={f"R{i}": Room(f"R{i}", 50) for i in range(6)},
                    participants={"G": Participant("G", "group", 10),
                                  "L": Participant("L", "lecturer")},
                    lessons={},
                    breaks={d: frozenset({5}) if rng.random() < 0.5 else frozenset()
                            for d in range(1, D + 1)})
    tt = {}
    busy: set[int] = set()
    for i in range(rng.randint(1, 6)):
        dur = rng.choice([1, 1, 2])
        for _ in range(20):
            d, p = rng.randint(1, D), rng.randint(1, H - dur + 1)
            t = (d - 1) * H + p - 1
            if not busy & set(range(t, t + dur)):
                break
        else:
            continue
        busy |= set(range(t, t + dur))
        e = f"E{i}"
        inst.lessons[e] = Lesson(e, dur, ("L",), ("G",), None, (f"R{i}",), frozenset({t}))
        tt[e] = (t, f"R{i}")
    return inst, tt


@pytest.mark.parametrize("seed", range(25))
def test_gap_formulation_is_exact(seed):
    """Claim 5 on 25 random fixed timetables (1-6 lessons of one group over two
    days, random lunch break): minimising G over the b/a/g variables of (6)-(7)
    gives exactly the true gap count, and forcing G one lower is infeasible -
    so the linear formulation is exact without integrality on b, a, g."""
    inst, tt = random_fixed_instance(seed)
    if not tt:
        pytest.skip("empty")
    ctx = StepContext.build(inst, inst, tt, tt, Params(workers=1), gap_protected=set())
    sub = SubModel(ctx, inst.lessons, protected=False, locks=False, assumptions=False)
    r = sub.minimize(sub.G, 10)
    assert r.status == "OPTIMAL"
    assert r.value == ev.total_gaps(inst, ev.gaps(inst, tt))
    # lower-bound property: g can never be pushed below the true gap count
    sub.bound(sub.G, r.value - 1)
    assert sub.check(10).infeasible
