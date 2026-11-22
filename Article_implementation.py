"""Demo (Visual Studio start-up file): the worked examples of Section 6 of
"Incremental cascaded rescheduling of dynamic university timetables with
protected soft constraints and lesson locks", in English and Armenian.

    python Article_implementation.py            # both languages
    python Article_implementation.py --lang hy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cascade import Disruption, TimetableStore, handle_request, load_scenario  # noqa: E402
from cascade.baselines import run_method  # noqa: E402


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def show(h, langs) -> None:
    r = h.result
    print(f"outcome: {r.outcome.value}   time: {r.seconds:.2f}s   "
          f"P* = {r.p_star}   P-bar = {r.p_bar}   extra expansion: {r.extra_expansion}")
    if r.trace:
        print("active-set trace (j, F_j, B_j, solver status, core -> action):")
        for it in r.trace:
            core = ", ".join("/".join(map(str, c)) for c in (it.core or []))
            print(f"  j={it.j}  F={{{', '.join(it.lessons)}}}  B={it.budget}  {it.status:<10} "
                  f"[{core}] -> {it.action}")
    for lang in langs:
        print(f"--- [{lang}]")
        print(h.explanation(lang))


def store_for(sc, **over) -> TimetableStore:
    return TimetableStore(sc.instance, sc.timetable, sc.initial, over.get("locks", sc.locks),
                          over.get("protected", sc.protected), sc.gap_protected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "hy", "both"], default="both")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    langs = ["en", "hy"] if a.lang == "both" else [a.lang]
    ex = ROOT / "examples"

    chain = load_scenario(ex / "chain_example.json")
    banner("1. Chain repair (Section 6): room of A closes, A's only option is held by B")
    show(handle_request(store_for(chain), chain.disruptions[0], chain.params), langs)

    banner("1b. Same request with baselines: B1 local check [7] and B2 hard-only chain")
    for m in ("B1", "B2"):
        h = handle_request(store_for(chain), chain.disruptions[0], chain.params,
                           solver=lambda c, m=m: run_method(m, c))
        print(f"{m}: {h.result.outcome.value}; protected violations: "
              f"{h.result.metrics.get('protected_violations', '-')}")
        if h.result.timetable:
            print(h.explanation("en"))

    banner("2. Lesson C fully locked: regulatory infeasibility with MUS and verified MCS")
    show(handle_request(store_for(chain, locks={"C": "A"}), chain.disruptions[0], chain.params), langs)

    banner("3. Voluntary request of lecturer LA (A -> hour 2, room R1): chain needs approval")
    req = Disruption("move_request", lesson="A", day=1, hour=2, room="R1", voluntary=True)

    def approver(result, ctx):
        print(f"  [approver] chain moves lessons of {', '.join(result.affected_lecturers)} -> approve")
        return True

    show(handle_request(store_for(chain), req, chain.params, approver=approver), langs)

    claim3 = load_scenario(ex / "claim3_counterexample.json")
    banner("4. Counterexample to Claim 3, delta_k = w^T: quality trigger adds X")
    show(handle_request(store_for(claim3), claim3.disruptions[0], claim3.params), langs)
    banner("4b. Same, delta_k = 0: the frozen threshold keeps the single move A -> 6")
    show(handle_request(store_for(claim3), claim3.disruptions[0],
                        claim3.params.with_(delta_external=0)), langs)


if __name__ == "__main__":
    main()
