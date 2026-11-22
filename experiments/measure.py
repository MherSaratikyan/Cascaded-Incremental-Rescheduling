"""Measured execution of one rescheduling request.

Every request ("test case") runs in a **fresh process** (multiprocessing
``spawn``), because Windows' peak-memory counters cannot be reset inside a
process.  The child records

* ``elapsed_s``   wall-clock (real) time of the request: step 1 to commit
* ``cpu_s``       process CPU time of the same span (all CP-SAT threads)
* ``startup_s``   process start + imports (reported separately, not compared)
* peak virtual size / committed private bytes / working set of the process,
  plus the committed bytes just before the request (``base_commit_mb``)

and the parent enforces a memory guard (committed private bytes) and a hard
wall-clock guard; a killed child yields the outcomes ``memory_limit`` or
``killed_timeout``.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from typing import Optional

from cascade import TimetableStore, handle_request
from cascade import evaluate as ev
from cascade.baselines import run_method
from cascade.disruptions import Disruption
from cascade.instrument import MB, Stopwatch, memory_snapshot, private_bytes
from cascade.model import Instance, Timetable
from cascade.params import Params

TIME_FIELDS = ["elapsed_s", "cpu_s", "startup_s", "process_wall_s", "t_search", "t_optimise",
               "t_extra", "t_explain", "solver_seconds", "solver_calls"]
MEMORY_FIELDS = ["base_commit_mb", "peak_commit_mb", "request_commit_mb", "peak_virtual_mb",
                 "peak_wset_mb"]
QUALITY_FIELDS = ["F_ratio", "F_size", "P_k", "P_0", "G", "dG_plus", "dG_lecturers", "dG_groups",
                  "Q_O", "Q_O_k", "dV_pref", "dV_load", "dV_room", "changed",
                  "lecturers_affected", "groups_affected", "hours_moved", "day_changes",
                  "room_only_moves", "protected_violations", "lock_violations",
                  "gap_protection_violations", "hard_violations"]
CERT_FIELDS = ["pstar", "pstar_proven", "lexmin_proven", "lexmin_stages", "opt_gap_max",
               "extra_expansion", "mus_size", "mus_minimal", "mcs_count", "mcs_verified",
               "requires_approval"]
EFFORT_FIELDS = ["iterations", "max_model_vars", "max_model_constraints", "final_model_vars",
                 "final_model_constraints"]
REQUEST_FIELDS = ["kind", "voluntary", "outcome", "attempts", "error"]
ALL_METRICS = (REQUEST_FIELDS + TIME_FIELDS + MEMORY_FIELDS + QUALITY_FIELDS + CERT_FIELDS
               + EFFORT_FIELDS)


@dataclass
class State:
    """What a request needs from the published timetable (picklable)."""
    instance: Instance
    timetable: Timetable
    x0: Timetable
    locks: dict
    protected: dict
    gap_protected: Optional[set] = None


def _round(v):
    return round(v, 6) if isinstance(v, float) else v


def make_row(h, disruption: Disruption) -> dict:
    r, m = h.result, h.result.metrics
    row = {"kind": disruption.kind, "voluntary": disruption.kind == "move_request",
           "outcome": r.outcome.value, "attempts": h.attempts, "pstar": r.p_star,
           "extra_expansion": r.extra_expansion, "requires_approval": r.requires_approval}
    for f in QUALITY_FIELDS + CERT_FIELDS + EFFORT_FIELDS + TIME_FIELDS:
        if f in m and f not in row:
            row[f] = _round(m[f])
    row["changed"] = len(m.get("changed", []))
    if r.timetable is not None:
        g = ev.gaps(h.ctx.inst, r.timetable)
        split = {"lecturer": 0.0, "group": 0.0}
        for (u, d), v in g.items():
            p = h.ctx.inst.participants[u]
            split[p.kind] += p.weight * max(0, v - h.ctx.gaps_k.get((u, d), 0))
        row["dG_lecturers"], row["dG_groups"] = split["lecturer"], split["group"]
    return row


def run_request(state: State, disruption: Disruption, params: Params, method: str
                ) -> tuple[dict, Optional[State]]:
    """Run one request in *this* process.  Returns (row, new state or None)."""
    store = TimetableStore(state.instance, state.timetable, state.x0, state.locks,
                           state.protected, state.gap_protected)
    base = memory_snapshot()
    with Stopwatch() as sw:
        h = handle_request(store, disruption, params, solver=lambda c: run_method(method, c))
    peak = memory_snapshot()
    row = make_row(h, disruption)
    row.update(elapsed_s=round(sw.elapsed, 4), cpu_s=round(sw.cpu, 4),
               base_commit_mb=round(base.commit / MB, 2),
               peak_commit_mb=round(peak.peak_commit / MB, 2),
               request_commit_mb=round(max(0, peak.peak_commit - base.commit) / MB, 2),
               peak_virtual_mb=round(peak.peak_virtual / MB, 2),
               peak_wset_mb=round(peak.peak_wset / MB, 2))
    new_state = None
    if store.version > 0:
        new_state = State(store.instance, store.timetable, state.x0, state.locks,
                          state.protected, state.gap_protected)
    return row, new_state


def advance_state(state: State, disruption: Disruption, new_state: Optional[State]) -> State:
    """State for the next step: the committed one, or -- if nothing was
    committed (e.g. the child was killed) -- the old timetable on the instance
    after an *external* disruption, which happened regardless."""
    if new_state is not None:
        return new_state
    if disruption.external:
        from cascade.disruptions import apply
        return State(apply(state.instance, disruption)[0], state.timetable, state.x0,
                     state.locks, state.protected, state.gap_protected)
    return state


def _child(conn, state, disruption, params, method, t_spawn):
    startup = time.time() - t_spawn
    try:
        row, new_state = run_request(state, disruption, params, method)
        row["startup_s"] = round(startup, 3)
        conn.send(("ok", row, new_state))
    except BaseException:                       # report, never hang the parent
        conn.send(("error", traceback.format_exc(limit=8), None))
    finally:
        conn.close()


def run_isolated(state: State, disruption: Disruption, params: Params, method: str,
                 mem_limit_mb: float = 3072, kill_after: Optional[float] = None
                 ) -> tuple[dict, Optional[State]]:
    """Run one request in a fresh process with memory and wall-clock guards."""
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    kill_after = kill_after or (params.time_limit + params.explain_time_limit + 180)
    t0 = time.time()
    proc = ctx.Process(target=_child, args=(send, state, disruption, params, method, t0), daemon=True)
    proc.start()
    send.close()
    msg, peak_private, verdict = None, 0, None
    while True:
        if recv.poll(0.1):
            try:
                msg = recv.recv()
            except EOFError:
                msg = None
            break
        if not proc.is_alive():
            break
        try:
            peak_private = max(peak_private, private_bytes(proc.pid))
        except Exception:
            pass
        if peak_private > mem_limit_mb * MB:
            verdict = "memory_limit"
            break
        if time.time() - t0 > kill_after:
            verdict = "killed_timeout"
            break
    if verdict is not None:
        proc.kill()
    proc.join(10)
    wall = time.time() - t0
    base = {"kind": disruption.kind, "voluntary": disruption.kind == "move_request",
            "process_wall_s": round(wall, 3)}
    if verdict is not None:
        base.update(outcome=verdict, elapsed_s=round(wall, 3),
                    peak_commit_mb=round(peak_private / MB, 2))
        return base, None
    if msg is None:
        base.update(outcome="crashed", error=f"exit code {proc.exitcode}",
                    peak_commit_mb=round(peak_private / MB, 2))
        return base, None
    status, payload, new_state = msg
    if status != "ok":
        base.update(outcome="error", error=payload.strip().splitlines()[-1][:300])
        return base, None
    payload["process_wall_s"] = round(wall, 3)
    return payload, new_state
