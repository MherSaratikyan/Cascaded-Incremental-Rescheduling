"""Full procedure of Table 1 around :func:`reschedule`:

step 1  read x^(k) and the regulatory state from a versioned store
steps 2-5  :func:`cascade.rescheduler.reschedule`
step 6  optimistic concurrency: commit only if the version is unchanged,
        otherwise go back to step 1
step 7  approval of voluntary chains that move other lecturers' lessons,
        one-transaction commit, per-participant notification
"""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from .context import StepContext
from .disruptions import Disruption, apply
from .messages import change_lines, explain
from .model import Instance, Timetable
from .params import Params
from .rescheduler import Strategy, reschedule
from .result import Outcome, Result


class VersionConflict(Exception):
    pass


@dataclass
class Snapshot:
    instance: Instance
    timetable: Timetable
    x0: Timetable
    locks: dict[str, str]
    protected: dict[str, str]
    gap_protected: Optional[set[str]]
    version: int


class TimetableStore:
    """In-memory published timetable with a version counter."""

    def __init__(self, instance: Instance, timetable: Timetable, x0: Optional[Timetable] = None,
                 locks: Optional[dict[str, str]] = None, protected: Optional[dict[str, str]] = None,
                 gap_protected: Optional[set[str]] = None):
        self._mutex = threading.Lock()
        self.instance = instance
        self.timetable = dict(timetable)
        self.x0 = dict(x0 if x0 is not None else timetable)
        self.locks = dict(locks or {})
        self.protected = dict(protected or {})
        self.gap_protected = set(gap_protected) if gap_protected is not None else None
        self.version = 0
        self.log: list[str] = []

    def read(self) -> Snapshot:
        with self._mutex:
            return Snapshot(self.instance.copy(), dict(self.timetable), dict(self.x0),
                            dict(self.locks), dict(self.protected),
                            copy.copy(self.gap_protected), self.version)

    def commit(self, expected_version: int, instance: Optional[Instance] = None,
               timetable: Optional[Timetable] = None, note: str = "") -> int:
        with self._mutex:
            if self.version != expected_version:
                raise VersionConflict(f"expected v{expected_version}, store is at v{self.version}")
            self._write(instance, timetable, note)
            return self.version

    def force_write(self, instance: Optional[Instance] = None,
                    timetable: Optional[Timetable] = None, note: str = "") -> int:
        """An unconditional write by another client (used to simulate concurrency)."""
        with self._mutex:
            self._write(instance, timetable, note)
            return self.version

    def _write(self, instance, timetable, note):
        if instance is not None:
            self.instance = instance
        if timetable is not None:
            self.timetable = dict(timetable)
        self.version += 1
        self.log.append(f"v{self.version}: {note}")


Approver = Callable[[Result, StepContext], bool]


@dataclass
class Handled:
    result: Result
    ctx: StepContext
    attempts: int
    notifications: dict[str, list[str]] = field(default_factory=dict)

    def explanation(self, lang: str = "en") -> str:
        return explain(self.result, self.ctx, lang)


def notifications(ctx: StepContext, result: Result, lang: str = "en") -> dict[str, list[str]]:
    """Per-participant list of changes (step 7 notification)."""
    out: dict[str, list[str]] = {}
    if result.timetable is None:
        return out
    lines = dict(zip(sorted(e for e in result.timetable if result.timetable[e] != ctx.xk.get(e)),
                     change_lines(ctx, result.timetable, lang)))
    for e, line in lines.items():
        for u in ctx.inst.lessons[e].participants:
            out.setdefault(u, []).append(line)
    return out


def handle_request(store: TimetableStore, disruption: Disruption, params: Params,
                   strategy: Optional[Strategy] = None, approver: Optional[Approver] = None,
                   before_commit: Optional[Callable[[int], None]] = None,
                   solver: Optional[Callable[[StepContext], Result]] = None) -> Handled:
    solve = solver or (lambda c: reschedule(c, strategy))
    last: Optional[Handled] = None
    for attempt in range(params.max_retries + 1):
        snap = store.read()                                            # step 1
        inst_after, req = apply(snap.instance, disruption)
        ctx = StepContext.build(snap.instance, inst_after, snap.timetable, snap.x0, params,
                                snap.locks, snap.protected, snap.gap_protected, req)
        result = solve(ctx)                                            # steps 2-5
        if result.solved and result.requires_approval:                 # step 7 approval
            ok = approver(result, ctx) if approver is not None else True
            if not ok:
                result.outcome = Outcome.REJECTED_BY_APPROVER
                result.timetable = None
        if before_commit is not None:
            before_commit(attempt)
        try:                                                           # step 6 + commit
            if result.solved:
                store.commit(snap.version, inst_after, result.timetable,
                             f"{disruption.kind}: {result.outcome.value}")
            elif disruption.external:
                # the external event happened anyway; x^(k) stays published
                store.commit(snap.version, inst_after, None,
                             f"{disruption.kind}: {result.outcome.value} (timetable unchanged)")
            last = Handled(result, ctx, attempt + 1, notifications(ctx, result, params.lang))
            return last
        except VersionConflict as exc:
            result.notes.append(f"version conflict on attempt {attempt + 1}: {exc}")
            last = Handled(result, ctx, attempt + 1)
            continue
    assert last is not None
    last.result.outcome = Outcome.ALGORITHMIC_FAILURE
    last.result.timetable = None
    last.result.notes.append("optimistic concurrency: retries exhausted")
    return last
