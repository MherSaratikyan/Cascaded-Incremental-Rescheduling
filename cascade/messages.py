"""Plain-language explanations for staff, in English ("en") and Armenian ("hy")."""
from __future__ import annotations

from typing import Optional

from .context import StepContext
from .model import Assignment, Instance
from .result import Outcome, Result

LANGS = ("en", "hy")


def _ord_hy(n: int) -> str:
    return f"{n}-ին" if n == 1 else f"{n}-րդ"


def slot_text(inst: Instance, a: Assignment, lang: str) -> str:
    d, p = inst.day_hour(a[0])
    if lang == "hy":
        return f"{_ord_hy(d)} օր, {_ord_hy(p)} ժամ, {a[1]} լսարան"
    return f"day {d}, hour {p}, room {a[1]}"


def soft_text(inst: Instance, sid: str, lang: str) -> str:
    s = inst.soft[sid]
    if s.kind == "pref":
        return (f"{s.lecturer} դասախոսի ժամային նախընտրությունը" if lang == "hy"
                else f"time preference of lecturer {s.lecturer}")
    if s.kind == "load":
        return (f"{s.lecturer} դասախոսի օրական ծանրաբեռնվածությունը {_ord_hy(s.day)} օրը"
                if lang == "hy" else f"daily workload of {s.lecturer} on day {s.day}")
    return (f"{s.course} առարկայի լսարանային կայունությունը" if lang == "hy"
            else f"room stability of course {s.course}")


def item_text(inst: Instance, item: tuple, lang: str) -> str:
    kind = item[0]
    if kind == "lock":
        return (f"{item[1]} դասի ամրագրումը պետք է հանվի" if lang == "hy"
                else f"the lock on lesson {item[1]} must be lifted")
    if kind == "prot":
        return (f"{item[1]} պաշտպանվող պայմանը ({soft_text(inst, item[1], lang)}) պետք է թուլացվի"
                if lang == "hy" else
                f"protected constraint {item[1]} ({soft_text(inst, item[1], lang)}) must be relaxed")
    if kind == "gap":
        return (f"{item[1]}-ի համար նոր բացերի արգելքը պետք է թուլացվի" if lang == "hy"
                else f"the no-new-gaps protection of {item[1]} must be relaxed")
    if kind == "budget":
        return "փոփոխությունների սահմանը" if lang == "hy" else "the change budget"
    if kind == "fix":
        return (f"{item[1]} դասի ֆիքսումը" if lang == "hy" else f"the fixing of lesson {item[1]}")
    return repr(item)


def requirement_text(inst: Instance, item: tuple, lang: str) -> str:
    """Item as a requirement (used in a MUS), not as a correction."""
    kind = item[0]
    if kind == "lock":
        return f"{item[1]} դասի ամրագրում" if lang == "hy" else f"lock on lesson {item[1]}"
    if kind == "prot":
        return (f"{item[1]} ({soft_text(inst, item[1], lang)})")
    if kind == "gap":
        return (f"{item[1]}-ի նոր բացերի արգելք" if lang == "hy" else f"no new gaps for {item[1]}")
    return item_text(inst, item, lang)


def change_lines(ctx: StepContext, tt, lang: str) -> list[str]:
    out = []
    for e in sorted(tt):
        if e in ctx.xk and tt[e] != ctx.xk[e]:
            a, b = slot_text(ctx.inst, ctx.xk[e], lang), slot_text(ctx.inst, tt[e], lang)
            out.append(f"{e} դաս՝ {a} → {b}" if lang == "hy" else f"Lesson {e}: {a} -> {b}")
    return out


_HEAD = {
    Outcome.SOLVED: ("Solution found: {n} lesson(s) changed.",
                     "Լուծումը գտնվել է. փոփոխվել է {n} դաս։"),
    Outcome.PHYSICAL_INFEASIBLE: (
        "Physical infeasibility: no timetable satisfies the hard constraints.",
        "Ֆիզիկական անիրագործելիություն. խիստ սահմանափակումները բավարարող դասացուցակ գոյություն չունի։"),
    Outcome.REGULATORY_INFEASIBLE: (
        "Regulatory infeasibility: the hard constraints can be met, but not under the current regulations.",
        "Կանոնակարգային անիրագործելիություն. խիստ սահմանափակումները բավարարելի են, սակայն ոչ գործող կանոնակարգի պայմաններում։"),
    Outcome.LOCK_CONFLICT: (
        "Lock conflict: fully locked lesson(s) {locks} can no longer keep their assignment.",
        "Ամրագրման հակասություն. լրիվ ամրագրված {locks} դաս(եր)ը այլևս չի կարող պահպանել իր նշանակումը։"),
    Outcome.ALGORITHMIC_FAILURE: ("No solution was found in time.",
                                  "Լուծումը ժամանակին չի գտնվել։"),
    Outcome.REJECTED_LOCAL: (
        "Rejected by the local check: the lesson cannot be placed without moving other lessons.",
        "Մերժվել է տեղային ստուգմամբ. դասը հնարավոր չէ տեղադրել առանց այլ դասեր տեղափոխելու։"),
    Outcome.REJECTED_BY_APPROVER: (
        "The change chain was rejected by the authorised person.",
        "Փոփոխությունների շղթան մերժվել է իրավասու անձի կողմից։"),
}


def explain(result: Result, ctx: Optional[StepContext], lang: str = "en") -> str:
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}")
    li = 1 if lang == "hy" else 0
    inst = ctx.inst if ctx else None
    n = len(result.metrics.get("changed", [])) if result.metrics else 0
    lines = [_HEAD[result.outcome][li].format(n=n, locks=", ".join(result.lock_conflicts))]
    if result.outcome == Outcome.SOLVED and ctx is not None and result.timetable:
        lines += ["  " + s for s in change_lines(ctx, result.timetable, lang)]
    if result.requires_approval:
        who = ", ".join(result.affected_lecturers)
        lines.append(f"Պահանջվում է իրավասու անձի համաձայնություն. շղթան տեղափոխում է {who} դասախոս(ներ)ի դասերը։"
                     if lang == "hy" else
                     f"Requires approval by the authorised person: the chain moves lessons of {who}.")
    if result.mus and inst is not None:
        req = "; ".join(requirement_text(inst, i, lang) for i in result.mus)
        lines.append(("Անհամատեղելի պահանջներ (նվազագույն անհամատեղելի ենթաբազմություն)՝ "
                      if lang == "hy" else "Conflicting requirements (minimal unsatisfiable subset): ") + req)
    for k, c in enumerate(result.mcs, 1):
        fix = "; ".join(item_text(inst, i, lang) for i in c.items)
        if lang == "hy":
            tag = "ստուգված է կրկնակի լուծմամբ" if c.verified else "ստուգումը ձախողվել է. մասնակի թուլացում"
            lines.append(f"  Ուղղման տարբերակ {k}՝ {fix} ({tag})")
        else:
            tag = "verified by re-solving" if c.verified else "verification failed: partial relaxation"
            lines.append(f"  Correction option {k}: {fix} ({tag})")
    if result.review and inst is not None:
        names = ", ".join(result.review)
        lines.append(f"Ուղարկվել է վերանայման (արդեն խախտված էր, դարձել է չվատթարացվող)՝ {names}"
                     if lang == "hy" else
                     f"Sent for review (already violated, demoted to non-worsening): {names}")
    return "\n".join(lines)
