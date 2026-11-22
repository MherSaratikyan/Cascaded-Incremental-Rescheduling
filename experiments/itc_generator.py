"""ITC 2019-style instance generator.

The real ITC 2019 instances need a registered login, so this writes *new*
instances in the official ITC 2019 problem XML format
(https://www.itc2019.org/format) with the same structural features:

* 5-minute slots (slotsPerDay=288), 7-day weeks used Monday-Friday, 13 weeks
* multi-day time patterns: MWF 50 min, TTh 75 min, MW 75 min, single-day
  recitations, 110-minute labs on all / odd / even / first-half weeks,
  each option with a time penalty
* courses with 1-2 configurations, subparts Lecture / Recitation / Laboratory,
  parent-child classes, class limits, room options with penalties,
  classes without a room
* rooms in buildings with travel times, weekly and partial-week unavailability
* distributions: SameAttendees (instructors, parent-child), NotOverlap,
  Precedence, SameRoom, SameTime, SameStart, SameDays, DifferentDays, MaxDays,
  MinGap, WorkDay, MaxDayLoad, MaxBreaks, MaxBlock - required and soft
* students with course demands

They are synthetic ("similar, not the same"), deterministic per seed, and are
read through the same loader (cascade.itc) as real ITC files would be.

    python -m experiments.itc_generator            # synth-s, synth-m, synth-l
    python -m experiments.itc_generator --sizes s --seed 3
"""
from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.generator import ITC_DIR  # noqa: E402

WEEKS = 13
ALL_W, ODD_W, EVEN_W, HALF_W = "1" * 13, "1010101010101", "0101010101010", "1111111000000"
MWF, TTH, MW = "1010100", "0101000", "1010000"
DAY = ["1000000", "0100000", "0010000", "0001000", "0000100"]

# (days, start slot, length in slots); 1 slot = 5 minutes, 90 = 7:30
LECTURE = ([(MWF, s, 10) for s in range(90, 199, 12)]                 # MWF 7:30 ... 16:30, 50 min
           + [(TTH, s, 15) for s in (96, 114, 132, 150, 168, 186)]    # TTh 8:00 ... 15:30, 75 min
           + [(MW, s, 15) for s in (96, 132, 168)])                   # MW 75 min
RECITATION = [(d, s, 10) for d in DAY for s in range(96, 205, 12)]    # one day, 50 min
LAB = [(d, s, 22) for d in DAY for s in (96, 120, 156, 180)]          # one day, 110 min

SIZES = {   # courses, rooms, buildings, students
    "s": dict(courses=40, rooms=16, buildings=3, students=300),
    "m": dict(courses=100, rooms=36, buildings=5, students=800),
    "l": dict(courses=250, rooms=85, buildings=8, students=2000),
}


def _pen(rng: random.Random, p_zero: float = 0.7, hi: int = 4) -> int:
    return 0 if rng.random() < p_zero else rng.randint(1, hi)


def generate_itc_xml(name: str, courses: int, rooms: int, buildings: int, students: int,
                     seed: int) -> ET.ElementTree:
    rng = random.Random(f"itcgen|{name}|{seed}")
    root = ET.Element("problem", name=name, nrDays="7", nrWeeks=str(WEEKS), slotsPerDay="288")
    ET.SubElement(root, "optimization", time="2", room="1", distribution="10", student="5")

    # ------------------------------------------------------------- rooms
    caps = [20, 25, 30, 40, 40, 50, 60, 80, 100, 150, 220]
    room_el = ET.SubElement(root, "rooms")
    room_cap: dict[int, int] = {}
    building = {}
    for r in range(1, rooms + 1):
        cap = caps[(r - 1) % len(caps)] if r <= len(caps) else rng.choice(caps)
        room_cap[r] = cap
        building[r] = rng.randrange(buildings)
        el = ET.SubElement(room_el, "room", id=str(r), capacity=str(cap))
        for o in range(1, r):                       # symmetric travel, listed once
            if building[o] != building[r]:
                ET.SubElement(el, "travel", room=str(o), value=str(2 + abs(building[o] - building[r])))
        if rng.random() < 0.15:                     # weekly unavailability
            d = rng.choice(DAY)
            ET.SubElement(el, "unavailable", days=d, start=str(rng.choice((96, 144, 180))),
                          length=str(rng.choice((24, 36, 48))), weeks=ALL_W)
        if rng.random() < 0.10:                     # some weeks only (dropped by the mapping)
            ET.SubElement(el, "unavailable", days=rng.choice(DAY), start="144", length="48",
                          weeks=rng.choice((ODD_W, HALF_W)))

    # ----------------------------------------------------------- courses
    courses_el = ET.SubElement(root, "courses")
    cid = 0
    classes: list[dict] = []
    sub_classes: list[list[int]] = []               # classes per subpart
    subpart_no = config_no = 0

    def add_class(sub_el, kind: str, limit: int, parent=None) -> int:
        nonlocal cid
        cid += 1
        attrs = dict(id=str(cid), limit=str(limit))
        if parent is not None:
            attrs["parent"] = str(parent)
        no_room = kind != "Lec" and rng.random() < 0.05
        if no_room:
            attrs["room"] = "false"
        el = ET.SubElement(sub_el, "class", **attrs)
        if not no_room:
            fit = [r for r, c in room_cap.items() if c >= limit] or [max(room_cap, key=room_cap.get)]
            for r in sorted(rng.sample(fit, min(len(fit), rng.randint(2, 6)))):
                ET.SubElement(el, "room", id=str(r), penalty=str(_pen(rng, 0.6, 3)))
        pool = {"Lec": LECTURE, "Rec": RECITATION, "Lab": LAB}[kind]
        k = {"Lec": rng.randint(4, 8), "Rec": rng.randint(3, 8), "Lab": rng.randint(2, 6)}[kind]
        for days, start, length in rng.sample(pool, k):
            weeks = ALL_W
            if kind == "Lab" and rng.random() < 0.3:
                weeks = rng.choice((ODD_W, EVEN_W, HALF_W))
            ET.SubElement(el, "time", days=days, start=str(start), length=str(length),
                          weeks=weeks, penalty=str(_pen(rng)))
        classes.append(dict(id=cid, kind=kind, parent=parent, limit=limit))
        return cid

    course_classes: dict[int, list[int]] = {}
    for c in range(1, courses + 1):
        ce = ET.SubElement(courses_el, "course", id=str(c))
        enrol = rng.choice((25, 40, 60, 90, 120, 180))
        course_classes[c] = []
        for _ in range(1 if rng.random() < 0.85 else 2):
            config_no += 1
            cfg = ET.SubElement(ce, "config", id=str(config_no))
            n_lec = 1 if enrol <= 60 else rng.randint(1, 3)
            subpart_no += 1
            lec_sub = ET.SubElement(cfg, "subpart", id=str(subpart_no))
            lecs = [add_class(lec_sub, "Lec", max(20, enrol // n_lec)) for _ in range(n_lec)]
            sub_classes.append(lecs)
            course_classes[c] += lecs
            parents = lecs
            if rng.random() < 0.7:
                subpart_no += 1
                rec_sub = ET.SubElement(cfg, "subpart", id=str(subpart_no))
                recs = []
                for p in lecs:
                    for _ in range(rng.randint(1, 3)):
                        recs.append(add_class(rec_sub, "Rec", 25, parent=p))
                sub_classes.append(recs)
                course_classes[c] += recs
                parents = recs
            if rng.random() < 0.35:
                subpart_no += 1
                lab_sub = ET.SubElement(cfg, "subpart", id=str(subpart_no))
                labs = [add_class(lab_sub, "Lab", 20, parent=p) for p in parents]
                sub_classes.append(labs)
                course_classes[c] += labs

    # ------------------------------------------------------ distributions
    dist_el = ET.SubElement(root, "distributions")

    def dist(typ: str, ids, required: bool, penalty: int = 1):
        ids = list(dict.fromkeys(ids))
        if len(ids) < 2:
            return
        attrs = {"type": typ}
        if required:
            attrs["required"] = "true"
        else:
            attrs["penalty"] = str(penalty)
        el = ET.SubElement(dist_el, "distribution", **attrs)
        for i in ids:
            ET.SubElement(el, "class", id=str(i))

    by_id = {k["id"]: k for k in classes}
    # instructors: 2-4 classes each -> SameAttendees (required) + soft workload rules
    pool = [k["id"] for k in classes]
    rng.shuffle(pool)
    while pool:
        n = rng.randint(2, 4)
        take, pool = pool[:n], pool[n:]
        dist("SameAttendees", take, True)
        r = rng.random()
        if r < 0.25:
            dist("MaxDays(3)", take, False, 2)
        elif r < 0.40:
            dist("MinGap(6)", take, False, 1)
        elif r < 0.55:
            dist("WorkDay(108)", take, False, 1)
        elif r < 0.65:
            dist("MaxDayLoad(84)", take, False, 2)
        elif r < 0.75:
            dist("MaxBreaks(1,12)", take, False, 1)
        elif r < 0.85:
            dist("MaxBlock(48,6)", take, False, 1)
    # course structure: a child cannot overlap its parent; lecture before its labs
    for k in classes:
        if k["parent"] is not None:
            dist("SameAttendees", [k["parent"], k["id"]], True)
            if k["kind"] == "Lab":
                top = k["parent"]
                while by_id[top]["parent"] is not None:
                    top = by_id[top]["parent"]
                dist("Precedence", [top, k["id"]], rng.random() < 0.3, 2)
    # sections of one subpart: spread in time; lecture sections share a room / days
    for ids in sub_classes:
        if len(ids) < 2:
            continue
        dist("NotOverlap", ids, len(ids) <= 3 and rng.random() < 0.3, 1)
        kind = by_id[ids[0]]["kind"]
        if kind == "Lec":
            r = rng.random()
            if r < 0.3:
                dist("SameRoom", ids, False, 1)
            elif r < 0.45:
                dist("SameDays", ids, False, 1)
            elif r < 0.55:
                dist("SameTime", ids[:2], True)
        elif kind == "Rec" and rng.random() < 0.2:
            dist("DifferentDays", ids[:3], False, 1)
        elif kind == "Lab" and rng.random() < 0.1:
            dist("SameStart", ids[:2], True)

    # ----------------------------------------------------------- students
    stud_el = ET.SubElement(root, "students")
    all_courses = list(course_classes)
    for s in range(1, students + 1):
        se = ET.SubElement(stud_el, "student", id=str(s))
        for c in rng.sample(all_courses, min(len(all_courses), rng.randint(3, 5))):
            ET.SubElement(se, "course", id=str(c))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t")
    return tree


def make(size: str, seed: int = 1, out_dir: Path = ITC_DIR, check: bool = True,
         tries: int = 10) -> Path:
    """Write experiments/itc/synth-<size>.xml; with ``check`` the mapped instance
    must admit a feasible x^(0) (else the next seed is tried) - the x^(0) is
    cached in experiments/instances like for any ITC instance."""
    from experiments.generator import load_itc_instance
    cfg = SIZES[size]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"synth-{size}.xml"
    cache = ROOT / "experiments" / "instances"
    for k in range(tries):
        tree = generate_itc_xml(f"synth-{size}", seed=seed * 100 + k, **cfg)
        with path.open("wb") as fh:
            fh.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            fh.write(b"<!-- Synthetic ITC 2019-style instance generated by experiments/itc_generator.py"
                     b" (not an official ITC 2019 instance) -->\n")
            tree.write(fh, encoding="utf-8", xml_declaration=False)
        if not check:
            return path
        (cache / f"itc-synth-{size}.json").unlink(missing_ok=True)
        try:
            load_itc_instance(f"itc-synth-{size}", cache)
            return path
        except RuntimeError as exc:
            print(f"  synth-{size} seed {seed * 100 + k}: {exc}; retrying", flush=True)
    raise RuntimeError(f"no feasible synth-{size} instance in {tries} tries")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=["s", "m", "l"], choices=sorted(SIZES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args(argv)
    from cascade.itc import fidelity_text, load_itc
    for size in a.sizes:
        path = make(size, a.seed, check=not a.no_check)
        print(fidelity_text(load_itc(path)), flush=True)


if __name__ == "__main__":
    main()
