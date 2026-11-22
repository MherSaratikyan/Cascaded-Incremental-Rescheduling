"""Fetch pinned official ITC 2019 test instances and generate ITC-style data.

    python -m experiments.itc_data --synthetic s m l

The official XMLs come from the competition organization's public solver
repository. Their original bytes are verified before use. Synthetic cases
are recorded separately in the dataset manifest, together with the adapter's
fidelity report. Additional XMLs in experiments/itc are also discovered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from cascade.itc import fidelity_text, load_itc
from experiments.generator import ITC_DIR
from experiments.itc_generator import make

SOURCE_REPO = "itc2019/alexandre-lemos"
SOURCE_COMMIT = "c33d15797686a27c192eabb90948baa54d3ddef5"
SOURCES = {
    "lums-sum17": "c055ded38574dc6765334326fdcd9c1c832f9b18955680668edd89e8521afd77",
    "wbg-fal10": "dcfca95a4fdc34cdb85d2b84dd072e2bdfed05747abb5dfd0afc61dfe7514ed4",
    "pu-cs-fal07": "e9f1b7941e6b06919db69d80e1c326a02224aa5305d6bb9ebcfbdfa987e2b0db",
}


def source_url(name: str) -> str:
    return (f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}"
            f"/data/input/ITC-2019/{name}.xml")


def fetch_official(directory: Path = ITC_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCES.items():
        path = directory / f"{name}.xml"
        if path.exists():
            data = path.read_bytes()
        else:
            request = Request(source_url(name), headers={"User-Agent": "rescheduling-benchmarks"})
            with urlopen(request, timeout=90) as response:
                data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"{path.name}: SHA-256 mismatch; expected {expected}, got {actual}")
        if not path.exists():
            path.write_bytes(data)
        print(f"Official ITC 2019 test instance: {path.name} (SHA-256 verified)", flush=True)


def write_manifest(path: Path, generated: set[str], seed: int) -> None:
    datasets = []
    for xml in sorted(ITC_DIR.glob("*.xml")):
        inst = load_itc(xml)
        entry = dict(size=f"itc-{xml.stem}", file=xml.name,
                     sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                     lessons=len(inst.lessons), adaptation="A", fidelity=inst.meta["itc"])
        if xml.stem in SOURCES:
            entry.update(kind="official_itc2019_test_instance", source=source_url(xml.stem),
                         source_commit=SOURCE_COMMIT,
                         license=f"https://github.com/{SOURCE_REPO}/blob/{SOURCE_COMMIT}/LICENSE")
        elif xml.stem in generated:
            entry.update(kind="synthetic_itc2019_style", generator="experiments.itc_generator",
                         generator_seed=seed)
        else:
            entry.update(kind="user_supplied_xml")
        datasets.append(entry)
        print(fidelity_text(inst), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"datasets": datasets}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Dataset manifest: {path}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", nargs="+", choices=("s", "m", "l"), default=["s", "m", "l"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=Path("benchmark-metadata/itc-manifest.json"))
    args = parser.parse_args(argv)
    fetch_official()
    for size in args.synthetic:
        make(size, seed=args.seed)
    write_manifest(args.manifest, {f"synth-{s}" for s in args.synthetic}, args.seed)


if __name__ == "__main__":
    main()
