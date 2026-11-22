import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade import TimetableStore, load_scenario  # noqa: E402


@pytest.fixture
def examples() -> Path:
    return ROOT / "examples"


def make_store(sc, **over):
    return TimetableStore(sc.instance, sc.timetable, sc.initial,
                          over.get("locks", sc.locks), over.get("protected", sc.protected),
                          over.get("gap_protected", sc.gap_protected))


@pytest.fixture
def chain(examples):
    return load_scenario(examples / "chain_example.json")


@pytest.fixture
def claim3(examples):
    return load_scenario(examples / "claim3_counterexample.json")
