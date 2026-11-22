"""Dataset integrity and provenance for the ITC benchmark inputs (offline)."""
import hashlib
import io
import json
from pathlib import Path

import pytest

from experiments import itc_data


def set_source(monkeypatch, payload):
    monkeypatch.setattr(itc_data, "SOURCES", {"test-input": hashlib.sha256(payload).hexdigest()})


def test_cached_official_xml_is_checked_without_network(tmp_path, monkeypatch):
    payload = b"original XML bytes\n"
    set_source(monkeypatch, payload)
    path = tmp_path / "test-input.xml"
    path.write_bytes(payload)
    monkeypatch.setattr(itc_data, "urlopen", lambda *a, **kw: pytest.fail("Unexpected network request"))
    itc_data.fetch_official(tmp_path)
    assert path.read_bytes() == payload


def test_changed_cached_xml_is_rejected_and_preserved(tmp_path, monkeypatch):
    set_source(monkeypatch, b"expected bytes")
    path = tmp_path / "test-input.xml"
    path.write_bytes(b"modified bytes")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        itc_data.fetch_official(tmp_path)
    assert path.read_bytes() == b"modified bytes"


def test_corrupt_download_does_not_create_a_dataset(tmp_path, monkeypatch):
    set_source(monkeypatch, b"expected bytes")
    monkeypatch.setattr(itc_data, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong download"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        itc_data.fetch_official(tmp_path)
    assert not (tmp_path / "test-input.xml").exists()


def test_download_preserves_original_bom_and_newlines(tmp_path, monkeypatch):
    payload = b"\xef\xbb\xbf<problem/>\r\n"
    set_source(monkeypatch, payload)
    monkeypatch.setattr(itc_data, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    itc_data.fetch_official(tmp_path)
    assert (tmp_path / "test-input.xml").read_bytes() == payload


def test_manifest_separates_generated_and_user_data_and_records_fidelity(tmp_path, monkeypatch):
    sample = Path(__file__).resolve().parents[1] / "examples" / "itc_sample.xml"
    (tmp_path / "synth-s.xml").write_bytes(sample.read_bytes())
    (tmp_path / "additional.xml").write_bytes(sample.read_bytes())
    monkeypatch.setattr(itc_data, "ITC_DIR", tmp_path)
    out = tmp_path / "metadata" / "datasets.json"
    itc_data.write_manifest(out, {"synth-s"}, 3)
    rows = {d["size"]: d for d in json.loads(out.read_text(encoding="utf-8"))["datasets"]}
    assert rows["itc-synth-s"]["kind"] == "synthetic_itc2019_style"
    assert rows["itc-synth-s"]["generator_seed"] == 3
    assert rows["itc-additional"]["kind"] == "user_supplied_xml"
    assert rows["itc-additional"]["adaptation"] == "A"
    assert rows["itc-additional"]["fidelity"]["lessons"] == 6
    assert rows["itc-additional"]["fidelity"]["students_ignored"] == 2
