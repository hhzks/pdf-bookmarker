"""Tests for training/fetch_ccpdf.py (offline: the remote ZIP is served from memory)."""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import fetch_ccpdf


def test_zip_url_groups_by_thousand():
    assert fetch_ccpdf.zip_url(0).endswith("/zipfiles/0000-0999/0000.zip")
    assert fetch_ccpdf.zip_url(7900).endswith("/zipfiles/7000-7999/7900.zip")


def test_parse_zip_spec_ranges_and_lists():
    assert fetch_ccpdf.parse_zip_spec("0-3,17,2") == [0, 1, 2, 3, 17]
    with pytest.raises(ValueError):
        fetch_ccpdf.parse_zip_spec(str(fetch_ccpdf.LAST_ZIP + 1))


def _row(**overrides):
    row = {"file_name": "0000001.pdf", "parse_status": "OK", "encrypted": "f",
           "num_pages": "20", "tika_eval_num_alpha_tokens": "5000",
           "tika_eval_lang": "eng", "pdf_producer": "Microsoft Word",
           "xmp_creator_tool": "", "url_id": "42"}
    row.update(overrides)
    return row


@pytest.mark.parametrize("overrides, reason", [
    ({}, None),
    ({"parse_status": "EXCEPTION"}, "tika-parse-failed"),
    ({"encrypted": "t"}, "encrypted"),
    ({"num_pages": "2"}, "too-short"),
    ({"tika_eval_num_alpha_tokens": "0"}, "no-text"),
    ({"tika_eval_lang": "deu"}, "language"),
    ({"pdf_producer": "pdfTeX-1.40.21"}, "tex"),
])
def test_metadata_reject(overrides, reason):
    got = fetch_ccpdf.metadata_reject(_row(**overrides), min_pages=4,
                                      include_tex=False, lang="eng")
    assert got == reason


def test_metadata_reject_keeps_tex_when_asked():
    row = _row(pdf_producer="pdfTeX-1.40.21")
    assert fetch_ccpdf.metadata_reject(row, min_pages=4, include_tex=True, lang=None) is None


def test_load_metadata_keeps_requested_zips_only(tmp_path):
    path = tmp_path / "tika.csv"
    path.write_text("﻿file_name,num_pages\n0000001.pdf,3\n0001001.pdf,9\n",
                    encoding="utf-8")
    rows = fetch_ccpdf.load_metadata(path, {1})
    assert list(rows) == ["0001001.pdf"]


def test_gate_matches_harvest(outlined_toc_pdf, plain_pdf, encrypted_pdf):
    reason, info = fetch_ccpdf.gate(outlined_toc_pdf.read_bytes(), min_pages=4,
                                    include_tex=False)
    assert reason is None and info["outline_entries"] == 4
    assert fetch_ccpdf.gate(plain_pdf.read_bytes(), min_pages=1,
                            include_tex=False)[0] == "no-outline"
    assert fetch_ccpdf.gate(encrypted_pdf.read_bytes(), min_pages=1,
                            include_tex=False)[0] == "encrypted"
    assert fetch_ccpdf.gate(b"<html>", min_pages=1, include_tex=False)[0] == "not-pdf"
    assert fetch_ccpdf.gate(outlined_toc_pdf.read_bytes(), min_pages=50,
                            include_tex=False)[0] == "too-short"


class _Response:
    def __init__(self, data, headers=None):
        self._data = data
        self.headers = headers or {}

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def remote_zip(monkeypatch, outlined_toc_pdf, plain_pdf):
    """Serve an in-memory ZIP through urlopen, honouring HEAD and Range."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("0000000.pdf", outlined_toc_pdf.read_bytes())
        zf.writestr("0000001.pdf", plain_pdf.read_bytes())
    blob = buf.getvalue()
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.get_method() == "HEAD":
            return _Response(b"", {"Content-Length": str(len(blob))})
        start, end = map(int, re.fullmatch(
            r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        return _Response(blob[start:end + 1])

    monkeypatch.setattr(fetch_ccpdf.urllib.request, "urlopen", fake_urlopen)
    return requests


def test_fetch_keeps_bookmarked_and_records_every_try(remote_zip, tmp_path):
    kept, rejected = fetch_ccpdf.fetch([0], tmp_path, max_kept=10, min_pages=1,
                                       include_tex=False, metadata=None, lang=None)
    assert (kept, rejected) == (1, 1)
    assert [p.name for p in tmp_path.glob("*.pdf")] == ["cc-0000000.pdf"]
    lines = [json.loads(l) for l in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert {(l["member"], l["reason"]) for l in lines} == {
        ("0000000.pdf", None), ("0000001.pdf", "no-outline")}


def test_fetch_resumes_without_refetching(remote_zip, tmp_path):
    fetch_ccpdf.fetch([0], tmp_path, max_kept=10, min_pages=1,
                      include_tex=False, metadata=None, lang=None)
    remote_zip.clear()
    kept, rejected = fetch_ccpdf.fetch([0], tmp_path, max_kept=10, min_pages=1,
                                       include_tex=False, metadata=None, lang=None)
    assert (kept, rejected) == (0, 0)
    # Only the directory is read again; no member bytes are requested.
    assert len(remote_zip) <= 4


def test_fetch_metadata_skips_before_download(remote_zip, tmp_path):
    metadata = {"0000000.pdf": _row(file_name="0000000.pdf", num_pages="1")}
    kept, rejected = fetch_ccpdf.fetch([0], tmp_path, max_kept=10, min_pages=4,
                                       include_tex=False, metadata=metadata, lang=None)
    # One member is too short per Tika, the other has no row: neither is fetched
    # nor recorded, so a later run with other filters reconsiders both.
    assert (kept, rejected) == (0, 0)
    assert not (tmp_path / "manifest.jsonl").read_text()


def test_range_file_retries_a_dropped_transfer(monkeypatch):
    calls = []

    def flaky(request, timeout):
        if request.get_method() == "HEAD":
            return _Response(b"", {"Content-Length": "10"})
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("reset")
        return _Response(b"0123456789"[2:6])

    monkeypatch.setattr(fetch_ccpdf.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fetch_ccpdf.time, "sleep", lambda s: None)
    remote = fetch_ccpdf.HttpRangeFile("https://x/0000.zip")
    remote.seek(2)
    assert remote.read(4) == b"2345"
    assert remote.tell() == 6 and len(calls) == 2
