"""Tests for training/harvest.py and training/fetch_arxiv.py (offline parts)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import fetch_arxiv
import harvest


# --- harvest_pdf ------------------------------------------------------------

def test_harvest_toc_path(outlined_toc_pdf):
    records, reason = harvest.harvest_pdf(outlined_toc_pdf)
    assert reason is None
    (record,) = records
    assert record["context_kind"] == "toc"
    assert record["context"].startswith("Table of contents text:")
    assert record["alignment"] == 1.0
    assert record["page_count"] == 5
    assert record["entries"] == [
        {"title": "1 Introduction", "level": 1, "printed_page": 3},
        {"title": "1.1 Background", "level": 2, "printed_page": 3},
        {"title": "2 Methods", "level": 1, "printed_page": 4},
        {"title": "3 Results", "level": 1, "printed_page": 5},
    ]


def test_harvest_headings_path(bookmarked_pdf):
    """Outline but no TOC page -> heading-candidate context, no printed pages."""
    records, reason = harvest.harvest_pdf(bookmarked_pdf, min_pages=1)
    assert reason is None
    (record,) = records
    assert record["context_kind"] == "headings"
    assert record["context"].startswith("Candidate heading lines")
    assert record["alignment"] is None
    assert record["entries"] == [
        {"title": "Existing", "level": 1, "printed_page": None}
    ]


def test_harvest_skips_unlabeled(plain_pdf):
    records, reason = harvest.harvest_pdf(plain_pdf)
    assert records is None
    assert reason == "no-embedded-outline"


def test_harvest_skips_encrypted(encrypted_pdf):
    records, reason = harvest.harvest_pdf(encrypted_pdf)
    assert records is None
    assert reason == "encrypted"


def test_harvest_skips_short(outlined_toc_pdf):
    records, reason = harvest.harvest_pdf(outlined_toc_pdf, min_pages=10)
    assert records is None
    assert reason == "too-short"


def test_harvest_augment_headings(outlined_toc_pdf):
    records, reason = harvest.harvest_pdf(outlined_toc_pdf, augment_headings=True)
    assert reason is None
    real, synth = records
    assert real["context_kind"] == "toc"
    assert synth["context_kind"] == "headings-synthetic"
    assert synth["sha256"] == real["sha256"]  # same doc -> same split later
    # Context is rebuilt from body lines only, so it takes the candidate
    # branch and must not contain the TOC rows.
    assert synth["context"].startswith("Candidate heading lines")
    assert ".........." not in synth["context"]
    # Same gold titles/levels, but printed pages are unknowable body-side.
    assert [e["title"] for e in synth["entries"]] == [e["title"] for e in real["entries"]]
    assert all(e["printed_page"] is None for e in synth["entries"])


def test_harvest_augment_skips_headings_docs(bookmarked_pdf):
    """No TOC pages -> nothing to synthesize; still a single record."""
    records, reason = harvest.harvest_pdf(
        bookmarked_pdf, min_pages=1, augment_headings=True
    )
    assert reason is None
    assert len(records) == 1


def test_harvest_cli_writes_jsonl(outlined_toc_pdf, plain_pdf, tmp_path):
    pdf_dir = tmp_path / "corpus"
    pdf_dir.mkdir()
    (pdf_dir / "good.pdf").write_bytes(Path(outlined_toc_pdf).read_bytes())
    (pdf_dir / "unlabeled.pdf").write_bytes(Path(plain_pdf).read_bytes())
    out = tmp_path / "records.jsonl"

    assert harvest.main([str(pdf_dir), "-o", str(out)]) == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["file"].endswith("good.pdf")
    assert records[0]["sha256"]


# --- title alignment --------------------------------------------------------

def test_normalize_title():
    assert harvest.normalize_title("  1.1   Background . ") == "1.1 background"


def test_align_printed_pages_partial_match():
    from pdf_bookmarker.models import OutlineEntry

    toc_rows = [
        OutlineEntry(title="1 Introduction", level=1, printed_page=3),
        OutlineEntry(title="2 Methods", level=1, printed_page=7),
    ]
    printed, alignment = harvest.align_printed_pages(
        ["1 Introduction", "Totally Absent Chapter", "2 Methods"], toc_rows
    )
    assert printed == [3, None, 7]
    assert alignment == 2 / 3


# --- arXiv feed parsing (offline) --------------------------------------------

_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <title>A Very
      Long  Title</title>
    <published>2024-01-20T12:00:00Z</published>
    <link href="http://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/math/0309136v1</id>
    <title>Old Style Id</title>
    <published>2003-09-08T12:00:00Z</published>
  </entry>
</feed>
"""


def test_parse_feed():
    papers = fetch_arxiv.parse_feed(_FEED)
    assert papers == [
        {
            "arxiv_id": "2401.12345v2",
            "title": "A Very Long Title",
            "published": "2024-01-20T12:00:00Z",
            "pdf_url": "http://arxiv.org/pdf/2401.12345v2",
        },
        {
            "arxiv_id": "math/0309136v1",
            "title": "Old Style Id",
            "published": "2003-09-08T12:00:00Z",
            "pdf_url": "http://arxiv.org/pdf/math/0309136v1",
        },
    ]


def test_safe_filename_old_style_id():
    assert fetch_arxiv._safe_filename("math/0309136v1") == "math_0309136v1.pdf"


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_get_backs_off_on_429(monkeypatch):
    import email.message
    import urllib.error

    calls = []
    headers = email.message.Message()
    headers["Retry-After"] = "0"  # arXiv really sends this on 503s

    def flaky_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests", headers, None
            )
        return _FakeResponse(b"<feed/>")

    sleeps = []
    monkeypatch.setattr(fetch_arxiv.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(fetch_arxiv.time, "sleep", sleeps.append)

    assert fetch_arxiv._get("https://x/api") == b"<feed/>"
    assert len(calls) == 3
    assert sleeps == [30, 60]  # Retry-After: 0 must not defeat the backoff


def test_search_keeps_partial_results_on_failure(monkeypatch):
    """A dead search page yields the papers already collected, not a crash."""
    pages = iter([_FEED.encode("utf-8"), None])

    def one_good_page(url):
        page = next(pages)
        if page is None:
            raise TimeoutError("boom")
        return page

    monkeypatch.setattr(fetch_arxiv, "_get", one_good_page)
    monkeypatch.setattr(fetch_arxiv.time, "sleep", lambda s: None)

    papers = fetch_arxiv.search("cat:math.LO", 100)
    assert [p["arxiv_id"] for p in papers] == ["2401.12345v2", "math/0309136v1"]
