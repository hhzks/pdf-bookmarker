"""Tests for training/fetch_nist.py (offline parts)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import fetch_nist


def test_candidate_urls_order():
    urls = fetch_nist.candidate_urls(53)
    # Highest revision first, unrevised next, legacy path last.
    assert urls[0].endswith("NIST.SP.800-53r5.pdf")
    assert urls[4].endswith("NIST.SP.800-53r1.pdf")
    assert urls[5].endswith("NIST.SP.800-53.pdf")
    assert urls[6].endswith("nistspecialpublication800-53.pdf")
    assert len(urls) == 7


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_try_download_retries_transient_timeouts(monkeypatch):
    """A stalled transfer costs a retry, never the whole run."""
    calls = []

    def flaky_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) < 3:
            raise TimeoutError("The read operation timed out")
        return _FakeResponse(b"%PDF-1.7 fake")

    monkeypatch.setattr(fetch_nist.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(fetch_nist.time, "sleep", lambda s: None)

    assert fetch_nist.try_download("https://x/NIST.SP.800-53r5.pdf") == b"%PDF-1.7 fake"
    assert len(calls) == 3


def test_try_download_gives_up_after_retries(monkeypatch):
    def always_timeout(request, timeout):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(fetch_nist.urllib.request, "urlopen", always_timeout)
    monkeypatch.setattr(fetch_nist.time, "sleep", lambda s: None)

    assert fetch_nist.try_download("https://x/NIST.SP.800-53r5.pdf") is None
