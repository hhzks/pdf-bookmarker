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
