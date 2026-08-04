"""Tests for training/hybrid.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import hybrid


def doc(sha, entries_per_page):
    return {"sha256": sha, "entries_per_page": entries_per_page}


def test_full_budget_routes_everything():
    docs = [doc("a", 0.1), doc("b", 5.0)]
    assert hybrid.route(docs, 1.0) == {"a", "b"}


def test_zero_budget_routes_nothing():
    assert hybrid.route([doc("a", 0.1), doc("b", 5.0)], 0.0) == set()


def test_the_sparsest_documents_are_routed_first():
    """A sparse outline is where the labeler found least and the LLM adds most."""
    docs = [doc("dense", 5.0), doc("sparse", 0.1), doc("middling", 1.0)]
    assert hybrid.route(docs, 0.34) == {"sparse"}


def test_budget_is_a_fraction_of_documents():
    docs = [doc(str(i), i) for i in range(10)]
    assert len(hybrid.route(docs, 0.5)) == 5


def test_ranking_is_per_page_not_per_document():
    """A 400-page book with 40 headings is sparse; an 8-page paper with 6 is not."""
    book = {"sha256": "book", "entries_per_page": 40 / 400}
    paper = {"sha256": "paper", "entries_per_page": 6 / 8}
    assert hybrid.route([book, paper], 0.5) == {"book"}


def test_entries_carry_their_pages_through_the_conversion():
    raw = [{"title": "Introduction", "level": 1, "page": 4, "printed_page": 2}]
    entries = hybrid.to_entries(raw)
    assert entries[0].page == 4
    assert entries[0].printed_page == 2


def test_missing_pages_become_none_rather_than_failing():
    """Labeler entries have no printed_page; LLM entries have no page."""
    entries = hybrid.to_entries([{"title": "Methods", "level": 2}])
    assert entries[0].page is None
    assert entries[0].printed_page is None
