"""Tests for training/line_labeler.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import line_labeler as ll


def row(text="Body text", label=0, **kw):
    base = {
        "text": text,
        "page": 0,
        "x": 72.0,
        "y": 100.0,
        "size": 10.0,
        "bold": False,
        "size_ratio": 1.0,
        "gap_above": 0.0,
        "words": len(text.split()),
        "label": label,
    }
    base.update(kw)
    return base


# --- reading an outline back off the labels ---------------------------------

def test_entries_come_from_the_labeled_lines():
    rows = [
        row("Introduction", label=1, page=2),
        row("some body text"),
        row("Methods", label=2, page=5),
    ]
    entries = ll.entries_from_labels(rows, [r["label"] for r in rows])
    assert [(e["title"], e["level"], e["page"]) for e in entries] == [
        ("Introduction", 1, 2),
        ("Methods", 2, 5),
    ]


def test_no_labels_gives_no_entries():
    rows = [row("a"), row("b")]
    assert ll.entries_from_labels(rows, [0, 0]) == []


def test_entry_titles_are_the_line_text_verbatim():
    """The whole point: titles are exact, never regenerated."""
    rows = [row("2.2.2 Notation and Basic Definitions", label=3)]
    entries = ll.entries_from_labels(rows, [3])
    assert entries[0]["title"] == "2.2.2 Notation and Basic Definitions"


def test_entries_keep_document_order():
    rows = [row("B", label=1, page=9), row("A", label=1, page=1)]
    entries = ll.entries_from_labels(rows, [1, 1])
    assert [e["title"] for e in entries] == ["B", "A"]


def test_blank_titles_are_dropped():
    """A labeled line with no visible text cannot become a bookmark."""
    rows = [row("   ", label=1), row("Real", label=1)]
    entries = ll.entries_from_labels(rows, [1, 1])
    assert [e["title"] for e in entries] == ["Real"]


# --- features ---------------------------------------------------------------

def test_feature_vector_is_numeric_and_fixed_width():
    vectors = [ll.feature_vector(row("Short"), 10), ll.feature_vector(row("x" * 90), 10)]
    assert len({len(v) for v in vectors}) == 1
    assert all(isinstance(x, float) for v in vectors for x in v)
    assert len(ll.FEATURE_NAMES) == len(vectors[0])


def test_bold_and_size_ratio_reach_the_vector():
    plain = ll.feature_vector(row("Heading", bold=False, size_ratio=1.0), 10)
    heavy = ll.feature_vector(row("Heading", bold=True, size_ratio=2.0), 10)
    assert plain != heavy
    i = ll.FEATURE_NAMES.index("bold")
    assert plain[i] == 0.0 and heavy[i] == 1.0


def test_page_position_is_relative_to_the_document():
    early = ll.feature_vector(row("x", page=0), 100)
    late = ll.feature_vector(row("x", page=99), 100)
    i = ll.FEATURE_NAMES.index("page_frac")
    assert early[i] == 0.0
    assert late[i] == 0.99


def test_single_page_document_does_not_divide_by_zero():
    v = ll.feature_vector(row("x", page=0), 1)
    assert all(x == x for x in v)  # no NaN


def test_numbering_shape_is_a_feature():
    """"3.1 Results" looks like a heading in a way "Results" does not."""
    numbered = ll.feature_vector(row("3.1 Results"), 10)
    plain = ll.feature_vector(row("Results"), 10)
    i = ll.FEATURE_NAMES.index("starts_numbered")
    assert numbered[i] == 1.0
    assert plain[i] == 0.0


def test_gap_above_is_scaled_by_font_size():
    """Raw points are not comparable across documents; ems are."""
    v = ll.feature_vector(row("x", gap_above=20.0, size=10.0), 10)
    i = ll.FEATURE_NAMES.index("gap_ems")
    assert v[i] == 2.0
