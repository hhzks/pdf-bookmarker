"""Tests for training/build_line_dataset.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import build_line_dataset as bld

from pdf_bookmarker.extractor import Line


def line(text, page=0, y=100.0, size=10.0, bold=False, x=72.0):
    return Line(text, page, x, y, size, bold)


# --- aligning gold bookmarks onto extracted lines ---------------------------

def test_labels_the_line_the_bookmark_points_at():
    lines = [
        line("Introduction", page=2, y=72, size=16, bold=True),
        line("body text", page=2, y=100),
    ]
    toc = [(1, "Introduction", 3)]  # get_toc pages are 1-based physical
    labels, stats = bld.align_labels(lines, toc)
    assert labels == [1, 0]
    assert stats["aligned"] == 1


def test_level_is_carried_through():
    lines = [
        line("Methods", page=0, y=72, size=14, bold=True),
        line("Sampling", page=0, y=200, size=12, bold=True),
    ]
    toc = [(1, "Methods", 1), (2, "Sampling", 1)]
    labels, _ = bld.align_labels(lines, toc)
    assert labels == [1, 2]


def test_matches_across_a_section_number_difference():
    """Bookmarks usually omit the numbering the printed heading carries."""
    lines = [line("1 Introduction", page=0, y=72, size=16, bold=True)]
    toc = [(1, "Introduction", 1)]
    labels, _ = bld.align_labels(lines, toc)
    assert labels == [1]


def test_prefers_the_heading_over_an_identical_toc_row():
    """A TOC row matches the title too; the bookmark's page decides."""
    lines = [
        line("Chapter One  1", page=0, y=100),   # TOC row on the contents page
        line("Chapter One", page=3, y=72, size=16, bold=True),  # real heading
    ]
    toc = [(1, "Chapter One", 4)]
    labels, _ = bld.align_labels(lines, toc)
    assert labels == [0, 1]


def test_repeated_titles_take_different_lines():
    lines = [
        line("Summary", page=1, y=72, size=14, bold=True),
        line("Summary", page=5, y=72, size=14, bold=True),
    ]
    toc = [(2, "Summary", 2), (2, "Summary", 6)]
    labels, stats = bld.align_labels(lines, toc)
    assert labels == [2, 2]
    assert stats["aligned"] == 2


def test_unmatched_bookmark_labels_nothing_and_is_counted():
    lines = [line("Something else", page=0, y=72)]
    toc = [(1, "Missing Heading", 1)]
    labels, stats = bld.align_labels(lines, toc)
    assert labels == [0]
    assert stats["aligned"] == 0
    assert stats["unaligned"] == 1


def test_falls_back_to_neighbouring_pages():
    """A bookmark may point one page off from the heading it names."""
    lines = [line("Results", page=4, y=72, size=16, bold=True)]
    toc = [(1, "Results", 4)]  # 1-based 4 -> physical 3, heading is on 4
    labels, _ = bld.align_labels(lines, toc)
    assert labels == [1]


def test_does_not_reach_across_the_whole_document():
    """A far-away same-titled line is not the heading the bookmark means."""
    lines = [line("Results", page=40, y=72, size=16, bold=True)]
    toc = [(1, "Results", 1)]
    labels, stats = bld.align_labels(lines, toc)
    assert labels == [0]
    assert stats["unaligned"] == 1


def test_deeper_levels_are_capped():
    """Level is a small classification target, not an unbounded integer."""
    lines = [line("Deep", page=0, y=72)]
    toc = [(9, "Deep", 1)]
    labels, _ = bld.align_labels(lines, toc, max_level=4)
    assert labels == [4]


# --- per-document feature rows ----------------------------------------------

def test_features_are_relative_to_the_document_body_size():
    lines = [
        line("Heading", page=0, y=72, size=20, bold=True),
        line("body text that dominates the document", page=0, y=100, size=10),
    ]
    rows = bld.line_features(lines)
    assert rows[0]["size_ratio"] == 2.0   # 20pt against a 10pt body
    assert rows[1]["size_ratio"] == 1.0
    assert rows[0]["bold"] is True


def test_features_include_the_gap_above_each_line():
    """Headings sit under more whitespace than body text does."""
    lines = [
        line("body", page=0, y=100, size=10),
        line("Heading", page=0, y=160, size=10),
    ]
    rows = bld.line_features(lines)
    assert rows[0]["gap_above"] == 0.0     # first line on the page
    assert rows[1]["gap_above"] == 60.0


def test_gap_resets_at_a_page_boundary():
    lines = [
        line("last line", page=0, y=700, size=10),
        line("first line", page=1, y=72, size=10),
    ]
    rows = bld.line_features(lines)
    assert rows[1]["gap_above"] == 0.0


def test_feature_rows_are_parallel_to_the_lines():
    lines = [line("a", page=0, y=72), line("b", page=0, y=90)]
    rows = bld.line_features(lines)
    assert [r["text"] for r in rows] == ["a", "b"]
    assert [r["page"] for r in rows] == [0, 0]
