import fitz
import pytest

from pdf_bookmarker.extractor import extract_lines
from pdf_bookmarker.heading_detector import body_text_size, detect_headings


def test_body_text_size(headings_pdf):
    lines = extract_lines(fitz.open(headings_pdf))
    assert body_text_size(lines) == pytest.approx(10.0, abs=0.5)


def test_detect_headings(headings_pdf):
    lines = extract_lines(fitz.open(headings_pdf))
    entries = detect_headings(lines)
    assert [(e.title, e.level, e.page) for e in entries] == [
        ("Chapter 1 Getting Started", 1, 0),
        ("1.1 Installation", 2, 0),
        ("Chapter 2 Advanced Usage", 1, 1),
        ("2.1 Configuration", 2, 1),
    ]
    assert all(e.y is not None for e in entries)


def test_detect_headings_empty_when_uniform():
    from pdf_bookmarker.extractor import Line

    lines = [Line(f"para {i}", 0, 72, 72 + i * 14, 10, False) for i in range(20)]
    assert detect_headings(lines) == []


def test_wrapped_heading_is_one_entry():
    """A title wrapping onto the next line is one heading, not two."""
    from pdf_bookmarker.extractor import Line

    lines = [
        Line("Question 1: Critical Points, Hessian, and Local", 0, 52, 199, 20, True),
        Line("Approximation", 0, 52, 222, 20, True),  # ~1.2 em below: a wrap
        *[Line(f"body {i}", 0, 52, 260 + i * 16, 13.5, False) for i in range(20)],
        Line("Question 2: Change of Variables", 0, 52, 600, 20, True),
    ]
    entries = detect_headings(lines)
    assert [e.title for e in entries] == [
        "Question 1: Critical Points, Hessian, and Local Approximation",
        "Question 2: Change of Variables",
    ]
    assert entries[0].y == 199  # bookmark points at the first wrapped line


def test_adjacent_distinct_headings_not_merged():
    """Consecutive headings separated by block spacing stay separate."""
    from pdf_bookmarker.extractor import Line

    lines = [
        Line("High Yield Questions", 0, 52, 52, 22, True),
        Line("High-Yield Vector Calculus Questions", 0, 52, 90, 22, True),  # 1.7 em
        Line("3 Revision", 1, 52, 100, 16, True),
        Line("3.1 Propositional Logic", 1, 52, 124, 16, True),  # numbered: new heading
        *[Line(f"body {i}", 0, 52, 200 + i * 16, 13.5, False) for i in range(20)],
    ]
    titles = [e.title for e in detect_headings(lines)]
    assert "High Yield Questions" in titles
    assert "High-Yield Vector Calculus Questions" in titles
    assert "3 Revision" in titles
    assert "3.1 Propositional Logic" in titles
