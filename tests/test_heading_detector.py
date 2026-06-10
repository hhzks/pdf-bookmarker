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
