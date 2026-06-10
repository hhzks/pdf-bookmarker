import fitz

from pdf_bookmarker.extractor import Line, extract_lines
from pdf_bookmarker.toc_detector import find_toc_pages, parse_toc


def _lines(path):
    doc = fitz.open(path)
    return extract_lines(doc), doc.page_count


def test_find_toc_pages(toc_pdf):
    lines, page_count = _lines(toc_pdf)
    assert find_toc_pages(lines, page_count) == [1]


def test_no_toc_detected(headings_pdf):
    lines, page_count = _lines(headings_pdf)
    assert find_toc_pages(lines, page_count) == []


def test_multipage_toc_continuation():
    def entry(page, i):
        return Line(f"{i} Title {i} .......... {i + 2}", page, 72, 72 + i * 18, 10, False)

    lines = [Line("Contents", 1, 72, 60, 16, True)]
    lines += [entry(1, i) for i in range(1, 5)]   # TOC page with heading
    lines += [entry(2, i) for i in range(5, 9)]   # continuation page, no heading
    lines += [Line("Body text here", 3, 72, 72, 10, False)]
    assert find_toc_pages(lines, 30) == [1, 2]


def test_parse_toc_levels_from_numbering(toc_pdf):
    lines, _ = _lines(toc_pdf)
    entries = parse_toc(lines, [1])
    assert [(e.title, e.level, e.printed_page) for e in entries] == [
        ("1 Introduction", 1, 3),
        ("1.1 Background", 2, 3),
        ("2 Methods", 1, 4),
        ("3 Results", 1, 5),
    ]


def test_parse_toc_unnumbered_uses_indentation(offset_toc_pdf):
    lines, _ = _lines(offset_toc_pdf)
    entries = parse_toc(lines, [1])
    assert [(e.title, e.level, e.printed_page) for e in entries] == [
        ("Chapter One", 1, 1),
        ("Chapter Two", 1, 3),
        ("Chapter Three", 1, 5),
    ]
