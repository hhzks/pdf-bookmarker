import fitz

from pdf_bookmarker.extractor import Line, extract_lines
from pdf_bookmarker.locator import locate_entries
from pdf_bookmarker.models import OutlineEntry
from pdf_bookmarker.toc_detector import parse_toc


def test_locates_exact_pages(toc_pdf):
    lines = extract_lines(fitz.open(toc_pdf))
    entries = parse_toc(lines, [1])
    located, failures = locate_entries(entries, lines, skip_pages={1})
    assert failures == 0
    assert [e.page for e in located] == [2, 2, 3, 4]
    assert all(e.y is not None for e in located)


def test_offset_correction(offset_toc_pdf):
    lines = extract_lines(fitz.open(offset_toc_pdf))
    entries = parse_toc(lines, [1])
    located, failures = locate_entries(entries, lines, skip_pages={1})
    assert failures == 0
    assert [e.page for e in located] == [3, 5, 7]


def test_finds_heading_on_page_shared_with_toc_tail():
    """A section can start on the same physical page where the TOC ends."""
    lines = [
        Line("Contents", 0, 72, 72, 16, True),
        Line("1 Reading  2", 0, 72, 100, 10, False),
        Line("5.1 Tail . . . . .  40", 1, 72, 72, 10, False),  # TOC continuation
        Line("1 Reading", 1, 72, 400, 14, True),               # real heading below it
    ]
    entries = [OutlineEntry("1 Reading", 1, printed_page=2)]
    located, failures = locate_entries(entries, lines, skip_pages={0, 1})
    assert failures == 0
    assert located[0].page == 1
    assert located[0].y == 400


def test_does_not_match_toc_rows_on_toc_pages():
    """The TOC's own entry rows must never count as the section heading."""
    lines = [
        Line("Chapter One .......... 1", 1, 72, 100, 10, False),  # TOC row
        Line("Chapter One", 3, 72, 72, 16, True),                 # real heading
    ]
    entries = [OutlineEntry("Chapter One", 1, printed_page=1)]
    located, failures = locate_entries(entries, lines, skip_pages={1})
    assert failures == 0
    assert located[0].page == 3


def test_unfound_entry_falls_back_to_hint():
    lines = [Line("Hello world", 0, 72, 72, 10, False)]
    entries = [OutlineEntry("Missing Chapter", 1, printed_page=1)]
    located, failures = locate_entries(entries, lines)
    assert failures == 1
    assert located[0].page == 0  # offset-corrected hint, clamped to the document
    assert located[0].y is None


def test_unfound_entry_without_hint_is_dropped():
    lines = [Line("Hello world", 0, 72, 72, 10, False)]
    entries = [OutlineEntry("Missing Chapter", 1)]
    located, failures = locate_entries(entries, lines)
    assert located == []
    assert failures == 1
