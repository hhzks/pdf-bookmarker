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


# --- snapping titles to the text actually on the page -----------------------

def test_locates_across_a_section_number_difference():
    """An LLM or TOC row may drop the number the printed heading carries."""
    lines = [
        Line("1 Introduction", 0, 72, 72, 16, True),
        Line("body text here", 0, 72, 100, 10, False),
    ]
    entries = [OutlineEntry("Introduction", 1, printed_page=1)]
    located, failures = locate_entries(entries, lines)
    assert failures == 0
    assert located[0].page == 0          # found despite the numbering
    assert located[0].title == "Introduction"   # not snapped: default is off


def test_snaps_title_to_the_matched_body_line_when_asked():
    lines = [Line("1 Introduction", 0, 72, 72, 16, True)]
    entries = [OutlineEntry("Introduction", 1, printed_page=1)]
    located, _ = locate_entries(entries, lines, snap_titles=True)
    assert located[0].title == "1 Introduction"


def test_snapping_leaves_an_exact_match_alone():
    lines = [Line("2 Methods", 0, 72, 72, 16, True)]
    entries = [OutlineEntry("2 Methods", 1, printed_page=1)]
    located, _ = locate_entries(entries, lines, snap_titles=True)
    assert located[0].title == "2 Methods"


def test_snapping_never_truncates_a_longer_title():
    """The line is a fragment of the entry; adopting it would lose text."""
    lines = [Line("Results and", 0, 72, 72, 16, True)]
    entries = [OutlineEntry("Results and Discussion", 1, printed_page=1)]
    located, _ = locate_entries(entries, lines, snap_titles=True)
    assert located[0].title == "Results and Discussion"


def test_snapping_is_off_by_default():
    """It measurably hurt on the fixed test set; see locate_entries docstring."""
    lines = [Line("1 Introduction", 0, 72, 72, 16, True)]
    entries = [OutlineEntry("Introduction", 1, printed_page=1)]
    located, _ = locate_entries(entries, lines)
    assert located[0].title == "Introduction"


def test_unlocated_entry_keeps_its_title():
    lines = [Line("Hello world", 0, 72, 72, 10, False)]
    entries = [OutlineEntry("Missing Chapter", 1, printed_page=1)]
    located, failures = locate_entries(entries, lines)
    assert failures == 1
    assert located[0].title == "Missing Chapter"
