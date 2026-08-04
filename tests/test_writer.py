import fitz

from pdf_bookmarker.models import OutlineEntry
from pdf_bookmarker.writer import sanitize_levels, write_outline


def test_sanitize_levels_clamps_jumps():
    entries = [OutlineEntry("A", 2), OutlineEntry("B", 4), OutlineEntry("C", 1)]
    out = sanitize_levels(entries)
    assert [e.level for e in out] == [1, 2, 1]


def test_sanitize_levels_raises_levels_below_one():
    """set_toc rejects level < 1 outright, and an LLM can emit 0."""
    entries = [OutlineEntry("A", 0), OutlineEntry("B", 2), OutlineEntry("C", -3)]
    out = sanitize_levels(entries)
    # A floors to 1; B's nesting survives on top of it; C floors to 1.
    assert [e.level for e in out] == [1, 2, 1]


def test_write_outline_accepts_a_zero_level_entry(tmp_path):
    """Regression: a level-0 entry used to reach set_toc and raise ValueError."""
    doc = fitz.open()
    for _ in range(2):
        doc.new_page()
    out = tmp_path / "out.pdf"
    count = write_outline(doc, [OutlineEntry("Zero", 0, page=0)], str(out))
    assert count == 1
    assert fitz.open(out).get_toc()[0][0] == 1


def test_write_outline_sets_toc(tmp_path):
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    entries = [
        OutlineEntry("One", 1, page=0, y=72.0),
        OutlineEntry("Sub", 2, page=1, y=100.0),
        OutlineEntry("Unlocated", 1, page=None),
    ]
    out = tmp_path / "out.pdf"
    count = write_outline(doc, entries, str(out))
    assert count == 2  # the unlocated entry is skipped
    toc = fitz.open(str(out)).get_toc()
    assert [item[:3] for item in toc] == [[1, "One", 1], [2, "Sub", 2]]
