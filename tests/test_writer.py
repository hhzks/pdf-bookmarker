import fitz

from pdf_bookmarker.models import OutlineEntry
from pdf_bookmarker.writer import sanitize_levels, write_outline


def test_sanitize_levels_clamps_jumps():
    entries = [OutlineEntry("A", 2), OutlineEntry("B", 4), OutlineEntry("C", 1)]
    out = sanitize_levels(entries)
    assert [e.level for e in out] == [1, 2, 1]


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
