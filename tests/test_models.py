from pdf_bookmarker.models import OutlineEntry


def test_outline_entry_defaults():
    e = OutlineEntry(title="Intro", level=1)
    assert e.page is None
    assert e.y is None
    assert e.printed_page is None
