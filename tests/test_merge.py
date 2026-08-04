from pdf_bookmarker.merge import merge_outlines
from pdf_bookmarker.models import OutlineEntry


def entry(title, level=1, page=0, y=100.0, printed_page=None):
    return OutlineEntry(title, level, page, y, printed_page)


def test_keeps_every_primary_entry():
    primary = [entry("Introduction", page=1), entry("Methods", page=5)]
    merged = merge_outlines(primary, [])
    assert [e.title for e in merged] == ["Introduction", "Methods"]


def test_adds_entries_the_primary_missed():
    primary = [entry("Introduction", page=1)]
    secondary = [entry("Results", page=9)]
    merged = merge_outlines(primary, secondary)
    assert [e.title for e in merged] == ["Introduction", "Results"]


def test_orders_the_result_by_position_in_the_document():
    primary = [entry("Conclusion", page=9)]
    secondary = [entry("Introduction", page=1)]
    merged = merge_outlines(primary, secondary)
    assert [e.title for e in merged] == ["Introduction", "Conclusion"]


def test_orders_within_a_page_by_height():
    primary = [entry("Lower", page=3, y=400)]
    secondary = [entry("Upper", page=3, y=100)]
    assert [e.title for e in merge_outlines(primary, secondary)] == ["Upper", "Lower"]


def test_a_duplicate_title_does_not_appear_twice():
    primary = [entry("Introduction", page=1)]
    secondary = [entry("Introduction", page=1)]
    merged = merge_outlines(primary, secondary)
    assert len(merged) == 1


def test_the_primary_wins_a_duplicate():
    """The primary knows its exact page; the secondary's is inferred."""
    primary = [entry("Introduction", level=1, page=4)]
    secondary = [entry("Introduction", level=3, page=99)]
    merged = merge_outlines(primary, secondary)
    assert merged[0].level == 1
    assert merged[0].page == 4


def test_duplicates_are_matched_across_a_section_number_difference():
    """"1 Introduction" and "Introduction" are one heading, not two."""
    primary = [entry("1 Introduction", page=1)]
    secondary = [entry("Introduction", page=1)]
    assert len(merge_outlines(primary, secondary)) == 1


def test_a_title_repeated_in_the_document_survives_twice():
    """Three chapters can each have a "Summary"; they are distinct headings."""
    primary = [entry("Summary", page=3), entry("Summary", page=40)]
    merged = merge_outlines(primary, [])
    assert len(merged) == 2


def test_duplicates_are_absorbed_one_for_one_not_as_a_set():
    """Counts matter: a third "Summary" is a heading the primary never found."""
    primary = [entry("Summary", page=3), entry("Summary", page=40)]
    secondary = [entry("Summary", page=3), entry("Summary", page=40),
                 entry("Summary", page=70)]
    merged = merge_outlines(primary, secondary)
    assert len(merged) == 3
    assert [e.page for e in merged] == [3, 40, 70]


def test_matching_is_by_count_not_position():
    """Known limitation, asserted so a change to it is deliberate.

    The secondary's second "Summary" sits at page 70 where the primary has
    nothing, but the primary already proposed two, so it is absorbed. Position
    -aware matching would keep it; that is not what was measured, so it is not
    what ships.
    """
    primary = [entry("Summary", page=3), entry("Summary", page=40)]
    secondary = [entry("Summary", page=3), entry("Summary", page=70)]
    assert len(merge_outlines(primary, secondary)) == 2


def test_entries_without_a_page_go_last_rather_than_first():
    """An unlocated entry must not sort ahead of the whole document."""
    primary = [entry("Located", page=2)]
    secondary = [OutlineEntry("Unlocated", 1)]
    merged = merge_outlines(primary, secondary)
    assert [e.title for e in merged] == ["Located", "Unlocated"]


def test_empty_inputs_give_an_empty_outline():
    assert merge_outlines([], []) == []


def test_inputs_are_not_mutated():
    primary = [entry("Introduction", page=1)]
    secondary = [entry("Results", page=9)]
    merge_outlines(primary, secondary)
    assert len(primary) == 1 and len(secondary) == 1
