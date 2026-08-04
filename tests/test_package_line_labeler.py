"""Tests for pdf_bookmarker/line_labeler.py."""
from pdf_bookmarker import line_labeler as ll


def row(text="Body text", label=0, **kw):
    base = {
        "text": text,
        "page": 0,
        "x": 72.0,
        "y": 100.0,
        "size": 10.0,
        "bold": False,
        "size_ratio": 1.0,
        "gap_above": 0.0,
        "words": len(text.split()),
        "label": label,
    }
    base.update(kw)
    return base


def vec(r, page_count=10, among=()):
    """One row's vector, featurized as part of a document.

    Rows are never featurized alone in production -- the window features read
    the lines around each one -- so tests must not either.
    """
    return ll.feature_matrix([r, *among], page_count)[0]


# --- reading an outline back off the labels ---------------------------------

def test_entries_come_from_the_labeled_lines():
    rows = [
        row("Introduction", label=1, page=2),
        row("some body text"),
        row("Methods", label=2, page=5),
    ]
    entries = ll.entries_from_labels(rows, [r["label"] for r in rows])
    assert [(e["title"], e["level"], e["page"]) for e in entries] == [
        ("Introduction", 1, 2),
        ("Methods", 2, 5),
    ]


def test_no_labels_gives_no_entries():
    rows = [row("a"), row("b")]
    assert ll.entries_from_labels(rows, [0, 0]) == []


def test_entry_titles_are_the_line_text_verbatim():
    """The whole point: titles are exact, never regenerated."""
    rows = [row("2.2.2 Notation and Basic Definitions", label=3)]
    entries = ll.entries_from_labels(rows, [3])
    assert entries[0]["title"] == "2.2.2 Notation and Basic Definitions"


def test_entries_keep_document_order():
    rows = [row("B", label=1, page=9), row("A", label=1, page=1)]
    entries = ll.entries_from_labels(rows, [1, 1])
    assert [e["title"] for e in entries] == ["B", "A"]


def test_blank_titles_are_dropped():
    """A labeled line with no visible text cannot become a bookmark."""
    rows = [row("   ", label=1), row("Real", label=1)]
    entries = ll.entries_from_labels(rows, [1, 1])
    assert [e["title"] for e in entries] == ["Real"]


# --- features ---------------------------------------------------------------

def test_feature_vector_is_numeric_and_fixed_width():
    vectors = [vec(row("Short")), vec(row("x" * 90))]
    assert len({len(v) for v in vectors}) == 1
    assert all(isinstance(x, float) for v in vectors for x in v)
    assert len(ll.FEATURE_NAMES) == len(vectors[0])


def test_bold_and_size_ratio_reach_the_vector():
    plain = vec(row("Heading", bold=False, size_ratio=1.0))
    heavy = vec(row("Heading", bold=True, size_ratio=2.0))
    assert plain != heavy
    i = ll.FEATURE_NAMES.index("bold")
    assert plain[i] == 0.0 and heavy[i] == 1.0


def test_page_position_is_relative_to_the_document():
    early = vec(row("x", page=0), 100)
    late = vec(row("x", page=99), 100)
    i = ll.FEATURE_NAMES.index("page_frac")
    assert early[i] == 0.0
    assert late[i] == 0.99


def test_single_page_document_does_not_divide_by_zero():
    v = vec(row("x", page=0), 1)
    assert all(x == x for x in v)  # no NaN


def test_numbering_shape_is_a_feature():
    """"3.1 Results" looks like a heading in a way "Results" does not."""
    numbered = vec(row("3.1 Results"))
    plain = vec(row("Results"))
    i = ll.FEATURE_NAMES.index("starts_numbered")
    assert numbered[i] == 1.0
    assert plain[i] == 0.0


# --- local contrast: what the neighbouring lines look like --------------------

def window(name, r, among):
    i = ll.FEATURE_NAMES.index(name)
    return ll.feature_matrix([r, *among], 10)[0][i]


def test_a_heading_is_bigger_than_the_line_under_it():
    """The strongest local signal, and one no single-line feature can carry.

    size_ratio compares a line to the *document's* body size. That cannot tell
    a heading from an equally large line inside a figure block; the ratio to
    the following line can.
    """
    assert window("size_vs_next", row("Introduction", size=18.0),
                  [row("body text", size=10.0)]) > 1.5
    assert window("size_vs_next", row("body text", size=10.0),
                  [row("more body", size=10.0)]) == 1.0


def test_bold_means_nothing_inside_an_all_bold_block():
    """`bold` fires identically for a heading and for any line of a bold block.

    The fraction of bold lines nearby is what separates them.
    """
    alone = window("window_bold_frac", row("Heading", bold=True),
                   [row("body", bold=False), row("body", bold=False)])
    inside = window("window_bold_frac", row("Heading", bold=True),
                    [row("also bold", bold=True), row("still bold", bold=True)])
    assert alone < 0.5 < inside


def test_whitespace_below_reaches_the_vector():
    assert window("gap_below", row("Heading", size=10.0),
                  [row("body", gap_above=25.0)]) == 2.5


def test_the_next_line_being_body_text_is_evidence():
    assert window("next_size_ratio", row("Heading"), [row("body", size_ratio=1.0)]) == 1.0
    assert window("next_size_ratio", row("Heading"), [row("big", size_ratio=2.0)]) == 2.0


def test_indentation_contrast_with_the_following_line():
    assert window("x_minus_next", row("Heading", x=72.0), [row("body", x=90.0)]) == -18.0


def test_neighbours_do_not_reach_across_a_page_break():
    """The line under a heading is on the same page; the next page's first line
    is not adjacent to it in any layout sense."""
    same = window("size_vs_next", row("Heading", size=20.0, page=0),
                  [row("body", size=10.0, page=0)])
    across = window("size_vs_next", row("Heading", size=20.0, page=0),
                    [row("body", size=10.0, page=1)])
    assert same == 2.0
    assert across == 1.0  # falls back to itself rather than borrowing


def test_a_lone_line_does_not_divide_by_zero():
    for value in ll.feature_matrix([row("Only line")], 1)[0]:
        assert value == value          # no NaN
        assert abs(value) != float("inf")


def test_window_features_need_the_document():
    """Featurizing a row alone would silently zero these out.

    That is the train/serve skew that cost 1.3 title F1 as float64, which is
    why feature_matrix is the only public entry point.
    """
    alone = ll.feature_matrix([row("Heading", size=18.0)], 10)[0]
    among = ll.feature_matrix([row("Heading", size=18.0), row("body", size=10.0)], 10)[0]
    assert alone != among


def test_feature_matrix_returns_one_vector_per_row():
    rows = [row("A", size=18.0), row("body"), row("B", size=18.0)]
    matrix = ll.feature_matrix(rows, 10)
    assert len(matrix) == 3
    assert {len(v) for v in matrix} == {len(ll.FEATURE_NAMES)}
    assert all(isinstance(x, float) for v in matrix for x in v)


def test_feature_matrix_of_nothing_is_nothing():
    assert ll.feature_matrix([], 10) == []


# --- text fed to the lexical model ------------------------------------------

def test_text_is_lowercased_for_the_lexical_model():
    assert ll.text_for_model("Introduction") == ll.text_for_model("INTRODUCTION")


def test_digits_collapse_so_sections_share_a_token():
    """"1 Introduction" and "7 Introduction" are the same lexical evidence."""
    assert ll.text_for_model("1 Introduction") == ll.text_for_model("7 Introduction")
    assert ll.text_for_model("2.3 Methods") == ll.text_for_model("10.4 Methods")


def test_digit_collapsing_keeps_the_words():
    assert "introduction" in ll.text_for_model("1. Introduction")


def test_whitespace_is_normalized():
    assert ll.text_for_model("  Results   and  Discussion ") == (
        ll.text_for_model("Results and Discussion")
    )


def test_distinct_wording_stays_distinct():
    assert ll.text_for_model("References") != ll.text_for_model("Appendix")


def test_gap_above_is_scaled_by_font_size():
    """Raw points are not comparable across documents; ems are."""
    v = vec(row("x", gap_above=20.0, size=10.0))
    i = ll.FEATURE_NAMES.index("gap_ems")
    assert v[i] == 2.0


# --- numbering depth ----------------------------------------------------------

def depth(text):
    i = ll.FEATURE_NAMES.index("number_depth")
    return vec(row(text))[i]


def numbered(text):
    i = ll.FEATURE_NAMES.index("starts_numbered")
    return vec(row(text))[i]


def test_depth_counts_the_components():
    """"4.1.2" is almost certainly a heading; "4" alone could be a list item,
    and the depth is also what the level classifier needs."""
    assert depth("4 Results") == 1.0
    assert depth("4.1 Setup") == 2.0
    assert depth("2.2.2 Notation") == 3.0


def test_unnumbered_lines_have_depth_zero():
    assert depth("Introduction") == 0.0
    assert depth("the model was trained on 4.1 million rows") == 0.0


def test_appendix_labels_count():
    assert depth("A.3 Proofs") == 2.0
    assert depth("B. Related Work") == 1.0
    assert depth("IV. Discussion") == 1.0


def test_an_appendix_letter_now_reads_as_numbered():
    """The labeler's rule was narrower than the locator's: "B. Related Work"
    scored 0 while the metric treated "B." as a section label."""
    assert numbered("B. Related Work") == 1.0
    assert numbered("IV. Discussion") == 1.0


def test_numbered_and_depth_agree():
    for text in ("4.1 Setup", "Introduction", "B. Related Work", "12 Appendix"):
        assert (numbered(text) == 1.0) == (depth(text) > 0)
