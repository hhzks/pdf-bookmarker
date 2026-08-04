"""Combine two outlines of the same document into one.

Two detectors of the same PDF disagree more than they overlap. Measured on the
77-document evaluation set, a line-labeling detector and the LLM proposed 2752
distinct titles between them and agreed on only 62.7% — 19.1% came from one,
18.1% from the other. Taking the union rather than letting the second replace
the first is worth +4.3 title F1 (0.7671 -> 0.8103), almost all of it recall
(0.7343 -> 0.8525), against roughly 2 points of precision.

The primary outline is the one whose positions are trusted: its entries are
kept verbatim, and the secondary only contributes headings the primary never
proposed.
"""
from .locator import strip_section_numbers
from .models import OutlineEntry

_LAST = float("inf")


def _key(title: str) -> str:
    """Number-insensitive form, so "1 Introduction" and "Introduction" are one
    heading rather than two — the two sources rarely agree on numbering."""
    stripped = strip_section_numbers(title.strip()).lower()
    return " ".join(
        "".join(c for c in stripped if c.isalnum() or c.isspace()).split()
    )


def merge_outlines(
    primary: list[OutlineEntry], secondary: list[OutlineEntry]
) -> list[OutlineEntry]:
    """Union of two outlines, ordered by position in the document.

    A title the primary already proposed is dropped from the secondary. Counts
    are respected rather than sets: a document with three "Summary" headings
    keeps three, and a secondary that proposes a fourth contributes one.

    Matching is by count, not position, so a secondary "Summary" somewhere the
    primary has nothing is still absorbed if the primary already proposed as
    many. Position-aware matching would keep it, but that is not the variant
    the +4.3 F1 was measured on, so it is not the variant that ships.

    Entries with no page sort last: an unlocated entry is not evidence that the
    heading belongs at the front of the document, and ordering it there would
    corrupt the nesting that sanitize_levels then has to repair.
    """
    counts: dict[str, int] = {}
    for e in primary:
        counts[_key(e.title)] = counts.get(_key(e.title), 0) + 1

    merged = list(primary)
    for e in secondary:
        key = _key(e.title)
        if counts.get(key):
            counts[key] -= 1  # already proposed by the primary; skip this one
            continue
        merged.append(e)

    # Stable sort, so entries sharing a position keep primary-then-secondary
    # order and an outline that was already in reading order is undisturbed.
    return sorted(
        merged,
        key=lambda e: (
            _LAST if e.page is None else e.page,
            _LAST if e.y is None else e.y,
        ),
    )
