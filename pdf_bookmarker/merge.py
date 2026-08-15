"""Combine two outlines of the same document into one.

Two detectors of the same PDF disagree far more than they overlap. Measured on
the 76-document evaluation set, the shipped labeler and the shipped GGUF
proposed 3665 distinct titles between them and agreed on only **30.8%** —
34.6% came from the labeler alone, 34.7% from the LLM alone. Taking the union
rather than letting the second replace the first trades precision for recall.

That complementarity is what this is for, and it pays: end-to-end,
0.8009 -> 0.8312 with `--llm` (+0.0303, CI [+0.0092, +0.0525]) and
0.8009 -> 0.8211 at the default routing (+0.0202, CI [+0.0000, +0.0418]).
Recall carries it, 0.7664 -> 0.8470.

Any figure here is only true of the pair it was measured on, so re-run
`training/route_check.py` before quoting one — and check what the prediction
file actually contains. A file labelled as the GGUF's output but holding a
labeler union made this merge look worthless for eleven days (issue #17).

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
