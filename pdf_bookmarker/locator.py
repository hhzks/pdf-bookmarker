"""Match outline entries to their physical location in the document."""
import re
from statistics import median

from .extractor import Line
from .models import OutlineEntry
from .toc_detector import is_toc_row

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

# A leading section label: a dotted sequence ("4.1", "2.2.2"), a number with a
# trailing dot ("1."), or a bare one- or two-digit number ("1", "12"), each
# followed by whitespace. Bare numbers are capped at two digits so a year in a
# real title ("2026 Annual Report") is not mistaken for a label, and the
# required whitespace leaves "3D Reconstruction" alone.
_SECTION_NUMBER = re.compile(r"^(\d+(\.\d+)+\.?|\d+\.|\d{1,2})\s+")


def strip_section_numbers(title: str) -> str:
    """Drop a leading section label from a heading title.

    The printed heading usually carries numbering that a TOC entry or an LLM
    may omit (and PDFs' embedded bookmarks usually omit too), so "1
    Introduction" and "Introduction" have to be recognised as one heading.
    """
    return _SECTION_NUMBER.sub("", title.strip())


def locate_entries(
    entries: list[OutlineEntry],
    lines: list[Line],
    skip_pages: set[int] | None = None,
    snap_titles: bool = True,
) -> tuple[list[OutlineEntry], int]:
    """Find each entry's title in the body. Returns (located_entries, failure_count).

    Found entries get .page and .y set. Entries with a printed-page hint that
    cannot be found keep an offset-corrected page guess (counted as a failure).
    Entries with no hint and no match are dropped (also counted as a failure).

    skip_pages holds the TOC's own pages: their TOC-style rows must never be
    matched, but a section heading can share a page with the end of the TOC.

    snap_titles adopts the matched line's own text, so every located bookmark
    reads the way the heading is actually printed instead of however the TOC
    row or the model happened to write it. Without it the outline mixes
    conventions within one document: on the 77-document test set it rewrites
    69% of titles, which is how inconsistent the raw output is.

    It only ever adopts a section-label difference — the two must be the same
    heading once labels are stripped — because a prefix match can otherwise
    reach a far longer line ("Discussion" -> "Discussion of Results and Future
    Work") and rewrite the title outright.

    Cost is ~0.4 title F1 measured with evaluate.py --ignore-section-numbers
    (0.6966 -> 0.6929 for v3). The strict metric drops far more, but only
    because gold comes from embedded bookmarks that omit the numbering the page
    shows; that is a convention difference, not an accuracy one.
    """
    skip_pages = skip_pages or set()
    page_count = max((l.page for l in lines), default=0) + 1
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        if line.page in skip_pages and is_toc_row(line.text):
            continue
        by_page.setdefault(line.page, []).append(line)

    offsets: list[int] = []  # physical_page - (printed_page - 1), from successes
    located: list[OutlineEntry] = []
    failures = 0

    for entry in entries:
        hint = _hint_page(entry, offsets, page_count)
        match = None
        for page_index in _pages_nearest_first(hint, page_count):
            for line in by_page.get(page_index, []):
                if _matches(entry.title, line.text):
                    match = line
                    break
            if match:
                break
        if match:
            entry.page = match.page
            entry.y = match.y
            # Adopt the page's own wording only when the two are the same
            # heading apart from a section label. A prefix match can reach a
            # far longer line ("Discussion" -> "Discussion of Results and
            # Future Work"), and adopting that would rewrite the title.
            if snap_titles and _bare(match.text) == _bare(entry.title):
                entry.title = match.text.strip()
            if entry.printed_page is not None:
                offsets.append(match.page - (entry.printed_page - 1))
            located.append(entry)
        elif entry.printed_page is not None:
            entry.page = _hint_page(entry, offsets, page_count)
            failures += 1
            located.append(entry)
        else:
            failures += 1  # no hint and no match: drop the entry
    return located, failures


def _normalize(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", text.lower())).strip()


def _prefix_match(target: str, candidate: str) -> bool:
    if not target or not candidate:
        return False
    return (
        candidate == target
        or (len(target) >= 8 and candidate.startswith(target))
        or (len(candidate) >= 8 and target.startswith(candidate))
    )


def _bare(text: str) -> str:
    """Normalized form with any leading section label removed."""
    return _normalize(strip_section_numbers(text))


def _matches(target: str, candidate: str) -> bool:
    """Compare raw titles, tolerating a section number on either side.

    A TOC entry or an LLM may write "Introduction" where the printed heading
    says "1 Introduction"; without this the entry never locates and its page
    falls back to a guess. Stripping happens before normalization because
    normalization removes the dots that identify a label ("2.2.2" -> "222").
    """
    exact_target, exact_candidate = _normalize(target), _normalize(candidate)
    if _prefix_match(exact_target, exact_candidate):
        return True
    bare_target, bare_candidate = _bare(target), _bare(candidate)
    if bare_target == exact_target and bare_candidate == exact_candidate:
        return False  # neither side carried a label; nothing new to try
    return _prefix_match(bare_target, bare_candidate)


def _hint_page(entry: OutlineEntry, offsets: list[int], page_count: int) -> int:
    if entry.printed_page is None:
        return 0
    offset = round(median(offsets)) if offsets else 0
    return min(max(entry.printed_page - 1 + offset, 0), page_count - 1)


def _pages_nearest_first(hint: int, page_count: int):
    for delta in range(page_count):
        for page in dict.fromkeys((hint + delta, hint - delta)):
            if 0 <= page < page_count:
                yield page
