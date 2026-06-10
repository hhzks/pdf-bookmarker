"""Match outline entries to their physical location in the document."""
import re
from statistics import median

from .extractor import Line
from .models import OutlineEntry

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def locate_entries(
    entries: list[OutlineEntry],
    lines: list[Line],
    skip_pages: set[int] | None = None,
) -> tuple[list[OutlineEntry], int]:
    """Find each entry's title in the body. Returns (located_entries, failure_count).

    Found entries get .page and .y set. Entries with a printed-page hint that
    cannot be found keep an offset-corrected page guess (counted as a failure).
    Entries with no hint and no match are dropped (also counted as a failure).
    """
    skip_pages = skip_pages or set()
    page_count = max((l.page for l in lines), default=0) + 1
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        if line.page not in skip_pages:
            by_page.setdefault(line.page, []).append(line)

    offsets: list[int] = []  # physical_page - (printed_page - 1), from successes
    located: list[OutlineEntry] = []
    failures = 0

    for entry in entries:
        target = _normalize(entry.title)
        hint = _hint_page(entry, offsets, page_count)
        match = None
        for page_index in _pages_nearest_first(hint, page_count, skip_pages):
            for line in by_page.get(page_index, []):
                if _matches(target, _normalize(line.text)):
                    match = line
                    break
            if match:
                break
        if match:
            entry.page = match.page
            entry.y = match.y
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


def _matches(target: str, candidate: str) -> bool:
    if not target or not candidate:
        return False
    return (
        candidate == target
        or (len(target) >= 8 and candidate.startswith(target))
        or (len(candidate) >= 8 and target.startswith(candidate))
    )


def _hint_page(entry: OutlineEntry, offsets: list[int], page_count: int) -> int:
    if entry.printed_page is None:
        return 0
    offset = round(median(offsets)) if offsets else 0
    return min(max(entry.printed_page - 1 + offset, 0), page_count - 1)


def _pages_nearest_first(hint: int, page_count: int, skip_pages: set[int]):
    for delta in range(page_count):
        for page in dict.fromkeys((hint + delta, hint - delta)):
            if 0 <= page < page_count and page not in skip_pages:
                yield page
