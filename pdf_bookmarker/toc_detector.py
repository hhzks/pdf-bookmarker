"""Find and parse a table of contents."""
import re

from .extractor import Line
from .models import OutlineEntry

_TOC_HEADING = re.compile(r"^(table of )?contents$", re.IGNORECASE)
# "1.2 Title ...... 34" / "Title    34" — ≥2 separator chars before the page number
_TOC_LINE = re.compile(r"^(?P<title>.+?)[\s.]{2,}(?P<page>\d{1,4})$")
_NUMBERING = re.compile(r"^(?P<num>\d+(?:\.\d+)*)[.\s]\s*")


def is_toc_row(text: str) -> bool:
    """True for lines that belong to the TOC itself (its heading or entry rows)."""
    text = text.strip()
    return bool(_TOC_HEADING.match(text) or _TOC_LINE.match(text))


def find_toc_pages(lines: list[Line], page_count: int) -> list[int]:
    """Return the (contiguous) physical page indices that look like a TOC."""
    scan_limit = max(30, int(page_count * 0.15))
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        if line.page < scan_limit:
            by_page.setdefault(line.page, []).append(line)

    toc_pages: list[int] = []
    for page_index in sorted(by_page):
        page_lines = by_page[page_index]
        has_heading = any(_TOC_HEADING.match(l.text.strip()) for l in page_lines)
        entry_count = sum(1 for l in page_lines if _TOC_LINE.match(l.text.strip()))
        is_toc = (has_heading and entry_count >= 3) or entry_count >= 8
        if toc_pages and page_index == toc_pages[-1] + 1 and entry_count >= 3:
            is_toc = True  # continuation page of a multi-page TOC
        if is_toc:
            toc_pages.append(page_index)
        elif toc_pages:
            break  # TOC pages are contiguous; stop after the run ends
    return toc_pages


def parse_toc(lines: list[Line], toc_pages: list[int]) -> list[OutlineEntry]:
    """Parse TOC entry lines into OutlineEntries with levels and printed pages."""
    toc_page_set = set(toc_pages)
    raw: list[tuple[Line, str, int]] = []
    for line in lines:
        if line.page not in toc_page_set:
            continue
        m = _TOC_LINE.match(line.text.strip())
        if not m:
            continue
        title = m.group("title").strip(" .")
        raw.append((line, title, int(m.group("page"))))
    if not raw:
        return []

    numbered = [_level_from_numbering(title) for _, title, _ in raw]
    if all(level is not None for level in numbered):
        levels = numbered
    else:
        levels = _levels_from_indentation([line.x for line, _, _ in raw])

    return [
        OutlineEntry(title=title, level=level, printed_page=page)
        for (_, title, page), level in zip(raw, levels)
    ]


def _level_from_numbering(title: str) -> int | None:
    m = _NUMBERING.match(title)
    if m:
        return m.group("num").count(".") + 1
    return None


def _levels_from_indentation(xs: list[float]) -> list[int]:
    """Cluster left edges into tiers; deeper indentation = deeper level."""
    tiers: list[float] = []
    for x in sorted(set(xs)):
        if not tiers or x - tiers[-1] >= 5:  # merge offsets closer than 5pt
            tiers.append(x)

    def tier_of(x: float) -> int:
        for i, t in enumerate(tiers):
            if x < t + 5:
                return i
        return len(tiers) - 1

    return [tier_of(x) + 1 for x in xs]
