"""Heuristic chapter/subchapter detection for PDFs with no TOC."""
import re
from collections import Counter

from .extractor import Line
from .models import OutlineEntry

_CHAPTER = re.compile(r"^(chapter|part|appendix)\s+([0-9ivxlc]+)\b", re.IGNORECASE)
_NUMBERED = re.compile(r"^(?P<num>\d+(?:\.\d+)*)[.\s]\s*\S")
_MAX_HEADING_WORDS = 12


def body_text_size(lines: list[Line]) -> float:
    """The dominant font size, weighted by amount of text."""
    weights: Counter[float] = Counter()
    for line in lines:
        weights[round(line.size, 1)] += len(line.text.split())
    return weights.most_common(1)[0][0]


def detect_headings(lines: list[Line]) -> list[OutlineEntry]:
    body = body_text_size(lines)
    candidates = [
        line
        for line in lines
        if len(line.text.split()) <= _MAX_HEADING_WORDS
        and not _is_page_furniture(line.text)
        and (
            line.size >= body * 1.15
            or (line.bold and (_CHAPTER.match(line.text) or _NUMBERED.match(line.text)))
        )
    ]
    if not candidates:
        return []

    # Size tiers: largest candidate size = level 1, next = level 2, ...
    sizes = sorted({round(c.size, 1) for c in candidates}, reverse=True)
    entries: list[OutlineEntry] = []
    for line in candidates:
        num = _NUMBERED.match(line.text)
        if num:
            level = num.group("num").count(".") + 1
        elif _CHAPTER.match(line.text):
            level = 1
        else:
            level = sizes.index(round(line.size, 1)) + 1
        entries.append(OutlineEntry(title=line.text, level=level, page=line.page, y=line.y))
    return entries


def _is_page_furniture(text: str) -> bool:
    return text.strip().isdigit()  # standalone printed page numbers
