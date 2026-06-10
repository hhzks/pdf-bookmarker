"""Heuristic chapter/subchapter detection for PDFs with no TOC."""
import re
from collections import Counter

from .extractor import Line
from .models import OutlineEntry

_CHAPTER = re.compile(r"^(chapter|part|appendix)\s+([0-9ivxlc]+)\b", re.IGNORECASE)
_NUMBERED = re.compile(r"^(?P<num>\d+(?:\.\d+)*)[.\s]\s*\S")
_MAX_HEADING_WORDS = 12
_WRAP_SPACING_EMS = 1.5  # wrapped title lines sit closer than this; new blocks farther


def body_text_size(lines: list[Line]) -> float:
    """The dominant font size, weighted by amount of text."""
    weights: Counter[float] = Counter()
    for line in lines:
        weights[round(line.size, 1)] += len(line.text)
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

    groups = _merge_wrapped(candidates)
    # Size tiers: largest candidate size = level 1, next = level 2, ...
    sizes = sorted({round(g[0].size, 1) for g in groups}, reverse=True)
    entries: list[OutlineEntry] = []
    for group in groups:
        first = group[0]
        title = " ".join(line.text for line in group)
        num = _NUMBERED.match(title)
        if num:
            level = num.group("num").count(".") + 1
        elif _CHAPTER.match(title):
            level = 1
        else:
            level = sizes.index(round(first.size, 1)) + 1
        entries.append(OutlineEntry(title=title, level=level, page=first.page, y=first.y))
    return entries


def _merge_wrapped(candidates: list[Line]) -> list[list[Line]]:
    """Group heading lines so a title wrapping onto the next line stays one
    heading. A continuation matches the previous line's style, sits within
    normal line spacing, and does not start a numbered/chapter heading."""
    groups = [[candidates[0]]]
    for line in candidates[1:]:
        prev = groups[-1][-1]
        if (
            line.page == prev.page
            and round(line.size, 1) == round(prev.size, 1)
            and line.bold == prev.bold
            and line.y - prev.y <= _WRAP_SPACING_EMS * prev.size
            and not _NUMBERED.match(line.text)
            and not _CHAPTER.match(line.text)
        ):
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def _is_page_furniture(text: str) -> bool:
    return text.strip().isdigit()  # standalone printed page numbers
