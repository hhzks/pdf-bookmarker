"""Text extraction with font metadata. This is the seam where OCR would plug in."""
import unicodedata
from dataclasses import dataclass

import fitz  # PyMuPDF

_BOLD_FLAG = 16     # bit 4 of the span flags
_BASELINE_TOL = 2.0  # fragments this close vertically sit on one visual line
_WIDE_GAP_EMS = 2.0  # horizontal gap (in ems) marking a layout break, e.g. TOC page numbers


@dataclass
class Line:
    text: str
    page: int      # 0-based physical page index
    x: float       # left edge of the line
    y: float       # top of the line
    size: float    # largest font size in the line
    bold: bool


@dataclass
class _Fragment:
    """One raw PyMuPDF "line"; several may share a visual baseline."""
    text: str
    x0: float
    x1: float
    y: float
    size: float
    bold: bool


def has_text_layer(doc: fitz.Document) -> bool:
    """True if any of the first 10 pages contains extractable text."""
    for page_index in range(min(10, doc.page_count)):
        if doc[page_index].get_text("text").strip():
            return True
    return False


def extract_lines(doc: fitz.Document) -> list[Line]:
    """Flatten the document into visual lines with position/size/weight metadata.

    PyMuPDF splits one visual line into several "line" dicts wherever the text
    jumps horizontally (LaTeX heading numbers, TOC page numbers), so fragments
    sharing a baseline are merged back into a single Line here.
    """
    lines: list[Line] = []
    for page_index, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        lines.extend(lines_from_blocks(blocks, page_index))
    return lines


def lines_from_blocks(blocks: list[dict], page_index: int) -> list[Line]:
    """Merge a page's text blocks (from get_text("dict")) into visual lines.

    Shared by the born-digital path (extract_lines) and the OCR path
    (ocr.extract_lines_via_ocr), which supplies OCR-derived blocks.
    """
    fragments: list[_Fragment] = []
    for block in blocks:
        if block.get("type") != 0:  # 0 = text block
            continue
        for raw_line in block["lines"]:
            fragment = _parse_fragment(raw_line)
            if fragment:
                fragments.append(fragment)
    lines: list[Line] = []
    for group in _baseline_groups(fragments):
        lines.append(
            Line(
                text=_join_fragments(group),
                page=page_index,
                x=group[0].x0,
                y=min(f.y for f in group),
                size=max(f.size for f in group),
                bold=any(f.bold for f in group),
            )
        )
    return lines


def _parse_fragment(raw_line: dict) -> _Fragment | None:
    inked = [s for s in raw_line["spans"] if s["text"].strip()]
    if not inked:
        return None
    # Join ALL spans: LaTeX PDFs emit inter-word spaces as whitespace-only
    # spans. NFKC folds ligatures (e.g. "ﬁ" -> "fi") into plain letters.
    text = "".join(s["text"] for s in raw_line["spans"])
    text = unicodedata.normalize("NFKC", text).strip()
    x0, y0, x1, _ = raw_line["bbox"]
    return _Fragment(
        text=text,
        x0=x0,
        x1=x1,
        y=y0,
        size=max(s["size"] for s in inked),
        bold=any(
            s["flags"] & _BOLD_FLAG or "bold" in s["font"].lower() for s in inked
        ),
    )


def _baseline_groups(fragments: list[_Fragment]) -> list[list[_Fragment]]:
    """Group fragments that share a baseline; each group reads left to right."""
    fragments.sort(key=lambda f: (f.y, f.x0))
    groups: list[list[_Fragment]] = []
    for fragment in fragments:
        if groups and fragment.y - groups[-1][0].y <= _BASELINE_TOL:
            groups[-1].append(fragment)
        else:
            groups.append([fragment])
    for group in groups:
        group.sort(key=lambda f: f.x0)
    return groups


def _join_fragments(group: list[_Fragment]) -> str:
    """Merge a baseline group; a wide gap becomes a double space so TOC
    parsing can still see the layout break before a page number."""
    parts = [group[0].text]
    for prev, frag in zip(group, group[1:]):
        wide = frag.x0 - prev.x1 > _WIDE_GAP_EMS * frag.size
        parts.append(("  " if wide else " ") + frag.text)
    return "".join(parts)
