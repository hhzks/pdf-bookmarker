"""Text extraction with font metadata. This is the seam where OCR would plug in."""
from dataclasses import dataclass

import fitz  # PyMuPDF

_BOLD_FLAG = 16  # bit 4 of the span flags


@dataclass
class Line:
    text: str
    page: int      # 0-based physical page index
    x: float       # left edge of the line
    y: float       # top of the line
    size: float    # largest font size in the line
    bold: bool


def has_text_layer(doc: fitz.Document) -> bool:
    """True if any of the first 10 pages contains extractable text."""
    for page_index in range(min(10, doc.page_count)):
        if doc[page_index].get_text("text").strip():
            return True
    return False


def extract_lines(doc: fitz.Document) -> list[Line]:
    """Flatten the document into Lines with position/size/weight metadata."""
    lines: list[Line] = []
    for page_index, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # 0 = text block
                continue
            for raw_line in block["lines"]:
                spans = [s for s in raw_line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                size = max(s["size"] for s in spans)
                bold = any(
                    s["flags"] & _BOLD_FLAG or "bold" in s["font"].lower()
                    for s in spans
                )
                x0, y0, _, _ = raw_line["bbox"]
                lines.append(
                    Line(text=text, page=page_index, x=x0, y=y0, size=size, bold=bold)
                )
    return lines
