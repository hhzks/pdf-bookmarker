"""OCR text recognition for scanned PDFs, via PyMuPDF's Tesseract integration.

This is the OCR half of the extraction seam: it renders each page, OCRs it
with Tesseract, and reuses extractor.lines_from_blocks so OCR'd pages become
the same Line objects as born-digital pages. Requires the `tesseract` system
binary (PyMuPDF shells out to it); no extra Python dependency.
"""
import shutil

import fitz

from .extractor import Line, lines_from_blocks

DPI = 300  # render resolution for OCR; 300 is the standard accuracy/speed point
MAX_OCR_PIXELS = 40_000_000  # cap the rendered page bitmap (A4@300dpi ≈ 8.7 MP)


def available() -> bool:
    """True if the Tesseract binary is on PATH (PyMuPDF shells out to it)."""
    return bool(shutil.which("tesseract"))


def _effective_dpi(page: fitz.Page) -> int:
    """Clamp DPI so the page renders within MAX_OCR_PIXELS.

    Page size is in points (1/72 inch); at `dpi` the bitmap is
    (w_in*dpi) x (h_in*dpi) pixels. Oversized scans (large-format or very
    high-res) would otherwise render to multi-hundred-MB bitmaps and risk
    OOM-killing the worker, so for such pages we lower the DPI — degraded OCR
    beats a crash. A 72-DPI floor keeps output usable; only absurdly large
    pages (>~7700 in²) could exceed the budget at the floor.
    """
    rect = page.rect
    area_in = max((rect.width / 72.0) * (rect.height / 72.0), 1e-6)
    budget_dpi = int((MAX_OCR_PIXELS / area_in) ** 0.5)
    return max(72, min(DPI, budget_dpi))


def extract_lines_via_ocr(doc: fitz.Document) -> list[Line]:
    """OCR every page and return merged visual lines (same shape as extractor)."""
    lines: list[Line] = []
    for page_index, page in enumerate(doc):
        textpage = page.get_textpage_ocr(language="eng", dpi=_effective_dpi(page), full=True)
        blocks = page.get_text("dict", textpage=textpage)["blocks"]
        lines.extend(lines_from_blocks(blocks, page_index))
    return lines
