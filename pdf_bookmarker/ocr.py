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


def available() -> bool:
    """True if the Tesseract binary is on PATH (PyMuPDF shells out to it)."""
    return bool(shutil.which("tesseract"))


def extract_lines_via_ocr(doc: fitz.Document) -> list[Line]:
    """OCR every page and return merged visual lines (same shape as extractor)."""
    lines: list[Line] = []
    for page_index, page in enumerate(doc):
        textpage = page.get_textpage_ocr(language="eng", dpi=DPI, full=True)
        blocks = page.get_text("dict", textpage=textpage)["blocks"]
        lines.extend(lines_from_blocks(blocks, page_index))
    return lines
