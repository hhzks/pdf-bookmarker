import fitz
import pytest

from pdf_bookmarker import extractor, ocr


def test_available_true_when_tesseract_on_path(monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda name: "/usr/bin/tesseract")
    assert ocr.available() is True


def test_available_false_when_tesseract_missing(monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda name: None)
    assert ocr.available() is False


def test_scanned_pdf_has_no_text_layer(scanned_text_pdf):
    # Sanity check on the fixture: it must look like a scan to the pipeline.
    doc = fitz.open(scanned_text_pdf)
    assert extractor.has_text_layer(doc) is False
    doc.close()


@pytest.mark.skipif(not ocr.available(), reason="tesseract not installed")
def test_extract_lines_via_ocr_recovers_text(scanned_text_pdf):
    doc = fitz.open(scanned_text_pdf)
    lines = ocr.extract_lines_via_ocr(doc)
    doc.close()
    joined = " ".join(line.text for line in lines).lower()
    assert "introduction" in joined
    assert "methods" in joined


def test_effective_dpi_uses_full_dpi_for_normal_page():
    doc = fitz.open()
    doc.new_page()  # A4 ≈ 595x842 pt, well under the pixel budget at 300 DPI
    assert ocr._effective_dpi(doc[0]) == ocr.DPI
    doc.close()


def test_effective_dpi_clamps_oversized_page():
    doc = fitz.open()
    doc.new_page(width=5000, height=5000)  # ~69x69 inch: 300 DPI would be huge
    dpi = ocr._effective_dpi(doc[0])
    assert 72 <= dpi < ocr.DPI
    doc.close()
