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
