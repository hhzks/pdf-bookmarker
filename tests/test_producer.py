"""Tests for pdf_bookmarker/producer.py — which model a document gets."""
import fitz
import pytest

from pdf_bookmarker import producer


@pytest.mark.parametrize("name, expected", [
    ("pdfTeX-1.40.21", True),
    ("LuaHBTeX, Version 1.13.0", True),
    ("dvips + GPL Ghostscript 9.50", True),
    ("xdvipdfmx (20200315)", True),
    ("LaTeX with hyperref", True),
    # arXiv rewrites every PDF it builds; "tex\b" alone misses this.
    ("pikepdf 8.15.1 arXiv GenPDF (tex2pdf:8def8d8)", True),
    ("Microsoft® Word for Microsoft 365", False),
    ("SoftMaker Textmaker", False),
    ("Adobe InDesign CC 2015", False),
    ("", False),
])
def test_is_tex(name, expected):
    assert producer.is_tex(name) is expected


def _doc(**metadata):
    doc = fitz.open()
    doc.new_page()
    doc.set_metadata(metadata)
    return doc


def test_document_is_tex_reads_the_producer():
    assert producer.document_is_tex(_doc(producer="pdfTeX-1.40.25"))


def test_document_is_tex_reads_the_creator_too():
    """arXiv's rebuild names TeX only in Creator."""
    doc = _doc(producer="pikepdf 8.15.1", creator="arXiv GenPDF (tex2pdf:8def8d8)")
    assert producer.document_is_tex(doc)


def test_a_document_with_no_metadata_is_not_tex():
    assert not producer.document_is_tex(_doc())
