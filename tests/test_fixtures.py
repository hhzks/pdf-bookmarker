import fitz


def test_toc_pdf_shape(toc_pdf):
    doc = fitz.open(toc_pdf)
    assert doc.page_count == 5
    assert "Contents" in doc[1].get_text()
    assert "1 Introduction" in doc[2].get_text()


def test_offset_toc_pdf_shape(offset_toc_pdf):
    doc = fitz.open(offset_toc_pdf)
    assert doc.page_count == 8
    assert "Chapter One" in doc[3].get_text()
    assert "Chapter Three" in doc[7].get_text()


def test_headings_pdf_has_no_toc_page(headings_pdf):
    doc = fitz.open(headings_pdf)
    assert doc.page_count == 2
    assert "Contents" not in doc[0].get_text()


def test_no_text_pdf_is_textless(no_text_pdf):
    doc = fitz.open(no_text_pdf)
    assert doc[0].get_text().strip() == ""


def test_bookmarked_pdf_has_outline(bookmarked_pdf):
    assert fitz.open(bookmarked_pdf).get_toc() == [[1, "Existing", 1]]


def test_encrypted_pdf_needs_password(encrypted_pdf):
    assert fitz.open(encrypted_pdf).needs_pass
