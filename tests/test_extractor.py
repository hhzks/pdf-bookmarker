import fitz

from pdf_bookmarker.extractor import extract_lines, has_text_layer


def test_has_text_layer(toc_pdf, no_text_pdf):
    assert has_text_layer(fitz.open(toc_pdf)) is True
    assert has_text_layer(fitz.open(no_text_pdf)) is False


def test_extract_lines_metadata(toc_pdf):
    lines = extract_lines(fitz.open(toc_pdf))
    contents = [l for l in lines if l.text == "Contents"]
    assert len(contents) == 1
    heading = contents[0]
    assert heading.page == 1
    assert heading.bold is True
    assert heading.size > 12
    body = [l for l in lines if l.page == 2 and "Lorem" in l.text]
    assert body
    assert all(not l.bold for l in body)
    assert all(l.size < 12 for l in body)


def test_extract_lines_positions(toc_pdf):
    lines = [l for l in lines_on_page(toc_pdf, 1)]
    ys = [l.y for l in lines]
    assert ys == sorted(ys)  # top-to-bottom order within the page
    indented = [l for l in lines if l.text.startswith("1.1")]
    flush = [l for l in lines if l.text.startswith("2 Methods")]
    assert indented[0].x > flush[0].x


def lines_on_page(path, page_index):
    return [l for l in extract_lines(fitz.open(path)) if l.page == page_index]
