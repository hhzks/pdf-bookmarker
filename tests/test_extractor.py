import fitz

from pdf_bookmarker.extractor import _parse_fragment, extract_lines, has_text_layer


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


def test_merges_same_baseline_fragments():
    """LaTeX PDFs emit the heading number and title as separate text objects."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "3.1", fontsize=14, fontname="hebo")
    page.insert_text((110, 100), "Propositional Logic", fontsize=14, fontname="hebo")
    lines = extract_lines(doc)
    assert [l.text for l in lines] == ["3.1 Propositional Logic"]
    assert lines[0].x == 72  # leftmost fragment anchors the merged line
    assert lines[0].size == 14


def test_wide_gap_kept_as_double_space():
    """TOC rows without dot leaders still need >=2 separator chars to parse."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "1", fontsize=10)
    page.insert_text((90, 100), "Reading", fontsize=10)
    page.insert_text((500, 100), "2", fontsize=10)
    lines = extract_lines(doc)
    assert [l.text for l in lines] == ["1 Reading  2"]


def _span(text, size=10.0, flags=0, font="Helvetica"):
    return {"text": text, "size": size, "flags": flags, "font": font}


def test_fragment_keeps_space_only_spans():
    """LaTeX PDFs emit inter-word spaces as their own spans; don't drop them."""
    raw_line = {
        "bbox": (72.0, 85.0, 200.0, 104.0),
        "spans": [_span("Propositional"), _span(" "), _span("Logic")],
    }
    assert _parse_fragment(raw_line).text == "Propositional Logic"


def test_fragment_normalizes_ligatures():
    raw_line = {
        "bbox": (72.0, 85.0, 200.0, 104.0),
        "spans": [_span("satisﬁable")],
    }
    assert _parse_fragment(raw_line).text == "satisfiable"
