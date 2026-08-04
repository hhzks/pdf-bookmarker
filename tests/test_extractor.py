import fitz

from pdf_bookmarker import extractor
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


def _pdf_with_an_image():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Chapter 1", fontsize=16, fontname="hebo")
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 64))
    pixmap.set_rect(pixmap.irect, (200, 30, 30))
    page.insert_image(fitz.Rect(72, 200, 136, 264), pixmap=pixmap)
    return doc


def test_images_are_never_decoded(monkeypatch):
    """Ask PyMuPDF for text without images: we discard them anyway.

    lines_from_blocks drops every non-text block, so decoding them is pure
    latency -- and it is not a small amount. Extraction is ~88% of the labeler
    path's runtime and image decoding is ~70% of that; over the 76-document
    evaluation set, skipping it ran 2.81x faster (39.6s -> 14.1s, worst
    document 14.0s -> 0.35s) for a byte-identical outline.
    """
    seen = []
    original = extractor.lines_from_blocks

    def spy(blocks, page_index):
        seen.extend(blocks)
        return original(blocks, page_index)

    monkeypatch.setattr(extractor, "lines_from_blocks", spy)
    lines = extract_lines(_pdf_with_an_image())

    assert [l.text for l in lines] == ["Chapter 1"]  # the text still arrives
    assert seen, "no blocks reached lines_from_blocks"
    assert [b for b in seen if b.get("type") == 0], "no text block reached it"
    assert not [b for b in seen if b.get("type") != 0], (
        "an image block was decoded and handed over just to be discarded"
    )


def test_text_flags_drop_images_and_nothing_else():
    """Only the image bit may differ from PyMuPDF's default for "dict".

    TEXT_PRESERVE_WHITESPACE is load-bearing: LaTeX emits inter-word spaces as
    whitespace-only spans and _parse_fragment joins all spans to recover them
    (see test_fragment_keeps_space_only_spans). Turning flags off wholesale to
    "go faster" would run those words together.
    """
    assert extractor._TEXT_FLAGS == fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
