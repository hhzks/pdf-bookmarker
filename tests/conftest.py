import fitz
import pytest

BODY = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."


def _add_page(doc, rows):
    """rows: list of (text, fontsize, fontname, x)."""
    page = doc.new_page()
    y = 72.0
    for text, size, font, x in rows:
        page.insert_text((x, y), text, fontsize=size, fontname=font)
        y += size * 1.8


def _body_rows(n=4):
    return [(BODY, 10, "helv", 72) for _ in range(n)]


def _save(doc, tmp_path_factory, name, **save_kwargs):
    path = tmp_path_factory.mktemp("pdfs") / name
    doc.save(str(path), **save_kwargs)
    doc.close()
    return path


@pytest.fixture(scope="session")
def toc_pdf(tmp_path_factory):
    """5 pages, numbered TOC on physical page 1; printed page N == physical index N-1."""
    doc = fitz.open()
    _add_page(doc, [("My Book", 24, "hebo", 72)])
    _add_page(doc, [
        ("Contents", 16, "hebo", 72),
        ("1 Introduction .......... 3", 10, "helv", 72),
        ("1.1 Background .......... 3", 10, "helv", 90),
        ("2 Methods .......... 4", 10, "helv", 72),
        ("3 Results .......... 5", 10, "helv", 72),
    ])
    _add_page(doc, [("1 Introduction", 16, "hebo", 72), *_body_rows(),
                    ("1.1 Background", 13, "hebo", 72), *_body_rows()])
    _add_page(doc, [("2 Methods", 16, "hebo", 72), *_body_rows()])
    _add_page(doc, [("3 Results", 16, "hebo", 72), *_body_rows()])
    return _save(doc, tmp_path_factory, "toc.pdf")


@pytest.fixture(scope="session")
def offset_toc_pdf(tmp_path_factory):
    """Unnumbered dotted TOC; printed page 1 is physical index 3 (offset +3)."""
    doc = fitz.open()
    _add_page(doc, [("Cover", 24, "hebo", 72)])
    _add_page(doc, [
        ("Contents", 16, "hebo", 72),
        ("Chapter One .......... 1", 10, "helv", 72),
        ("Chapter Two .......... 3", 10, "helv", 72),
        ("Chapter Three .......... 5", 10, "helv", 72),
    ])
    _add_page(doc, [("Preface", 14, "hebo", 72), *_body_rows()])
    _add_page(doc, [("Chapter One", 16, "hebo", 72), *_body_rows()])    # printed 1
    _add_page(doc, _body_rows(6))                                       # printed 2
    _add_page(doc, [("Chapter Two", 16, "hebo", 72), *_body_rows()])    # printed 3
    _add_page(doc, _body_rows(6))                                       # printed 4
    _add_page(doc, [("Chapter Three", 16, "hebo", 72), *_body_rows()])  # printed 5
    return _save(doc, tmp_path_factory, "offset.pdf")


@pytest.fixture(scope="session")
def headings_pdf(tmp_path_factory):
    """No TOC; chapter/section structure expressed via font size + bold."""
    doc = fitz.open()
    _add_page(doc, [("Chapter 1 Getting Started", 18, "hebo", 72), *_body_rows(6),
                    ("1.1 Installation", 14, "hebo", 72), *_body_rows(4)])
    _add_page(doc, [("Chapter 2 Advanced Usage", 18, "hebo", 72), *_body_rows(6),
                    ("2.1 Configuration", 14, "hebo", 72), *_body_rows(4)])
    return _save(doc, tmp_path_factory, "headings.pdf")


@pytest.fixture(scope="session")
def ghost_toc_pdf(tmp_path_factory):
    """TOC where 1 of 3 entries doesn't exist in the body → >20% location failures."""
    doc = fitz.open()
    _add_page(doc, [
        ("Contents", 16, "hebo", 72),
        ("1 Alpha .......... 2", 10, "helv", 72),
        ("2 Beta .......... 3", 10, "helv", 72),
        ("3 Ghost .......... 9", 10, "helv", 72),
    ])
    _add_page(doc, [("1 Alpha", 16, "hebo", 72), *_body_rows()])
    _add_page(doc, [("2 Beta", 16, "hebo", 72), *_body_rows()])
    return _save(doc, tmp_path_factory, "ghost.pdf")


@pytest.fixture(scope="session")
def no_text_pdf(tmp_path_factory):
    """Drawings only — simulates a scanned PDF with no text layer."""
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(72, 72, 300, 300), color=(0, 0, 0), width=2)
    return _save(doc, tmp_path_factory, "notext.pdf")


@pytest.fixture(scope="session")
def bookmarked_pdf(tmp_path_factory):
    """Already has an outline — used for --force behavior."""
    doc = fitz.open()
    _add_page(doc, [("Chapter 1 Test", 18, "hebo", 72), *_body_rows(6)])
    doc.set_toc([[1, "Existing", 1]])
    return _save(doc, tmp_path_factory, "bookmarked.pdf")


@pytest.fixture(scope="session")
def encrypted_pdf(tmp_path_factory):
    doc = fitz.open()
    _add_page(doc, [("Secret", 12, "helv", 72)])
    return _save(doc, tmp_path_factory, "encrypted.pdf",
                 encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
