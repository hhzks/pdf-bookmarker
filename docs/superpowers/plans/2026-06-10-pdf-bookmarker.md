# pdf-bookmarker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python CLI that adds a hierarchical bookmark outline to text-based PDFs by parsing the TOC (or detecting headings heuristically), locating each section in the body, with a model-agnostic LLM rescue layer.

**Architecture:** Pipeline of focused modules — extract text+font metadata (PyMuPDF) → detect/parse TOC → fall back to font-based heading detection → locate entries in the body (printed page № as hint) → write outline via `set_toc`. An `LLMBackend` protocol (Anthropic implementation included) verifies/repairs low-confidence outlines.

**Tech Stack:** Python 3.12, PyMuPDF (`fitz`), `anthropic` SDK, `pydantic`, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-pdf-bookmarker-design.md`

## File Structure

```
pyproject.toml
pdf_bookmarker/
  __init__.py
  models.py            OutlineEntry dataclass
  extractor.py         Line dataclass, extract_lines(), has_text_layer()  (OCR seam)
  toc_detector.py      find_toc_pages(), parse_toc()
  heading_detector.py  body_text_size(), detect_headings()
  locator.py           locate_entries()
  llm.py               LLMBackend protocol, AnthropicBackend, get_backend(), is_low_confidence()
  writer.py            sanitize_levels(), write_outline()
  cli.py               build_parser(), main()
tests/
  conftest.py          synthetic PDF fixtures
  test_fixtures.py … test_cli.py  (one test file per module)
```

All commands below are run from the repo root. On this machine use `python -m pytest` (Windows).

---

### Task 1: Project scaffolding + models

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `pdf_bookmarker/__init__.py`, `pdf_bookmarker/models.py`, `tests/test_models.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "pdf-bookmarker"
version = "0.1.0"
description = "Add hierarchical bookmarks to text-based PDFs"
requires-python = ">=3.12"
dependencies = ["pymupdf>=1.24", "anthropic>=0.40", "pydantic>=2.7"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
pdf-bookmarker = "pdf_bookmarker.cli:main"

[tool.setuptools.packages.find]
include = ["pdf_bookmarker*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.egg-info/
.pytest_cache/
build/
dist/
.venv/
*.bookmarked.pdf
```

- [ ] **Step 3: Create `pdf_bookmarker/__init__.py`**

```python
"""Add hierarchical bookmarks to text-based PDFs."""
```

- [ ] **Step 4: Write the failing test** — `tests/test_models.py`

```python
from pdf_bookmarker.models import OutlineEntry


def test_outline_entry_defaults():
    e = OutlineEntry(title="Intro", level=1)
    assert e.page is None
    assert e.y is None
    assert e.printed_page is None
```

- [ ] **Step 5: Install and verify the test fails**

Run: `pip install -e ".[dev]"` then `python -m pytest tests/test_models.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'pdf_bookmarker.models'`

- [ ] **Step 6: Create `pdf_bookmarker/models.py`**

```python
"""Shared data types."""
from dataclasses import dataclass


@dataclass
class OutlineEntry:
    title: str
    level: int                       # 1-based nesting depth (1 = chapter)
    page: int | None = None          # 0-based physical page index (set by locator)
    y: float | None = None           # vertical position of the heading on the page
    printed_page: int | None = None  # page number as printed in the TOC (1-based)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: 1 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore pdf_bookmarker tests
git commit -m "feat: project scaffolding and OutlineEntry model"
```

---

### Task 2: Synthetic PDF test fixtures

**Files:**
- Create: `tests/conftest.py`, `tests/test_fixtures.py`

- [ ] **Step 1: Write `tests/conftest.py`** — all fixtures are session-scoped generated PDFs

```python
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


def _save(doc, tmp_path_factory, name):
    path = tmp_path_factory.mktemp("pdfs") / name
    doc.save(str(path))
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
    path = tmp_path_factory.mktemp("pdfs") / "encrypted.pdf"
    doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256,
             user_pw="pw", owner_pw="pw")
    return path
```

- [ ] **Step 2: Write `tests/test_fixtures.py`** — sanity checks that the fixtures are what later tasks assume

```python
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
```

- [ ] **Step 3: Run the fixture tests**

Run: `python -m pytest tests/test_fixtures.py -v`
Expected: 6 passed. (If any assertion fails, fix the fixture in `conftest.py` — these page indices are relied on by every later task.)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_fixtures.py
git commit -m "test: synthetic PDF fixtures"
```

---

### Task 3: Extractor (text + font metadata)

**Files:**
- Create: `pdf_bookmarker/extractor.py`
- Test: `tests/test_extractor.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_extractor.py`

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extractor.py -v`
Expected: ERROR — `ModuleNotFoundError: No module named 'pdf_bookmarker.extractor'`

- [ ] **Step 3: Write `pdf_bookmarker/extractor.py`**

```python
"""Text extraction with font metadata. This is the seam where OCR would plug in."""
from dataclasses import dataclass

import fitz  # PyMuPDF

_BOLD_FLAG = 16  # bit 4 of the span flags


@dataclass
class Line:
    text: str
    page: int      # 0-based physical page index
    x: float       # left edge of the line
    y: float       # top of the line
    size: float    # largest font size in the line
    bold: bool


def has_text_layer(doc: fitz.Document) -> bool:
    """True if any of the first 10 pages contains extractable text."""
    for page_index in range(min(10, doc.page_count)):
        if doc[page_index].get_text("text").strip():
            return True
    return False


def extract_lines(doc: fitz.Document) -> list[Line]:
    """Flatten the document into Lines with position/size/weight metadata."""
    lines: list[Line] = []
    for page_index, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # 0 = text block
                continue
            for raw_line in block["lines"]:
                spans = [s for s in raw_line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                size = max(s["size"] for s in spans)
                bold = any(
                    s["flags"] & _BOLD_FLAG or "bold" in s["font"].lower()
                    for s in spans
                )
                x0, y0, _, _ = raw_line["bbox"]
                lines.append(
                    Line(text=text, page=page_index, x=x0, y=y0, size=size, bold=bold)
                )
    return lines
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extractor.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pdf_bookmarker/extractor.py tests/test_extractor.py
git commit -m "feat: text extraction with font metadata"
```

---

### Task 4: TOC detector

**Files:**
- Create: `pdf_bookmarker/toc_detector.py`
- Test: `tests/test_toc_detector.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_toc_detector.py`

```python
import fitz

from pdf_bookmarker.extractor import Line, extract_lines
from pdf_bookmarker.toc_detector import find_toc_pages, parse_toc


def _lines(path):
    doc = fitz.open(path)
    return extract_lines(doc), doc.page_count


def test_find_toc_pages(toc_pdf):
    lines, page_count = _lines(toc_pdf)
    assert find_toc_pages(lines, page_count) == [1]


def test_no_toc_detected(headings_pdf):
    lines, page_count = _lines(headings_pdf)
    assert find_toc_pages(lines, page_count) == []


def test_multipage_toc_continuation():
    def entry(page, i):
        return Line(f"{i} Title {i} .......... {i + 2}", page, 72, 72 + i * 18, 10, False)

    lines = [Line("Contents", 1, 72, 60, 16, True)]
    lines += [entry(1, i) for i in range(1, 5)]   # TOC page with heading
    lines += [entry(2, i) for i in range(5, 9)]   # continuation page, no heading
    lines += [Line("Body text here", 3, 72, 72, 10, False)]
    assert find_toc_pages(lines, 30) == [1, 2]


def test_parse_toc_levels_from_numbering(toc_pdf):
    lines, _ = _lines(toc_pdf)
    entries = parse_toc(lines, [1])
    assert [(e.title, e.level, e.printed_page) for e in entries] == [
        ("1 Introduction", 1, 3),
        ("1.1 Background", 2, 3),
        ("2 Methods", 1, 4),
        ("3 Results", 1, 5),
    ]


def test_parse_toc_unnumbered_uses_indentation(offset_toc_pdf):
    lines, _ = _lines(offset_toc_pdf)
    entries = parse_toc(lines, [1])
    assert [(e.title, e.level, e.printed_page) for e in entries] == [
        ("Chapter One", 1, 1),
        ("Chapter Two", 1, 3),
        ("Chapter Three", 1, 5),
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_toc_detector.py -v`
Expected: ERROR — `ModuleNotFoundError: No module named 'pdf_bookmarker.toc_detector'`

- [ ] **Step 3: Write `pdf_bookmarker/toc_detector.py`**

```python
"""Find and parse a table of contents."""
import re

from .extractor import Line
from .models import OutlineEntry

_TOC_HEADING = re.compile(r"^(table of )?contents$", re.IGNORECASE)
# "1.2 Title ...... 34" / "Title    34" — ≥2 separator chars before the page number
_TOC_LINE = re.compile(r"^(?P<title>.+?)[\s.]{2,}(?P<page>\d{1,4})$")
_NUMBERING = re.compile(r"^(?P<num>\d+(?:\.\d+)*)[.\s]\s*")


def find_toc_pages(lines: list[Line], page_count: int) -> list[int]:
    """Return the (contiguous) physical page indices that look like a TOC."""
    scan_limit = max(30, int(page_count * 0.15))
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        if line.page < scan_limit:
            by_page.setdefault(line.page, []).append(line)

    toc_pages: list[int] = []
    for page_index in sorted(by_page):
        page_lines = by_page[page_index]
        has_heading = any(_TOC_HEADING.match(l.text.strip()) for l in page_lines)
        entry_count = sum(1 for l in page_lines if _TOC_LINE.match(l.text.strip()))
        is_toc = (has_heading and entry_count >= 3) or entry_count >= 8
        if toc_pages and page_index == toc_pages[-1] + 1 and entry_count >= 3:
            is_toc = True  # continuation page of a multi-page TOC
        if is_toc:
            toc_pages.append(page_index)
        elif toc_pages:
            break  # TOC pages are contiguous; stop after the run ends
    return toc_pages


def parse_toc(lines: list[Line], toc_pages: list[int]) -> list[OutlineEntry]:
    """Parse TOC entry lines into OutlineEntries with levels and printed pages."""
    toc_page_set = set(toc_pages)
    raw: list[tuple[Line, str, int]] = []
    for line in lines:
        if line.page not in toc_page_set:
            continue
        m = _TOC_LINE.match(line.text.strip())
        if not m:
            continue
        title = m.group("title").strip(" .")
        raw.append((line, title, int(m.group("page"))))
    if not raw:
        return []

    numbered = [_level_from_numbering(title) for _, title, _ in raw]
    if all(level is not None for level in numbered):
        levels = numbered
    else:
        levels = _levels_from_indentation([line.x for line, _, _ in raw])

    return [
        OutlineEntry(title=title, level=level, printed_page=page)
        for (_, title, page), level in zip(raw, levels)
    ]


def _level_from_numbering(title: str) -> int | None:
    m = _NUMBERING.match(title)
    if m:
        return m.group("num").count(".") + 1
    return None


def _levels_from_indentation(xs: list[float]) -> list[int]:
    """Cluster left edges into tiers; deeper indentation = deeper level."""
    tiers: list[float] = []
    for x in sorted(set(xs)):
        if not tiers or x - tiers[-1] >= 5:  # merge offsets closer than 5pt
            tiers.append(x)

    def tier_of(x: float) -> int:
        for i, t in enumerate(tiers):
            if x < t + 5:
                return i
        return len(tiers) - 1

    return [tier_of(x) + 1 for x in xs]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_toc_detector.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pdf_bookmarker/toc_detector.py tests/test_toc_detector.py
git commit -m "feat: TOC page detection and entry parsing"
```

---

### Task 5: Heading detector (no-TOC fallback)

**Files:**
- Create: `pdf_bookmarker/heading_detector.py`
- Test: `tests/test_heading_detector.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_heading_detector.py`

```python
import fitz
import pytest

from pdf_bookmarker.extractor import extract_lines
from pdf_bookmarker.heading_detector import body_text_size, detect_headings


def test_body_text_size(headings_pdf):
    lines = extract_lines(fitz.open(headings_pdf))
    assert body_text_size(lines) == pytest.approx(10.0, abs=0.5)


def test_detect_headings(headings_pdf):
    lines = extract_lines(fitz.open(headings_pdf))
    entries = detect_headings(lines)
    assert [(e.title, e.level, e.page) for e in entries] == [
        ("Chapter 1 Getting Started", 1, 0),
        ("1.1 Installation", 2, 0),
        ("Chapter 2 Advanced Usage", 1, 1),
        ("2.1 Configuration", 2, 1),
    ]
    assert all(e.y is not None for e in entries)


def test_detect_headings_empty_when_uniform():
    from pdf_bookmarker.extractor import Line

    lines = [Line(f"para {i}", 0, 72, 72 + i * 14, 10, False) for i in range(20)]
    assert detect_headings(lines) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_heading_detector.py -v`
Expected: ERROR — `ModuleNotFoundError: No module named 'pdf_bookmarker.heading_detector'`

- [ ] **Step 3: Write `pdf_bookmarker/heading_detector.py`**

```python
"""Heuristic chapter/subchapter detection for PDFs with no TOC."""
import re
from collections import Counter

from .extractor import Line
from .models import OutlineEntry

_CHAPTER = re.compile(r"^(chapter|part|appendix)\s+([0-9ivxlc]+)\b", re.IGNORECASE)
_NUMBERED = re.compile(r"^(?P<num>\d+(?:\.\d+)*)[.\s]\s*\S")
_MAX_HEADING_WORDS = 12


def body_text_size(lines: list[Line]) -> float:
    """The dominant font size, weighted by amount of text."""
    weights: Counter[float] = Counter()
    for line in lines:
        weights[round(line.size, 1)] += len(line.text)
    return weights.most_common(1)[0][0]


def detect_headings(lines: list[Line]) -> list[OutlineEntry]:
    body = body_text_size(lines)
    candidates = [
        line
        for line in lines
        if len(line.text.split()) <= _MAX_HEADING_WORDS
        and not _is_page_furniture(line.text)
        and (
            line.size >= body * 1.15
            or (line.bold and (_CHAPTER.match(line.text) or _NUMBERED.match(line.text)))
        )
    ]
    if not candidates:
        return []

    # Size tiers: largest candidate size = level 1, next = level 2, ...
    sizes = sorted({round(c.size, 1) for c in candidates}, reverse=True)
    entries: list[OutlineEntry] = []
    for line in candidates:
        num = _NUMBERED.match(line.text)
        if num:
            level = num.group("num").count(".") + 1
        elif _CHAPTER.match(line.text):
            level = 1
        else:
            level = sizes.index(round(line.size, 1)) + 1
        entries.append(OutlineEntry(title=line.text, level=level, page=line.page, y=line.y))
    return entries


def _is_page_furniture(text: str) -> bool:
    return text.strip().isdigit()  # standalone printed page numbers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_heading_detector.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pdf_bookmarker/heading_detector.py tests/test_heading_detector.py
git commit -m "feat: heuristic heading detection fallback"
```

---

### Task 6: Locator (match TOC entries to physical pages)

**Files:**
- Create: `pdf_bookmarker/locator.py`
- Test: `tests/test_locator.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_locator.py`

```python
import fitz

from pdf_bookmarker.extractor import Line, extract_lines
from pdf_bookmarker.locator import locate_entries
from pdf_bookmarker.models import OutlineEntry
from pdf_bookmarker.toc_detector import parse_toc


def test_locates_exact_pages(toc_pdf):
    lines = extract_lines(fitz.open(toc_pdf))
    entries = parse_toc(lines, [1])
    located, failures = locate_entries(entries, lines, skip_pages={1})
    assert failures == 0
    assert [e.page for e in located] == [2, 2, 3, 4]
    assert all(e.y is not None for e in located)


def test_offset_correction(offset_toc_pdf):
    lines = extract_lines(fitz.open(offset_toc_pdf))
    entries = parse_toc(lines, [1])
    located, failures = locate_entries(entries, lines, skip_pages={1})
    assert failures == 0
    assert [e.page for e in located] == [3, 5, 7]


def test_unfound_entry_falls_back_to_hint():
    lines = [Line("Hello world", 0, 72, 72, 10, False)]
    entries = [OutlineEntry("Missing Chapter", 1, printed_page=1)]
    located, failures = locate_entries(entries, lines)
    assert failures == 1
    assert located[0].page == 0  # offset-corrected hint, clamped to the document
    assert located[0].y is None


def test_unfound_entry_without_hint_is_dropped():
    lines = [Line("Hello world", 0, 72, 72, 10, False)]
    entries = [OutlineEntry("Missing Chapter", 1)]
    located, failures = locate_entries(entries, lines)
    assert located == []
    assert failures == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_locator.py -v`
Expected: ERROR — `ModuleNotFoundError: No module named 'pdf_bookmarker.locator'`

- [ ] **Step 3: Write `pdf_bookmarker/locator.py`**

```python
"""Match outline entries to their physical location in the document."""
import re
from statistics import median

from .extractor import Line
from .models import OutlineEntry

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def locate_entries(
    entries: list[OutlineEntry],
    lines: list[Line],
    skip_pages: set[int] | None = None,
) -> tuple[list[OutlineEntry], int]:
    """Find each entry's title in the body. Returns (located_entries, failure_count).

    Found entries get .page and .y set. Entries with a printed-page hint that
    cannot be found keep an offset-corrected page guess (counted as a failure).
    Entries with no hint and no match are dropped (also counted as a failure).
    """
    skip_pages = skip_pages or set()
    page_count = max((l.page for l in lines), default=0) + 1
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        if line.page not in skip_pages:
            by_page.setdefault(line.page, []).append(line)

    offsets: list[int] = []  # physical_page - (printed_page - 1), from successes
    located: list[OutlineEntry] = []
    failures = 0

    for entry in entries:
        target = _normalize(entry.title)
        hint = _hint_page(entry, offsets, page_count)
        match = None
        for page_index in _pages_nearest_first(hint, page_count, skip_pages):
            for line in by_page.get(page_index, []):
                if _matches(target, _normalize(line.text)):
                    match = line
                    break
            if match:
                break
        if match:
            entry.page = match.page
            entry.y = match.y
            if entry.printed_page is not None:
                offsets.append(match.page - (entry.printed_page - 1))
            located.append(entry)
        elif entry.printed_page is not None:
            entry.page = _hint_page(entry, offsets, page_count)
            failures += 1
            located.append(entry)
        else:
            failures += 1  # no hint and no match: drop the entry
    return located, failures


def _normalize(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", text.lower())).strip()


def _matches(target: str, candidate: str) -> bool:
    if not target or not candidate:
        return False
    return (
        candidate == target
        or (len(target) >= 8 and candidate.startswith(target))
        or (len(candidate) >= 8 and target.startswith(candidate))
    )


def _hint_page(entry: OutlineEntry, offsets: list[int], page_count: int) -> int:
    if entry.printed_page is None:
        return 0
    offset = round(median(offsets)) if offsets else 0
    return min(max(entry.printed_page - 1 + offset, 0), page_count - 1)


def _pages_nearest_first(hint: int, page_count: int, skip_pages: set[int]):
    for delta in range(page_count):
        for page in dict.fromkeys((hint + delta, hint - delta)):
            if 0 <= page < page_count and page not in skip_pages:
                yield page
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_locator.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add pdf_bookmarker/locator.py tests/test_locator.py
git commit -m "feat: locate outline entries in document body"
```

---

### Task 7: Writer (apply the outline)

**Files:**
- Create: `pdf_bookmarker/writer.py`
- Test: `tests/test_writer.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_writer.py`

```python
import fitz

from pdf_bookmarker.models import OutlineEntry
from pdf_bookmarker.writer import sanitize_levels, write_outline


def test_sanitize_levels_clamps_jumps():
    entries = [OutlineEntry("A", 2), OutlineEntry("B", 4), OutlineEntry("C", 1)]
    out = sanitize_levels(entries)
    assert [e.level for e in out] == [1, 2, 1]


def test_write_outline_sets_toc(tmp_path):
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    entries = [
        OutlineEntry("One", 1, page=0, y=72.0),
        OutlineEntry("Sub", 2, page=1, y=100.0),
        OutlineEntry("Unlocated", 1, page=None),
    ]
    out = tmp_path / "out.pdf"
    count = write_outline(doc, entries, str(out))
    assert count == 2  # the unlocated entry is skipped
    toc = fitz.open(str(out)).get_toc()
    assert [item[:3] for item in toc] == [[1, "One", 1], [2, "Sub", 2]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_writer.py -v`
Expected: ERROR — `ModuleNotFoundError: No module named 'pdf_bookmarker.writer'`

- [ ] **Step 3: Write `pdf_bookmarker/writer.py`**

```python
"""Write the outline into a copy of the PDF."""
import fitz

from .models import OutlineEntry


def sanitize_levels(entries: list[OutlineEntry]) -> list[OutlineEntry]:
    """PDF outlines must start at level 1 and never jump by more than +1."""
    prev = 0
    for entry in entries:
        if entry.level > prev + 1:
            entry.level = prev + 1
        prev = entry.level
    return entries


def write_outline(doc: fitz.Document, entries: list[OutlineEntry], out_path: str) -> int:
    """Apply entries as the document outline and save to out_path.

    Returns the number of bookmarks written. Entries without a page are skipped.
    """
    entries = sanitize_levels([e for e in entries if e.page is not None])
    toc = []
    for e in entries:
        item = [e.level, e.title, e.page + 1]  # set_toc wants 1-based pages
        if e.y is not None:
            item.append({"kind": fitz.LINK_GOTO, "to": fitz.Point(0, e.y), "zoom": 0})
        toc.append(item)
    doc.set_toc(toc)
    doc.save(out_path)
    return len(toc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_writer.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pdf_bookmarker/writer.py tests/test_writer.py
git commit -m "feat: write outline via set_toc"
```

---

### Task 8: LLM layer (protocol, Anthropic backend, registry, confidence)

**Files:**
- Create: `pdf_bookmarker/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_llm.py`

```python
from types import SimpleNamespace

import pytest

from pdf_bookmarker import llm
from pdf_bookmarker.models import OutlineEntry


class FakeMessages:
    def __init__(self, captured):
        self._captured = captured

    def parse(self, **kwargs):
        self._captured.update(kwargs)
        return SimpleNamespace(
            parsed_output=llm._Outline(
                entries=[llm._OutlineItem(title="Intro", level=1, printed_page=3)]
            )
        )


def _fake_anthropic(monkeypatch, captured):
    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages(captured)

    monkeypatch.setattr("anthropic.Anthropic", FakeClient)


def test_anthropic_backend_parses_outline(monkeypatch):
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    backend = llm.AnthropicBackend(model="claude-opus-4-8")
    entries = backend.parse_outline("1 Intro .......... 3")
    assert entries == [OutlineEntry(title="Intro", level=1, printed_page=3)]
    assert captured["model"] == "claude-opus-4-8"
    assert "1 Intro" in captured["messages"][0]["content"]


def test_get_backend_passes_model_through(monkeypatch):
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    backend = llm.get_backend("anthropic:claude-haiku-4-5")
    backend.parse_outline("x")
    assert captured["model"] == "claude-haiku-4-5"


def test_get_backend_default_model(monkeypatch):
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    backend = llm.get_backend("anthropic")
    backend.parse_outline("x")
    assert captured["model"] == "claude-opus-4-8"


def test_get_backend_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm.get_backend("bogus:model-x")


@pytest.mark.parametrize(
    "detected,failures,used_toc,levels,page_count,expected",
    [
        (0, 0, False, [], 10, True),            # nothing detected
        (2, 0, True, [1, 1], 10, True),         # TOC parsed but <3 entries
        (10, 3, True, [1] * 10, 10, True),      # >20% location failures
        (4, 0, False, [1, 3, 1, 2], 10, True),  # incoherent level jump
        (5, 0, False, [1, 1, 1, 1, 1], 400, True),   # flat outline, 300+ pages
        (4, 0, True, [1, 2, 1, 1], 10, False),       # healthy TOC outline
        (4, 0, False, [1, 2, 1, 2], 50, False),      # healthy heading outline
    ],
)
def test_is_low_confidence(detected, failures, used_toc, levels, page_count, expected):
    assert llm.is_low_confidence(detected, failures, used_toc, levels, page_count) is expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm.py -v`
Expected: ERROR — `ModuleNotFoundError: No module named 'pdf_bookmarker.llm'`

- [ ] **Step 3: Write `pdf_bookmarker/llm.py`**

```python
"""Model-agnostic LLM verification layer.

To add a provider: implement the LLMBackend protocol and register the class
in _BACKENDS. Selection is via "provider:model-id" strings (e.g. --model).
"""
from typing import Protocol

from pydantic import BaseModel

from .models import OutlineEntry

DEFAULT_MODEL_SPEC = "anthropic:claude-opus-4-8"


class LLMBackend(Protocol):
    def parse_outline(self, context: str) -> list[OutlineEntry]:
        """Parse raw TOC text / heading candidates into a structured outline."""
        ...


class _OutlineItem(BaseModel):
    title: str
    level: int
    printed_page: int | None = None


class _Outline(BaseModel):
    entries: list[_OutlineItem]


_PROMPT = """The following text was extracted from a PDF. It contains either a table of
contents or a list of candidate section headings (with font metadata). Produce the
document outline: one entry per real section, in document order. `level` is the nesting
depth (1 = chapter, 2 = subchapter, ...). Set `printed_page` when a page number is shown
next to the entry. Exclude page furniture, running headers, and anything that is not a
section heading. Keep titles exactly as written (minus dot leaders and page numbers).

{context}"""


class AnthropicBackend:
    """Default backend using the official Anthropic SDK with structured output."""

    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic  # lazy import so heuristics-only runs don't need a key

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self._model = model

    def parse_outline(self, context: str) -> list[OutlineEntry]:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": _PROMPT.format(context=context)}],
            output_format=_Outline,
        )
        outline = response.parsed_output
        return [
            OutlineEntry(title=item.title, level=item.level, printed_page=item.printed_page)
            for item in outline.entries
        ]


_BACKENDS: dict[str, type] = {"anthropic": AnthropicBackend}


def get_backend(spec: str) -> LLMBackend:
    """Resolve a "provider:model-id" spec (model part optional) to a backend."""
    provider, _, model = spec.partition(":")
    if provider not in _BACKENDS:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Available: {', '.join(sorted(_BACKENDS))}"
        )
    backend_cls = _BACKENDS[provider]
    return backend_cls(model) if model else backend_cls()


def is_low_confidence(
    detected: int,
    failures: int,
    used_toc: bool,
    levels: list[int],
    page_count: int,
) -> bool:
    """Decide whether the heuristic outline needs LLM verification (auto mode)."""
    if detected == 0:
        return True
    if used_toc and detected < 3:
        return True
    if failures / detected > 0.2:
        return True
    if not used_toc:
        if any(b - a > 1 for a, b in zip(levels, levels[1:])):
            return True
        if page_count >= 300 and len(set(levels)) == 1:
            return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm.py -v`
Expected: 11 passed (4 + 7 parametrized)

- [ ] **Step 5: Commit**

```bash
git add pdf_bookmarker/llm.py tests/test_llm.py
git commit -m "feat: model-agnostic LLM layer with Anthropic backend"
```

---

### Task 9: CLI orchestration

**Files:**
- Create: `pdf_bookmarker/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_cli.py`

```python
import fitz
import pytest

from pdf_bookmarker import cli
from pdf_bookmarker.models import OutlineEntry


def test_dry_run_prints_outline(toc_pdf, capsys):
    rc = cli.main([str(toc_pdf), "--dry-run", "--no-llm"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 Introduction" in out
    assert "  1.1 Background" in out  # indented one level


def test_writes_bookmarks_from_toc(toc_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    rc = cli.main([str(toc_pdf), "-o", str(out), "--no-llm"])
    assert rc == 0
    toc = fitz.open(str(out)).get_toc()
    assert [item[:3] for item in toc] == [
        [1, "1 Introduction", 3],
        [2, "1.1 Background", 3],
        [1, "2 Methods", 4],
        [1, "3 Results", 5],
    ]


def test_writes_bookmarks_from_headings(headings_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    rc = cli.main([str(headings_pdf), "-o", str(out), "--no-llm"])
    assert rc == 0
    toc = fitz.open(str(out)).get_toc()
    assert [item[:3] for item in toc] == [
        [1, "Chapter 1 Getting Started", 1],
        [2, "1.1 Installation", 1],
        [1, "Chapter 2 Advanced Usage", 2],
        [2, "2.1 Configuration", 2],
    ]


def test_default_output_path(toc_pdf, tmp_path, monkeypatch):
    import shutil

    src = tmp_path / "book.pdf"
    shutil.copy(toc_pdf, src)
    rc = cli.main([str(src), "--no-llm"])
    assert rc == 0
    assert (tmp_path / "book.bookmarked.pdf").exists()


def test_no_text_layer_errors(no_text_pdf, capsys):
    rc = cli.main([str(no_text_pdf), "--no-llm"])
    assert rc == 2
    assert "OCR" in capsys.readouterr().err


def test_encrypted_pdf_errors(encrypted_pdf, capsys):
    rc = cli.main([str(encrypted_pdf), "--no-llm"])
    assert rc == 2
    assert "encrypted" in capsys.readouterr().err


def test_missing_file_errors(tmp_path, capsys):
    rc = cli.main([str(tmp_path / "nope.pdf"), "--no-llm"])
    assert rc == 2


def test_existing_bookmarks_require_force(bookmarked_pdf, tmp_path, capsys):
    rc = cli.main([str(bookmarked_pdf), "--no-llm"])
    assert rc == 2
    assert "--force" in capsys.readouterr().err
    out = tmp_path / "out.pdf"
    rc = cli.main([str(bookmarked_pdf), "-o", str(out), "--no-llm", "--force"])
    assert rc == 0
    toc = fitz.open(str(out)).get_toc()
    assert toc and toc[0][1] != "Existing"


def test_unknown_provider_errors(toc_pdf, capsys):
    rc = cli.main([str(toc_pdf), "--llm", "--model", "bogus:x"])
    assert rc == 2
    assert "Unknown LLM provider" in capsys.readouterr().err


def test_llm_flag_uses_backend(headings_pdf, monkeypatch, tmp_path):
    class FakeBackend:
        def parse_outline(self, context):
            return [OutlineEntry("Chapter 1 Getting Started", 1)]

    monkeypatch.setattr(cli.llm, "get_backend", lambda spec: FakeBackend())
    out = tmp_path / "out.pdf"
    rc = cli.main([str(headings_pdf), "--llm", "-o", str(out)])
    assert rc == 0
    toc = fitz.open(str(out)).get_toc()
    assert toc[0][1] == "Chapter 1 Getting Started"


def test_auto_mode_low_confidence_calls_llm(ghost_toc_pdf, monkeypatch, tmp_path):
    calls = []

    class FakeBackend:
        def parse_outline(self, context):
            calls.append(context)
            return [
                OutlineEntry("1 Alpha", 1, printed_page=2),
                OutlineEntry("2 Beta", 1, printed_page=3),
            ]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(cli.llm, "get_backend", lambda spec: FakeBackend())
    out = tmp_path / "out.pdf"
    rc = cli.main([str(ghost_toc_pdf), "-o", str(out)])
    assert rc == 0
    assert len(calls) == 1
    assert [item[1] for item in fitz.open(str(out)).get_toc()] == ["1 Alpha", "2 Beta"]


def test_auto_mode_without_key_warns_and_continues(ghost_toc_pdf, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "out.pdf"
    rc = cli.main([str(ghost_toc_pdf), "-o", str(out)])
    assert rc == 0
    assert "without LLM" in capsys.readouterr().err
    assert len(fitz.open(str(out)).get_toc()) == 3  # heuristic outline kept


def test_auto_mode_high_confidence_skips_llm(toc_pdf, monkeypatch, tmp_path):
    def boom(spec):
        raise AssertionError("LLM should not be called for a healthy outline")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(cli.llm, "get_backend", boom)
    rc = cli.main([str(toc_pdf), "-o", str(tmp_path / "out.pdf")])
    assert rc == 0


def test_llm_failure_in_auto_mode_falls_back(ghost_toc_pdf, monkeypatch, tmp_path, capsys):
    class FailingBackend:
        def parse_outline(self, context):
            raise RuntimeError("api down")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(cli.llm, "get_backend", lambda spec: FailingBackend())
    out = tmp_path / "out.pdf"
    rc = cli.main([str(ghost_toc_pdf), "-o", str(out)])
    assert rc == 0
    assert "warning" in capsys.readouterr().err.lower()
    assert len(fitz.open(str(out)).get_toc()) == 3


def test_llm_failure_with_llm_flag_errors(ghost_toc_pdf, monkeypatch, tmp_path, capsys):
    class FailingBackend:
        def parse_outline(self, context):
            raise RuntimeError("api down")

    monkeypatch.setattr(cli.llm, "get_backend", lambda spec: FailingBackend())
    rc = cli.main([str(ghost_toc_pdf), "--llm", "-o", str(tmp_path / "out.pdf")])
    assert rc == 1
    assert "LLM verification failed" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -v`
Expected: ERROR — `ModuleNotFoundError: No module named 'pdf_bookmarker.cli'`

- [ ] **Step 3: Write `pdf_bookmarker/cli.py`**

```python
"""Command-line entry point and pipeline orchestration."""
import argparse
import os
import sys
from pathlib import Path

import fitz

from . import extractor, heading_detector, llm, locator, toc_detector, writer
from .extractor import Line
from .models import OutlineEntry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-bookmarker",
        description="Add a hierarchical bookmark outline to a text-based PDF.",
    )
    parser.add_argument("input", type=Path, help="input PDF")
    parser.add_argument("-o", "--output", type=Path,
                        help="output path (default: <input>.bookmarked.pdf)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--llm", action="store_true",
                      help="always verify the outline with the LLM")
    mode.add_argument("--no-llm", action="store_true", help="never call the LLM")
    parser.add_argument("--model", default=llm.DEFAULT_MODEL_SPEC,
                        help="LLM backend as PROVIDER:MODEL_ID (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the detected outline without writing")
    parser.add_argument("--force", action="store_true",
                        help="replace existing bookmarks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        doc = fitz.open(args.input)
    except Exception as exc:
        print(f"error: cannot open {args.input}: {exc}", file=sys.stderr)
        return 2
    if doc.needs_pass:
        print("error: PDF is encrypted", file=sys.stderr)
        return 2
    if doc.get_toc() and not args.force:
        print("error: PDF already has bookmarks; use --force to replace them",
              file=sys.stderr)
        return 2
    if not extractor.has_text_layer(doc):
        print("error: no extractable text layer (scanned PDF? OCR is not supported yet)",
              file=sys.stderr)
        return 2

    lines = extractor.extract_lines(doc)
    entries, failures, used_toc, toc_pages = build_outline(lines, doc.page_count)

    if decide_llm(args, entries, failures, used_toc, doc.page_count):
        try:
            backend = llm.get_backend(args.model)
            context = build_llm_context(lines, toc_pages)
            llm_entries = backend.parse_outline(context)
            entries, failures = locator.locate_entries(
                llm_entries, lines, skip_pages=set(toc_pages)
            )
        except ValueError as exc:  # unknown provider
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            if args.llm:
                print(f"error: LLM verification failed: {exc}", file=sys.stderr)
                return 1
            print(f"warning: LLM call failed ({exc}); using heuristic outline",
                  file=sys.stderr)

    if not entries:
        print("error: no outline could be detected", file=sys.stderr)
        return 1

    if args.dry_run:
        print_outline(entries)
        return 0

    out_path = args.output or args.input.with_suffix(".bookmarked.pdf")
    count = writer.write_outline(doc, entries, str(out_path))
    print(f"wrote {count} bookmarks to {out_path}")
    return 0


def build_outline(
    lines: list[Line], page_count: int
) -> tuple[list[OutlineEntry], int, bool, list[int]]:
    """Run TOC detection with heading-detection fallback.

    Returns (entries, location_failures, used_toc, toc_pages).
    """
    toc_pages = toc_detector.find_toc_pages(lines, page_count)
    entries = toc_detector.parse_toc(lines, toc_pages) if toc_pages else []
    if entries:
        located, failures = locator.locate_entries(
            entries, lines, skip_pages=set(toc_pages)
        )
        return located, failures, True, toc_pages
    # Fallback: headings already carry page/y, no location step needed.
    return heading_detector.detect_headings(lines), 0, False, toc_pages


def decide_llm(
    args: argparse.Namespace,
    entries: list[OutlineEntry],
    failures: int,
    used_toc: bool,
    page_count: int,
) -> bool:
    if args.no_llm:
        return False
    if args.llm:
        return True
    levels = [e.level for e in entries]
    if not llm.is_low_confidence(len(entries), failures, used_toc, levels, page_count):
        return False
    if args.model.startswith("anthropic") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("warning: outline confidence is low but ANTHROPIC_API_KEY is not set; "
              "continuing without LLM", file=sys.stderr)
        return False
    return True


def build_llm_context(lines: list[Line], toc_pages: list[int]) -> str:
    if toc_pages:
        toc_page_set = set(toc_pages)
        toc_text = "\n".join(l.text for l in lines if l.page in toc_page_set)
        return f"Table of contents text:\n{toc_text}"
    body = heading_detector.body_text_size(lines)
    candidates = [
        f"physical_page={l.page} size={l.size:.1f} bold={l.bold} text={l.text!r}"
        for l in lines
        if l.size >= body * 1.1 or l.bold
    ]
    return (
        f"Candidate heading lines (body text size {body:.1f}; physical_page is "
        f"0-based, not a printed page number):\n" + "\n".join(candidates[:400])
    )


def print_outline(entries: list[OutlineEntry]) -> None:
    for e in entries:
        page = "?" if e.page is None else e.page + 1
        print(f"{'  ' * (e.level - 1)}{e.title}  (page {page})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add pdf_bookmarker/cli.py tests/test_cli.py
git commit -m "feat: CLI orchestration with auto/forced/disabled LLM modes"
```

---

### Task 10: README, full-suite verification, manual smoke test

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -v`
Expected: all tests pass (≈49). Fix any failures before continuing.

- [ ] **Step 2: Manual smoke test against a generated fixture**

```bash
python -c "import sys; sys.path.insert(0, 'tests'); import fitz; from conftest import _add_page, _body_rows; doc = fitz.open(); _add_page(doc, [('My Book', 24, 'hebo', 72)]); _add_page(doc, [('Contents', 16, 'hebo', 72), ('1 Introduction .......... 3', 10, 'helv', 72), ('1.1 Background .......... 3', 10, 'helv', 90), ('2 Methods .......... 4', 10, 'helv', 72), ('3 Results .......... 5', 10, 'helv', 72)]); _add_page(doc, [('1 Introduction', 16, 'hebo', 72), *_body_rows(), ('1.1 Background', 13, 'hebo', 72)]); _add_page(doc, [('2 Methods', 16, 'hebo', 72)]); _add_page(doc, [('3 Results', 16, 'hebo', 72)]); doc.save('smoke.pdf')"
pdf-bookmarker smoke.pdf --dry-run --no-llm
pdf-bookmarker smoke.pdf --no-llm
```

Expected: dry-run prints the 4-entry indented outline; second command prints `wrote 4 bookmarks to smoke.bookmarked.pdf`. Open `smoke.bookmarked.pdf` in a viewer if available and confirm the bookmarks jump to the right headings. Then delete `smoke.pdf` and `smoke.bookmarked.pdf`.

- [ ] **Step 3: Write `README.md`**

```markdown
# pdf-bookmarker

Add a hierarchical bookmark outline to text-based PDFs. Parses the table of
contents when one exists (preserving chapter/subchapter structure and linking
each bookmark to the section's real location), and falls back to font-based
heading detection when there is no TOC. An optional LLM pass verifies or
repairs low-confidence outlines.

## Install

    pip install -e .

## Usage

    pdf-bookmarker input.pdf                 # writes input.bookmarked.pdf
    pdf-bookmarker input.pdf -o out.pdf      # explicit output
    pdf-bookmarker input.pdf --dry-run       # print outline, write nothing
    pdf-bookmarker input.pdf --force         # replace existing bookmarks
    pdf-bookmarker input.pdf --llm           # always verify with the LLM
    pdf-bookmarker input.pdf --no-llm        # never call the LLM

By default the LLM is only consulted when the heuristic outline looks
unreliable (auto mode). Set `ANTHROPIC_API_KEY` to enable it; without a key,
auto mode warns and continues heuristics-only.

### Choosing a model

    pdf-bookmarker input.pdf --model anthropic:claude-opus-4-8

The LLM layer is provider-agnostic: implement `pdf_bookmarker.llm.LLMBackend`
and register the class in `_BACKENDS` to add another provider.

## Limitations

- Text-based PDFs only. Scanned PDFs (no text layer) are rejected; OCR is a
  planned extension (`extractor.py` is the seam).
- Encrypted PDFs are not supported.

## Development

    pip install -e ".[dev]"
    python -m pytest
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README with usage and provider extension notes"
```

---

## Self-Review Notes (already applied)

- **Spec coverage:** CLI flags (Task 9), TOC detection incl. multi-page (Task 4), indentation + numbering levels (Task 4), printed→physical offset + y-position (Task 6), heading fallback (Task 5), confidence thresholds exactly as spec'd (Task 8), provider registry + `--model` (Task 8/9), all error-handling rows from the spec table (Tasks 9 tests), fixtures 1–5 from spec testing section (Task 2; the "already bookmarked" and "encrypted" cases included).
- **Type consistency:** `Line(text, page, x, y, size, bold)`; `OutlineEntry(title, level, page, y, printed_page)`; `locate_entries(entries, lines, skip_pages) -> (list, int)`; `is_low_confidence(detected, failures, used_toc, levels, page_count)` — verified consistent across all tasks.
- **Known judgment calls:** locator matches by normalized prefix (≥8 chars) to tolerate subtitle suffixes; `set_toc` dest dict uses `{"kind": fitz.LINK_GOTO, "to": fitz.Point(0, y)}` (PyMuPDF accepts a 4th item per TOC entry). If a PyMuPDF version quirk surfaces on these, the writer tests in Task 7 will catch it first.
