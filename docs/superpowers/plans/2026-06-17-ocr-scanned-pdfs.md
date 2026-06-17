# OCR for Scanned PDFs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recognise text in scanned (no-text-layer) PDFs via Tesseract OCR so the existing outline-detection pipeline can bookmark them.

**Architecture:** OCR plugs into the documented extractor seam. PyMuPDF's `page.get_textpage_ocr()` yields a TextPage that `page.get_text("dict", textpage=tp)` reads like a born-digital page, producing the same `Line` objects. A new `ocr.py` owns the Tesseract dependency; `extractor.py` is refactored so both paths share one line-building helper. `pipeline.process_pdf` gains `ocr_mode`/`ocr_max_pages`; the web bounds OCR with an env-configurable page cap.

**Tech Stack:** PyMuPDF (`get_textpage_ocr`), Tesseract system binary (no new Python dependency), FastAPI backend, React frontend, pytest.

## Global Constraints

- OCR engine: PyMuPDF `page.get_textpage_ocr(language="eng", dpi=DPI, full=True)`; `DPI = 300`. English only — `language` is a constant, not a parameter.
- **No new Python dependency** — `pyproject.toml` `dependencies` unchanged. Tesseract is a system binary.
- `ocr.available() -> bool` returns `bool(shutil.which("tesseract"))`.
- `pipeline.process_pdf` new params: `ocr_mode: str = "auto"` (`"auto" | "force" | "never"`), `ocr_max_pages: int | None = None` (None = no cap). `PipelineResult` gains `used_ocr: bool`.
- New typed errors (subclasses of `pipeline.PipelineError`): `OcrUnavailableError`, `OcrPageLimitError`.
- OCR branch rule: run OCR when `ocr_mode == "force"` OR (`ocr_mode == "auto"` AND no text layer). `never` + no text layer → `NoTextLayerError` (today's behavior).
- The `NoTextLayerError` message string must contain the substring `OCR` (existing tests match it). The web friendly string for `NoTextLayerError` must contain the substring `scanned` (existing tests match it).
- Web: `OCR_MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "50"))`; the web always uses `ocr_mode="auto"` and passes `ocr_max_pages=OCR_MAX_PAGES`. No new upload form field.
- Detection only — do NOT embed a searchable text layer in the output PDF.
- The baseline-merging logic must live in ONE place, shared by `extractor` and `ocr` (no duplication).
- No AI attribution in commits (no `Co-Authored-By` trailer).

---

### Task 1: OCR module + shared extractor helper

**Files:**
- Modify: `pdf_bookmarker/extractor.py` (refactor `extract_lines`, lines 41-69)
- Create: `pdf_bookmarker/ocr.py`
- Modify: `tests/conftest.py` (add a fixture)
- Test: `tests/test_ocr.py` (new)

**Interfaces:**
- Produces: `extractor.lines_from_blocks(blocks: list[dict], page_index: int) -> list[Line]`; `ocr.available() -> bool`; `ocr.extract_lines_via_ocr(doc: fitz.Document) -> list[Line]`; `ocr.DPI = 300`; pytest fixture `scanned_text_pdf` (path to an image-only PDF with no text layer).

- [ ] **Step 1: Refactor the per-page loop out of `extract_lines`**

In `pdf_bookmarker/extractor.py`, replace the body of `extract_lines` (lines 41-69) with a thin loop that delegates to a new public helper:

```python
def extract_lines(doc: fitz.Document) -> list[Line]:
    """Flatten the document into visual lines with position/size/weight metadata.

    PyMuPDF splits one visual line into several "line" dicts wherever the text
    jumps horizontally (LaTeX heading numbers, TOC page numbers), so fragments
    sharing a baseline are merged back into a single Line here.
    """
    lines: list[Line] = []
    for page_index, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        lines.extend(lines_from_blocks(blocks, page_index))
    return lines


def lines_from_blocks(blocks: list[dict], page_index: int) -> list[Line]:
    """Merge a page's text blocks (from get_text("dict")) into visual lines.

    Shared by the born-digital path (extract_lines) and the OCR path
    (ocr.extract_lines_via_ocr), which supplies OCR-derived blocks.
    """
    fragments: list[_Fragment] = []
    for block in blocks:
        if block.get("type") != 0:  # 0 = text block
            continue
        for raw_line in block["lines"]:
            fragment = _parse_fragment(raw_line)
            if fragment:
                fragments.append(fragment)
    lines: list[Line] = []
    for group in _baseline_groups(fragments):
        lines.append(
            Line(
                text=_join_fragments(group),
                page=page_index,
                x=group[0].x0,
                y=min(f.y for f in group),
                size=max(f.size for f in group),
                bold=any(f.bold for f in group),
            )
        )
    return lines
```

- [ ] **Step 2: Verify the refactor didn't change behavior**

Run: `python -m pytest tests/test_extractor.py -v`
Expected: PASS (the existing extractor tests still pass — this is a pure refactor).

- [ ] **Step 3: Commit the refactor**

```bash
git add pdf_bookmarker/extractor.py
git commit -m "refactor: extract shared lines_from_blocks helper"
```

- [ ] **Step 4: Add the `scanned_text_pdf` fixture**

In `tests/conftest.py`, add after the `no_text_pdf` fixture (around line 139):

```python
@pytest.fixture(scope="session")
def scanned_text_pdf(tmp_path_factory):
    """Real headings rendered, then rasterized to an image-only page.

    The result has NO text layer (has_text_layer is False) but is visually
    legible, so OCR can recover the text.
    """
    src = fitz.open()
    page = src.new_page()
    page.insert_text((72, 120), "Chapter 1 Introduction", fontsize=26, fontname="hebo")
    page.insert_text((72, 220), "Chapter 2 Methods", fontsize=26, fontname="hebo")
    pix = page.get_pixmap(dpi=150)
    src.close()
    out = fitz.open()
    img_page = out.new_page()
    img_page.insert_image(img_page.rect, pixmap=pix)
    return _save(out, tmp_path_factory, "scanned.pdf")
```

- [ ] **Step 5: Write the failing OCR tests**

Create `tests/test_ocr.py`:

```python
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
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ocr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_bookmarker.ocr'` (module not created yet).

- [ ] **Step 7: Create the OCR module**

Create `pdf_bookmarker/ocr.py`:

```python
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
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ocr.py -v`
Expected: the `available()` tests and the no-text-layer sanity test PASS; `test_extract_lines_via_ocr_recovers_text` PASSES if Tesseract is installed, otherwise SKIPS. No failures.

- [ ] **Step 9: Commit**

```bash
git add pdf_bookmarker/ocr.py tests/test_ocr.py tests/conftest.py
git commit -m "feat: OCR module recovering text from scanned pages"
```

---

### Task 2: Pipeline integration (ocr_mode, page cap, errors)

**Files:**
- Modify: `pdf_bookmarker/pipeline.py` (imports line 8; errors near lines 25-30; `PipelineResult` lines 48-54; `process_pdf` lines 57-122)
- Test: `tests/test_pipeline.py` (add tests; update one existing test)

**Interfaces:**
- Consumes: `ocr.available()`, `ocr.extract_lines_via_ocr(doc)` from Task 1.
- Produces: `pipeline.OcrUnavailableError`, `pipeline.OcrPageLimitError`; `process_pdf(..., ocr_mode="auto", ocr_max_pages=None)`; `PipelineResult.used_ocr: bool`.

- [ ] **Step 1: Write the failing pipeline tests**

Add to `tests/test_pipeline.py` (note the new import of `Line` and `OutlineEntry` is already imported):

```python
from pdf_bookmarker.extractor import Line


def _fixed_outline(monkeypatch):
    """Bypass detection so OCR-decision tests are deterministic."""
    monkeypatch.setattr(
        pipeline, "build_outline",
        lambda lines, page_count: ([OutlineEntry("Chapter 1", 1, page=0, y=100.0)], 0, False, []),
    )


def test_auto_ocr_runs_when_no_text_layer(no_text_pdf, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(pipeline.ocr, "available", lambda: True)
    monkeypatch.setattr(
        pipeline.ocr, "extract_lines_via_ocr",
        lambda doc: called.append(True) or [Line("Chapter 1", 0, 72, 100, 24, True)],
    )
    _fixed_outline(monkeypatch)
    result = pipeline.process_pdf(no_text_pdf, tmp_path / "o.pdf", llm_mode="never")
    assert called == [True]
    assert result.used_ocr is True
    assert result.bookmark_count == 1


def test_auto_ocr_unavailable_raises(no_text_pdf, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.ocr, "available", lambda: False)
    with pytest.raises(pipeline.OcrUnavailableError):
        pipeline.process_pdf(no_text_pdf, tmp_path / "o.pdf", llm_mode="never")


def test_never_ocr_no_text_raises(no_text_pdf, tmp_path):
    with pytest.raises(pipeline.NoTextLayerError, match="OCR"):
        pipeline.process_pdf(
            no_text_pdf, tmp_path / "o.pdf", llm_mode="never", ocr_mode="never"
        )


def test_force_ocr_overrides_text_layer(toc_pdf, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(pipeline.ocr, "available", lambda: True)
    monkeypatch.setattr(
        pipeline.ocr, "extract_lines_via_ocr",
        lambda doc: called.append(True) or [Line("Chapter 1", 0, 72, 100, 24, True)],
    )
    _fixed_outline(monkeypatch)
    result = pipeline.process_pdf(
        toc_pdf, tmp_path / "o.pdf", llm_mode="never", ocr_mode="force"
    )
    assert called == [True]  # OCR ran even though toc_pdf has a text layer
    assert result.used_ocr is True


def test_ocr_page_cap_raises(no_text_pdf, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.ocr, "available", lambda: True)
    with pytest.raises(pipeline.OcrPageLimitError):
        pipeline.process_pdf(
            no_text_pdf, tmp_path / "o.pdf", llm_mode="never", ocr_max_pages=0
        )


def test_invalid_ocr_mode_raises(toc_pdf, tmp_path):
    with pytest.raises(ValueError, match="ocr_mode"):
        pipeline.process_pdf(
            toc_pdf, tmp_path / "o.pdf", llm_mode="never", ocr_mode="sometimes"
        )


def test_born_digital_pdf_does_not_use_ocr(toc_pdf, tmp_path):
    result = pipeline.process_pdf(toc_pdf, tmp_path / "o.pdf", llm_mode="never")
    assert result.used_ocr is False
```

Also UPDATE the existing `test_no_text_layer_raises` (currently lines 43-45) to pin `ocr_mode="never"`, because the new default (`auto`) would route it through OCR and make it environment-dependent:

```python
def test_no_text_layer_raises(no_text_pdf, tmp_path):
    with pytest.raises(pipeline.NoTextLayerError, match="OCR"):
        pipeline.process_pdf(
            no_text_pdf, tmp_path / "o.pdf", llm_mode="never", ocr_mode="never"
        )
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -k "ocr or used_ocr" -v`
Expected: FAIL (e.g. `AttributeError: module 'pdf_bookmarker.pipeline' has no attribute 'ocr'` / `OcrUnavailableError`, and `process_pdf` rejects unknown `ocr_mode`/`ocr_max_pages` kwargs).

- [ ] **Step 3: Add the import, errors, and result field**

In `pdf_bookmarker/pipeline.py`, line 8, add `ocr` to the package import:

```python
from . import extractor, heading_detector, llm, locator, ocr, toc_detector, writer
```

Reword `_NO_TEXT_MESSAGE` (line 29) so it still contains `OCR` but no longer claims OCR is unsupported:

```python
_NO_TEXT_MESSAGE = "no extractable text layer (scanned PDF; enable OCR to read it)"
```

Add two new error classes near the other error subclasses (e.g. after `NoTextLayerError`, around line 27):

```python
class OcrUnavailableError(PipelineError):
    """OCR was needed or requested but the Tesseract binary is not available."""


class OcrPageLimitError(PipelineError):
    """The document has more pages than the configured OCR page cap."""
```

Add `used_ocr` to `PipelineResult` (after `used_toc`, line 52):

```python
@dataclass
class PipelineResult:
    entries: list[OutlineEntry]
    bookmark_count: int  # 0 when output_path is None (dry run)
    used_llm: bool
    used_toc: bool
    used_ocr: bool = False
    warnings: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Add the params and OCR branch to `process_pdf`**

Change the signature (lines 57-65) to add the two params:

```python
def process_pdf(
    input_path: Path | str,
    output_path: Path | str | None,
    *,
    llm_mode: str = "auto",  # "auto" | "always" | "never"
    model_spec: str = llm.DEFAULT_MODEL_SPEC,
    api_key: str | None = None,
    replace_existing: bool = True,
    ocr_mode: str = "auto",  # "auto" | "force" | "never"
    ocr_max_pages: int | None = None,
) -> PipelineResult:
```

After the existing `llm_mode` validation (line 71-72), add `ocr_mode` validation:

```python
    if ocr_mode not in ("auto", "force", "never"):
        raise ValueError(f"ocr_mode must be auto, force or never, not {ocr_mode!r}")
```

Replace the current text-layer guard + extraction (lines 83-88):

```python
        if not extractor.has_text_layer(doc):
            raise NoTextLayerError(_NO_TEXT_MESSAGE)

        lines = extractor.extract_lines(doc)
        if not lines:
            raise NoTextLayerError(_NO_TEXT_MESSAGE)
```

with the OCR-aware version:

```python
        has_text = extractor.has_text_layer(doc)
        use_ocr = ocr_mode == "force" or (ocr_mode == "auto" and not has_text)
        if use_ocr:
            if not ocr.available():
                raise OcrUnavailableError(
                    "OCR is required to read this PDF but Tesseract is not available"
                )
            if ocr_max_pages is not None and doc.page_count > ocr_max_pages:
                raise OcrPageLimitError(
                    f"document has {doc.page_count} pages; OCR limit is {ocr_max_pages}"
                )
            lines = ocr.extract_lines_via_ocr(doc)
            if not lines:
                raise NoTextLayerError("OCR found no readable text in this scanned PDF")
            used_ocr = True
        else:
            if not has_text:
                raise NoTextLayerError(_NO_TEXT_MESSAGE)
            lines = extractor.extract_lines(doc)
            if not lines:
                raise NoTextLayerError(_NO_TEXT_MESSAGE)
            used_ocr = False
```

Thread `used_ocr` into the returned result. Change the final `return` (line 119) to include it:

```python
        return PipelineResult(entries, count, used_llm, used_toc, used_ocr, warnings)
```

- [ ] **Step 5: Run the pipeline tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: all PASS (new OCR tests + updated `test_no_text_layer_raises` + all pre-existing tests, since `used_ocr` defaults to False and born-digital flow is unchanged).

- [ ] **Step 6: Commit**

```bash
git add pdf_bookmarker/pipeline.py tests/test_pipeline.py
git commit -m "feat: OCR-aware pipeline with ocr_mode and page cap"
```

---

### Task 3: CLI `--ocr` flag

**Files:**
- Modify: `pdf_bookmarker/cli.py` (parser, lines 18-28; `main`, lines 31-45)
- Test: `tests/test_cli.py` (update one existing test; add two)

**Interfaces:**
- Consumes: `pipeline.process_pdf(..., ocr_mode=...)`, `pipeline.ocr` (Task 2).
- Produces: `--ocr {auto,force,never}` CLI option (default `auto`).

- [ ] **Step 1: Write/Update the failing CLI tests**

UPDATE the existing `test_no_text_layer_errors` (lines 80-83) to pin `--ocr never` so it deterministically tests the OCR-disabled path regardless of whether Tesseract is installed on the test machine:

```python
def test_no_text_layer_errors(no_text_pdf, capsys):
    rc = cli.main([str(no_text_pdf), "--no-llm", "--ocr", "never"])
    assert rc == 2
    assert "OCR" in capsys.readouterr().err
```

ADD two tests:

```python
def test_ocr_flag_defaults_to_auto(toc_pdf, tmp_path):
    # A born-digital PDF: --ocr defaults to auto and never touches OCR.
    rc = cli.main([str(toc_pdf), "-o", str(tmp_path / "out.pdf"), "--no-llm"])
    assert rc == 0


def test_force_ocr_invokes_ocr(toc_pdf, tmp_path, monkeypatch):
    from pdf_bookmarker.extractor import Line
    called = []
    monkeypatch.setattr(cli.pipeline.ocr, "available", lambda: True)
    monkeypatch.setattr(
        cli.pipeline.ocr, "extract_lines_via_ocr",
        lambda doc: called.append(True) or [Line("Chapter 1 X", 0, 72, 100, 24, True)],
    )
    rc = cli.main([str(toc_pdf), "-o", str(tmp_path / "out.pdf"), "--no-llm", "--ocr", "force"])
    assert rc == 0
    assert called == [True]  # OCR ran despite the text layer
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -k "ocr" -v`
Expected: FAIL — `--ocr` is an unrecognized argument (argparse exits with code 2 / SystemExit), so the new tests fail.

- [ ] **Step 3: Add the `--ocr` argument and wire it through**

In `pdf_bookmarker/cli.py`, add to `build_parser` (after the `--model` argument, around line 23):

```python
    parser.add_argument("--ocr", choices=("auto", "force", "never"), default="auto",
                        help="OCR scanned PDFs: auto (when no text layer), force, "
                             "or never (default: %(default)s)")
```

In `main`, pass it to `process_pdf` (add to the call at lines 39-45):

```python
        result = pipeline.process_pdf(
            args.input,
            out_path,
            llm_mode=llm_mode,
            model_spec=args.model,
            replace_existing=args.force,
            ocr_mode=args.ocr,
        )
```

- [ ] **Step 4: Run the CLI tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -v`
Expected: all PASS (updated no-text test + two new tests + all pre-existing).

- [ ] **Step 5: Commit**

```bash
git add pdf_bookmarker/cli.py tests/test_cli.py
git commit -m "feat: --ocr flag on the CLI"
```

---

### Task 4: Web backend OCR pass-through + page cap

**Files:**
- Modify: `backend/app/routes.py` (imports/config lines 1-12; `create_job` lines 24-56)
- Modify: `backend/app/jobs.py` (`friendly_error` list lines 34-47; `submit` lines 64-82; `_run` lines 115-131)
- Test: `tests/test_webapp_api.py` (add tests)

**Interfaces:**
- Consumes: `pipeline.OcrUnavailableError`, `pipeline.OcrPageLimitError`, `process_pdf(..., ocr_mode=, ocr_max_pages=)` (Task 2).
- Produces: `routes.OCR_MAX_PAGES`; `JobStore.submit(..., ocr_mode="auto", ocr_max_pages=None)` with those defaults.

- [ ] **Step 1: Write the failing web tests**

Add to `tests/test_webapp_api.py`:

```python
def test_ocr_options_reach_pipeline(client, fake_pipeline):
    from app import routes
    upload(client)
    call = _first_call(fake_pipeline)  # helper added in the model-choice work
    assert call["ocr_mode"] == "auto"
    assert call["ocr_max_pages"] == routes.OCR_MAX_PAGES


def test_page_limit_reports_friendly_error(monkeypatch):
    from pdf_bookmarker.pipeline import OcrPageLimitError

    def boom(input_path, output_path, **kwargs):
        raise OcrPageLimitError("too long")

    monkeypatch.setattr(jobs_module, "process_pdf", boom)
    with TestClient(create_app(rate_limit_per_hour=1000)) as client:
        job_id = upload(client).json()["job_id"]
        body = poll_until_finished(client, job_id)
        assert body["status"] == "failed"
        assert "too long" in body["error"].lower()


def test_ocr_unavailable_reports_friendly_error(monkeypatch):
    from pdf_bookmarker.pipeline import OcrUnavailableError

    def boom(input_path, output_path, **kwargs):
        raise OcrUnavailableError("no tesseract")

    monkeypatch.setattr(jobs_module, "process_pdf", boom)
    with TestClient(create_app(rate_limit_per_hour=1000)) as client:
        job_id = upload(client).json()["job_id"]
        body = poll_until_finished(client, job_id)
        assert body["status"] == "failed"
        assert "scanned" in body["error"].lower()
```

> Note: `_first_call`, `upload`, `poll_until_finished`, `jobs_module`, and `create_app` already exist in this test file. If `_first_call` is absent, add `def _first_call(fake_pipeline, timeout=10): ...` busy-wait returning `fake_pipeline[0]`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_webapp_api.py -k "ocr or page_limit" -v`
Expected: FAIL — `routes.OCR_MAX_PAGES` doesn't exist; `ocr_mode`/`ocr_max_pages` not in the recorded kwargs; the two new error types map to the generic "Processing failed unexpectedly." string (so the `too long`/`scanned` assertions fail).

- [ ] **Step 3: Add config + pass-through in `routes.py`**

At the top of `backend/app/routes.py`, add the `os` import and the config constant (the file currently imports `DEFAULT_MODEL_SPEC`-free after the model-choice work; keep `SERVER_MODEL_SPEC` as-is):

```python
import os
```

Add near `SERVER_MODEL_SPEC`:

```python
# Bound OCR cost on the free tier: scanned PDFs longer than this are rejected.
OCR_MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "50"))
```

In `create_job`, pass the OCR options to `store.submit(...)`:

```python
    job = store.submit(
        bytes(data),
        file.filename or "document.pdf",
        llm_mode=llm_mode,
        model_spec=model_spec,
        api_key=api_key or None,
        ocr_mode="auto",
        ocr_max_pages=OCR_MAX_PAGES,
    )
```

- [ ] **Step 4: Add params + friendly errors in `jobs.py`**

In `backend/app/jobs.py`, extend `submit` (keep the new params keyword-only with defaults so existing direct callers in `tests/test_webapp_jobs.py` keep working):

```python
    def submit(
        self,
        pdf_bytes: bytes,
        original_name: str,
        *,
        llm_mode: str,
        model_spec: str,
        api_key: str | None,
        ocr_mode: str = "auto",
        ocr_max_pages: int | None = None,
    ) -> Job:
```

Change the pool dispatch line to forward them:

```python
        self._pool.submit(
            self._run, job, llm_mode, model_spec, api_key, ocr_mode, ocr_max_pages
        )
```

Update `_run` to accept and forward them:

```python
    def _run(self, job: Job, llm_mode: str, model_spec: str,
             api_key: str | None, ocr_mode: str = "auto",
             ocr_max_pages: int | None = None) -> None:
        job.status = "processing"
        try:
            result = process_pdf(
                job.input_path, job.output_path,
                llm_mode=llm_mode, model_spec=model_spec, api_key=api_key,
                ocr_mode=ocr_mode, ocr_max_pages=ocr_max_pages,
            )
        except Exception as exc:
            job.error = friendly_error(exc)
            job.status = "failed"
        else:
            job.bookmark_count = result.bookmark_count
            job.status = "done"
```

Add entries to the `_FRIENDLY` list (after the `LLMVerificationError` entry, around line 45) and reword the `NoTextLayerError` message so it still contains `scanned`:

```python
    (pipeline.NoTextLayerError,
     "This PDF appears to be scanned and no text could be read from it, even "
     "after OCR."),
    (pipeline.OcrPageLimitError,
     "This scanned PDF is too long to process (the OCR page limit is set by "
     "the server). Try a shorter document."),
    (pipeline.OcrUnavailableError,
     "This server can't process scanned PDFs right now."),
```

(Replace the existing `NoTextLayerError` tuple with the reworded one; add the two new tuples.)

- [ ] **Step 5: Run the web tests to verify they pass**

Run: `python -m pytest tests/test_webapp_api.py tests/test_webapp_jobs.py -v`
Expected: all PASS (new OCR tests, plus the pre-existing `test_failed_job_*` tests which still find `scanned` in the reworded `NoTextLayerError` string).

- [ ] **Step 6: Commit**

```bash
git add backend/app/routes.py backend/app/jobs.py tests/test_webapp_api.py
git commit -m "feat: web OCR pass-through with page cap and friendly errors"
```

---

### Task 5: Deployment + docs + frontend note

**Files:**
- Modify: `backend/Dockerfile`
- Modify: `render.yaml`
- Modify: `README.md`
- Modify: `frontend/src/App.jsx` (static note under the dropzone)

**Interfaces:**
- Consumes: the `OCR_MAX_PAGES` env-var contract (Task 4) and the Tesseract runtime requirement (Task 1).
- Produces: documentation + deployment config only (plus one static UI string).

- [ ] **Step 1: Install Tesseract in the Docker image**

In `backend/Dockerfile`, between the `pip install` line (line 7) and `COPY backend/app ./app` (line 9), add the Tesseract install and tessdata path. The exact `TESSDATA_PREFIX` depends on the base image's Tesseract version — confirm it by checking where `tesseract-ocr-eng` installs `eng.traineddata` (Debian bookworm, which `python:3.12-slim` is based on, ships Tesseract 5 with data under `/usr/share/tesseract-ocr/5/tessdata`):

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
```

- [ ] **Step 2: Add OCR_MAX_PAGES to render.yaml**

In `render.yaml`, append to the `envVars` list (matching the existing indentation, after the model-choice work's `VERIFICATION_MODEL` entry):

```yaml
      - key: OCR_MAX_PAGES
        sync: false   # optional; defaults to 50
```

- [ ] **Step 3: Document OCR in the README**

In `README.md`, add a short "Scanned PDFs (OCR)" subsection covering: scanned/no-text PDFs are auto-OCR'd; the CLI needs the `tesseract` binary installed locally (e.g. `apt install tesseract-ocr`, `brew install tesseract`, or the Windows installer) on PATH; the `--ocr {auto,force,never}` flag; and the `OCR_MAX_PAGES` env var (default 50, web only). Add `OCR_MAX_PAGES` to the env-var list where `ALLOWED_ORIGINS`/`VERIFICATION_MODEL` are documented, matching that format.

- [ ] **Step 4: Add the frontend note**

In `frontend/src/App.jsx`, under the dropzone `<p>` that says "Drop a PDF here..." (around line 130), the dropzone shows either the filename or the prompt. Add a static helper line inside the idle-card `<section>`, just after the closing `</div>` of the dropzone (before the `<fieldset>`):

```jsx
          <p className="note">Scanned PDFs are supported via OCR.</p>
```

- [ ] **Step 5: Verify**

Run: `python -c "import yaml; yaml.safe_load(open('render.yaml'))"` (if pyyaml is importable; otherwise visually confirm indentation matches the existing entries).
Run: `cd frontend && npm run build` and confirm it builds with no errors.
Expected: YAML valid; frontend builds. (The Dockerfile change cannot be runtime-verified without a Docker build; confirm the lines are syntactically correct and the tessdata path matches the documented Debian location.)

- [ ] **Step 6: Commit**

```bash
git add backend/Dockerfile render.yaml README.md frontend/src/App.jsx
git commit -m "docs: OCR deployment (Tesseract), OCR_MAX_PAGES, and usage notes"
```

---

## Self-Review

**Spec coverage:**
- OCR engine = PyMuPDF Tesseract, English, DPI 300, no new Python dep → Task 1 (`ocr.py`).
- Shared line-building (no duplication) → Task 1 (`extractor.lines_from_blocks`).
- `ocr_mode`/`ocr_max_pages`/`used_ocr`/new errors/branch rule → Task 2.
- `NoTextLayerError` contains `OCR`; reworded; OCR-found-nothing case → Task 2.
- CLI `--ocr` → Task 3.
- Web `OCR_MAX_PAGES`, pass-through, friendly errors (incl. `scanned` substring kept) → Task 4.
- Deployment (Tesseract in Docker, `TESSDATA_PREFIX`), render.yaml, README, frontend note → Task 5.
- Detection-only (no searchable layer) → honored everywhere; no writer change.
- Success criteria: scanned→bookmarks (Task 1+2), OcrUnavailable clear (Task 2), web cap rejection (Task 2+4), `--ocr never`/`force` (Task 2+3), no duplicated merge logic (Task 1), pytest passes incl. skip (all tasks).
- Existing-test interactions: `test_pipeline.py::test_no_text_layer_raises` updated (Task 2), `test_cli.py::test_no_text_layer_errors` updated (Task 3), web `scanned`-substring tests preserved (Task 4).

**Placeholder scan:** none — every step has concrete code/commands. The `TESSDATA_PREFIX` path carries a concrete default plus a verification instruction (legitimate deploy caveat, not a placeholder).

**Type/name consistency:** `lines_from_blocks`, `ocr.available`, `ocr.extract_lines_via_ocr`, `ocr.DPI`, `ocr_mode`, `ocr_max_pages`, `used_ocr`, `OcrUnavailableError`, `OcrPageLimitError`, `OCR_MAX_PAGES`, `scanned_text_pdf` used consistently across tasks and against the current code read during planning.
