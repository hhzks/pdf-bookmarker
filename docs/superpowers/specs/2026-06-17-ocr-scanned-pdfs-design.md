# OCR for scanned (no-text-layer) PDFs

**Date:** 2026-06-17
**Status:** Approved (pending spec review)

## Problem

The tool only works on PDFs that already have a text layer. Scanned documents
(image-only pages) currently fail with `NoTextLayerError`. We want to recognise
their text via OCR so the existing outline-detection pipeline can run on them.

## Decisions (locked during brainstorming)

- **OCR engine:** Tesseract via PyMuPDF's built-in `page.get_textpage_ocr()`.
  OCR'd text flows through the same `get_text("dict")` path, producing the same
  `Line` objects, so the rest of the pipeline is unchanged. Tesseract is a
  **system binary** (PyMuPDF shells out to it); no new Python dependency.
- **Cost control:** a configurable OCR page cap. On the web, a scanned doc with
  more pages than the cap is rejected with a clear message. The CLI has no cap
  by default (local users accept the wait).
- **Output:** detection only — the output PDF is the original plus the bookmark
  outline. No searchable text layer is embedded.

## Architecture

OCR plugs into the documented extractor seam. For a scanned page,
`tp = page.get_textpage_ocr(full=True, dpi=300, language="eng")` yields a
TextPage that `page.get_text("dict", textpage=tp)` reads exactly like a
born-digital page. The resulting blocks/lines feed the same baseline-merging
logic used today, producing `Line` objects with real font sizes (Tesseract
reports glyph heights) but no bold flag (`bold=False`). Heuristic detection
still runs; the existing LLM-verification stage (auto mode) layers on top for
quality, which matters more on noisy OCR output.

## Components

### New: `pdf_bookmarker/ocr.py`

Owns the Tesseract dependency and the rendering details.

- `DPI = 300` — module constant (render resolution for OCR).
- `available() -> bool` — returns `bool(shutil.which("tesseract"))`.
- `extract_lines_via_ocr(doc) -> list[Line]` — for each page, build an OCR
  TextPage and convert it to `Line`s via the shared extractor helper. (The
  page-cap guard lives in the pipeline, which only calls this once the document
  is known to be within the cap.)

### Refactor: `pdf_bookmarker/extractor.py` (DRY)

Pull the per-page `blocks → list[Line]` loop out of `extract_lines` into a
shared helper:

```python
def _lines_from_blocks(blocks: list[dict], page_index: int) -> list[Line]: ...
```

`extract_lines(doc)` becomes a thin loop: `page.get_text("dict")["blocks"]` →
`_lines_from_blocks(blocks, page_index)`. `ocr.extract_lines_via_ocr` calls the
same helper with OCR-derived blocks, so the baseline-merging logic exists in
exactly one place.

### `pdf_bookmarker/pipeline.py`

New parameters on `process_pdf`:
- `ocr_mode: str = "auto"` — one of `"auto" | "force" | "never"`.
- `ocr_max_pages: int | None = None` — `None` means no cap.

New typed errors (PipelineError subclasses):
- `OcrUnavailableError` — OCR was needed/requested but Tesseract is not installed.
- `OcrPageLimitError` — the document exceeds `ocr_max_pages`.

`PipelineResult` gains `used_ocr: bool`.

Replacement for the current no-text guard, after `doc` is opened/validated and
before `extract_lines`:

```python
if ocr_mode not in ("auto", "force", "never"):
    raise ValueError(...)

has_text = extractor.has_text_layer(doc)
use_ocr = ocr_mode == "force" or (ocr_mode == "auto" and not has_text)
if use_ocr:
    if not ocr.available():
        raise OcrUnavailableError("OCR is required but Tesseract is not available")
    if ocr_max_pages is not None and doc.page_count > ocr_max_pages:
        raise OcrPageLimitError(
            f"document has {doc.page_count} pages; OCR limit is {ocr_max_pages}"
        )
    lines = ocr.extract_lines_via_ocr(doc)
    used_ocr = True
else:
    if not has_text:
        raise NoTextLayerError(_NO_TEXT_MESSAGE)
    lines = extractor.extract_lines(doc)
    used_ocr = False

if not lines:
    raise NoTextLayerError(_NO_TEXT_MESSAGE)
```

`used_ocr` is threaded into the returned `PipelineResult`.

### `pdf_bookmarker/cli.py`

Add `--ocr {auto,force,never}` (default `auto`), passed as `ocr_mode`. No
page-cap flag (CLI uses `ocr_max_pages=None`). The two new errors are already
caught by the generic `except (pipeline.PipelineError, llm.UnknownProviderError)`
handler (exit code 2), printing their message — no extra CLI branches needed.

### Web backend (`backend/app/`)

- `routes.py`: `OCR_MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "50"))`.
  `create_job` always uses `ocr_mode="auto"` and passes `ocr_max_pages=OCR_MAX_PAGES`
  through to the job. No new form field.
- `jobs.py`: `JobStore.submit` and `_run` gain `ocr_mode` and `ocr_max_pages`
  parameters, forwarded to `process_pdf` (mirrors `model_spec`). Add to
  `friendly_error`:
  - `OcrPageLimitError` → "This scanned PDF is too long to process (the OCR
    limit is set by the server). Try a shorter document."
  - `OcrUnavailableError` → "This server can't process scanned PDFs right now."
  - Reword the `NoTextLayerError` string: OCR now runs automatically, so this
    fires only when OCR recovered no readable text — e.g. "No readable text
    could be found in this PDF, even after OCR."
- `frontend/src/App.jsx`: a static note under the dropzone — "Scanned PDFs are
  supported via OCR." (minor copy only).

## Deployment

- `backend/Dockerfile`: install the Tesseract binary and English data, and set
  the tessdata path so PyMuPDF can find it:
  ```dockerfile
  RUN apt-get update \
      && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
      && rm -rf /var/lib/apt/lists/*
  ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
  ```
  The exact `TESSDATA_PREFIX` path depends on the base image's Tesseract
  version (Debian bookworm ships Tesseract 5); confirm it during implementation
  by checking where `tesseract-ocr-eng` installs `eng.traineddata`.
- `render.yaml`: add an optional `OCR_MAX_PAGES` env var entry (`sync: false`,
  commented with its default of 50).
- `README.md`: document the OCR feature, the local Tesseract install
  requirement (so the CLI works locally), and the `OCR_MAX_PAGES` env var.

## Testing

- `tests/test_ocr.py`:
  - `available()` returns True/False based on a monkeypatched `shutil.which`.
  - **Real OCR integration test**, `@pytest.mark.skipif(not ocr.available(), ...)`:
    use the `scanned_text_pdf` fixture, run `extract_lines_via_ocr`, assert the
    known words are recovered (case-insensitive substring match — OCR is not
    exact).
- `tests/conftest.py`: new `scanned_text_pdf` fixture — render real text to a
  pixmap and insert it as a full-page image into a fresh page, so the page has
  **no text layer** but is visually legible (`has_text_layer` returns False).
- `tests/test_pipeline.py` (no Tesseract needed — monkeypatch `ocr.available`
  and `ocr.extract_lines_via_ocr`):
  - `auto` + no text layer + OCR available → OCR path runs, `used_ocr=True`.
  - `auto` + no text layer + OCR unavailable → `OcrUnavailableError`.
  - `never` + no text layer → `NoTextLayerError` (unchanged behavior).
  - `force` + a text PDF → OCR path runs despite the existing text layer.
  - `ocr_max_pages` smaller than the page count → `OcrPageLimitError`.
- `tests/test_webapp_api.py`: a job over the cap surfaces the page-limit
  friendly error; confirm `ocr_max_pages`/`ocr_mode` reach the pipeline via the
  `fake_pipeline` recorded kwargs.

## Success criteria

- A scanned PDF (no text layer) gets a bookmark outline when Tesseract is
  available, via the unchanged detection pipeline.
- With OCR unavailable, the failure is a clear `OcrUnavailableError`, not a
  silent crash.
- On the web, a scanned doc longer than `OCR_MAX_PAGES` is rejected with a
  friendly message; within the cap it is processed.
- `--ocr never` reproduces today's behavior; `--ocr force` re-OCRs even
  text-bearing PDFs.
- The baseline-merging logic is not duplicated between `extractor` and `ocr`.
- `python -m pytest` passes (the real-OCR test skips cleanly where Tesseract is
  absent).

## Out of scope

- Embedding a searchable text layer in the output PDF.
- Multi-language OCR (English only; `language` is a constant, not a parameter).
- Per-page lazy OCR / OCRing only the pages the locator needs.
- Auto-OCR of mixed text+scanned documents (`--ocr force` is the manual escape
  hatch).
