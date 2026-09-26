# pdf-bookmarker

Adds a clickable bookmark outline (chapters, sections, subsections) to PDFs
that have none.

**[Try it in the browser](https://pdf-bookmarker.vercel.app/)**, or run it
locally as a command-line tool.

## How it works

1. **Read the text** of every page, with its font size, weight and position.
   Scanned PDFs are OCR'd first.
2. **Find the headings**, using the first available method:
   - a small **trained heading model** (recommended, see below), or
   - the PDF's own **table of contents**, or
   - **font rules** (large or bold lines).
3. **Optionally ask an LLM** to check the outline when it looks incomplete.
4. **Write the bookmarks** into a copy of the PDF.

## Quick start

    pip install -e .
    pdf-bookmarker input.pdf              # writes input.bookmarked.pdf

| option | effect |
|---|---|
| `-o out.pdf` | choose the output file |
| `--dry-run` | print the outline, write nothing |
| `--force` | replace bookmarks the PDF already has |
| `--labeler PATH` | use the trained heading model |
| `--labeler-nontex PATH` | use a second heading model for PDFs not made with LaTeX |
| `--llm` / `--no-llm` | always / never call the LLM (default: only when needed) |
| `--model SPEC` | choose the LLM (see below) |
| `--ocr auto\|force\|never` | when to OCR (default `auto`: only if there is no text) |

## Heading model (recommended)

A ~5 MB model that looks at each line's typography and decides whether it is
a heading, and at what level. It runs on CPU in milliseconds and needs no API
key.

    pip install -e ".[labeler]"
    curl -LO https://github.com/hhzks/pdf-bookmarker/releases/download/labeler-v1/labeler.joblib
    pdf-bookmarker input.pdf --labeler labeler.joblib

Set `PDF_BOOKMARKER_LABELER=labeler.joblib` to use it without the flag.

The model above was trained mostly on LaTeX papers. For PDFs made with other
software (Word, InDesign and similar), add the second model. The tool reads
the PDF's producer and uses the second model for every PDF not made with
LaTeX:

    curl -LO https://github.com/hhzks/pdf-bookmarker/releases/download/labeler-nontex-v1/labeler-nontex.joblib
    pdf-bookmarker input.pdf --labeler labeler.joblib --labeler-nontex labeler-nontex.joblib

Set `PDF_BOOKMARKER_LABELER_NONTEX` to use it without the flag.

### Results

Measured on 76 held-out documents (reproduce with `training/route_check.py`):

| configuration | LLM calls | title F1 | level accuracy |
|---|---|---|---|
| font rules only (default install) | — | 0.62 | 0.82 |
| LLM only | 100% | 0.76 | 0.75 |
| heading model | 0% | 0.80 | **0.89** |
| heading model + LLM when needed (default) | 38% | 0.82 | 0.89 |
| heading model + `--llm` | 100% | **0.83** | 0.88 |

66 of these 76 documents are LaTeX. On 81 held-out web PDFs (Word, InDesign,
HTML converters and others), with no LLM:

| configuration | title F1 |
|---|---|
| font rules only | 0.54 |
| heading model | 0.42 |
| heading model + non-LaTeX model | **0.65** |

- **Title F1:** how well the found headings match the real ones.
- **Level accuracy:** how often a found heading sits at the correct depth.

The heading model and the LLM find different headings, so the pipeline
combines their outlines. With the heading model, the LLM is called only when
the model finds 0.5 headings per page or fewer. Use `--llm-density` to change
this threshold: a higher value costs more LLM calls and gives slightly better
results.

## LLM verification

The default LLM is Claude. Set `ANTHROPIC_API_KEY` to use it. If no key is
set, the tool continues without an LLM.

    pdf-bookmarker input.pdf --model anthropic:claude-opus-4-8
    pdf-bookmarker input.pdf --model gemini             # needs pip install -e ".[gemini]"
                                                        # and GEMINI_API_KEY

**Local model (no key, nothing leaves your machine):** a fine-tuned
Qwen3.5-2B in GGUF format (~2 GB). To build it, see `training/README.md`.

    pip install -e ".[local]"
    pdf-bookmarker input.pdf --llm --model "local:models/outline.gguf"

On CPU it takes minutes per document. To use a GPU, set
`PDF_BOOKMARKER_LOCAL_N_GPU_LAYERS=-1`.

To add another provider, implement `pdf_bookmarker.llm.LLMBackend` and
register it in `_BACKENDS`.

## Scanned PDFs

OCR needs the `tesseract` program on your `PATH`:

    apt install tesseract-ocr     # Debian/Ubuntu
    brew install tesseract        # macOS
                                  # Windows: use the Tesseract installer

OCR text has no bold or exact font sizes, so outlines of scans are less
accurate. For scans, use the LLM too.

## Web app

A React frontend (`frontend/`) and a FastAPI backend (`backend/`). Uploaded
files are deleted after one hour.

    pip install -e ".[dev]"
    cd backend && uvicorn app.main:app --port 8000

    cd frontend && npm install && npm run dev    # in a second terminal

### Server settings

| env var | effect |
|---|---|
| `ALLOWED_ORIGINS` | comma-separated CORS allowlist |
| `PDF_BOOKMARKER_LABELER` | path to the heading model; unset means font rules only |
| `PDF_BOOKMARKER_LABELER_NONTEX` | path to the model for PDFs not made with LaTeX; unset means the heading model for all PDFs |
| `REQUIRE_LABELER` | if `true`, refuse to start without a working heading model |
| `VERIFICATION_MODEL` | server-side LLM; unset by default (the server runs no LLM) |
| `PDF_BOOKMARKER_LOCAL_N_GPU_LAYERS` | GPU layers for a local model (`-1` = all) |
| `OCR_MAX_PAGES` | reject scans longer than this (default 50) |

The Docker image (`backend/Dockerfile`) downloads the heading model from the
`labeler-v1` release and the non-LaTeX model from the `labeler-nontex-v1`
release, and checks the SHA-256 of each. To ship a retrained model, upload a
new release asset, then update that model's two build arguments
(`LABELER_VERSION` and `LABELER_SHA256`, or `LABELER_NONTEX_VERSION` and
`LABELER_NONTEX_SHA256`).

If you enable `VERIFICATION_MODEL`, also update the frontend's "This server
runs the heading model only" note.

## Development

    pip install -e ".[dev]"
    python -m pytest
