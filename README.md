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
    pdf-bookmarker input.pdf --model gemini                        # gemini-3.5-flash
    pdf-bookmarker input.pdf --model gemini:gemini-3.1-pro-preview

The Google Gemini backend needs the `gemini` extra and a key in
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`):

    pip install -e ".[gemini]"

The LLM layer is provider-agnostic: implement `pdf_bookmarker.llm.LLMBackend`,
register the class in `_BACKENDS`, and (optionally) list its key env vars in
`ENV_KEYS` to add another provider.

### Scanned PDFs (OCR)

Scanned PDFs (no text layer) are automatically OCR'd and run through the same
outline detection as born-digital PDFs. The CLI needs the `tesseract` binary
installed and on `PATH`:

    apt install tesseract-ocr        # Debian/Ubuntu
    brew install tesseract           # macOS
    # or the Tesseract Windows installer

OCR'd text loses the bold/exact-size cues that font-heuristic heading
detection relies on, so outline quality on scans is generally lower than on
born-digital PDFs. Pairing OCR with LLM verification (`--ocr` together with
`--llm`, or auto mode with an API key set) is recommended for scanned
documents.

Control OCR behavior with `--ocr`:

    pdf-bookmarker scanned.pdf --ocr auto    # OCR only if there's no text layer (default)
    pdf-bookmarker scanned.pdf --ocr force   # always OCR, even if text exists
    pdf-bookmarker scanned.pdf --ocr never   # never OCR; fail on scanned PDFs

On the web app, OCR runs in `auto` mode and scanned PDFs longer than
`OCR_MAX_PAGES` (default 50) are rejected to bound processing cost.

## Development

    pip install -e ".[dev]"
    python -m pytest

## Web app

Web UI lives in `frontend/` (React + Vite) with a FastAPI backend in `backend/`. Uploaded and processed files are deleted after one hour. You can access the website [here](https://pdf-bookmarker.vercel.app/), or alternatively run it yourself.

### Run locally

    pip install -e ".[dev]"
    cd backend
    uvicorn app.main:app --port 8000

    # in a second terminal
    cd frontend
    npm install
    npm run dev   # proxies /api to :8000
