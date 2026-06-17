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

Scanned PDFs (no text layer) are automatically OCR'd and bookmarked like any
other PDF. The CLI needs the `tesseract` binary installed and on `PATH`:

    apt install tesseract-ocr        # Debian/Ubuntu
    brew install tesseract           # macOS
    # or the Tesseract Windows installer

Control OCR behavior with `--ocr`:

    pdf-bookmarker scanned.pdf --ocr auto    # OCR only if there's no text layer (default)
    pdf-bookmarker scanned.pdf --ocr force   # always OCR, even if text exists
    pdf-bookmarker scanned.pdf --ocr never   # never OCR; fail on scanned PDFs

On the web app, OCR runs in `auto` mode and scanned PDFs longer than
`OCR_MAX_PAGES` (default 50) are rejected to bound processing cost.

## Limitations

- Encrypted PDFs are not supported.

## Development

    pip install -e ".[dev]"
    python -m pytest

## Web app

An ilovepdf-style web UI lives in `frontend/` (React + Vite, hosted on
Vercel) with a FastAPI backend in `backend/` (hosted on Render). Uploaded
and processed files are deleted after one hour.

### Run locally

    pip install -e ".[dev]"
    cd backend
    uvicorn app.main:app --port 8000

    # in a second terminal
    cd frontend
    npm install
    npm run dev   # proxies /api to :8000

### Deploy

**Backend (Render):** create a Blueprint from this repo (`render.yaml`) or a
Docker web service with context `.` and Dockerfile `backend/Dockerfile`.
Set environment variables:

- `ALLOWED_ORIGINS` — your frontend URL, e.g. `https://your-app.vercel.app`
  (comma-separate several origins)
- `ANTHROPIC_API_KEY` and/or `GEMINI_API_KEY` — server-side LLM keys
  (optional; users can also bring their own key in the UI)
- `VERIFICATION_MODEL` — optional. The `provider:model-id` used for LLM
  verification when the user does not supply their own API key. Defaults to
  `gemini:gemini-3.5-flash`. The matching provider key (e.g. `GEMINI_API_KEY`)
  must be set.
- `OCR_MAX_PAGES` — optional. Caps the page count of scanned PDFs eligible
  for OCR. Defaults to `50`.

The free tier sleeps after idle: the first request afterwards takes ~1 min,
and in-memory jobs are lost on restart (files are temporary by design).

**Frontend (Vercel):** import the repo, set the project root directory to
`frontend/`, and set `VITE_API_BASE_URL` to the Render URL
(e.g. `https://pdf-bookmarker-api.onrender.com`).
