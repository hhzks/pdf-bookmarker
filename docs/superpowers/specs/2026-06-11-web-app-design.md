# pdf-bookmarker Web App — Design

**Date:** 2026-06-11
**Status:** Approved

## Goal

Turn the `pdf-bookmarker` CLI into a public, free-to-host web application
(ilovepdf-style): a user uploads a text-based PDF, the backend adds a
hierarchical bookmark outline, and the user downloads the result. Files are
temporary and deleted after one hour.

## Decisions made during brainstorming

- **Hosting:** public deployment on a free platform. Primary target:
  Hugging Face Spaces (free Docker hosting, no card, no sleep). The image
  must also run anywhere generic (`$PORT` env var).
- **LLM verification:** same three modes as the CLI — `auto` (default),
  `always`, `never` — selectable in the UI. Uses the server-side API key by
  default; the user may optionally supply their own key, used only for that
  job and never stored or logged.
- **Stack:** FastAPI backend + React/Vite frontend, served from one origin.
- **Processing model:** async jobs with in-process workers (no Redis/Celery),
  job-ID + polling, to survive slow LLM calls without gateway timeouts.

## Security note (pre-existing issue)

`pdf_bookmarker/llm.py` currently hardcodes a real Google API key in
`GeminiBackend.__init__`. This design removes it (keys come from an explicit
parameter or env vars), but the key is in git history and must be **revoked
by the owner in Google AI Studio**.

## Architecture

```
repo/
  pdf_bookmarker/      # existing engine (refactored, CLI unchanged in behaviour)
    pipeline.py        # NEW: shared processing pipeline
    cli.py             # becomes a thin wrapper over pipeline.py
    llm.py             # backends gain api_key parameter; hardcoded key removed
  backend/             # NEW: FastAPI app
    app/
      main.py          # app factory, static file serving
      jobs.py          # job store, worker pool, cleanup task
      routes.py        # API endpoints
  frontend/            # NEW: React + Vite single page
  Dockerfile           # multi-stage: node build -> python runtime
```

### 1. Pipeline extraction (`pdf_bookmarker/pipeline.py`)

One reusable entry point shared by the CLI and the web backend:

```python
def process_pdf(
    input_path: Path,
    output_path: Path,
    *,
    llm_mode: str = "auto",        # "auto" | "always" | "never"
    model_spec: str = DEFAULT_MODEL_SPEC,
    api_key: str | None = None,    # overrides env-var key when given
) -> PipelineResult               # bookmark_count, used_llm, used_toc
```

Failures raise typed exceptions instead of printing to stderr:
`EncryptedPdfError`, `NoTextLayerError`, `NoOutlineError`,
`llm.UnknownProviderError`, plus the existing LLM-failure behaviour
(auto mode falls back to heuristics; `always` mode raises).

The pipeline always replaces existing bookmarks (CLI `--force` semantics);
the CLI keeps its own already-has-bookmarks guard so its behaviour is
unchanged. LLM backends (`AnthropicBackend`, `GeminiBackend`) accept
`api_key: str | None = None`, falling back to their env vars as today.

### 2. Backend API (FastAPI)

| Endpoint | Behaviour |
|---|---|
| `POST /api/jobs` | Multipart: `file` (required), `llm_mode` (default `auto`), `model` (default server default), `api_key` (optional). Validates PDF magic bytes and 50 MB cap. Returns `{"job_id": "<uuid>"}` (202). |
| `GET /api/jobs/{id}` | `{"status": "queued"\|"processing"\|"done"\|"failed", "error": str?, "bookmark_count": int?}`. 404 for unknown/expired ids. |
| `GET /api/jobs/{id}/download` | Streams the result as `<original-stem>.bookmarked.pdf`. 404 if not done or expired. |

- **Job execution:** `ThreadPoolExecutor(max_workers=2)` in-process.
- **Job store:** in-memory dict `{job_id: Job}`; per-job temp directory
  holding `input.pdf` and `output.pdf`.
- **Cleanup:** periodic task deletes jobs (state + files) older than 1 hour.
- **Rate limiting:** per-IP, 10 job submissions per hour, to protect the
  server-side LLM key. Returns 429.
- **API keys:** user-supplied keys live only in the in-memory job record for
  the duration of the job, are passed to the backend constructor, and are
  excluded from all logging.
- **Error mapping:** typed pipeline exceptions map to friendly messages
  (e.g. "This PDF appears to be scanned — it has no text layer").
- Serves the built frontend from `/` as static files (same origin, no CORS).

### 3. Frontend (React + Vite, single page)

Flow: drag-and-drop / file-picker card → options panel → upload with
progress → "Processing…" polling state → Download button → "files are
deleted after 1 hour" note, with a "process another file" reset.

Options panel:
- LLM verification: radio — Auto (default) / Always / Never.
- Model: dropdown of provider:model presets (server default preselected).
- Collapsible "Use my own API key" text input (password-style field).

Errors render as a friendly message card with a retry option. Polling
interval ~1.5 s. The Vite build (`frontend/dist`) is copied into the backend
image and served by FastAPI.

### 4. Deployment

Multi-stage `Dockerfile`:
1. `node` stage: `npm ci && npm run build` in `frontend/`.
2. `python:3.12-slim` stage: install `pdf_bookmarker` + backend deps
   (incl. `[gemini]` extra), copy frontend build, run `uvicorn` on `$PORT`
   (default 7860 to match Hugging Face Spaces).

Server LLM key supplied via platform secret (`ANTHROPIC_API_KEY` /
`GEMINI_API_KEY`). No other persistent state — fully ephemeral filesystem is
fine because files are temporary by design.

### 5. Testing

- **Pipeline:** unit tests for `process_pdf` (success, typed errors, api_key
  pass-through) reusing existing PDF fixtures.
- **API:** FastAPI `TestClient` tests with the pipeline monkeypatched:
  job lifecycle (submit → poll → download), validation rejections (non-PDF,
  oversize), unknown job 404, error mapping, rate limit 429, cleanup TTL.
- **Existing tests:** the current CLI test suite must keep passing.
- **Frontend:** no automated tests initially (single page, manual
  verification); revisit if the UI grows.

## Out of scope

- OCR for scanned PDFs (existing limitation).
- Accounts, persistent history, multi-file batches.
- Distributed queue / horizontal scaling.
