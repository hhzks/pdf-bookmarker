# pdf-bookmarker Web App — Design

**Date:** 2026-06-11
**Status:** Approved

## Goal

Turn the `pdf-bookmarker` CLI into a public, free-to-host web application
(ilovepdf-style): a user uploads a text-based PDF, the backend adds a
hierarchical bookmark outline, and the user downloads the result. Files are
temporary and deleted after one hour.

## Decisions made during brainstorming

- **Hosting:** public deployment on free platforms, with the frontend and
  backend hosted separately. Frontend: Vercel (free static hosting; Netlify
  or Cloudflare Pages work identically). Backend: Render free tier as a
  Docker web service (Fly.io or Railway are drop-in alternatives). The
  backend image must run anywhere generic (`$PORT` env var).
- **LLM verification:** same three modes as the CLI — `auto` (default),
  `always`, `never` — selectable in the UI. Uses the server-side API key by
  default; the user may optionally supply their own key, used only for that
  job and never stored or logged.
- **Stack:** FastAPI backend + React/Vite frontend, served from one origin.
- **Processing model:** async jobs with in-process workers (no Redis/Celery),
  job-ID + polling, to survive slow LLM calls without gateway timeouts.

## Security note (resolved)

A real Google API key was briefly hardcoded in `GeminiBackend.__init__` in
the local working copy. It was never committed: no commit, dangling object,
or remote ref contains it (verified with `git log -S`, `git grep` over all
revs, and a dangling-blob check), and it has been removed from the working
file. Rotating the key remains a sensible precaution. The design keeps keys
out of source permanently (explicit parameter or env vars only).

## Architecture

```
repo/
  pdf_bookmarker/      # existing engine (refactored, CLI unchanged in behaviour)
    pipeline.py        # NEW: shared processing pipeline
    cli.py             # becomes a thin wrapper over pipeline.py
    llm.py             # backends gain api_key parameter; hardcoded key removed
  backend/             # NEW: FastAPI app (deployed to Render)
    app/
      main.py          # app factory, CORS config
      jobs.py          # job store, worker pool, cleanup task
      routes.py        # API endpoints
    Dockerfile         # python runtime for the API only
  frontend/            # NEW: React + Vite single page (deployed to Vercel)
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
  server-side LLM key. Returns 429. Client IP is taken from
  `X-Forwarded-For` (the backend sits behind the platform's proxy).
- **API keys:** user-supplied keys live only in the in-memory job record for
  the duration of the job, are passed to the backend constructor, and are
  excluded from all logging.
- **Error mapping:** typed pipeline exceptions map to friendly messages
  (e.g. "This PDF appears to be scanned — it has no text layer").
- **CORS:** the API allows only the frontend origin, configured via an
  `ALLOWED_ORIGINS` env var (comma-separated, to cover preview deploys).

### 3. Frontend (React + Vite, single page)

Flow: drag-and-drop / file-picker card → options panel → upload with
progress → "Processing…" polling state → Download button → "files are
deleted after 1 hour" note, with a "process another file" reset.

Options panel:
- LLM verification: radio — Auto (default) / Always / Never.
- Model: dropdown of provider:model presets (server default preselected).
- Collapsible "Use my own API key" text input (password-style field).

Errors render as a friendly message card with a retry option. Polling
interval ~1.5 s. The API base URL comes from `VITE_API_BASE_URL` at build
time (empty in local dev, where Vite's dev-server proxy forwards `/api` to
the local backend).

### 4. Deployment (split hosting)

**Backend — Render free tier**, Docker web service:
- `backend/Dockerfile`: `python:3.12-slim`, installs `pdf_bookmarker` +
  backend deps (incl. `[gemini]` extra), runs `uvicorn` on `$PORT`.
- Env vars set in the platform dashboard: `ANTHROPIC_API_KEY` /
  `GEMINI_API_KEY` (server LLM keys), `ALLOWED_ORIGINS` (frontend URL).
- Render's free tier sleeps after ~15 min idle (first request after sleep
  takes ~50 s) and restarts lose in-memory jobs and temp files. Acceptable:
  files are temporary by design, and the frontend shows expired jobs as
  "session expired — please re-upload".
- Fly.io and Railway run the same image unchanged.

**Frontend — Vercel** (Netlify / Cloudflare Pages equivalent):
- Git-integration build of `frontend/` (`npm run build`), with
  `VITE_API_BASE_URL` pointing at the Render URL.

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
