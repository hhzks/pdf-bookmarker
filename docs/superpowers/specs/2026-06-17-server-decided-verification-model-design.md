# Server-decided verification model

**Date:** 2026-06-17
**Status:** Approved (pending spec review)

## Problem

The web UI lets any user choose the LLM model used for the verification stage.
The model should instead be decided by the server. The sole exception: when a
user supplies their own API key, they may choose which model to use with that
key.

The concrete server model is Gemini 3.5 Flash (`gemini:gemini-3.5-flash`), but
swapping it later must be trivial.

## Rule

The server picks the verification model, **unless** the caller supplies their
own API key — in which case the caller may choose the model for that key. This
is enforced on the server, not merely hidden in the UI: a client-supplied model
without an accompanying key is silently ignored (not a 400).

## Changes

### 1. Backend — `backend/app/routes.py`

- Add a module-level config read from the environment:
  ```python
  SERVER_MODEL_SPEC = os.environ.get("VERIFICATION_MODEL", "gemini:gemini-3.5-flash")
  ```
  This is the "trivial change" seam: swap the model at deploy time via the
  `VERIFICATION_MODEL` env var, matching the existing `ALLOWED_ORIGINS` /
  API-key env pattern. No code change needed to switch models.
- Resolve the model spec in `create_job`:
  ```python
  # The server decides the model; a caller may only override it when they
  # bring their own API key.
  model_spec = (model or SERVER_MODEL_SPEC) if api_key else SERVER_MODEL_SPEC
  ```
- Remove the now-unused `from pdf_bookmarker.llm import DEFAULT_MODEL_SPEC`
  import.
- The `model` form field is still accepted (so own-key callers can pass it) but
  is ignored when no `api_key` is present.

### 2. Frontend — `frontend/src/App.jsx`

- Move the Model `<select>` **inside** the "Use my own API key" (`showKeyField`)
  block. It renders only when the user has chosen to supply their own key.
- When using the server's key (default), show no model picker. Add a one-line
  note, e.g. *"Verification model is chosen by the server."*
- In `start()`, send `model` only when an API key is present. (The backend
  ignores it otherwise; this keeps the request honest.)
- Drop "(default)" from the Claude Opus 4.8 label — it is no longer a global
  default, just the first option offered for the own-key path. `MODELS` list
  otherwise unchanged.

### 3. Library / CLI — unchanged

`llm.DEFAULT_MODEL_SPEC` stays the library default and the CLI `--model` flag is
retained. This task targets the web app's end users; the CLI is a
local/developer tool documented in CLAUDE.md.

### 4. Tests — `tests/test_webapp_api.py`

- Keep `test_options_forwarded_to_pipeline` (key + model still flow through to
  the pipeline).
- Add cases asserting on the `model_spec` passed to the fake pipeline:
  - no key + client `model` supplied → pipeline receives `SERVER_MODEL_SPEC`
    (override ignored).
  - no key + no model → pipeline receives `SERVER_MODEL_SPEC`.
  - key + no model → pipeline receives `SERVER_MODEL_SPEC` (own-key fallback).

### 5. Deployment / docs

No code-path blockers: the Docker image already installs `[gemini,web]` and
`render.yaml` already provisions `GEMINI_API_KEY`. Document the optional
`VERIFICATION_MODEL` env var:
- a commented/`sync: false` entry (or a note) in `render.yaml`.
- a brief mention in the deployment section of the README.

## Out of scope

- Heuristic confidence logic (`llm.is_low_confidence`) — unchanged.
- The LLM-mode radios (`auto` / `always` / `never`) — unchanged.
- The set of models offered in the own-key dropdown — unchanged.

## Success criteria

- With no API key, the pipeline always runs with `SERVER_MODEL_SPEC`, regardless
  of any `model` value a client sends.
- With an API key, the caller's chosen model is honored (falling back to
  `SERVER_MODEL_SPEC` if none given).
- The model picker is visible in the UI only when the user supplies their own
  key.
- Setting `VERIFICATION_MODEL` changes the server model with no code change.
- `python -m pytest` passes.
