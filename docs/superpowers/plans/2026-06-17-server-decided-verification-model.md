# Server-decided Verification Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the server decide the LLM verification model (default `gemini:gemini-3.5-flash`), allowing a model choice only when the caller supplies their own API key.

**Architecture:** Enforce the rule in the backend route (`routes.py`): a client-supplied `model` is honored only alongside an `api_key`; otherwise the server's `SERVER_MODEL_SPEC` is used. The frontend hides the model picker behind the "use my own API key" toggle. The Python library default and CLI `--model` are untouched.

**Tech Stack:** FastAPI (backend), React 19 + Vite (frontend), pytest (backend tests). No JS test runner is configured — frontend verification is `npm run build` plus reading the diff.

## Global Constraints

- Server model config: `SERVER_MODEL_SPEC = os.environ.get("VERIFICATION_MODEL", "gemini:gemini-3.5-flash")` — env var overridable, no code change to swap models.
- Rule: caller may override the model **only** when an `api_key` is present; a `model` without a key is silently ignored (no 400).
- Do NOT change `pdf_bookmarker.llm.DEFAULT_MODEL_SPEC` or the CLI `--model` flag.
- Do NOT change the LLM-mode radios (`auto`/`always`/`never`) or `llm.is_low_confidence`.
- Backend tests import the backend as `app` (conftest prepends `backend/` to `sys.path`); the `fake_pipeline` fixture records pipeline kwargs in a list, including `model_spec`.
- No AI attribution in commits (no `Co-Authored-By` trailer).

---

### Task 1: Backend resolves the model server-side

**Files:**
- Modify: `backend/app/routes.py` (imports near line 7; `create_job` body near lines 24-56)
- Test: `tests/test_webapp_api.py`

**Interfaces:**
- Consumes: `JobStore.submit(pdf_bytes, original_name, *, llm_mode, model_spec, api_key)` (unchanged), recorded by the `fake_pipeline` fixture as `calls[i]["model_spec"]` / `calls[i]["api_key"]`.
- Produces: module constant `SERVER_MODEL_SPEC` in `app.routes`; the `model` form field stays accepted but is ignored without `api_key`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_webapp_api.py` (the `upload` helper and busy-wait pattern already exist in this file):

```python
def _first_call(fake_pipeline, timeout=10):
    deadline = time.time() + timeout
    while not fake_pipeline and time.time() < deadline:
        time.sleep(0.02)
    assert fake_pipeline, "pipeline was never called"
    return fake_pipeline[0]


def test_no_key_ignores_client_model(client, fake_pipeline):
    upload(client, llm_mode="always", model="anthropic:claude-opus-4-8")
    call = _first_call(fake_pipeline)
    assert call["model_spec"] == "gemini:gemini-3.5-flash"
    assert call["api_key"] is None


def test_no_key_no_model_uses_server_model(client, fake_pipeline):
    upload(client, llm_mode="always")
    call = _first_call(fake_pipeline)
    assert call["model_spec"] == "gemini:gemini-3.5-flash"


def test_key_without_model_falls_back_to_server_model(client, fake_pipeline):
    upload(client, llm_mode="always", api_key="user-secret")
    call = _first_call(fake_pipeline)
    assert call["model_spec"] == "gemini:gemini-3.5-flash"
    assert call["api_key"] == "user-secret"


def test_key_with_model_is_honored(client, fake_pipeline):
    upload(client, llm_mode="always", model="anthropic:claude-sonnet-4-6",
           api_key="user-secret")
    call = _first_call(fake_pipeline)
    assert call["model_spec"] == "anthropic:claude-sonnet-4-6"
    assert call["api_key"] == "user-secret"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_webapp_api.py -k "client_model or server_model or honored" -v`
Expected: `test_no_key_ignores_client_model` FAILS (model_spec is `anthropic:claude-opus-4-8`, the current default); the others may pass incidentally.

- [ ] **Step 3: Implement the server-side resolution**

In `backend/app/routes.py`, replace the `DEFAULT_MODEL_SPEC` import:

```python
import os
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

MAX_SIZE = 50 * 1024 * 1024  # 50 MB
VALID_MODES = {"auto", "always", "never"}

# The server decides the verification model. Override at deploy time with the
# VERIFICATION_MODEL env var (e.g. "anthropic:claude-opus-4-8") — no code change.
SERVER_MODEL_SPEC = os.environ.get("VERIFICATION_MODEL", "gemini:gemini-3.5-flash")
```

Then in `create_job`, replace the `model_spec=model or DEFAULT_MODEL_SPEC` line in the `store.submit(...)` call:

```python
    # The server decides the model; a caller may only override it when they
    # bring their own API key. A model sent without a key is ignored.
    model_spec = (model or SERVER_MODEL_SPEC) if api_key else SERVER_MODEL_SPEC

    store = request.app.state.jobs
    job = store.submit(
        bytes(data),
        file.filename or "document.pdf",
        llm_mode=llm_mode,
        model_spec=model_spec,
        api_key=api_key or None,
    )
    return {"job_id": job.id}
```

(The `model: str | None = Form(None)` parameter stays in the signature.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_webapp_api.py -v`
Expected: all PASS (including the existing `test_options_forwarded_to_pipeline`, which sends key + `gemini:gemini-3.5-flash`).

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes.py tests/test_webapp_api.py
git commit -m "feat: server decides verification model unless caller brings own key"
```

---

### Task 2: Frontend gates the model picker behind the own-key toggle

**Files:**
- Modify: `frontend/src/App.jsx` (`MODELS` const lines 4-8; options `fieldset` lines 153-185; `start()` lines 57-73)

**Interfaces:**
- Consumes: `createJob(file, { llmMode, model, apiKey }, onProgress)` in `frontend/src/api.js` — already appends `model` only when truthy; no change needed there.
- Produces: model `<select>` rendered only inside the `showKeyField` block; `model` sent only when an API key is present.

- [ ] **Step 1: Update the MODELS label**

Drop "(default)" — the model is no longer a global default, just the first own-key option:

```jsx
const MODELS = [
  { value: "anthropic:claude-opus-4-8", label: "Claude Opus 4.8" },
  { value: "anthropic:claude-sonnet-4-6", label: "Claude Sonnet 4.6" },
  { value: "gemini:gemini-3.5-flash", label: "Gemini 3.5 Flash" },
];
```

- [ ] **Step 2: Move the model select inside the own-key block**

Replace the whole `{llmMode !== "never" && (...)}` block (currently lines 153-185, the `<>...</>` containing the Model `<label>`, the toggle button, and the key field) with:

```jsx
            {llmMode !== "never" && (
              <>
                <p className="note">Verification model is chosen by the server.</p>
                <button
                  type="button"
                  className="linklike"
                  onClick={() => setShowKeyField(!showKeyField)}
                >
                  {showKeyField ? "Use the server's API key" : "Use my own API key"}
                </button>
                {showKeyField && (
                  <>
                    <label className="field">
                      Model
                      <select value={model} onChange={(e) => setModel(e.target.value)}>
                        {MODELS.map((m) => (
                          <option key={m.value} value={m.value}>
                            {m.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="field">
                      API key (used only for this job, never stored)
                      <input
                        type="password"
                        value={apiKey}
                        autoComplete="off"
                        placeholder="sk-… or AIza…"
                        onChange={(e) => setApiKey(e.target.value)}
                      />
                    </label>
                  </>
                )}
              </>
            )}
```

- [ ] **Step 3: Send model only when a key is present**

In `start()`, change the `createJob` options so the model is omitted without a key (the backend ignores it anyway; this keeps the request honest):

```jsx
      const trimmedKey = apiKey.trim();
      const id = await createJob(
        file,
        { llmMode, model: trimmedKey ? model : undefined, apiKey: trimmedKey },
        setProgress
      );
```

- [ ] **Step 4: Verify the build**

Run: `cd frontend && npm run build`
Expected: build succeeds with no errors. Read the diff to confirm: the `<select>` appears only inside `showKeyField`, the server-model note shows whenever `llmMode !== "never"`, and `model` is sent only with a key.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat: show model picker only when using own API key"
```

---

### Task 3: Document the VERIFICATION_MODEL env var

**Files:**
- Modify: `render.yaml` (`envVars` list, lines 8-14)
- Modify: `README.md` (deployment / environment section)

**Interfaces:**
- Consumes: `SERVER_MODEL_SPEC` env-var contract from Task 1 (`VERIFICATION_MODEL`).
- Produces: documentation only — no code behavior.

- [ ] **Step 1: Add VERIFICATION_MODEL to render.yaml**

Append to the `envVars` list (after `GEMINI_API_KEY`):

```yaml
      - key: VERIFICATION_MODEL
        sync: false   # optional; defaults to gemini:gemini-3.5-flash
```

- [ ] **Step 2: Document it in the README**

Find the deployment/environment-variables section (search `ALLOWED_ORIGINS` in `README.md`) and add a row/line describing `VERIFICATION_MODEL`:

> `VERIFICATION_MODEL` — optional. The `provider:model-id` used for LLM verification when the user does not supply their own API key. Defaults to `gemini:gemini-3.5-flash`. The matching provider key (e.g. `GEMINI_API_KEY`) must be set.

Match the surrounding formatting (table row vs. bullet) exactly.

- [ ] **Step 3: Verify**

Run: `python -c "import yaml; yaml.safe_load(open('render.yaml'))"` (if pyyaml is available) or visually confirm `render.yaml` indentation matches the existing entries.
Expected: no YAML error; README mentions `VERIFICATION_MODEL`.

- [ ] **Step 4: Commit**

```bash
git add render.yaml README.md
git commit -m "docs: document VERIFICATION_MODEL env var"
```

---

## Self-Review

**Spec coverage:**
- Server picks model / env-var configurable → Task 1 (`SERVER_MODEL_SPEC`).
- Override only with own key, enforced server-side → Task 1 (`(model or SERVER_MODEL_SPEC) if api_key else SERVER_MODEL_SPEC`) + tests.
- UI hides picker unless own key → Task 2.
- CLI/library unchanged → no task touches them (constraint stated).
- Deployment/docs → Task 3.
- Success criteria (no-key always server model; key honored; picker gated; env swap; pytest passes) → covered by Task 1 tests + Task 2 build + Task 3.

**Placeholder scan:** none — all steps carry concrete code/commands.

**Type/name consistency:** `SERVER_MODEL_SPEC`, `VERIFICATION_MODEL`, `model_spec`, `model`, `api_key`, `showKeyField`, `MODELS` used consistently across tasks and match the existing code read during brainstorming.
