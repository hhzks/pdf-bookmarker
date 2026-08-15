# pdf-bookmarker

Add a hierarchical bookmark outline to text-based PDFs. Parses the table of
contents when one exists (preserving chapter/subchapter structure and linking
each bookmark to the section's real location), and falls back to font-based
heading detection when there is no TOC. Two optional detectors improve on that
substantially: a [trained heading model](#heading-model-recommended) and an
[LLM pass](#choosing-a-model) that verifies or repairs low-confidence outlines.

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

### Local model (no API key)

A fine-tuned local model can replace the cloud LLM entirely — nothing leaves
your machine, and there is no per-document cost:

    pip install -e ".[local]"     # llama-cpp-python, CPU is fine
    pdf-bookmarker input.pdf --llm --model "local:models/outline.gguf"

The shipped model is a QLoRA fine-tune of Qwen3.5-2B, quantized to q8_0
(~2 GB). The web backend can serve it (`VERIFICATION_MODEL`) but does not by
default — see the deployment settings below.
It is loaded once and reused, and generation is serialized — llama.cpp keeps
state in the context, so concurrent calls on one model would corrupt it.

On a CUDA machine, offload it to the GPU (roughly 50s for a 130-page document
here, versus minutes on CPU):

    PDF_BOOKMARKER_LOCAL_N_GPU_LAYERS=-1     # -1 = every layer
    PDF_BOOKMARKER_LOCAL_N_CTX=16384         # prompt + generated outline

Generation is grammar-constrained to the outline JSON schema, so the model
cannot produce malformed output. See `training/README.md` for how to build
the GGUF (harvest data, QLoRA fine-tune, `export_gguf.py`).

The LLM layer is provider-agnostic: implement `pdf_bookmarker.llm.LLMBackend`,
register the class in `_BACKENDS`, and (optionally) list its key env vars in
`ENV_KEYS` to add another provider.

### Heading model (recommended)

An optional trained heading detector classifies every line of the PDF as
not-a-heading or its nesting level. Titles come from the page itself, so they
are exact, and each bookmark already knows its physical page — no searching for
it afterwards. It is a ~5 MB gradient-boosted tree over typography features
(size, weight, position, spacing): CPU-only, milliseconds per document, no key.

    pip install -e ".[labeler]"                        # scikit-learn + joblib
    pdf-bookmarker input.pdf --labeler models/labeler.joblib

Or point `PDF_BOOKMARKER_LABELER` at it once and drop the flag. Measured over
76 held-out documents (macro-averaged, section numbering ignored). Every row is
the outline `process_pdf` actually produces — extraction, detection, locator and
merge included — against the shipped model, so the rows are comparable to each
other. Reproduce with `training/route_check.py`:

| configuration | LLM calls | title F1 | level accuracy |
|---|---|---|---|
| font heuristics (default install) | — | 0.6208 | 0.8218 |
| LLM alone | 100% | 0.7970 | 0.8181 |
| `--labeler` | 0% | **0.8009** | **0.8928** |
| `--labeler` + auto mode (default) | 38% | **0.8124** | 0.8853 |
| `--labeler --llm` | 100% | 0.8117 | 0.8783 |

The LLM row is the shipped GGUF, replayed from cached predictions so the whole
table is one model measured one way. **On titles the two are tied** — paired
per document, +0.0039 for the heading model with a 95% CI of [−0.0196,
+0.0263], 28 documents better against 26. Its advantage is everything else:
hierarchy (+0.0747, CI [+0.0464, +0.1060], 39 better against 5), exact physical
pages by construction, no API key, and milliseconds per document against
minutes.

The two detectors miss different headings — the model cannot name a heading
that is not a line of text, the LLM reconstructs wrapped and merged ones — so
the last two rows merge the outlines rather than letting one replace the other.

The two level-accuracy figures answer different questions. **0.8928 is the
heading model alone**; **0.8853 is end-to-end at the default routing**, where
merging the LLM's entries in costs a little hierarchy. Neither contradicts the
other.

With a labeler configured, auto mode calls the LLM when the model found
**0.5 headings per page or fewer**. That routes 38% of documents, and it is the
peak of the curve rather than a compromise on it — escalating more buys nothing
measurable (0.8112 at `1.0`, 0.8117 at `--llm`) and costs 2.6x the calls.
`--llm-density` moves the threshold; `0` never escalates.

**Be careful reading the last two rows as a gain.** Paired per document, the
default's +0.0115 over the labeler alone has a 95% CI of [−0.0069, +0.0318]
(16 documents better, 9 worse, 51 unchanged) — the union is no longer
distinguishable from the model alone on a set this size. Only the smallest
budget is individually significant (`--llm-density 0.25`, 8% routed, +0.0086,
CI [+0.0010, +0.0192], 5 better against 1 worse). This is a change from earlier
revisions of this table: as the heading model improved, the LLM's marginal
contribution shrank. Treat the LLM pass as insurance for documents where the
model finds little, which is exactly what routing on density selects for.

Train one with `training/train_line_labeler.py --save-model` (see
`training/README.md`); the bundle records the feature set it was fitted on and
is refused if it does not match the code loading it.

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

### Deployment settings

| env var | effect |
|---|---|
| `ALLOWED_ORIGINS` | comma-separated CORS allowlist; unset blocks other origins |
| `PDF_BOOKMARKER_LABELER` | path to the heading model; unset means heuristics only |
| `REQUIRE_LABELER` | refuse to boot without a working heading model |
| `VERIFICATION_MODEL` | server-side LLM; **unset by default — the server runs none** |
| `PDF_BOOKMARKER_LOCAL_N_GPU_LAYERS` | GPU offload for a local model (`-1` = all) |
| `OCR_MAX_PAGES` | reject scanned PDFs longer than this (default 50) |

The labeler is loaded and validated once at startup: a path that cannot be used
stops the server there, rather than failing every upload with a message about a
model the user never asked for. Replacing the file on disk is picked up without
a restart.

An *unset* path is treated differently, because heuristics-only is a legitimate
way to run this server. That default is wrong for a deployment built around the
model — it would answer every job at 0.6208 title F1 instead of 0.8009 and look
healthy doing it — so such a deployment sets `REQUIRE_LABELER=true` and gets a
boot failure instead. `render.yaml` sets it.

### The heading model in the image

`backend/Dockerfile` installs the `labeler` extra and downloads the model from
the [`labeler-v1`](https://github.com/hhzks/pdf-bookmarker/releases/tag/labeler-v1)
release, checking it against a pinned SHA-256 so the image either has the exact
model that was measured or fails to build. The model is not tracked in git: it
is ~5 MB of fitted estimators that a retrain replaces wholesale.

scikit-learn is pinned to the version that fitted the bundle. It is pickled
estimators, and unpickling across scikit-learn versions is not guaranteed — the
pin keeps an unrelated upstream release from breaking a deploy. **Retraining
means a new release and three edits in step:** upload the asset, then update
`LABELER_VERSION` and `LABELER_SHA256` in the Dockerfile.

**The deployed server runs no LLM of its own.** The heading model produces the
outline; verification happens only when a caller supplies their own API key.
On a CPU host the local GGUF costs minutes per document for +0.0115 title F1 at
the default routing — an interval that spans zero — which is not a trade worth
making by default. (That was believed to be roughly 3 F1 when this default was
chosen; re-measuring against the current heading model made the case stronger,
not weaker.) Setting `VERIFICATION_MODEL`
turns it back on — worthwhile on a GPU host — in which case the model is
*checked* at startup but not loaded, and a missing file is a warning rather
than a failure. (The frontend's "This server runs the heading model only" note
assumes the default; update it if you enable server-side verification.)

    # in a second terminal
    cd frontend
    npm install
    npm run dev   # proxies /api to :8000
