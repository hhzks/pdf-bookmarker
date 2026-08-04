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
76 held-out documents (macro title F1, section numbering ignored):

| configuration | title F1 |
|---|---|
| font heuristics (default install) | 0.6205 |
| LLM alone | 0.7642 |
| `--labeler` | 0.7797 |
| `--labeler` + auto mode | 0.7978 |
| `--labeler --llm` | 0.8187 |

The two detectors miss different headings — the model cannot name a heading
that is not a line of text, the LLM reconstructs wrapped and merged ones — so
the outlines are merged rather than one replacing the other.

Level accuracy is 0.8879. The LLM rows were measured against an earlier
revision of the heading model (0.7685 titles, 0.8368 levels), so they
understate the pairing slightly. The auto row is end-to-end with the shipped GGUF; the last row replayed the
same model's predictions over every document. Paired per document, the GGUF
and the 4-bit adapter it was merged from are indistinguishable (6 wins, 8
losses, 62 ties; 95% CI on the difference [−0.023, +0.004]).

With a labeler configured, auto mode calls the LLM when the model found
**0.5 headings per page or fewer**, which is where it adds most: that routes
43% of documents and buys about three quarters of the gain. `--llm-density`
moves the threshold (`0` never escalates; `1.5` maximises quality and escalates
on nearly everything, i.e. `--llm`).

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
| `VERIFICATION_MODEL` | server-side LLM; **unset by default — the server runs none** |
| `PDF_BOOKMARKER_LOCAL_N_GPU_LAYERS` | GPU offload for a local model (`-1` = all) |
| `OCR_MAX_PAGES` | reject scanned PDFs longer than this (default 50) |

The labeler is loaded and validated once at startup: a path that cannot be used
stops the server there, rather than failing every upload with a message about a
model the user never asked for. Replacing the file on disk is picked up without
a restart.

**The deployed server runs no LLM of its own.** The heading model produces the
outline; verification happens only when a caller supplies their own API key.
On a CPU host the local GGUF costs minutes per document for roughly 3 title F1,
which is not a trade worth making by default. Setting `VERIFICATION_MODEL`
turns it back on — worthwhile on a GPU host — in which case the model is
*checked* at startup but not loaded, and a missing file is a warning rather
than a failure. (The frontend's "This server runs the heading model only" note
assumes the default; update it if you enable server-side verification.)

    # in a second terminal
    cd frontend
    npm install
    npm run dev   # proxies /api to :8000
