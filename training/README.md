# Training data tooling

Scripts for building a labeled dataset to fine-tune a small local model that
implements the `LLMBackend.parse_outline` contract (see `pdf_bookmarker/llm.py`).

Not part of the installed package — run from the repo root.

## 1. Fetch a seed corpus

```bash
python training/fetch_arxiv.py --query "cat:math.LO" --max 25 -o corpus/arxiv
```

arXiv PDFs compiled with `hyperref` often carry an embedded outline; the ones
that don't are filtered out at harvest time, so over-fetching is cheap. The
fetcher sleeps 3s between requests (arXiv API guidelines), skips files it
already has, and appends provenance to `corpus/arxiv/manifest.jsonl`.

```bash
python training/fetch_nist.py -o corpus/nist
```

NIST SP 800-series publications (public domain) are richly bookmarked with
formal printed TOCs — the best printed_page training data. There is no
directory listing, so the fetcher probes the predictable nvlpubs URL space
(highest revision first); 404s are expected.

You can also drop PDFs from any other source into a directory — harvesting is
source-agnostic. Sources checked and rejected: govinfo CFR volumes carry no
embedded outline (fail the harvest gate); OAPEN's legacy REST API no longer
exposes files.

**Licensing:** the arXiv API does not report per-paper licenses. Check
`https://arxiv.org/abs/<id>` before redistributing a corpus; prefer
CC-licensed material.

## 2. Harvest training records

```bash
python training/harvest.py corpus/ -o records.jsonl
```

Keeps only PDFs that are readable, unencrypted, have a text layer, and have a
non-empty embedded outline (`doc.get_toc()`) — that outline is the gold label.
For each kept PDF it emits one JSON line:

- `context` — the exact string the serving path would send to the model
  (`pipeline.build_llm_context`), either TOC text or candidate heading lines.
- `entries` — gold `{title, level, printed_page}` items in the shape of
  `llm.Outline`.
- `context_kind` — `"toc"` or `"headings"`, `alignment`, `sha256`, etc.

`printed_page` is recovered by fuzzy-aligning gold titles against the parsed
TOC rows — **not** taken from `get_toc()`, whose page numbers are physical
indices, not printed ones. TOC-path documents whose outline aligns with fewer
than `--min-alignment` (default 0.6) of TOC rows are dropped as label noise.

## 3. Build the SFT dataset

```bash
python training/build_dataset.py records.jsonl [silver.jsonl ...] -o dataset/
```

Dedups by `sha256`, splits by document (deterministically — a doc's split is
derived from its hash, so re-running with more data never moves an existing
doc between splits), and writes `dataset/{train,val,test}.jsonl` of
`{"prompt", "completion", "meta"}`. The prompt is `llm.PROMPT` with the
record's context and the completion is `llm.Outline` JSON — the exact
serving format, so train == serve by construction.

## 4. Evaluate

```bash
python training/evaluate.py records.jsonl --backend heuristic     # baseline
python training/evaluate.py records.jsonl --predictions preds.jsonl
```

Macro-averaged title F1, level accuracy on matched titles, and printed-page
accuracy. The heuristic baseline (the non-LLM pipeline path) is the score any
fine-tuned model must beat. To evaluate a model, run it over the records'
contexts and write `{"sha256", "entries"}` lines.

## 5. Distill silver labels (optional, costs API money)

```bash
python training/distill.py corpus/ -o silver.jsonl --limit 20
```

Bookmarked corpora over-represent the TOC path, so the heading-candidate path
is data-starved. This runs a shipped backend (default `anthropic`, needs the
API key in the env) over PDFs with **no** embedded outline and records its
outline as a silver label (`"silver": true`) in the harvest record shape —
`build_dataset.py` consumes it directly. Resumable; one LLM call per PDF.

## 6. Fine-tune (QLoRA)

```bash
pip install -r training/requirements.txt   # torch/trl/peft — training only, never app deps
python training/finetune.py dataset/ -o checkpoints/outline-lora
```

QLoRA (4-bit NF4) over the SFT splits; defaults to `Qwen/Qwen2.5-1.5B-Instruct`,
LoRA r=16 on all linear layers, completion-only loss, 2 epochs. Needs a CUDA
GPU (~6 GB VRAM); on CPU pass `--no-4bit` (slow — use a smaller `--base-model`).
Trains on the raw prompt/completion text (no chat template) because the planned
`local:` backend will prompt with raw `llm.PROMPT` text too — train == serve.

Then evaluate: generate outlines for `test.jsonl` contexts with the adapter,
write `{"sha256", "entries"}` lines, and score with `evaluate.py` against the
heuristic baseline.

## Remaining (not yet built)

- Merge the adapter + GGUF export, and a `local:` backend in
  `pdf_bookmarker/llm.py` with grammar-constrained decoding (llama.cpp GBNF
  from the `llm.Outline` schema).
- End-to-end page-placement metric via `locator.locate_entries`.
