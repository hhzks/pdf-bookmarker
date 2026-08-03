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

**Use `--ignore-section-numbers` for model-vs-model comparisons.** Titles match
by exact string by default, but gold comes from PDFs' embedded bookmarks, which
usually omit the section numbering the printed heading shows — only ~20% of gold
entries carry a leading number. A model that copies the heading as printed
(what `llm.PROMPT` asks for) is then charged both a false positive and a false
negative for each one. This bias is large enough to invert a verdict: it made a
4.7-point win look like a 2.6-point loss. See
`comparison-qwen35-2b-vs-v3.md`.

Predictions must also be generated at **the same quantization the adapter was
trained at** — `predict.py --no-4bit` against a 4-bit QLoRA adapter cost 8.3 F1
on an otherwise identical run.

## 5. Distill silver labels (optional)

Bookmarked corpora over-represent the TOC path, so the heading-candidate path
is data-starved. This runs a backend over PDFs with **no** embedded outline and
records its outline as a silver label (`"silver": true`) in the harvest record
shape — `build_dataset.py` consumes it directly. Resumable; one call per PDF.

**The teacher must be stronger than the student.** Distillation transfers what
the teacher knows and the student does not, so a local teacher has to be a
*larger* model than the one being trained. Pointing this at the shipped
`outline.gguf` is self-distillation — it adds no knowledge and promotes the
student's own errors to labels.

Local teacher (free, offline, nothing leaves the machine):

```bash
python training/distill.py corpus/ -o silver.jsonl --limit 0 \
    --model "local:models/Qwen3.5-9B-Q5_K_M.gguf" \
    --chat --n-gpu-layers -1 --n-ctx 8192
```

`--chat` is **required for an off-the-shelf instruct teacher**: without it the
model gets the bare `PROMPT` via raw completion, which is what the fine-tuned
student was trained on but not what an instruct model expects. `--no-think`
additionally appends the Qwen3-family `/no_think` switch; grammar-constrained
decoding already blocks `<think>` blocks, so this only asks the model not to
plan them.

`--n-gpu-layers -1` offloads everything to the GPU and **needs a CUDA build of
llama-cpp-python** — the default PyPI wheel is CPU-only. Check with
`llama_cpp.llama_supports_gpu_offload()`; if it prints False, install from
`https://abetlen.github.io/llama-cpp-python/whl/<cuXXX>`, matching the CUDA
**runtime** you have installed (`nvcc --version`), not the higher version
`nvidia-smi` reports — that is driver capability, and a wheel built for a
runtime you lack fails with `Failed to load shared library 'llama.dll'`.

Sizing on a 12 GB card: Qwen3.5-9B at `Q5_K_M` (6.6 GB) loads to ~8.8 GB used
with an 8k context, leaving ~3.2 GB. Drop `--n-ctx` before dropping model size,
since the KV cache grows with context. If throughput collapses, check **power
draw, not utilization** — the Windows spill trap in `REMOTE.md` applies here too.
`--limit 0` means no limit, which is only sensible when calls are free.

API teacher (**costs money** — one billed call per kept PDF, so start small):

```bash
python training/distill.py corpus/ -o silver.jsonl --limit 20
python training/distill.py corpus/ -o silver.jsonl --model gemini:gemini-3.5-flash
```

### Quality gate

Silver labels are unverified, so every record is checked against the locator: a
title the locator cannot find in the body text is very likely hallucinated.
Records where more than `--max-unlocatable` (default 0.2) of the teacher's
titles fail to locate are dropped and counted as `teacher-unlocatable`. Pass
`1.0` to disable the gate.

After a run, check the `context_kind` mix in the output — the point of this step
is `headings` records. A corpus full of PDFs that print a TOC without embedding
one yields `toc` records, which you already have plenty of.

Skip reasons worth reading: `teacher-unlocatable` is the gate rejecting likely
hallucination, `context-too-long` means the prompt exceeded the teacher's
`--n-ctx`, and `teacher-error` is any other backend failure. All three skip a
single PDF — a failing teacher never ends the run.

### Silver labels are train-only

`build_dataset.py` routes any record with `"silver": true` to the **train split
only**, the same way it treats `headings-synthetic`, and reports the count as
`silver-skipped-eval`. Distilled labels are a teacher's output, not ground
truth: scoring against them would measure agreement with the teacher rather
than correctness. Evaluation stays on harvested gold records.

## 6. Fine-tune (QLoRA)

```bash
pip install -r training/requirements.txt   # torch/trl/peft — training only, never app deps
python training/finetune.py dataset/ -o checkpoints/outline-lora
```

QLoRA (4-bit NF4) over the SFT splits; defaults to `Qwen/Qwen3.5-2B`,
LoRA r=16 on all linear layers, completion-only loss, 2 epochs. Needs a CUDA
GPU (~7 GB VRAM); on CPU pass `--no-4bit` (slow — use a smaller `--base-model`).

Qwen3.5 has no `-Instruct` suffix — `Qwen/Qwen3.5-2B` is the post-trained model,
`Qwen/Qwen3.5-2B-Base` the raw pretrained one. Its thinking mode is irrelevant
here: `<think>` lives in the chat template, and this path never applies one.
**The same default is duplicated in `predict.py` and `export_gguf.py`** —
predicting with, or merging into, a different base than the adapter was trained
on produces garbage silently, so change all three together (a test guards it).

Switching base model invalidates existing checkpoints and the measured context
ceiling: 5120 was established for the 1.5B, and a 2B needs more VRAM per token.
Re-check the first stderr line (`train: N (dropped M over-long)`) and watch for
the spill trap before trusting a long run.
Trains on the raw prompt/completion text (no chat template) because the planned
`local:` backend will prompt with raw `llm.PROMPT` text too — train == serve.

Then evaluate: generate outlines for `test.jsonl` contexts with the adapter,
write `{"sha256", "entries"}` lines, and score with `evaluate.py` against the
heuristic baseline.

## 7. Export for serving

```bash
python training/export_gguf.py checkpoints/outline-lora-v2 -o models/outline.gguf
pdf-bookmarker input.pdf --llm --model "local:models/outline.gguf"
```

Merges the adapter into the base model (peft) and converts to GGUF (q8_0 by
default, ~1.7 GB) via llama.cpp's converter, shallow-cloned automatically.
The `local:` backend in `pdf_bookmarker/llm.py` runs it with llama-cpp-python
and grammar-constrained decoding — output is forced to the `llm.Outline` JSON
schema, eliminating parse failures.

## Remaining (not yet built)

- End-to-end page-placement metric via `locator.locate_entries`.
