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

You can also drop PDFs from any other source (OpenStax, govinfo, NIST, DOAB…)
into a directory — harvesting is source-agnostic.

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
  `llm._Outline`.
- `context_kind` — `"toc"` or `"headings"`, `alignment`, `sha256`, etc.

`printed_page` is recovered by fuzzy-aligning gold titles against the parsed
TOC rows — **not** taken from `get_toc()`, whose page numbers are physical
indices, not printed ones. TOC-path documents whose outline aligns with fewer
than `--min-alignment` (default 0.6) of TOC rows are dropped as label noise.

## Next steps (not yet built)

- Dedup by `sha256`; split by document into train/val/test.
- Format records as `llm._PROMPT` + `_Outline` JSON for SFT (train == serve).
- Distill the frontier backend over no-bookmark PDFs to cover the
  heading-candidate path, which is underrepresented in bookmarked corpora.
- Eval harness: entry-set F1, level accuracy, end-to-end page placement via
  `locator.locate_entries`.
