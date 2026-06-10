# pdf-bookmarker — Design

**Date:** 2026-06-10
**Status:** Approved

## Overview

A Python CLI tool that adds a hierarchical bookmark outline to text-based PDFs.

Pipeline: **detect TOC → parse entries → locate sections in body → write bookmarks**, with heuristic heading detection as a fallback when no TOC exists, and an LLM as a verification/rescue layer when heuristic confidence is low.

## Requirements

- Input: a text-based PDF (has an extractable text layer). Scanned/image PDFs are rejected with a clear error; the architecture keeps a seam for adding OCR later.
- If the PDF contains a table of contents, parse it, preserving chapter/subchapter hierarchy, and link each bookmark to the actual location of that section in the document (found by searching for the heading text, not by trusting printed page numbers).
- If no TOC exists, detect chapters/subchapters heuristically from font size, weight, and numbering patterns, and build the outline from those.
- LLM assistance is model-agnostic: ships with an Anthropic backend, users can select other providers/models.
- Never modify the input file; write a new bookmarked copy.

## CLI

```
pdf-bookmarker input.pdf [-o output.pdf] [--llm | --no-llm] [--model PROVIDER:MODEL_ID] [--dry-run] [--force]
```

| Flag | Behavior |
|---|---|
| (default output) | `input.bookmarked.pdf` next to the original |
| `-o output.pdf` | explicit output path |
| `--dry-run` | print the detected outline tree; write nothing |
| `--force` | required to replace existing bookmarks (otherwise warn and exit) |
| `--llm` | always verify the detected outline with the LLM |
| `--no-llm` | never call the LLM |
| (neither) | auto mode: LLM is called only when heuristic confidence is low; if no API key is configured, warn and continue heuristics-only |
| `--model` | LLM backend selection, `provider:model-id` form. Default: `anthropic:claude-opus-4-8` |

Error exits: encrypted PDF; no text layer (message notes OCR is not yet supported); existing bookmarks without `--force`.

## Architecture

```
pdf_bookmarker/
  cli.py               argparse entry point, orchestrates the pipeline
  models.py            OutlineEntry (title, level, page, y-position), shared types
  extractor.py         text extraction w/ font metadata (PyMuPDF); the seam where OCR would plug in
  toc_detector.py      find TOC pages + parse entries → hierarchy from numbering/indentation
  heading_detector.py  fallback: font-size tiers, bold, numbering patterns → outline
  locator.py           match each entry's title to its actual page (printed page № as hint)
  llm.py               LLMBackend protocol + AnthropicBackend; verify/repair outline
  writer.py            doc.set_toc() with page numbers + positions, save output
```

Each module has one responsibility and communicates through `OutlineEntry` lists, so any stage can be tested or replaced independently.

## Key logic

### TOC detection (`toc_detector.py`)
- Scan the first ~30 pages (or 15% of the document, whichever is larger) for pages that look like a TOC: a "Contents" / "Table of Contents" heading, and/or many lines ending in page numbers (dot leaders or right-aligned numbers).
- Parse each entry: title text, printed page number, hierarchy level. Level comes from numbering depth (`2.4.1` → level 3) or indentation tiers when entries are unnumbered.
- Multi-page TOCs: keep consuming consecutive pages that match the TOC line pattern.

### Locating sections (`locator.py`)
- Printed page numbers rarely equal physical PDF page indices (front matter, roman numerals), so each entry's title is searched for in the body text.
- The printed page number is used as a hint: estimate the printed→physical offset from the first few successful matches, then search nearest-first around the hinted page.
- A match requires the title to appear as a standalone-ish line (not mid-paragraph), preferring larger/bold spans. The match yields physical page + y-coordinate so the bookmark lands exactly on the heading.
- Entries that cannot be located: bookmark to the offset-corrected hinted page with a warning; dropped (with a warning) if there is no usable hint.

### Heuristic fallback (`heading_detector.py`)
- Cluster font sizes across the document to find the body-text size.
- Heading candidates: lines noticeably larger than body text or bold, short (< ~12 words), standalone, optionally matching numbering patterns ("Chapter 3", "3.2 Title", roman numerals).
- Outline levels from font-size tiers, refined by numbering depth when present.

### Confidence scoring
Low confidence (triggers LLM in auto mode) when any of:
- TOC was detected but parsing produced fewer than 3 entries
- More than 20% of entries failed to locate in the body
- Fallback path found 0 headings, or produced incoherent levels (e.g., level jumps > 1, or a single level for a 300-page book)

### LLM layer (`llm.py`)
- `LLMBackend` protocol: one method, `parse_outline(context: str) -> list[OutlineEntry]`, taking raw TOC text or heading candidates (with font metadata) and returning a structured outline.
- `AnthropicBackend` implements it with the official `anthropic` SDK using `client.messages.parse()` and a Pydantic outline schema (validated structured output). Default model `claude-opus-4-8`; reads `ANTHROPIC_API_KEY` from the environment.
- Backend selection via `--model provider:model-id`, resolved through a registry dict — adding a new provider is one small class implementing the protocol plus a registry entry. Unknown provider → clear error listing available providers.
- LLM output is advisory: returned entries still pass through `locator.py`, so the LLM cannot invent page destinations.
- API errors in auto mode → warn and continue with the heuristic result; in `--llm` mode → exit with the error.

## Error handling summary

| Condition | Behavior |
|---|---|
| Encrypted PDF | exit with error |
| No text layer | exit with error mentioning OCR not yet supported |
| Existing bookmarks, no `--force` | warn and exit |
| Entry can't be located | bookmark to offset-corrected hint + warning, or drop + warning |
| LLM API error (auto mode) | warn, continue heuristics-only |
| LLM API error (`--llm`) | exit with error |
| Unknown `--model` provider | exit with error listing providers |

## Testing

- pytest; synthetic PDF fixtures generated programmatically with PyMuPDF:
  1. clean numbered TOC
  2. dotted-leader TOC with front-matter offset (printed ≠ physical pages)
  3. no TOC, styled headings (font-size hierarchy)
  4. no text layer (drawn-only pages) → expect the OCR error
  5. PDF that already has bookmarks → expect `--force` behavior
- Unit tests per module (TOC line parsing, level inference, font clustering, locator offset estimation).
- `llm.py` tested with a fake backend and mocked Anthropic responses; one optional live integration test gated behind an env var.

## Dependencies

- Python 3.12
- `pymupdf` (extraction, TOC write), `anthropic` (default LLM backend), `pydantic` (structured output schema)
- Packaged with `pyproject.toml`; `pip install -e .` exposes the `pdf-bookmarker` console script.

## Out of scope (for now)

- OCR for scanned PDFs (architecture leaves `extractor.py` as the seam)
- Non-Anthropic backend implementations (interface + registry ship; implementations are user-added)
- GUI
