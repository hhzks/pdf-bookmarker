# pdf-bookmarker

Add a hierarchical bookmark outline to text-based PDFs. Parses the table of
contents when one exists (preserving chapter/subchapter structure and linking
each bookmark to the section's real location), and falls back to font-based
heading detection when there is no TOC. An optional LLM pass verifies or
repairs low-confidence outlines.

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

The LLM layer is provider-agnostic: implement `pdf_bookmarker.llm.LLMBackend`
and register the class in `_BACKENDS` to add another provider.

## Limitations

- Text-based PDFs only. Scanned PDFs (no text layer) are rejected; OCR is a
  planned extension (`extractor.py` is the seam).
- Encrypted PDFs are not supported.

## Development

    pip install -e ".[dev]"
    python -m pytest
