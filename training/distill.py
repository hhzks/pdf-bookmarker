"""Distill a frontier LLM backend into silver training records.

Bookmarked corpora over-represent the TOC path (PDFs with embedded outlines
usually also print a TOC), so the heading-candidate path is data-starved.
This fills the gap: run a shipped backend (anthropic/gemini) over PDFs that
have NO embedded outline, and record its parsed outline as a silver label in
the same JSONL shape harvest.py emits — so build_dataset.py consumes both.

COSTS MONEY: every kept PDF is one LLM call. Start small (--limit).
Requires the provider's API key in the environment (see llm.ENV_KEYS).

Usage:
    python training/distill.py corpus/ -o silver.jsonl --limit 20
    python training/distill.py corpus/ -o silver.jsonl --model gemini:gemini-3.5-flash
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from pdf_bookmarker import extractor, llm, pipeline, toc_detector


def distill_pdf(path: Path, backend: llm.LLMBackend, *, min_pages: int = 4) -> tuple[dict | None, str | None]:
    """One PDF -> one silver record, mirroring harvest.harvest_pdf's contract.

    Only PDFs WITHOUT an embedded outline are used (the ones harvest.py can't),
    and the teacher's output is the label — no alignment to compute.
    """
    try:
        doc = fitz.open(path)
    except Exception:
        return None, "unreadable"
    try:
        if doc.needs_pass:
            return None, "encrypted"
        if doc.get_toc():
            return None, "has-embedded-outline"  # harvest.py's job, not ours
        if doc.page_count < min_pages:
            return None, "too-short"
        if not extractor.has_text_layer(doc):
            return None, "no-text-layer"
        lines = extractor.extract_lines(doc)
        if not lines:
            return None, "no-text-layer"

        toc_pages = toc_detector.find_toc_pages(lines, doc.page_count)
        context = pipeline.build_llm_context(lines, toc_pages)
        entries = backend.parse_outline(context)
        if not entries:
            return None, "teacher-empty"
        return {
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "page_count": doc.page_count,
            "context_kind": "toc" if toc_pages else "headings",
            "context": context,
            "entries": [
                {"title": e.title, "level": e.level, "printed_page": e.printed_page}
                for e in entries
            ],
            "alignment": None,
            "silver": True,  # distilled label, not ground truth
        }, None
    finally:
        doc.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf_dir", type=Path, help="directory scanned recursively for *.pdf")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output JSONL (appended)")
    parser.add_argument("--model", default=llm.DEFAULT_MODEL_SPEC, help="provider:model-id")
    parser.add_argument("--limit", type=int, default=20, help="max LLM calls this run")
    parser.add_argument("--min-pages", type=int, default=4)
    args = parser.parse_args(argv)

    backend = llm.get_backend(args.model)
    done = set()
    if args.output.exists():  # resumable: skip PDFs already distilled
        with open(args.output, encoding="utf-8") as f:
            done = {json.loads(line)["sha256"] for line in f}

    skips: Counter[str] = Counter()
    written = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for pdf in sorted(args.pdf_dir.rglob("*.pdf")):
            if written >= args.limit:
                break
            if hashlib.sha256(pdf.read_bytes()).hexdigest() in done:
                skips["already-distilled"] += 1
                continue
            record, reason = distill_pdf(pdf, backend, min_pages=args.min_pages)
            if record is None:
                skips[reason] += 1
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"distilled {pdf.name} ({len(record['entries'])} entries)", file=sys.stderr)

    print(f"wrote {written} silver records to {args.output}", file=sys.stderr)
    for reason, count in skips.most_common():
        print(f"  skipped {count}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
