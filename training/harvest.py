"""Harvest LLM training records from already-bookmarked PDFs.

Any PDF whose embedded outline (doc.get_toc()) is non-empty is a free labeled
example: the input context is built exactly the way the serving path builds it
(pipeline.build_llm_context), and the embedded outline supplies gold
titles/levels. For documents with a detected TOC, printed page numbers are
recovered by aligning the gold titles against the parsed TOC rows — never from
get_toc(), whose pages are 1-based *physical* indices, not printed numbers.

Usage:
    python training/harvest.py PDF_DIR -o records.jsonl
    python training/harvest.py PDF_DIR -o records.jsonl --min-pages 8 --min-alignment 0.7

Each output line is one JSON record:
    {
      "file": "corpus/foo.pdf",
      "sha256": "...",
      "page_count": 214,
      "context_kind": "toc" | "headings",
      "context": "<exact string the model will see>",
      "entries": [{"title": ..., "level": ..., "printed_page": ...}, ...],
      "alignment": 0.94        # null for headings-path docs
    }

`entries` matches the shape of llm.Outline, so dataset construction is just
prompt formatting.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from pdf_bookmarker import extractor, pipeline, toc_detector

_MATCH_THRESHOLD = 0.85  # normalized-title similarity for a gold<->TOC-row match


def normalize_title(text: str) -> str:
    """Normalize a heading title for fuzzy comparison."""
    return " ".join(text.strip().strip(".").lower().split())


def align_printed_pages(
    gold_titles: list[str], toc_entries: list
) -> tuple[list[int | None], float]:
    """Match each gold outline title to a parsed TOC row to recover its printed
    page number. Greedy in-order matching; each TOC row is used at most once.

    Returns (printed_pages parallel to gold_titles, matched fraction).
    """
    if not gold_titles:
        return [], 0.0
    normalized_rows = [normalize_title(e.title) for e in toc_entries]
    used: set[int] = set()
    printed: list[int | None] = []
    matched = 0
    for title in gold_titles:
        target = normalize_title(title)
        best_index, best_score = None, _MATCH_THRESHOLD
        for i, row in enumerate(normalized_rows):
            if i in used:
                continue
            score = SequenceMatcher(None, target, row).ratio()
            if score > best_score or (score == best_score and best_index is None):
                best_index, best_score = i, score
        if best_index is None:
            printed.append(None)
        else:
            used.add(best_index)
            matched += 1
            printed.append(toc_entries[best_index].printed_page)
    return printed, matched / len(gold_titles)


def harvest_pdf(
    path: Path | str,
    *,
    min_pages: int = 4,
    min_alignment: float = 0.6,
    augment_headings: bool = False,
) -> tuple[list[dict] | None, str | None]:
    """Turn one PDF into training record(s).

    Returns (records, None) on success or (None, skip_reason) when the document
    is unusable as a training example. With augment_headings, a TOC-path
    document yields a second, synthetic record whose context is built as if
    the TOC pages did not exist — real gold labels for the data-starved
    heading-candidate path. Synthetic records share the parent's sha256 (same
    split, no leakage) and are marked context_kind "headings-synthetic".
    """
    path = Path(path)
    try:
        doc = fitz.open(path)
    except Exception:
        return None, "unreadable"
    try:
        if doc.needs_pass:
            return None, "encrypted"
        gold = doc.get_toc()
        if not gold:
            return None, "no-embedded-outline"
        if doc.page_count < min_pages:
            return None, "too-short"
        if not extractor.has_text_layer(doc):
            return None, "no-text-layer"
        lines = extractor.extract_lines(doc)
        if not lines:
            return None, "no-text-layer"

        toc_pages = toc_detector.find_toc_pages(lines, doc.page_count)
        context = pipeline.build_llm_context(lines, toc_pages)
        gold_titles = [title for _, title, _ in gold]
        gold_levels = [level for level, _, _ in gold]

        if toc_pages:
            parsed = toc_detector.parse_toc(lines, toc_pages)
            printed, alignment = align_printed_pages(gold_titles, parsed)
            if alignment < min_alignment:
                return None, "poor-alignment"
            kind = "toc"
        else:
            # Heading-candidate path: the model never sees printed page
            # numbers here, so there is nothing to align.
            printed = [None] * len(gold)
            alignment = None
            kind = "headings"

        base = {
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "page_count": doc.page_count,
        }
        records = [
            {
                **base,
                "context_kind": kind,
                "context": context,
                "entries": [
                    {"title": title, "level": level, "printed_page": page}
                    for title, level, page in zip(gold_titles, gold_levels, printed)
                ],
                "alignment": alignment,
            }
        ]

        if augment_headings and toc_pages:
            toc_page_set = set(toc_pages)
            body_lines = [l for l in lines if l.page not in toc_page_set]
            if body_lines:
                records.append(
                    {
                        **base,
                        "context_kind": "headings-synthetic",
                        # No toc_pages -> build_llm_context takes the
                        # heading-candidate branch over the body lines.
                        "context": pipeline.build_llm_context(body_lines, []),
                        "entries": [
                            {"title": t, "level": l, "printed_page": None}
                            for t, l in zip(gold_titles, gold_levels)
                        ],
                        "alignment": None,
                    }
                )
        return records, None
    finally:
        doc.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf_dir", type=Path, help="directory scanned recursively for *.pdf")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output JSONL path")
    parser.add_argument("--min-pages", type=int, default=4)
    parser.add_argument(
        "--min-alignment",
        type=float,
        default=0.6,
        help="drop TOC-path docs whose gold outline matches fewer than this "
        "fraction of parsed TOC rows (label-noise guard)",
    )
    parser.add_argument(
        "--augment-headings",
        action="store_true",
        help="also emit a synthetic heading-candidate record per TOC-path doc "
        "(context built as if the TOC pages did not exist)",
    )
    args = parser.parse_args(argv)

    pdfs = sorted(args.pdf_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"no PDFs found under {args.pdf_dir}", file=sys.stderr)
        return 1

    skips: Counter[str] = Counter()
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for pdf in pdfs:
            records, reason = harvest_pdf(
                pdf,
                min_pages=args.min_pages,
                min_alignment=args.min_alignment,
                augment_headings=args.augment_headings,
            )
            if records is None:
                skips[reason] += 1
                continue
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"wrote {written} records to {args.output}", file=sys.stderr)
    for reason, count in skips.most_common():
        print(f"  skipped {count}: {reason}", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
