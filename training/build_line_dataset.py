"""Turn bookmarked PDFs into per-line labels for a sequence labeler.

The generative path asks a model to *write* the outline. This builds data for
the alternative: classify every extracted line as `0` (not a heading) or its
heading level, then read the outline straight off the labels. Titles are then
exact by construction and there is no unparseable output.

Measured recall ceilings on the fixed 77-document test set, against the
generative model's 0.7535:

    per-line labels          0.8162 macro / 0.8606 micro
    BIO spans (<= 3 lines)   0.8533 / 0.8856
    substring offsets        0.9364 / 0.9534

Note the labeler sees *every* extracted line, which is why it clears the
generative model while pointer output into `build_llm_context`'s filtered
400-line context does not (0.7469, below current recall).

Labels come from `doc.get_toc()` — the same free supervision harvest.py uses —
aligned onto extracted lines by title, constrained to the page the bookmark
points at. That page constraint is what keeps a TOC row on the contents page
from being labeled instead of the heading it refers to.

Usage:
    python training/build_line_dataset.py PDF_DIR -o lines.jsonl
    python training/build_line_dataset.py PDF_DIR -o lines.jsonl --max-level 4

Output: one JSON object per document,
    {"file", "sha256", "page_count", "lines": [{text, page, x, y, size, bold,
     size_ratio, gap_above, words, label}, ...], "aligned", "unaligned"}
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from pdf_bookmarker import extractor
from pdf_bookmarker.extractor import Line
from pdf_bookmarker.heading_detector import body_text_size
from pdf_bookmarker.locator import strip_section_numbers

# How far from the page a bookmark points at we will look for its heading.
# Bookmarks routinely land a page early or late; anything further away is a
# different section that happens to share a title.
_PAGE_WINDOW = 1
_MAX_LEVEL = 4


def _key(text: str) -> str:
    """Normalized, section-label-insensitive form used to match a title."""
    stripped = strip_section_numbers(text.strip())
    return " ".join("".join(c for c in stripped.lower() if c.isalnum() or c.isspace()).split())


def align_labels(
    lines: list[Line],
    toc: list,
    max_level: int = _MAX_LEVEL,
) -> tuple[list[int], dict[str, int]]:
    """Label each line with its heading level, or 0.

    toc is doc.get_toc(): (level, title, page) with page 1-based *physical*.
    Each bookmark claims at most one line, searched on the page it points at
    and then outward, so repeated titles ("Summary" in three chapters) take
    different lines instead of all collapsing onto the first.
    """
    labels = [0] * len(lines)
    by_page: dict[int, list[int]] = {}
    for i, line in enumerate(lines):
        by_page.setdefault(line.page, []).append(i)

    keys = [_key(line.text) for line in lines]
    # Both keys always present: a caller reading stats["aligned"] should not
    # have to guard against a document where nothing aligned.
    stats: Counter[str] = Counter({"aligned": 0, "unaligned": 0})
    for entry in toc:
        level, title, page = entry[0], entry[1], entry[2]
        target = _key(title)
        if not target:
            stats["unaligned"] += 1
            continue
        # get_toc pages are 1-based; a broken bookmark can be 0 or -1.
        hint = max(page - 1, 0)
        index = _find(keys, labels, by_page, target, hint)
        if index is None:
            stats["unaligned"] += 1
            continue
        labels[index] = min(max(level, 1), max_level)
        stats["aligned"] += 1
    return labels, dict(stats)


def _find(
    keys: list[str],
    labels: list[int],
    by_page: dict[int, list[int]],
    target: str,
    hint: int,
) -> int | None:
    """First unclaimed line matching target, searching outward from hint."""
    for delta in range(_PAGE_WINDOW + 1):
        for page in dict.fromkeys((hint + delta, hint - delta)):
            for i in by_page.get(page, []):
                if labels[i] == 0 and keys[i] == target:
                    return i
    return None


def line_features(lines: list[Line]) -> list[dict]:
    """Per-line features, parallel to lines.

    Sizes are expressed relative to the document's dominant body size so the
    model sees "twice body text" rather than "20pt", which is what actually
    distinguishes a heading across documents typeset at different scales.
    """
    if not lines:
        return []
    body = body_text_size(lines) or 1.0
    rows = []
    for i, line in enumerate(lines):
        previous = lines[i - 1] if i else None
        same_page = previous is not None and previous.page == line.page
        rows.append(
            {
                "text": line.text,
                "page": line.page,
                "x": round(line.x, 2),
                "y": round(line.y, 2),
                "size": round(line.size, 2),
                "bold": line.bold,
                "size_ratio": round(line.size / body, 4),
                "gap_above": round(line.y - previous.y, 2) if same_page else 0.0,
                "words": len(line.text.split()),
            }
        )
    return rows


def build_document(path: Path | str, max_level: int = _MAX_LEVEL) -> tuple[dict | None, str | None]:
    """Turn one PDF into a labeled line sequence, or (None, skip_reason)."""
    path = Path(path)
    try:
        doc = fitz.open(path)
    except Exception:
        return None, "unreadable"
    try:
        if doc.needs_pass:
            return None, "encrypted"
        toc = doc.get_toc()
        if not toc:
            return None, "no-embedded-outline"
        if not extractor.has_text_layer(doc):
            return None, "no-text-layer"
        lines = extractor.extract_lines(doc)
        if not lines:
            return None, "no-text-layer"
        labels, stats = align_labels(lines, toc, max_level)
        if not stats.get("aligned"):
            return None, "no-alignment"
        rows = line_features(lines)
        for row, label in zip(rows, labels):
            row["label"] = label
        return {
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "page_count": doc.page_count,
            "lines": rows,
            "aligned": stats.get("aligned", 0),
            "unaligned": stats.get("unaligned", 0),
        }, None
    finally:
        doc.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf_dir", type=Path, help="directory scanned recursively for *.pdf")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--max-level", type=int, default=_MAX_LEVEL,
                        help="levels deeper than this are clamped (default 4)")
    args = parser.parse_args(argv)

    pdfs = sorted(args.pdf_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"no PDFs found under {args.pdf_dir}", file=sys.stderr)
        return 1

    skips: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for n, pdf in enumerate(pdfs, 1):
            try:
                record, reason = build_document(pdf, args.max_level)
            except Exception as exc:  # a corrupt PDF must not kill the run
                skips[f"error:{type(exc).__name__}"] += 1
                continue
            if record is None:
                skips[reason] += 1
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            totals["lines"] += len(record["lines"])
            totals["aligned"] += record["aligned"]
            totals["unaligned"] += record["unaligned"]
            print(f"\r{n}/{len(pdfs)} documents", end="", file=sys.stderr)

    print(file=sys.stderr)
    print(f"wrote {written} documents to {args.output}", file=sys.stderr)
    gold = totals["aligned"] + totals["unaligned"]
    if gold:
        print(f"  bookmarks aligned onto a line: {totals['aligned']}/{gold} "
              f"({totals['aligned'] / gold:.1%})", file=sys.stderr)
    if totals["lines"]:
        print(f"  lines: {totals['lines']} "
              f"({totals['aligned'] / totals['lines']:.2%} positive)", file=sys.stderr)
    for reason, count in skips.most_common():
        print(f"  skipped {count}: {reason}", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
