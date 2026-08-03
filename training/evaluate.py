"""Score predicted outlines against harvested gold records.

Metrics per document, macro-averaged over the set:
  - title F1        — precision/recall on normalized titles (greedy in-order match)
  - level accuracy  — of the matched titles, how many have the right level
  - page accuracy   — of the matched titles with a gold printed_page, how many
                      predicted it exactly (TOC-path docs only)

Two prediction sources:
  --backend heuristic   re-run the non-LLM pipeline (build_outline) on each
                        record's PDF — the baseline any model must beat
  --predictions X.jsonl lines of {"sha256": ..., "entries": [...]} from any
                        model you are evaluating

Usage:
    python training/evaluate.py records.jsonl --backend heuristic
    python training/evaluate.py records.jsonl --predictions preds.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitz

from harvest import normalize_title
from pdf_bookmarker import extractor, locator, pipeline


# Gold entries come from PDFs' embedded bookmarks, which usually omit the
# numbering that the printed heading shows — but only usually, so the same
# section can appear as "Introduction" in one document and "1 Introduction" in
# another. Comparing without it measures whether a model found the right
# heading rather than which of two defensible conventions it copied. The
# locator needs exactly the same rule to match an entry to its heading, so the
# implementation lives in the package and is shared rather than duplicated.
strip_section_numbers = locator.strip_section_numbers


def match_entries(
    pred: list[dict], gold: list[dict], ignore_section_numbers: bool = False
) -> list[tuple[dict, dict]]:
    """Greedy in-order matching on normalized titles; each side used once."""

    def key(title: str) -> str:
        return normalize_title(
            strip_section_numbers(title) if ignore_section_numbers else title
        )

    pairs = []
    used: set[int] = set()
    for g in gold:
        target = key(g["title"])
        for i, p in enumerate(pred):
            if i not in used and key(p["title"]) == target:
                used.add(i)
                pairs.append((p, g))
                break
    return pairs


def score_outline(
    pred: list[dict], gold: list[dict], ignore_section_numbers: bool = False
) -> dict:
    """Score one document. pred/gold: [{title, level, printed_page}, ...]."""
    pairs = match_entries(pred, gold, ignore_section_numbers)
    matched = len(pairs)
    precision = matched / len(pred) if pred else 0.0
    recall = matched / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    level_ok = sum(1 for p, g in pairs if p["level"] == g["level"])
    paged = [(p, g) for p, g in pairs if g.get("printed_page") is not None]
    page_ok = sum(1 for p, g in paged if p.get("printed_page") == g["printed_page"])
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "level_accuracy": level_ok / matched if matched else None,
        "page_accuracy": page_ok / len(paged) if paged else None,
    }


def heuristic_predict(pdf_path: str) -> list[dict]:
    """Run the non-LLM pipeline path and shape its output like gold entries."""
    doc = fitz.open(pdf_path)
    try:
        lines = extractor.extract_lines(doc)
        entries, _, _, _ = pipeline.build_outline(lines, doc.page_count)
    finally:
        doc.close()
    return [
        {"title": e.title, "level": e.level, "printed_page": e.printed_page}
        for e in entries
    ]


def evaluate(
    records: list[dict],
    predictions: dict[str, list[dict]],
    ignore_section_numbers: bool = False,
) -> dict:
    """Macro-average scores over records; skips records with no prediction."""
    scores = []
    skipped = 0
    for record in records:
        pred = predictions.get(record["sha256"])
        if pred is None:
            skipped += 1
            continue
        scores.append(
            score_outline(pred, record["entries"], ignore_section_numbers)
        )

    def avg(key):
        values = [s[key] for s in scores if s[key] is not None]
        return sum(values) / len(values) if values else None

    return {
        "documents": len(scores),
        "skipped": skipped,
        "precision": avg("precision"),
        "recall": avg("recall"),
        "f1": avg("f1"),
        "level_accuracy": avg("level_accuracy"),
        "page_accuracy": avg("page_accuracy"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path, help="harvest records.jsonl (the gold)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--backend", choices=["heuristic"])
    source.add_argument("--predictions", type=Path, help="JSONL of {sha256, entries}")
    parser.add_argument("--ignore-section-numbers", action="store_true",
                        help="match titles ignoring a leading section label "
                        "(\"1. Introduction\" == \"Introduction\"). Gold comes from "
                        "embedded bookmarks, which usually strip the numbering the "
                        "printed heading shows, so the strict metric penalises a "
                        "model for copying the page faithfully")
    args = parser.parse_args(argv)

    records = [json.loads(l) for l in args.records.read_text(encoding="utf-8").splitlines()]

    if args.backend == "heuristic":
        predictions = {}
        for record in records:
            try:
                predictions[record["sha256"]] = heuristic_predict(record["file"])
            except Exception as exc:
                print(f"prediction failed for {record['file']}: {exc}", file=sys.stderr)
    else:
        predictions = {
            p["sha256"]: p["entries"]
            for p in (
                json.loads(l)
                for l in args.predictions.read_text(encoding="utf-8").splitlines()
            )
        }

    result = evaluate(records, predictions, args.ignore_section_numbers)
    print(json.dumps(result, indent=2))
    return 0 if result["documents"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
