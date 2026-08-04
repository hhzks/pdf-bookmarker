"""Routed union of the line labeler and the LLM.

The two detectors disagree more than they overlap — on the 76-document
evaluation set they proposed 2752 distinct titles and agreed on 62.7% — so
taking the union rather than picking one is worth +4.3 title F1 (0.7671 ->
0.8103), almost all of it recall (0.7343 -> 0.8525).

Routing decides which documents are worth an LLM call. The signal is entries
per page from the labeler alone: a sparse outline means the labeler found
little, which is where the LLM adds most. Validated by repeated 50/50 splits,
fitting the threshold on one half and scoring the other:

    LLM budget   held-out F1   gain    share of always-union
         0%         0.7659    -0.0012          -3%
        25%         0.7913    +0.0242          56%
        50%         0.7995    +0.0324          75%
       100%         0.8121    +0.0451         104%

Sublinear, so a budget buys a disproportionate share of the gain. But note
that tuning the threshold purely for quality picks ~1.5 entries/page, which
routes 94% of documents — routing is a *cost* control, not a quality one. If
compute is free, use --budget 1.0.

**The union only works when both sources are precise.** Merging the font
heuristics (precision 0.606) into the LLM's output instead *costs* 6.1 F1, as
precision collapses from 0.804 to 0.650. That is why pipeline.process_pdf
still lets the LLM replace the heuristic outline rather than merging it.

Usage:
    python training/hybrid.py records.jsonl --labeler preds_lines.jsonl \
        --llm preds_qwen35_2b.jsonl --lines lines_all.jsonl -o preds_hybrid.jsonl
    ... --budget 0.5      # send only the sparsest half to the LLM
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_bookmarker.merge import merge_outlines
from pdf_bookmarker.models import OutlineEntry


def to_entries(raw: list[dict]) -> list[OutlineEntry]:
    return [
        OutlineEntry(
            title=e["title"],
            level=e["level"],
            page=e.get("page"),
            printed_page=e.get("printed_page"),
        )
        for e in raw
    ]


def route(documents: list[dict], budget: float) -> set[str]:
    """The sha256s worth an LLM call: the sparsest `budget` fraction.

    Sparsest by entries per page, not by entry count, so a 400-page book with
    40 headings is treated as sparse while an 8-page paper with 6 is not.
    """
    if budget >= 1.0:
        return {d["sha256"] for d in documents}
    ranked = sorted(documents, key=lambda d: d["entries_per_page"])
    return {d["sha256"] for d in ranked[: round(len(ranked) * budget)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path, help="gold records, for the doc list")
    parser.add_argument("--labeler", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--lines", type=Path, required=True,
                        help="lines.jsonl, for page counts")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=1.0,
                        help="fraction of documents to send to the LLM "
                        "(default 1.0: every document, the best-quality setting)")
    args = parser.parse_args(argv)

    def load(path):
        return {
            p["sha256"]: p["entries"]
            for p in map(json.loads, path.read_text(encoding="utf-8").splitlines())
        }

    labeler, llm = load(args.labeler), load(args.llm)
    page_counts = {}
    with open(args.lines, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            page_counts[doc["sha256"]] = doc["page_count"]

    documents = []
    for record in (
        json.loads(l) for l in args.records.read_text(encoding="utf-8").splitlines()
    ):
        sha = record["sha256"]
        if sha not in labeler:
            continue
        documents.append({
            "sha256": sha,
            "entries_per_page": len(labeler[sha]) / max(page_counts.get(sha, 1), 1),
        })

    routed = route(documents, args.budget)
    added = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for doc in documents:
            sha = doc["sha256"]
            entries = to_entries(labeler[sha])
            if sha in routed and sha in llm:
                before = len(entries)
                entries = merge_outlines(entries, to_entries(llm[sha]))
                added += len(entries) - before
            out.write(json.dumps({
                "sha256": sha,
                "entries": [
                    {"title": e.title, "level": e.level,
                     "printed_page": e.printed_page, "page": e.page}
                    for e in entries
                ],
            }, ensure_ascii=False) + "\n")

    print(f"{len(documents)} documents, {len(routed)} routed to the LLM "
          f"({len(routed) / max(len(documents), 1):.0%}), "
          f"{added} entries added by the merge", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
