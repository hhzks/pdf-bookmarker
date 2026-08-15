"""Score the shipped pipeline end-to-end at several auto-mode routing thresholds.

This is the harness behind the README's labeler/routing table. It runs the real
`pipeline.process_pdf` — real extractor, real labeler, real locator, real merge
— over the evaluation set, and replays the LLM from cached predictions instead
of calling one. That keeps the run free and deterministic, and it is the only
way the routed configurations can be re-measured after a labeler change without
re-paying for a corpus of LLM calls.

The replay is faithful to where it matters: `process_pdf` still decides *which*
documents to escalate (`llm.is_sparse_outline` on the labeler's own output), and
the entries the fake backend returns still go through `locator.locate_entries`
and `merge.merge_outlines`. Only the network call is substituted.

Usage:
    python training/route_check.py records_test_eval_76.jsonl \\
        --labeler ~/models/labeler.joblib \\
        --llm-predictions preds_gguf_qwen35.jsonl \\
        --thresholds 0 0.5 1.5

`--thresholds 0` is the labeler alone (auto mode never escalates at 0 headings
per page); any positive threshold routes the documents at or below it. Add
`--always` for the `--llm` row, which unions on every document regardless of
density.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdf_bookmarker import llm, pipeline
from pdf_bookmarker.models import OutlineEntry

from evaluate import score_outline


class ReplayBackend:
    """An LLMBackend that returns one document's cached predictions.

    Page numbers are dropped deliberately. The cached `printed_page` is the
    model's own guess, and keeping it would let a wrong guess survive location
    as an offset-corrected fallback — the pipeline places these entries itself,
    which is what it does with a real backend's output too.
    """

    def __init__(self, entries: list[dict]):
        self._entries = entries

    def parse_outline(self, context: str) -> list[OutlineEntry]:
        return [
            OutlineEntry(title=e["title"], level=e["level"], printed_page=None)
            for e in self._entries
        ]


def run_config(
    records: list[dict],
    predictions: dict[str, list[dict]],
    labeler_path: str,
    *,
    llm_mode: str,
    density: float,
) -> dict:
    """Run every record through process_pdf under one routing configuration."""
    current: list[dict] = []

    def fake_get_backend(spec, api_key=None, **options):
        return ReplayBackend(current)

    real_get_backend = llm.get_backend
    llm.get_backend = fake_get_backend
    scores, routed, failed = [], 0, []
    try:
        for record in records:
            current = predictions.get(record["sha256"], [])
            try:
                result = pipeline.process_pdf(
                    record["file"],
                    None,  # dry run: detect only
                    llm_mode=llm_mode,
                    labeler_path=labeler_path,
                    llm_density=density,
                    # decide_llm short-circuits to True on a key rather than
                    # checking env vars for a provider we never reach.
                    api_key="replay",
                )
            except Exception as exc:
                failed.append((record["file"], str(exc)))
                continue
            routed += bool(result.used_llm)
            pred = [
                {"title": e.title, "level": e.level, "printed_page": e.printed_page}
                for e in result.entries
            ]
            scores.append(score_outline(pred, record["entries"], True))
    finally:
        llm.get_backend = real_get_backend

    def avg(key):
        values = [s[key] for s in scores if s[key] is not None]
        return sum(values) / len(values) if values else None

    return {
        "documents": len(scores),
        "routed": routed,
        "routed_share": routed / len(scores) if scores else 0.0,
        "f1": avg("f1"),
        "precision": avg("precision"),
        "recall": avg("recall"),
        "level_accuracy": avg("level_accuracy"),
        "failed": failed,
        # Kept per document because the macro averages do not settle anything
        # at this corpus size -- a routing change has to be paired to be
        # believed. level_accuracy is None when a document matched no titles,
        # and stays None here rather than becoming a zero.
        "per_document_f1": [s["f1"] for s in scores],
        "per_document_level": [s["level_accuracy"] for s in scores],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path, help="harvest records.jsonl (the gold)")
    parser.add_argument("--labeler", required=True, help="path to labeler.joblib")
    parser.add_argument("--llm-predictions", type=Path, required=True,
                        help="JSONL of {sha256, entries} to replay as the LLM")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.0, 0.5, 1.5],
                        help="auto-mode --llm-density values to score (0 = never)")
    parser.add_argument("--always", action="store_true",
                        help="also score llm_mode=always (the --llm row)")
    parser.add_argument("--no-labeler", action="store_true",
                        help="also score the LLM with no labeler configured, "
                        "which is the 'LLM alone' row: process_pdf replaces the "
                        "heuristic outline with the located LLM one rather than "
                        "merging, so this is not the same as scoring the cached "
                        "predictions directly")
    parser.add_argument("--json", type=Path, help="write full results here")
    args = parser.parse_args(argv)

    records = [
        json.loads(l)
        for l in args.records.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    predictions = {
        p["sha256"]: p["entries"]
        for p in (
            json.loads(l)
            for l in args.llm_predictions.read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    }
    missing = [r["sha256"] for r in records if r["sha256"] not in predictions]
    if missing:
        print(f"warning: {len(missing)} records have no cached LLM prediction; "
              "they replay as an empty outline", file=sys.stderr)

    configs = [(f"auto @ {t}", "auto", t, args.labeler) for t in args.thresholds]
    if args.always:
        configs.append(("always (--llm)", "always", 0.0, args.labeler))
    if args.no_labeler:
        configs.append(("LLM alone", "always", 0.0, None))

    results = {}
    for name, mode, density, labeler in configs:
        started = time.time()
        print(f"running {name} ...", file=sys.stderr, flush=True)
        result = run_config(
            records, predictions, labeler, llm_mode=mode, density=density
        )
        result["seconds"] = round(time.time() - started, 1)
        results[name] = result
        if result["documents"]:
            print(
                f"  {name:>16}  routed {result['routed']:>3}/{result['documents']}"
                f" ({result['routed_share']:.0%})  F1 {result['f1']:.4f}"
                f"  P {result['precision']:.4f}  R {result['recall']:.4f}"
                f"  level {result['level_accuracy']:.4f}  [{result['seconds']}s]",
                file=sys.stderr, flush=True,
            )
        else:
            print(f"  {name}: every document failed", file=sys.stderr)
        for path, exc in result["failed"]:
            print(f"    failed: {path}: {exc}", file=sys.stderr)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(
        {k: {m: v[m] for m in
             ("documents", "routed", "routed_share", "f1", "precision",
              "recall", "level_accuracy")}
         for k, v in results.items()},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
