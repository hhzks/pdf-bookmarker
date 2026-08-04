"""Distill a larger teacher model into silver training records.

Bookmarked corpora over-represent the TOC path (PDFs with embedded outlines
usually also print a TOC), so the heading-candidate path is data-starved.
This fills the gap: run a shipped backend over PDFs that have NO embedded
outline, and record its parsed outline as a silver label in the same JSONL
shape harvest.py emits — so build_dataset.py consumes both.

Teacher choice matters. Distillation only transfers what the teacher knows
that the student does not, so the teacher must be materially stronger than the
model being trained. Pointing this at the shipped outline.gguf is
self-distillation: it cannot add knowledge, and it promotes the student's own
errors to labels.

Two ways to run it:

  local (free, offline) — a larger GGUF than the one being trained, e.g. a
  14B at q4_K_M. No API key, no per-call cost, so --limit 0 is reasonable.
  Set --n-ctx to suit the teacher; the default fits the 1.5B student.

    python training/distill.py corpus/ -o silver.jsonl --limit 0 \
        --model "local:models/teacher.gguf" --n-ctx 8192

  api (COSTS MONEY) — every kept PDF is one billed call, so start small.
  Requires the provider's API key in the environment (see llm.ENV_KEYS).

    python training/distill.py corpus/ -o silver.jsonl --limit 20
    python training/distill.py corpus/ -o silver.jsonl --model gemini:gemini-3.5-flash

Silver labels are unverified, so every record is gated on the locator — see
distill_pdf and --max-unlocatable.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from pdf_bookmarker import extractor, llm, locator, pipeline, toc_detector
from pdf_bookmarker.models import OutlineEntry

DEFAULT_MAX_UNLOCATABLE = 0.2  # matches llm.is_low_confidence's failure threshold

# llama.cpp's wording when the prompt does not fit n_ctx; the API backends have
# their own, so anything unrecognised is reported generically rather than guessed at.
_CONTEXT_OVERFLOW = re.compile(r"exceed\w*\s+context\s+window", re.I)


def _failure_reason(exc: Exception) -> str:
    """Classify a teacher failure. One bad PDF must never end the run."""
    if _CONTEXT_OVERFLOW.search(str(exc)):
        return "context-too-long"
    return "teacher-error"


def distill_pdf(
    path: Path,
    backend: llm.LLMBackend,
    *,
    min_pages: int = 4,
    max_unlocatable: float = DEFAULT_MAX_UNLOCATABLE,
) -> tuple[dict | None, str | None]:
    """One PDF -> one silver record, mirroring harvest.harvest_pdf's contract.

    Only PDFs WITHOUT an embedded outline are used (the ones harvest.py can't),
    and the teacher's output is the label — no alignment to compute.

    Silver labels are unverified, so the teacher's outline is gated on the
    locator: a title that cannot be found in the body text is very likely
    hallucinated, and a record with more than max_unlocatable of those is
    dropped rather than trained on.
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
        try:
            entries = backend.parse_outline(context)
        except Exception as exc:  # noqa: BLE001 - one PDF must not end the run
            return None, _failure_reason(exc)
        if not entries:
            return None, "teacher-empty"
        # Validation only: locate_entries mutates what it is given, so it gets
        # copies and the recorded label keeps harvest.py's shape.
        _, failures = locator.locate_entries(
            [OutlineEntry(e.title, e.level, e.printed_page) for e in entries],
            lines,
            skip_pages=set(toc_pages),
        )
        if failures / len(entries) > max_unlocatable:
            return None, "teacher-unlocatable"
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
    parser.add_argument("--limit", type=int, default=20,
                        help="max LLM calls this run; 0 means no limit (safe for "
                        "a local teacher, which costs nothing per call)")
    parser.add_argument("--min-pages", type=int, default=4)
    parser.add_argument("--n-ctx", type=int, default=None,
                        help="context size for a local teacher; the default suits "
                        "the shipped 1.5B student, not a larger teacher")
    parser.add_argument("--n-gpu-layers", type=int, default=None,
                        help="layers to offload to the GPU for a local teacher; "
                        "-1 offloads all of them (needs a CUDA build of "
                        "llama-cpp-python). Default is CPU-only")
    parser.add_argument("--chat", action="store_true",
                        help="apply the local model's chat template. Needed for an "
                        "off-the-shelf instruct teacher; leave off for a model "
                        "fine-tuned on the bare prompt (the shipped student)")
    parser.add_argument("--no-think", action="store_true",
                        help="append the Qwen3-family /no_think switch (implies "
                        "--chat). Grammar-constrained decoding already blocks "
                        "<think> blocks; this asks the model not to plan them")
    parser.add_argument("--max-unlocatable", type=float, default=DEFAULT_MAX_UNLOCATABLE,
                        help="drop a record when more than this fraction of the "
                        "teacher's titles cannot be found in the body text "
                        f"(default {DEFAULT_MAX_UNLOCATABLE}); 1.0 disables the gate")
    args = parser.parse_args(argv)

    # Only forward what was asked for: these are local-only, and the API
    # backends reject keyword arguments they do not define.
    options = {}
    if args.n_ctx is not None:
        options["n_ctx"] = args.n_ctx
    if args.n_gpu_layers is not None:
        options["n_gpu_layers"] = args.n_gpu_layers
    if args.chat or args.no_think:
        options["chat"] = True  # /no_think is a chat-template switch
    if args.no_think:
        options["no_think"] = True
    backend = llm.get_backend(args.model, **options)
    done = set()
    if args.output.exists():  # resumable: skip PDFs already distilled
        with open(args.output, encoding="utf-8") as f:
            done = {json.loads(line)["sha256"] for line in f}

    skips: Counter[str] = Counter()
    written = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for pdf in sorted(args.pdf_dir.rglob("*.pdf")):
            if args.limit and written >= args.limit:
                break
            if hashlib.sha256(pdf.read_bytes()).hexdigest() in done:
                skips["already-distilled"] += 1
                continue
            record, reason = distill_pdf(
                pdf, backend,
                min_pages=args.min_pages,
                max_unlocatable=args.max_unlocatable,
            )
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
