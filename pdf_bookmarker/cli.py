"""Command-line entry point and pipeline orchestration."""
import argparse
import os
import sys
from pathlib import Path

import fitz

from . import extractor, heading_detector, llm, locator, toc_detector, writer
from .extractor import Line
from .models import OutlineEntry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-bookmarker",
        description="Add a hierarchical bookmark outline to a text-based PDF.",
    )
    parser.add_argument("input", type=Path, help="input PDF")
    parser.add_argument("-o", "--output", type=Path,
                        help="output path (default: <input>.bookmarked.pdf)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--llm", action="store_true",
                      help="always verify the outline with the LLM")
    mode.add_argument("--no-llm", action="store_true", help="never call the LLM")
    parser.add_argument("--model", default=llm.DEFAULT_MODEL_SPEC,
                        help="LLM backend as PROVIDER:MODEL_ID (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the detected outline without writing")
    parser.add_argument("--force", action="store_true",
                        help="replace existing bookmarks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        doc = fitz.open(args.input)
    except Exception as exc:
        print(f"error: cannot open {args.input}: {exc}", file=sys.stderr)
        return 2
    if doc.needs_pass:
        print("error: PDF is encrypted", file=sys.stderr)
        return 2
    if doc.get_toc() and not args.force:
        print("error: PDF already has bookmarks; use --force to replace them",
              file=sys.stderr)
        return 2
    if not extractor.has_text_layer(doc):
        print("error: no extractable text layer (scanned PDF? OCR is not supported yet)",
              file=sys.stderr)
        return 2

    lines = extractor.extract_lines(doc)
    entries, failures, used_toc, toc_pages = build_outline(lines, doc.page_count)

    if decide_llm(args, entries, failures, used_toc, doc.page_count):
        try:
            backend = llm.get_backend(args.model)
            context = build_llm_context(lines, toc_pages)
            llm_entries = backend.parse_outline(context)
            entries, failures = locator.locate_entries(
                llm_entries, lines, skip_pages=set(toc_pages)
            )
        except ValueError as exc:  # unknown provider
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            if args.llm:
                print(f"error: LLM verification failed: {exc}", file=sys.stderr)
                return 1
            print(f"warning: LLM call failed ({exc}); using heuristic outline",
                  file=sys.stderr)

    if not entries:
        print("error: no outline could be detected", file=sys.stderr)
        return 1

    if args.dry_run:
        print_outline(entries)
        return 0

    out_path = args.output or args.input.with_suffix(".bookmarked.pdf")
    count = writer.write_outline(doc, entries, str(out_path))
    print(f"wrote {count} bookmarks to {out_path}")
    return 0


def build_outline(
    lines: list[Line], page_count: int
) -> tuple[list[OutlineEntry], int, bool, list[int]]:
    """Run TOC detection with heading-detection fallback.

    Returns (entries, location_failures, used_toc, toc_pages).
    """
    toc_pages = toc_detector.find_toc_pages(lines, page_count)
    entries = toc_detector.parse_toc(lines, toc_pages) if toc_pages else []
    if entries:
        located, failures = locator.locate_entries(
            entries, lines, skip_pages=set(toc_pages)
        )
        return located, failures, True, toc_pages
    # Fallback: headings already carry page/y, no location step needed.
    return heading_detector.detect_headings(lines), 0, False, toc_pages


def decide_llm(
    args: argparse.Namespace,
    entries: list[OutlineEntry],
    failures: int,
    used_toc: bool,
    page_count: int,
) -> bool:
    if args.no_llm:
        return False
    if args.llm:
        return True
    levels = [e.level for e in entries]
    if not llm.is_low_confidence(len(entries), failures, used_toc, levels, page_count):
        return False
    # Key pre-check is Anthropic-specific by design (the only shipped backend);
    # other providers surface missing-key errors via the auto-mode exception path.
    if args.model.startswith("anthropic") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("warning: outline confidence is low but ANTHROPIC_API_KEY is not set; "
              "continuing without LLM", file=sys.stderr)
        return False
    return True


def build_llm_context(lines: list[Line], toc_pages: list[int]) -> str:
    if toc_pages:
        toc_page_set = set(toc_pages)
        toc_text = "\n".join(l.text for l in lines if l.page in toc_page_set)
        return f"Table of contents text:\n{toc_text}"
    body = heading_detector.body_text_size(lines)
    candidates = [
        f"physical_page={l.page} size={l.size:.1f} bold={l.bold} text={l.text!r}"
        for l in lines
        if l.size >= body * 1.1 or l.bold
    ]
    return (
        f"Candidate heading lines (body text size {body:.1f}; physical_page is "
        f"0-based, not a printed page number):\n"
        + "\n".join(candidates[:400])  # cap keeps the prompt within a sane token budget
    )


def print_outline(entries: list[OutlineEntry]) -> None:
    try:  # Windows consoles (cp1252) cannot encode every glyph; never crash
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass  # captured/redirected stdout may not support reconfigure
    for e in entries:
        page = "?" if e.page is None else e.page + 1
        print(f"{'  ' * (e.level - 1)}{e.title}  (page {page})")
