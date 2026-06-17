"""Command-line entry point — argument parsing around pipeline.process_pdf."""
import argparse
import sys
from pathlib import Path

from . import llm, pipeline
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
    parser.add_argument("--ocr", choices=("auto", "force", "never"), default="auto",
                        help="OCR scanned PDFs: auto (when no text layer), force, "
                             "or never (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the detected outline without writing")
    parser.add_argument("--force", action="store_true",
                        help="replace existing bookmarks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    llm_mode = "always" if args.llm else ("never" if args.no_llm else "auto")
    out_path = None if args.dry_run else (
        args.output or args.input.with_suffix(".bookmarked.pdf")
    )

    try:
        result = pipeline.process_pdf(
            args.input,
            out_path,
            llm_mode=llm_mode,
            model_spec=args.model,
            replace_existing=args.force,
            ocr_mode=args.ocr,
        )
    except pipeline.ExistingBookmarksError:
        print("error: PDF already has bookmarks; use --force to replace them",
              file=sys.stderr)
        return 2
    except (pipeline.NoOutlineError, pipeline.LLMVerificationError) as exc:
        for warning in getattr(exc, "warnings", []):
            print(f"warning: {warning}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (pipeline.PipelineError, llm.UnknownProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if args.dry_run:
        print_outline(result.entries)
        return 0
    print(f"wrote {result.bookmark_count} bookmarks to {out_path}")
    return 0


def print_outline(entries: list[OutlineEntry]) -> None:
    try:  # Windows consoles (cp1252) cannot encode every glyph; never crash
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass  # captured/redirected stdout may not support reconfigure
    for e in entries:
        page = "?" if e.page is None else e.page + 1
        print(f"{'  ' * (e.level - 1)}{e.title}  (page {page})")
