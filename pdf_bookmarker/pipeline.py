"""Shared processing pipeline used by the CLI and the web backend."""
import os
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from . import extractor, heading_detector, llm, locator, toc_detector, writer
from .extractor import Line
from .models import OutlineEntry


class PipelineError(Exception):
    """Base class for pipeline failures."""


class InvalidPdfError(PipelineError):
    """The file does not exist or cannot be opened as a PDF."""


class EncryptedPdfError(PipelineError):
    """The PDF requires a password."""


class NoTextLayerError(PipelineError):
    """The PDF has no extractable text (scanned image)."""


_NO_TEXT_MESSAGE = "no extractable text layer (scanned PDF? OCR is not supported yet)"


class ExistingBookmarksError(PipelineError):
    """The PDF already has an outline and replace_existing is False."""


class NoOutlineError(PipelineError):
    """No outline could be detected."""

    def __init__(self, message: str, warnings: list[str] | None = None):
        super().__init__(message)
        self.warnings = warnings or []


class LLMVerificationError(PipelineError):
    """The LLM pass failed and llm_mode was "always"."""


@dataclass
class PipelineResult:
    entries: list[OutlineEntry]
    bookmark_count: int  # 0 when output_path is None (dry run)
    used_llm: bool
    used_toc: bool
    warnings: list[str] = field(default_factory=list)


def process_pdf(
    input_path: Path | str,
    output_path: Path | str | None,
    *,
    llm_mode: str = "auto",  # "auto" | "always" | "never"
    model_spec: str = llm.DEFAULT_MODEL_SPEC,
    api_key: str | None = None,
    replace_existing: bool = True,
) -> PipelineResult:
    """Detect an outline in input_path and write it to output_path.

    output_path=None is a dry run: detect only, write nothing.
    Raises a PipelineError subclass (or llm.UnknownProviderError) on failure.
    """
    if llm_mode not in ("auto", "always", "never"):
        raise ValueError(f"llm_mode must be auto, always or never, not {llm_mode!r}")

    try:
        doc = fitz.open(input_path)
    except Exception as exc:
        raise InvalidPdfError(f"cannot open {input_path}: {exc}") from exc
    try:
        if doc.needs_pass:
            raise EncryptedPdfError("PDF is encrypted")
        if doc.get_toc() and not replace_existing:
            raise ExistingBookmarksError("PDF already has bookmarks")
        if not extractor.has_text_layer(doc):
            raise NoTextLayerError(_NO_TEXT_MESSAGE)

        lines = extractor.extract_lines(doc)
        if not lines:
            raise NoTextLayerError(_NO_TEXT_MESSAGE)
        entries, failures, used_toc, toc_pages = build_outline(lines, doc.page_count)

        warnings: list[str] = []
        used_llm = False
        run_llm, warning = decide_llm(
            llm_mode, api_key, entries, failures, used_toc, doc.page_count, model_spec
        )
        if warning:
            warnings.append(warning)
        if run_llm:
            try:
                backend = llm.get_backend(model_spec, api_key=api_key)
                llm_entries = backend.parse_outline(build_llm_context(lines, toc_pages))
                entries, failures = locator.locate_entries(
                    llm_entries, lines, skip_pages=set(toc_pages)
                )
                used_llm = True
            except llm.UnknownProviderError:
                raise
            except Exception as exc:
                if llm_mode == "always":
                    raise LLMVerificationError(f"LLM verification failed: {exc}") from exc
                warnings.append(f"LLM call failed ({exc}); using heuristic outline")

        if not entries:
            raise NoOutlineError("no outline could be detected", warnings)

        count = 0
        if output_path is not None:
            count = writer.write_outline(doc, entries, str(output_path))
        return PipelineResult(entries, count, used_llm, used_toc, warnings)
    finally:
        doc.close()


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
    llm_mode: str,
    api_key: str | None,
    entries: list[OutlineEntry],
    failures: int,
    used_toc: bool,
    page_count: int,
    model_spec: str,
) -> tuple[bool, str | None]:
    """Returns (run_llm, warning)."""
    if llm_mode == "never":
        return False, None
    if llm_mode == "always":
        return True, None
    levels = [e.level for e in entries]
    if not llm.is_low_confidence(len(entries), failures, used_toc, levels, page_count):
        return False, None
    if api_key:
        return True, None
    # Pre-check only covers shipped providers (llm.ENV_KEYS); unknown ones
    # surface missing-key errors via the auto-mode exception path.
    key_names = llm.ENV_KEYS.get(model_spec.partition(":")[0])
    if key_names and not any(os.environ.get(name) for name in key_names):
        return False, (
            f"outline confidence is low but {key_names[0]} is not set; "
            "continuing without LLM"
        )
    return True, None


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
