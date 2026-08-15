"""Shared processing pipeline used by the CLI and the web backend."""
import os
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from . import (
    extractor,
    heading_detector,
    labeler as labeler_module,
    llm,
    locator,
    merge,
    ocr,
    toc_detector,
    writer,
)
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


class OcrUnavailableError(PipelineError):
    """OCR was needed or requested but the Tesseract binary is not available."""


class OcrPageLimitError(PipelineError):
    """The document has more pages than the configured OCR page cap."""


_NO_TEXT_MESSAGE = "no extractable text layer (scanned PDF; enable OCR to read it)"


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
    used_ocr: bool = False
    warnings: list[str] = field(default_factory=list)
    used_labeler: bool = False  # appended last: callers construct positionally


def process_pdf(
    input_path: Path | str,
    output_path: Path | str | None,
    *,
    llm_mode: str = "auto",  # "auto" | "always" | "never"
    model_spec: str = llm.DEFAULT_MODEL_SPEC,
    api_key: str | None = None,
    replace_existing: bool = True,
    ocr_mode: str = "auto",  # "auto" | "force" | "never"
    ocr_max_pages: int | None = None,
    labeler_path: str | Path | None = None,
    llm_density: float = llm.SPARSE_ENTRIES_PER_PAGE,
) -> PipelineResult:
    """Detect an outline in input_path and write it to output_path.

    output_path=None is a dry run: detect only, write nothing.
    llm_density is the auto-mode routing threshold for the labeler path, in
    headings per page (0 never escalates on it).
    Raises a PipelineError subclass (or llm.UnknownProviderError) on failure.
    """
    if llm_mode not in ("auto", "always", "never"):
        raise ValueError(f"llm_mode must be auto, always or never, not {llm_mode!r}")
    if ocr_mode not in ("auto", "force", "never"):
        raise ValueError(f"ocr_mode must be auto, force or never, not {ocr_mode!r}")

    try:
        doc = fitz.open(input_path)
    except Exception as exc:
        raise InvalidPdfError(f"cannot open {input_path}: {exc}") from exc
    try:
        if doc.needs_pass:
            raise EncryptedPdfError("PDF is encrypted")
        if doc.get_toc() and not replace_existing:
            raise ExistingBookmarksError("PDF already has bookmarks")
        has_text = extractor.has_text_layer(doc)
        use_ocr = ocr_mode == "force" or (ocr_mode == "auto" and not has_text)
        warnings: list[str] = []
        if use_ocr:
            if not ocr.available():
                raise OcrUnavailableError(
                    "OCR is required to read this PDF but Tesseract is not available"
                )
            if ocr_max_pages is not None and doc.page_count > ocr_max_pages:
                raise OcrPageLimitError(
                    f"document has {doc.page_count} pages; OCR limit is {ocr_max_pages}"
                )
            if has_text:
                warnings.append("forced OCR: the existing text layer was ignored")
            lines = ocr.extract_lines_via_ocr(doc)
            if not lines:
                raise NoTextLayerError("OCR found no readable text in this scanned PDF")
            used_ocr = True
        else:
            if not has_text:
                raise NoTextLayerError(_NO_TEXT_MESSAGE)
            lines = extractor.extract_lines(doc)
            if not lines:
                raise NoTextLayerError(_NO_TEXT_MESSAGE)
            used_ocr = False
        model = resolve_labeler(labeler_path)
        used_labeler = False
        entries = []
        if model is not None:
            # The labeler beats the heuristic path outright (0.8009 vs 0.6208
            # title F1) and its entries already carry an exact physical page,
            # so there is nothing for the locator to do and no TOC to parse.
            toc_pages = toc_detector.find_toc_pages(lines, doc.page_count)
            entries = model.detect(lines, doc.page_count)
            failures, used_toc, used_labeler = 0, False, True
        if not entries:
            # Nothing detected — either no labeler, or one that found nothing on
            # this document. Falling through to the heuristics is strictly
            # better than returning no outline at all, which is what a
            # confident-but-silent labeler would otherwise cause.
            entries, failures, used_toc, toc_pages = build_outline(lines, doc.page_count)
            used_labeler = False

        used_llm = False
        run_llm, warning = decide_llm(
            llm_mode, api_key, entries, failures, used_toc, doc.page_count, model_spec,
            used_labeler=used_labeler,
            density_threshold=llm_density,
        )
        if warning:
            warnings.append(warning)
        if run_llm:
            try:
                backend = llm.get_backend(model_spec, api_key=api_key)
                llm_entries = backend.parse_outline(build_llm_context(lines, toc_pages))
                located, failures = locator.locate_entries(
                    llm_entries, lines, skip_pages=set(toc_pages)
                )
                if used_labeler:
                    # Two precise detectors that agree on only 31% of proposed
                    # titles, so the union trades precision for recall:
                    # 0.8009 -> 0.8211 at the default routing (+0.0202, CI
                    # [+0.0000, +0.0418]) and 0.8312 with --llm, recall
                    # 0.7664 -> 0.8470. The labeler leads because its pages are
                    # exact. Re-run training/route_check.py before quoting
                    # these; they do not survive a labeler change.
                    #
                    # This merge is deliberately NOT applied to the heuristic
                    # outline. Merging that in instead costs 6.1 F1, because its
                    # precision (0.606) drags the result down -- the union needs
                    # both sources to be precise.
                    entries = merge.merge_outlines(entries, located)
                else:
                    entries = located
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
        return PipelineResult(
            entries,
            count,
            used_llm,
            used_toc,
            used_ocr,
            warnings=warnings,
            used_labeler=used_labeler,
        )
    finally:
        doc.close()


_LABELER_ENV = "PDF_BOOKMARKER_LABELER"

# Last model loaded, as ((path, mtime, size), model). One slot: a deployment
# serves one model, and a second path simply evicts the first.
_CACHED_LABELER: tuple[tuple[str, int, int], object] | None = None


def resolve_labeler(labeler_path: str | Path | None):
    """Load the line-labeling detector, or None when none is configured.

    Off unless asked for: an install with no model behaves exactly as before.
    The env var exists so a deployment can enable it without every caller
    threading a path through.

    The bundle is cached across calls — the web worker resolves once per job,
    and unpickling a 1.7 MB model per PDF is pure latency. The cache key
    includes the file's mtime and size, so replacing the model on disk takes
    effect without a restart. Two threads racing here both load and one wins
    the slot; either model is valid, so the store stays lock-free like the rest
    of the worker path.
    """
    global _CACHED_LABELER
    path = labeler_path or os.environ.get(_LABELER_ENV)
    if not path:
        return None
    path = Path(path)
    try:
        info = path.stat()
    except OSError:
        return labeler_module.Labeler.load(path)  # missing/unreadable: let it say so
    key = (str(path), info.st_mtime_ns, info.st_size)
    cached = _CACHED_LABELER
    if cached is not None and cached[0] == key:
        return cached[1]
    model = labeler_module.Labeler.load(path)
    _CACHED_LABELER = (key, model)
    return model


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
    *,
    used_labeler: bool = False,
    density_threshold: float = llm.SPARSE_ENTRIES_PER_PAGE,
) -> tuple[bool, str | None]:
    """Returns (run_llm, warning)."""
    if llm_mode == "never":
        return False, None
    if llm_mode == "always":
        return True, None
    if used_labeler:
        # A different outline needs a different question. The structural checks
        # below read location failures and level jumps, neither of which the
        # labeler produces; what predicts its need for help is how little it
        # found. See llm.is_sparse_outline for the measured curve.
        needs_llm = llm.is_sparse_outline(len(entries), page_count, density_threshold)
    else:
        levels = [e.level for e in entries]
        needs_llm = llm.is_low_confidence(
            len(entries), failures, used_toc, levels, page_count
        )
    if not needs_llm:
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
