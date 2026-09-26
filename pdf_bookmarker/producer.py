"""Tell LaTeX-produced PDFs from the rest by their metadata.

The heading models are population-specific: the LaTeX-trained labeler scores
0.8009 title F1 on arXiv/NIST but 0.4221 on web PDFs, where font heuristics
beat it. This is the switch the pipeline uses to pick a model per document,
and the filter `training/fetch_ccpdf.py` uses to skip LaTeX at fetch time, so
the two cannot disagree about what counts as LaTeX.
"""
import re

# pdfTeX, XeTeX, LuaHBTeX, dvips, dvipdfmx, and arXiv's rewritten
# "arXiv GenPDF (tex2pdf:...)" — without tex2pdf, 66 of the 76 evaluation
# documents read as non-LaTeX. "tex\b" so "Textmaker" is not one.
_TEX_RE = re.compile(r"tex\b|tex2pdf|latex|dvips|dvipdfm", re.IGNORECASE)


def is_tex(producer: str) -> bool:
    """True if a producer/creator string names a TeX toolchain."""
    return bool(_TEX_RE.search(producer))


def document_is_tex(doc) -> bool:
    """True if an open PyMuPDF document was produced by TeX.

    Reads both fields: pdfTeX writes Producer, while arXiv's rebuild and many
    LaTeX front ends ("LaTeX with hyperref") only fill Creator.
    """
    meta = doc.metadata or {}
    return is_tex(f"{meta.get('producer') or ''} {meta.get('creator') or ''}")
