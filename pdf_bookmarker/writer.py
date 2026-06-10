"""Write the outline into a copy of the PDF."""
import fitz

from .models import OutlineEntry


def sanitize_levels(entries: list[OutlineEntry]) -> list[OutlineEntry]:
    """PDF outlines must start at level 1 and never jump by more than +1."""
    prev = 0
    for entry in entries:
        if entry.level > prev + 1:
            entry.level = prev + 1
        prev = entry.level
    return entries


def write_outline(doc: fitz.Document, entries: list[OutlineEntry], out_path: str) -> int:
    """Apply entries as the document outline and save to out_path.

    Returns the number of bookmarks written. Entries without a page are skipped.
    """
    entries = sanitize_levels([e for e in entries if e.page is not None])
    toc = []
    for e in entries:
        item = [e.level, e.title, e.page + 1]  # set_toc wants 1-based pages
        if e.y is not None:
            item.append({"kind": fitz.LINK_GOTO, "to": fitz.Point(0, e.y), "zoom": 0})
        toc.append(item)
    doc.set_toc(toc)
    doc.save(out_path)
    return len(toc)
