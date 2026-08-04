"""Featurization and outline assembly for the line-labeling outline model.

Lives in the package, not in training/, because the trainer and the serving
path must compute features from one implementation — the same "train == serve"
property that build_dataset.py gives the generative path. training/ imports
these; nothing here imports training/.

The features are deliberately layout-first: relative font size, weight,
indentation, whitespace above, and the shape of any leading section label.
These are the signals the font heuristics in heading_detector.py already use,
made continuous and learnable instead of thresholded. Text semantics are
handled separately (by an encoder over the line text); keeping the two apart
makes it measurable how much each contributes.
"""
import re

from .extractor import Line
from .heading_detector import body_text_size
from .locator import section_label

_ALL_CAPS = re.compile(r"^[^a-z]*[A-Z][^a-z]*$")
_DIGITS = re.compile(r"\d+")


def number_depth(text: str) -> int:
    """How deep a line's section label goes: "4" 1, "4.1" 2, "2.2.2" 3, none 0.

    Depth separates a subsection from a list item — "4." opens either, "4.1.2"
    only ever opens a heading — and it is the same evidence the level
    classifier needs. The label itself comes from locator.section_label, so
    this cannot drift from the rule the locator and the metric apply.
    """
    label = section_label(text)
    if not label:
        return 0
    return len([part for part in label.rstrip(".").split(".") if part])


def text_for_model(text: str) -> str:
    """Normalized line text for the lexical model.

    Digits collapse to a single token so that "1 Introduction" and
    "7 Introduction" are one piece of evidence rather than two rare ones —
    the *shape* of a section label is already a layout feature
    (`starts_numbered`), so the specific number only fragments the vocabulary.
    """
    return " ".join(_DIGITS.sub("#", text.lower()).split())

FEATURE_NAMES = [
    "size_ratio",       # font size relative to the document's body size
    "bold",
    "words",
    "chars",
    "gap_ems",          # whitespace above, in ems of this line's own size
    "x",                # left edge: indentation tracks nesting depth
    "y_frac",           # height down the page
    "page_frac",        # how far into the document
    "starts_numbered",  # "3.1 Results", "A.2 Method", "IV. Discussion"
    "number_depth",     # 4 -> 1, 4.1 -> 2, 2.2.2 -> 3; also evidence of level
    "all_caps",
    "ends_colon",
    "ends_period",      # prose ends in a full stop; headings rarely do
    "digit_frac",
]

_PAGE_HEIGHT = 792.0  # US Letter; only used to scale y into roughly [0, 1]


def feature_vector(row: dict, page_count: int) -> list[float]:
    """Numeric features for one line. Order matches FEATURE_NAMES."""
    text = row["text"]
    stripped = text.strip()
    size = row.get("size") or 1.0
    depth = number_depth(stripped)
    return [
        float(row.get("size_ratio", 1.0)),
        1.0 if row.get("bold") else 0.0,
        float(row.get("words", len(text.split()))),
        float(len(stripped)),
        float(row.get("gap_above", 0.0)) / size,
        float(row.get("x", 0.0)),
        float(row.get("y", 0.0)) / _PAGE_HEIGHT,
        float(row.get("page", 0)) / max(page_count, 1),
        1.0 if depth else 0.0,
        float(depth),
        1.0 if stripped and _ALL_CAPS.match(stripped) else 0.0,
        1.0 if stripped.endswith(":") else 0.0,
        1.0 if stripped.endswith(".") else 0.0,
        sum(c.isdigit() for c in stripped) / len(stripped) if stripped else 0.0,
    ]


def line_features(lines: list[Line]) -> list[dict]:
    """Per-line feature rows, parallel to lines.

    Sizes are expressed relative to the document's dominant body size so the
    model sees "twice body text" rather than "20pt", which is what actually
    distinguishes a heading across documents typeset at different scales.
    """
    if not lines:
        return []
    body = body_text_size(lines) or 1.0
    rows = []
    for i, line in enumerate(lines):
        previous = lines[i - 1] if i else None
        same_page = previous is not None and previous.page == line.page
        rows.append(
            {
                "text": line.text,
                "page": line.page,
                "x": round(line.x, 2),
                "y": round(line.y, 2),
                "size": round(line.size, 2),
                "bold": line.bold,
                "size_ratio": round(line.size / body, 4),
                "gap_above": round(line.y - previous.y, 2) if same_page else 0.0,
                "words": len(line.text.split()),
            }
        )
    return rows


def entries_from_labels(rows: list[dict], labels: list[int]) -> list[dict]:
    """Read an outline off per-line labels, in document order.

    Titles are the line's own text, so they are exact by construction — the
    property that motivated this whole approach. `page` is the 0-based
    physical page, which is what the bookmark actually needs; the generative
    path has to recover it through the locator.
    """
    entries = []
    for index, (row, label) in enumerate(zip(rows, labels)):
        if not label:
            continue
        title = row["text"].strip()
        if not title:
            continue
        # `index` lets a caller recover the row this came from without
        # reproducing the filtering above and drifting out of step with it.
        entries.append({
            "title": title,
            "level": int(label),
            "page": row["page"],
            "index": index,
        })
    return entries
