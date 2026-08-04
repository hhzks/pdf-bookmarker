"""Featurization and outline assembly for the line-labeling outline model.

Shared by the trainer and the predictor so that the features a model was fit
on are exactly the features it is served — the same "train == serve" property
that build_dataset.py gives the generative path.

The features are deliberately layout-first: relative font size, weight,
indentation, whitespace above, and the shape of any leading section label.
These are the signals the font heuristics in heading_detector.py already use,
made continuous and learnable instead of thresholded. Text semantics are
handled separately (by an encoder over the line text); keeping the two apart
makes it measurable how much each contributes.
"""
import re

_NUMBERED = re.compile(r"^(\d+(\.\d+)*|[A-Za-z](\.\d+)+|[IVXLivxl]{1,5})[.\s]\s*\S")
_ALL_CAPS = re.compile(r"^[^a-z]*[A-Z][^a-z]*$")
_DIGITS = re.compile(r"\d+")


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
    return [
        float(row.get("size_ratio", 1.0)),
        1.0 if row.get("bold") else 0.0,
        float(row.get("words", len(text.split()))),
        float(len(stripped)),
        float(row.get("gap_above", 0.0)) / size,
        float(row.get("x", 0.0)),
        float(row.get("y", 0.0)) / _PAGE_HEIGHT,
        float(row.get("page", 0)) / max(page_count, 1),
        1.0 if _NUMBERED.match(stripped) else 0.0,
        1.0 if stripped and _ALL_CAPS.match(stripped) else 0.0,
        1.0 if stripped.endswith(":") else 0.0,
        1.0 if stripped.endswith(".") else 0.0,
        sum(c.isdigit() for c in stripped) / len(stripped) if stripped else 0.0,
    ]


def entries_from_labels(rows: list[dict], labels: list[int]) -> list[dict]:
    """Read an outline off per-line labels, in document order.

    Titles are the line's own text, so they are exact by construction — the
    property that motivated this whole approach. `page` is the 0-based
    physical page, which is what the bookmark actually needs; the generative
    path has to recover it through the locator.
    """
    entries = []
    for row, label in zip(rows, labels):
        if not label:
            continue
        title = row["text"].strip()
        if not title:
            continue
        entries.append({"title": title, "level": int(label), "page": row["page"]})
    return entries
