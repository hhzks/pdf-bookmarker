"""Shared data types."""
from dataclasses import dataclass


@dataclass
class OutlineEntry:
    title: str
    level: int                       # 1-based nesting depth (1 = chapter)
    page: int | None = None          # 0-based physical page index (set by locator)
    y: float | None = None           # vertical position of the heading on the page
    printed_page: int | None = None  # page number as printed in the TOC (1-based)
