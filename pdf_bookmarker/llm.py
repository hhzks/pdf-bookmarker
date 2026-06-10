"""Model-agnostic LLM verification layer.

To add a provider: implement the LLMBackend protocol and register the class
in _BACKENDS. Selection is via "provider:model-id" strings (e.g. --model).
"""
from typing import Protocol

from pydantic import BaseModel

from .models import OutlineEntry

DEFAULT_MODEL_SPEC = "anthropic:claude-opus-4-8"


class LLMBackend(Protocol):
    def parse_outline(self, context: str) -> list[OutlineEntry]:
        """Parse raw TOC text / heading candidates into a structured outline."""
        ...


class _OutlineItem(BaseModel):
    title: str
    level: int
    printed_page: int | None = None


class _Outline(BaseModel):
    entries: list[_OutlineItem]


_PROMPT = """The following text was extracted from a PDF. It contains either a table of
contents or a list of candidate section headings (with font metadata). Produce the
document outline: one entry per real section, in document order. `level` is the nesting
depth (1 = chapter, 2 = subchapter, ...). Set `printed_page` when a page number is shown
next to the entry. Exclude page furniture, running headers, and anything that is not a
section heading. Keep titles exactly as written (minus dot leaders and page numbers).

{context}"""


class AnthropicBackend:
    """Default backend using the official Anthropic SDK with structured output."""

    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic  # lazy import so heuristics-only runs don't need a key

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self._model = model

    def parse_outline(self, context: str) -> list[OutlineEntry]:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": _PROMPT.format(context=context)}],
            output_format=_Outline,
        )
        outline = response.parsed_output
        return [
            OutlineEntry(title=item.title, level=item.level, printed_page=item.printed_page)
            for item in outline.entries
        ]


_BACKENDS: dict[str, type] = {"anthropic": AnthropicBackend}


def get_backend(spec: str) -> LLMBackend:
    """Resolve a "provider:model-id" spec (model part optional) to a backend."""
    provider, _, model = spec.partition(":")
    if provider not in _BACKENDS:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Available: {', '.join(sorted(_BACKENDS))}"
        )
    backend_cls = _BACKENDS[provider]
    return backend_cls(model) if model else backend_cls()


def is_low_confidence(
    detected: int,
    failures: int,
    used_toc: bool,
    levels: list[int],
    page_count: int,
) -> bool:
    """Decide whether the heuristic outline needs LLM verification (auto mode)."""
    if detected == 0:
        return True
    if used_toc and detected < 3:
        return True
    if failures / detected > 0.2:
        return True
    if not used_toc:
        if any(b - a > 1 for a, b in zip(levels, levels[1:])):
            return True
        if page_count >= 300 and len(set(levels)) == 1:
            return True
    return False
