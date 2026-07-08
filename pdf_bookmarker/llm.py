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


# Public: the training tooling (training/) builds SFT datasets from PROMPT and
# Outline so that training format == serving format. Change them together.
class OutlineItem(BaseModel):
    title: str
    level: int
    printed_page: int | None = None


class Outline(BaseModel):
    entries: list[OutlineItem]


PROMPT = """The following text was extracted from a PDF. It contains either a table of
contents or a list of candidate section headings (with font metadata). Produce the
document outline: one entry per real section, in document order. `level` is the nesting
depth (1 = chapter, 2 = subchapter, ...). Set `printed_page` when a page number is shown
next to the entry. Exclude page furniture, running headers, and anything that is not a
section heading. Keep titles exactly as written (minus dot leaders and page numbers).

{context}"""


class AnthropicBackend:
    """Default backend using the official Anthropic SDK with structured output."""

    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None):
        import anthropic  # lazy import so heuristics-only runs don't need a key

        # api_key=None falls back to ANTHROPIC_API_KEY from the environment.
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def parse_outline(self, context: str) -> list[OutlineEntry]:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": PROMPT.format(context=context)}],
            output_format=Outline,
        )
        outline = response.parsed_output
        return [
            OutlineEntry(title=item.title, level=item.level, printed_page=item.printed_page)
            for item in outline.entries
        ]


class GeminiBackend:
    """Google Gemini backend using the google-genai SDK with structured output."""

    def __init__(self, model: str = "gemini-3.5-flash", api_key: str | None = None):
        try:
            from google import genai  # lazy import: shipped as the [gemini] extra
        except ImportError as exc:
            raise ImportError(
                'google-genai is not installed; run pip install "pdf-bookmarker[gemini]"'
            ) from exc

        # api_key=None falls back to GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def parse_outline(self, context: str) -> list[OutlineEntry]:
        response = self._client.models.generate_content(
            model=self._model,
            contents=PROMPT.format(context=context),
            config={
                "response_mime_type": "application/json",
                "response_schema": Outline,
            },
        )
        outline = response.parsed
        return [
            OutlineEntry(title=item.title, level=item.level, printed_page=item.printed_page)
            for item in outline.entries
        ]


_BACKENDS: dict[str, type] = {
    "anthropic": AnthropicBackend,
    "gemini": GeminiBackend,
}

# Env vars each provider's SDK reads its key from (first name used in warnings).
ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


class UnknownProviderError(ValueError):
    """The provider part of a "provider:model-id" spec is not registered."""


def get_backend(spec: str, api_key: str | None = None) -> LLMBackend:
    """Resolve a "provider:model-id" spec (model part optional) to a backend."""
    provider, _, model = spec.partition(":")
    if provider not in _BACKENDS:
        raise UnknownProviderError(
            f"Unknown LLM provider {provider!r}. Available: {', '.join(sorted(_BACKENDS))}"
        )
    backend_cls = _BACKENDS[provider]
    return backend_cls(model, api_key=api_key) if model else backend_cls(api_key=api_key)


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
