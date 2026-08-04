"""Model-agnostic LLM verification layer.

To add a provider: implement the LLMBackend protocol and register the class
in _BACKENDS. Selection is via "provider:model-id" strings (e.g. --model).
"""
import json
import os
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


class LocalBackend:
    """Local GGUF model via llama-cpp-python; no API key, nothing leaves the
    machine. The model part of the spec is the path to a .gguf file, e.g.
    --model "local:models/outline.gguf" (produce one with
    training/export_gguf.py). Output is grammar-constrained to the Outline
    schema, so the model cannot emit malformed JSON."""

    _N_CTX = 16384  # must cover prompt + generated outline
    _N_CTX_ENV = "PDF_BOOKMARKER_LOCAL_N_CTX"
    _N_GPU_LAYERS = 0  # llama.cpp's own default: CPU only
    _N_GPU_LAYERS_ENV = "PDF_BOOKMARKER_LOCAL_N_GPU_LAYERS"

    def __init__(
        self,
        model: str = "",
        api_key: str | None = None,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        chat: bool = False,
        no_think: bool = False,
    ):
        # api_key is accepted for LLMBackend compatibility and ignored.
        if not model:
            raise ValueError(
                "the local backend needs a model path, e.g. "
                '--model "local:models/outline.gguf"'
            )
        try:
            import llama_cpp  # lazy import: shipped as the [local] extra
        except ImportError as exc:
            raise ImportError(
                'llama-cpp-python is not installed; run pip install "pdf-bookmarker[local]"'
            ) from exc

        self._chat = chat
        self._no_think = no_think
        self._grammar = llama_cpp.LlamaGrammar.from_json_schema(
            json.dumps(Outline.model_json_schema())
        )
        self._llm = llama_cpp.Llama(
            model_path=model,
            n_ctx=self._int_option(n_ctx, cls_default=self._N_CTX, env=self._N_CTX_ENV),
            n_gpu_layers=self._int_option(
                n_gpu_layers, cls_default=self._N_GPU_LAYERS, env=self._N_GPU_LAYERS_ENV
            ),
            verbose=False,
        )

    @staticmethod
    def _int_option(value: int | None, *, cls_default: int, env: str) -> int:
        """Explicit kwarg wins, then the env var, then the shipped default.

        The env vars exist because the CLI selects a model with a bare
        "local:path" spec, which has nowhere to put tuning knobs — but a
        teacher-sized model (see training/distill.py) needs different ones
        than the 1.5B student these defaults were chosen for.
        """
        if value is not None:
            return value
        raw = os.environ.get(env)
        if raw is None:
            return cls_default
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"{env} must be an integer, not {raw!r}") from None

    def parse_outline(self, context: str) -> list[OutlineEntry]:
        prompt = PROMPT.format(context=context)
        if self._chat:
            # An off-the-shelf instruct model expects its own chat template;
            # the fine-tuned student instead expects the bare prompt it was
            # trained on, so raw completion stays the default.
            if self._no_think:
                prompt += "\n\n/no_think"
            result = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.0,
                grammar=self._grammar,
            )
            text = result["choices"][0]["message"]["content"]
        else:
            result = self._llm(
                prompt, max_tokens=4096, temperature=0.0, grammar=self._grammar
            )
            text = result["choices"][0]["text"]
        outline = Outline.model_validate_json(text)
        return [
            OutlineEntry(title=item.title, level=item.level, printed_page=item.printed_page)
            for item in outline.entries
        ]


_BACKENDS: dict[str, type] = {
    "anthropic": AnthropicBackend,
    "gemini": GeminiBackend,
    "local": LocalBackend,
}

# Env vars each provider's SDK reads its key from (first name used in warnings).
ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


class UnknownProviderError(ValueError):
    """The provider part of a "provider:model-id" spec is not registered."""


def get_backend(spec: str, api_key: str | None = None, **options) -> LLMBackend:
    """Resolve a "provider:model-id" spec (model part optional) to a backend.

    Extra keyword options are passed to the backend constructor; they are
    provider-specific (e.g. n_ctx for "local"), so an option the selected
    backend does not accept is a TypeError.
    """
    provider, _, model = spec.partition(":")
    if provider not in _BACKENDS:
        raise UnknownProviderError(
            f"Unknown LLM provider {provider!r}. Available: {', '.join(sorted(_BACKENDS))}"
        )
    backend_cls = _BACKENDS[provider]
    if model:
        return backend_cls(model, api_key=api_key, **options)
    return backend_cls(api_key=api_key, **options)


# Entries per page below which a line-labeler outline is worth an LLM call.
# Measured on the 76-document evaluation set: routing by this signal is
# sublinear, so a fraction of the calls buys a disproportionate share of the
# union's +4.3 title F1.
#
#   threshold   documents routed   title F1   gain
#      0.00            0%           0.7685      —      (labeler alone)
#      0.25            8%           0.7829   +0.0144
#      0.50           45%           0.7983   +0.0298
#      1.00           80%           0.8056   +0.0372
#      1.50           99%           0.8120   +0.0435   (quality-maximising)
#
# 0.5 is a cost/quality choice, not the best-scoring one: it takes 70% of the
# gain for 45% of the calls, which is what "auto" means here. Cross-fitting the
# threshold for quality alone picks 1.50 — i.e. "call the LLM on everything".
# Anyone who wants that should pass it, or just use --llm.
SPARSE_ENTRIES_PER_PAGE = 0.5


def is_sparse_outline(
    detected: int, page_count: int, threshold: float = SPARSE_ENTRIES_PER_PAGE
) -> bool:
    """Decide whether a line-labeler outline needs the LLM too (auto mode).

    Density, not the structural checks `is_low_confidence` applies: those were
    tuned for the font heuristics and are actively misleading here. On labeler
    outlines they fire on the documents that need the LLM *least* — 26% of the
    corpus, gaining +0.039 F1 where they fire against +0.044 where they stay
    silent — while density at a comparable budget (22% routed) gains more.

    A sparse outline means the labeler found little, which is where the LLM
    adds most. Per page rather than in total, so a 400-page book with 40
    headings counts as sparse and an 8-page paper with 6 does not.
    """
    if detected == 0:
        return True
    return detected / max(page_count, 1) <= threshold


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
