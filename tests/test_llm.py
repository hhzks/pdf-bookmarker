import os
from types import SimpleNamespace

import pytest

from pdf_bookmarker import llm
from pdf_bookmarker.models import OutlineEntry


@pytest.mark.skipif(
    not os.environ.get("PDF_BOOKMARKER_LIVE_LLM"),
    reason="live API test; set PDF_BOOKMARKER_LIVE_LLM=1 (requires ANTHROPIC_API_KEY)",
)
def test_anthropic_backend_live():
    backend = llm.AnthropicBackend()
    entries = backend.parse_outline(
        "Table of contents text:\n"
        "1 Introduction .......... 3\n"
        "1.1 Background .......... 3\n"
        "2 Methods .......... 4"
    )
    assert entries
    assert entries[0].level == 1
    assert any(e.level == 2 for e in entries)


class FakeMessages:
    def __init__(self, captured):
        self._captured = captured

    def parse(self, **kwargs):
        self._captured.update(kwargs)
        return SimpleNamespace(
            parsed_output=llm.Outline(
                entries=[llm.OutlineItem(title="Intro", level=1, printed_page=3)]
            )
        )


def _fake_anthropic(monkeypatch, captured):
    class FakeClient:
        def __init__(self, api_key=None):
            captured["client_api_key"] = api_key
            self.messages = FakeMessages(captured)

    monkeypatch.setattr("anthropic.Anthropic", FakeClient)


def test_anthropic_backend_parses_outline(monkeypatch):
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    backend = llm.AnthropicBackend(model="claude-opus-4-8")
    entries = backend.parse_outline("1 Intro .......... 3")
    assert entries == [OutlineEntry(title="Intro", level=1, printed_page=3)]
    assert captured["model"] == "claude-opus-4-8"
    assert "1 Intro" in captured["messages"][0]["content"]


def test_get_backend_passes_model_through(monkeypatch):
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    backend = llm.get_backend("anthropic:claude-haiku-4-5")
    backend.parse_outline("x")
    assert captured["model"] == "claude-haiku-4-5"


def test_get_backend_default_model(monkeypatch):
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    backend = llm.get_backend("anthropic")
    backend.parse_outline("x")
    assert captured["model"] == "claude-opus-4-8"


def test_get_backend_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm.get_backend("bogus:model-x")


class FakeGeminiModels:
    def __init__(self, captured):
        self._captured = captured

    def generate_content(self, **kwargs):
        self._captured.update(kwargs)
        return SimpleNamespace(
            parsed=llm.Outline(
                entries=[llm.OutlineItem(title="Intro", level=1, printed_page=3)]
            )
        )


def _fake_gemini(monkeypatch, captured):
    class FakeClient:
        def __init__(self, api_key=None):
            captured["client_api_key"] = api_key
            self.models = FakeGeminiModels(captured)

    monkeypatch.setattr("google.genai.Client", FakeClient)


def test_gemini_backend_parses_outline(monkeypatch):
    captured = {}
    _fake_gemini(monkeypatch, captured)
    backend = llm.GeminiBackend(model="gemini-3.5-flash")
    entries = backend.parse_outline("1 Intro .......... 3")
    assert entries == [OutlineEntry(title="Intro", level=1, printed_page=3)]
    assert captured["model"] == "gemini-3.5-flash"
    assert "1 Intro" in captured["contents"]
    assert captured["config"]["response_schema"] is llm.Outline
    assert captured["config"]["response_mime_type"] == "application/json"


def test_get_backend_gemini_default_model(monkeypatch):
    captured = {}
    _fake_gemini(monkeypatch, captured)
    llm.get_backend("gemini").parse_outline("x")
    assert captured["model"] == "gemini-3.5-flash"


def test_get_backend_gemini_model_passthrough(monkeypatch):
    captured = {}
    _fake_gemini(monkeypatch, captured)
    llm.get_backend("gemini:gemini-3.1-pro-preview").parse_outline("x")
    assert captured["model"] == "gemini-3.1-pro-preview"


@pytest.mark.skipif(
    not os.environ.get("PDF_BOOKMARKER_LIVE_LLM"),
    reason="live API test; set PDF_BOOKMARKER_LIVE_LLM=1 (requires GEMINI_API_KEY)",
)
def test_gemini_backend_live():
    backend = llm.GeminiBackend()
    entries = backend.parse_outline(
        "Table of contents text:\n"
        "1 Introduction .......... 3\n"
        "1.1 Background .......... 3\n"
        "2 Methods .......... 4"
    )
    assert entries
    assert entries[0].level == 1
    assert any(e.level == 2 for e in entries)


@pytest.mark.parametrize(
    "detected,failures,used_toc,levels,page_count,expected",
    [
        (0, 0, False, [], 10, True),            # nothing detected
        (2, 0, True, [1, 1], 10, True),         # TOC parsed but <3 entries
        (10, 3, True, [1] * 10, 10, True),      # >20% location failures
        (4, 0, False, [1, 3, 1, 2], 10, True),  # incoherent level jump
        (5, 0, False, [1, 1, 1, 1, 1], 400, True),   # flat outline, 300+ pages
        (4, 0, True, [1, 2, 1, 1], 10, False),       # healthy TOC outline
        (4, 0, False, [1, 2, 1, 2], 50, False),      # healthy heading outline
    ],
)
def test_is_low_confidence(detected, failures, used_toc, levels, page_count, expected):
    assert llm.is_low_confidence(detected, failures, used_toc, levels, page_count) is expected


def test_anthropic_backend_passes_api_key(monkeypatch):
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    llm.AnthropicBackend(api_key="sk-user")
    assert captured["client_api_key"] == "sk-user"


def test_gemini_backend_passes_api_key(monkeypatch):
    captured = {}
    _fake_gemini(monkeypatch, captured)
    llm.GeminiBackend(api_key="g-user")
    assert captured["client_api_key"] == "g-user"


def test_get_backend_forwards_api_key(monkeypatch):
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    llm.get_backend("anthropic:claude-opus-4-8", api_key="sk-user")
    assert captured["client_api_key"] == "sk-user"


def test_get_backend_forwards_api_key_without_model(monkeypatch):
    captured = {}
    _fake_gemini(monkeypatch, captured)
    llm.get_backend("gemini", api_key="g-user")
    assert captured["client_api_key"] == "g-user"


def test_get_backend_api_key_defaults_to_none(monkeypatch):
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    llm.get_backend("anthropic")
    assert captured["client_api_key"] is None
