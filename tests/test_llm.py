from types import SimpleNamespace

import pytest

from pdf_bookmarker import llm
from pdf_bookmarker.models import OutlineEntry


class FakeMessages:
    def __init__(self, captured):
        self._captured = captured

    def parse(self, **kwargs):
        self._captured.update(kwargs)
        return SimpleNamespace(
            parsed_output=llm._Outline(
                entries=[llm._OutlineItem(title="Intro", level=1, printed_page=3)]
            )
        )


def _fake_anthropic(monkeypatch, captured):
    class FakeClient:
        def __init__(self):
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
