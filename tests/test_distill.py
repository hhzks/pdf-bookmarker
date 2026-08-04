"""Tests for training/distill.py (the teacher runs offline via a fake backend)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import distill

from pdf_bookmarker.models import OutlineEntry


class _FakeTeacher:
    """Stands in for an LLMBackend; returns a fixed outline."""

    def __init__(self, entries):
        self._entries = entries

    def parse_outline(self, context):
        return [OutlineEntry(e.title, e.level, e.printed_page) for e in self._entries]


def _entry(title, level=1, printed_page=None):
    return OutlineEntry(title, level, printed_page)


# --- locator quality gate ---------------------------------------------------

def test_distill_rejects_teacher_output_that_does_not_locate(toc_pdf):
    """Titles absent from the body are hallucinated, so the record is dropped."""
    teacher = _FakeTeacher([
        _entry("Chapter On Phlogiston", 1, 3),
        _entry("Chapter On Aether", 1, 4),
        _entry("Chapter On Caloric", 1, 5),
    ])
    record, reason = distill.distill_pdf(toc_pdf, teacher)
    assert record is None
    assert reason == "teacher-unlocatable"


def test_distill_keeps_teacher_output_that_locates(toc_pdf):
    teacher = _FakeTeacher([
        _entry("1 Introduction", 1, 3),
        _entry("2 Methods", 1, 4),
        _entry("3 Results", 1, 5),
    ])
    record, reason = distill.distill_pdf(toc_pdf, teacher)
    assert reason is None
    assert record["silver"] is True
    assert [e["title"] for e in record["entries"]] == [
        "1 Introduction", "2 Methods", "3 Results"
    ]


def test_distill_tolerates_failures_under_the_threshold(toc_pdf):
    """One bad entry out of four is 25% -- kept when the threshold allows it."""
    teacher = _FakeTeacher([
        _entry("1 Introduction", 1, 3),
        _entry("1.1 Background", 2, 3),
        _entry("2 Methods", 1, 4),
        _entry("Chapter On Phlogiston", 1, 5),
    ])
    record, reason = distill.distill_pdf(toc_pdf, teacher, max_unlocatable=0.3)
    assert reason is None
    assert len(record["entries"]) == 4

    record, reason = distill.distill_pdf(toc_pdf, teacher, max_unlocatable=0.2)
    assert record is None
    assert reason == "teacher-unlocatable"


def test_distill_gate_runs_on_the_headings_path(headings_pdf):
    """The headings path has no printed pages, so unmatched entries are dropped."""
    teacher = _FakeTeacher([_entry("Nowhere To Be Found")])
    record, reason = distill.distill_pdf(headings_pdf, teacher, min_pages=1)
    assert record is None
    assert reason == "teacher-unlocatable"

    teacher = _FakeTeacher([
        _entry("Chapter 1 Getting Started"),
        _entry("Chapter 2 Advanced Usage"),
    ])
    record, reason = distill.distill_pdf(headings_pdf, teacher, min_pages=1)
    assert reason is None
    assert record["context_kind"] == "headings"


def test_distill_does_not_record_locator_page_fields(toc_pdf):
    """The gate is validation only -- silver labels keep harvest.py's shape."""
    teacher = _FakeTeacher([_entry("1 Introduction", 1, 3)])
    record, _ = distill.distill_pdf(toc_pdf, teacher)
    assert set(record["entries"][0]) == {"title", "level", "printed_page"}


# --- existing skip reasons still apply --------------------------------------

def test_distill_skips_pdfs_with_an_embedded_outline(outlined_toc_pdf):
    teacher = _FakeTeacher([_entry("1 Introduction", 1, 3)])
    record, reason = distill.distill_pdf(outlined_toc_pdf, teacher)
    assert record is None
    assert reason == "has-embedded-outline"


def test_distill_skips_when_the_teacher_returns_nothing(toc_pdf):
    record, reason = distill.distill_pdf(toc_pdf, _FakeTeacher([]))
    assert record is None
    assert reason == "teacher-empty"


# --- a failing teacher skips one PDF, it does not end the run ---------------

class _ExplodingTeacher:
    def __init__(self, exc):
        self._exc = exc

    def parse_outline(self, context):
        raise self._exc


def test_distill_skips_a_context_the_teacher_cannot_fit(toc_pdf):
    """llama.cpp raises this when the prompt exceeds n_ctx."""
    teacher = _ExplodingTeacher(
        ValueError("Requested tokens (11095) exceed context window of 8192")
    )
    record, reason = distill.distill_pdf(toc_pdf, teacher)
    assert record is None
    assert reason == "context-too-long"


def test_distill_skips_when_the_teacher_errors(toc_pdf):
    """An API timeout or a bad generation must not end a 200-PDF run."""
    record, reason = distill.distill_pdf(toc_pdf, _ExplodingTeacher(RuntimeError("boom")))
    assert record is None
    assert reason == "teacher-error"


def test_main_keeps_going_after_a_teacher_failure(tmp_path, monkeypatch, toc_pdf,
                                                  headings_pdf, offset_toc_pdf):
    """One poisoned PDF must not cost the whole run."""
    good = _FakeTeacher([_entry("1 Introduction", 1, 3)])
    calls = {"n": 0}

    class _FlakyTeacher:
        def parse_outline(self, context):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("Requested tokens (99) exceed context window of 8")
            return good.parse_outline(context)

    monkeypatch.setattr(distill.llm, "get_backend", lambda spec, **kw: _FlakyTeacher())
    corpus = _corpus(tmp_path, toc_pdf, headings_pdf, offset_toc_pdf)
    out = tmp_path / "silver.jsonl"
    assert distill.main([str(corpus), "-o", str(out), "--min-pages", "1",
                         "--max-unlocatable", "1.0"]) == 0
    assert calls["n"] == 3                       # all three were attempted
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2  # first one skipped


# --- CLI --------------------------------------------------------------------

def _corpus(tmp_path, *pdfs):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i, pdf in enumerate(pdfs):
        (corpus / f"{i}.pdf").write_bytes(Path(pdf).read_bytes())
    return corpus


def _run(monkeypatch, tmp_path, corpus, *argv):
    teacher = _FakeTeacher([_entry("1 Introduction", 1, 3)])
    monkeypatch.setattr(distill.llm, "get_backend", lambda spec, **kw: teacher)
    out = tmp_path / "silver.jsonl"
    # max-unlocatable 1.0 disables the gate: this exercises limit handling only.
    distill.main([str(corpus), "-o", str(out), "--min-pages", "1",
                  "--max-unlocatable", "1.0", *argv])
    return out.read_text(encoding="utf-8").splitlines()


def test_main_limit_zero_means_no_limit(tmp_path, monkeypatch, toc_pdf, headings_pdf):
    corpus = _corpus(tmp_path, toc_pdf, headings_pdf)
    assert len(_run(monkeypatch, tmp_path, corpus, "--limit", "0")) == 2


def test_main_limit_caps_the_run(tmp_path, monkeypatch, toc_pdf, headings_pdf):
    corpus = _corpus(tmp_path, toc_pdf, headings_pdf)
    assert len(_run(monkeypatch, tmp_path, corpus, "--limit", "1")) == 1


def test_main_forwards_n_ctx_to_the_backend(tmp_path, monkeypatch, toc_pdf):
    captured = {}

    def fake_get_backend(spec, **options):
        captured.update(spec=spec, **options)
        return _FakeTeacher([_entry("1 Introduction", 1, 3)])

    monkeypatch.setattr(distill.llm, "get_backend", fake_get_backend)
    corpus = _corpus(tmp_path, toc_pdf)
    distill.main([str(corpus), "-o", str(tmp_path / "s.jsonl"),
                  "--model", "local:teacher.gguf", "--n-ctx", "32768"])
    assert captured["spec"] == "local:teacher.gguf"
    assert captured["n_ctx"] == 32768


def test_main_omits_n_ctx_when_not_given(tmp_path, monkeypatch, toc_pdf):
    """API backends reject unknown kwargs, so n_ctx must not be sent blindly."""
    captured = {}

    def fake_get_backend(spec, **options):
        captured.update(spec=spec, options=options)
        return _FakeTeacher([_entry("1 Introduction", 1, 3)])

    monkeypatch.setattr(distill.llm, "get_backend", fake_get_backend)
    corpus = _corpus(tmp_path, toc_pdf)
    distill.main([str(corpus), "-o", str(tmp_path / "s.jsonl")])
    assert captured["options"] == {}


def _captured_options(tmp_path, monkeypatch, toc_pdf, *argv):
    captured = {}

    def fake_get_backend(spec, **options):
        captured.update(options)
        return _FakeTeacher([_entry("1 Introduction", 1, 3)])

    monkeypatch.setattr(distill.llm, "get_backend", fake_get_backend)
    corpus = _corpus(tmp_path, toc_pdf)
    distill.main([str(corpus), "-o", str(tmp_path / "s.jsonl"), *argv])
    return captured


def test_main_forwards_gpu_and_chat_options(tmp_path, monkeypatch, toc_pdf):
    options = _captured_options(
        tmp_path, monkeypatch, toc_pdf,
        "--model", "local:teacher.gguf", "--n-gpu-layers", "-1",
        "--chat", "--no-think",
    )
    assert options == {"n_gpu_layers": -1, "chat": True, "no_think": True}


def test_main_omits_chat_options_when_not_given(tmp_path, monkeypatch, toc_pdf):
    """--chat and --no-think are local-only; API backends must not see them."""
    options = _captured_options(tmp_path, monkeypatch, toc_pdf)
    assert options == {}
