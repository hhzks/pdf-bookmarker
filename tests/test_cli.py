import fitz
import pytest

from pdf_bookmarker import cli
from pdf_bookmarker.models import OutlineEntry


def test_dry_run_prints_outline(toc_pdf, capsys):
    rc = cli.main([str(toc_pdf), "--dry-run", "--no-llm"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 Introduction" in out
    assert "  1.1 Background" in out  # indented one level


def test_writes_bookmarks_from_toc(toc_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    rc = cli.main([str(toc_pdf), "-o", str(out), "--no-llm"])
    assert rc == 0
    toc = fitz.open(str(out)).get_toc()
    assert [item[:3] for item in toc] == [
        [1, "1 Introduction", 3],
        [2, "1.1 Background", 3],
        [1, "2 Methods", 4],
        [1, "3 Results", 5],
    ]


def test_writes_bookmarks_from_fragmented_pdf(fragmented_pdf, tmp_path):
    """LaTeX-style PDFs: split number/title fragments, chapter TOC rows
    without dot leaders, subchapters nested under their parent chapter."""
    out = tmp_path / "out.pdf"
    rc = cli.main([str(fragmented_pdf), "-o", str(out), "--no-llm"])
    assert rc == 0
    toc = fitz.open(str(out)).get_toc()
    assert [item[:3] for item in toc] == [
        [1, "1 Reading", 2],
        [1, "2 Logic and background", 3],
        [1, "3 Revision", 4],
        [2, "3.1 Propositional Logic", 4],
        [2, "3.2 Predicate Logic", 5],
    ]


def test_print_outline_survives_unencodable_console(monkeypatch):
    """A cp1252 Windows console cannot encode every glyph; never crash."""
    import io
    import sys

    buf = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buf, encoding="cp1252"))
    cli.print_outline([OutlineEntry(title="A ∨ B", level=1, page=0)])
    sys.stdout.flush()
    assert b"A ? B" in buf.getvalue()


def test_writes_bookmarks_from_headings(headings_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    rc = cli.main([str(headings_pdf), "-o", str(out), "--no-llm"])
    assert rc == 0
    toc = fitz.open(str(out)).get_toc()
    assert [item[:3] for item in toc] == [
        [1, "Chapter 1 Getting Started", 1],
        [2, "1.1 Installation", 1],
        [1, "Chapter 2 Advanced Usage", 2],
        [2, "2.1 Configuration", 2],
    ]


def test_default_output_path(toc_pdf, tmp_path, monkeypatch):
    import shutil

    src = tmp_path / "book.pdf"
    shutil.copy(toc_pdf, src)
    rc = cli.main([str(src), "--no-llm"])
    assert rc == 0
    assert (tmp_path / "book.bookmarked.pdf").exists()


def test_no_text_layer_errors(no_text_pdf, capsys):
    rc = cli.main([str(no_text_pdf), "--no-llm"])
    assert rc == 2
    assert "OCR" in capsys.readouterr().err


def test_encrypted_pdf_errors(encrypted_pdf, capsys):
    rc = cli.main([str(encrypted_pdf), "--no-llm"])
    assert rc == 2
    assert "encrypted" in capsys.readouterr().err


def test_missing_file_errors(tmp_path, capsys):
    rc = cli.main([str(tmp_path / "nope.pdf"), "--no-llm"])
    assert rc == 2


def test_existing_bookmarks_require_force(bookmarked_pdf, tmp_path, capsys):
    rc = cli.main([str(bookmarked_pdf), "--no-llm"])
    assert rc == 2
    assert "--force" in capsys.readouterr().err
    out = tmp_path / "out.pdf"
    rc = cli.main([str(bookmarked_pdf), "-o", str(out), "--no-llm", "--force"])
    assert rc == 0
    toc = fitz.open(str(out)).get_toc()
    assert toc and toc[0][1] != "Existing"


def test_unknown_provider_errors(toc_pdf, capsys):
    rc = cli.main([str(toc_pdf), "--llm", "--model", "bogus:x"])
    assert rc == 2
    assert "Unknown LLM provider" in capsys.readouterr().err


def test_llm_flag_uses_backend(headings_pdf, monkeypatch, tmp_path):
    class FakeBackend:
        def parse_outline(self, context):
            return [OutlineEntry("Chapter 1 Getting Started", 1)]

    monkeypatch.setattr(cli.llm, "get_backend", lambda spec: FakeBackend())
    out = tmp_path / "out.pdf"
    rc = cli.main([str(headings_pdf), "--llm", "-o", str(out)])
    assert rc == 0
    toc = fitz.open(str(out)).get_toc()
    assert toc[0][1] == "Chapter 1 Getting Started"


def test_auto_mode_low_confidence_calls_llm(ghost_toc_pdf, monkeypatch, tmp_path):
    calls = []

    class FakeBackend:
        def parse_outline(self, context):
            calls.append(context)
            return [
                OutlineEntry("1 Alpha", 1, printed_page=2),
                OutlineEntry("2 Beta", 1, printed_page=3),
            ]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(cli.llm, "get_backend", lambda spec: FakeBackend())
    out = tmp_path / "out.pdf"
    rc = cli.main([str(ghost_toc_pdf), "-o", str(out)])
    assert rc == 0
    assert len(calls) == 1
    assert [item[1] for item in fitz.open(str(out)).get_toc()] == ["1 Alpha", "2 Beta"]


def test_auto_mode_without_key_warns_and_continues(ghost_toc_pdf, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "out.pdf"
    rc = cli.main([str(ghost_toc_pdf), "-o", str(out)])
    assert rc == 0
    assert "without LLM" in capsys.readouterr().err
    assert len(fitz.open(str(out)).get_toc()) == 3  # heuristic outline kept


def test_auto_mode_without_gemini_key_warns_and_continues(
    ghost_toc_pdf, monkeypatch, tmp_path, capsys
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    out = tmp_path / "out.pdf"
    rc = cli.main([str(ghost_toc_pdf), "-o", str(out), "--model", "gemini"])
    assert rc == 0
    assert "without LLM" in capsys.readouterr().err
    assert len(fitz.open(str(out)).get_toc()) == 3  # heuristic outline kept


def test_llm_flag_with_gemini_but_no_key_is_pipeline_error(
    ghost_toc_pdf, monkeypatch, tmp_path, capsys
):
    """Missing key surfaces as an LLM failure (rc 1), same as other providers."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rc = cli.main([str(ghost_toc_pdf), "-o", str(tmp_path / "out.pdf"),
                   "--llm", "--model", "gemini"])
    assert rc == 1
    assert "LLM verification failed" in capsys.readouterr().err


def test_auto_mode_high_confidence_skips_llm(toc_pdf, monkeypatch, tmp_path):
    def boom(spec):
        raise AssertionError("LLM should not be called for a healthy outline")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(cli.llm, "get_backend", boom)
    rc = cli.main([str(toc_pdf), "-o", str(tmp_path / "out.pdf")])
    assert rc == 0


def test_llm_failure_in_auto_mode_falls_back(ghost_toc_pdf, monkeypatch, tmp_path, capsys):
    class FailingBackend:
        def parse_outline(self, context):
            raise RuntimeError("api down")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(cli.llm, "get_backend", lambda spec: FailingBackend())
    out = tmp_path / "out.pdf"
    rc = cli.main([str(ghost_toc_pdf), "-o", str(out)])
    assert rc == 0
    assert "warning" in capsys.readouterr().err.lower()
    assert len(fitz.open(str(out)).get_toc()) == 3


def test_llm_failure_with_llm_flag_errors(ghost_toc_pdf, monkeypatch, tmp_path, capsys):
    class FailingBackend:
        def parse_outline(self, context):
            raise RuntimeError("api down")

    monkeypatch.setattr(cli.llm, "get_backend", lambda spec: FailingBackend())
    rc = cli.main([str(ghost_toc_pdf), "--llm", "-o", str(tmp_path / "out.pdf")])
    assert rc == 1
    assert "LLM verification failed" in capsys.readouterr().err
