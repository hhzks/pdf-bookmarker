import shutil

import fitz
import pytest

from pdf_bookmarker import pipeline
from pdf_bookmarker.models import OutlineEntry


def test_process_pdf_writes_bookmarks(toc_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    result = pipeline.process_pdf(toc_pdf, out, llm_mode="never")
    assert result.bookmark_count == 4
    assert result.used_toc is True
    assert result.used_llm is False
    assert result.warnings == []
    toc = fitz.open(str(out)).get_toc()
    assert [item[:3] for item in toc] == [
        [1, "1 Introduction", 3],
        [2, "1.1 Background", 3],
        [1, "2 Methods", 4],
        [1, "3 Results", 5],
    ]


def test_dry_run_returns_entries_without_writing(toc_pdf, tmp_path):
    result = pipeline.process_pdf(toc_pdf, None, llm_mode="never")
    assert result.bookmark_count == 0
    assert result.entries[0].title == "1 Introduction"
    assert list(tmp_path.iterdir()) == []  # nothing written anywhere we control


def test_missing_file_raises(tmp_path):
    with pytest.raises(pipeline.InvalidPdfError):
        pipeline.process_pdf(tmp_path / "nope.pdf", tmp_path / "o.pdf", llm_mode="never")


def test_encrypted_raises(encrypted_pdf, tmp_path):
    with pytest.raises(pipeline.EncryptedPdfError):
        pipeline.process_pdf(encrypted_pdf, tmp_path / "o.pdf", llm_mode="never")


def test_no_text_layer_raises(no_text_pdf, tmp_path):
    with pytest.raises(pipeline.NoTextLayerError, match="OCR"):
        pipeline.process_pdf(no_text_pdf, tmp_path / "o.pdf", llm_mode="never")


def test_empty_extraction_raises_no_text_layer(toc_pdf, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.extractor, "extract_lines", lambda doc: [])
    with pytest.raises(pipeline.NoTextLayerError):
        pipeline.process_pdf(toc_pdf, tmp_path / "o.pdf", llm_mode="never")


def test_no_outline_raises(plain_pdf, tmp_path):
    with pytest.raises(pipeline.NoOutlineError):
        pipeline.process_pdf(plain_pdf, tmp_path / "o.pdf", llm_mode="never")


def test_no_outline_error_carries_warnings(plain_pdf, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(pipeline.NoOutlineError) as excinfo:
        pipeline.process_pdf(plain_pdf, tmp_path / "o.pdf", llm_mode="auto")
    assert any("without LLM" in w for w in excinfo.value.warnings)


def test_existing_bookmarks_replaced_by_default(bookmarked_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    result = pipeline.process_pdf(bookmarked_pdf, out, llm_mode="never")
    assert result.bookmark_count >= 1
    assert fitz.open(str(out)).get_toc()[0][1] != "Existing"


def test_existing_bookmarks_guard(bookmarked_pdf, tmp_path):
    with pytest.raises(pipeline.ExistingBookmarksError):
        pipeline.process_pdf(
            bookmarked_pdf, tmp_path / "o.pdf", llm_mode="never", replace_existing=False
        )


def test_input_file_released_after_processing(toc_pdf, tmp_path):
    src = tmp_path / "copy.pdf"
    shutil.copy(toc_pdf, src)
    pipeline.process_pdf(src, tmp_path / "out.pdf", llm_mode="never")
    src.unlink()  # raises PermissionError on Windows if the handle leaked
    assert not src.exists()


def test_input_file_released_after_failure(encrypted_pdf, tmp_path):
    src = tmp_path / "enc.pdf"
    shutil.copy(encrypted_pdf, src)
    with pytest.raises(pipeline.EncryptedPdfError):
        pipeline.process_pdf(src, tmp_path / "out.pdf", llm_mode="never")
    src.unlink()
    assert not src.exists()


def test_invalid_llm_mode_raises(toc_pdf, tmp_path):
    with pytest.raises(ValueError, match="llm_mode"):
        pipeline.process_pdf(toc_pdf, tmp_path / "o.pdf", llm_mode="sometimes")


def test_api_key_passed_to_backend(ghost_toc_pdf, tmp_path, monkeypatch):
    captured = {}

    class FakeBackend:
        def parse_outline(self, context):
            return [OutlineEntry("1 Alpha", 1, printed_page=2)]

    def fake_get_backend(spec, api_key=None):
        captured["spec"], captured["api_key"] = spec, api_key
        return FakeBackend()

    monkeypatch.setattr(pipeline.llm, "get_backend", fake_get_backend)
    result = pipeline.process_pdf(
        ghost_toc_pdf, tmp_path / "o.pdf", llm_mode="always", api_key="user-key"
    )
    assert captured["api_key"] == "user-key"
    assert result.used_llm is True


def test_auto_mode_user_key_enables_llm_without_env(ghost_toc_pdf, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = []

    class FakeBackend:
        def parse_outline(self, context):
            calls.append(context)
            return [OutlineEntry("1 Alpha", 1, printed_page=2)]

    monkeypatch.setattr(pipeline.llm, "get_backend", lambda spec, api_key=None: FakeBackend())
    result = pipeline.process_pdf(
        ghost_toc_pdf, tmp_path / "o.pdf", llm_mode="auto", api_key="user-key"
    )
    assert len(calls) == 1
    assert result.used_llm is True


def test_auto_mode_without_key_warns_and_continues(ghost_toc_pdf, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "o.pdf"
    result = pipeline.process_pdf(ghost_toc_pdf, out, llm_mode="auto")
    assert any("without LLM" in w for w in result.warnings)
    assert result.used_llm is False
    assert len(fitz.open(str(out)).get_toc()) == 3  # heuristic outline kept


def test_llm_failure_auto_falls_back(ghost_toc_pdf, tmp_path, monkeypatch):
    class FailingBackend:
        def parse_outline(self, context):
            raise RuntimeError("api down")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(pipeline.llm, "get_backend", lambda spec, api_key=None: FailingBackend())
    out = tmp_path / "o.pdf"
    result = pipeline.process_pdf(ghost_toc_pdf, out, llm_mode="auto", api_key="secret-sentinel")
    assert any("LLM call failed" in w for w in result.warnings)
    assert all("secret-sentinel" not in w for w in result.warnings)
    assert len(fitz.open(str(out)).get_toc()) == 3


def test_llm_failure_always_raises(ghost_toc_pdf, tmp_path, monkeypatch):
    class FailingBackend:
        def parse_outline(self, context):
            raise RuntimeError("api down")

    monkeypatch.setattr(pipeline.llm, "get_backend", lambda spec, api_key=None: FailingBackend())
    with pytest.raises(pipeline.LLMVerificationError, match="LLM verification failed") as excinfo:
        pipeline.process_pdf(
            ghost_toc_pdf, tmp_path / "o.pdf", llm_mode="always", api_key="secret-sentinel"
        )
    assert "secret-sentinel" not in str(excinfo.value)


def test_unknown_provider_propagates(toc_pdf, tmp_path):
    from pdf_bookmarker import llm

    with pytest.raises(llm.UnknownProviderError):
        pipeline.process_pdf(
            toc_pdf, tmp_path / "o.pdf", llm_mode="always", model_spec="bogus:x"
        )
