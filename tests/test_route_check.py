"""Tests for training/route_check.py."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import route_check
from pdf_bookmarker import llm, pipeline
from pdf_bookmarker.pipeline import PipelineResult
from pdf_bookmarker.models import OutlineEntry


def test_replay_backend_returns_the_cached_entries():
    backend = route_check.ReplayBackend(
        [{"title": "Introduction", "level": 1, "printed_page": 3}]
    )
    entries = backend.parse_outline("any context")
    assert [(e.title, e.level) for e in entries] == [("Introduction", 1)]


def test_replay_drops_the_cached_page_so_the_locator_places_it():
    """A cached printed_page is the LLM's own guess; keeping it lets a wrong
    guess survive location as an offset-corrected fallback."""
    backend = route_check.ReplayBackend(
        [{"title": "Introduction", "level": 1, "printed_page": 3}]
    )
    assert backend.parse_outline("ctx")[0].printed_page is None


@pytest.fixture
def fake_process(monkeypatch):
    """Replace process_pdf with a stub, recording the kwargs it was called with."""
    calls = []

    def stub(input_path, output_path, **kwargs):
        calls.append(kwargs)
        return PipelineResult(
            entries=[OutlineEntry(title="Introduction", level=1, page=0)],
            bookmark_count=0,
            used_llm=kwargs["llm_mode"] == "always",
            used_toc=False,
            used_labeler=True,
        )

    monkeypatch.setattr(pipeline, "process_pdf", stub)
    return calls


RECORDS = [
    {"file": "a.pdf", "sha256": "a", "entries": [{"title": "Introduction", "level": 1}]},
    {"file": "b.pdf", "sha256": "b", "entries": [{"title": "Introduction", "level": 1}]},
]


def test_the_density_threshold_reaches_the_pipeline(fake_process):
    """The whole point of the harness: process_pdf does the routing, not us."""
    route_check.run_config(
        RECORDS, {}, "model.joblib", llm_mode="auto", density=1.25
    )
    assert [c["llm_density"] for c in fake_process] == [1.25, 1.25]
    assert all(c["labeler_path"] == "model.joblib" for c in fake_process)


def test_an_api_key_is_passed_so_auto_mode_does_not_skip_on_a_missing_env_var(
    fake_process,
):
    route_check.run_config(RECORDS, {}, "m", llm_mode="auto", density=0.5)
    assert all(c["api_key"] for c in fake_process)


def test_routed_counts_the_documents_that_actually_called_the_llm(fake_process):
    result = route_check.run_config(
        RECORDS, {}, "m", llm_mode="always", density=0.0
    )
    assert result["routed"] == 2
    assert result["routed_share"] == 1.0
    assert result["f1"] == 1.0


def test_a_failing_document_is_recorded_and_does_not_sink_the_run(monkeypatch):
    def stub(input_path, output_path, **kwargs):
        if input_path == "a.pdf":
            raise pipeline.InvalidPdfError("broken")
        return PipelineResult(
            entries=[OutlineEntry(title="Introduction", level=1, page=0)],
            bookmark_count=0,
            used_llm=False,
            used_toc=False,
            used_labeler=True,
        )

    monkeypatch.setattr(pipeline, "process_pdf", stub)
    result = route_check.run_config(
        RECORDS, {}, "m", llm_mode="never", density=0.0
    )
    assert result["documents"] == 1
    assert [path for path, _ in result["failed"]] == ["a.pdf"]


def test_the_real_backend_is_restored_even_when_a_run_raises(monkeypatch):
    """run_config swaps llm.get_backend globally; leaking that would silently
    corrupt any later measurement in the same process."""
    original = llm.get_backend

    def boom(input_path, output_path, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "process_pdf", boom)
    with pytest.raises(KeyboardInterrupt):
        route_check.run_config(RECORDS, {}, "m", llm_mode="auto", density=0.5)
    assert llm.get_backend is original


def test_per_document_f1_is_kept_for_paired_comparisons(fake_process):
    """Macro F1 alone cannot say whether a routing change is significant."""
    result = route_check.run_config(
        RECORDS, {}, "m", llm_mode="never", density=0.0
    )
    assert len(result["per_document_f1"]) == 2


def test_no_labeler_run_configures_none_so_the_llm_stands_alone(fake_process):
    """The 'LLM alone' row has to go through process_pdf like the others; with
    no labeler it replaces the heuristic outline rather than merging."""
    route_check.run_config(RECORDS, {}, None, llm_mode="always", density=0.0)
    assert all(c["labeler_path"] is None for c in fake_process)


def test_main_reports_every_configuration(tmp_path, fake_process, capsys):
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(json.dumps(r) for r in RECORDS), encoding="utf-8"
    )
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        json.dumps({"sha256": "a", "entries": []}), encoding="utf-8"
    )
    out = tmp_path / "out.json"

    assert route_check.main([
        str(records), "--labeler", "m",
        "--llm-predictions", str(preds),
        "--thresholds", "0", "0.5", "--always",
        "--json", str(out),
    ]) == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    assert set(written) == {"auto @ 0.0", "auto @ 0.5", "always (--llm)"}
