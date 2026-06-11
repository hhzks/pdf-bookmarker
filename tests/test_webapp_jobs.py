import time

import pytest

from pdf_bookmarker.pipeline import NoTextLayerError

from app import jobs as jobs_module
from app.jobs import JobStore


def wait_for(job, timeout=10.0):
    deadline = time.time() + timeout
    while job.status in ("queued", "processing"):
        if time.time() > deadline:
            raise AssertionError(f"job stuck in state {job.status!r}")
        time.sleep(0.02)


def test_submit_runs_pipeline_and_completes(fake_pipeline):
    store = JobStore(ttl_seconds=3600)
    job = store.submit(
        b"%PDF-1.4 input", "mybook.pdf",
        llm_mode="auto", model_spec="anthropic:claude-opus-4-8", api_key="user-key",
    )
    wait_for(job)
    assert job.status == "done"
    assert job.bookmark_count == 4
    assert job.output_path.read_bytes().startswith(b"%PDF")
    assert store.get(job.id) is job
    assert fake_pipeline[0]["llm_mode"] == "auto"
    assert fake_pipeline[0]["model_spec"] == "anthropic:claude-opus-4-8"
    assert fake_pipeline[0]["api_key"] == "user-key"
    assert fake_pipeline[0]["input"].read_bytes() == b"%PDF-1.4 input"


def test_api_key_not_stored_on_job(fake_pipeline):
    store = JobStore()
    job = store.submit(b"%PDF-1.4", "a.pdf", llm_mode="auto",
                       model_spec="anthropic", api_key="secret")
    wait_for(job)
    assert "secret" not in repr(job)


def test_failed_job_gets_friendly_error(monkeypatch):
    def boom(input_path, output_path, **kwargs):
        raise NoTextLayerError("no extractable text layer")

    monkeypatch.setattr(jobs_module, "process_pdf", boom)
    store = JobStore()
    job = store.submit(b"%PDF-1.4", "scan.pdf", llm_mode="auto",
                       model_spec="anthropic", api_key=None)
    wait_for(job)
    assert job.status == "failed"
    assert "scanned" in job.error


def test_unexpected_error_gets_generic_message(monkeypatch):
    def boom(input_path, output_path, **kwargs):
        raise RuntimeError("internal details that must not leak")

    monkeypatch.setattr(jobs_module, "process_pdf", boom)
    store = JobStore()
    job = store.submit(b"%PDF-1.4", "a.pdf", llm_mode="auto",
                       model_spec="anthropic", api_key=None)
    wait_for(job)
    assert job.status == "failed"
    assert "internal details" not in job.error


def test_cleanup_removes_expired_jobs(fake_pipeline):
    store = JobStore(ttl_seconds=3600)
    job = store.submit(b"%PDF-1.4", "a.pdf", llm_mode="auto",
                       model_spec="anthropic", api_key=None)
    wait_for(job)
    store.cleanup_expired(now=time.time() + 3601)
    assert store.get(job.id) is None
    assert not job.dir.exists()


def test_cleanup_keeps_fresh_jobs(fake_pipeline):
    store = JobStore(ttl_seconds=3600)
    job = store.submit(b"%PDF-1.4", "a.pdf", llm_mode="auto",
                       model_spec="anthropic", api_key=None)
    wait_for(job)
    store.cleanup_expired(now=time.time() + 60)
    assert store.get(job.id) is job
    assert job.dir.exists()
