import time

import pytest
from fastapi.testclient import TestClient

from pdf_bookmarker.pipeline import NoTextLayerError

from app import jobs as jobs_module
from app import routes
from app.main import create_app

PDF_BYTES = b"%PDF-1.4 minimal test bytes"


@pytest.fixture
def client(fake_pipeline):
    with TestClient(create_app(rate_limit_per_hour=1000)) as c:
        yield c


def upload(client, filename="mybook.pdf", body=PDF_BYTES, **form):
    return client.post(
        "/api/jobs",
        files={"file": (filename, body, "application/pdf")},
        data=form,
    )


def poll_until_finished(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("job never finished")


def test_job_lifecycle(client):
    res = upload(client)
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    body = poll_until_finished(client, job_id)
    assert body["status"] == "done"
    assert body["bookmark_count"] == 4

    dl = client.get(f"/api/jobs/{job_id}/download")
    assert dl.status_code == 200
    assert dl.headers["content-type"] == "application/pdf"
    assert 'filename="mybook.bookmarked.pdf"' in dl.headers["content-disposition"]
    assert dl.content.startswith(b"%PDF")


def test_rejects_non_pdf(client):
    res = upload(client, filename="x.txt", body=b"hello")
    assert res.status_code == 400


def test_rejects_oversize(client, monkeypatch):
    monkeypatch.setattr("app.routes.MAX_SIZE", 10)
    res = upload(client)
    assert res.status_code == 413


def test_rejects_bad_llm_mode(client):
    res = upload(client, llm_mode="sometimes")
    assert res.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/download").status_code == 404


def test_failed_job_reports_friendly_error(monkeypatch):
    def boom(input_path, output_path, **kwargs):
        raise NoTextLayerError("no extractable text layer")

    monkeypatch.setattr(jobs_module, "process_pdf", boom)
    with TestClient(create_app(rate_limit_per_hour=1000)) as client:
        job_id = upload(client).json()["job_id"]
        body = poll_until_finished(client, job_id)
        assert body["status"] == "failed"
        assert "scanned" in body["error"]
        # a failed job has nothing to download
        assert client.get(f"/api/jobs/{job_id}/download").status_code == 404


def _first_call(fake_pipeline, timeout=10):
    deadline = time.time() + timeout
    while not fake_pipeline and time.time() < deadline:
        time.sleep(0.02)
    assert fake_pipeline, "pipeline was never called"
    return fake_pipeline[0]


def test_no_key_ignores_client_model(client, fake_pipeline):
    upload(client, llm_mode="always", model="anthropic:claude-opus-4-8")
    call = _first_call(fake_pipeline)
    assert call["model_spec"] == routes.SERVER_MODEL_SPEC
    assert call["api_key"] is None


def test_no_key_no_model_uses_server_model(client, fake_pipeline):
    upload(client, llm_mode="always")
    call = _first_call(fake_pipeline)
    assert call["model_spec"] == routes.SERVER_MODEL_SPEC


def test_key_with_model_is_honored(client, fake_pipeline):
    upload(client, llm_mode="always", model="anthropic:claude-sonnet-4-6",
           api_key="user-secret")
    call = _first_call(fake_pipeline)
    assert call["model_spec"] == "anthropic:claude-sonnet-4-6"
    assert call["api_key"] == "user-secret"


def test_options_forwarded_to_pipeline(client, fake_pipeline):
    upload(client, llm_mode="always", model="gemini:gemini-3.5-flash",
           api_key="user-secret")
    call = _first_call(fake_pipeline)
    assert call["llm_mode"] == "always"
    assert call["model_spec"] == "gemini:gemini-3.5-flash"
    assert call["api_key"] == "user-secret"


def test_rate_limit_returns_429(fake_pipeline):
    with TestClient(create_app(rate_limit_per_hour=2)) as client:
        assert upload(client).status_code == 202
        assert upload(client).status_code == 202
        assert upload(client).status_code == 429


def test_cors_allows_configured_origin(fake_pipeline):
    app = create_app(rate_limit_per_hour=1000,
                     allowed_origins=["http://frontend.test"])
    with TestClient(app) as client:
        res = client.options("/api/jobs", headers={
            "Origin": "http://frontend.test",
            "Access-Control-Request-Method": "POST",
        })
        assert res.headers.get("access-control-allow-origin") == "http://frontend.test"


def test_rate_limit_keyed_on_proxy_appended_ip(fake_pipeline):
    """The rightmost X-Forwarded-For entry (trusted proxy) is the key; a
    spoofed leftmost value must not reset the quota."""
    with TestClient(create_app(rate_limit_per_hour=1)) as client:
        res = client.post(
            "/api/jobs",
            files={"file": ("a.pdf", PDF_BYTES, "application/pdf")},
            headers={"x-forwarded-for": "spoof-1, 198.51.100.7"},
        )
        assert res.status_code == 202
        res = client.post(
            "/api/jobs",
            files={"file": ("a.pdf", PDF_BYTES, "application/pdf")},
            headers={"x-forwarded-for": "spoof-2, 198.51.100.7"},
        )
        assert res.status_code == 429


def test_failed_validation_does_not_consume_quota(fake_pipeline):
    with TestClient(create_app(rate_limit_per_hour=1)) as client:
        res = client.post(
            "/api/jobs", files={"file": ("x.txt", b"hello", "text/plain")}
        )
        assert res.status_code == 400
        res = client.post(
            "/api/jobs", files={"file": ("a.pdf", PDF_BYTES, "application/pdf")}
        )
        assert res.status_code == 202


def test_ocr_options_reach_pipeline(client, fake_pipeline):
    from app import routes
    upload(client)
    call = _first_call(fake_pipeline)  # helper added in the model-choice work
    assert call["ocr_mode"] == "auto"
    assert call["ocr_max_pages"] == routes.OCR_MAX_PAGES


def test_page_limit_reports_friendly_error(monkeypatch):
    from pdf_bookmarker.pipeline import OcrPageLimitError

    def boom(input_path, output_path, **kwargs):
        raise OcrPageLimitError("too long")

    monkeypatch.setattr(jobs_module, "process_pdf", boom)
    with TestClient(create_app(rate_limit_per_hour=1000)) as client:
        job_id = upload(client).json()["job_id"]
        body = poll_until_finished(client, job_id)
        assert body["status"] == "failed"
        assert "too long" in body["error"].lower()


def test_ocr_unavailable_reports_friendly_error(monkeypatch):
    from pdf_bookmarker.pipeline import OcrUnavailableError

    def boom(input_path, output_path, **kwargs):
        raise OcrUnavailableError("no tesseract")

    monkeypatch.setattr(jobs_module, "process_pdf", boom)
    with TestClient(create_app(rate_limit_per_hour=1000)) as client:
        job_id = upload(client).json()["job_id"]
        body = poll_until_finished(client, job_id)
        assert body["status"] == "failed"
        assert "scanned" in body["error"].lower()


def test_a_broken_labeler_path_stops_startup(monkeypatch, tmp_path):
    """Better one loud failure at boot than every upload failing obscurely."""
    monkeypatch.setenv("PDF_BOOKMARKER_LABELER", str(tmp_path / "missing.joblib"))
    with pytest.raises(RuntimeError, match="PDF_BOOKMARKER_LABELER"):
        create_app()


def test_startup_without_a_labeler_is_unaffected(monkeypatch):
    monkeypatch.delenv("PDF_BOOKMARKER_LABELER", raising=False)
    assert create_app() is not None


def test_the_labeler_is_loaded_at_startup(monkeypatch):
    """Validates the config and warms the cache, so no job pays the load."""
    from app import main as main_module

    calls = []
    monkeypatch.setattr(
        main_module, "resolve_labeler", lambda path: calls.append(path) or "model"
    )
    create_app()
    assert calls == [None]  # resolved from the environment, not a hardcoded path


def test_startup_logs_whether_the_labeler_is_active(monkeypatch, caplog):
    """create_app runs at import, before uvicorn configures logging; a status
    line emitted there is swallowed and the operator never sees it."""
    import logging

    monkeypatch.delenv("PDF_BOOKMARKER_LABELER", raising=False)
    app = create_app()
    with caplog.at_level(logging.INFO, logger="pdf_bookmarker.web"):
        with TestClient(app):
            pass
    assert any("labeler" in record.message.lower() for record in caplog.records)


def test_the_deployment_can_override_the_verification_model(monkeypatch):
    import importlib

    monkeypatch.setenv("VERIFICATION_MODEL", "gemini:gemini-3.5-flash")
    reloaded = importlib.reload(routes)
    try:
        assert reloaded.SERVER_MODEL_SPEC == "gemini:gemini-3.5-flash"
    finally:
        monkeypatch.delenv("VERIFICATION_MODEL", raising=False)
        importlib.reload(routes)


def test_a_missing_local_model_warns_but_starts(monkeypatch, caplog):
    """Unlike the labeler, the LLM is optional by design: auto mode degrades to
    the heuristic outline, so a missing file must not take the server down."""
    import logging

    monkeypatch.setattr(routes, "SERVER_MODEL_SPEC", "local:no/such/model.gguf")
    app = create_app()
    with caplog.at_level(logging.WARNING, logger="pdf_bookmarker.web"):
        with TestClient(app):
            pass
    assert any("no/such/model.gguf" in record.message for record in caplog.records)


def test_a_present_local_model_is_reported(monkeypatch, caplog, tmp_path):
    import logging

    model = tmp_path / "outline.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(routes, "SERVER_MODEL_SPEC", f"local:{model}")
    app = create_app()
    with caplog.at_level(logging.INFO, logger="pdf_bookmarker.web"):
        with TestClient(app):
            pass
    assert any(str(model) in record.message for record in caplog.records)
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("verification model" in message for message in warnings)


# --- labeler-only deployment: the server runs no LLM of its own ---------------

def test_the_server_runs_no_llm_of_its_own_by_default(monkeypatch):
    """A CPU-bound host cannot afford a local model in the request path, and
    the routed gain does not pay for the latency."""
    import importlib

    monkeypatch.delenv("VERIFICATION_MODEL", raising=False)
    reloaded = importlib.reload(routes)
    try:
        assert reloaded.SERVER_MODEL_SPEC == ""
    finally:
        importlib.reload(routes)


def test_a_keyless_job_never_reaches_the_llm(client, fake_pipeline, monkeypatch):
    monkeypatch.setattr(routes, "SERVER_MODEL_SPEC", "")
    upload(client, llm_mode="always")
    assert _first_call(fake_pipeline)["llm_mode"] == "never"


def test_a_keyless_job_still_uses_a_configured_server_model(
    client, fake_pipeline, monkeypatch
):
    """Setting VERIFICATION_MODEL puts the deployment back in charge."""
    monkeypatch.setattr(routes, "SERVER_MODEL_SPEC", "local:models/outline.gguf")
    upload(client, llm_mode="always")
    call = _first_call(fake_pipeline)
    assert call["llm_mode"] == "always"
    assert call["model_spec"] == "local:models/outline.gguf"


def test_a_caller_with_a_key_still_gets_the_llm(client, fake_pipeline, monkeypatch):
    monkeypatch.setattr(routes, "SERVER_MODEL_SPEC", "")
    upload(client, llm_mode="always", model="anthropic:claude-sonnet-4-6",
           api_key="user-secret")
    call = _first_call(fake_pipeline)
    assert call["llm_mode"] == "always"
    assert call["model_spec"] == "anthropic:claude-sonnet-4-6"


def test_a_key_without_a_model_falls_back_to_the_shipped_default(
    client, fake_pipeline, monkeypatch
):
    """With no server model there is nothing to inherit, so use the CLI's."""
    from pdf_bookmarker import llm

    monkeypatch.setattr(routes, "SERVER_MODEL_SPEC", "")
    upload(client, llm_mode="always", api_key="user-secret")
    assert _first_call(fake_pipeline)["model_spec"] == llm.DEFAULT_MODEL_SPEC
