import time

import pytest
from fastapi.testclient import TestClient

from pdf_bookmarker.pipeline import NoTextLayerError

from app import jobs as jobs_module
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


def test_options_forwarded_to_pipeline(client, fake_pipeline):
    upload(client, llm_mode="always", model="gemini:gemini-3.5-flash",
           api_key="user-secret")
    deadline = time.time() + 10
    while not fake_pipeline and time.time() < deadline:
        time.sleep(0.02)
    assert fake_pipeline[0]["llm_mode"] == "always"
    assert fake_pipeline[0]["model_spec"] == "gemini:gemini-3.5-flash"
    assert fake_pipeline[0]["api_key"] == "user-secret"


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
