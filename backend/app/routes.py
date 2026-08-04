"""HTTP endpoints for the job API."""
import os
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from pdf_bookmarker import llm

MAX_SIZE = 50 * 1024 * 1024  # 50 MB
VALID_MODES = {"auto", "always", "never"}

# The server runs no LLM of its own: the heading model produces the outline and
# verification is opt-in. On a CPU host the local GGUF costs minutes per
# document for roughly 3 title F1, which is not a trade worth making by default.
#
# Set VERIFICATION_MODEL to put the deployment back in charge — a local GGUF
# ("local:models/outline.gguf", worthwhile on a GPU host) or a cloud model. A
# caller who brings their own API key can always select one per request.
SERVER_MODEL_SPEC = os.environ.get("VERIFICATION_MODEL", "")

# Bound OCR cost on the free tier: scanned PDFs longer than this are rejected.
OCR_MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "50"))

router = APIRouter(prefix="/api")


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # The last entry is appended by the trusted platform proxy (one hop
        # on Render); everything to its left is client-controlled.
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@router.post("/jobs", status_code=202)
async def create_job(
    request: Request,
    file: UploadFile,
    llm_mode: str = Form("auto"),
    model: str | None = Form(None),
    api_key: str | None = Form(None),
):
    if llm_mode not in VALID_MODES:
        raise HTTPException(400, "llm_mode must be auto, always or never.")

    # Buffered fully in memory (capped at MAX_SIZE = 50 MB); an accepted
    # tradeoff at free-tier traffic levels.
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > MAX_SIZE:
            raise HTTPException(413, "File exceeds the 50 MB limit.")
    if not bytes(data[:5]) == b"%PDF-":
        raise HTTPException(400, "This file is not a PDF.")

    if not request.app.state.limiter.allow(client_ip(request)):
        raise HTTPException(429, "Rate limit exceeded — try again later.")

    # The server decides the model; a caller may only override it when they
    # bring their own API key. A model sent without a key is ignored.
    if api_key:
        model_spec = model or SERVER_MODEL_SPEC or llm.DEFAULT_MODEL_SPEC
    else:
        model_spec = SERVER_MODEL_SPEC
        # Nothing to call: say so up front rather than letting the pipeline
        # build a backend for an empty spec and fail per job.
        if not model_spec:
            llm_mode = "never"

    store = request.app.state.jobs
    job = store.submit(
        bytes(data),
        file.filename or "document.pdf",
        llm_mode=llm_mode,
        model_spec=model_spec,
        api_key=api_key or None,
        ocr_mode="auto",
        ocr_max_pages=OCR_MAX_PAGES,
    )
    return {"job_id": job.id}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    body: dict = {"status": job.status}
    if job.error is not None:
        body["error"] = job.error
    if job.bookmark_count is not None:
        body["bookmark_count"] = job.bookmark_count
    return body


@router.get("/jobs/{job_id}/download")
async def download(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "Unknown, unfinished, or expired job.")
    filename = Path(job.original_name).stem + ".bookmarked.pdf"
    return FileResponse(job.output_path, media_type="application/pdf",
                        filename=filename)
