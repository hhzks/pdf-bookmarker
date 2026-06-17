"""HTTP endpoints for the job API."""
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from pdf_bookmarker.llm import DEFAULT_MODEL_SPEC

MAX_SIZE = 50 * 1024 * 1024  # 50 MB
VALID_MODES = {"auto", "always", "never"}

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

    store = request.app.state.jobs
    job = store.submit(
        bytes(data),
        file.filename or "document.pdf",
        llm_mode=llm_mode,
        model_spec=model or DEFAULT_MODEL_SPEC,
        api_key=api_key or None,
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
