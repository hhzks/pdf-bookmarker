"""In-memory job store running the PDF pipeline on a thread pool."""
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from pdf_bookmarker import llm, pipeline
from pdf_bookmarker.pipeline import process_pdf


@dataclass
class Job:
    id: str
    original_name: str
    dir: Path
    created_at: float
    status: str = "queued"  # queued | processing | done | failed
    error: str | None = None
    bookmark_count: int | None = None

    @property
    def input_path(self) -> Path:
        return self.dir / "input.pdf"

    @property
    def output_path(self) -> Path:
        return self.dir / "output.pdf"


_FRIENDLY: list[tuple[type[Exception], str]] = [
    (pipeline.InvalidPdfError, "This file could not be read as a PDF."),
    (pipeline.EncryptedPdfError,
     "This PDF is password-protected. Remove the encryption and try again."),
    (pipeline.NoTextLayerError,
     "This PDF appears to be scanned — it has no text layer, so headings "
     "cannot be detected."),
    (pipeline.NoOutlineError,
     "No table of contents or headings could be detected in this PDF."),
    (pipeline.LLMVerificationError,
     "LLM verification failed. Check the API key and model, or set LLM mode "
     "to Auto or Never and retry."),
    (llm.UnknownProviderError, "Unknown LLM provider in the model selection."),
]


def friendly_error(exc: Exception) -> str:
    for exc_type, message in _FRIENDLY:
        if isinstance(exc, exc_type):
            return message
    return "Processing failed unexpectedly. Please try again."


class JobStore:
    def __init__(self, ttl_seconds: int = 3600, max_workers: int = 2):
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        pdf_bytes: bytes,
        original_name: str,
        *,
        llm_mode: str,
        model_spec: str,
        api_key: str | None,
    ) -> Job:
        job_dir = Path(tempfile.mkdtemp(prefix="pdfjob-"))
        job = Job(id=uuid.uuid4().hex, original_name=original_name,
                  dir=job_dir, created_at=time.time())
        job.input_path.write_bytes(pdf_bytes)
        with self._lock:
            self._jobs[job.id] = job
        # The api_key travels only as a call argument: it is never stored on
        # the job record and never logged.
        self._pool.submit(self._run, job, llm_mode, model_spec, api_key)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cleanup_expired(self, now: float | None = None) -> None:
        """Drop expired jobs and delete their files.

        done/failed jobs expire after the TTL. Jobs that never reached a
        terminal state (a hung worker, or queued forever under a full pool)
        are reclaimed after twice the TTL so abandoned temp dirs cannot
        accumulate. If a directory cannot be removed (Windows file locks,
        e.g. a download still streaming), the job is kept so the next pass
        retries; a download that starts just before expiry may still fail
        mid-stream — accepted, the file is already an hour stale by then.
        """
        now = time.time() if now is None else now
        with self._lock:
            expired = [
                job for job in self._jobs.values()
                if (job.status in ("done", "failed")
                    and now - job.created_at > self._ttl)
                or now - job.created_at > 2 * self._ttl
            ]
            for job in expired:
                del self._jobs[job.id]
        for job in expired:
            shutil.rmtree(job.dir, ignore_errors=True)
            if job.dir.exists():  # locked on Windows; retry next pass
                with self._lock:
                    self._jobs[job.id] = job

    def _run(self, job: Job, llm_mode: str, model_spec: str,
             api_key: str | None) -> None:
        # Job fields are published lockless: this worker thread writes them,
        # request threads poll them. CPython's GIL makes the attribute stores
        # atomic and visible; the terminal status is always written last.
        job.status = "processing"
        try:
            result = process_pdf(
                job.input_path, job.output_path,
                llm_mode=llm_mode, model_spec=model_spec, api_key=api_key,
            )
        except Exception as exc:
            job.error = friendly_error(exc)
            job.status = "failed"
        else:
            job.bookmark_count = result.bookmark_count
            job.status = "done"
