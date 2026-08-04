"""App factory: wiring, CORS, and the periodic cleanup loop."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pdf_bookmarker.labeler import LabelerError
from pdf_bookmarker.pipeline import resolve_labeler

from .jobs import JobStore
from .ratelimit import RateLimiter
from . import routes
from .routes import router

logger = logging.getLogger("pdf_bookmarker.web")

CLEANUP_INTERVAL_SECONDS = 300


def create_app(
    *,
    ttl_seconds: int = 3600,
    rate_limit_per_hour: int = 10,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    if allowed_origins is None:
        allowed_origins = [
            origin.strip()
            for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",")
            if origin.strip()
        ]

    startup_notes = [
        (logging.INFO, _load_labeler()),
        _check_verification_model(),
    ]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Logged here, not in create_app: that runs at import time, before
        # uvicorn installs its logging config, and anything emitted then goes
        # nowhere.
        for level, message in app.state.startup_notes:
            logger.log(level, message)
        task = asyncio.create_task(_cleanup_loop(app))
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title="pdf-bookmarker", lifespan=lifespan)
    app.state.startup_notes = startup_notes
    app.state.jobs = JobStore(ttl_seconds=ttl_seconds)
    app.state.limiter = RateLimiter(rate_limit_per_hour)
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        logger.warning(
            "ALLOWED_ORIGINS is not set; browsers on other origins will be "
            "blocked by CORS"
        )
    app.include_router(router)
    return app


def _load_labeler() -> str:
    """Resolve the configured line labeler at boot: validate it and warm it.

    The pipeline would load it lazily on the first job, which puts a
    misconfigured path behind an upload — every request failing with a message
    about a model the user never asked for. Failing at startup instead makes
    the operator's mistake obvious to the operator. It also pays the unpickling
    cost once, before any request is waiting on it.

    Returns the line to log once logging is configured (see lifespan).
    """
    try:
        model = resolve_labeler(None)
    except LabelerError as exc:
        raise RuntimeError(
            f"PDF_BOOKMARKER_LABELER is set but the model cannot be used: {exc}"
        ) from exc
    if model is None:
        return (
            "no line labeler configured (PDF_BOOKMARKER_LABELER); the pipeline "
            "will use TOC parsing and font heuristics"
        )
    return f"line labeler loaded from {os.environ.get('PDF_BOOKMARKER_LABELER')}"


def _check_verification_model() -> tuple[int, str]:
    """Report on the server's LLM, without loading it.

    Only the file's existence is checked. Loading a multi-gigabyte GGUF here
    would tie the web app to the [local] extra and delay every boot, and the
    first job that needs it pays the load once anyway (llm.get_backend caches
    it). Missing is a warning, not a failure: unlike the labeler this is a
    shipped default rather than something an operator asked for, and auto mode
    already degrades to the heuristic outline when the LLM cannot be built.
    """
    spec = routes.SERVER_MODEL_SPEC  # read now, not at import: tests and
    provider, _, model = spec.partition(":")  # embedders can set it
    if provider != "local":
        return logging.INFO, f"verification model: {spec}"
    if model and Path(model).is_file():
        return logging.INFO, f"verification model: local GGUF at {model}"
    return logging.WARNING, (
        f"the configured verification model {model or '(no path)'} does not "
        "exist; outlines will not be LLM-verified (set VERIFICATION_MODEL)"
    )


async def _cleanup_loop(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            app.state.jobs.cleanup_expired()
            app.state.limiter.cleanup_expired()
        except Exception:
            logger.exception("cleanup pass failed; will retry next interval")


# uvicorn entry point (app.main:app); creates the worker pool at import time.
#
# uvicorn's logging config only covers its own loggers and leaves the root
# unconfigured, so without this every INFO line this app emits is dropped and
# only WARNING+ escapes via logging's last-resort handler. basicConfig is a
# no-op when the host has already configured logging, so it cannot override a
# deployment's own setup.
logging.basicConfig(level=logging.INFO)
app = create_app()
