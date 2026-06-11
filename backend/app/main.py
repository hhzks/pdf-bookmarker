"""App factory: wiring, CORS, and the periodic cleanup loop."""
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .jobs import JobStore
from .ratelimit import RateLimiter
from .routes import router

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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(_cleanup_loop(app))
        yield
        task.cancel()

    app = FastAPI(title="pdf-bookmarker", lifespan=lifespan)
    app.state.jobs = JobStore(ttl_seconds=ttl_seconds)
    app.state.limiter = RateLimiter(rate_limit_per_hour)
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(router)
    return app


async def _cleanup_loop(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        app.state.jobs.cleanup_expired()
        app.state.limiter.cleanup_expired()


app = create_app()
