"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from backend_app.api.routes import inference_router, router as api_router
from backend_app.core.config import get_settings


def create_app() -> FastAPI:
    """Create the FastAPI application instance."""

    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(api_router, prefix="/api")
    app.include_router(inference_router)
    return app


app = create_app()
