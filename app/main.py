#!/usr/bin/env python3
"""
Anot Medical ASR — FastAPI Application Server
=============================================
Stage 11: Production application entry point with lifespan model management.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.core.logging import logger, log_request_event
from app.security.privacy import generate_request_id
from app.services.asr_service import ASRService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Loads MedASR once at startup; cleans up resources on shutdown.
    """
    logger.info(f"Starting {settings.SERVICE_NAME} v{settings.VERSION} on {settings.HOST}:{settings.PORT}...")
    asr = ASRService.get_instance()
    try:
        asr.initialize()
        logger.info("Anot ASR engine initialized and ready for production traffic.")
    except Exception as e:
        logger.error(f"Critical error during MedASR startup: {e}")
        # Service continues running so /health and /ready report failure appropriately

    yield

    logger.info("Shutting down Anot ASR service...")
    try:
        asr.active_model().unload()
    except Exception as e:
        logger.warning(f"Notice during shutdown cleanup: {e}")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="Anot Medical ASR Service",
        description="Production-grade, private clinical speech-to-text service powered by Google MedASR.",
        version=settings.VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan
    )

    # CORS configuration. In production, restrict to the configured origins
    # (ANOT_CORS_ORIGINS, comma-separated, e.g. your web app's domain).
    cors_origins = (["*"] if settings.DEBUG else settings.cors_origins_list) or \
        ["http://localhost:3000", "http://127.0.0.1:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Custom exception handlers for safe, non-leaking error responses
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        req_id = getattr(request.state, "request_id", generate_request_id())
        log_request_event("http_error", req_id, {
            "status_code": exc.status_code,
            "path": request.url.path
        }, level=logger.level)

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": str(exc.detail),
                "request_id": req_id
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", generate_request_id())
        logger.error(f"Unhandled server error [request_id={req_id}]: {exc.__class__.__name__}")

        # Strictly conceal internal stack traces from clients
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal error occurred during audio processing.",
                "request_id": req_id
            }
        )

    # Include API routes
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
