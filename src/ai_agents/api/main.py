from __future__ import annotations

from fastapi import FastAPI

from ai_agents.api.routers.health import router as health_router
from ai_agents.api.routers.rag import router as rag_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="ai_agents API",
        version="0.1.0",
    )

    app.include_router(health_router)
    app.include_router(rag_router)

    return app


app = create_app()