from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_agents.api.routers.health import router as health_router
from ai_agents.api.routers.rag import router as rag_router
from ai_agents.api.routers.coding_agent import router as coding_agent_router
from ai_agents.api.auth import ApiKeyMiddleware


def create_app() -> FastAPI:

    app = FastAPI(
        title="ai_agents API",
        version="0.1.0",
    )

    # Middleware
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health_router)
    app.include_router(rag_router)
    app.include_router(coding_agent_router)

    return app


app = create_app()


