from __future__ import annotations

import os
import secrets
from dotenv import load_dotenv

from fastapi import Request, WebSocket, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ai_agents.config.settings import settings

load_dotenv()


def resolved_api_key() -> str | None:
    return os.environ.get("AI_AGENTS_API_KEY") or settings.resolved_ai_agents_api_key()


def is_valid_api_key(provided: str | None) -> bool:
    expected = resolved_api_key()

    if not expected or not provided:
        return False

    return secrets.compare_digest(provided, expected)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Allow health checks/docs without auth.
        if request.url.path in {"/health", "/docs", "/openapi.json"}:
            return await call_next(request)

        expected = resolved_api_key()
        provided = request.headers.get("x-api-key")

        if not expected:
            return JSONResponse(
                status_code=500,
                content={"detail": "AI_AGENTS_API_KEY is not configured"},
            )

        if not is_valid_api_key(provided):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )

        return await call_next(request)


async def authorize_websocket(websocket: WebSocket) -> bool:
    """
    Browser WebSocket clients cannot set arbitrary headers, so allow either:
    - x-api-key header for non-browser clients
    - api_key query parameter for the Electron/browser renderer
    """
    expected = resolved_api_key()

    if not expected:
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR,
            reason="AI_AGENTS_API_KEY is not configured",
        )
        return False

    provided = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")

    if not is_valid_api_key(provided):
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Unauthorized",
        )
        return False

    return True




