from __future__ import annotations

import os
import secrets
import time
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

        if request.method == "OPTIONS":
            return await call_next(request)
        
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


_websocket_tokens: dict[str, float] = {}
MAX_TOKEN_AGE_SECONDS = 60


def generate_websocket_token() -> str:
    now = time.monotonic()
    _cleanup_expired_tokens(now)
    token = secrets.token_urlsafe(32)
    expiry = now + MAX_TOKEN_AGE_SECONDS
    _websocket_tokens[token] = expiry
    return token


def _cleanup_expired_tokens(now: float) -> None:
    expired = [t for t, exp in _websocket_tokens.items() if exp <= now]
    for t in expired:
        del _websocket_tokens[t]


async def authorize_websocket(websocket: WebSocket) -> bool:
    """
    Browser WebSocket clients cannot set arbitrary headers, so allow either:
    - x-api-key header for non-browser clients
    - api_key query parameter for the Electron/browser renderer
    - token query parameter (short-lived websocket token) for frontend clients
    """
    token = websocket.query_params.get("token")

    if token:
        now = time.monotonic()
        stored_expiry = _websocket_tokens.get(token)
        if stored_expiry is None or stored_expiry <= now:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Invalid or expired token",
            )
            return False
        # single-use token: remove after successful validation
        del _websocket_tokens[token]
        _cleanup_expired_tokens(now)
        return True

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




