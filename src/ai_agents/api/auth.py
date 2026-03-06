from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from ai_agents.config.settings import settings


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Allow health checks without auth
        if request.url.path in {"/health", "/docs", "/openapi.json"}:
            return await call_next(request)

        expected = os.environ.get(("AI_AGENTS_API_KEY"), settings.resolved_ai_agents_api_key())
        provided = request.headers.get("x-api-key")

        if not expected:
            return JSONResponse(
                status_code=500,
                content={"detail": "AI_AGENTS_API_KEY is not configured"},
            )

        if not provided or not secrets.compare_digest(provided, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )

        return await call_next(request)