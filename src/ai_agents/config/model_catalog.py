from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from groq import Groq

from ai_agents.config.constants import (
    FALLBACK_MODELS,
    PROVIDER_CAPABILITIES,
    ModelCapability,
)
from ai_agents.config.runtime_configuration import (
    ChatProvider,
    runtime_agent_configuration,
)


class ModelCatalogError(RuntimeError):
    """Raised when a provider's model catalog cannot be retrieved."""


def provider_supports(provider: ChatProvider, capability: ModelCapability) -> bool:
    return capability in PROVIDER_CAPABILITIES[provider]


def _fallback(provider: ChatProvider, capability: ModelCapability) -> list[str]:
    return list(FALLBACK_MODELS.get((provider, capability), []))


def _groq_base_url() -> str:
    """Return a Groq SDK base URL ending in ``/openai/v1``.

    Older environment files sometimes configure ``GROQ_URL`` as only
    ``https://api.groq.com``. Appending ``/models`` to that value targets the wrong
    route. Normalize both the host-only and full OpenAI-compatible forms here.
    """
    base_url = runtime_agent_configuration.provider_base_url("groq").rstrip("/")

    if base_url.endswith("/models"):
        base_url = base_url[: -len("/models")]

    if base_url.endswith("/openai/v1"):
        return base_url
    if base_url.endswith("/openai"):
        return f"{base_url}/v1"

    return f"{base_url}/openai/v1"


def _models_url(provider: ChatProvider) -> str:
    if provider == "groq":
        return f"{_groq_base_url()}/models"

    base_url = runtime_agent_configuration.provider_base_url(provider).rstrip("/")
    if provider == "anthropic":
        return f"{base_url}/v1/models"    
    if provider == "google":
        return f"{base_url}/models"
    if provider == "openrouter":
        return f"{base_url}/models/user"
    return f"{base_url}/models"


def _headers(provider: ChatProvider, api_key: str | None) -> dict[str, str]:
    headers = {
        "accept": "application/json",
        "user-agent": "ai-agents-model-catalog/1.0",
    }
    if provider == "anthropic":
        if api_key:
            headers["x-api-key"] = api_key.strip()
        headers["anthropic-version"] = "2023-06-01"
    elif provider == "google":
        if api_key:
            headers["x-goog-api-key"] = api_key.strip()
    elif api_key:
        headers["authorization"] = f"Bearer {api_key.strip()}"
    return headers


def _sdk_item_dict(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return item

    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        return value if isinstance(value, dict) else None

    model_id = getattr(item, "id", None)
    if not isinstance(model_id, str) or not model_id.strip():
        return None

    return {
        "id": model_id,
        "active": getattr(item, "active", True),
        "owned_by": getattr(item, "owned_by", None),
    }


def _exception_detail(exc: Exception) -> str:
    """Extract provider error details without exposing the API key."""
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    body = getattr(exc, "body", None)

    parts: list[str] = []
    if status_code is not None:
        parts.append(f"HTTP {status_code}")

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code") or error.get("type")
            message = error.get("message")
            if code:
                parts.append(str(code))
            if message:
                parts.append(str(message))
        elif error:
            parts.append(str(error))
        elif body:
            parts.append(json.dumps(body, ensure_ascii=False))
    elif isinstance(body, str) and body.strip():
        parts.append(body.strip())

    if request_id:
        parts.append(f"request_id={request_id}")

    return ": ".join(parts) if parts else str(exc)


def _get_groq_json(api_key: str) -> dict[str, Any]:
    """Retrieve Groq models through the provider's supported Python client."""
    try:
        client = Groq(api_key=api_key.strip())
        page = client.models.list()
    except Exception as exc:
        raise ModelCatalogError(
            f"Groq model catalog request failed: {_exception_detail(exc)}"
        ) from exc

    raw_data = getattr(page, "data", None)
    if raw_data is None and isinstance(page, dict):
        raw_data = page.get("data")
    if not isinstance(raw_data, list):
        try:
            raw_data = list(raw_data or [])
        except TypeError as exc:
            raise ModelCatalogError(
                "Groq returned an unreadable model catalog response."
            ) from exc

    data = [
        item_dict
        for item in raw_data
        if (item_dict := _sdk_item_dict(item)) is not None
    ]
    return {"object": "list", "data": data}


def _http_error_detail(exc: HTTPError) -> str:
    body_text = ""
    try:
        body_text = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body_text = ""

    if body_text:
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            body = None

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                code = error.get("code") or error.get("type")
                message = error.get("message")
                detail = ": ".join(str(value) for value in (code, message) if value)
                if detail:
                    return f"HTTP {exc.code}: {detail}"
            return f"HTTP {exc.code}: {json.dumps(body, ensure_ascii=False)}"

        return f"HTTP {exc.code}: {body_text[:500]}"

    return f"HTTP {exc.code}: {exc.reason}"


def _get_json(provider: ChatProvider, api_key: str | None) -> dict[str, Any]:
    if provider == "groq":
        if not api_key:
            raise ModelCatalogError("Groq API key is not configured.")
        return _get_groq_json(api_key)

    request = Request(
        _models_url(provider),
        headers=_headers(provider, api_key),
        method="GET",
    )
    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310 - fixed provider URLs
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ModelCatalogError(_http_error_detail(exc)) from exc

    if not isinstance(payload, dict):
        raise ValueError("Provider returned a non-object model catalog.")
    return payload


def _model_id(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    raw = item.get("id") or item.get("name")
    if not isinstance(raw, str) or not raw.strip():
        return None
    if item.get("active") is False:
        return None
    return raw.strip()


def _openrouter_modalities(item: dict[str, Any]) -> tuple[set[str], set[str]]:
    architecture = item.get("architecture")
    if not isinstance(architecture, dict):
        architecture = {}

    raw_inputs = (
        architecture.get("input_modalities")
        or item.get("input_modalities")
        or []
    )
    raw_outputs = (
        architecture.get("output_modalities")
        or item.get("output_modalities")
        or []
    )

    inputs = {
        str(value).strip().lower()
        for value in raw_inputs
        if isinstance(value, str) and value.strip()
    }
    outputs = {
        str(value).strip().lower()
        for value in raw_outputs
        if isinstance(value, str) and value.strip()
    }
    return inputs, outputs


def _matches_capability(
    provider: ChatProvider,
    capability: ModelCapability,
    item: dict[str, Any],
    model_id: str,
) -> bool:
    lowered = model_id.lower()

    if provider == "anthropic":
        return capability in {"chat", "vision"}    
    if provider == "google":
        return capability in {"chat", "vision"}

    if provider == "deepseek":
        return capability == "chat"

    if provider == "openrouter":
        inputs, outputs = _openrouter_modalities(item)
        if capability == "vision":
            return "image" in inputs and (not outputs or "text" in outputs)
        return capability == "chat" and (not inputs or "text" in inputs)

    if capability == "stt":
        return "whisper" in lowered or "transcribe" in lowered

    if capability == "tts":
        return any(token in lowered for token in ("tts", "speech", "orpheus")) and not any(
            token in lowered for token in ("transcribe", "whisper")
        )

    audio_only = any(
        token in lowered
        for token in ("whisper", "transcribe", "tts", "orpheus", "embedding", "moderation")
    )
    if audio_only:
        return False

    if capability == "vision":
        if provider == "groq":
            return any(
                token in lowered
                for token in ("vision", "scout", "maverick", "qwen3.6")
            )
        if provider == "openai":
            return lowered.startswith(("gpt-", "o", "chatgpt-")) and not any(
                token in lowered for token in ("realtime", "audio", "image")
            )

    return capability == "chat"


def _extract_models(
    provider: ChatProvider,
    capability: ModelCapability,
    payload: dict[str, Any],
) -> list[str]:
    raw_data = payload.get("data")
    if not isinstance(raw_data, list):
        raw_data = payload.get("models")
    if not isinstance(raw_data, list):
        return []

    models: list[str] = []
    for item in raw_data:
        model_id = _model_id(item)
        if model_id is None or not isinstance(item, dict):
            continue
        if provider == "google" and model_id.startswith("models/"):
            model_id = model_id.removeprefix("models/")
        if _matches_capability(provider, capability, item, model_id):
            models.append(model_id)

    return sorted(set(models), key=str.casefold)


def discover_models(
    provider: ChatProvider,
    capability: ModelCapability,
) -> dict[str, Any]:
    if not provider_supports(provider, capability):
        return {
            "provider": provider,
            "capability": capability,
            "models": [],
            "source": "fallback",
            "secret_configured": runtime_agent_configuration.secret_configured(provider),
            "error": f"{provider} does not support the {capability} capability in this application.",
        }

    api_key = runtime_agent_configuration.provider_api_key(provider)
    secret_configured = runtime_agent_configuration.secret_configured(provider)

    # Provider model catalogs are authenticated and can be account-specific. Return
    # a curated fallback until a credential is configured.
    if not api_key:
        return {
            "provider": provider,
            "capability": capability,
            "models": _fallback(provider, capability),
            "source": "fallback",
            "secret_configured": False,
            "error": "Configure this provider's API key to load its live model list.",
        }

    try:
        payload = _get_json(provider, api_key)
        models = _extract_models(provider, capability, payload)
        if not models:
            raise ValueError("The provider returned no compatible models.")
        return {
            "provider": provider,
            "capability": capability,
            "models": models,
            "source": "live",
            "secret_configured": secret_configured,
            "error": None,
        }
    except (
        ModelCatalogError,
        URLError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return {
            "provider": provider,
            "capability": capability,
            "models": _fallback(provider, capability),
            "source": "fallback",
            "secret_configured": secret_configured,
            "error": f"Live model discovery failed: {exc}",
        }