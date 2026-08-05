from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from ai_agents.agents.coding.coding_agent_settings import settings as coding_settings
from ai_agents.config.runtime_configuration import (
    ChatProvider,
    runtime_agent_configuration,
)
from ai_agents.config.settings import settings


def _require_api_key(provider: ChatProvider) -> str:
    api_key = runtime_agent_configuration.provider_api_key(provider)
    if not api_key:
        raise RuntimeError(
            f"No API key is configured for provider '{provider}'. "
            "Set it in Agent configuration, an environment variable, or Secrets Manager."
        )
    return api_key


def build_chat_model(
    *,
    provider: ChatProvider,
    model_name: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    api_key = _require_api_key(provider)
    optional: dict[str, Any] = {
        "max_retries": 1,
        "timeout": coding_settings.model_timeout_seconds,
    }

    # Current Claude reasoning models can reject non-default sampling parameters.
    if temperature is not None and provider != "anthropic":
        optional["temperature"] = temperature
    if max_tokens is not None:
        optional["max_tokens"] = max_tokens

    if provider == "groq":
        return ChatGroq(model=model_name, api_key=api_key, **optional)

    if provider == "anthropic":
        return ChatAnthropic(model=model_name, api_key=api_key, **optional)

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=runtime_agent_configuration.provider_base_url(provider),
        **optional,
    )


def coding_model(*, max_tokens: int | None = None) -> BaseChatModel:
    """Fast model used for bounded planning and optional routing/navigation."""

    return build_chat_model(
        provider=settings.coding_provider,
        model_name=settings.coding_model,
        max_tokens=max_tokens,
    )


def reasoning_model(*, max_tokens: int | None = None) -> BaseChatModel:
    """Higher-quality model reserved for patch generation and repair loops."""

    return build_chat_model(
        provider=settings.reasoning_provider,
        model_name=settings.reasoning_model,
        max_tokens=max_tokens,
    )


def caption_model() -> BaseChatModel:
    return build_chat_model(
        provider=settings.caption_provider,
        model_name=settings.caption_model,
    )


def voice_chat_model(
    *,
    temperature: float = 0.5,
    max_tokens: int | None = None,
) -> BaseChatModel:
    return build_chat_model(
        provider=settings.voice_chat_provider,
        model_name=settings.voice_chat_model,
        temperature=temperature,
        max_tokens=max_tokens or settings.voice_chat_max_tokens,
    )
