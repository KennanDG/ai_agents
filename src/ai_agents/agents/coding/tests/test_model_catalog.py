from __future__ import annotations

from ai_agents.config.model_catalog import provider_supports, _fallback, discover_models
from ai_agents.config.runtime_configuration import runtime_agent_configuration


def test_capability_matrix_excludes_anthropic_audio() -> None:
    assert provider_supports("anthropic", "chat")
    assert provider_supports("anthropic", "vision")
    assert not provider_supports("anthropic", "stt")
    assert not provider_supports("anthropic", "tts")


def test_unconfigured_provider_uses_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_agent_configuration,
        "provider_api_key",
        lambda provider: None,
    )
    monkeypatch.setattr(
        runtime_agent_configuration,
        "secret_configured",
        lambda provider: False,
    )

    result = discover_models("anthropic", "chat")

    assert result["source"] == "fallback"
    assert result["models"]
    assert result["secret_configured"] is False


def test_unsupported_capability_returns_no_models(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_agent_configuration,
        "secret_configured",
        lambda provider: False,
    )

    result = discover_models("anthropic", "tts")

    assert result["models"] == []
    assert "does not support" in result["error"]


    
def test_google_supports_chat() -> None:
    assert provider_supports("google", "chat") is True


def test_google_supports_vision() -> None:
    assert provider_supports("google", "vision") is True


def test_google_does_not_support_stt() -> None:
    assert provider_supports("google", "stt") is False


def test_google_fallback_chat() -> None:
    models = _fallback("google", "chat")
    assert len(models) > 0
    assert "gemini-2.0-flash" in models


def test_google_fallback_vision() -> None:
    models = _fallback("google", "vision")
    assert len(models) > 0
