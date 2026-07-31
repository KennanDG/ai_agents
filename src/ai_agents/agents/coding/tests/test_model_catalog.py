from __future__ import annotations

from ai_agents.config.model_catalog import discover_models, provider_supports
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
