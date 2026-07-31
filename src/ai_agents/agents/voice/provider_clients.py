from __future__ import annotations

from openai import OpenAI

from ai_agents.config.runtime_configuration import (
    ChatProvider,
    runtime_agent_configuration,
)


AudioProvider = ChatProvider
_SUPPORTED_AUDIO_PROVIDERS = frozenset({"groq", "openai"})


def build_audio_client(provider: AudioProvider) -> OpenAI:
    """Build an OpenAI-compatible audio client for the selected provider.

    Groq and OpenAI expose compatible transcription and speech endpoints. Keep STT
    and TTS clients separate because the user may route those capabilities to
    different providers.
    """
    if provider not in _SUPPORTED_AUDIO_PROVIDERS:
        raise RuntimeError(
            f"Provider '{provider}' is not configured for speech-to-text or text-to-speech."
        )

    api_key = runtime_agent_configuration.provider_api_key(provider)
    if not api_key:
        raise RuntimeError(
            f"No API key is configured for audio provider '{provider}'. "
            "Set it in Agent configuration, an environment variable, or Secrets Manager."
        )

    return OpenAI(
        api_key=api_key,
        base_url=runtime_agent_configuration.provider_base_url(provider),
        max_retries=2,
    )
