import uuid
from typing import Any, Literal, Mapping

QDRANT_ID_NAMESPACE = uuid.UUID("4c0f9c2a-3db6-4f6d-9dd4-6c5c3c70a3f1")

ChatProvider = Literal["groq", "deepseek", "openrouter", "openai", "anthropic"]
ModelCapability = Literal["chat", "vision", "stt", "tts"]
AgentKind = Literal["coding", "voice"]

PROVIDER_CAPABILITIES: dict[ChatProvider, frozenset[ModelCapability]] = {
    "groq": frozenset({"chat", "vision", "stt", "tts"}),
    "deepseek": frozenset({"chat"}),
    "openrouter": frozenset({"chat", "vision"}),
    "openai": frozenset({"chat", "vision", "stt", "tts"}),
    "anthropic": frozenset({"chat", "vision"}),
}

PUBLIC_FIELDS = (
    "coding_provider",
    "coding_model",
    "reasoning_provider",
    "reasoning_model",
    "caption_provider",
    "caption_model",
    "voice_chat_provider",
    "voice_chat_model",
    "voice_stt_provider",
    "voice_stt_model",
    "voice_tts_provider",
    "voice_tts_model",
    "voice_tts_voice",
    "voice_tts_enabled",
)


PROVIDER_FIELDS: dict[ChatProvider, tuple[str, str, str | None]] = {
    "groq": ("groq_api_key", "GROQ_API_KEY", "groq_secret_arn"),
    "deepseek": ("deepseek_api_key", "DEEPSEEK_API_KEY", "deepseek_secret_arn"),
    "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY", "openrouter_secret_arn"),
    "openai": ("openai_api_key", "OPENAI_API_KEY", "openai_secret_arn"),
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY", "anthropic_secret_arn"),
}


PROVIDER_URL_FIELDS: dict[ChatProvider, str] = {
    "groq": "groq_api_url",
    "deepseek": "deepseek_api_url",
    "openrouter": "openrouter_api_url",
    "openai": "openai_api_url",
    "anthropic": "anthropic_api_url",
}



PROVIDER_SLOT_CAPABILITY: dict[str, ModelCapability] = {
    "coding_provider": "chat",
    "reasoning_provider": "chat",
    "caption_provider": "vision",
    "voice_chat_provider": "chat",
    "voice_stt_provider": "stt",
    "voice_tts_provider": "tts",
}






# These are only a fallback for an unconfigured provider or a temporary provider API
# failure. The live /models endpoint remains the source of truth whenever possible.
FALLBACK_MODELS: dict[tuple[ChatProvider, ModelCapability], list[str]] = {
    ("groq", "chat"): [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ],
    ("groq", "vision"): [
        "qwen/qwen3.6-27b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ],
    ("groq", "stt"): [
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "distil-whisper-large-v3-en",
    ],
    ("groq", "tts"): [
        "canopylabs/orpheus-v1-english",
        "canopylabs/orpheus-arabic-saudi",
    ],
    ("deepseek", "chat"): ["deepseek-v4-pro", "deepseek-v4-flash"],
    ("openrouter", "chat"): [
        "anthropic/claude-sonnet-5",
        "deepseek/deepseek-v4-pro",
        "openai/gpt-5.4",
    ],
    ("openrouter", "vision"): [
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.4",
    ],
    ("openai", "chat"): ["gpt-4.1", "gpt-4.1-mini", "o3"],
    ("openai", "vision"): ["gpt-4.1", "gpt-4.1-mini", "o3"],
    ("openai", "stt"): [
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "whisper-1",
    ],
    ("openai", "tts"): ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
    ("anthropic", "chat"): [
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-haiku-4-5",
    ],
    ("anthropic", "vision"): [
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-haiku-4-5",
    ],
}





