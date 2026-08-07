import uuid
import re
from pathlib import Path
import threading
from typing import Any, Literal, Mapping

from ai_agents.agents.coding.coding_agent_settings import settings as default_coding_settings



###############################################################################################
############################################# API #############################################
###############################################################################################


################ SCHEMAS ################
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
MAX_SKILL_CHARS = 50_000
MAX_TOOL_CHARS = 100_000



################ VOICE_AGENT ################
MAX_VOICE_ATTACHMENTS = 5
MAX_VOICE_ATTACHMENT_CONTENT_CHARS = 20_000
MAX_TOTAL_VOICE_ATTACHMENT_CONTENT_CHARS = 60_000
MAX_VOICE_SKILL_CONTEXT_CHARS = 3_600
VOICE_SKILLS_DIR = Path(__file__).resolve().parents[2] / "agents" / "voice" / "skills"



################ GITHUB ################
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
IMPORT_LOCK = threading.RLock()
AUTO_STASH_PREFIX = "ai-agents:auto-stash:"






################ CODING_AGENT ################
IGNORED_REPOSITORY_FILES = {
    ".DS_Store",
    "Thumbs.db",
}

IGNORED_REPOSITORY_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
MAX_REPOSITORY_FILE_BYTES = 1_000_000
MAX_ATTACHED_FILES = default_coding_settings.max_attached_files
MAX_ATTACHMENT_CHARS = default_coding_settings.max_attachment_storage_chars
MAX_TOTAL_ATTACHMENT_CHARS = default_coding_settings.max_total_attachment_storage_chars
MAX_ATTACHED_IMAGE_BYTES = 5_000_000
ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
IMAGE_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>image/(?:png|jpeg|jpg|webp));base64,(?P<data>[A-Za-z0-9+/=\r\n]+)$",
    re.IGNORECASE,
)
LANGUAGE_BY_EXTENSION = {
    ".css": "css",
    ".html": "html",
    ".js": "javascript",
    ".jsx": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".sql": "sql",
    ".ts": "typescript",
    ".tsx": "typescript",
    # ".txt": "plaintext",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".java": "java",
}





################ ADMIN ################
AI_AGENTS_ROOT = Path(__file__).resolve().parents[2]


CUSTOM_PREFIX = "custom_"

CODING_RUNTIME_FIELDS = (
    "coding_subagent_count",
    "coding_route_max_tokens",
    "coding_planner_max_tokens",
    "coding_repo_navigation_max_tokens",
    "coding_simple_patch_max_tokens",
    "coding_patch_max_tokens",
    "coding_progress_max_tokens",
)
CODING_RUNTIME_BOUNDS = {
    "coding_subagent_count": (1, 6),
    "coding_route_max_tokens": (256, 2_000),
    "coding_planner_max_tokens": (512, 6_000),
    "coding_repo_navigation_max_tokens": (512, 4_000),
    "coding_simple_patch_max_tokens": (2_000, 16_000),
    "coding_patch_max_tokens": (4_000, 32_000),
    "coding_progress_max_tokens": (512, 4_000),
}
MODEL_CONFIGURATION_FIELDS = (
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




###############################################################################################
############################################# CONFIG ##########################################
###############################################################################################
ChatProvider = Literal["groq", "deepseek", "openrouter", "openai", "anthropic", "google"]
ModelCapability = Literal["chat", "vision", "stt", "tts"]
AgentKind = Literal["coding", "voice"]

PROVIDER_CAPABILITIES: dict[ChatProvider, frozenset[ModelCapability]] = {
    "groq": frozenset({"chat", "vision", "stt", "tts"}),
    "deepseek": frozenset({"chat"}),
    "openrouter": frozenset({"chat", "vision"}),
    "openai": frozenset({"chat", "vision", "stt", "tts"}),
    "anthropic": frozenset({"chat", "vision"}),
    "google": frozenset({"chat", "vision"}),
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
    "google": ("google_api_key", "GOOGLE_API_KEY", "google_secret_arn"),
}


PROVIDER_URL_FIELDS: dict[ChatProvider, str] = {
    "groq": "groq_api_url",
    "deepseek": "deepseek_api_url",
    "openrouter": "openrouter_api_url",
    "openai": "openai_api_url",
    "anthropic": "anthropic_api_url",
    "google": "google_api_url",
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
    ("google", "chat"): [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-pro",
    ],
    ("google", "vision"): [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-pro",
    ],
}



QDRANT_ID_NAMESPACE = uuid.UUID("4c0f9c2a-3db6-4f6d-9dd4-6c5c3c70a3f1")

