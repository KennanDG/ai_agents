from __future__ import annotations

LLM_DECISION_MAX_ATTEMPTS = 3
MAX_PATCH_ATTEMPTS = 2
MAX_FILES_TO_INSPECT = 10
VALIDATION_OUTPUT_MAX_CHARS = 2_000
VALIDATION_PROFILE_NAME = "coding-agent-default"




#################################### Search ####################################
IGNORED_CONTEXT_PATH_PARTS = (
    ("logs", "runs"),
)

DEFAULT_EXCLUDED_PATH_HINTS = [
    "agents/coding/logs/runs",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "code",
    "for",
    "from",
    "help",
    "how",
    "i",
    "in",
    "into",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "repo",
    "repository",
    "should",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "we",
    "with",
    "you",
}


TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".sql",
    ".sh",
    ".tf",
    ".dockerfile",
}


PYTHON_SYMBOL_KINDS = {
    "class",
    "function",
    "async_function",
    "import",
    "constant",
}