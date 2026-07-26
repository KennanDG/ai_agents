MAX_REPO_FILES = 700
MAX_TREE_FILES = 250
MAX_SEARCH_MATCHES = 10
MAX_FILE_BYTES = 250_000
MAX_EXPLICIT_FILE_CHARS = 8_000
MAX_ATTACHMENT_CONTENT_CHARS = 6_000
MAX_TOTAL_ATTACHMENT_CONTENT_CHARS = 18_000
MAX_CONTEXT_JSON_CHARS = 16_000
MAX_LLM_TREE_PATHS = 60
MAX_LLM_EXPLICIT_FILE_CHARS = 1_500
MAX_LLM_SEARCH_EXCERPT_CHARS = 900
MAX_LLM_ATTACHMENT_EXCERPT_CHARS = 750





IGNORED_DIRS = {
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
    "agents/coding/logs"
}

TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".css",
    ".csv",
    ".h",
    ".hh",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

STOP_WORDS = {
    "about",
    "agent",
    "attached",
    "coding",
    "could",
    "files",
    "from",
    "have",
    "into",
    "please",
    "should",
    "that",
    "their",
    "this",
    "update",
    "voice",
    "with",
    "would",
}


CLARIFICATION_TOPIC_ORDER = (
    "objective",
    "current_behavior",
    "scope",
    "environment",
    "constraints",
    "acceptance_criteria",
    "priority",
)

CLARIFICATION_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "current_behavior": (
        "current behavior",
        "currently",
        "error",
        "failure",
        "failing",
        "slow",
        "slowdown",
        "repeating",
        "false result",
        "symptom",
    ),
    "scope": (
        "scope",
        "only this file",
        "related files",
        "other modules",
        "stay inside",
        "limit the change",
    ),
    "environment": (
        "environment",
        "runtime",
        "local",
        "ci",
        "deployment",
        "production",
        "operating system",
        "python version",
        "node version",
    ),
    "constraints": (
        "constraint",
        "must remain",
        "must preserve",
        "must not",
        "cannot change",
        "avoid changing",
        "keep unchanged",
    ),
    "acceptance_criteria": (
        "acceptance",
        "success",
        "done",
        "complete",
        "working correctly",
        "observable result",
        "verify",
    ),
    "priority": (
        "priority",
        "most important",
        "first pass",
        "matters most",
        "tradeoff",
    ),
    "objective": (
        "main outcome",
        "goal",
        "objective",
        "want to achieve",
        "desired result",
    ),
}



CLARIFICATION_FALLBACK_QUESTIONS: dict[str, str] = {
    "objective": "What is the main outcome you want this change to achieve?",
    "current_behavior": (
        "What concrete failure, slowdown, or false validation result are you seeing now?"
    ),
    "scope": (
        "Should the change stay in the current file, or may it update related helpers, "
        "settings, and tests?"
    ),
    "environment": (
        "Which local, CI, or deployment environments must the updated behavior support?"
    ),
    "constraints": "Which existing behavior or validation commands must remain unchanged?",
    "acceptance_criteria": (
        "What observable result should the coding agent use to consider this work complete?"
    ),
    "priority": (
        "For the first pass, which matters most: speed, fewer false failures, clearer "
        "diagnostics, or maintainability?"
    ),
}



QUESTION_FILLER_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "clarify",
    "could",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "like",
    "me",
    "of",
    "or",
    "please",
    "specific",
    "tell",
    "the",
    "this",
    "to",
    "want",
    "what",
    "which",
    "would",
    "you",
    "your",
}

QUESTION_REPEAT_THRESHOLD = 0.64

