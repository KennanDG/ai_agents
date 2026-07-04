from __future__ import annotations

import re
from collections import Counter

STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "in", "on", "at", "to", "for", "of", "and", "or", "it", "that",
    "this", "these", "those", "with", "as", "by", "from", "but",
})


def summarize_text(text: str, max_length: int = 200) -> str:
    """Truncate text to a summary of at most max_length characters."""
    return text[:max_length]


def extract_keywords(text: str, num_keywords: int = 5) -> list[str]:
    """Extract the top keywords from text, excluding common stop words."""
    words = re.findall(r"\b\w+\b", text.lower())
    filtered = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    counter = Counter(filtered)
    return [word for word, _ in counter.most_common(num_keywords)]


def translate_text(text: str, target_language: str = "en") -> str:
    """Translate text to the target language (stub)."""
    # A real implementation would call a translation API.
    return f"[Translated to {target_language}]: {text}"
