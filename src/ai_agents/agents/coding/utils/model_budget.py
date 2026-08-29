from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Iterable


_TOKENISH_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class ModelExecutionProfile:
    provider: str
    model_name: str
    context_window_tokens: int
    requested_output_tokens: int
    reserve_tokens: int
    safety_tokens: int
    max_input_tokens: int


def estimate_tokens(text: str) -> int:
    """Conservatively estimate prompt tokens without adding a tokenizer dependency.

    Code is punctuation-heavy, so bytes/3.2 is intentionally more conservative than
    the common prose chars/4 heuristic. The lexical estimate protects short,
    symbol-dense snippets. Exact provider tokenizers can still reject an over-limit
    request, so callers retain a safety reserve.
    """

    if not text:
        return 0

    byte_estimate = math.ceil(len(text.encode("utf-8")) / 3.2)
    lexical_estimate = math.ceil(len(_TOKENISH_RE.findall(text)) * 1.15)

    return max(1, byte_estimate, lexical_estimate)


def configured_context_window(
    *,
    provider: str,
    model_name: str,
    fallback_tokens: int,
    overrides_json: str = "{}",
) -> int:
    """Resolve an exact provider/model override or use the conservative slot fallback."""

    try:
        raw = json.loads(overrides_json or "{}")
    except (TypeError, json.JSONDecodeError):
        raw = {}

    if isinstance(raw, dict):
        keys = (
            f"{provider}:{model_name}",
            model_name,
        )

        for key in keys:
            value = raw.get(key)
            if isinstance(value, int) and value > 0:
                return value

    return max(1, int(fallback_tokens))



def configured_max_output_tokens(
    *,
    provider: str,
    model_name: str,
    fallback_tokens: int,
    overrides_json: str = "{}",
) -> int:
    """Resolve a model-specific output ceiling without hardcoding a stale catalog."""

    try:
        raw = json.loads(overrides_json or "{}")
    except (TypeError, json.JSONDecodeError):
        raw = {}

    if isinstance(raw, dict):
        for key in (f"{provider}:{model_name}", model_name):
            value = raw.get(key)
            if isinstance(value, int) and value > 0:
                return value

    return max(1, int(fallback_tokens))

def resolve_model_profile(
    *,
    provider: str,
    model_name: str,
    context_window_tokens: int,
    requested_output_tokens: int,
    configured_max_input_tokens: int,
    reserve_tokens: int,
    safety_tokens: int,
) -> ModelExecutionProfile:
    """Resolve the usable input budget for one concrete provider/model slot."""

    window = max(1, int(context_window_tokens))
    output = max(1, int(requested_output_tokens))
    reserve = max(0, int(reserve_tokens))
    safety = max(0, int(safety_tokens))

    available = max(1, window - output - reserve - safety)
    max_input = max(1, min(int(configured_max_input_tokens), available))

    return ModelExecutionProfile(
        provider=provider,
        model_name=model_name,
        context_window_tokens=window,
        requested_output_tokens=output,
        reserve_tokens=reserve,
        safety_tokens=safety,
        max_input_tokens=max_input,
    )


def fit_blocks_to_token_budget(
    blocks: Iterable[str],
    *,
    max_tokens: int,
) -> tuple[list[str], list[str], int]:
    """Keep exact context blocks whole and in priority order."""

    included: list[str] = []
    omitted: list[str] = []
    used = 0
    seen: set[str] = set()

    for raw in blocks:
        block = str(raw).strip()
        if not block or block in seen:
            continue

        seen.add(block)
        tokens = estimate_tokens(block)
        
        if used + tokens > max_tokens:
            first_line = block.splitlines()[0][:160] if block else "(empty)"
            omitted.append(first_line)
            continue

        included.append(block)
        used += tokens

    return included, omitted, used
