from ai_agents.agents.coding.utils.model_budget import (
    configured_context_window,
    configured_max_output_tokens,
    estimate_tokens,
    fit_blocks_to_token_budget,
    resolve_model_profile,
)


def test_model_profile_clamps_input_to_concrete_model_window() -> None:
    profile = resolve_model_profile(
        provider="deepseek",
        model_name="deepseek-v4-pro",
        context_window_tokens=131_072,
        requested_output_tokens=20_000,
        configured_max_input_tokens=96_000,
        reserve_tokens=10_000,
        safety_tokens=6_000,
    )

    assert profile.max_input_tokens == 95_072


def test_model_context_override_is_provider_and_model_specific() -> None:
    resolved = configured_context_window(
        provider="openrouter",
        model_name="vendor/model",
        fallback_tokens=131_072,
        overrides_json='{"openrouter:vendor/model": 262144}',
    )

    assert resolved == 262_144


def test_context_blocks_are_kept_whole_in_priority_order() -> None:
    first = "a " * 100
    second = "b " * 10_000
    third = "c " * 100

    first_tokens = estimate_tokens(first)
    third_tokens = estimate_tokens(third)
    included, omitted, used = fit_blocks_to_token_budget(
        [first, second, third],
        max_tokens=first_tokens + third_tokens + 5,
    )

    assert included == [first.strip(), third.strip()]
    assert omitted
    assert used <= first_tokens + third_tokens + 5


def test_model_output_override_caps_requested_generation_budget() -> None:
    cap = configured_max_output_tokens(
        provider="openrouter",
        model_name="vendor/model",
        fallback_tokens=32_000,
        overrides_json='{"openrouter:vendor/model": 8192}',
    )

    assert cap == 8_192
