from __future__ import annotations


def skill_instructions_for_llm(skill_instructions: str) -> str:
    """
    Skill files list allowed tools for the graph runner.

    The model itself is not a tool-calling agent in this workflow. If we pass
    the raw "Allowed tools" section to the LLM, some models try to call those
    tool names directly, which fails because they are not bound in request.tools.
    """
    if not skill_instructions:
        return ""

    lines = skill_instructions.splitlines()
    cleaned: list[str] = []
    skipping_allowed_tools = False

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("allowed tools"):
            skipping_allowed_tools = True
            cleaned.append(
                "Repository operations are executed by the LangGraph runner. "
                "Do not call tools directly. Return structured output only."
            )
            continue

        if skipping_allowed_tools:
            if not stripped:
                skipping_allowed_tools = False
                continue

            if stripped.startswith("-"):
                continue

            skipping_allowed_tools = False

        cleaned.append(line)

    return "\n".join(cleaned).strip()
