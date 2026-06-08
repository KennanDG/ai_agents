# Skill: Implement Change

Purpose: Make a focused, low-risk code change using existing project patterns.

Use when:
- The user asks for a small feature, refactor, or fix.
- The task is localized to a few files.

Repository operations are executed by the LangGraph runner. Do not call tools directly. Return structured output only.

Steps:
1. Inspect relevant files.
2. Identify the smallest safe change.
3. Preserve existing style and architecture.
4. Update or add tests when appropriate.
5. Run targeted validation.
6. Report files changed and validation results.

Rules:
- Avoid broad rewrites.
- Do not touch secrets or unrelated files.
- Prefer small patches.