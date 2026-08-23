# Skill: Repository Context Verification for Coding Handoff

Purpose: Verify and synthesize repository context before handing off a coding request to the coding agent. Inspect gathered repository evidence, identify relevant files and implementation areas, resolve ambiguities with one concise clarification if needed, and prepare a precise handoff with resolved objective, constraints, and validation expectations.

Use when:
- User gives a spoken coding request that references the current repository
- Request mentions an active file, attached files, or existing implementation
- Request refers to errors, bugs, or source code in the repository
- Clarification is needed before the coding agent can proceed safely
- Repository context must be grounded in facts before handoff

Allowed tools:
- custom_voice_context_audit
- robust_search
- search_repo
- list_files
- read_file
- resolve_in_repo
- file_size
- format_search_results

Steps:
1. Call custom_voice_context_audit to retrieve the compact audit of available repository context, including paths, files, search matches, and attachments.
2. Review the audit output to identify explicit files, implementation areas, and whether sufficient evidence exists to understand the request.
3. If the audit indicates missing critical context, use robust_search or search_repo to locate relevant files or implementation details mentioned in the user's request.
4. Synthesize repository-grounded facts: list relevant file paths, key implementation areas, constraints, and dependencies identified from the audit and search results.
5. Identify one unresolved detail that would materially change the implementation approach; if found, ask one concise clarification question and wait for the user's response.
6. Prepare a precise coding-agent handoff that includes: resolved objective, relevant file paths, important constraints, likely implementation steps, and validation expectations.
7. Communicate the handoff to the user in clear, actionable language suitable for transfer to the coding agent.

Rules:
- Treat custom_voice_context_audit output as read-only backend-generated evidence; do not assume additional context exists.
- Never invent repository paths, APIs, functions, dependencies, or tool names; use only paths and entities confirmed by the audit or search results.
- Prefer repository-grounded facts over assumptions; if a detail is uncertain, ask for clarification rather than guess.
- Do not imply that the voice model itself called a tool, opened a file, modified the repository, or ran tests.
- Ask at most one clarification question; if multiple details are unclear, prioritize the one that most materially affects implementation.
- Keep the skill focused on requirement synthesis, context interpretation, and handoff preparation—not direct code modification or execution.
- Use only tools from the executable tool catalog; do not reference unavailable tools.
- Ensure the final handoff is concise, specific, and ready for the coding agent to act upon without further voice-agent interpretation.
