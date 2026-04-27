# Skill: Write Tests

Purpose: Add focused tests for behavior that changed or broke.

Use when:
- The user asks for tests.
- A fix needs regression coverage.

Allowed tools:
- list_files
- search_repo
- read_file
- write_file
- run_validation_suite

Steps:
1. Always target the `src/ai_agents/agents/coding/tests/` directory for test creation or modification.
2. Match existing test style and fixtures found in that directory.
3. Add the smallest meaningful test.
4. Prefer unit tests before integration tests.
5. Run the targeted test file.

Rules:
- All test files must be created or updated strictly within `src/ai_agents/agents/coding/tests/`
- Do not add brittle tests tied to implementation details unless necessary.
- Do not require external services unless existing tests already do.
- Always validate through the coding agent validation module.