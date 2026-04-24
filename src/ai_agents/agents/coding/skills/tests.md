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
- run_command

Steps:
1. Locate existing test structure.
2. Match existing test style and fixtures.
3. Add the smallest meaningful test.
4. Prefer unit tests before integration tests.
5. Run the targeted test file.

Rules:
- Do not add brittle tests tied to implementation details unless necessary.
- Do not require external services unless existing tests already do.