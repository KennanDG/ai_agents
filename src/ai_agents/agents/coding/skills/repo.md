# Skill: Inspect Repository

Purpose: Understand the repository structure before editing.

Use when:
- Starting a coding task.
- The relevant files are unknown.
- The agent needs to learn project conventions.

Allowed tools:
- list_files
- search_repo
- read_file

Steps:
1. List root-level files and important source folders.
2. Identify framework, package manager, test framework, and entrypoints.
3. Search for relevant filenames, symbols, routes, or error terms.
4. Read only the most relevant files.
5. Summarize findings before editing.

Rules:
- Do not edit files during this skill.
- Do not assume architecture before inspecting files.