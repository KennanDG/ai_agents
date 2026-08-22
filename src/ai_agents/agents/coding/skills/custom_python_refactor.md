# Skill: Python Module Refactoring

Purpose: Safely refactor Python modules in an existing repository by inspecting package structure, identifying import dependencies, detecting circular imports, and validating changes with minimal disruption to the codebase.

Use when:
- Moving or renaming Python files within a repository
- Reorganizing package structure or module hierarchy
- Extracting logic into new modules or subpackages
- Changing import statements across multiple files
- Consolidating or splitting existing modules
- Updating public API exposure or module boundaries

Allowed tools:
- list_files
- read_file
- write_file
- python_import_map
- python_symbol_details
- search_repository
- robust_search
- resolve_in_repo
- unified_diff
- run_command
- is_allowed_command

Steps:
1. Use list_files to inspect the current package structure and identify all Python files in the target directory.
2. Use python_import_map to extract all imports from affected modules and identify external dependencies.
3. Use python_symbol_details to locate all public APIs, classes, functions, and constants that may be referenced elsewhere.
4. Use search_repository to find all references to symbols being moved or renamed across the entire codebase.
5. Use python_import_map on dependent modules to detect potential circular import chains before making changes.
6. Plan refactoring as small, incremental steps; document the change scope and affected files.
7. Apply changes incrementally: move/rename files, update imports in affected modules, preserve public API signatures.
8. Use read_file and write_file to update import statements in all dependent modules.
9. Use run_command to validate Python syntax with `python -m py_compile` on modified files.
10. Use search_repository to locate and review test files that reference moved or renamed modules.
11. Update test imports and references using read_file and write_file.
12. Use run_command to execute the most relevant test suite to validate the refactoring.

Rules:
- Never invent or assume the existence of modules, functions, or dependencies not present in the repository.
- Always inspect the package structure before proposing changes.
- Preserve public APIs and maintain backward compatibility where possible; document breaking changes explicitly.
- Detect and report circular import risks before executing changes.
- Make changes incrementally; avoid rewriting entire modules in a single operation.
- Only modify files directly affected by the refactoring; do not alter unrelated code.
- Validate Python syntax on all modified files before considering the refactoring complete.
- Update all affected test files and verify tests pass after refactoring.
- Use unified_diff to review changes before writing files to disk.
- If a capability is needed but no tool exists for it, express it as guidance rather than inventing a tool name.
