# Skill: Build React Component

Purpose: Implement or refactor a focused React TypeScript component using the frontend's existing patterns.

Use when:
- The user asks to add or change a React component, screen, hook, or client-side interaction.
- The task primarily affects `.tsx` or frontend TypeScript files.

Allowed tools:
- list_files
- robust_search
- read_file
- write_file
- run_command
- scaffold_component
- lint_component_imports
- validate_component_props

Steps:
1. Inspect `package.json`, nearby components, shared types, and existing state patterns.
2. Define typed props and state at the narrowest useful boundary.
3. Prefer composition and existing primitives over new abstractions or dependencies.
4. Cover loading, empty, error, and disabled states when data or actions are involved.
5. Keep rendering pure; place effects and transport logic behind hooks or services.
6. Run the frontend typecheck and the smallest relevant test or build command.

Rules:
- Do not use `any` when a local type, generic, or `unknown` is appropriate.
- Do not invent backend response shapes; inspect contracts or create an explicit adapter boundary.
- Preserve existing component, import, and file-naming conventions.
- Keep the change focused and report validation failures.
