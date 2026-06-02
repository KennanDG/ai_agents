# Skill: Debug Error

Purpose: Diagnose and fix a coding error using logs, tracebacks, or failed command output.

Use when:
- The user provides a traceback, failed test, log output, or runtime exception.

Allowed tools:
- search_repo
- read_file
- write_file
- run_command

Steps:
1. Extract the exact error message.
2. Identify the likely failing file, function, or command.
3. Search the repo for related code.
4. Read relevant files before editing.
5. Apply the smallest safe fix.
6. Re-run the failing command or targeted tests.
7. Report root cause, fix, and validation result.

Rules:
- Do not guess without inspecting files.
- If multiple causes are possible, rank them.
- Do not ever hide the failed validation.
