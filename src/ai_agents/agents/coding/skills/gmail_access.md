# Skill: Gmail Access

Purpose: Allow the AI agent to search and read emails from a connected Gmail account using the Gmail API.

Use when:
- The user asks to find specific emails, check recent messages, or search for emails containing certain keywords.
- You need to retrieve information stored in the user's Gmail inbox.

The agent must never send, delete, or modify emails. Only read access is permitted.

Allowed tools:
- gmail_search

Steps:
1. Formulate a Gmail search query using standard Gmail search operators (e.g., `from:user@example.com`, `is:unread`, `subject:meeting`).
2. Use the `gmail_search` tool with the query and a reasonable maximum number of results (default 5).
3. Parse the returned email metadata to extract the required information.
4. Present the findings to the user in a clear format without exposing any raw tokens or credentials.

Rules:
- Never print, log, or expose OAuth credentials or refresh tokens.
- Only access the Gmail account for which the user has explicitly granted permission.
- Do not read email content beyond the snippet; if full message bodies are needed, inform the user that additional permission may be required.
- Respect privacy – only retrieve emails that are directly relevant to the user's request.
- Never attempt to send, delete, or modify any email.
- If authentication fails, report the error without revealing sensitive details.