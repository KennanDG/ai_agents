# Skill: Web Search

Purpose: Search the web for current information using an external search API.

Use when:
- The user asks for up-to-date or external information not available in the repository.
- The user asks to google something.

Allowed tools:
- web_search

Steps:
1. Parse the user's search query.
2. Use the web_search tool with the query.
3. Return a summary of the top results.

Rules:
- Use the SERPAPI_API_KEY environment variable for authentication.
- Do not expose the API key.
- Respect the tool's output format.
