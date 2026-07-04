# Skill: Web Search

Purpose: Search the web for information and retrieve web page content.

Use when:
- The user asks to find or verify web information.
- The task requires fetching URLs or extracting links.

Steps:
1. Generate search queries from the user request.
2. Use search_web to get result URLs.
3. Use fetch_url to retrieve page content.
4. Use extract_links to gather related links.

Rules:
- Limit search to trusted sources.
- Do not expose sensitive data.
- Validate URLs before fetching.
