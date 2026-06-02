# Skill: Web Search

Purpose: Perform a web search and return summarized information from relevant web pages.

Use when:
- The user asks a question that requires up-to-date or external information.
- The user asks for news, facts, or recent events.

Allowed tools:
- web_search, browse_page (or equivalent content retrieval)

Steps:
1. Extract the search query from the user's request.
2. Use a web search tool to find relevant results.
3. Optionally open and extract content from top results for more detail.
4. Summarize the key information, citing sources.
5. Present the answer in a conversational tone.

Rules:
- Do not verbatim copy large blocks of copyright-protected content.
- Limit results to the top 3–5 most relevant sources.
- If conflicting information is found, note the discrepancy and present both views.
- Respect safe-search guidelines and avoid adult or harmful content.
- Clearly state when information might be outdated or uncertain.
