from agents.voice.tools.web_search import web_search


def run(query: str) -> str:
    """Execute a web search and return a voice-friendly summary."""
    result = web_search(query)
    if isinstance(result, dict):
        if "error" in result:
            return f"Search error: {result['error']}"
        results = result.get("results", [])
        if not results:
            return "No results found."
        top = results[:3]
        lines = [
            f"- {r.get('title', 'No title')}: {r.get('snippet', '')}"
            for r in top
        ]
        summary = "\n".join(lines)
        return f"Top web results for '{query}':\n{summary}"
    return "Unexpected search output."
