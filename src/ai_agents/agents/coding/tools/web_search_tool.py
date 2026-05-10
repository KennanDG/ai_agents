"""Simple placeholder web search tool.

In a real implementation this would call an external search API.
Here it returns a fixed placeholder string for safety.
"""

def search(query: str) -> str:
    """Return a placeholder result for the given query.

    Args:
        query: The search query string.

    Returns:
        A placeholder string indicating what would be searched.
    """
    if not query:
        return "No query provided."
    return f"[Placeholder] Results for '{query}'."
