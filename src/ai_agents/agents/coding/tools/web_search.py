from __future__ import annotations

import json
import os
from dotenv import load_dotenv
import urllib.parse
import urllib.request
from typing import Any


load_dotenv()

SERPAPI_API_KEY=os.environ['SERPAPI_API_KEY']

def web_search(query: str, num_results: int = 5) -> str:
    """Perform a web search using the SerpApi service.

    Requires SERPAPI_API_KEY environment variable.

    Args:
        query: The search query string.
        num_results: Number of top results to return (default 5).

    Returns:
        A JSON string containing an array of result objects (title, link, snippet).
    """
    api_key = os.environ.get("SERPAPI_API_KEY")

    if not api_key:
        return json.dumps({"error": "SERPAPI_API_KEY not set"})

    params = {
        "q": query,
        "api_key": api_key,
        "num": str(num_results),
        "engine": "google",
    }

    url = "https://serpapi.com/search?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        return json.dumps({"error": str(e)})

    organic_results = data.get("organic_results", [])
    results = []
    
    for item in organic_results[:num_results]:
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
        })

    return json.dumps(results)
