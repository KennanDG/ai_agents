import os
import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


def web_search(query: str) -> Dict[str, Any]:
    """Search the web via SerpAPI using the SERPAPI_API_KEY env variable."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        logger.error("SERPAPI_API_KEY not set")
        return {"error": "Search API key not configured"}
    try:
        params = {
            "q": query,
            "api_key": api_key,
            "engine": "google",
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("organic_results", [])
        return {"results": results}
    except Exception as e:
        logger.exception("Web search failed")
        return {"error": str(e)}
