from __future__ import annotations

import urllib.request
from html.parser import HTMLParser
from typing import Optional


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)


def search_web(query: str, num_results: int = 10) -> list[str]:
    """Perform a web search and return a list of result URLs (stub implementation)."""
    # In a real implementation, this would query a search engine API.
    return [f"https://example.com/result{i}?q={query}" for i in range(num_results)]


def fetch_url(url: str) -> str:
    """Fetch the HTML content of a given URL."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_links(html: str) -> list[str]:
    """Extract all hyperlinks from HTML content."""
    parser = LinkExtractor()
    parser.feed(html)
    return parser.links
