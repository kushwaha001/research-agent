"""Tool interface + registry.

Every tool is an async `search(query, limit)` that returns validated `Document`s.
Tools advertise *what they are good for* so the planner LLM can pick sources itself.
"""
from __future__ import annotations

import abc
import html
import re
from typing import ClassVar

import httpx

from ..models import Document

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,application/atom+xml;q=0.9,application/json;q=0.9,*/*;q=0.8"


class ToolError(RuntimeError):
    """Raised for any tool failure; the executor decides whether to retry or recover."""


class Tool(abc.ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    best_for: ClassVar[str]

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout

    async def _get(self, url: str, **params) -> httpx.Response:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, headers={"User-Agent": USER_AGENT, "Accept": ACCEPT, "Accept-Language": "en-US,en;q=0.9"}, follow_redirects=True
            ) as c:
                r = await c.get(url, params=params or None)
        except httpx.TimeoutException as e:
            raise ToolError(f"{self.name}: timed out after {self.timeout}s") from e
        except httpx.HTTPError as e:
            raise ToolError(f"{self.name}: network error {type(e).__name__}") from e
        if r.status_code == 429:
            raise ToolError(f"{self.name}: rate limited (429)")
        if r.status_code >= 400:
            raise ToolError(f"{self.name}: HTTP {r.status_code}")
        return r

    @abc.abstractmethod
    async def search(self, query: str, limit: int = 6) -> list[Document]: ...

    async def run(self, query: str, limit: int = 6) -> list[Document]:
        """Validated entry point: sanitises input and output."""
        query = (query or "").strip()
        if not query:
            raise ToolError(f"{self.name}: empty query")
        docs = await self.search(query[:300], limit)
        clean = [d for d in docs if d.url.startswith("http") and d.title.strip()]
        return clean[:limit]


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()
