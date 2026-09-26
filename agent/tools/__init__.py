"""Tool registry, simulated tools for offline mode, and fault injection."""
from __future__ import annotations

import asyncio
import hashlib

from ..models import Document
from .base import Tool, ToolError
from .sources import Arxiv, GitHub, HackerNews, NewsSearch, PageReader, WebSearch, Wikipedia

SEARCH_TOOLS: list[type[Tool]] = [WebSearch, NewsSearch, Wikipedia, Arxiv, HackerNews, GitHub]

# If a tool fails outright, the executor tries its fallback with the same query.
FALLBACKS = {
    "web_search": "wikipedia",
    "news": "web_search",
    "hackernews": "web_search",
    "arxiv": "web_search",
    "github": "web_search",
    "wikipedia": "web_search",
}


ANGLES = [
    "Benchmarks published this month show {q} closing the gap with larger systems on reasoning tasks, per {src}.",
    "Enterprises report that adopting {q} cut inference cost substantially; deployment on edge devices is growing.",
    "Critics note that evaluation of {q} is inconsistent and several claimed gains did not replicate.",
    "A new open-source release in {q} gained thousands of stars within days, signalling strong developer demand.",
    "Regulators and standards bodies have begun drafting guidance that touches on {q} safety and transparency.",
    "Researchers propose distillation and quantisation recipes that make {q} cheaper to train and serve.",
    "Background: {q} emerged from earlier work on efficiency; its history explains current design trade-offs.",
]


class SimulatedTool(Tool):
    """Deterministic stand-in used when OFFLINE_MODE=1 (no network / CI / demos).

    Results are clearly labelled [SIMULATED] and seeded from the query so tests are stable.
    """

    def __init__(self, real: type[Tool], timeout: float = 12.0):
        super().__init__(timeout)
        self.name = real.name  # type: ignore[misc]
        self.description = real.description  # type: ignore[misc]
        self.best_for = real.best_for  # type: ignore[misc]

    async def search(self, query: str, limit: int = 6) -> list[Document]:
        await asyncio.sleep(0.15)
        if self.name == "page_reader":
            return [Document(title="[SIMULATED] page", url=query, source=self.name,
                             snippet=f"[SIMULATED] Full text of {query}. " * 8)]
        h = int(hashlib.md5(f"{self.name}{query}".encode()).hexdigest(), 16)
        docs = []
        for i in range(min(limit, 4)):
            slug = f"{h % 9973}-{i}"
            docs.append(
                Document(
                    title=f"[SIMULATED] {query.title()}: perspective {i + 1} from {self.name}",
                    url=f"https://example.org/{self.name}/{slug}",
                    snippet=f"[SIMULATED] {ANGLES[(h + i) % len(ANGLES)].format(q=query, src=self.name)}",
                    source=self.name,
                    published="2026-09-2" + str(i),
                )
            )
        # A cross-source duplicate so the dedup stage has something real to do.
        docs.append(Document(title=f"[SIMULATED] Overview of {query}", url="https://example.org/shared/overview",
                             snippet=f"[SIMULATED] A widely syndicated overview of {query}.", source=self.name))
        return docs


class FaultInjector:
    """Deliberately induces failures to demonstrate recovery (assignment requirement #4).

    modes:
      timeout  - first call of `target` raises a timeout (-> retry with backoff succeeds)
      outage   - every call of `target` fails (-> fallback tool is used)
      empty    - first call of `target` returns nothing (-> LLM reformulates the query)
      garbage  - first call of `target` returns malformed docs (-> validation rejects, retry)
    """

    def __init__(self, mode: str | None, target: str | None):
        self.mode, self.target = mode, target
        self.fired: set[str] = set()

    def wrap(self, tool: Tool) -> Tool:
        if not self.mode or tool.name != self.target:
            return tool
        inj = self
        original = tool.search

        async def search(query: str, limit: int = 6):
            key = f"{tool.name}"
            first = key not in inj.fired
            inj.fired.add(key)
            if inj.mode == "outage":
                raise ToolError(f"{tool.name}: [INJECTED] service unavailable (503)")
            if first and inj.mode == "timeout":
                await asyncio.sleep(0.2)
                raise ToolError(f"{tool.name}: [INJECTED] timed out after {tool.timeout}s")
            if first and inj.mode == "empty":
                return []
            if first and inj.mode == "garbage":
                return [Document(title="", url="not-a-url", snippet="\x00\x00", source=tool.name)]
            return await original(query, limit)

        tool.search = search  # type: ignore[method-assign]
        return tool


def build_tools(offline: bool, timeout: float, injector: FaultInjector | None = None) -> dict[str, Tool]:
    tools: dict[str, Tool] = {}
    for cls in [*SEARCH_TOOLS, PageReader]:
        t: Tool = SimulatedTool(cls, timeout) if offline else cls(timeout)
        if injector:
            t = injector.wrap(t)
        tools[t.name] = t
    return tools


__all__ = ["Tool", "ToolError", "build_tools", "FaultInjector", "FALLBACKS", "SEARCH_TOOLS"]
