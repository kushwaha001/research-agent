"""Live, key-free public data sources."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, unquote, urlparse

from ..models import Document
from .base import Tool, ToolError, strip_html


class WebSearch(Tool):
    name = "web_search"
    description = "General web search (Bing, with DuckDuckGo as backup). Broad coverage of any topic."
    best_for = "general questions, companies, products, how-tos, anything not covered by a specialised source"

    async def search(self, query: str, limit: int = 6) -> list[Document]:
        # Bing's RSS endpoint works from cloud hosts; DuckDuckGo often blocks datacenter IPs.
        try:
            docs = await self._bing(query)
            if docs:
                return docs[:limit]
        except ToolError:
            pass
        return (await self._ddg(query))[:limit]

    async def _bing(self, query: str) -> list[Document]:
        r = await self._get("https://www.bing.com/search", q=query, format="rss", setlang="en-US", cc="US")
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            raise ToolError("web_search: invalid Bing RSS") from e
        return [Document(title=i.findtext("title") or "", url=i.findtext("link") or "",
                         snippet=strip_html(i.findtext("description") or ""), source=self.name,
                         published=i.findtext("pubDate"))
                for i in root.iter("item")]

    async def _ddg(self, query: str) -> list[Document]:
        r = await self._get("https://html.duckduckgo.com/html/", q=query)
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</a>',
            r.text,
            re.S,
        )
        docs = []
        for href, title, snippet in blocks:
            if "uddg=" in href:  # DDG wraps links as //duckduckgo.com/l/?uddg=<real url>
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
            if "duckduckgo.com/y.js" in href:  # ads
                continue
            docs.append(Document(title=strip_html(title), url=href, snippet=strip_html(snippet), source=self.name))
        if not docs and "anomaly" in r.text.lower():
            raise ToolError("web_search: blocked by bot protection")
        return docs


class NewsSearch(Tool):
    name = "news"
    description = "Recent news articles (Google News RSS), sorted by recency."
    best_for = "recent developments, 'last week/this month', current events, announcements"

    async def search(self, query: str, limit: int = 6) -> list[Document]:
        r = await self._get("https://news.google.com/rss/search", q=f"{query} when:14d", hl="en-US", gl="US", ceid="US:en")
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            raise ToolError("news: invalid RSS response") from e
        docs = []
        for item in root.iter("item"):
            title = item.findtext("title") or ""
            src = item.findtext("source") or ""
            docs.append(
                Document(
                    title=title,
                    url=item.findtext("link") or "",
                    snippet=f"{strip_html(item.findtext('description') or '')} ({src})".strip(),
                    source=self.name,
                    published=item.findtext("pubDate"),
                )
            )
        return docs[:limit]


class Wikipedia(Tool):
    name = "wikipedia"
    description = "Wikipedia article search with intro extracts."
    best_for = "definitions, background, history, well-established concepts, organisations, people, places"

    async def search(self, query: str, limit: int = 6) -> list[Document]:
        r = await self._get(
            "https://en.wikipedia.org/w/api.php",
            action="query", generator="search", gsrsearch=query, gsrlimit=str(min(limit, 8)),
            prop="extracts|info", exintro="1", explaintext="1", exsentences="4",
            inprop="url", format="json",
        )
        pages = (r.json().get("query") or {}).get("pages") or {}
        pages = sorted(pages.values(), key=lambda p: p.get("index", 99))
        return [
            Document(title=p["title"], url=p.get("fullurl", ""), snippet=p.get("extract", "")[:900], source=self.name)
            for p in pages
        ]


class Arxiv(Tool):
    name = "arxiv"
    description = "Research papers: arXiv preprints newest first (Semantic Scholar as backup), with abstracts."
    best_for = "scientific / ML / AI / physics / maths research, state of the art, papers"

    async def search(self, query: str, limit: int = 6) -> list[Document]:
        errors = []
        # arXiv sometimes refuses cloud-host IPs on one hostname but not the other.
        for host in ("https://export.arxiv.org/api/query", "https://arxiv.org/api/query"):
            try:
                return await self._arxiv(host, query, limit)
            except ToolError as e:
                errors.append(str(e))
        try:
            return await self._semantic_scholar(query, limit)
        except ToolError as e:
            errors.append(str(e))
        raise ToolError("arxiv: all paper sources failed (" + "; ".join(errors) + ")")

    async def _arxiv(self, url: str, query: str, limit: int) -> list[Document]:
        r = await self._get(url, search_query=f"all:{query}", sortBy="submittedDate",
                            sortOrder="descending", max_results=str(limit))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            raise ToolError("arxiv: invalid Atom response") from e
        return [
            Document(
                title=re.sub(r"\s+", " ", e.findtext("a:title", "", ns)).strip(),
                url=e.findtext("a:id", "", ns),
                snippet=re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip()[:900],
                source=self.name,
                published=e.findtext("a:published", None, ns),
            )
            for e in root.findall("a:entry", ns)
        ]

    async def _semantic_scholar(self, query: str, limit: int) -> list[Document]:
        r = await self._get("https://api.semanticscholar.org/graph/v1/paper/search", query=query, limit=str(limit),
                            fields="title,abstract,url,year,publicationDate,externalIds", sort="publicationDate:desc")
        docs = []
        for p in r.json().get("data", []):
            arxiv_id = (p.get("externalIds") or {}).get("ArXiv")
            docs.append(Document(title=p.get("title") or "", source=self.name,
                                 url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (p.get("url") or ""),
                                 snippet=(p.get("abstract") or "")[:900],
                                 published=p.get("publicationDate") or (str(p["year"]) if p.get("year") else None)))
        return docs


class HackerNews(Tool):
    name = "hackernews"
    description = "Hacker News stories from the last 30 days (Algolia API), ranked by relevance."
    best_for = "tech industry, startups, developer tools, programming, what engineers are discussing"

    async def search(self, query: str, limit: int = 6) -> list[Document]:
        since = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())
        r = await self._get(
            "https://hn.algolia.com/api/v1/search",
            query=query, tags="story", numericFilters=f"created_at_i>{since}", hitsPerPage=str(limit),
        )
        docs = []
        for h in r.json().get("hits", []):
            hn = f"https://news.ycombinator.com/item?id={h['objectID']}"
            docs.append(
                Document(
                    title=h.get("title") or "",
                    url=h.get("url") or hn,
                    snippet=f"{h.get('points', 0)} points, {h.get('num_comments', 0)} comments on HN. "
                    + strip_html(h.get("story_text") or "")[:400],
                    source=self.name,
                    published=h.get("created_at"),
                )
            )
        return docs


class GitHub(Tool):
    name = "github"
    description = "GitHub repository search, sorted by stars."
    best_for = "open-source libraries, frameworks, code tools, implementations"

    async def search(self, query: str, limit: int = 6) -> list[Document]:
        r = await self._get("https://api.github.com/search/repositories", q=query, sort="stars", per_page=str(limit))
        return [
            Document(
                title=i["full_name"],
                url=i["html_url"],
                snippet=f"★{i.get('stargazers_count', 0)} · {i.get('language') or 'n/a'} · {i.get('description') or ''}",
                source=self.name,
                published=i.get("pushed_at"),
            )
            for i in r.json().get("items", [])
        ]


class PageReader(Tool):
    """Not a search tool: fetches a URL and returns its readable text (used to enrich top hits)."""
    name = "page_reader"
    description = "Fetch a web page and extract its main text."
    best_for = "reading a specific URL in depth"

    async def search(self, query: str, limit: int = 1) -> list[Document]:
        r = await self._get(query)
        body = re.sub(r"(?is)<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>", " ", r.text)
        paras = [strip_html(p) for p in re.findall(r"(?is)<p[^>]*>(.*?)</p>", body)]
        text = " ".join(p for p in paras if len(p) > 60)[:2500]
        if len(text) < 200:
            raise ToolError("page_reader: no readable content")
        title = strip_html((re.search(r"(?is)<title>(.*?)</title>", r.text) or [None, query])[1])
        return [Document(title=title, url=query, snippet=text, source=self.name)]
