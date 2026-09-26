"""Evidence post-processing: relevance scoring, near-duplicate removal, citation numbering."""
from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

from .models import Document

STOP = set("""a an the of to in on for and or with by from at is are was were be been this that these those
it its as about into over under what which who whom how why when where top latest last week month year new
""".split())
# Instruction words that describe the task, not the topic (dropped when building search queries).
INSTRUCTION = set("""research summarize summarise summary find give produce list explain analyze analyse report brief
developments development me please short detailed overview identify describe past recent days""".split())


def topic_terms(goal: str) -> list[str]:
    return [t for t in tokens(goal) if t not in INSTRUCTION and not t.isdigit()]


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


def canonical_url(url: str) -> str:
    p = urlparse(url.strip())
    host = p.netloc.lower().removeprefix("www.").removeprefix("m.")
    path = p.path.rstrip("/")
    # arXiv: collapse versions (2401.00001v2 -> 2401.00001)
    path = re.sub(r"(/abs/\d{4}\.\d{4,5})v\d+$", r"\1", path)
    return urlunparse(("https", host, path, "", "", ""))


def shingles(text: str, k: int = 3) -> set[str]:
    t = tokens(text)
    return {" ".join(t[i : i + k]) for i in range(max(len(t) - k + 1, 1))} if t else set()


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def relevance(doc: Document, query_terms: set[str]) -> float:
    """Cheap lexical relevance in [0,1]: weighted term coverage of title + snippet."""
    if not query_terms:
        return 0.5
    title, body = set(tokens(doc.title)), set(tokens(doc.snippet))
    cov_title = len(query_terms & title) / len(query_terms)
    cov_body = len(query_terms & body) / len(query_terms)
    return round(min(1.0, 0.6 * cov_title + 0.6 * cov_body), 3)


def dedupe_and_rank(
    docs: list[Document], goal: str, *, min_score: float = 0.12, near_dup: float = 0.6, max_docs: int = 18
) -> tuple[list[Document], dict[str, int]]:
    """Returns (kept docs with citation ids, stats)."""
    q = set(topic_terms(goal)) or set(tokens(goal))
    for d in docs:
        d.score = relevance(d, q)

    stats = {"raw": len(docs), "url_dupes": 0, "near_dupes": 0, "irrelevant": 0}
    by_url: dict[str, Document] = {}
    for d in sorted(docs, key=lambda d: -len(d.snippet)):  # keep the richest copy
        key = canonical_url(d.url)
        if key in by_url:
            stats["url_dupes"] += 1
            continue
        by_url[key] = d

    kept: list[Document] = []
    sigs: list[set[str]] = []
    for d in sorted(by_url.values(), key=lambda d: -d.score):
        if d.score < min_score:
            stats["irrelevant"] += 1
            continue
        sig = shingles(f"{d.title} {d.snippet}")
        if any(jaccard(sig, s) >= near_dup for s in sigs):
            stats["near_dupes"] += 1
            continue
        kept.append(d)
        sigs.append(sig)

    kept = kept[:max_docs]
    for i, d in enumerate(kept, 1):
        d.id = i
    stats["kept"] = len(kept)
    return kept, stats
