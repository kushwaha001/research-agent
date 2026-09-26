"""Turns ranked evidence into a structured, cited report, then self-checks it."""
from __future__ import annotations

import json
import re

from .llm import LLMClient
from .models import Document, Finding, Plan, Report
from .processing import tokens

SYNTH_SYSTEM = """You are the synthesis module of a research agent. Write a report using ONLY the numbered
sources provided. Every key point and finding MUST list its source ids in its "citations" array (do NOT write [n] markers inside "text"). Never invent facts, numbers,
dates or sources. If the evidence is thin or conflicting, say so in "limitations".
Return JSON only:
{"title": str,
 "summary": str (3-4 sentences, executive summary),
 "key_points": [{"text": str, "citations": [int]}]  (4-6 items, the most important takeaways),
 "findings": [{"text": str, "citations": [int]}]    (4-8 items, specific facts/details/numbers),
 "insights": [str]  (3-5 concrete, actionable recommendations for the reader),
 "limitations": [str]}"""

CRITIC_SYSTEM = """You review research evidence for coverage. Given sub-questions and source titles/snippets,
decide which sub-questions are NOT adequately answered. For each gap propose one tool call.
Return JSON: {"gaps": [{"sub_question": str, "tool": str, "query": str}]}  (empty list if coverage is fine; max 2)"""


def evidence_block(docs: list[Document], max_chars: int = 700) -> str:
    return "\n\n".join(
        f"[{d.id}] {d.title} ({d.source}{', ' + d.published[:16] if d.published else ''})\n{d.snippet[:max_chars]}"
        for d in docs
    )


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 40 <= len(s.strip()) <= 320]


class Synthesizer:
    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    async def find_gaps(self, plan: Plan, docs: list[Document], tool_names: list[str]) -> list[dict]:
        """Reflection step: which sub-questions lack evidence?"""
        if self.llm:
            try:
                r = await self.llm.chat_json(CRITIC_SYSTEM, json.dumps({
                    "sub_questions": plan.sub_questions, "tools": tool_names,
                    "evidence": [{"id": d.id, "title": d.title, "snippet": d.snippet[:200]} for d in docs]}))
                return [g for g in r.get("gaps", []) if g.get("tool") in tool_names and g.get("query")][:2]
            except Exception:
                return []
        # Heuristic: a sub-question is a gap if <2 docs share any of its content words.
        gaps = []
        for q in plan.sub_questions:
            qt = set(tokens(q)) - {"current", "state", "key", "developments", "practitioner", "should"}
            hits = sum(1 for d in docs if qt & set(tokens(d.title + " " + d.snippet)))
            if qt and hits < 2:
                gaps.append({"sub_question": q, "tool": "web_search", "query": " ".join(list(qt)[:5])})
        return gaps[:1]

    async def write(self, plan: Plan, docs: list[Document]) -> tuple[Report, list[str]]:
        """Returns (report, warnings from the citation self-check)."""
        if not docs:
            return Report(title=f"Research brief: {plan.goal}", summary="No usable evidence could be gathered; every source failed or returned irrelevant results.",
                          key_points=[], findings=[], insights=["Retry later or rephrase the goal more specifically."],
                          limitations=["All tools failed or returned nothing relevant."], references=[]), []
        report = None
        if self.llm:
            try:
                raw = await self.llm.chat_json(
                    SYNTH_SYSTEM,
                    f"Goal: {plan.goal}\nInterpretation: {plan.interpretation}\nSub-questions: {plan.sub_questions}\n\nSOURCES:\n{evidence_block(docs)}",
                    temperature=0.3)
                report = Report(title=raw.get("title") or plan.goal, summary=raw.get("summary", ""),
                                key_points=[Finding(**f) for f in raw.get("key_points", []) if isinstance(f, dict)],
                                findings=[Finding(**f) for f in raw.get("findings", []) if isinstance(f, dict)],
                                insights=[str(i) for i in raw.get("insights", [])],
                                limitations=[str(i) for i in raw.get("limitations", [])])
            except Exception:
                report = None
        if report is None:
            report = self.extractive(plan, docs)
        warnings = self.verify_citations(report, docs)
        return report, warnings

    @staticmethod
    def verify_citations(report: Report, docs: list[Document]) -> list[str]:
        """Self-check: drop citations to non-existent sources and flag uncited claims."""
        valid = {d.id for d in docs}
        warnings = []
        marker = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")
        for f in (*report.key_points, *report.findings):
            # Move inline [n] markers the model wrote into the citations list.
            for m in marker.findall(f.text):
                f.citations += [int(x) for x in m.split(",") if int(x) not in f.citations]
            f.text = marker.sub("", f.text).strip()
        report.summary = marker.sub("", report.summary)
        report.insights = [marker.sub("", i).strip() for i in report.insights]
        report.limitations = [marker.sub("", i).strip() for i in report.limitations]
        for section in (report.key_points, report.findings):
            for f in section:
                bad = [c for c in f.citations if c not in valid]
                if bad:
                    warnings.append(f"Removed hallucinated citation(s) {bad} from: '{f.text[:60]}…'")
                    f.citations = [c for c in f.citations if c in valid]
                if not f.citations:
                    warnings.append(f"Uncited claim flagged: '{f.text[:60]}…'")
                    f.text = f.text.rstrip() + " (unverified)"
        cited = {c for s in (report.key_points, report.findings) for f in s for c in f.citations}
        report.references = [d for d in docs if d.id in cited] or docs
        return warnings

    def extractive(self, plan: Plan, docs: list[Document]) -> Report:
        """LLM-free fallback: pick the most goal-relevant sentences from the top sources."""
        q = set(tokens(plan.goal))
        scored = []
        for d in docs:
            for s in _sentences(d.snippet) or [d.snippet[:250]]:
                overlap = len(q & set(tokens(s))) / (len(q) or 1)
                scored.append((overlap + d.score, s, d.id))
        scored.sort(key=lambda x: -x[0])
        seen, picks = set(), []
        for _, s, i in scored:
            key = " ".join(tokens(s)[:6])
            if key in seen:
                continue
            seen.add(key)
            picks.append(Finding(text=s, citations=[i]))
        sources = sorted({d.source for d in docs})
        return Report(
            title=f"Research brief: {plan.goal}",
            summary=(f"Gathered {len(docs)} distinct, relevant sources from {', '.join(sources)}. "
                     "This summary was produced extractively (no LLM configured), so points are quoted from sources rather than synthesised."),
            key_points=[Finding(text=d.title, citations=[d.id]) for d in docs[:5]],
            findings=picks[:6],
            insights=[f"Read the top-ranked sources first: {', '.join(f'[{d.id}]' for d in docs[:3])}.",
                      "Cross-check claims that appear in only one source before acting on them.",
                      "Configure an LLM key for synthesised, non-extractive insights."],
            limitations=["Extractive mode: no cross-source reasoning was performed."],
        )
