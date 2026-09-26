"""Offline test-suite: no network or API key needed (simulated tools + heuristic LLM)."""
from __future__ import annotations

import asyncio

import pytest

from agent import ResearchAgent
from agent.config import Settings
from agent.llm import extract_json
from agent.models import Document, Finding, Plan, PlanStep, Report
from agent.planner import Planner
from agent.processing import canonical_url, dedupe_and_rank, topic_terms
from agent.report import to_markdown, to_pdf
from agent.synthesizer import Synthesizer
from agent.tools import build_tools

GOAL = "Research and summarize the top 3 developments in small language models from the last week"


def settings(tmp_path) -> Settings:
    s = Settings(offline=True, db_path=str(tmp_path / "m.db"))
    s.offline = True
    return s


def run(tmp_path, **kw):
    events = []

    async def emit(e):
        events.append(e)

    res = asyncio.run(ResearchAgent(settings(tmp_path), **kw).run(GOAL, emit))
    return res, events


# ---------- processing ----------
def test_canonical_url_collapses_variants():
    assert canonical_url("https://www.x.com/a/?utm=1") == canonical_url("http://x.com/a")
    assert canonical_url("https://arxiv.org/abs/2401.00001v3") == canonical_url("https://arxiv.org/abs/2401.00001")


def test_dedupe_removes_url_and_near_duplicates_and_irrelevant():
    docs = [
        Document(title="Small language models beat giants", url="https://a.com/1", snippet="small language models new benchmark results today", source="news"),
        Document(title="Small language models beat giants", url="https://www.a.com/1/", snippet="short", source="web_search"),
        Document(title="Small language models beat giants!", url="https://b.com/2", snippet="small language models new benchmark results today", source="web_search"),
        Document(title="Cooking pasta at home", url="https://c.com", snippet="boil water add salt", source="web_search"),
    ]
    kept, stats = dedupe_and_rank(docs, GOAL)
    assert stats["url_dupes"] == 1 and stats["near_dupes"] == 1 and stats["irrelevant"] == 1
    assert [d.id for d in kept] == [1]


def test_topic_terms_drop_instruction_words():
    assert topic_terms(GOAL) == ["small", "language", "models"]


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('blah {"b": [1,2]} trailing') == {"b": [1, 2]}


# ---------- planning ----------
def test_heuristic_plan_uses_multiple_distinct_tools():
    p = Planner(None, build_tools(True, 5)).heuristic_plan(GOAL)
    assert len({s.tool for s in p.steps}) >= 2
    assert p.steps[0].tool == "news"  # "last week" -> recency source first


def test_llm_plan_validation_drops_hallucinated_tools():
    pl = Planner(None, build_tools(True, 5))
    raw = {"interpretation": "x", "sub_questions": ["q"], "steps": [
        {"id": 1, "tool": "news", "query": "a", "rationale": ""},
        {"id": 2, "tool": "made_up_tool", "query": "b", "rationale": ""},
        {"id": 3, "tool": "arxiv", "query": "c", "rationale": ""}]}
    plan = pl._validate(GOAL, raw)
    assert [s.tool for s in plan.steps] == ["news", "arxiv"]
    with pytest.raises(ValueError):
        pl._validate(GOAL, {"steps": [{"id": 1, "tool": "news", "query": "a"}]})


# ---------- synthesis self-check ----------
def test_citation_verification_removes_hallucinated_ids():
    docs = [Document(title="t", url="https://a.com", source="news", id=1)]
    rep = Report(title="t", summary="s", key_points=[Finding(text="real", citations=[1, 7])],
                 findings=[Finding(text="uncited", citations=[])], insights=[])
    warnings = Synthesizer.verify_citations(rep, docs)
    assert rep.key_points[0].citations == [1]
    assert rep.findings[0].text.endswith("(unverified)")
    assert len(warnings) == 2


# ---------- end-to-end + failure recovery ----------
def test_end_to_end_happy_path(tmp_path):
    res, events = run(tmp_path)
    types = [e.type for e in events]
    assert types.index("plan") < types.index("step_start")  # plans BEFORE acting
    assert res.report.references and res.report.key_points and res.report.insights
    assert all(s.status == "ok" for s in res.steps)
    assert "## References" in to_markdown(res)
    assert to_pdf(res)[:4] == b"%PDF"


@pytest.mark.parametrize("mode,expect", [("timeout", "retry"), ("garbage", "retry"), ("empty", "reformulate"), ("outage", "fallback")])
def test_injected_failures_are_recovered(tmp_path, mode, expect):
    res, events = run(tmp_path, fault=mode, fault_tool="news")
    news = next(s for s in res.steps if s.tool == "news")
    assert news.status == "recovered", news
    assert any(e.type == "recovery" and e.data.get("action") == expect for e in events)
    assert res.stats["recovered"] >= 1


def test_memory_recalls_similar_previous_run(tmp_path):
    s = settings(tmp_path)
    asyncio.run(ResearchAgent(s).run(GOAL))
    events = []

    async def emit(e):
        events.append(e)

    asyncio.run(ResearchAgent(s).run("latest developments in small language models", emit))
    assert any(e.type == "memory" for e in events)


def test_total_failure_is_reported_not_crashed(tmp_path):
    s = settings(tmp_path)
    agent = ResearchAgent(s, use_memory=False)
    for t in agent.tools.values():
        async def boom(*a, **k):
            from agent.tools import ToolError
            raise ToolError("down")
        t.search = boom
    agent.planner.heuristic_plan = lambda g: Plan(goal=g, interpretation="", sub_questions=["q"],
                                                  steps=[PlanStep(id=1, tool="news", query="x", rationale="")])
    res = asyncio.run(agent.run(GOAL))
    assert res.steps[0].status == "failed"
    assert "No usable evidence" in res.report.summary


def test_llm_path_with_stubbed_model(tmp_path, monkeypatch):
    """Exercises planner/critic/synthesiser prompts via a fake LLM (incl. a hallucinated citation)."""
    import json as _j

    from agent.llm import LLMClient

    async def fake_chat(self, system, user, **kw):
        self.calls += 1
        if "planning module" in system:
            return _j.dumps({"interpretation": "SLM news, past 7 days", "sub_questions": ["What launched?"],
                             "source_rationale": "news for recency, arxiv for papers",
                             "steps": [{"id": 1, "tool": "news", "query": "small language model release", "rationale": "r"},
                                       {"id": 2, "tool": "arxiv", "query": "small language models", "rationale": "r"}]})
        if "coverage" in system:
            return '{"gaps": []}'
        return "```json\n" + _j.dumps({"title": "SLM brief", "summary": "s", "insights": ["do x"], "limitations": [],
                                       "key_points": [{"text": "a", "citations": [1]}, {"text": "b", "citations": [99]}],
                                       "findings": [{"text": "c", "citations": [2]}]}) + "\n```"

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    s = settings(tmp_path)
    agent = ResearchAgent(s)
    agent.llm = LLMClient(s)
    agent.planner.llm = agent.synth.llm = agent.llm
    events = []

    async def emit(e):
        events.append(e)

    res = asyncio.run(agent.run(GOAL, emit))
    assert res.plan.interpretation == "SLM news, past 7 days"
    assert res.report.title == "SLM brief"
    assert res.report.key_points[1].text.endswith("(unverified)")  # [99] was stripped
    assert any("hallucinated" in e.message for e in events)
