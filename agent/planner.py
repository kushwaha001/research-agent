"""Planning: turn a goal into sub-questions and tool calls. The LLM picks the sources itself."""
from __future__ import annotations

import json

from pydantic import ValidationError

from .llm import LLMClient
from .models import Plan, PlanStep
from .processing import tokens, topic_terms
from .tools import Tool

PLANNER_SYSTEM = """You are the planning module of an autonomous research agent.
Given a research goal and a catalogue of tools, produce a plan BEFORE any tool is called.

Rules:
- Restate how you interpret the goal (resolve ambiguity explicitly; state assumptions).
- Break it into 2-4 focused sub-questions.
- Choose the MOST APPROPRIATE tools for this goal from the catalogue (use 2-4 distinct tools).
  Prefer 'news' for recent events, 'arxiv' for research, 'github' for code, 'wikipedia' for background.
- Write 3-7 steps. Each step = one tool call with a concise keyword query (not a sentence) and a rationale.
- Steps with no dependencies run in parallel, so keep them independent where possible.
Return JSON only:
{"interpretation": str, "sub_questions": [str], "source_rationale": str,
 "steps": [{"id": int, "tool": str, "query": str, "rationale": str, "depends_on": [int]}]}"""

REFORMULATE_SYSTEM = """A search returned no useful results. Propose ONE better keyword query
(broader or with synonyms, max 8 words). Return JSON: {"query": str, "why": str}"""


def tool_catalogue(tools: dict[str, Tool]) -> str:
    return "\n".join(f"- {t.name}: {t.description} Best for: {t.best_for}" for t in tools.values() if t.name != "page_reader")


class Planner:
    def __init__(self, llm: LLMClient | None, tools: dict[str, Tool]):
        self.llm, self.tools = llm, tools

    async def plan(self, goal: str, memory_context: str = "") -> tuple[Plan, bool]:
        """Returns (plan, used_fallback)."""
        if self.llm:
            user = f"Goal: {goal}\n\nTools:\n{tool_catalogue(self.tools)}"
            if memory_context:
                user += f"\n\nRelated past research (avoid repeating it, build on it):\n{memory_context}"
            for _ in range(2):  # one retry if the plan fails validation
                try:
                    raw = await self.llm.chat_json(PLANNER_SYSTEM, user)
                    return self._validate(goal, raw), False
                except (ValidationError, ValueError, KeyError, TypeError) as e:
                    user += f"\n\nYour previous plan was invalid ({e}). Fix it."
                except Exception:
                    break
        return self.heuristic_plan(goal), True

    def _validate(self, goal: str, raw: dict) -> Plan:
        steps = []
        for i, s in enumerate(raw.get("steps", []), 1):
            tool = str(s.get("tool", "")).strip()
            if tool not in self.tools or tool == "page_reader":
                continue  # drop hallucinated tools instead of failing the whole plan
            steps.append(PlanStep(id=i, tool=tool, query=str(s["query"])[:200],
                                  rationale=str(s.get("rationale", "")),
                                  depends_on=[d for d in s.get("depends_on", []) if isinstance(d, int) and d < i]))
        if len({s.tool for s in steps}) < 2:
            raise ValueError("plan must use at least two distinct tools that exist in the catalogue")
        return Plan(goal=goal, interpretation=raw.get("interpretation", goal),
                    sub_questions=raw.get("sub_questions", [])[:5] or [goal],
                    source_rationale=raw.get("source_rationale", ""), steps=steps[:8])

    def heuristic_plan(self, goal: str) -> Plan:
        """Deterministic rule-based planner (offline mode / LLM outage)."""
        g = goal.lower()
        kw = " ".join(topic_terms(goal)[:6]) or goal
        picks: list[tuple[str, str]] = []
        if any(w in g for w in ["latest", "recent", "news", "week", "today", "month", "developments", "2026"]):
            picks.append(("news", "recent developments are best covered by news"))
        if any(w in g for w in ["paper", "research", "model", "llm", "ai", "learning", "algorithm", "science"]):
            picks.append(("arxiv", "the topic has an academic research dimension"))
        if any(w in g for w in ["library", "framework", "open source", "github", "code", "tool", "repo"]):
            picks.append(("github", "open-source implementations are relevant"))
        if any(w in g for w in ["startup", "developer", "software", "tech", "programming", "ai"]):
            picks.append(("hackernews", "practitioner discussion adds signal"))
        picks += [("web_search", "broad coverage"), ("wikipedia", "background and definitions")]
        seen, steps = set(), []
        for tool, why in picks:
            if tool in seen or tool not in self.tools:
                continue
            seen.add(tool)
            steps.append(PlanStep(id=len(steps) + 1, tool=tool, query=kw, rationale=why))
            if len(steps) == 4:
                break
        return Plan(goal=goal, interpretation=f"Research '{goal}' and summarise the most relevant, well-sourced points.",
                    sub_questions=[f"What is the current state of {kw}?", f"What are the key recent developments in {kw}?",
                                   f"What should a practitioner do about {kw}?"],
                    source_rationale="Rule-based source selection from keywords in the goal.", steps=steps)

    async def reformulate(self, goal: str, query: str, tool: str) -> tuple[str, str]:
        if self.llm:
            try:
                r = await self.llm.chat_json(REFORMULATE_SYSTEM,
                                             json.dumps({"goal": goal, "failed_query": query, "tool": tool}))
                if r.get("query") and r["query"].strip().lower() != query.lower():
                    return r["query"].strip(), r.get("why", "")
            except Exception:
                pass
        # Heuristic: keep only the 3 most informative words (broader query).
        broader = " ".join(sorted(tokens(query), key=len, reverse=True)[:3]) or goal
        if broader.lower() == query.lower():
            broader = " ".join(tokens(goal)[:3])
        return broader, "broadened to the most distinctive keywords"
