"""The orchestrator: recall → plan → execute (parallel) → dedupe/rank → reflect & fill gaps → synthesise → verify → store."""
from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

from .config import Settings, get_settings
from .executor import Executor
from .llm import LLMClient
from .memory import Memory
from .models import PlanStep, RunResult, TraceEvent
from .planner import Planner
from .processing import dedupe_and_rank
from .synthesizer import Synthesizer
from .tools import FaultInjector, build_tools

Emit = Callable[[TraceEvent], Awaitable[None]]


async def _noop(_: TraceEvent) -> None:
    return None


class ResearchAgent:
    def __init__(self, settings: Settings | None = None, *, fault: str | None = None, fault_tool: str | None = None,
                 use_memory: bool = True, max_gap_rounds: int = 1):
        self.s = settings or get_settings()
        self.injector = FaultInjector(fault, fault_tool) if fault else None
        self.tools = build_tools(self.s.offline, self.s.tool_timeout, self.injector)
        self.llm = None if self.s.offline else LLMClient(self.s)
        self.planner = Planner(self.llm, self.tools)
        self.synth = Synthesizer(self.llm)
        self.memory = Memory(self.s.db_path) if use_memory else None
        self.max_gap_rounds = max_gap_rounds

    async def run(self, goal: str, emit: Emit = _noop) -> RunResult:
        goal = " ".join(goal.split())[:500]
        if len(goal) < 3:
            raise ValueError("Goal is too short. Describe what you want researched.")
        run_id = uuid.uuid4().hex[:10]
        trace: list[TraceEvent] = []
        t0 = time.perf_counter()

        async def log(ev: TraceEvent) -> None:
            trace.append(ev)
            await emit(ev)

        await log(TraceEvent(type="info", message=f"Goal received. LLM: {self.s.llm_label}",
                             data={"run_id": run_id, "goal": goal, "llm": self.s.llm_label, "offline": self.s.offline,
                                   "fault": self.injector and {"mode": self.injector.mode, "tool": self.injector.target}}))

        # 1. Memory recall
        memory_ctx = ""
        if self.memory:
            past = self.memory.recall(goal)
            if past:
                memory_ctx = "\n".join(f"- ({p['similarity']}) {p['goal']}: {p['summary'][:300]}" for p in past)
                await log(TraceEvent(type="memory", message=f"Recalled {len(past)} related past run(s)", data={"matches": past}))

        # 2. Plan
        plan, fallback = await self.planner.plan(goal, memory_ctx)
        if fallback and self.llm:
            await log(TraceEvent(type="warning", message="LLM planner failed validation twice → using rule-based planner"))
        await log(TraceEvent(type="plan", message=f"Plan: {len(plan.steps)} steps across {len({s.tool for s in plan.steps})} sources",
                             data=plan.model_dump()))

        # 3. Execute
        ex = Executor(self.tools, self.planner, log, max_retries=self.s.max_retries, timeout=self.s.tool_timeout)
        results = await ex.run(plan.steps, goal)
        docs = [d for r in results for d in r.docs]
        kept, stats = dedupe_and_rank(docs, goal)
        await log(TraceEvent(type="info", message=f"Dedup & relevance filter: kept {stats['kept']} of {stats['raw']} results", data=stats))

        # 4. Reflect: are the sub-questions covered? If not, run targeted follow-up searches.
        for rnd in range(self.max_gap_rounds):
            gaps = await self.synth.find_gaps(plan, kept, [n for n in self.tools if n != "page_reader"])
            if not gaps:
                await log(TraceEvent(type="info", message="Reflection: evidence covers all sub-questions"))
                break
            base = max(s.id for s in plan.steps)
            extra = [PlanStep(id=base + i + 1, tool=g["tool"], query=g["query"], rationale=f"Gap: {g.get('sub_question', '')}")
                     for i, g in enumerate(gaps)]
            plan.steps += extra
            await log(TraceEvent(type="plan", message=f"Reflection found {len(gaps)} gap(s) → adding follow-up step(s)",
                                 data={**plan.model_dump(), "added": [s.id for s in extra]}))
            more = await ex.run(extra, goal)
            results += more
            docs += [d for r in more for d in r.docs]
            kept, stats = dedupe_and_rank(docs, goal)

        # 5. Read top pages for depth
        n = await ex.enrich(kept)
        if n:
            await log(TraceEvent(type="info", message=f"Read full text of {n} top page(s) for extra depth"))

        # 6. Synthesise + verify citations
        await log(TraceEvent(type="info", message=f"Synthesising report from {len(kept)} sources"))
        report, warnings = await self.synth.write(plan, kept)
        for w in warnings[:6]:
            await log(TraceEvent(type="warning", message=f"Self-check: {w}"))

        failed = [r for r in results if r.status == "failed"]
        if failed:
            report.limitations.append(
                "Some sources could not be reached: " + ", ".join(f"{r.tool} (step {r.step_id})" for r in failed) + ".")
        stats.update(tool_calls=ex.tool_calls, steps=len(results), recovered=sum(r.status == "recovered" for r in results),
                     failed=len(failed), llm_calls=self.llm.calls if self.llm else 0,
                     llm_tokens=self.llm.tokens if self.llm else 0, citation_warnings=len(warnings),
                     duration_s=round(time.perf_counter() - t0, 2),
                     sources_used=sorted({d.source for d in kept}))
        result = RunResult(run_id=run_id, goal=goal, llm=self.s.llm_label, plan=plan, steps=results,
                           report=report, stats=stats, trace=trace)
        await log(TraceEvent(type="report", message="Report ready", data=report.model_dump()))
        await log(TraceEvent(type="done", message=f"Done in {stats['duration_s']}s", data={"run_id": run_id, "stats": stats}))
        result.trace = trace
        if self.memory:
            self.memory.save(run_id, goal, self.s.llm_label, report.summary, result.model_dump_json())
        return result
