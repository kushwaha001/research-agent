"""Executes plan steps: parallel waves, retries with backoff, reformulation and fallbacks."""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from .models import Document, PlanStep, StepResult, TraceEvent
from .planner import Planner
from .tools import FALLBACKS, Tool, ToolError

Emit = Callable[[TraceEvent], Awaitable[None]]


class Executor:
    def __init__(self, tools: dict[str, Tool], planner: Planner, emit: Emit, *, max_retries: int = 2, timeout: float = 12.0):
        self.tools, self.planner, self.emit = tools, planner, emit
        self.max_retries, self.timeout = max_retries, timeout
        self.tool_calls = 0

    async def _call(self, tool: str, query: str) -> list[Document]:
        """One guarded tool call. Raises ToolError for timeouts, errors and invalid payloads."""
        self.tool_calls += 1
        t = self.tools[tool]
        try:
            raw = await asyncio.wait_for(t.search(query, 6), timeout=self.timeout + 2)
        except asyncio.TimeoutError as e:
            raise ToolError(f"{tool}: timed out") from e
        valid = [d for d in raw if d.url.startswith("http") and d.title.strip()]
        if raw and not valid:
            raise ToolError(f"{tool}: returned {len(raw)} malformed result(s) (schema validation failed)")
        return valid[:6]

    async def _call_with_retry(self, step: PlanStep, tool: str, query: str) -> tuple[list[Document], int, list[str]]:
        errors, attempts = [], 0
        for attempt in range(self.max_retries + 1):
            attempts += 1
            try:
                return await self._call(tool, query), attempts, errors
            except ToolError as e:
                errors.append(str(e))
                if attempt < self.max_retries:
                    delay = 0.5 * (2**attempt)
                    await self.emit(TraceEvent(type="recovery", message=f"Step {step.id}: {e} → retrying in {delay:.1f}s",
                                               data={"step": step.id, "action": "retry", "attempt": attempts}))
                    await asyncio.sleep(delay)
        raise ToolError(f"{errors[-1]} (x{len(errors)})")

    async def run_step(self, step: PlanStep, goal: str) -> StepResult:
        t0 = time.perf_counter()
        await self.emit(TraceEvent(type="step_start", message=f"Step {step.id}: {step.tool}('{step.query}')",
                                   data=step.model_dump()))
        recovery: list[str] = []
        attempts, docs, status, error = 0, [], "ok", None
        try:
            docs, attempts, errs = await self._call_with_retry(step, step.tool, step.query)
            if errs:
                status, recovery = "recovered", [f"succeeded after {len(errs)} retr{'y' if len(errs) == 1 else 'ies'}"]
            if not docs:
                # Self-correction #1: the query was probably too narrow -> reformulate.
                new_q, why = await self.planner.reformulate(goal, step.query, step.tool)
                await self.emit(TraceEvent(type="recovery", data={"step": step.id, "action": "reformulate", "query": new_q},
                                           message=f"Step {step.id}: no results → reformulated query to '{new_q}' ({why})"))
                docs, a, _ = await self._call_with_retry(step, step.tool, new_q)
                attempts += a
                status, recovery = ("recovered", recovery + [f"reformulated to '{new_q}'"]) if docs else ("failed", recovery)
                if not docs:
                    error = "no results even after reformulation"
        except ToolError as e:
            error, status = str(e), "failed"
            attempts = attempts or self.max_retries + 1

        if status == "failed" and step.tool in FALLBACKS:
            # Self-correction #2: switch to an alternative source.
            fb = FALLBACKS[step.tool]
            await self.emit(TraceEvent(type="recovery", data={"step": step.id, "action": "fallback", "tool": fb},
                                       message=f"Step {step.id}: {step.tool} unavailable ({error}) → falling back to {fb}"))
            try:
                docs, a, _ = await self._call_with_retry(step, fb, step.query)
                attempts += a
                if docs:
                    status, recovery = "recovered", recovery + [f"fell back to {fb}"]
            except ToolError as e:
                error = f"{error}; fallback {fb} also failed: {e}"

        res = StepResult(step_id=step.id, tool=step.tool, query=step.query, status=status, attempts=attempts,
                         docs=docs, error=error if status == "failed" else None,
                         recovery="; ".join(recovery) or None, duration_ms=int((time.perf_counter() - t0) * 1000))
        await self.emit(TraceEvent(
            type="step_end",
            message=f"Step {step.id} {status}: {len(docs)} result(s) in {res.duration_ms} ms" + (f" — {error}" if status == "failed" else ""),
            data={**res.model_dump(exclude={"docs"}), "titles": [d.title for d in docs[:4]]}))
        return res

    async def run(self, steps: list[PlanStep], goal: str) -> list[StepResult]:
        """Run steps in dependency 'waves'; everything inside a wave runs concurrently."""
        done: dict[int, StepResult] = {}
        pending = list(steps)
        while pending:
            wave = [s for s in pending if all(d in done for d in s.depends_on)]
            if not wave:  # cyclic / dangling deps -> just run the rest
                wave = pending
            if len(wave) > 1:
                await self.emit(TraceEvent(type="info", message=f"Running {len(wave)} steps in parallel",
                                           data={"steps": [s.id for s in wave]}))
            for r in await asyncio.gather(*(self.run_step(s, goal) for s in wave)):
                done[r.step_id] = r
            pending = [s for s in pending if s.id not in done]
        return [done[s.id] for s in steps]

    async def enrich(self, docs: list[Document], n: int = 3) -> int:
        """Read the full text of the top web/news hits (best-effort, parallel)."""
        targets = [d for d in docs if d.source in ("web_search", "news") and len(d.snippet) < 400][:n]

        async def one(d: Document) -> bool:
            try:
                page = await asyncio.wait_for(self.tools["page_reader"].search(d.url), timeout=self.timeout)
                self.tool_calls += 1
                d.snippet = (d.snippet + " … " + page[0].snippet)[:1800]
                return True
            except Exception:
                return False

        return sum(await asyncio.gather(*(one(d) for d in targets)))
