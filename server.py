"""FastAPI server: streams the agent's trace over Server-Sent Events and serves the UI."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from agent import ResearchAgent
from agent.config import get_settings
from agent.memory import Memory
from agent.models import RunResult, TraceEvent
from agent.report import to_markdown, to_pdf
from agent.tools import SEARCH_TOOLS

app = FastAPI(title="Autonomous Research Agent")
WEB = Path(__file__).parent / "web"
settings = get_settings()
memory = Memory(settings.db_path)
_sem = asyncio.Semaphore(4)  # cap concurrent runs on small free-tier hosts


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "llm": settings.llm_label, "offline": settings.offline,
            "tools": [{"name": t.name, "description": t.description} for t in SEARCH_TOOLS]}


@app.get("/api/run")
async def run(goal: str = Query(..., min_length=3, max_length=500), fault: str | None = None,
              fault_tool: str | None = None, offline: bool = False) -> StreamingResponse:
    if fault and fault not in {"timeout", "outage", "empty", "garbage"}:
        raise HTTPException(400, "invalid fault mode")
    q: asyncio.Queue[TraceEvent | None] = asyncio.Queue()

    async def emit(ev: TraceEvent) -> None:
        await q.put(ev)

    async def worker() -> None:
        s = get_settings()
        if offline:
            s.offline = True
        try:
            async with _sem:
                await ResearchAgent(s, fault=fault, fault_tool=fault_tool or "web_search").run(goal, emit)
        except Exception as e:  # surfaced to the UI instead of a silent hang
            await q.put(TraceEvent(type="error", message=f"{type(e).__name__}: {e}"))
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())

    async def stream():
        try:
            while (ev := await q.get()) is not None:
                yield f"event: {ev.type}\ndata: {json.dumps(ev.model_dump(), ensure_ascii=False)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/history")
def history() -> list[dict]:
    return memory.history()


def _load(run_id: str) -> RunResult:
    data = memory.get(run_id)
    if not data:
        raise HTTPException(404, "run not found")
    return RunResult(**data)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    return JSONResponse(_load(run_id).model_dump())


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    memory.delete(run_id)
    return {"ok": True}


@app.get("/api/runs/{run_id}/report.md")
def report_md(run_id: str) -> Response:
    return Response(to_markdown(_load(run_id)), media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="research-{run_id}.md"'})


@app.get("/api/runs/{run_id}/report.pdf")
def report_pdf(run_id: str) -> Response:
    return Response(to_pdf(_load(run_id)), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="research-{run_id}.pdf"'})
