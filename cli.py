"""Command-line interface.

    python cli.py "Research the top 3 developments in small language models from the last week"
    python cli.py "..." --fault timeout --fault-tool news     # demo recovery
    python cli.py "..." --offline --out samples/run1          # no network needed
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agent import ResearchAgent
from agent.config import get_settings
from agent.models import TraceEvent
from agent.report import to_markdown, to_pdf

ICONS = {"plan": "🧭", "step_start": "→", "step_end": "✓", "recovery": "⟲", "warning": "⚠", "memory": "🧠",
         "info": "·", "report": "📄", "done": "✔", "error": "✗"}


async def printer(ev: TraceEvent) -> None:
    print(f"{ICONS.get(ev.type, '·')} [{ev.type}] {ev.message}", flush=True)
    if ev.type == "plan" and "steps" in ev.data:
        d = ev.data
        print(f"    interpretation: {d['interpretation']}")
        for q in d["sub_questions"]:
            print(f"    ? {q}")
        for s in d["steps"]:
            print(f"    {s['id']}. {s['tool']:<11} '{s['query']}'  — {s['rationale']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Autonomous research agent")
    ap.add_argument("goal")
    ap.add_argument("--offline", action="store_true", help="simulated tools + heuristic LLM (no network)")
    ap.add_argument("--fault", choices=["timeout", "outage", "empty", "garbage"], help="inject a failure")
    ap.add_argument("--fault-tool", default="web_search")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--out", help="write <out>.md, <out>.pdf, <out>.json and <out>.log")
    a = ap.parse_args()

    s = get_settings()
    if a.offline:
        s.offline = True
    agent = ResearchAgent(s, fault=a.fault, fault_tool=a.fault_tool, use_memory=not a.no_memory)
    log_lines: list[str] = []

    async def emit(ev: TraceEvent) -> None:
        await printer(ev)
        log_lines.append(json.dumps(ev.model_dump(), ensure_ascii=False))

    result = asyncio.run(agent.run(a.goal, emit))
    md = to_markdown(result)
    print("\n" + md)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text(md)
        out.with_suffix(".pdf").write_bytes(to_pdf(result))
        out.with_suffix(".json").write_text(result.model_dump_json(indent=2))
        out.with_suffix(".log").write_text("\n".join(log_lines))
        print(f"\nSaved {out}.md/.pdf/.json/.log", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
