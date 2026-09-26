"""Render a RunResult to Markdown and PDF."""
from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from .models import Finding, RunResult


def _cite(f: Finding) -> str:
    return f.text + ("" if not f.citations else " " + "".join(f"[{c}]" for c in f.citations))


def to_markdown(r: RunResult) -> str:
    rep, s = r.report, r.stats
    out = [f"# {rep.title}", "",
           f"> **Goal:** {r.goal}  ",
           f"> **Generated:** {datetime.fromtimestamp(r.trace[0].ts if r.trace else 0):%Y-%m-%d %H:%M} · **LLM:** {r.llm} · "
           f"**Sources kept:** {s.get('kept', 0)}/{s.get('raw', 0)} · **Tool calls:** {s.get('tool_calls', 0)} · "
           f"**Recovered failures:** {s.get('recovered', 0)}", "",
           "## Executive summary", "", rep.summary, "", "## Key points", ""]
    out += [f"- {_cite(f)}" for f in rep.key_points] + ["", "## Important findings", ""]
    out += [f"- {_cite(f)}" for f in rep.findings] + ["", "## Actionable insights", ""]
    out += [f"{i}. {x}" for i, x in enumerate(rep.insights, 1)] + [""]
    if rep.limitations:
        out += ["## Limitations & caveats", ""] + [f"- {x}" for x in rep.limitations] + [""]
    out += ["## References", ""]
    out += [f"{d.id}. [{d.title}]({d.url}) — *{d.source}*" + (f", {d.published[:16]}" if d.published else "") for d in rep.references]
    out += ["", "## How this report was produced", "",
            f"**Interpretation:** {r.plan.interpretation}", "",
            f"**Source selection:** {r.plan.source_rationale}", "",
            "| # | Tool | Query | Status | Attempts | Results | Recovery |", "|---|---|---|---|---|---|---|"]
    out += [f"| {x.step_id} | {x.tool} | {x.query} | {x.status} | {x.attempts} | {len(x.docs)} | {x.recovery or x.error or '—'} |"
            for x in r.steps]
    out += ["", f"Deduplication: {s.get('url_dupes', 0)} duplicate URLs, {s.get('near_dupes', 0)} near-duplicate texts and "
            f"{s.get('irrelevant', 0)} irrelevant results removed.", ""]
    return "\n".join(out)


def to_pdf(r: RunResult) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    ink, accent = colors.HexColor("#1b1f24"), colors.HexColor("#2f5bd3")
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, leading=24, textColor=ink, alignment=0, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, textColor=accent, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14, textColor=ink)
    meta = ParagraphStyle("m", parent=body, fontSize=8.5, textColor=colors.HexColor("#5a6270"))
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10)

    def P(t: str, st=body) -> Paragraph:
        return Paragraph(escape(t), st)

    def bullets(items, numbered=False):
        return ListFlowable([ListItem(P(i), leftIndent=12) for i in items],
                            bulletType="1" if numbered else "bullet", start="1" if numbered else None, leftIndent=14)

    rep, s = r.report, r.stats
    story = [P(rep.title, h1),
             P(f"Goal: {r.goal}  ·  LLM: {r.llm}  ·  Sources kept {s.get('kept', 0)}/{s.get('raw', 0)}  ·  "
               f"Tool calls {s.get('tool_calls', 0)}  ·  Recovered failures {s.get('recovered', 0)}", meta),
             Spacer(1, 6), P("Executive summary", h2), P(rep.summary),
             P("Key points", h2), bullets([_cite(f) for f in rep.key_points]),
             P("Important findings", h2), bullets([_cite(f) for f in rep.findings]),
             P("Actionable insights", h2), bullets(rep.insights, numbered=True)]
    if rep.limitations:
        story += [P("Limitations & caveats", h2), bullets(rep.limitations)]
    story += [P("References", h2)]
    for d in rep.references:
        story.append(Paragraph(f'[{d.id}] <link href="{escape(d.url)}" color="#2f5bd3">{escape(d.title)}</link> '
                               f'<font color="#5a6270">— {d.source}</font>', small))
    story += [P("Execution trace", h2)]
    rows = [["#", "Tool", "Query", "Status", "Tries", "Hits"]] + [
        [str(x.step_id), x.tool, Paragraph(escape(x.query), small), x.status, str(x.attempts), str(len(x.docs))] for x in r.steps]
    t = Table(rows, colWidths=[8 * mm, 25 * mm, 80 * mm, 22 * mm, 12 * mm, 12 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")), ("FONTSIZE", (0, 0), (-1, -1), 8),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9ced8")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                      title=rep.title, author="Autonomous Research Agent").build(story)
    return buf.getvalue()
