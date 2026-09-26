"""Typed data structures shared across the agent."""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    """One piece of evidence returned by a tool."""
    title: str
    url: str
    snippet: str = ""
    source: str  # tool name
    published: str | None = None
    score: float = 0.0  # relevance score, filled by processing
    id: int = 0  # citation number, assigned after dedup


class PlanStep(BaseModel):
    id: int
    tool: str
    query: str
    rationale: str
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    interpretation: str
    sub_questions: list[str]
    steps: list[PlanStep]
    source_rationale: str = ""


class StepResult(BaseModel):
    step_id: int
    tool: str
    query: str
    status: Literal["ok", "recovered", "failed", "skipped"]
    attempts: int
    docs: list[Document] = Field(default_factory=list)
    error: str | None = None
    recovery: str | None = None
    duration_ms: int = 0


class Finding(BaseModel):
    text: str
    citations: list[int] = Field(default_factory=list)


class Report(BaseModel):
    title: str
    summary: str
    key_points: list[Finding]
    findings: list[Finding]
    insights: list[str]
    limitations: list[str] = Field(default_factory=list)
    references: list[Document] = Field(default_factory=list)


class TraceEvent(BaseModel):
    """Everything the agent does is emitted as a TraceEvent (UI + logs + transcripts)."""
    type: str  # plan | step_start | step_end | recovery | info | warning | report | done | error | memory
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)


class RunResult(BaseModel):
    run_id: str
    goal: str
    llm: str
    plan: Plan
    steps: list[StepResult]
    report: Report
    stats: dict[str, Any]
    trace: list[TraceEvent]
