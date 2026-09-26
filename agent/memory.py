"""Long-term memory of past research runs (SQLite; zero setup)."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .processing import jaccard, tokens


class Memory:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, goal TEXT, created REAL, llm TEXT,
                summary TEXT, result_json TEXT)""")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save(self, run_id: str, goal: str, llm: str, summary: str, result_json: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
                      (run_id, goal, time.time(), llm, summary, result_json))

    def history(self, limit: int = 30) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, goal, created, llm, summary FROM runs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [dict(zip(["id", "goal", "created", "llm", "summary"], r)) for r in rows]

    def get(self, run_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT result_json FROM runs WHERE id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, run_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def recall(self, goal: str, threshold: float = 0.35, k: int = 2) -> list[dict]:
        """Find earlier runs on a similar goal (token Jaccard)."""
        q = set(tokens(goal))
        with self._conn() as c:
            rows = c.execute("SELECT id, goal, created, summary FROM runs ORDER BY created DESC LIMIT 200").fetchall()
        scored = [(jaccard(q, set(tokens(g))), i, g, t, s) for i, g, t, s in rows]
        scored = [x for x in scored if x[0] >= threshold]
        scored.sort(key=lambda x: -x[0])
        return [{"similarity": round(s, 2), "id": i, "goal": g, "created": t, "summary": su} for s, i, g, t, su in scored[:k]]
