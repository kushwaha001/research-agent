"""Minimal async client for any OpenAI-compatible chat API, with JSON-mode + repair.

Keeping this dependency-free (just httpx) makes the provider swappable: Groq, Gemini,
OpenAI or a local Ollama all speak the same /chat/completions protocol.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from .config import Settings


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating ```json fences and leading chatter."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} block.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise LLMError(f"Model did not return JSON: {text[:200]!r}")


class LLMClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.calls = 0
        self.tokens = 0

    async def chat(self, system: str, user: str, *, json_mode: bool = True, temperature: float = 0.2) -> str:
        if self.s.provider == "gemini":
            return await self._gemini(system, user, json_mode, temperature)
        return await self._openai(system, user, json_mode, temperature)

    async def _gemini(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        """Native Gemini generateContent. Works with AI Studio keys (AIza...) and, via the
        Vertex AI express endpoint, with newer AQ.* keys."""
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, **({"responseMimeType": "application/json"} if json_mode else {})},
        }
        m = self.s.model
        urls = [f"{self.s.base_url}/models/{m}:generateContent",
                f"https://aiplatform.googleapis.com/v1/publishers/google/models/{m}:generateContent"]
        if self.s.api_key.startswith("AQ."):
            urls.reverse()  # express-mode keys usually belong to Vertex AI
        last: Exception | None = None
        for attempt in range(3):
            for url in urls:
                try:
                    async with httpx.AsyncClient(timeout=90) as c:
                        r = await c.post(url, json=body, headers={"x-goog-api-key": self.s.api_key})
                    if r.status_code in (401, 403, 404):
                        last = LLMError(f"{url.split('/')[2]} HTTP {r.status_code}: {r.text[:200]}")
                        continue  # try the other endpoint
                    if r.status_code == 429 or r.status_code >= 500:
                        raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                    if r.status_code >= 400:
                        raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                    data = r.json()
                    self.calls += 1
                    self.tokens += data.get("usageMetadata", {}).get("totalTokenCount", 0)
                    urls = [url]  # stick with the endpoint that works
                    return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
                except (httpx.HTTPError, LLMError, KeyError, IndexError) as e:
                    last = e
            await asyncio.sleep(1.5 * (2**attempt))
        raise LLMError(f"Gemini call failed: {last}")

    async def _openai(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        payload: dict[str, Any] = {
            "model": self.s.model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.s.api_key}"}
        last: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=60) as c:
                    r = await c.post(f"{self.s.base_url}/chat/completions", json=payload, headers=headers)
                if r.status_code == 429 or r.status_code >= 500:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    # Some providers reject response_format; retry once without it.
                    if json_mode and "response_format" in payload:
                        payload.pop("response_format")
                        continue
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                body = r.json()
                self.calls += 1
                self.tokens += body.get("usage", {}).get("total_tokens", 0)
                return body["choices"][0]["message"]["content"]
            except (httpx.HTTPError, LLMError, KeyError) as e:
                last = e
                await asyncio.sleep(1.5 * (2**attempt))
        raise LLMError(f"LLM call failed after retries: {last}")

    async def chat_json(self, system: str, user: str, **kw: Any) -> dict:
        reply = await self.chat(system, user, **kw)
        try:
            return extract_json(reply)
        except (LLMError, json.JSONDecodeError):
            # Self-repair: ask the model to fix its own output once.
            fixed = await self.chat(
                "You convert text into strictly valid JSON. Output JSON only.",
                f"Convert this into valid JSON with the same content:\n\n{reply}",
            )
            return extract_json(fixed)
