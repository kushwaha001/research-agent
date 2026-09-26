"""Runtime configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# LLM providers (Gemini uses its native API; the rest are OpenAI-compatible).
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "gemini-2.5-flash", "GEMINI_API_KEY"),  # native API
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", "llama3.1", ""),
}


@dataclass
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "").lower())
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    offline: bool = field(default_factory=lambda: os.getenv("OFFLINE_MODE", "0") == "1")
    tool_timeout: float = field(default_factory=lambda: float(os.getenv("TOOL_TIMEOUT", "8")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "2")))
    db_path: str = field(default_factory=lambda: os.getenv("MEMORY_DB", "data/memory.sqlite3"))
    reports_dir: str = field(default_factory=lambda: os.getenv("REPORTS_DIR", "data/reports"))

    def __post_init__(self) -> None:
        # Auto-detect a provider from whichever key is present.
        if not self.provider:
            for name, (_, _, env) in PROVIDERS.items():
                if env and os.getenv(env):
                    self.provider = name
                    break
        if self.provider in PROVIDERS:
            url, model, env = PROVIDERS[self.provider]
            self.base_url = self.base_url or url
            self.model = self.model or model
            self.api_key = self.api_key or (os.getenv(env, "") if env else "ollama")
        if not self.api_key:
            self.offline = True  # no LLM available -> deterministic offline mode

    @property
    def llm_label(self) -> str:
        return "offline (heuristic)" if self.offline else f"{self.provider}:{self.model}"


def get_settings() -> Settings:
    return Settings()
