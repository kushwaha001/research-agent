# Write-up: Autonomous Research Agent

**Task chosen.** A research agent: goal → plan → multi-source evidence → cited, actionable report. I chose it because it exercises every part of the rubric (planning, tool orchestration, failure recovery) against real, messy public data, with no curated dataset.

## Design decisions

- **An explicit pipeline with LLM decisions inside it, not a free-form ReAct loop.** The control flow (recall → plan → execute → dedupe → reflect → synthesise → verify) is fixed code. The LLM decides *what* happens inside each stage: which sources, which queries, how to reformulate, which gaps to fill, what the evidence means. This keeps runs bounded (at most one reflection round and a capped number of steps), cheap (about 3–5 LLM calls), and easy to debug, while the agent still makes its own choices.
- **Plan first, as validated data.** The plan is JSON checked with pydantic. Tools the LLM invents are dropped, at least two distinct real tools are required, and an invalid plan gets one corrective retry before the rule-based planner takes over. The plan is shown to the user before any tool runs.
- **Layered self-correction per step.** Timeouts and errors get exponential-backoff retries. Empty results get an LLM query reformulation. A dead source gets a fallback source. If everything fails, the report says so instead of breaking silently. Malformed payloads are caught by schema validation. A `FaultInjector` can trigger each of these on any tool, and there is a test for each.
- **Grounding over fluency.** Sources are numbered and the model must cite `[n]`. After synthesis, any citation to a source that doesn't exist is stripped, uncited claims are marked "(unverified)", and both are logged as warnings.
- **Everything is an event.** One `TraceEvent` stream feeds the live UI (SSE), the CLI, the `.log` transcripts and the report's "How this report was produced" section, so the audit trail can't drift from what actually happened.
- **Works in degraded mode.** Without an LLM key it plans by rules and summarises extractively. In `OFFLINE_MODE` it uses deterministic simulated tools, which makes the 15-test suite fast and independent of the network.
- **Few dependencies.** It needs only httpx, FastAPI, pydantic and reportlab, with no agent framework. The LLM client speaks the OpenAI-compatible protocol, so Groq, Gemini, OpenAI and Ollama are interchangeable through one environment variable.

## Limitations

- Relevance and dedup are lexical. Paraphrased duplicates or relevant results that share no keywords can slip past the filter.
- Scraped sources (DuckDuckGo HTML, page text extraction) are fragile and can be rate-limited. The fallbacks cover this, but a keyed search API would be more reliable.
- Google News links are redirects, so full-text enrichment of news items often fails; the agent then falls back to the snippets.
- Checking citations confirms that each cited source *exists*, not that it actually *supports* the claim.
- Memory recall uses token overlap, and SQLite on free hosts is wiped on redeploy.

## With more time

1. Embedding-based relevance, near-dup clustering and memory recall, plus cross-encoder re-ranking.
2. Entailment checking of each claim against the source it cites, for a real faithfulness score.
3. An evaluation harness: a set of goals with rubric grading (coverage, citation precision, recovery rate) tracked over time, with the fault-injection matrix run in CI.
4. Source credibility weighting and conflict detection (flagging when sources disagree).
5. Postgres for memory, per-user history and response caching, and streaming token output during synthesis.
