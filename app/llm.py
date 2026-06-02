from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .beaver_data import QueryExample, TableSchema
from .settings import settings

logger = logging.getLogger(__name__)


_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    m = _SQL_FENCE_RE.search(text or "")
    if m:
        return m.group(1).strip().rstrip(";")

    s = (text or "").strip()
    for prefix in ["WITH", "SELECT"]:
        idx = s.upper().find(prefix)
        if idx != -1:
            return s[idx:].strip().rstrip(";")
    return s.rstrip(";")


def build_prompt(
    question: str,
    retrieved_tables: list[str],
    table_schemas: dict[str, TableSchema],
    examples: list[QueryExample],
) -> str:
    schema_blocks: list[str] = []
    for t in retrieved_tables:
        ts = table_schemas.get(t)
        if ts:
            schema_blocks.append(f"TABLE: {t}\n{ts.schema_text}")

    example_blocks: list[str] = []
    for ex in examples[:5]:
        example_blocks.append(f"Q: {ex.question}\nSQL: {ex.sql}")

    schema_section = "\n\n".join(schema_blocks) if schema_blocks else "(no schema context available)"
    examples_section = "\n\n".join(example_blocks) if example_blocks else ""

    parts: list[str] = [
        "You are an expert text-to-SQL system.",
        "Task: produce a single SQLite-compatible SQL query answering the user question.",
        "Rules:",
        "- Output only SQL (no explanation).",
        "- Use only tables/columns from the provided schema context.",
        "- Prefer explicit JOINs and clear aliases.",
        "- If the question is ambiguous, make the most reasonable assumption based on schema names.",
        "",
        "Schema context:",
        schema_section,
        "",
    ]

    if examples_section:
        parts.extend(["Examples:", examples_section, ""])

    parts.extend(["User question:", question.strip(), "", "SQL:"])
    return "\n".join(parts).strip()


@dataclass(frozen=True)
class LlmResult:
    raw_text: str
    sql: str


class LlmClient:
    async def generate(self, prompt: str) -> LlmResult:
        raise NotImplementedError


class OpenAICompatibleClient(LlmClient):
    def __init__(self, base_url: str, api_key: str, model: str, extra_headers: dict[str, str] | None = None):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._extra_headers = extra_headers or {}

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    async def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
            resp = await client.post(f"{self._base_url}/chat/completions", headers=headers, json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = (e.response.text or "").strip()
                raise RuntimeError(f"LLM HTTP {e.response.status_code}: {body[:2000]}") from e
            return resp.json()

    async def generate(self, prompt: str) -> LlmResult:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "Return only SQL."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        data = await self._call(payload)
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        sql = extract_sql(content)
        return LlmResult(raw_text=content, sql=sql)


class OllamaClient(LlmClient):
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    async def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = (e.response.text or "").strip()
                raise RuntimeError(f"Ollama HTTP {e.response.status_code}: {body[:2000]}") from e
            return resp.json()

    async def generate(self, prompt: str) -> LlmResult:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        data = await self._call(payload)
        content = (data.get("message") or {}).get("content", "")
        sql = extract_sql(content)
        return LlmResult(raw_text=content, sql=sql)


def build_llm_client() -> LlmClient:
    provider = settings.llm_provider

    if provider == "groq":
        if settings.groq_api_key:
            return OpenAICompatibleClient(settings.groq_base_url, settings.groq_api_key, settings.llm_model)

    if provider == "openai":
        if settings.openai_api_key:
            return OpenAICompatibleClient(settings.openai_base_url, settings.openai_api_key, settings.llm_model)

    if provider == "openrouter":
        if settings.openrouter_api_key:
            return OpenAICompatibleClient(
                settings.openrouter_base_url,
                settings.openrouter_api_key,
                settings.llm_model,
                extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "nst-challenge"},
            )

    if provider == "together":
        if settings.together_api_key:
            return OpenAICompatibleClient(
                settings.together_base_url, settings.together_api_key, settings.llm_model
            )

    if provider == "ollama":
        return OllamaClient(settings.ollama_base_url, settings.llm_model)

    candidates: list[tuple[str, str | None, str]] = [
        ("groq", settings.groq_api_key, settings.groq_base_url),
        ("openrouter", settings.openrouter_api_key, settings.openrouter_base_url),
        ("together", settings.together_api_key, settings.together_base_url),
        ("openai", settings.openai_api_key, settings.openai_base_url),
    ]
    for prov, key, base in candidates:
        if key:
            return OpenAICompatibleClient(base, key, settings.llm_model)

    raise RuntimeError(
        "No LLM API key found. Set GROQ_API_KEY / OPENROUTER_API_KEY / TOGETHER_API_KEY / OPENAI_API_KEY "
        "or switch LLM_PROVIDER=ollama."
    )
