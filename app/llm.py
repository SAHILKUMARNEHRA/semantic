from __future__ import annotations

import asyncio
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


def _sanitize_header_token(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").strip())


def _sanitize_header_value(value: str) -> str:
    v = (value or "").strip()
    v = re.sub(r"[\u2028\u2029]+", " ", v)
    v = re.sub(r"\s+", " ", v)
    return v.strip()


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
        self._base_url = (base_url or "").strip().rstrip("/")
        self._api_key = _sanitize_header_token(api_key)
        self._model = model
        self._extra_headers = {k: _sanitize_header_value(v) for k, v in (extra_headers or {}).items()}

    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5), reraise=True)
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
                if e.response.status_code == 429:
                    ra = (e.response.headers.get("retry-after") or "").strip()
                    wait_s = 0.0
                    if ra:
                        try:
                            wait_s = float(ra)
                        except Exception:
                            wait_s = 0.0
                    if wait_s <= 0.0:
                        body_text = (e.response.text or "")
                        m = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", body_text, re.IGNORECASE)
                        if m:
                            try:
                                wait_s = float(m.group(1))
                            except Exception:
                                wait_s = 0.0
                    if wait_s > 0.0:
                        await asyncio.sleep(min(wait_s + 0.2, 35.0))
                body = (e.response.text or "").strip()
                raise RuntimeError(f"LLM HTTP {e.response.status_code}: {body[:2000]}") from e
            return resp.json()

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    async def _list_models(self) -> list[str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
            resp = await client.get(f"{self._base_url}/models", headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = (e.response.text or "").strip()
                raise RuntimeError(f"LLM models HTTP {e.response.status_code}: {body[:2000]}") from e
            data = resp.json()
            ids: list[str] = []
            for item in data.get("data", []) or []:
                mid = item.get("id")
                if isinstance(mid, str) and mid.strip():
                    ids.append(mid.strip())
            return ids

    def _pick_model(self, model_ids: list[str]) -> str | None:
        if not model_ids:
            return None
        lowered = [(m, m.lower()) for m in model_ids]
        for m, ml in lowered:
            if "llama" in ml and "70" in ml:
                return m
        for m, ml in lowered:
            if "llama" in ml:
                return m
        return model_ids[0]

    async def generate(self, prompt: str) -> LlmResult:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "Return only SQL."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        try:
            data = await self._call(payload)
        except RuntimeError as e:
            msg = str(e).lower()
            if any(k in msg for k in ["model_decommissioned", "model_not_found", "does not exist", "no longer supported"]):
                model_ids = await self._list_models()
                picked = self._pick_model(model_ids)
                if not picked:
                    raise
                self._model = picked
                payload["model"] = picked
                data = await self._call(payload)
            else:
                raise
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


def _normalize_groq_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return m
    lowered = m.lower()
    if lowered == "llama3-70b-8192":
        return "llama-3.1-70b-versatile"
    if lowered == "llama3-8b-8192":
        return "llama-3.1-8b-instant"
    return m


def build_llm_client() -> LlmClient:
    provider = settings.llm_provider

    if provider == "groq":
        if settings.groq_api_key:
            model = _normalize_groq_model(settings.groq_model or settings.llm_model)
            return OpenAICompatibleClient(settings.groq_base_url, settings.groq_api_key, model)

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
