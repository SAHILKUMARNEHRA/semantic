from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hf_cache_dir: Path = Field(default=Path(".cache/hf"))

    sqlite_path: Path = Field(default=Path(".cache/beaver.sqlite"))

    local_table_dir: Path = Field(default=Path("../table"), alias="LOCAL_TABLE_DIR")
    local_query_dir: Path = Field(default=Path("../query"), alias="LOCAL_QUERY_DIR")

    max_question_length: int = Field(default=512)
    retrieval_top_k: int = Field(default=10)

    llm_provider: Literal["groq", "openai", "openrouter", "together", "ollama", "none"] = Field(
        default="groq"
    )
    llm_model: str = Field(default="llama3-70b-8192", alias="LLM_MODEL")

    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")

    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    together_base_url: str = Field(default="https://api.together.xyz/v1", alias="TOGETHER_BASE_URL")
    together_api_key: str | None = Field(default=None, alias="TOGETHER_API_KEY")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    request_timeout_s: float = Field(default=60.0)

    def ensure_cache_dirs(self) -> None:
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
