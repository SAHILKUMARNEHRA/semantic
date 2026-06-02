from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableSchema:
    table_name: str
    schema_text: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class QueryExample:
    question: str
    sql: str
    gold_tables: list[str]
    raw: dict[str, Any]


def _hf_token_arg() -> str | bool | None:
    if settings.hf_token:
        return settings.hf_token
    return None


def _list_parquet_files(folder: Path) -> list[str]:
    if not folder.exists() or not folder.is_dir():
        return []
    files = sorted(str(p) for p in folder.glob("*.parquet") if p.is_file())
    return files


def _load_local_parquet_dataset(folder: Path):
    files = _list_parquet_files(folder)
    if not files:
        return None
    ds = load_dataset(
        "parquet",
        data_files={"train": files},
        cache_dir=str(settings.hf_cache_dir),
    )
    return ds["train"]


def _guess_first_existing_key(row: dict[str, Any], candidates: list[str]) -> str | None:
    for k in candidates:
        if k in row and row[k] not in (None, ""):
            return k
    return None


def _row_to_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in row.items():
        if v is None:
            continue
        if isinstance(v, str):
            vv = v.strip()
            if vv:
                parts.append(f"{k}: {vv}")
            continue
        if isinstance(v, (int, float, bool)):
            parts.append(f"{k}: {v}")
            continue
        if isinstance(v, list):
            flat = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        flat.append(s)
                else:
                    flat.append(str(item))
            if flat:
                parts.append(f"{k}: {', '.join(flat)}")
            continue
        parts.append(f"{k}: {str(v)}")
    return "\n".join(parts)


def load_tables() -> list[TableSchema]:
    settings.ensure_cache_dirs()

    local = _load_local_parquet_dataset(settings.local_table_dir)
    if local is not None:
        rows = local
    else:
        token = _hf_token_arg()
        try:
            ds = load_dataset(
                "beaverbench/beaver-table",
                token=token,
                cache_dir=str(settings.hf_cache_dir),
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to download/load beaverbench/beaver-table from Hugging Face. "
                "Provide access via HF_TOKEN (env var) or run `huggingface-cli login`, "
                "or set LOCAL_TABLE_DIR to a folder containing the downloaded *.parquet files. "
                f"Original error: {e}"
            ) from e

        split_name = "train" if "train" in ds else list(ds.keys())[0]
        rows = ds[split_name]

    tables: list[TableSchema] = []
    for row in rows:
        row_dict = dict(row)
        name_key = _guess_first_existing_key(
            row_dict, ["table_name", "table", "name", "table_id", "tableName"]
        )
        if not name_key:
            continue
        table_name = str(row_dict[name_key]).strip()
        if not table_name:
            continue

        schema_key = _guess_first_existing_key(
            row_dict,
            [
                "create_table_sql",
                "ddl",
                "schema",
                "schema_text",
                "table_schema",
                "columns",
                "column_names",
            ],
        )
        if schema_key and isinstance(row_dict.get(schema_key), str):
            schema_text = str(row_dict[schema_key]).strip()
        else:
            schema_text = _row_to_text(row_dict)

        tables.append(TableSchema(table_name=table_name, schema_text=schema_text, raw=row_dict))

    if not tables:
        logger.warning("Loaded beaver-table but found no tables after parsing")
    return tables


def load_queries(limit: int | None = None) -> list[QueryExample]:
    settings.ensure_cache_dirs()

    local = _load_local_parquet_dataset(settings.local_query_dir)
    if local is not None:
        rows = local
    else:
        token = _hf_token_arg()
        try:
            ds = load_dataset(
                "beaverbench/beaver-query",
                token=token,
                cache_dir=str(settings.hf_cache_dir),
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to download/load beaverbench/beaver-query from Hugging Face. "
                "Provide access via HF_TOKEN (env var) or run `huggingface-cli login`, "
                "or set LOCAL_QUERY_DIR to a folder containing the downloaded *.parquet files. "
                f"Original error: {e}"
            ) from e

        split_name = "train" if "train" in ds else list(ds.keys())[0]
        rows = ds[split_name]

    examples: list[QueryExample] = []
    for row in rows:
        row_dict = dict(row)
        q_key = _guess_first_existing_key(row_dict, ["question", "nl_question", "query", "text"])
        s_key = _guess_first_existing_key(row_dict, ["sql", "gold_sql", "query_sql"])
        t_key = _guess_first_existing_key(
            row_dict,
            ["gold_tables", "tables", "table_names", "used_tables", "table_ids"],
        )

        if not q_key or not s_key:
            continue

        question = str(row_dict[q_key]).strip()
        sql = str(row_dict[s_key]).strip()
        if not question or not sql:
            continue

        gold_tables: list[str] = []
        if t_key:
            t_val = row_dict.get(t_key)
            if isinstance(t_val, list):
                gold_tables = [str(x).strip() for x in t_val if str(x).strip()]
            elif isinstance(t_val, str) and t_val.strip():
                gold_tables = [x.strip() for x in t_val.split(",") if x.strip()]

        examples.append(QueryExample(question=question, sql=sql, gold_tables=gold_tables, raw=row_dict))
        if limit and len(examples) >= limit:
            break

    return examples
