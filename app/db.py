from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

import sqlglot

from .beaver_data import TableSchema
from .settings import settings

logger = logging.getLogger(__name__)


_CREATE_RE = re.compile(r"create\s+table\s+.+?\(", re.IGNORECASE | re.DOTALL)


def _extract_column_names_from_ddl(ddl: str) -> list[str]:
    ddl = ddl.strip().rstrip(";")
    if not ddl:
        return []
    try:
        expr = sqlglot.parse_one(ddl, read="ansi")
    except Exception:
        expr = None

    if expr is not None and expr.args.get("this") is not None:
        cols = expr.args.get("expressions") or []
        names: list[str] = []
        for c in cols:
            try:
                name = c.this.name
            except Exception:
                name = None
            if name:
                names.append(str(name))
        return names

    body_start = ddl.find("(")
    body_end = ddl.rfind(")")
    if body_start == -1 or body_end == -1 or body_end <= body_start:
        return []
    body = ddl[body_start + 1 : body_end]
    names: list[str] = []
    for chunk in body.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        first = chunk.split()[0].strip("`\"[]")
        if first.lower() in {"primary", "foreign", "unique", "constraint"}:
            continue
        names.append(first)
    return names


def _extract_columns(table: TableSchema) -> list[str]:
    raw = table.raw

    for k in ["column_names", "columns"]:
        v = raw.get(k)
        if isinstance(v, list) and v:
            if all(isinstance(x, str) for x in v):
                return [str(x).strip() for x in v if str(x).strip()]
            if all(isinstance(x, dict) for x in v):
                out = []
                for item in v:
                    for nk in ["name", "column_name", "column", "field"]:
                        if nk in item and item[nk]:
                            out.append(str(item[nk]).strip())
                            break
                if out:
                    return out

    for k in ["create_table_sql", "ddl", "schema", "table_schema"]:
        v = raw.get(k)
        if isinstance(v, str) and _CREATE_RE.search(v):
            cols = _extract_column_names_from_ddl(v)
            if cols:
                return cols

    if _CREATE_RE.search(table.schema_text):
        cols = _extract_column_names_from_ddl(table.schema_text)
        if cols:
            return cols

    return []


def ensure_sqlite_schema(sqlite_path: Path, tables: list[TableSchema]) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(sqlite_path))
    try:
        con.execute("PRAGMA foreign_keys = OFF;")

        for t in tables:
            cols = _extract_columns(t)
            if not cols:
                cols = ["__dummy__"]
            col_defs = ", ".join([f'\"{c}\" TEXT' for c in cols])
            sql = f'CREATE TABLE IF NOT EXISTS \"{t.table_name}\" ({col_defs});'
            try:
                con.execute(sql)
            except Exception:
                safe_name = t.table_name.replace("\"", "")
                sql = f'CREATE TABLE IF NOT EXISTS \"{safe_name}\" ({col_defs});'
                con.execute(sql)

        con.commit()
    finally:
        con.close()


def execute_sql(sqlite_path: Path, sql: str, limit: int = 200) -> list[tuple]:
    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute("PRAGMA query_only = ON;")
        cur.execute(f"EXPLAIN QUERY PLAN {sql}")
        cur.execute(sql)
        rows = cur.fetchmany(limit)
        return rows
    finally:
        con.close()


def init_default_db(tables: list[TableSchema]) -> None:
    settings.ensure_cache_dirs()
    ensure_sqlite_schema(settings.sqlite_path, tables)
