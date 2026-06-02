from __future__ import annotations

import logging

import sqlglot

from .db import execute_sql
from .settings import settings

logger = logging.getLogger(__name__)


def validate_sql_syntax(sql: str) -> tuple[bool, str | None]:
    try:
        sqlglot.parse_one(sql, read="ansi")
        return True, None
    except Exception as e:
        return False, str(e)


def is_readonly_query(sql: str) -> bool:
    s = sql.strip().lower()
    return s.startswith("select") or s.startswith("with")


def validate_and_execute(sql: str) -> tuple[bool, str | None, list[tuple] | None]:
    is_valid, err = validate_sql_syntax(sql)
    if not is_valid:
        return False, err, None
    if not is_readonly_query(sql):
        return False, "Only SELECT/WITH queries are allowed", None

    try:
        rows = execute_sql(settings.sqlite_path, sql)
        return True, None, rows
    except Exception as e:
        return True, str(e), None
