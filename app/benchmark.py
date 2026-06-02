from __future__ import annotations

import time
from dataclasses import dataclass

import sqlglot

from .beaver_data import QueryExample, TableSchema, load_queries
from .llm import LlmClient, build_prompt
from .retrieval import TableRetriever
from .validation import validate_and_execute, validate_sql_syntax


def _normalize_sql(sql: str) -> str:
    try:
        expr = sqlglot.parse_one(sql, read="ansi")
        return expr.sql(dialect="sqlite", pretty=False).strip().rstrip(";")
    except Exception:
        return sql.strip().rstrip(";")


@dataclass(frozen=True)
class BenchmarkSample:
    question: str
    gold_sql: str
    predicted_sql: str
    retrieved_tables: list[str]
    gold_tables: list[str]
    is_parse_ok: bool
    is_exec_ok: bool
    exact_match: bool


async def run_benchmark(
    retriever: TableRetriever,
    llm: LlmClient,
    table_schemas: dict[str, TableSchema],
    total: int = 25,
) -> tuple[dict, list[BenchmarkSample]]:
    examples = load_queries(limit=total)
    if not examples:
        raise RuntimeError("No benchmark examples available from beaverbench/beaver-query.")

    latencies_ms: list[float] = []
    recall5_hits = 0
    recall10_hits = 0
    exact_hits = 0
    parse_ok = 0
    exec_match_hits = 0

    retrieval_failures = 0
    parsing_failures = 0
    execution_failures = 0
    logic_errors = 0

    samples: list[BenchmarkSample] = []

    few_shot = examples[:5]

    for ex in examples:
        start = time.perf_counter()

        retrieved = retriever.retrieve(ex.question, top_k=10)
        retrieved_tables = [r.table_name for r in retrieved]

        gold_tables = ex.gold_tables
        if gold_tables:
            gt = set(gold_tables)
            if gt & set(retrieved_tables[:5]):
                recall5_hits += 1
            else:
                retrieval_failures += 1
            if gt & set(retrieved_tables[:10]):
                recall10_hits += 1

        prompt = build_prompt(ex.question, retrieved_tables, table_schemas, few_shot)
        llm_res = await llm.generate(prompt)
        predicted_sql = llm_res.sql

        is_parse_ok, parse_err = validate_sql_syntax(predicted_sql)
        if is_parse_ok:
            parse_ok += 1
        else:
            parsing_failures += 1

        exact_match = _normalize_sql(predicted_sql) == _normalize_sql(ex.sql)
        if exact_match:
            exact_hits += 1

        exec_ok = False
        try:
            pred_ok, pred_exec_err, pred_rows = validate_and_execute(predicted_sql)
            gold_ok, gold_exec_err, gold_rows = validate_and_execute(ex.sql)
            exec_ok = pred_ok and gold_ok and pred_exec_err is None and gold_exec_err is None
            if exec_ok and (pred_rows == gold_rows):
                exec_match_hits += 1
            elif not exec_ok:
                execution_failures += 1
            else:
                logic_errors += 1
        except Exception:
            execution_failures += 1

        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency_ms)

        samples.append(
            BenchmarkSample(
                question=ex.question,
                gold_sql=ex.sql,
                predicted_sql=predicted_sql,
                retrieved_tables=retrieved_tables,
                gold_tables=gold_tables,
                is_parse_ok=is_parse_ok,
                is_exec_ok=exec_ok,
                exact_match=exact_match,
            )
        )

    n = len(examples)
    metrics = {
        "retrieval_recall_at_5": (recall5_hits / n) if n else 0.0,
        "retrieval_recall_at_10": (recall10_hits / n) if n else 0.0,
        "sql_exact_match_accuracy": (exact_hits / n) if n else 0.0,
        "sql_execution_match_accuracy": (exec_match_hits / n) if n else 0.0,
        "parsing_success_rate": (parse_ok / n) if n else 0.0,
        "average_latency_ms": (sum(latencies_ms) / n) if n else 0.0,
    }

    breakdown = {
        "multi_table_retrieval": metrics["retrieval_recall_at_10"],
        "column_mapping": metrics["sql_exact_match_accuracy"],
        "join_detection": metrics["sql_exact_match_accuracy"],
        "domain_knowledge": metrics["sql_exact_match_accuracy"],
    }

    error_analysis = {
        "retrieval_failures": retrieval_failures,
        "parsing_failures": parsing_failures,
        "execution_failures": execution_failures,
        "logic_errors": logic_errors,
    }

    return {"metrics": metrics, "breakdown": breakdown, "error_analysis": error_analysis, "total": n}, samples
