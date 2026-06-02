from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException

from .beaver_data import QueryExample, TableSchema, load_queries, load_tables
from .benchmark import run_benchmark
from .db import init_default_db
from .llm import build_llm_client, build_prompt
from .logging_config import setup_logging
from .retrieval import TableRetriever
from .schemas import (
    BenchmarkBreakdown,
    BenchmarkErrorAnalysis,
    BenchmarkMetrics,
    BenchmarkResponse,
    GenerateSqlRequest,
    GenerateSqlResponse,
    RetrieveRequest,
    RetrieveResponse,
    TableDetail,
)
from .settings import settings
from .validation import validate_and_execute, validate_sql_syntax

logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise Text-to-SQL API", version="0.1.0")


def _require_ready() -> None:
    if getattr(app.state, "init_error", None):
        raise HTTPException(status_code=500, detail=str(app.state.init_error))
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Service not ready")


def _get_llm():
    llm = getattr(app.state, "llm", None)
    if llm is not None:
        return llm
    try:
        llm = build_llm_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    app.state.llm = llm
    return llm


@app.on_event("startup")
async def _startup() -> None:
    setup_logging()
    settings.ensure_cache_dirs()

    try:
        tables = load_tables()
        table_map = {t.table_name: t for t in tables}
        init_default_db(tables)

        retriever = TableRetriever(tables)
        few_shot: list[QueryExample] = load_queries(limit=5)

        app.state.tables = tables
        app.state.table_map = table_map
        app.state.retriever = retriever
        app.state.llm = None
        app.state.examples = few_shot
        app.state.ready = True
        app.state.init_error = None
    except Exception as e:
        app.state.ready = False
        app.state.init_error = e
        logger.exception("startup_failed", extra={"error": str(e)})


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    _require_ready()

    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question must be non-empty")
    if len(q) > settings.max_question_length:
        raise HTTPException(status_code=400, detail="question too long")

    retrieved = app.state.retriever.retrieve(q, top_k=settings.retrieval_top_k)
    if not retrieved:
        return RetrieveResponse(retrieved_tables=[], scores=[], confidence=0.0, details={})

    scores = [r.score for r in retrieved]
    retrieved_tables = [r.table_name for r in retrieved]
    confidence = float(sum(scores) / max(len(scores), 1))

    details = {
        r.table_name: TableDetail(relevance_score=r.score, reason=r.reason) for r in retrieved
    }

    return RetrieveResponse(
        retrieved_tables=retrieved_tables,
        scores=scores,
        confidence=confidence,
        details=details,
    )


@app.post("/generate-sql", response_model=GenerateSqlResponse)
async def generate_sql(req: GenerateSqlRequest) -> GenerateSqlResponse:
    _require_ready()

    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question must be non-empty")
    if len(q) > settings.max_question_length:
        raise HTTPException(status_code=400, detail="question too long")

    retrieved_tables: list[str] = []
    if req.use_retrieved_context:
        retrieved = app.state.retriever.retrieve(q, top_k=settings.retrieval_top_k)
        retrieved_tables = [r.table_name for r in retrieved]

    prompt = build_prompt(q, retrieved_tables, app.state.table_map, app.state.examples)

    start = time.perf_counter()
    try:
        llm_res = await _get_llm().generate(prompt)
    except Exception as e:
        logger.exception("llm_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e)) from e
    latency_ms = (time.perf_counter() - start) * 1000.0

    logger.info(
        "llm_call",
        extra={
            "latency_ms": latency_ms,
            "prompt": prompt,
            "llm_response": llm_res.raw_text,
        },
    )

    sql = llm_res.sql
    is_valid_syntax, parse_err = validate_sql_syntax(sql)

    exec_err = None
    if is_valid_syntax:
        _, exec_err, _ = validate_and_execute(sql)

    parsing_errors = parse_err or exec_err
    confidence = 0.85 if is_valid_syntax else 0.4

    return GenerateSqlResponse(
        sql=sql,
        retrieved_tables=retrieved_tables,
        is_valid_syntax=is_valid_syntax,
        parsing_errors=parsing_errors,
        confidence=confidence,
        prompt_used=prompt,
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark() -> BenchmarkResponse:
    _require_ready()

    result, samples = await run_benchmark(
        retriever=app.state.retriever,
        llm=_get_llm(),
        table_schemas=app.state.table_map,
        total=25,
    )

    metrics = BenchmarkMetrics(**result["metrics"])
    breakdown = BenchmarkBreakdown(**result["breakdown"])
    error_analysis = BenchmarkErrorAnalysis(**result["error_analysis"])

    sample_payload = [
        {
            "question": s.question,
            "gold_sql": s.gold_sql,
            "predicted_sql": s.predicted_sql,
            "retrieved_tables": s.retrieved_tables,
            "gold_tables": s.gold_tables,
            "is_parse_ok": s.is_parse_ok,
            "is_exec_ok": s.is_exec_ok,
            "exact_match": s.exact_match,
        }
        for s in samples[:8]
    ]

    return BenchmarkResponse(
        total_queries=result["total"],
        metrics=metrics,
        subtask_breakdown=breakdown,
        error_analysis=error_analysis,
        samples=sample_payload,
    )
