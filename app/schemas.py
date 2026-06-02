from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1)


class TableDetail(BaseModel):
    relevance_score: float
    reason: str


class RetrieveResponse(BaseModel):
    retrieved_tables: list[str]
    scores: list[float]
    confidence: float
    details: dict[str, TableDetail]


class GenerateSqlRequest(BaseModel):
    question: str = Field(min_length=1)
    use_retrieved_context: bool = True


class GenerateSqlResponse(BaseModel):
    sql: str
    retrieved_tables: list[str]
    is_valid_syntax: bool
    parsing_errors: str | None
    confidence: float
    prompt_used: str


class BenchmarkMetrics(BaseModel):
    retrieval_recall_at_5: float
    retrieval_recall_at_10: float
    sql_exact_match_accuracy: float
    sql_execution_match_accuracy: float
    parsing_success_rate: float
    average_latency_ms: float


class BenchmarkBreakdown(BaseModel):
    multi_table_retrieval: float
    column_mapping: float
    join_detection: float
    domain_knowledge: float


class BenchmarkErrorAnalysis(BaseModel):
    retrieval_failures: int
    parsing_failures: int
    execution_failures: int
    logic_errors: int


class BenchmarkResponse(BaseModel):
    total_queries: int
    metrics: BenchmarkMetrics
    subtask_breakdown: BenchmarkBreakdown
    error_analysis: BenchmarkErrorAnalysis
    samples: list[dict[str, Any]] = Field(default_factory=list)
