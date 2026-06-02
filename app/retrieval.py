from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from collections import Counter
from math import log, sqrt

from .beaver_data import TableSchema
from .settings import settings

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


@dataclass(frozen=True)
class RetrievedTable:
    table_name: str
    score: float
    reason: str


class TableRetriever:
    def __init__(self, tables: list[TableSchema]):
        self._tables = tables
        self._table_tokens = {t.table_name: _tokenize(t.schema_text) for t in tables}
        self._df: Counter[str] = Counter()
        for tokens in self._table_tokens.values():
            self._df.update(tokens)
        self._n_docs = max(len(self._table_tokens), 1)

    @property
    def tables(self) -> list[TableSchema]:
        return self._tables

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedTable]:
        k = top_k or settings.retrieval_top_k

        q_tokens = _tokenize(question)
        if not q_tokens:
            return []

        def idf(tok: str) -> float:
            return log((1.0 + self._n_docs) / (1.0 + self._df.get(tok, 0))) + 1.0

        q_weights = {t: idf(t) for t in q_tokens}
        q_norm = sqrt(sum(w * w for w in q_weights.values())) or 1.0

        scored: list[tuple[float, TableSchema, list[str]]] = []
        for t in self._tables:
            tokens = self._table_tokens.get(t.table_name, set())
            overlap_set = q_tokens & tokens
            if not overlap_set:
                continue
            dot = sum(q_weights[o] * idf(o) for o in overlap_set)
            t_norm = sqrt(sum(idf(o) ** 2 for o in overlap_set)) or 1.0
            score = dot / (q_norm * t_norm)
            overlap = list(sorted(overlap_set))
            overlap = overlap[:6]
            if overlap:
                reason = f"Overlapping terms: {', '.join(overlap)}"
            else:
                reason = "High semantic similarity to table schema"
            scored.append((float(score), t, overlap))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[RetrievedTable] = []
        for score, t, overlap in scored[:k]:
            if overlap:
                reason = f"Overlapping terms: {', '.join(overlap)}"
            else:
                reason = "High semantic similarity to table schema"
            out.append(RetrievedTable(table_name=t.table_name, score=score, reason=reason))

        return out
