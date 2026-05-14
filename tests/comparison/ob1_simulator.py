"""Simulator of OB1 / OpenBrain semantics.

Captures what OB1 actually does — and equally importantly, what it cannot do.
Pure-vector, single table, no relationships, no temporal, no provenance edges,
no scope.

A FACT query like "who owns Acme" hits the vector store with no understanding
that it should expand through edges; a CHANGE query gets the same fuzzy match
with no awareness of validity windows.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from thought.embeddings.base import bytes_to_vector, vector_to_bytes
from thought.embeddings.deterministic import DeterministicEmbedder


@dataclass
class OB1Hit:
    text: str
    score: float


class OB1Simulator:
    """Single ``thoughts`` table + pgvector-style similarity."""

    def __init__(self, *, dim: int = 128) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE thoughts (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " text TEXT NOT NULL, embedding BLOB NOT NULL,"
            " created_at TEXT NOT NULL)"
        )
        self._embedder = DeterministicEmbedder(dim=dim)

    def ingest(self, content: str, *, now: datetime, **_: object) -> None:
        v = self._embedder.embed(content)
        self._conn.execute(
            "INSERT INTO thoughts (text, embedding, created_at) VALUES (?, ?, ?)",
            (content, vector_to_bytes(v), now.isoformat()),
        )

    def recall(self, query: str, *, limit: int = 10, **_: object) -> tuple[list[OB1Hit], float]:
        start = time.perf_counter()
        q = self._embedder.embed(query)
        results: list[OB1Hit] = []
        for row in self._conn.execute("SELECT text, embedding FROM thoughts"):
            v = bytes_to_vector(row["embedding"], self._embedder.dim)
            score = float(np.dot(q, v))
            results.append(OB1Hit(text=row["text"], score=score))
        results.sort(key=lambda h: h.score, reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return results[:limit], elapsed_ms

    def names_in_top_k(self, query: str, k: int = 10) -> set[str]:
        hits, _ = self.recall(query, limit=k)
        names: set[str] = set()
        for h in hits:
            for tok in re.findall(r"[A-Za-z]+", h.text):
                if tok[:1].isupper() and len(tok) >= 2 and tok.lower() not in {"the", "a", "is", "in", "at"}:
                    names.add(tok.lower())
        return names
