"""Simulator of Karpathy LLM-Wiki retrieval semantics.

The wiki is a set of markdown pages plus a flat ``index.md``. Retrieval = scan
the index, pick pages by name overlap, dump them into context. There is no
embedding similarity, no graph, no temporal awareness, no contradiction model.

To make the comparison fair we score on token-overlap-with-page-name. This
mirrors the way Claude would scan an index in practice (per the gist).
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]+", s) if len(t) >= 2}


@dataclass
class WikiHit:
    page: str
    score: float


class KarpathyWikiSimulator:
    def __init__(self) -> None:
        # page name -> list of sentences (the page body).
        self._pages: dict[str, list[str]] = defaultdict(list)
        self._index: list[str] = []  # all page names, in insertion order

    def ingest(self, content: str, **_: object) -> None:
        # Heuristic: the page name is the first noun phrase / capitalized word
        # in the content. Fall back to a hash if nothing capitalized.
        match = re.search(r"\b([A-Z][A-Za-z0-9]{1,})\b", content)
        page = match.group(1) if match else f"page_{len(self._pages):05d}"
        if page not in self._pages:
            self._index.append(page)
        self._pages[page].append(content)

    def recall(self, query: str, *, limit: int = 10, **_: object) -> tuple[list[WikiHit], float]:
        start = time.perf_counter()
        qtoks = _tokens(query)
        results: list[WikiHit] = []
        for page in self._index:  # linear scan over the index — Karpathy's known ceiling
            page_toks = _tokens(page) | _tokens(" ".join(self._pages[page]))
            inter = len(qtoks & page_toks)
            if inter:
                score = inter / max(1, len(qtoks))
                results.append(WikiHit(page=page, score=score))
        results.sort(key=lambda h: h.score, reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return results[:limit], elapsed_ms

    def names_in_top_k(self, query: str, k: int = 10) -> set[str]:
        hits, _ = self.recall(query, limit=k)
        return {h.page.lower() for h in hits}
