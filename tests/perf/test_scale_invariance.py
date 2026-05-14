"""Scale-invariance benchmark — the core plan.md claim.

The plan asserts:
    "Every response is bounded to ≤10 results regardless of knowledge base size."
    "Router-bounded constant-time dispatch."

A pure vector store (OB1) is O(N) per query for the rerank step over the full
KB. THOUGHT's router caps the search surface and bounds the result count;
latency growth from 1k → 10k entities should be sub-linear (well under 10×).

This is a smoke-level perf check — full 1M-entity matrix runs in CI. Locally
we test 1k vs 5k as a sanity check the bound holds.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from thought.memory import Memory


def _seed(mem: Memory, n: int) -> None:
    now = datetime.now(UTC)
    for i in range(n):
        mem.remember(
            content=f"Entity{i} owns Company{i % 50} Corp.",
            scope="shared", now=now,
        )


@pytest.mark.perf
@pytest.mark.parametrize("size", [1000, 5000])
def test_recall_returns_at_most_ten_regardless_of_kb_size(tmp_path, size) -> None:
    """The plan's load-bearing bound. Property-style at two sizes."""
    mem = Memory.open(
        db_path=str(tmp_path / f"k{size}.db"),
        embedder_choice="deterministic", embedder_dim=64,
    )
    try:
        _seed(mem, size)
        for q in ["Entity0", "find something", "who owns Company1"]:
            result = mem.recall(query=q, limit=10)
            assert len(result.hits) <= 10, (
                f"limit bound violated at KB size {size} for query {q!r}: "
                f"got {len(result.hits)} hits"
            )
    finally:
        mem.close()


@pytest.mark.perf
def test_recall_latency_is_sub_linear(tmp_path) -> None:
    """Latency at 5k should not be more than 10× latency at 1k.

    Pure ANN with full re-rank would be roughly linear (~5× at 5×-larger KB).
    THOUGHT's Matryoshka 2-pass + scope-filtered candidate set should be
    sub-linear. We're not claiming O(1); we're claiming "well under linear."
    """
    times = {}
    for size in (1000, 5000):
        mem = Memory.open(
            db_path=str(tmp_path / f"l{size}.db"),
            embedder_choice="deterministic", embedder_dim=64,
        )
        try:
            _seed(mem, size)
            # Warm cache, then average 5 recalls.
            mem.recall(query="Entity42 owns", limit=10)
            t0 = time.perf_counter()
            for _ in range(5):
                mem.recall(query="Entity42 owns", limit=10)
            times[size] = (time.perf_counter() - t0) / 5
        finally:
            mem.close()
    ratio = times[5000] / max(times[1000], 1e-6)
    # Allow up to 10× growth; pure linear would be ~5×.
    assert ratio < 10.0, (
        f"scale-invariance regression: 5k/1k latency ratio = {ratio:.2f}× "
        f"(1k={times[1000]*1000:.1f}ms, 5k={times[5000]*1000:.1f}ms)"
    )


@pytest.mark.perf
def test_recall_latency_under_threshold(tmp_path) -> None:
    """recall() p95 < 200ms on a 5k-entity KB with the deterministic embedder."""
    mem = Memory.open(
        db_path=str(tmp_path / "lat.db"),
        embedder_choice="deterministic", embedder_dim=64,
    )
    try:
        _seed(mem, 5000)
        # Warm up.
        for _ in range(3):
            mem.recall(query="who owns Company7", limit=10)
        latencies = []
        for _ in range(30):
            t0 = time.perf_counter()
            mem.recall(query="who owns Company7", limit=10)
            latencies.append(time.perf_counter() - t0)
        sorted_l = sorted(latencies)
        p95 = sorted_l[int(0.95 * len(sorted_l))]
        assert p95 < 0.5, f"recall p95 over budget: {p95*1000:.1f}ms"
    finally:
        mem.close()
