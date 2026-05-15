"""Tests for the Ollama embedder (mocked via httpx.MockTransport)."""
from __future__ import annotations

import httpx
import numpy as np
import pytest

from thought.embeddings.ollama import OllamaEmbedder


def _make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_embed_returns_normalised_vector() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]]})
    e = OllamaEmbedder(host="http://x", model="m", dim=2, client=_make_client(handler))
    v = e.embed("hello")
    assert v.shape == (2,)
    # L2-normalised → unit vector.
    assert pytest.approx(np.linalg.norm(v), abs=1e-5) == 1.0


def test_embed_many_batch() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # Return two vectors regardless of body for this test.
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]})
    e = OllamaEmbedder(host="http://x", model="m", dim=2, client=_make_client(handler))
    arr = e.embed_many(["a", "b"])
    assert arr.shape == (2, 2)


def test_dim_mismatch_raises() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0, 3.0]]})  # 3-d
    e = OllamaEmbedder(host="http://x", model="m", dim=4, client=_make_client(handler))
    with pytest.raises(ValueError, match="3-d embeddings"):
        e.embed("hello")


def test_connection_error_message_is_actionable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("daemon down")
    e = OllamaEmbedder(host="http://x", model="m", dim=2, client=_make_client(handler))
    with pytest.raises(RuntimeError, match="ollama serve"):
        e.embed("hello")


def test_falls_back_to_legacy_endpoint_when_batch_unavailable() -> None:
    """Older Ollama returns ``embedding`` (singular) — embedder should fall back."""
    call_log: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url.path)
        call_log.append(url)
        if url.endswith("/api/embed"):
            # Newer endpoint absent → return malformed (no embeddings key) so
            # the embedder triggers the legacy fallback.
            return httpx.Response(200, json={"unexpected": "shape"})
        if url.endswith("/api/embeddings"):
            return httpx.Response(200, json={"embedding": [0.6, 0.8]})
        return httpx.Response(404)

    e = OllamaEmbedder(host="http://x", model="m", dim=2, client=_make_client(handler))
    v = e.embed("hello")
    assert v.shape == (2,)
    assert any("api/embeddings" in u for u in call_log), call_log


def test_empty_input_returns_empty_array() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")
    e = OllamaEmbedder(host="http://x", model="m", dim=4, client=_make_client(handler))
    arr = e.embed_many([])
    assert arr.shape == (0, 4)


def test_model_name_and_version() -> None:
    e = OllamaEmbedder(host="http://x", model="nomic-embed-text", dim=768,
                       client=_make_client(lambda r: httpx.Response(200)))
    assert e.model_name == "ollama:nomic-embed-text"
    assert e.model_version  # non-empty
    assert e.dim == 768
