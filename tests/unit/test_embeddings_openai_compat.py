"""Tests for the OpenAI-compatible embedder (LM Studio, vLLM, llama.cpp, OpenAI)."""
from __future__ import annotations

import httpx
import numpy as np
import pytest

from thought.embeddings.openai_compat import (
    LMStudioEmbedder,
    OpenAICompatibleEmbedder,
    OpenAIEmbedder,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compat_embed_returns_normalised() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [3.0, 4.0]}]})
    e = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model="m", dim=2, client=_client(handler),
    )
    v = e.embed("hello")
    assert pytest.approx(float(np.linalg.norm(v)), abs=1e-5) == 1.0


def test_openai_compat_batch() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}],
        })
    e = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model="m", dim=2, client=_client(handler),
    )
    arr = e.embed_many(["a", "b"])
    assert arr.shape == (2, 2)


def test_openai_compat_dim_mismatch() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0, 3.0]}]})
    e = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model="m", dim=4, client=_client(handler),
    )
    with pytest.raises(ValueError, match="3-d embeddings"):
        e.embed("hello")


def test_openai_compat_response_count_mismatch() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # Two inputs requested, one returned.
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]}]})
    e = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model="m", dim=2, client=_client(handler),
    )
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 inputs"):
        e.embed_many(["a", "b"])


def test_openai_compat_connection_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("server down")
    e = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model="m", dim=2, client=_client(handler),
    )
    with pytest.raises(RuntimeError, match="OpenAI-compat embed call failed"):
        e.embed("hello")


def test_api_key_added_to_authorization_header() -> None:
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]}]})

    e = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model="m", dim=2, api_key="sk-test",
        client=httpx.Client(
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer sk-test"},
        ),
    )
    e.embed("hello")
    assert captured["auth"] == "Bearer sk-test"


# ---------------------------------------------------------------- LMStudioEmbedder

def test_lmstudio_defaults_to_localhost_1234() -> None:
    e = LMStudioEmbedder(dim=4, client=_client(lambda r: httpx.Response(200)))
    assert e._base_url.endswith(":1234/v1")
    assert e.model_name.startswith("lmstudio:")
    assert e.dim == 4


def test_lmstudio_embed() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert "1234" in str(req.url)
        return httpx.Response(200, json={"data": [{"embedding": [0.6, 0.8]}]})
    e = LMStudioEmbedder(dim=2, client=_client(handler))
    v = e.embed("hi")
    assert v.shape == (2,)


# ---------------------------------------------------------------- OpenAIEmbedder

def test_openai_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(dim=4)


def test_openai_picks_up_env_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    e = OpenAIEmbedder(dim=4, client=_client(lambda r: httpx.Response(200)))
    assert e.model_name.startswith("openai:")
