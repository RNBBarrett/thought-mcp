"""Ollama embedder.

POSTs to Ollama's native ``/api/embed`` (preferred, batched) or
``/api/embeddings`` (legacy, per-text). L2-normalises results on receive so
the rest of the vector layer sees the same shape as in-process embedders.

Requires the ``[llm-ollama]`` extra (``httpx``).

Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
"""
from __future__ import annotations

import numpy as np


class OllamaEmbedder:
    """Talks to a local Ollama daemon.

    Defaults match the canonical Ollama install: daemon on
    ``http://localhost:11434``, model ``nomic-embed-text`` (768-dim).
    Pass any other model the user has pulled — the dim is verified on the
    first ``embed`` call so misconfiguration fails fast with a clear message.
    """

    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dim: int = 768,
        timeout: float = 30.0,
        client=None,
    ) -> None:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover — gated by extra
            raise ImportError(
                "OllamaEmbedder requires httpx — install with "
                "`pip install thought-mcp[llm-ollama]`."
            ) from e
        self._host = host.rstrip("/")
        self._model = model
        self._dim = dim
        # Allow tests to inject a mocked transport-backed client.
        self._client = client or httpx.Client(timeout=timeout)
        self._dim_verified = False

    @property
    def model_name(self) -> str:
        return f"ollama:{self._model}"

    @property
    def model_version(self) -> str:
        # Ollama doesn't expose a version per model via the embed endpoint;
        # the model name itself acts as the version key for storage.
        return "ollama-1"

    @property
    def dim(self) -> int:
        return self._dim

    # ---------- public API

    def embed(self, text: str) -> np.ndarray:
        vec = self.embed_many([text])[0]
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        # Try the newer batched endpoint first; fall back to per-text loop.
        try:
            r = self._client.post(
                f"{self._host}/api/embed",
                json={"model": self._model, "input": texts},
            )
            r.raise_for_status()
            data = r.json()
            vectors = data.get("embeddings")
            if vectors is None:
                # Older Ollama: returned "embedding" (singular) for one input.
                raise _OllamaBatchUnavailableError
            arr = np.asarray(vectors, dtype=np.float32)
        except _OllamaBatchUnavailableError:
            arr = self._fallback_per_text(texts)
        except Exception as e:
            # httpx 4xx / 5xx / connection errors — make it actionable.
            raise RuntimeError(
                f"Ollama embed call failed at {self._host}/api/embed: {e}. "
                f"Is the daemon running? Try `ollama serve`."
            ) from e

        if arr.shape[1] != self._dim:
            raise ValueError(
                f"Ollama model {self._model!r} returned {arr.shape[1]}-d "
                f"embeddings, but [embedding] dim={self._dim} in your config. "
                f"Update your config or pick a model whose output matches."
            )
        if not self._dim_verified:
            self._dim_verified = True
        # L2-normalise (Ollama's outputs are usually not unit-norm).
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).astype(np.float32, copy=False)

    # ---------- legacy single-text fallback

    def _fallback_per_text(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            r = self._client.post(
                f"{self._host}/api/embeddings",
                json={"model": self._model, "prompt": t},
            )
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding")
            if vec is None:
                raise RuntimeError(
                    f"Ollama /api/embeddings returned no 'embedding' field "
                    f"for model {self._model!r}: {data!r}"
                )
            out[i] = np.asarray(vec, dtype=np.float32)
        return out


class _OllamaBatchUnavailableError(Exception):
    """Internal signal that /api/embed isn't available on this Ollama version."""
