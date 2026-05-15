"""OpenAI-compatible embedder.

Works against any server that speaks the OpenAI ``/v1/embeddings`` shape:

- **LM Studio** at ``http://localhost:1234/v1``
- **vLLM** at ``http://localhost:8000/v1``
- **llama.cpp --api** at ``http://localhost:8080/v1``
- **OpenAI proper** at ``https://api.openai.com/v1`` (with an api key)

``LMStudioEmbedder`` is a convenience subclass with LM Studio's default
defaults; ``OpenAIEmbedder`` is the same with OpenAI's defaults + the key.

Requires ``httpx`` (in the ``[llm-ollama]`` extra; you don't need an actual
Ollama daemon to use this class).
"""
from __future__ import annotations

import numpy as np


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dim: int,
        api_key: str | None = None,
        timeout: float = 30.0,
        client=None,
    ) -> None:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover — gated by extra
            raise ImportError(
                "OpenAICompatibleEmbedder requires httpx — install with "
                "`pip install thought-mcp[llm-ollama]`."
            ) from e
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._api_key = api_key
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.Client(timeout=timeout, headers=headers)
        self._dim_verified = False

    @property
    def model_name(self) -> str:
        return f"openai-compat:{self._model}"

    @property
    def model_version(self) -> str:
        return "openai-compat-1"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        try:
            r = self._client.post(
                f"{self._base_url}/embeddings",
                json={"model": self._model, "input": texts},
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise RuntimeError(
                f"OpenAI-compat embed call failed at {self._base_url}/embeddings: {e}. "
                f"Is the server running? Check the URL and that the model {self._model!r} "
                f"is loaded."
            ) from e
        # OpenAI shape: {"data": [{"embedding": [...]}, ...]}
        items = data.get("data") or []
        if len(items) != len(texts):
            raise RuntimeError(
                f"Embeddings server returned {len(items)} vectors for {len(texts)} "
                f"inputs: {data!r}"
            )
        arr = np.asarray(
            [item["embedding"] for item in items], dtype=np.float32,
        )
        if arr.shape[1] != self._dim:
            raise ValueError(
                f"Model {self._model!r} returned {arr.shape[1]}-d embeddings "
                f"but [embedding] dim={self._dim} in your config."
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).astype(np.float32, copy=False)


class LMStudioEmbedder(OpenAICompatibleEmbedder):
    """LM Studio defaults to ``http://localhost:1234/v1`` and accepts any key."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:1234/v1",
        model: str = "nomic-embed-text-v1.5",
        dim: int = 768,
        timeout: float = 30.0,
        client=None,
    ) -> None:
        super().__init__(
            base_url=base_url, model=model, dim=dim,
            api_key=None, timeout=timeout, client=client,
        )

    @property
    def model_name(self) -> str:
        return f"lmstudio:{self._model}"

    @property
    def model_version(self) -> str:
        return "lmstudio-1"


class OpenAIEmbedder(OpenAICompatibleEmbedder):
    """OpenAI proper. Requires ``api_key`` (or set ``OPENAI_API_KEY``)."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        api_key: str | None = None,
        timeout: float = 30.0,
        client=None,
    ) -> None:
        import os
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAIEmbedder requires an api_key (or set OPENAI_API_KEY)."
            )
        super().__init__(
            base_url=base_url, model=model, dim=dim,
            api_key=api_key, timeout=timeout, client=client,
        )

    @property
    def model_name(self) -> str:
        return f"openai:{self._model}"

    @property
    def model_version(self) -> str:
        return "openai-1"
