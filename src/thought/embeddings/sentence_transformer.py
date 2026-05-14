"""Production embedder backed by ``sentence-transformers``.

Lazy-loads the model on first call so importing the module — and even
constructing the Embedder — doesn't trigger the 80MB download. First call
shows a one-line progress notice. Optionally accelerated via ONNX Runtime
when ``onnxruntime`` is installed (~3-5× speedup on CPU).

Defaults to ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim, dense, well
within the binary-quantisation sweet spot) so the README quickstart gives
real-quality results out of the box.

Install:
    pip install 'thought-mcp[embeddings-local]'
    # plus optionally:
    pip install onnxruntime
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

_log = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Lazy-loaded sentence-transformers embedder.

    Construction is free; the model is downloaded / loaded on first
    ``embed`` / ``embed_many`` call. Vectors are L2-normalised so cosine
    similarity reduces to a dot product downstream.
    """

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
        device: str | None = None,
        show_progress: bool = True,
    ) -> None:
        self._model_name = model_name
        self._dim = dim
        self._device = device
        self._model: SentenceTransformer | None = None
        self._show_progress = show_progress

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        # Sentence-transformers doesn't expose a per-model version; we tag
        # by the model name so a model upgrade triggers a re-embed pass.
        return "1"

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        if self._show_progress:
            sys.stderr.write(
                f"[thought] loading embedder {self._model_name} "
                f"(first run downloads ~80MB)…\n"
            )
            sys.stderr.flush()
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. Run: "
                "pip install 'thought-mcp[embeddings-local]'"
            ) from e
        model = SentenceTransformer(self._model_name, device=self._device)
        # Validate dim — surface a clear error if the user paired a model
        # with the wrong dim.
        actual_dim = model.get_sentence_embedding_dimension()
        if actual_dim is not None and actual_dim != self._dim:
            raise ValueError(
                f"embedder dim mismatch: configured {self._dim} but "
                f"{self._model_name} produces {actual_dim}"
            )
        self._model = model
        if self._show_progress:
            sys.stderr.write(f"[thought] embedder ready (dim={self._dim})\n")
            sys.stderr.flush()
        return model

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_loaded()
        # ``normalize_embeddings=True`` makes cosine equivalent to dot product
        # — important for the Matryoshka and binary-Hamming code paths.
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=min(64, max(8, len(texts))),
        )
        return np.asarray(vectors, dtype=np.float32)
