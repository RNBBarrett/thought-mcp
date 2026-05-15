"""Configuration loading.

Settings come from (in priority order):
1. Constructor arguments.
2. Environment variables prefixed ``THOUGHT_``.
3. ``thought.toml`` in the working directory.
4. Sensible defaults.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    choice: Literal[
        "auto", "deterministic", "minilm", "bge-m3",
        # v0.4 — local-LLM + remote OpenAI-compatible:
        "ollama", "lmstudio", "openai-compat", "openai",
    ] = "auto"
    dim: int = 384
    model_name: str | None = None
    # Ollama (native API)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    # LM Studio (OpenAI-compatible)
    lmstudio_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "nomic-embed-text-v1.5"
    # Generic OpenAI-compatible (vLLM, llama.cpp, OpenAI proper, …)
    openai_compat_url: str = "http://localhost:8000/v1"
    openai_compat_model: str = "text-embedding-3-small"
    openai_compat_api_key: str = ""  # blank for local servers


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    provider: Literal[
        "anthropic", "openai", "openai-compat", "ollama", "lmstudio", "none",
    ] = "none"
    model: str | None = None
    base_url: str | None = None
    api_key: str = ""


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = 8765


class ConsolidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    cycle_seconds: float = 60.0
    cold_demotion_days: int = 30
    staleness_days: int = 30
    batch_size: int = 100


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    db_path: str = ".thought/thought.db"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)


def find_config(start: Path | None = None, name: str = "thought.toml") -> Path | None:
    """Walk up from ``start`` (or cwd) looking for ``name`` — git-style.

    Returns the first matching path or ``None``. Stops at the filesystem root
    or at a directory containing a ``.git`` marker, whichever comes first.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        cfg = candidate / name
        if cfg.is_file():
            return cfg
        # Stop at repo boundaries — don't accidentally pick up a sibling
        # project's config three levels up.
        if (candidate / ".git").exists() and candidate != here:
            return None
    return None


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings, optionally walking up the directory tree.

    When ``path`` is explicitly provided we honour it as-is (and tolerate it
    being absent, returning defaults). When omitted we search upward from
    the current working directory for ``thought.toml``.
    """
    if path is None:
        p = find_config() or Path("thought.toml")
    else:
        p = Path(path)
    data: dict = {}
    if p.exists():
        with p.open("rb") as fh:
            data = tomllib.load(fh)
    # Env overrides — simple flat keys for the MVP.
    if env_db := os.environ.get("THOUGHT_DB_PATH"):
        data["db_path"] = env_db
    if env_emb := os.environ.get("THOUGHT_EMBEDDER"):
        data.setdefault("embedding", {})["choice"] = env_emb
    # v0.4: local-LLM provider env overrides
    emb_overrides = {
        "THOUGHT_OLLAMA_HOST": "ollama_host",
        "THOUGHT_OLLAMA_MODEL": "ollama_model",
        "THOUGHT_LMSTUDIO_URL": "lmstudio_url",
        "THOUGHT_LMSTUDIO_MODEL": "lmstudio_model",
        "THOUGHT_OPENAI_COMPAT_URL": "openai_compat_url",
        "THOUGHT_OPENAI_COMPAT_MODEL": "openai_compat_model",
        "THOUGHT_OPENAI_COMPAT_API_KEY": "openai_compat_api_key",
    }
    for env_key, cfg_key in emb_overrides.items():
        if v := os.environ.get(env_key):
            data.setdefault("embedding", {})[cfg_key] = v
    return Settings.model_validate(data)
