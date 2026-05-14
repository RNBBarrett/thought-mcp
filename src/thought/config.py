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
    choice: Literal["auto", "deterministic", "minilm", "bge-m3", "openai"] = "auto"
    dim: int = 384
    model_name: str | None = None


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    provider: Literal["anthropic", "openai", "ollama", "none"] = "none"
    model: str | None = None
    base_url: str | None = None


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
    return Settings.model_validate(data)
