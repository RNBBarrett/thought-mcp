"""Setup helpers for the local-LLM embedders.

Pings the daemon, lists models, optionally writes a ``thought.toml`` snippet.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SetupResult:
    reachable: bool
    models: list[str]
    suggested_model: str | None = None
    error: str = ""


# A small list of known embedding-capable Ollama models for the suggestion.
KNOWN_OLLAMA_EMBED_MODELS: frozenset[str] = frozenset({
    "nomic-embed-text", "mxbai-embed-large", "all-minilm",
    "snowflake-arctic-embed", "bge-large", "bge-m3",
})


def ping_ollama(host: str = "http://localhost:11434", *, client=None) -> SetupResult:
    """Check if Ollama is reachable and list installed models."""
    try:
        import httpx
    except ImportError:
        return SetupResult(
            reachable=False, models=[],
            error="httpx not installed — pip install 'thought-mcp[llm-ollama]'",
        )
    c = client or httpx.Client(timeout=5.0)
    try:
        r = c.get(f"{host.rstrip('/')}/api/tags")
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return SetupResult(
            reachable=False, models=[],
            error=f"Ollama daemon unreachable at {host} ({e}). "
                  f"Start it with `ollama serve`.",
        )
    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    embed_models = [m for m in models if any(km in m for km in KNOWN_OLLAMA_EMBED_MODELS)]
    suggested = embed_models[0] if embed_models else None
    return SetupResult(reachable=True, models=models, suggested_model=suggested)


def ping_lmstudio(base_url: str = "http://localhost:1234/v1", *, client=None) -> SetupResult:
    """Check if LM Studio is reachable and list loaded models."""
    try:
        import httpx
    except ImportError:
        return SetupResult(
            reachable=False, models=[],
            error="httpx not installed — pip install 'thought-mcp[llm-ollama]'",
        )
    c = client or httpx.Client(timeout=5.0)
    try:
        r = c.get(f"{base_url.rstrip('/')}/models")
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return SetupResult(
            reachable=False, models=[],
            error=f"LM Studio unreachable at {base_url} ({e}). "
                  f"Launch LM Studio and start its local server.",
        )
    # OpenAI-style: {"data": [{"id": "model-name"}, ...]}
    items = data.get("data", [])
    models = [m.get("id", "") for m in items if m.get("id")]
    return SetupResult(reachable=True, models=models)


TOML_OLLAMA_SNIPPET = """\
[embedding]
choice = "ollama"
dim = 768
ollama_host = "{host}"
ollama_model = "{model}"

[llm]
enabled = true
provider = "ollama"
model = "mistral"
base_url = "{host}"
"""

TOML_LMSTUDIO_SNIPPET = """\
[embedding]
choice = "lmstudio"
dim = 768
lmstudio_url = "{base_url}"
lmstudio_model = "{model}"

[llm]
enabled = true
provider = "lmstudio"
model = "openai/gpt-oss-20b"
base_url = "{base_url}"
"""
