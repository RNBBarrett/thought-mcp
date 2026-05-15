"""Auto-write hook for Claude Code's ``Stop`` event.

Reads the hook payload (JSON) from stdin, extracts the last user turn + the
final assistant turn from the conversation transcript, ingests both as
sources so the contents become first-class memory entities. Idempotent via
the ingest pipeline's content-sha256 keyed dedup (replaying a transcript
won't double-ingest).

Two modes:
- ``raw`` (default): pipe each turn straight into ``Memory.remember``. The
  existing Jaccard dedup + the ingest pipeline's NER-style fact extractor
  absorb low-signal phrasing.
- ``extract``: route each turn through a small LLM (Anthropic Haiku by
  default) first, ingest only the distilled facts. Lower noise but costs
  ~$0.001/turn. Falls back to ``raw`` with a stderr warning if the
  Anthropic SDK or API key isn't available.

Spec: https://code.claude.com/docs/en/hooks
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from ..memory import Memory

WriteMode = Literal["raw", "extract"]

# Defaults the contradiction-detection engine activates for auto-write so
# user-preference style facts ("prefers X" / "lives in Y") supersede their
# earlier counterparts instead of accumulating side-by-side. Matches the
# verb vocabulary in ``thought.ingest.entities``.
DEFAULT_UNIQUE_PREDICATES: frozenset[str] = frozenset({
    "PREFERS", "WORKS_AT", "OWNS", "REPORTS_TO",
})

# Per-turn cap to keep one autonomous turn from monopolizing the KB and to
# bound the ingest cost. 8k chars ≈ 2k tokens — comfortably below the model
# limits even on long assistant responses.
MAX_TURN_CHARS = 8_000


def _read_transcript(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract message dicts from the hook payload.

    Claude Code's Stop hook payload includes a ``transcript_path`` pointing
    at a JSONL file with the session's messages. For testability we also
    accept an inline ``messages`` list.
    """
    if "messages" in payload and isinstance(payload["messages"], list):
        return [m for m in payload["messages"] if isinstance(m, dict)]
    tpath = payload.get("transcript_path")
    if not tpath:
        return []
    p = Path(tpath)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _flatten_content(message: dict[str, Any]) -> str:
    """Collapse a message's ``content`` (Claude's content-block format) to text."""
    raw = message.get("content")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _select_turns_for_ingest(
    messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Pick the (last_user, last_assistant) pair we want to ingest.

    Returns a list of ``(role, content)`` tuples. Skipping intermediate
    tool-call traffic is intentional — those are not memory-grade signal.
    """
    last_user: str | None = None
    last_assistant: str | None = None
    for m in messages:
        role = m.get("role")
        content = _flatten_content(m).strip()
        if not content:
            continue
        if role == "user":
            last_user = content
        elif role == "assistant":
            last_assistant = content
    out: list[tuple[str, str]] = []
    if last_user:
        out.append(("user", last_user[:MAX_TURN_CHARS]))
    if last_assistant:
        out.append(("assistant", last_assistant[:MAX_TURN_CHARS]))
    return out


def run(
    *,
    memory: Memory,
    payload: dict[str, Any],
    mode: WriteMode = "raw",
    scope: str = "private",
    owner_id: str | None = None,
    unique_predicates: frozenset[str] | None = None,
    llm_cfg: object | None = None,
) -> dict[str, Any]:
    """Execute one auto-write hook invocation.

    Pure-Python; the CLI wrapper handles stdin / stdout / lifecycle. Returns
    a summary dict ``{"ingested": N, "duplicates": M, "mode": ..., "skipped": ...}``.
    """
    if unique_predicates is None:
        unique_predicates = DEFAULT_UNIQUE_PREDICATES
    messages = _read_transcript(payload)
    pairs = _select_turns_for_ingest(messages)
    if not pairs:
        return {"ingested": 0, "duplicates": 0, "mode": mode, "skipped": "no turns to ingest"}

    items: list[dict[str, Any]] = []
    if mode == "extract":
        for _role, body in pairs:
            for fact in _extract_facts(body, llm_cfg=llm_cfg):
                items.append({
                    "content": fact,
                    "scope": scope, "owner_id": owner_id,
                    "unique_predicates": tuple(unique_predicates),
                })
        if not items:
            return {"ingested": 0, "duplicates": 0, "mode": "extract", "skipped": "no facts extracted"}
    else:
        for _role, body in pairs:
            items.append({
                "content": body,
                "scope": scope, "owner_id": owner_id,
                "unique_predicates": tuple(unique_predicates),
            })

    results = memory.remember_many(items)
    n_new = sum(1 for r in results if r.duplicate_of_source is None)
    n_dup = len(results) - n_new
    return {
        "ingested": n_new,
        "duplicates": n_dup,
        "mode": mode,
        "contradictions": sum(len(r.contradictions_detected) for r in results),
    }


EXTRACT_PROMPT = (
    "Extract durable, third-person factual statements from the following "
    "conversation turn. Output one fact per line, no numbering, no "
    "preamble. Skip ephemeral content (greetings, hedging, in-progress "
    "thoughts). If no durable facts are present, output nothing.\n\n"
    "---\n{text}\n---"
)


def _extract_facts(text: str, *, llm_cfg: object | None = None) -> list[str]:
    """Dispatch to the configured LLM provider for fact extraction.

    Provider precedence:
    1. ``llm_cfg.provider`` if supplied
    2. Anthropic (back-compat for hooks installed in v0.3 without llm config)

    Falls back to returning ``[text]`` (raw mode equivalent) with a clear
    stderr message when the configured provider is unavailable.
    """
    provider = getattr(llm_cfg, "provider", None) or "anthropic"
    if provider == "none":
        return [text]
    if provider == "anthropic":
        return _extract_via_anthropic(text, llm_cfg)
    if provider == "ollama":
        return _extract_via_ollama(text, llm_cfg)
    if provider in {"lmstudio", "openai-compat", "openai"}:
        return _extract_via_openai_compat(text, llm_cfg, provider=provider)
    print(
        f"thought hook write: unknown llm provider {provider!r}; falling back to raw.",
        file=sys.stderr,
    )
    return [text]


def _extract_via_anthropic(text: str, llm_cfg: object | None) -> list[str]:
    try:
        import anthropic
    except ImportError:
        print(
            "thought hook write: --mode extract anthropic requires "
            "'pip install thought-mcp[llm-anthropic]'; falling back to raw mode.",
            file=sys.stderr,
        )
        return [text]
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "thought hook write: --mode extract anthropic requires ANTHROPIC_API_KEY; "
            "falling back to raw mode.",
            file=sys.stderr,
        )
        return [text]
    model = getattr(llm_cfg, "model", None) or "claude-haiku-4-5-20251001"
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=text)}],
        )
        body = "".join(
            block.text  # type: ignore[union-attr]
            for block in resp.content
            if getattr(block, "type", None) == "text"
        )
    except Exception as e:
        print(
            f"thought hook write: anthropic extract failed ({e}); falling back to raw.",
            file=sys.stderr,
        )
        return [text]
    return [line.strip() for line in body.splitlines() if line.strip()]


def _extract_via_ollama(text: str, llm_cfg: object | None) -> list[str]:
    try:
        import httpx
    except ImportError:
        print(
            "thought hook write: ollama extract requires 'pip install thought-mcp[llm-ollama]'; "
            "falling back to raw mode.",
            file=sys.stderr,
        )
        return [text]
    host = (getattr(llm_cfg, "base_url", None) or "http://localhost:11434").rstrip("/")
    model = getattr(llm_cfg, "model", None) or "mistral"
    try:
        r = httpx.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": EXTRACT_PROMPT.format(text=text), "stream": False},
            timeout=60.0,
        )
        r.raise_for_status()
        body = r.json().get("response", "")
    except Exception as e:
        print(
            f"thought hook write: ollama extract failed ({e}); falling back to raw.",
            file=sys.stderr,
        )
        return [text]
    return [line.strip() for line in body.splitlines() if line.strip()]


def _extract_via_openai_compat(
    text: str,
    llm_cfg: object | None,
    *,
    provider: str,
) -> list[str]:
    try:
        import httpx
    except ImportError:
        print(
            f"thought hook write: {provider} extract requires httpx; falling back to raw.",
            file=sys.stderr,
        )
        return [text]
    # Provider defaults.
    if provider == "lmstudio":
        default_base = "http://localhost:1234/v1"
        default_model = "openai/gpt-oss-20b"
    elif provider == "openai":
        default_base = "https://api.openai.com/v1"
        default_model = "gpt-4o-mini"
    else:  # openai-compat
        default_base = "http://localhost:8000/v1"
        default_model = "gpt-4o-mini"
    base_url = (getattr(llm_cfg, "base_url", None) or default_base).rstrip("/")
    model = getattr(llm_cfg, "model", None) or default_model
    api_key = getattr(llm_cfg, "api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": EXTRACT_PROMPT.format(text=text)},
                ],
                "stream": False,
                "max_tokens": 512,
            },
            headers=headers,
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        body = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(
            f"thought hook write: {provider} extract failed ({e}); falling back to raw.",
            file=sys.stderr,
        )
        return [text]
    return [line.strip() for line in body.splitlines() if line.strip()]


def cli_main(
    *,
    db_path: str,
    mode: WriteMode = "raw",
    scope: str = "private",
    owner_id: str | None = None,
    embedder_choice: str = "auto",
    embedder_dim: int = 384,
    embedding_cfg=None,
    llm_cfg=None,
) -> int:
    """CLI entrypoint. Reads stdin, writes a one-line summary on stderr,
    returns exit code 0 unless we couldn't even parse the payload."""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        print(f"thought hook write: invalid JSON on stdin: {e}", file=sys.stderr)
        return 0  # don't surface as a hook error
    mem = Memory.open(
        db_path=db_path,
        embedder_choice=embedder_choice,
        embedder_dim=embedder_dim,
        embedding_cfg=embedding_cfg,
    )
    try:
        summary = run(
            memory=mem, payload=payload,
            mode=mode, scope=scope, owner_id=owner_id,
            llm_cfg=llm_cfg,
        )
    finally:
        mem.close()
    print(
        f"thought hook write: {summary}",
        file=sys.stderr,
    )
    return 0
