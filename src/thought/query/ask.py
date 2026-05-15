"""Natural-language → Cypher translation, routed through whichever LLM
provider is configured in ``[llm] provider``.

Reuses the provider dispatch from :mod:`thought.hooks.write` but with a
different prompt aimed at structured Cypher emission. Validates the result
against the parser before executing — bad translations degrade gracefully
to a plain ``recall(question)`` fallback so the user always gets *something*.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from . import cypher

_PROMPT_TEMPLATE = """\
You translate English questions into Cypher queries against a memory database.

CONSTRAINTS (any violation will be rejected):
- Output a single Cypher query, no explanations, no markdown fences.
- Use ONLY these read-only Cypher features:
  MATCH (var:Type {{prop:value}}) (-[:RELATION]-> (var:Type))? ...
  WHERE expr (AND expr)*  using = <> < > <= >= CONTAINS "STARTS WITH" IN
  RETURN identlist
  LIMIT N (optional)
  AS_OF 'iso-date' (optional, for time-travel queries)
- Use ONLY entity types and relation types from the SCHEMA below.
- Never use MERGE / CREATE / DELETE / SET / WITH / variable-length paths.

SCHEMA:
  entity types: {entity_types}
  relation types: {relation_types}

QUESTION: {question}

CYPHER:"""


@dataclass
class AskResult:
    cypher: str | None
    sql: str | None
    rows: list[dict[str, Any]] | None
    fallback_used: bool
    fallback_reason: str | None
    error: str | None


def _build_prompt(schema: dict, question: str) -> str:
    et = ", ".join(schema.get("entity_types", {}).keys()) or "(none yet)"
    rt = ", ".join(schema.get("relation_types", {}).keys()) or "(none yet)"
    return _PROMPT_TEMPLATE.format(
        entity_types=et, relation_types=rt, question=question,
    )


def _extract_cypher(text: str) -> str:
    """Strip markdown fences and leading/trailing whitespace from LLM output."""
    s = text.strip()
    if s.startswith("```"):
        # Remove ```cypher / ```sql / ``` opener
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def ask(
    memory,
    question: str,
    *,
    llm_cfg: object | None = None,
    scope: str = "all",
    owner_id: str | None = None,
    no_fallback: bool = False,
) -> AskResult:
    """English → Cypher, validate, execute. Falls back to ``recall`` on bad
    translations unless ``no_fallback`` is set."""
    provider = getattr(llm_cfg, "provider", None) or "none"
    schema = memory.schema_summary()
    if provider == "none":
        if no_fallback:
            return AskResult(
                cypher=None, sql=None, rows=None, fallback_used=False,
                fallback_reason=None,
                error="thought ask requires [llm] provider to be set. "
                      "Configure anthropic / ollama / lmstudio / openai-compat in thought.toml.",
            )
        rows = _recall_fallback(memory, question, scope=scope, owner_id=owner_id)
        return AskResult(
            cypher=None, sql=None, rows=rows, fallback_used=True,
            fallback_reason="no LLM provider configured", error=None,
        )

    prompt = _build_prompt(schema, question)
    try:
        translation = _translate(prompt, llm_cfg=llm_cfg, provider=provider)
    except Exception as e:
        if no_fallback:
            return AskResult(
                cypher=None, sql=None, rows=None, fallback_used=False,
                fallback_reason=None, error=f"LLM call failed: {e}",
            )
        rows = _recall_fallback(memory, question, scope=scope, owner_id=owner_id)
        return AskResult(
            cypher=None, sql=None, rows=rows, fallback_used=True,
            fallback_reason=f"LLM call failed: {e}", error=None,
        )

    cypher_source = _extract_cypher(translation)
    try:
        query = cypher.parse(cypher_source)
        from ..models import ScopeFilter
        sql, params, columns = cypher.compile_to_sql(
            query, scope_filter=ScopeFilter(scope=scope, owner_id=owner_id),  # type: ignore[arg-type]
        )
        rows_raw = memory._backend._conn.execute(sql, params).fetchall()
        rows = []
        for r in rows_raw:
            row_dict: dict[str, Any] = {}
            for i, col in enumerate(columns):
                v = r[i]
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    import json as _json
                    try:
                        row_dict[col] = _json.loads(v)
                        continue
                    except _json.JSONDecodeError:
                        pass
                row_dict[col] = v
            rows.append(row_dict)
        return AskResult(
            cypher=cypher_source, sql=sql, rows=rows,
            fallback_used=False, fallback_reason=None, error=None,
        )
    except cypher.CypherError as e:
        if no_fallback:
            return AskResult(
                cypher=cypher_source, sql=None, rows=None,
                fallback_used=False, fallback_reason=None,
                error=f"emitted Cypher invalid: {e}",
            )
        rows = _recall_fallback(memory, question, scope=scope, owner_id=owner_id)
        return AskResult(
            cypher=cypher_source, sql=None, rows=rows,
            fallback_used=True,
            fallback_reason=f"emitted Cypher invalid ({e}); fell back to recall",
            error=None,
        )


def _translate(prompt: str, *, llm_cfg, provider: str) -> str:
    """Delegate to the right provider. Reuses the prompt structure across
    Anthropic / Ollama / LM Studio / OpenAI-compatible."""
    if provider == "anthropic":
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        model = getattr(llm_cfg, "model", None) or "claude-haiku-4-5-20251001"
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            b.text  # type: ignore[union-attr]
            for b in resp.content
            if getattr(b, "type", None) == "text"
        )
    if provider == "ollama":
        import httpx
        host = (getattr(llm_cfg, "base_url", None) or "http://localhost:11434").rstrip("/")
        model = getattr(llm_cfg, "model", None) or "mistral"
        r = httpx.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    if provider in {"lmstudio", "openai-compat", "openai"}:
        import httpx
        defaults = {
            "lmstudio":      ("http://localhost:1234/v1", "openai/gpt-oss-20b"),
            "openai":        ("https://api.openai.com/v1", "gpt-4o-mini"),
            "openai-compat": ("http://localhost:8000/v1", "gpt-4o-mini"),
        }
        default_base, default_model = defaults[provider]
        base_url = (getattr(llm_cfg, "base_url", None) or default_base).rstrip("/")
        model = getattr(llm_cfg, "model", None) or default_model
        api_key = getattr(llm_cfg, "api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        r = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False, "max_tokens": 512,
            },
            headers=headers, timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"unknown provider {provider!r}")


def _recall_fallback(memory, question: str, *, scope: str, owner_id: str | None):
    """Return ``recall(question)`` hits formatted as row dicts."""
    result = memory.recall(query=question, limit=5, scope=scope, owner_id=owner_id)
    return [
        {
            "entity_name": h.entity.name,
            "entity_type": h.entity.type,
            "layer": h.layer,
            "score": h.score,
            "confidence": h.confidence_class,
        }
        for h in result.hits
    ]
