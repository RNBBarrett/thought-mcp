"""Claude Agent SDK adapter — drop-in memory provider for the Anthropic
Agent SDK so an agent loop can call ``mcp__thought__*`` tools as its memory
backend without writing any plumbing.

Usage:

    from anthropic import Anthropic
    from thought.adapters.claude_sdk import ThoughtMemoryProvider

    memory = ThoughtMemoryProvider(db_path=".thought/thought.db", agent="vuln-scanner")
    client = Anthropic()
    # ... build the agent loop using `memory.context_for(target)` between turns,
    # and `memory.record(content)` after each assistant action.

This adapter is intentionally thin — it wraps the existing ``Memory`` facade
so the SDK doesn't need to know about entities, edges, or PPR. Three methods
cover the typical loop:

- ``context_for(target, role)`` — call before each LLM turn; returns a
  ``working_context`` payload to inject as a system-prompt augmentation.
- ``record(content, source_ref=None)`` — call after the agent does
  something worth remembering.
- ``scan(repo_path)`` — for code-scanning agents; incremental.
"""
from __future__ import annotations

from typing import Any

from ..memory import Memory


class ThoughtMemoryProvider:
    """A minimal adapter from THOUGHT to any Claude-Agent-SDK-shaped agent.

    Construct once at agent startup; reuse across the agent loop.
    """

    def __init__(
        self,
        *,
        db_path: str = ".thought/thought.db",
        agent: str | None = None,
        embedder_choice: str = "auto",
        embedder_dim: int = 384,
    ) -> None:
        self._memory = Memory.open(
            db_path=db_path,
            embedder_choice=embedder_choice,
            embedder_dim=embedder_dim,
        )
        self._agent = agent
        if agent:
            self._memory.register_agent(agent)

    # ---- context primitives

    def context_for(
        self,
        target: str,
        *,
        role: str | None = None,
        budget_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Get a working-context payload for the LLM's next turn."""
        return self._memory.working_context(
            target, role=role or self._agent, budget_tokens=budget_tokens,
        )

    def render_context(
        self,
        target: str,
        *,
        role: str | None = None,
        budget_tokens: int = 2000,
    ) -> str:
        """Convenience: ``context_for`` formatted as plain text for system-prompt injection."""
        payload = self.context_for(
            target, role=role, budget_tokens=budget_tokens,
        )
        lines = []
        if payload.get("anchor"):
            a = payload["anchor"]
            lines.append(f"# Working on: {a['name']} ({a['type']})")
        if payload.get("neighbours"):
            lines.append("\n## Relevant prior knowledge:")
            for n in payload["neighbours"]:
                lines.append(f"- {n['name']} ({n['type']}, score={n['score']:.2f})")
        if payload.get("recent_contradictions"):
            lines.append("\n## Recent contradictions in scope:")
            for c in payload["recent_contradictions"]:
                lines.append(f"- {c.get('source_id')} CONTRADICTS {c.get('target_id')}")
        if payload.get("role_view"):
            lines.append(f"\n## Role-specific view ({role}):")
            lines.append(payload["role_view"]["cypher"])
        return "\n".join(lines) if lines else ""

    # ---- write primitives

    def record(
        self,
        content: str,
        *,
        source_ref: str | None = None,
        scope: str = "shared",
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist a fact the agent just learned."""
        r = self._memory.remember(
            content=content, source_ref=source_ref,
            scope=scope, owner_id=owner_id,  # type: ignore[arg-type]
        )
        return r.model_dump(mode="json")

    # ---- code-vertical primitives

    def scan(
        self,
        repo_path: str,
        *,
        since: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Incremental code-scan. Use this from a code-aware agent's main loop."""
        return self._memory.scan(
            repo_path, agent=self._agent, since=since, note=note,
        )

    # ---- lifecycle

    def close(self) -> None:
        self._memory.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
