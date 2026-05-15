"""Auto-recall hook for Claude Code's ``UserPromptSubmit`` event.

Reads a Claude-Code hook payload from stdin (JSON), pulls the user prompt,
runs ``Memory.recall`` against it, and writes a JSON response on stdout that
Claude Code merges into the next turn's context via the ``additionalContext``
field. The hook stays silent when the recall is low-confidence so it never
pollutes context with empty / unrelated noise.

Claude Code hooks spec: https://code.claude.com/docs/en/hooks
"""
from __future__ import annotations

import json
import sys
from typing import Any

from ..memory import Memory

# Claude Code caps additionalContext at 10k chars per hook. We aim well under
# that so even a multi-hit response stays well-formed and predictable.
MAX_CONTEXT_CHARS = 8_000


def run(
    *,
    memory: Memory,
    payload: dict[str, Any],
    limit: int = 5,
    scope: str = "all",
    owner_id: str | None = None,
) -> dict[str, Any]:
    """Execute one auto-recall hook invocation.

    Pure-Python — no I/O. Tests call this directly. Returns the Claude-Code
    hook response shape; the CLI wrapper handles stdin / stdout / exit codes.
    """
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return _empty_response("no prompt in hook payload")

    result = memory.recall(
        query=prompt, limit=limit,
        scope=scope,  # type: ignore[arg-type]
        owner_id=owner_id,
    )

    # Low-confidence gate: don't push noise into the context window.
    if result.low_confidence or not result.hits:
        return _empty_response("low confidence; skipping injection")

    lines = [f"--- thought recall ({len(result.hits)} hit{'s' if len(result.hits) > 1 else ''}) ---"]
    for h in result.hits:
        layer = h.layer
        conf = h.confidence_class
        cls = result.query_class.value if hasattr(result.query_class, "value") else str(result.query_class)
        lines.append(
            f"  • {h.entity.name}  "
            f"(score={h.score:.2f}  layer={layer}  "
            f"class={cls}  conf={conf})"
        )
    block = "\n".join(lines)
    if len(block) > MAX_CONTEXT_CHARS:
        block = block[:MAX_CONTEXT_CHARS - 24] + "\n  … (truncated)"

    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": block,
        },
    }


def _empty_response(reason: str) -> dict[str, Any]:
    """Quiet no-op response. Reason goes to stderr in the CLI wrapper."""
    return {"_skip_reason": reason}


def cli_main(
    *,
    db_path: str,
    limit: int = 5,
    scope: str = "all",
    owner_id: str | None = None,
    embedder_choice: str = "auto",
    embedder_dim: int = 384,
) -> int:
    """CLI entrypoint. Reads stdin, writes stdout, returns an exit code.

    Exit code is always 0 on a successful hook execution — even when nothing
    is injected — so Claude Code doesn't surface a transient hook warning to
    the user for an expected no-op.
    """
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        print(f"thought hook recall: invalid JSON on stdin: {e}", file=sys.stderr)
        return 0  # don't surface as a hook error

    mem = Memory.open(
        db_path=db_path,
        embedder_choice=embedder_choice,
        embedder_dim=embedder_dim,
    )
    try:
        response = run(
            memory=mem, payload=payload,
            limit=limit, scope=scope, owner_id=owner_id,
        )
    finally:
        mem.close()

    if "_skip_reason" in response:
        # Silent skip — emit nothing on stdout so Claude Code's
        # additionalContext stays untouched. Reason goes to stderr for the
        # MCP-logs panel.
        print(f"thought hook recall: skipped ({response['_skip_reason']})",
              file=sys.stderr)
        return 0

    json.dump(response, sys.stdout)
    sys.stdout.write("\n")
    return 0
