"""Idempotent installer for Claude Code hook entries in ``.claude/settings.json``.

Mirrors :mod:`thought.clients` (the MCP-server installer) in shape: read the
existing settings file, merge our entry in without disturbing anything else,
back up the original to ``settings.json.thought.bak`` before any write, return
a structured result.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

HookKind = Literal["recall", "write", "context"]
ALL_KINDS: tuple[HookKind, ...] = ("recall", "write", "context")

# Mapping from hook kind → Claude Code event name + the command to register.
_HOOK_SPEC: dict[HookKind, tuple[str, str]] = {
    "recall":  ("UserPromptSubmit", "thought hook recall"),
    "write":   ("Stop",             "thought hook write"),
    "context": ("SessionStart",     "thought hook context"),
}


@dataclass(frozen=True)
class HookInstallResult:
    kind: HookKind
    path: Path
    status: Literal["installed", "already_present", "error"]
    detail: str = ""


def settings_path(*, scope: Literal["project", "user"] = "project") -> Path:
    """Return the ``.claude/settings.json`` path for the requested scope.

    Project scope is the recommended default — it travels with the repo and
    is what most users actually want for THOUGHT-flavoured auto-memory.
    """
    if scope == "project":
        return Path.cwd() / ".claude" / "settings.json"
    return Path.home() / ".claude" / "settings.json"


def install_hook(
    kind: HookKind,
    *,
    scope: Literal["project", "user"] = "project",
    command_override: str | None = None,
    backup: bool = True,
) -> HookInstallResult:
    """Install (or no-op-reinstall) a single hook entry.

    Settings JSON shape (per Claude Code docs):
        {
          "hooks": {
            "<event>": [
              {"hooks": [{"type": "command", "command": "..."}]}
            ]
          }
        }
    """
    event, default_cmd = _HOOK_SPEC[kind]
    cmd = command_override or default_cmd
    path = settings_path(scope=scope)
    existing: dict
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as e:
            return HookInstallResult(
                kind=kind, path=path, status="error",
                detail=f"existing settings is not valid JSON: {e}",
            )
        if not isinstance(existing, dict):
            return HookInstallResult(
                kind=kind, path=path, status="error",
                detail="existing settings is not a JSON object",
            )
    else:
        existing = {}

    hooks = existing.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        return HookInstallResult(
            kind=kind, path=path, status="error",
            detail="existing 'hooks' is not an object",
        )

    event_entries = hooks.setdefault(event, [])
    if not isinstance(event_entries, list):
        return HookInstallResult(
            kind=kind, path=path, status="error",
            detail=f"existing 'hooks.{event}' is not a list",
        )

    # Idempotence: if any nested entry already references our command, skip.
    for entry in event_entries:
        if not isinstance(entry, dict):
            continue
        inner = entry.get("hooks") or []
        for h in inner:
            if isinstance(h, dict) and h.get("command") == cmd:
                return HookInstallResult(
                    kind=kind, path=path, status="already_present",
                    detail=f"{event} already registers {cmd!r}",
                )

    event_entries.append({"hooks": [{"type": "command", "command": cmd}]})

    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".thought.bak"))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return HookInstallResult(
        kind=kind, path=path, status="installed",
        detail=f"registered {cmd!r} on {event}",
    )


def install_many(
    kinds: tuple[HookKind, ...],
    *,
    scope: Literal["project", "user"] = "project",
) -> list[HookInstallResult]:
    return [install_hook(k, scope=scope) for k in kinds]
