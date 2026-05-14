"""MCP client config installer.

Knows about the five MCP clients in the README quickstart and how to merge a
``thought`` server entry into each one's config file without disturbing the
user's other settings.

Per-client config locations (per-platform):

| Client       | Path (~ = Path.home())                                                       |
|--------------|------------------------------------------------------------------------------|
| claude-code  | ~/.claude.json (newer) or ~/.claude/settings.json (project-scoped works too) |
| cursor       | ~/.cursor/mcp.json                                                            |
| cline        | VS Code globalStorage path (see below)                                        |
| continue     | ~/.continue/config.json                                                       |
| windsurf     | ~/.codeium/windsurf/mcp_config.json                                           |

All five use the same shape:

    { "mcpServers": { "<name>": { "command": "...", "args": [...] } } }

Cline lives inside VS Code's globalStorage and uses a slightly different
filename — handled in ``_cline_path``. We back up the existing file (suffix
``.thought.bak``) before writing.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ClientName = Literal["claude-code", "cursor", "cline", "continue", "windsurf"]
ALL_CLIENTS: tuple[ClientName, ...] = (
    "claude-code", "cursor", "cline", "continue", "windsurf",
)


@dataclass(frozen=True)
class ClientInstallResult:
    client: ClientName
    path: Path | None
    status: Literal["installed", "already_present", "no_path", "error"]
    detail: str = ""


def _claude_code_path() -> Path | None:
    """Claude Code reads MCP servers from ``~/.claude.json``."""
    p = Path.home() / ".claude.json"
    return p


def _cursor_path() -> Path | None:
    p = Path.home() / ".cursor" / "mcp.json"
    return p


def _cline_path() -> Path | None:
    """Cline lives in VS Code's globalStorage.

    The path differs per platform. We probe the common candidates and return
    the first existing parent directory; if none exist we return ``None``
    (the user likely doesn't have Cline installed).
    """
    home = Path.home()
    candidates: list[Path] = []
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(
                Path(appdata) / "Code" / "User" / "globalStorage"
                / "saoudrizwan.claude-dev" / "settings"
                / "cline_mcp_settings.json"
            )
    elif sys.platform == "darwin":
        candidates.append(
            home / "Library" / "Application Support" / "Code" / "User"
            / "globalStorage" / "saoudrizwan.claude-dev" / "settings"
            / "cline_mcp_settings.json"
        )
    else:  # linux / freebsd / etc.
        candidates.append(
            home / ".config" / "Code" / "User" / "globalStorage"
            / "saoudrizwan.claude-dev" / "settings"
            / "cline_mcp_settings.json"
        )
    # Also the per-user fallback path documented by Cline.
    candidates.append(home / ".cline" / "cline_mcp_settings.json")
    for c in candidates:
        if c.parent.exists():
            return c
    # Default to the home-dir fallback even if the parent isn't there yet —
    # ``mkdir(parents=True)`` will create it.
    return candidates[-1] if candidates else None


def _continue_path() -> Path | None:
    return Path.home() / ".continue" / "config.json"


def _windsurf_path() -> Path | None:
    return Path.home() / ".codeium" / "windsurf" / "mcp_config.json"


_PATH_FNS = {
    "claude-code": _claude_code_path,
    "cursor": _cursor_path,
    "cline": _cline_path,
    "continue": _continue_path,
    "windsurf": _windsurf_path,
}


def server_block(*, command: str = "uvx", args: tuple[str, ...] = ("thought-mcp", "serve")) -> dict:
    """The ``mcpServers`` entry every client uses.

    Defaults to ``uvx thought-mcp serve`` which works without any prior
    install — uvx fetches the package on first run. Users who want to pin
    a specific install can swap ``command`` to a path to ``thought``.
    """
    return {"command": command, "args": list(args)}


def detect_paths() -> dict[ClientName, Path | None]:
    """Map each known client to its config path (whether or not it exists yet)."""
    return {name: fn() for name, fn in _PATH_FNS.items()}


def install(
    client: ClientName,
    *,
    server_name: str = "thought",
    block: dict | None = None,
    backup: bool = True,
) -> ClientInstallResult:
    """Merge a ``thought`` MCP server entry into ``client``'s config.

    - If the config file doesn't exist, create it.
    - If a server entry with ``server_name`` already exists and matches the
      desired block exactly, return ``already_present``.
    - Otherwise back up the existing file (suffix ``.thought.bak``) and
      write the merged config back.
    """
    block = block or server_block()
    path = _PATH_FNS[client]()
    if path is None:
        return ClientInstallResult(
            client=client, path=None, status="no_path",
            detail=f"no known config path for {client}",
        )

    existing: dict
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as e:
            return ClientInstallResult(
                client=client, path=path, status="error",
                detail=f"existing config is not valid JSON: {e}",
            )
        if not isinstance(existing, dict):
            return ClientInstallResult(
                client=client, path=path, status="error",
                detail="existing config is not a JSON object",
            )
    else:
        existing = {}

    servers = existing.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        return ClientInstallResult(
            client=client, path=path, status="error",
            detail="existing 'mcpServers' is not an object",
        )

    if servers.get(server_name) == block:
        return ClientInstallResult(
            client=client, path=path, status="already_present",
            detail="thought server entry already configured",
        )

    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".thought.bak"))

    servers[server_name] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return ClientInstallResult(
        client=client, path=path, status="installed",
        detail=f"added '{server_name}' to mcpServers",
    )


def install_many(clients: Iterable[ClientName]) -> list[ClientInstallResult]:
    return [install(c) for c in clients]


# ---------------------------------------------------------------- upgrade

def pin_server_block(
    *, version: str | None = None,
    command: str = "uvx",
    extras: tuple[str, ...] = ("mcp", "sqlite-vec"),
) -> dict:
    """Return an ``mcpServers`` block that pins a specific ``thought-mcp`` version.

    Uses ``uvx --from "thought-mcp[mcp,sqlite-vec]==<ver>" thought serve``.
    The extras are mandatory in practice — the server crashes at startup
    without ``mcp``; the ANN path is unusably slow without ``sqlite-vec``.
    uvx re-resolves the named version each invocation, so cached older
    versions are bypassed without any user-visible cache management.

    ``version=None`` pins to the running CLI's ``__version__``.
    """
    if version is None:
        from . import __version__
        version = __version__
    extras_str = f"[{','.join(extras)}]" if extras else ""
    return {
        "command": command,
        "args": ["--from", f"thought-mcp{extras_str}=={version}", "thought", "serve"],
    }


def upgrade(
    client: ClientName,
    *,
    version: str | None = None,
    server_name: str = "thought",
    backup: bool = True,
) -> ClientInstallResult:
    """Update the ``thought`` entry in ``client``'s config to pin a version.

    Same safety guarantees as ``install``: idempotent on rerun, backs up
    the existing config before writing, refuses to touch non-JSON or
    non-object files.
    """
    return install(
        client,
        server_name=server_name,
        block=pin_server_block(version=version),
        backup=backup,
    )


def upgrade_many(
    clients_to_upgrade: Iterable[ClientName],
    *,
    version: str | None = None,
) -> list[ClientInstallResult]:
    return [upgrade(c, version=version) for c in clients_to_upgrade]
