"""``thought`` CLI.

Commands:
- ``thought init``               — create db + config + CLAUDE.md hint
- ``thought serve``              — start the MCP server (stdio by default;
                                   ``--transport streamable-http`` for HTTP)
- ``thought ingest TEXT``        — one-shot remember from the command line
- ``thought ingest --file PATH`` — ingest a single file
- ``thought ingest --glob PAT``  — bulk-ingest matching files (one per item)
- ``thought ingest --stdin``     — bulk-ingest one line-per-item from stdin
- ``thought recall QUERY``       — pretty-printed recall results
- ``thought repl``               — interactive query shell
- ``thought stats``              — what's in the KB
- ``thought forget PATTERN``     — soft-delete entities matching a SQL LIKE pattern
- ``thought consolidate``        — run one consolidation cycle
- ``thought doctor``             — environment health check
"""
from __future__ import annotations

import glob as _glob
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdio so emoji / em-dashes / box-drawing don't UnicodeEncodeError
# on Windows consoles defaulting to cp1252. Safe no-op on Linux/macOS.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover — non-TTY pipes etc.
        pass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from . import __version__
from . import clients as mcp_clients
from .config import Settings, find_config, load_settings
from .ingest.pipeline import IngestItem
from .memory import Memory

app = typer.Typer(
    name="thought",
    add_completion=False,
    help="THOUGHT — Temporal Hierarchical Object Union & Graph Hybrid Toolkit. "
         "Local MCP memory for any LLM client.",
)
hook_app = typer.Typer(
    name="hook",
    help="Claude Code hook integrations for auto-write + auto-recall.",
)
app.add_typer(hook_app, name="hook")
console = Console(stderr=False)
err_console = Console(stderr=True)


def _open_memory(settings: Settings) -> Memory:
    return Memory.open(
        db_path=settings.db_path,
        embedder_choice=settings.embedding.choice,
        embedder_dim=settings.embedding.dim,
        consolidation_enabled=False,
    )


# ---------------------------------------------------------------- root

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
):
    if version:
        console.print(f"thought-mcp [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# ---------------------------------------------------------------- init

CLAUDE_MD_TEMPLATE = """\
# THOUGHT — agent-facing usage

This project uses [thought-mcp](https://github.com/RNBBarrett/thought-mcp)
as a persistent memory server. It exposes two MCP tools:

- `remember(content, scope='private', owner_id=None)` — stores a piece of text,
  extracts entities + typed edges, and tracks provenance. Idempotent on
  content sha256 — calling it twice with the same content is a no-op.

- `recall(query, limit=10, scope='all', as_of=None)` — retrieves up to 10
  results. The router classifies the query (VIBE / FACT / CHANGE / HYBRID)
  and dispatches to the right layer: vector search for similarity, graph
  PageRank for relationships, temporal scan for "as of" history.

## Guidance

- Use **`scope='private'` with an `owner_id`** for user-specific facts.
  Use `scope='shared'` for project- or org-wide facts.
- Every `recall` hit carries `confidence_class`: `source_grounded` (read
  from a real source), `inferred` (derived via the graph), or
  `hallucination_risk` (low evidence). Trust them in that order.
- `as_of=<ISO timestamp>` with `as_of_kind='valid'` answers "what was true
  on date X". `as_of_kind='learned'` answers "what did the system know on
  date X" — they differ when facts are corrected after the fact.
- Contradictions surface as `CONTRADICTS` edges. They're data, not
  warnings — feel free to query them.
- Bounded results: `recall` never returns more than 10 hits regardless
  of KB size. Use `as_of` and `scope` to narrow further.
"""


@app.command()
def init(
    config: Path = typer.Option("thought.toml", help="Path to config file."),
    db_path: str = typer.Option(".thought/thought.db", help="SQLite database path."),
    embedder: str = typer.Option(
        "auto", help="'auto' picks sentence-transformers if available, else deterministic.",
    ),
    write_claude_md: bool = typer.Option(
        True, "--write-claude-md/--no-claude-md",
        help="Drop a CLAUDE.md so MCP clients learn how to use the tool.",
    ),
    quick: bool = typer.Option(
        False, "--quick", help="Skip first-run embedder warmup.",
    ),
) -> None:
    """Create database file + config + agent-facing CLAUDE.md."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if not config.exists():
        # Use POSIX-style separators in the TOML so Windows paths like
        # ``C:\Users\...\thought.db`` don't blow up the TOML parser on the
        # next CLI call — backslashes in TOML basic strings are escape
        # sequences (``\U`` / ``\x`` etc.). SQLite accepts forward slashes
        # on Windows.
        db_path_for_toml = db_path.replace("\\", "/")
        config.write_text(
            f'db_path = "{db_path_for_toml}"\n\n[embedding]\nchoice = "{embedder}"\ndim = 384\n',
            encoding="utf-8",
        )
        console.print(f"  [ok] wrote [bold]{config}[/bold]")
    if write_claude_md:
        claude_md = Path("CLAUDE.md")
        if claude_md.exists():
            console.print(f"  [yellow]![/yellow] {claude_md} exists; not overwriting")
        else:
            claude_md.write_text(CLAUDE_MD_TEMPLATE, encoding="utf-8")
            console.print(f"  [ok] wrote [bold]{claude_md}[/bold]")
    mem = Memory.open(db_path=db_path, embedder_choice=embedder, embedder_dim=384)
    if not quick and embedder == "auto":
        try:
            mem._embedder.embed("warmup")
        except Exception as e:  # pragma: no cover
            err_console.print(f"[yellow]embedder warmup skipped: {e}[/yellow]")
    mem.close()
    console.print(f"  [ok] initialised [bold]{db_path}[/bold]")
    console.print(
        Panel(
            "[green]Ready.[/green]  Next:\n"
            "  [bold]thought install --client cursor[/bold]  wire it into your IDE\n"
            "  [bold]thought start[/bold]                     one-command run\n"
            "  [bold]thought ingest 'Alice owns Acme.'[/bold]\n"
            "  [bold]thought recall 'who owns Acme'[/bold]",
            title="Next steps",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------- serve

def _precheck() -> list[str]:
    """Lightweight fail-fast checks before binding the server port.

    Returns a list of warnings (empty = all good). Hard errors raise.
    """
    warnings: list[str] = []
    import sqlite3
    conn = sqlite3.connect(":memory:")
    if not hasattr(conn, "enable_load_extension"):
        warnings.append(
            "sqlite extension loading unavailable (Anaconda?) — "
            "vec search will use the slow Python fallback."
        )
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        warnings.append(
            "sqlite-vec not installed — vec search slow. "
            "pip install 'thought-mcp[sqlite-vec]'"
        )
    try:
        import mcp  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "MCP transport not installed. Run: pip install 'thought-mcp[mcp]'"
        ) from e
    return warnings


def _resolve_config(config: Path | None) -> Path:
    """Return the config path to load, walking up if unspecified."""
    if config is not None and str(config) != "thought.toml":
        return config
    found = find_config()
    return found if found else Path("thought.toml")


@app.command()
def serve(
    config: Path = typer.Option(Path("thought.toml"), help="Config file."),
    host: str | None = typer.Option(None, help="Bind host (streamable-http only)."),
    port: int | None = typer.Option(None, help="Bind port (streamable-http only)."),
    transport: str = typer.Option(
        "stdio", "--transport", "-t",
        help="MCP transport: 'stdio' (default, used by MCP-client child-process invocations) "
             "or 'streamable-http' (binds an HTTP listener — useful for local dev / remote clients).",
    ),
    skip_precheck: bool = typer.Option(
        False, "--skip-precheck", help="Skip the doctor precheck before serving.",
    ),
) -> None:
    """Start the MCP server.

    Default transport is stdio: every MCP client config wired up by
    ``thought install`` / ``thought upgrade`` invokes
    ``uvx --from "thought-mcp[mcp,sqlite-vec]==X" thought serve`` and expects
    to speak MCP over the child process's stdin/stdout. Pass
    ``--transport streamable-http`` for the HTTP transport.
    """
    if transport not in {"stdio", "streamable-http"}:
        err_console.print(
            f"[red]unknown transport {transport!r}[/red] — choose 'stdio' or 'streamable-http'"
        )
        raise typer.Exit(2)
    if not skip_precheck:
        warnings = _precheck()
        for w in warnings:
            err_console.print(f"[yellow]warn[/yellow]  {w}")
    settings = load_settings(_resolve_config(config))
    if host:
        settings.server.host = host
    if port is not None:
        settings.server.port = port
    mem = _open_memory(settings)
    if settings.consolidation.enabled:
        mem._consolidator.start()
    from .server import build_app
    mcp_app = build_app(mem)
    if transport == "streamable-http":
        # FastMCP stores its own host/port in ``mcp_app.settings``; without
        # this push-through our ``--host`` / ``--port`` flags are silently
        # ignored and the server binds 0.0.0.0:8000 regardless.
        mcp_app.settings.host = settings.server.host
        mcp_app.settings.port = settings.server.port
        err_console.print(
            f"[bold]thought-mcp {__version__}[/bold] serving on "
            f"http://{settings.server.host}:{settings.server.port}"
        )
    else:
        # stdio: banner goes to stderr so it doesn't corrupt the MCP frames
        # on stdout. Most MCP clients surface stderr in their logs panel.
        err_console.print(
            f"[dim]thought-mcp {__version__} ready (stdio transport)[/dim]"
        )
    try:
        mcp_app.run(transport=transport)  # type: ignore[arg-type]
    finally:
        mem.close()


# ---------------------------------------------------------------- install

@app.command()
def install(
    client: str | None = typer.Option(
        None, "--client", "-c",
        help="Client to install into: claude-code, cursor, cline, continue, windsurf.",
    ),
    all_clients: bool = typer.Option(
        False, "--all", help="Install into every detected client.",
    ),
    detect: bool = typer.Option(
        False, "--detect", help="Just print detected config paths and exit.",
    ),
) -> None:
    """Auto-wire ``thought`` into a supported MCP client.

    Writes (or merges) an ``mcpServers`` entry into the client's config file.
    Backs up the original to ``<file>.thought.bak``. Idempotent on rerun.

    Examples:
        thought install --client cursor
        thought install --all
        thought install --detect
    """
    if detect:
        table = Table(title="Detected MCP client config paths", border_style="cyan")
        table.add_column("Client", style="bold")
        table.add_column("Path")
        table.add_column("Exists?")
        for name, p in mcp_clients.detect_paths().items():
            table.add_row(
                name,
                str(p) if p else "[dim]unknown[/dim]",
                "[green]yes[/green]" if p and p.exists() else "[dim]no[/dim]",
            )
        console.print(table)
        return

    if all_clients:
        targets: tuple[str, ...] = mcp_clients.ALL_CLIENTS
    elif client is not None:
        if client not in mcp_clients.ALL_CLIENTS:
            err_console.print(
                f"[red]unknown client {client!r}[/red] — "
                f"choose from {', '.join(mcp_clients.ALL_CLIENTS)} or use --all"
            )
            raise typer.Exit(2)
        targets = (client,)
    else:
        err_console.print(
            "[red]specify --client <name> or --all[/red] "
            f"(known clients: {', '.join(mcp_clients.ALL_CLIENTS)})"
        )
        raise typer.Exit(2)

    table = Table(title="Install results", border_style="cyan")
    table.add_column("Client", style="bold")
    table.add_column("Status")
    table.add_column("Path")
    for r in mcp_clients.install_many(targets):  # type: ignore[arg-type]
        style = {
            "installed": "green",
            "already_present": "yellow",
            "no_path": "dim",
            "error": "red",
        }[r.status]
        table.add_row(
            r.client,
            f"[{style}]{r.status}[/{style}]",
            str(r.path) if r.path else "—",
        )
        if r.status == "error":
            err_console.print(f"[red]error[/red] ({r.client}): {r.detail}")
    console.print(table)
    console.print(
        "[dim]restart your client(s) to pick up the new server entry.[/dim]"
    )


# ---------------------------------------------------------------- upgrade

@app.command()
def upgrade(
    client: str | None = typer.Option(
        None, "--client", "-c",
        help="Client to upgrade: claude-code, cursor, cline, continue, windsurf.",
    ),
    all_clients: bool = typer.Option(
        False, "--all", help="Upgrade every detected client.",
    ),
    version: str | None = typer.Option(
        None, "--version", "-V",
        help="Specific version to pin (e.g. 0.2.0). Default: this CLI's version.",
    ),
) -> None:
    """Re-pin one or all MCP clients to a specific thought-mcp version.

    Forces ``uvx`` to fetch the named version instead of using its cached
    older copy. Use this whenever you upgrade ``thought-mcp`` itself and
    want your IDE's MCP server to actually pick up the new version on
    next restart.

    Examples:
        thought upgrade --all                  # pin every client to this CLI's version
        thought upgrade --client cursor -V 0.2.1
    """
    target_version = version or __version__
    if all_clients:
        targets: tuple[str, ...] = mcp_clients.ALL_CLIENTS
    elif client is not None:
        if client not in mcp_clients.ALL_CLIENTS:
            err_console.print(
                f"[red]unknown client {client!r}[/red] — "
                f"choose from {', '.join(mcp_clients.ALL_CLIENTS)} or use --all"
            )
            raise typer.Exit(2)
        targets = (client,)
    else:
        err_console.print(
            "[red]specify --client <name> or --all[/red] "
            f"(known clients: {', '.join(mcp_clients.ALL_CLIENTS)})"
        )
        raise typer.Exit(2)

    table = Table(
        title=f"Upgrade results — pinned to thought-mcp=={target_version}",
        border_style="cyan",
    )
    table.add_column("Client", style="bold")
    table.add_column("Status")
    table.add_column("Path")
    for r in mcp_clients.upgrade_many(targets, version=target_version):  # type: ignore[arg-type]
        style = {
            "installed": "green",
            "already_present": "yellow",
            "no_path": "dim",
            "error": "red",
        }[r.status]
        table.add_row(
            r.client,
            f"[{style}]{r.status}[/{style}]",
            str(r.path) if r.path else "—",
        )
        if r.status == "error":
            err_console.print(f"[red]error[/red] ({r.client}): {r.detail}")
    console.print(table)
    console.print(
        "[dim]restart your client(s) to pick up the new MCP server version.[/dim]"
    )


# ---------------------------------------------------------------- start

@app.command()
def start(
    client: str | None = typer.Option(
        None, "--client", "-c", help="Also wire this MCP client before serving.",
    ),
    config: Path = typer.Option(Path("thought.toml")),
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
) -> None:
    """One-command bootstrap: init-if-needed + (optional) install + serve.

    The fastest path from zero to a running memory server:

        thought start --client cursor
    """
    cfg_path = _resolve_config(config)
    if not cfg_path.exists():
        console.print("[dim]no config found — running init…[/dim]")
        init(config=cfg_path, db_path=".thought/thought.db",
             embedder="auto", write_claude_md=True, quick=False)
    if client:
        install(client=client, all_clients=False, detect=False)
    # ``start`` is the human-facing "run a server in a terminal" entrypoint
    # — pin streamable-http so stdout isn't claimed by MCP frames and the
    # user can ctrl-C / read logs normally.
    serve(
        config=cfg_path, host=host, port=port,
        transport="streamable-http", skip_precheck=False,
    )


# ---------------------------------------------------------------- ingest

@app.command()
def ingest(
    content: str | None = typer.Argument(None, help="Text content to remember."),
    scope: str = typer.Option("private", help="'shared' or 'private'."),
    owner_id: str | None = typer.Option(None),
    file: Path | None = typer.Option(
        None, "--file", "-f", help="Read content from this file.",
    ),
    glob: str | None = typer.Option(
        None, "--glob", "-g",
        help="Glob pattern; ingests each matching file as one item.",
    ),
    stdin: bool = typer.Option(
        False, "--stdin", help="Bulk-ingest one item per stdin line.",
    ),
    config: Path = typer.Option("thought.toml"),
) -> None:
    """Remember content from CLI / file / glob / stdin.

    Bulk paths (``--glob``, ``--stdin``) batch all writes into one DB
    transaction — measured 2.3× faster than a loop of single ingests.
    """
    settings = load_settings(config)
    mem = _open_memory(settings)
    try:
        if stdin:
            items = [
                IngestItem(content=line, scope=scope, owner_id=owner_id)  # type: ignore[arg-type]
                for line in (line.strip() for line in sys.stdin)
                if line
            ]
            _bulk_ingest(mem, items, "stdin")
        elif glob is not None:
            paths = [Path(p) for p in _glob.glob(glob, recursive=True)]
            paths = [p for p in paths if p.is_file()]
            if not paths:
                err_console.print(f"[red]no files matched[/red] {glob!r}")
                raise typer.Exit(1)
            items = [
                IngestItem(
                    content=p.read_text(encoding="utf-8"),
                    scope=scope,  # type: ignore[arg-type]
                    owner_id=owner_id,
                )
                for p in paths
            ]
            _bulk_ingest(mem, items, f"{len(paths)} files")
        elif file is not None:
            text = file.read_text(encoding="utf-8")
            result = mem.remember(content=text, scope=scope, owner_id=owner_id)  # type: ignore[arg-type]
            console.print_json(data=result.model_dump(mode="json"))
        else:
            if not content:
                err_console.print(
                    "[red]must provide content, --file, --glob, or --stdin[/red]"
                )
                raise typer.Exit(1)
            result = mem.remember(content=content, scope=scope, owner_id=owner_id)  # type: ignore[arg-type]
            console.print_json(data=result.model_dump(mode="json"))
    finally:
        mem.close()


def _bulk_ingest(mem: Memory, items: list[IngestItem], label: str) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=err_console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"ingesting {label}", total=len(items))
        results = mem.remember_many(items)
        progress.update(task, completed=len(items))
    n_new = sum(1 for r in results if r.duplicate_of_source is None)
    n_dup = len(results) - n_new
    n_entities = sum(len(r.entity_ids) for r in results)
    n_contradictions = sum(len(r.contradictions_detected) for r in results)
    table = Table(title="Ingest summary", show_header=False, border_style="cyan")
    table.add_column(style="bold")
    table.add_column()
    table.add_row("items processed", str(len(results)))
    table.add_row("new sources", str(n_new))
    table.add_row("duplicates skipped", str(n_dup))
    table.add_row("entities created", str(n_entities))
    table.add_row("contradictions", str(n_contradictions))
    console.print(table)


# ---------------------------------------------------------------- recall

@app.command()
def recall(
    query: str = typer.Argument(..., help="Query text."),
    limit: int = typer.Option(10),
    scope: str = typer.Option("all"),
    owner_id: str | None = typer.Option(None),
    as_of: str | None = typer.Option(None, help="ISO-8601 timestamp."),
    as_of_kind: str = typer.Option("valid", help="'valid' (world-time) or 'learned' (transaction-time)."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of pretty output."),
    config: Path = typer.Option("thought.toml"),
) -> None:
    """Recall hits for ``query`` — pretty table by default."""
    settings = load_settings(config)
    mem = _open_memory(settings)
    try:
        result = mem.recall(
            query=query, limit=limit, scope=scope,  # type: ignore[arg-type]
            owner_id=owner_id,
            as_of=datetime.fromisoformat(as_of) if as_of else None,
            as_of_kind=as_of_kind,  # type: ignore[arg-type]
        )
    finally:
        mem.close()
    if json_out:
        console.print_json(data=result.model_dump(mode="json"))
        return
    _render_recall(query, result)


_CONFIDENCE_STYLE = {
    "source_grounded": "green",
    "inferred": "yellow",
    "hallucination_risk": "red",
}


def _render_recall(query: str, result) -> None:
    header = (
        f"[bold]{query}[/bold]   [dim]class={result.query_class.value}  "
        f"elapsed={result.elapsed_ms:.1f}ms  "
        f"{'[red]low-confidence[/red]' if result.low_confidence else '[green]ok[/green]'}[/dim]"
    )
    if not result.hits:
        console.print(Panel("[red]no hits[/red]", title=header, border_style="red"))
        return
    table = Table(title=header, border_style="cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("score", justify="right", width=7)
    table.add_column("layer", width=8)
    table.add_column("confidence", width=18)
    table.add_column("entity")
    for i, h in enumerate(result.hits, 1):
        cstyle = _CONFIDENCE_STYLE.get(h.confidence_class, "white")
        table.add_row(
            str(i),
            f"{h.score:.3f}",
            h.layer,
            f"[{cstyle}]{h.confidence_class}[/{cstyle}]",
            h.entity.name,
        )
    console.print(table)
    if result.sources:
        console.print(f"[dim]{len(result.sources)} unique source(s)[/dim]")


# ---------------------------------------------------------------- repl

@app.command()
def repl(
    config: Path = typer.Option("thought.toml"),
) -> None:
    """Interactive query shell.

    Type a query and hit enter to recall. Empty line exits. Prefix with
    ``+`` to remember instead. ``?`` shows help. ``stats`` shows the KB.
    """
    settings = load_settings(config)
    mem = _open_memory(settings)
    s = mem.stats()
    console.print(Panel(
        f"[bold]thought {__version__}[/bold]\n"
        f"{s['entities_current']} entities • {s['edges_total']} edges • "
        f"{s['sources']} sources • {s['contradictions']} contradictions\n"
        f"[dim]type a query, or +text to remember, or 'stats' / 'q' / '?' [/dim]",
        title="Interactive shell",
        border_style="cyan",
    ))
    try:
        while True:
            try:
                line = Prompt.ask("[bold cyan]thought›[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                break
            line = line.strip()
            if not line or line in {"q", "quit", "exit"}:
                break
            if line == "?":
                console.print(
                    "  + <text>          remember\n"
                    "  <text>            recall (default top-10)\n"
                    "  stats             show KB summary\n"
                    "  q / quit          exit"
                )
                continue
            if line == "stats":
                _render_stats(mem.stats())
                continue
            if line.startswith("+"):
                content = line[1:].strip()
                if not content:
                    continue
                r = mem.remember(content=content, scope="private")
                console.print(
                    f"  [green][ok][/green] source={r.source_id[:12]}... "
                    f"{len(r.entity_ids)} entities, "
                    f"{len(r.contradictions_detected)} contradictions"
                )
                continue
            result = mem.recall(query=line, limit=10)
            _render_recall(line, result)
    finally:
        mem.close()


# ---------------------------------------------------------------- stats

@app.command()
def stats(
    config: Path = typer.Option("thought.toml"),
) -> None:
    """Show what's in the memory."""
    settings = load_settings(config)
    mem = _open_memory(settings)
    try:
        _render_stats(mem.stats())
    finally:
        mem.close()


def _render_stats(s: dict) -> None:
    head = Table(title="KB summary", show_header=False, border_style="cyan")
    head.add_column(style="bold")
    head.add_column(justify="right")
    head.add_row("entities (currently valid)", str(s["entities_current"]))
    head.add_row("entities (total incl. retired)", str(s["entities_total"]))
    head.add_row("edges", str(s["edges_total"]))
    head.add_row("contradictions", str(s["contradictions"]))
    head.add_row("sources", str(s["sources"]))
    head.add_row("write version", str(s["write_version"]))
    console.print(head)
    tiers = Table(title="Tier distribution", show_header=False, border_style="dim")
    tiers.add_column(style="bold")
    tiers.add_column(justify="right")
    tiers.add_row("hot", str(s["tier_hot"]))
    tiers.add_row("warm", str(s["tier_warm"]))
    tiers.add_row("cold", str(s["tier_cold"]))
    console.print(tiers)
    if s["top_accessed"]:
        top = Table(title="Top 10 by access count", border_style="dim")
        top.add_column("name")
        top.add_column("count", justify="right")
        for row in s["top_accessed"]:
            top.add_row(row["name"], str(row["count"]))
        console.print(top)


# ---------------------------------------------------------------- forget

@app.command()
def forget(
    pattern: str = typer.Argument(..., help="SQL LIKE pattern, e.g. 'kendra%'."),
    scope: str = typer.Option("all"),
    owner_id: str | None = typer.Option(None),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
    config: Path = typer.Option("thought.toml"),
) -> None:
    """Soft-delete entities by canonical-name pattern.

    Sets ``valid_until = now`` on currently-valid matching rows and writes
    a FORGET audit log entry. Append-only — no rows are deleted; the change
    is reversible by editing the DB directly.
    """
    settings = load_settings(config)
    mem = _open_memory(settings)
    try:
        if not yes:
            confirm = Prompt.ask(
                f"[yellow]Retire entities matching {pattern!r} in scope={scope!r}? "
                f"[y/N][/yellow]",
                default="N",
            )
            if confirm.strip().lower() not in {"y", "yes"}:
                console.print("[dim]aborted[/dim]")
                return
        retired = mem.forget(pattern, scope=scope, owner_id=owner_id)  # type: ignore[arg-type]
    finally:
        mem.close()
    console.print(f"  [green][ok][/green] retired {len(retired)} entit{'y' if len(retired)==1 else 'ies'}")


# ---------------------------------------------------------------- consolidate

@app.command()
def consolidate(
    config: Path = typer.Option("thought.toml"),
) -> None:
    """Run one consolidation cycle."""
    settings = load_settings(config)
    mem = _open_memory(settings)
    try:
        n = mem.consolidate()
    finally:
        mem.close()
    console.print(f"  [green][ok][/green] consolidation complete: {n} audit entries")


# ---------------------------------------------------------------- doctor

# ---------------------------------------------------------------- code-vertical (v0.2)

@app.command("ingest-code")
def ingest_code_cmd(
    path: Path = typer.Argument(..., help="File or directory to ingest."),
    glob_pattern: str = typer.Option(
        "**/*.py", "--glob", "-g",
        help="Glob pattern (when path is a directory). Default: **/*.py.",
    ),
    lang: str = typer.Option(
        "auto", "--lang", help="'auto' detects from extension; or 'python' / 'typescript'.",
    ),
    scope: str = typer.Option("shared"),
    owner_id: str | None = typer.Option(None),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Ingest source code via tree-sitter (Python supported; TS arrives in v0.2.x).

    Examples:
        thought ingest-code src/auth.py
        thought ingest-code src/ --glob '**/*.py'
        thought ingest-code mypkg/ --lang python
    """
    from .ingest.code.call_graph import build_call_graph
    from .ingest.code.pipeline import CodeIngestPipeline
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        pipe = CodeIngestPipeline(
            backend=mem._backend, embedder=mem._embedder,
            scope=scope, owner_id=owner_id,  # type: ignore[arg-type]
        )
        files = (
            [path] if path.is_file()
            else sorted(p for p in path.rglob(glob_pattern) if p.is_file())
        )
        if not files:
            err_console.print(f"[red]no files matched[/red] {glob_pattern!r}")
            raise typer.Exit(1)

        from datetime import UTC, datetime
        now = datetime.now(UTC)
        total_entities = 0
        total_calls = 0
        root = path if path.is_dir() else path.parent

        # Two-pass: ingest ALL files first (so cross-file references can
        # resolve), then walk the call graph. Doing them per-file pollutes
        # the resolver with stubs that win over real qualified methods on
        # later files.
        per_file: list[tuple[Path, str, str]] = []
        with Progress(
            SpinnerColumn(), TextColumn("[bold]{task.description}"),
            BarColumn(), MofNCompleteColumn(),
            console=err_console, transient=True,
        ) as prog:
            task = prog.add_task("pass 1/2: ingesting entities", total=len(files))
            for f in files:
                detected = lang if lang != "auto" else None
                r = pipe.ingest_code_file(
                    f, commit_sha=None, language=detected, now=now,
                    repo_root=root,
                )
                total_entities += len(r.entity_ids)
                rel = f.resolve().relative_to(root.resolve()).as_posix()
                per_file.append((f, rel, r.source_id))
                prog.advance(task)
            task2 = prog.add_task("pass 2/2: building call graph", total=len(per_file))
            for f, rel, source_id in per_file:
                detected = lang if lang != "auto" else None
                total_calls += build_call_graph(
                    backend=mem._backend, file_path=rel,
                    source=f.read_text(encoding="utf-8"),
                    language=detected or "python",
                    commit_sha=None,
                    scope=scope, owner_id=owner_id,  # type: ignore[arg-type]
                    source_ref=source_id, now=now,
                )
                prog.advance(task2)

        table = Table(title="Code-ingest summary", show_header=False, border_style="cyan")
        table.add_column(style="bold")
        table.add_column(justify="right")
        table.add_row("files processed", str(len(files)))
        table.add_row("entities created", str(total_entities))
        table.add_row("CALLS edges", str(total_calls))
        console.print(table)
    finally:
        mem.close()


@app.command("ingest-git")
def ingest_git_cmd(
    repo_path: Path = typer.Argument(Path("."), help="Path to a git repo."),
    mode: str = typer.Option(
        "snapshot", "--mode", help="'snapshot' (HEAD only) or 'full' (every commit).",
    ),
    paths: str = typer.Option(
        "*.py", "--paths", help="Comma-separated glob patterns to ingest.",
    ),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Ingest a git repository's code with commit-stamped provenance.

    ``--mode snapshot`` (default) ingests only HEAD. ``--mode full`` walks
    every commit and stamps each entity with its commit SHA — enables
    ``thought diff --from <sha1> --to <sha2>`` queries.
    """
    from .ingest.code.git_pipeline import GitIngestPipeline
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        pipe = GitIngestPipeline(
            backend=mem._backend, embedder=mem._embedder,
        )
        from datetime import UTC, datetime
        path_patterns = tuple(p.strip() for p in paths.split(",") if p.strip())
        with Progress(
            SpinnerColumn(), TextColumn("[bold]{task.description}"),
            console=err_console, transient=True,
        ) as prog:
            prog.add_task(f"git-ingest ({mode})", total=None)
            r = pipe.ingest_history(
                repo_path, mode=mode,  # type: ignore[arg-type]
                paths=path_patterns, now=datetime.now(UTC),
            )

        table = Table(title="Git-ingest summary", show_header=False, border_style="cyan")
        table.add_column(style="bold")
        table.add_column(justify="right")
        table.add_row("HEAD", r.head_sha[:12] + "…")
        table.add_row("mode", r.mode)
        table.add_row("commits visited", str(r.commits_visited))
        table.add_row("files ingested", str(r.files_ingested))
        table.add_row("CALLS edges", str(r.call_edges))
        console.print(table)
    finally:
        mem.close()


@app.command()
def topics(
    scope: str = typer.Option("all", help="'shared', 'private', or 'all'."),
    owner_id: str | None = typer.Option(None),
    min_count: int = typer.Option(1, help="Only show types with >= this many entities."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of pretty output."),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Show entity-type buckets in the KB ('topics').

    First step in topic browsing: see what *kinds* of facts the memory
    holds (PERSON, ORGANIZATION, CONCEPT, function, …) before drilling
    into specifics with ``thought browse <name>``.
    """
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        result = mem.list_topics(
            scope=scope, owner_id=owner_id,  # type: ignore[arg-type]
            min_count=min_count,
        )
    finally:
        mem.close()
    if json_out:
        console.print_json(data={"topics": result})
        return
    if not result:
        console.print("[dim]no topics — KB is empty[/dim]")
        return
    table = Table(title="Topics in memory", border_style="cyan")
    table.add_column("type", style="bold")
    table.add_column("count", justify="right")
    table.add_column("examples")
    for t in result:
        table.add_row(
            str(t["type"]),
            str(t["count"]),
            ", ".join(t["examples"]) if t["examples"] else "",  # type: ignore[arg-type]
        )
    console.print(table)


@app.command()
def browse(
    name: str = typer.Argument(..., help="Topic name: a type ('PERSON') or an entity ('dessert')."),
    depth: int = typer.Option(1, help="Graph-traversal depth when drilling into an entity."),
    limit: int = typer.Option(20),
    scope: str = typer.Option("all"),
    owner_id: str | None = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Drill into a topic.

    ``name`` is matched against entity-type names first ('PERSON',
    'function', 'CONCEPT' …); if no type matches it's treated as an
    entity name and the PPR-ranked neighbourhood is returned.

    Examples:
        thought browse PERSON           # all known people
        thought browse Acme             # everything connected to 'Acme'
        thought browse desserts --depth 2
    """
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        items = mem.browse_topic(
            name, depth=depth, limit=limit,
            scope=scope, owner_id=owner_id,  # type: ignore[arg-type]
        )
    finally:
        mem.close()
    if json_out:
        console.print_json(data={"items": items})
        return
    if not items:
        err_console.print(f"[red]no matches for[/red] {name!r}")
        raise typer.Exit(1)
    table = Table(
        title=f"Browsing [bold]{name}[/bold] ({items[0]['via']})",
        border_style="cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("type", width=14)
    table.add_column("entity")
    table.add_column("score", justify="right")
    for i, it in enumerate(items, 1):
        score = it.get("score")
        score_str = f"{score:.4f}" if isinstance(score, float) else "—"
        table.add_row(str(i), str(it["type"]), str(it["name"]), score_str)
    console.print(table)


@app.command()
def callers(
    name: str = typer.Argument(..., help="Function or method name."),
    limit: int = typer.Option(10),
    code_file: str | None = typer.Option(
        None, "--file", help="Restrict to this file.",
    ),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Show direct callers of a function/method, ranked by PageRank."""
    from .layers.code import CodeLayer
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        hits = CodeLayer(mem._backend).callers_of(
            name, code_file=code_file, limit=limit,
        )
    finally:
        mem.close()
    if not hits:
        err_console.print(f"[red]no callers found for[/red] {name!r}")
        raise typer.Exit(1)
    table = Table(
        title=f"Callers of [bold]{name}[/bold]",
        border_style="cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("score", justify="right")
    table.add_column("type")
    table.add_column("entity")
    table.add_column("file")
    for i, h in enumerate(hits, 1):
        table.add_row(
            str(i), f"{h.score:.4f}", h.entity.type, h.entity.name,
            h.entity.attrs.get("class") or h.entity.canonical_name,
        )
    console.print(table)


@app.command()
def impact(
    name: str = typer.Argument(..., help="Function or method name."),
    limit: int = typer.Option(20),
    code_file: str | None = typer.Option(None, "--file"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Transitive impact set — what's affected if you change ``name``.

    Seeds Personalized PageRank at ``name`` and walks the call graph
    bidirectionally; returns the highest-scoring affected entities.
    """
    from .layers.code import CodeLayer
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        hits = CodeLayer(mem._backend).impact_set(
            name, code_file=code_file, limit=limit,
        )
    finally:
        mem.close()
    if not hits:
        err_console.print(f"[red]no impact set for[/red] {name!r}")
        raise typer.Exit(1)
    table = Table(
        title=f"Impact set: what's affected by changing [bold]{name}[/bold]",
        border_style="cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("score", justify="right")
    table.add_column("type")
    table.add_column("entity")
    for i, h in enumerate(hits, 1):
        table.add_row(
            str(i), f"{h.score:.4f}", h.entity.type, h.entity.name,
        )
    console.print(table)


@app.command()
def diff(
    from_sha: str = typer.Option(..., "--from", help="Earlier commit SHA."),
    to_sha: str = typer.Option(..., "--to", help="Later commit SHA."),
    code_file: str | None = typer.Option(None, "--file"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Show entities added/removed between two ingested commit SHAs.

    Both SHAs must have been ingested previously via ``thought ingest-git --mode full``.
    """
    from .layers.code import CodeLayer
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        d = CodeLayer(mem._backend).diff(
            from_sha=from_sha, to_sha=to_sha, code_file=code_file,
        )
    finally:
        mem.close()

    def _render(title: str, entities, style: str) -> None:
        if not entities:
            return
        t = Table(title=title, border_style=style)
        t.add_column("type")
        t.add_column("name")
        t.add_column("file")
        for e in entities:
            t.add_row(e.type, e.name, e.attrs.get("file_path", "") or "")
        console.print(t)

    _render(f"Added in {to_sha[:8]}", d["added"], "green")
    _render(f"Removed since {from_sha[:8]}", d["removed"], "red")
    if not (d["added"] or d["removed"]):
        console.print("[dim]no differences[/dim]")


@app.command()
def doctor() -> None:
    """Deep environment health check."""
    import sqlite3
    table = Table(title="thought doctor", border_style="cyan")
    table.add_column("Check", style="bold")
    table.add_column("Result")

    table.add_row("thought-mcp version", __version__)
    table.add_row("python", sys.version.split()[0])
    table.add_row("platform", sys.platform)

    conn = sqlite3.connect(":memory:")
    has_ext = hasattr(conn, "enable_load_extension")
    table.add_row(
        "sqlite enable_load_extension",
        "[green]yes[/green]" if has_ext else
        "[red]NO[/red] (Anaconda?) — install python.org Python or pysqlite3-binary",
    )

    try:
        import sqlite_vec
        ver = getattr(sqlite_vec, "__version__", "unknown")
        table.add_row("sqlite-vec", f"[green]installed[/green] (v{ver})")
        # Try loading.
        if has_ext:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                table.add_row(
                    "sqlite-vec load test",
                    "[green]ok[/green] — fast ANN available",
                )
            except Exception as e:  # pragma: no cover
                table.add_row("sqlite-vec load test", f"[red]failed[/red]: {e}")
    except ImportError:
        table.add_row(
            "sqlite-vec",
            "[red]missing[/red] — pip install 'thought-mcp[sqlite-vec]'",
        )

    for mod, label, hint in [
        ("mcp", "MCP SDK", "pip install 'thought-mcp[mcp]'"),
        ("scipy", "scipy (sparse PageRank)", "pip install scipy"),
        ("numpy", "numpy", "pip install numpy"),
        ("rich", "rich (CLI output)", "pip install rich"),
        ("sentence_transformers", "sentence-transformers (production embedder)",
         "pip install 'thought-mcp[embeddings-local]'"),
        ("spacy", "spaCy (optional NER)", "pip install 'thought-mcp[ner]'"),
        ("onnxruntime", "ONNX Runtime (optional acceleration)",
         "pip install onnxruntime"),
    ]:
        try:
            __import__(mod)
            table.add_row(label, "[green]installed[/green]")
        except ImportError:
            table.add_row(label, f"[dim]missing[/dim] — {hint}")

    console.print(table)


# ---------------------------------------------------------------- hook subcommands

@hook_app.command("recall")
def hook_recall_cmd(
    limit: int = typer.Option(5, help="Max hits to inject into context."),
    scope: str = typer.Option("all"),
    owner_id: str | None = typer.Option(None),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Auto-recall hook for Claude Code's UserPromptSubmit event.

    Reads the hook payload (JSON) from stdin, runs `recall(query=prompt)`,
    emits the result as additionalContext on stdout. Skips injection
    silently when the recall is low-confidence — Claude Code expects the
    hook to be a no-op in that case rather than emitting "no hits found".

    Wire it up with:

        thought hook install --recall

    Or by hand in .claude/settings.json:

        {"hooks": {"UserPromptSubmit": [
          {"hooks": [{"type": "command", "command": "thought hook recall"}]}]}}
    """
    from .hooks.recall import cli_main
    settings = load_settings(_resolve_config(config))
    rc = cli_main(
        db_path=settings.db_path,
        limit=limit, scope=scope, owner_id=owner_id,
        embedder_choice=settings.embedding.choice,
        embedder_dim=settings.embedding.dim,
    )
    raise typer.Exit(rc)


@hook_app.command("write")
def hook_write_cmd(
    mode: str = typer.Option(
        "raw", "--mode",
        help="'raw' ingests turns verbatim (cheap, Jaccard-dedup absorbs noise). "
             "'extract' LLM-extracts durable facts first ($/turn, lower noise).",
    ),
    scope: str = typer.Option("private"),
    owner_id: str | None = typer.Option(None),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Auto-write hook for Claude Code's Stop event.

    Reads the hook payload (JSON) from stdin, picks the last user + assistant
    turns out of the transcript, and ingests them. Idempotent via the
    ingest pipeline's content-sha256 dedup, so replaying a transcript does
    not double-ingest.

    Wire it up with:

        thought hook install --write          # default: --mode raw
        thought hook install --both           # auto-recall + auto-write
    """
    from .hooks.write import cli_main
    if mode not in {"raw", "extract"}:
        err_console.print(f"[red]unknown mode[/red] {mode!r}")
        raise typer.Exit(2)
    settings = load_settings(_resolve_config(config))
    rc = cli_main(
        db_path=settings.db_path,
        mode=mode,  # type: ignore[arg-type]
        scope=scope, owner_id=owner_id,
        embedder_choice=settings.embedding.choice,
        embedder_dim=settings.embedding.dim,
    )
    raise typer.Exit(rc)


@hook_app.command("install")
def hook_install_cmd(
    recall: bool = typer.Option(
        False, "--recall", help="Install the UserPromptSubmit auto-recall hook.",
    ),
    write: bool = typer.Option(
        False, "--write", help="Install the Stop auto-write hook.",
    ),
    both: bool = typer.Option(
        False, "--both", help="Install both hooks (shorthand for --recall --write).",
    ),
    scope: str = typer.Option(
        "project", help="'project' writes ./.claude/settings.json; 'user' writes ~/.claude/settings.json.",
    ),
) -> None:
    """Register thought hooks in Claude Code's settings.json.

    Idempotent — running twice is a no-op. Backs up the original to
    settings.json.thought.bak before write.

    Examples:
        thought hook install --recall            # auto-recall only
        thought hook install --both              # auto-recall + auto-write
        thought hook install --both --scope user # global, all projects
    """
    from .hooks import install as hook_install
    if scope not in {"project", "user"}:
        err_console.print(f"[red]unknown scope[/red] {scope!r}")
        raise typer.Exit(2)
    kinds: list[hook_install.HookKind] = []
    if both or recall:
        kinds.append("recall")
    if both or write:
        kinds.append("write")
    if not kinds:
        err_console.print(
            "[red]specify at least one of --recall / --write / --both[/red]"
        )
        raise typer.Exit(2)
    results = hook_install.install_many(tuple(kinds), scope=scope)  # type: ignore[arg-type]
    table = Table(
        title=f"Hook install ({scope} scope)", border_style="cyan",
    )
    table.add_column("hook", style="bold")
    table.add_column("status")
    table.add_column("path")
    for r in results:
        style = {
            "installed": "green",
            "already_present": "yellow",
            "error": "red",
        }[r.status]
        table.add_row(r.kind, f"[{style}]{r.status}[/{style}]", str(r.path))
        if r.status == "error":
            err_console.print(f"[red]error[/red] ({r.kind}): {r.detail}")
    console.print(table)
    console.print(
        "[dim]restart Claude Code (or any client that reads settings.json) "
        "to pick up the new hooks.[/dim]"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
