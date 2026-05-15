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
db_app = typer.Typer(
    name="db",
    help="Database lifecycle: size, flush, backup, load, inspect, query.",
)
app.add_typer(db_app, name="db")
view_app = typer.Typer(
    name="view",
    help="Saved Cypher views — named queries that derive new constructs.",
)
app.add_typer(view_app, name="view")
console = Console(stderr=False)
err_console = Console(stderr=True)


def _open_memory(settings: Settings) -> Memory:
    return Memory.open(
        db_path=settings.db_path,
        embedder_choice=settings.embedding.choice,
        embedder_dim=settings.embedding.dim,
        consolidation_enabled=False,
        embedding_cfg=settings.embedding,
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
def schema(
    json_out: bool = typer.Option(False, "--json"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Show entity types + relation types currently in the KB.

    Use this before composing Cypher queries to see what's queryable.
    """
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        data = mem.schema_summary()
    finally:
        mem.close()
    if json_out:
        console.print_json(data=data)
        return
    et = data.get("entity_types") or {}
    rt = data.get("relation_types") or {}
    if not et and not rt:
        console.print("[dim]KB is empty — ingest some facts and try again.[/dim]")
        return
    if et:
        t = Table(title="Entity types", show_header=False, border_style="cyan")
        t.add_column(style="bold")
        t.add_column(justify="right")
        for k, v in et.items():
            t.add_row(k, str(v))
        console.print(t)
    if rt:
        t = Table(title="Relation types", show_header=False, border_style="cyan")
        t.add_column(style="bold")
        t.add_column(justify="right")
        for k, v in rt.items():
            t.add_row(k, str(v))
        console.print(t)


@app.command()
def query(
    cypher_source: str = typer.Argument(..., help="Cypher query string."),
    as_of: str | None = typer.Option(None, "--as-of", help="ISO date for time-travel query."),
    scope: str = typer.Option("all"),
    owner_id: str | None = typer.Option(None),
    explain: bool = typer.Option(False, "--explain", help="Print the emitted SQL before running."),
    json_out: bool = typer.Option(False, "--json"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Run a Cypher query against the KB.

    Supported subset: MATCH (var:Type {prop:val}) [-[:REL]-> (var:Type)] ...
    WHERE expr AND expr ... RETURN identlist [AS_OF 'date'] [LIMIT N].

    See ``thought schema`` first to see what entity types and relation types
    are available.
    """
    from .query import cypher as cypher_mod
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        # Allow --as-of as a flag in addition to inline AS_OF in the cypher.
        if as_of and "AS_OF" not in cypher_source.upper():
            cypher_source = f'{cypher_source} AS_OF "{as_of}"'
        try:
            q = cypher_mod.parse(cypher_source)
            from .models import ScopeFilter
            sql, params, columns = cypher_mod.compile_to_sql(
                q, scope_filter=ScopeFilter(scope=scope, owner_id=owner_id),  # type: ignore[arg-type]
            )
        except cypher_mod.CypherError as e:
            err_console.print(f"[red]Cypher error:[/red] {e}")
            raise typer.Exit(2) from e
        if explain:
            console.print(f"[dim]SQL:[/dim] {sql}")
            console.print(f"[dim]params:[/dim] {params}")
        rows = mem._backend._conn.execute(sql, params).fetchall()
    finally:
        mem.close()

    if json_out:
        import json as _json
        out_rows = []
        for r in rows:
            d: dict[str, object] = {}
            for i, c in enumerate(columns):
                v = r[i]
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    try:
                        d[c] = _json.loads(v)
                        continue
                    except _json.JSONDecodeError:
                        pass
                d[c] = v
            out_rows.append(d)
        console.print_json(data={"rows": out_rows})
        return
    if not rows:
        console.print("[dim]no rows[/dim]")
        return
    t = Table(title="query", border_style="cyan")
    for c in columns:
        t.add_column(c)
    for r in rows:
        t.add_row(*[str(v) if v is not None else "" for v in r])
    console.print(t)


@view_app.command("save")
def view_save_cmd(
    name: str = typer.Argument(..., help="Saved view name (identifier-shape)."),
    cypher_source: str = typer.Argument(..., help="Cypher query to save."),
    replace: bool = typer.Option(False, "--replace", help="Overwrite if it exists."),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Save a Cypher query as a named view.

    Views re-evaluate against the live KB every time you call ``view run``.
    """
    from .query import views as views_mod
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        try:
            views_mod.save_view(mem, name, cypher_source, replace=replace)
        except (views_mod.ViewError, ValueError) as e:
            err_console.print(f"[red]{e}[/red]")
            raise typer.Exit(2) from e
    finally:
        mem.close()
    console.print(f"  [green][ok][/green] saved view [bold]{name}[/bold]")


@view_app.command("list")
def view_list_cmd(
    json_out: bool = typer.Option(False, "--json"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """List all saved views."""
    from .query import views as views_mod
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        data = views_mod.list_views(mem)
    finally:
        mem.close()
    if json_out:
        console.print_json(data={"views": data})
        return
    if not data:
        console.print("[dim]no saved views yet.[/dim]")
        return
    t = Table(title="Saved views", border_style="cyan")
    t.add_column("name", style="bold")
    t.add_column("cypher")
    t.add_column("last run")
    for v in data:
        t.add_row(
            str(v["name"]),
            str(v["cypher"])[:60] + ("…" if len(str(v["cypher"])) > 60 else ""),
            str(v.get("last_run_at") or "—"),
        )
    console.print(t)


@view_app.command("show")
def view_show_cmd(
    name: str = typer.Argument(...),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Print a saved view's definition."""
    from .query import views as views_mod
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        data = views_mod.show_view(mem, name)
    finally:
        mem.close()
    if not data:
        err_console.print(f"[red]no saved view[/red] {name!r}")
        raise typer.Exit(1)
    console.print_json(data=data)


@view_app.command("run")
def view_run_cmd(
    name: str = typer.Argument(...),
    scope: str = typer.Option("all"),
    owner_id: str | None = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Run a saved view; results are pull-evaluated against the live KB."""
    from .query import views as views_mod
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        try:
            rows = views_mod.run_view(mem, name, scope=scope, owner_id=owner_id)
        except views_mod.ViewError as e:
            err_console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
    finally:
        mem.close()
    if json_out:
        console.print_json(data={"rows": rows})
        return
    if not rows:
        console.print("[dim]no rows[/dim]")
        return
    cols = list(rows[0].keys())
    t = Table(title=f"view: {name}", border_style="cyan")
    for c in cols:
        t.add_column(c)
    for r in rows:
        t.add_row(*[str(r.get(c, "")) for c in cols])
    console.print(t)


@view_app.command("delete")
def view_delete_cmd(
    name: str = typer.Argument(...),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    from .query import views as views_mod
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        deleted = views_mod.delete_view(mem, name)
    finally:
        mem.close()
    if deleted:
        console.print(f"  [green][ok][/green] deleted [bold]{name}[/bold]")
    else:
        err_console.print(f"[yellow]no saved view named[/yellow] {name!r}")
        raise typer.Exit(1)


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Natural-language question."),
    scope: str = typer.Option("all"),
    owner_id: str | None = typer.Option(None),
    explain: bool = typer.Option(False, "--explain", help="Print emitted Cypher + SQL before results."),
    no_fallback: bool = typer.Option(False, "--no-fallback", help="Fail loudly instead of falling back to recall."),
    save_as: str | None = typer.Option(None, "--save-as", help="Save successful translation as a named view."),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Ask in English; THOUGHT translates to Cypher via your configured LLM.

    The provider is picked from [llm] provider in your thought.toml:
    anthropic / ollama / lmstudio / openai-compat / openai. For Ollama /
    LM Studio users this is zero-API-cost — the LLM is already on your box.

    Bad translations degrade gracefully to a plain `recall(question)` so
    you always get something. Pass --no-fallback to fail instead.
    """
    from .query import ask as ask_mod
    from .query import views as views_mod
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        result = ask_mod.ask(
            mem, question,
            llm_cfg=settings.llm,
            scope=scope, owner_id=owner_id,
            no_fallback=no_fallback,
        )
        if save_as and result.cypher and not result.fallback_used:
            try:
                views_mod.save_view(mem, save_as, result.cypher, replace=True)
            except views_mod.ViewError as e:
                err_console.print(f"[red]could not save view: {e}[/red]")
    finally:
        mem.close()

    if result.error:
        err_console.print(f"[red]{result.error}[/red]")
        raise typer.Exit(1)
    if explain and result.cypher:
        console.print(f"[dim]Cypher:[/dim] {result.cypher}")
    if explain and result.sql:
        console.print(f"[dim]SQL:[/dim] {result.sql}")
    if result.fallback_used:
        console.print(f"[yellow]fell back to recall:[/yellow] {result.fallback_reason}")
    if not result.rows:
        console.print("[dim]no rows[/dim]")
        return
    cols = list(result.rows[0].keys())
    t = Table(title=f"ask: {question}", border_style="cyan")
    for c in cols:
        t.add_column(c)
    for r in result.rows:
        t.add_row(*[str(r.get(c, "")) for c in cols])
    console.print(t)
    if save_as and result.cypher and not result.fallback_used:
        console.print(f"[dim]saved as view '{save_as}'[/dim]")


@app.command("ollama-setup")
def ollama_setup_cmd(
    host: str = typer.Option(
        "http://localhost:11434", "--host",
        help="Ollama daemon URL.",
    ),
    model: str | None = typer.Option(
        None, "--model", help="Embedding model to suggest (default: pick from installed).",
    ),
    write_config: bool = typer.Option(
        False, "--write", help="Rewrite thought.toml to point at Ollama.",
    ),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Check Ollama, list installed models, print a thought.toml snippet.

    Doesn't pull models for you — printing the ``ollama pull`` command is
    the user's signal to download something.
    """
    from .hooks.setup import KNOWN_OLLAMA_EMBED_MODELS, TOML_OLLAMA_SNIPPET, ping_ollama
    result = ping_ollama(host)
    if not result.reachable:
        err_console.print(f"[red]{result.error}[/red]")
        raise typer.Exit(1)
    if not result.models:
        console.print(
            f"[yellow]Ollama is running at {host}, but no models are installed.[/yellow]\n"
            f"  Try: [bold]ollama pull nomic-embed-text[/bold]"
        )
        raise typer.Exit(1)
    table = Table(title=f"Models installed at {host}", border_style="cyan")
    table.add_column("model", style="bold")
    table.add_column("embedding-capable?")
    for m in result.models:
        is_embed = any(km in m for km in KNOWN_OLLAMA_EMBED_MODELS)
        table.add_row(m, "[green]yes[/green]" if is_embed else "[dim]no[/dim]")
    console.print(table)
    chosen_model = model or result.suggested_model
    if not chosen_model:
        console.print(
            "[yellow]No embedding-capable model installed.[/yellow]\n"
            "  Try: [bold]ollama pull nomic-embed-text[/bold]"
        )
        raise typer.Exit(1)
    snippet = TOML_OLLAMA_SNIPPET.format(host=host, model=chosen_model)
    if write_config:
        cfg = _resolve_config(config)
        cfg.write_text(f'db_path = ".thought/thought.db"\n\n{snippet}', encoding="utf-8")
        console.print(f"  [green][ok][/green] wrote {cfg}")
    else:
        console.print("[dim]Add this to your thought.toml:[/dim]")
        console.print(snippet)


@app.command("lmstudio-setup")
def lmstudio_setup_cmd(
    base_url: str = typer.Option(
        "http://localhost:1234/v1", "--base-url",
        help="LM Studio server URL.",
    ),
    model: str | None = typer.Option(
        None, "--model", help="Embedding model to suggest (default: first loaded model).",
    ),
    write_config: bool = typer.Option(
        False, "--write", help="Rewrite thought.toml to point at LM Studio.",
    ),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Check LM Studio, list loaded models, print a thought.toml snippet."""
    from .hooks.setup import TOML_LMSTUDIO_SNIPPET, ping_lmstudio
    result = ping_lmstudio(base_url)
    if not result.reachable:
        err_console.print(f"[red]{result.error}[/red]")
        raise typer.Exit(1)
    if not result.models:
        console.print(
            f"[yellow]LM Studio is reachable at {base_url}, but no models are loaded.[/yellow]\n"
            f"  Load an embedding model in the LM Studio UI first."
        )
        raise typer.Exit(1)
    table = Table(title=f"Models loaded at {base_url}", border_style="cyan")
    table.add_column("model id", style="bold")
    for m in result.models:
        table.add_row(m)
    console.print(table)
    chosen_model = model or result.models[0]
    snippet = TOML_LMSTUDIO_SNIPPET.format(base_url=base_url, model=chosen_model)
    if write_config:
        cfg = _resolve_config(config)
        cfg.write_text(f'db_path = ".thought/thought.db"\n\n{snippet}', encoding="utf-8")
        console.print(f"  [green][ok][/green] wrote {cfg}")
    else:
        console.print("[dim]Add this to your thought.toml:[/dim]")
        console.print(snippet)


@app.command()
def reembed(
    to: str = typer.Option(
        ..., "--to",
        help="Target embedder choice: deterministic | minilm | ollama | lmstudio | openai-compat | openai.",
    ),
    dim: int | None = typer.Option(
        None, "--dim", help="Override target dim (default: keep current).",
    ),
    batch_size: int = typer.Option(32, "--batch-size"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Re-embed every entity through a different embedder.

    Lets you start with ``deterministic`` for cheap setup and upgrade to a
    production embedder (Ollama, MiniLM, etc.) later without re-ingesting
    from source. Doesn't touch entity rows / edges / sources — only the
    ``embeddings`` table.

    Examples:
        thought reembed --to ollama
        thought reembed --to minilm --dim 384
    """
    settings = load_settings(_resolve_config(config))
    # Build an override embedding config so the new embedder picks up the
    # user's [embedding] Ollama / LM Studio / OpenAI-compat fields.
    mem = _open_memory(settings)
    try:
        # Count first for the progress bar.
        n_rows = int(mem._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM entities WHERE valid_until IS NULL"
        ).fetchone()["n"])
        with Progress(
            SpinnerColumn(), TextColumn("[bold]{task.description}"),
            BarColumn(), MofNCompleteColumn(),
            console=err_console, transient=True,
        ) as prog:
            task = prog.add_task(f"reembedding via {to}", total=n_rows)
            result = mem.reembed_to(
                to,
                new_dim=dim,
                embedding_cfg=settings.embedding,
                batch_size=batch_size,
                progress=lambda n: prog.advance(task, n),
            )
    finally:
        mem.close()
    table = Table(title="Reembed summary", show_header=False, border_style="cyan")
    table.add_column(style="bold")
    table.add_column(justify="right")
    for k, v in result.items():
        table.add_row(k, str(v))
    console.print(table)
    console.print(
        "[dim]Tip: update [embedding] choice in thought.toml to make the new "
        "embedder the default on next open.[/dim]"
    )


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
        embedding_cfg=settings.embedding,
        llm_cfg=settings.llm,
    )
    raise typer.Exit(rc)


@db_app.command("size")
def db_size_cmd(
    json_out: bool = typer.Option(False, "--json"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Show on-disk size of the DB file + WAL/SHM sidecars + entity counts."""
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        data = mem.db_size()
    finally:
        mem.close()
    if json_out:
        console.print_json(data=data)
        return
    table = Table(title="Database on-disk size", show_header=False, border_style="cyan")
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("path", str(data["path"]))
    table.add_row("main (db)", _human_bytes(int(data["main"])))
    table.add_row("wal", _human_bytes(int(data["wal"])))
    table.add_row("shm", _human_bytes(int(data["shm"])))
    table.add_row("total", f"[bold]{_human_bytes(int(data['total_bytes']))}[/bold]")
    table.add_row("entities (current/total)",
                  f"{data['entities_current']} / {data['entities_total']}")
    table.add_row("edges", str(data["edges"]))
    table.add_row("sources", str(data["sources"]))
    console.print(table)


def _human_bytes(n: int) -> str:
    """Render a byte count like Linux's du -h."""
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n_f = n / 1024
        if n_f < 1024 or unit == "TB":
            return f"{n_f:.1f} {unit}"
        n = int(n_f)
    return f"{n} B"  # unreachable


def _parse_date(s: str | None) -> datetime | None:
    """Accept ISO date or datetime; UTC if no tz given."""
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        from datetime import UTC as _UTC
        dt = dt.replace(tzinfo=_UTC)
    return dt


@db_app.command("flush")
def db_flush_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
    before: str | None = typer.Option(
        None, "--before", help="Only delete entities created/valid/learned BEFORE this date (ISO).",
    ),
    since: str | None = typer.Option(
        None, "--since", help="Only delete entities created/valid/learned AT OR AFTER this date (ISO).",
    ),
    time_axis: str = typer.Option(
        "created", "--time-axis",
        help="Which timestamp to filter on: created | valid | learned.",
    ),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Wipe the KB. Destructive.

    Without date flags: full wipe of all KB tables.
    With ``--before X`` and/or ``--since X``: row-level delete of entities whose
    chosen timestamp falls outside the kept range. Cascades to edges + triples.

    Always auto-backs-up to ``<db_path>.bak.<timestamp>`` before any destructive
    operation, so you can roll forward via ``thought db load`` if you regret it.
    """
    if time_axis not in {"created", "valid", "learned"}:
        err_console.print(f"[red]unknown --time-axis[/red] {time_axis!r}")
        raise typer.Exit(2)
    settings = load_settings(_resolve_config(config))
    bounds_str = ""
    if before:
        bounds_str += f" before {before}"
    if since:
        bounds_str += f" since {since}"
    bounds_str = bounds_str or " ALL DATA"
    if not yes:
        confirm = Prompt.ask(
            f"[yellow]Flush KB at [bold]{settings.db_path}[/bold] —{bounds_str} "
            f"(axis: {time_axis})? [y/N][/yellow]",
            default="N",
        )
        if confirm.strip().lower() not in {"y", "yes"}:
            console.print("[dim]aborted[/dim]")
            return
    # Auto-backup before destructive op.
    bak = Path(settings.db_path).with_suffix(
        Path(settings.db_path).suffix
        + f".bak.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )
    mem = _open_memory(settings)
    try:
        mem.backup_to(bak, force=True)
        console.print(f"  [dim]auto-backup → {bak}[/dim]")
        result = mem.flush(
            confirm=True,
            before=_parse_date(before),
            since=_parse_date(since),
            time_axis=time_axis,  # type: ignore[arg-type]
        )
    finally:
        mem.close()
    table = Table(title="Flush summary", show_header=False, border_style="cyan")
    table.add_column(style="bold")
    table.add_column(justify="right")
    for k, v in result.items():
        table.add_row(k, str(v))
    console.print(table)
    console.print(f"[green]auto-backup preserved at[/green] {bak}")


@db_app.command("backup")
def db_backup_cmd(
    file: Path = typer.Argument(..., help="Path to write the snapshot to."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
    before: str | None = typer.Option(None, "--before"),
    since: str | None = typer.Option(None, "--since"),
    time_axis: str = typer.Option("created", "--time-axis"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Snapshot the current DB to ``file``. Works while the server is running.

    With ``--before`` / ``--since``, the snapshot only contains entities whose
    chosen timestamp falls within the requested range.
    """
    if time_axis not in {"created", "valid", "learned"}:
        err_console.print(f"[red]unknown --time-axis[/red] {time_axis!r}")
        raise typer.Exit(2)
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        try:
            bytes_written = mem.backup_to(
                file, force=force,
                before=_parse_date(before),
                since=_parse_date(since),
                time_axis=time_axis,  # type: ignore[arg-type]
            )
        except FileExistsError as e:
            err_console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
    finally:
        mem.close()
    console.print(
        f"  [green][ok][/green] wrote [bold]{_human_bytes(bytes_written)}[/bold] to {file}"
    )


@db_app.command("load")
def db_load_cmd(
    file: Path = typer.Argument(..., help="Snapshot file to load."),
    yes: bool = typer.Option(False, "--yes", "-y"),
    merge: bool = typer.Option(
        False, "--merge",
        help="Merge into the existing DB instead of replacing it.",
    ),
    before: str | None = typer.Option(None, "--before"),
    since: str | None = typer.Option(None, "--since"),
    time_axis: str = typer.Option("created", "--time-axis"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Load a snapshot. Replaces the current DB unless ``--merge`` is set.

    Replace mode auto-backs-up the current DB to ``<db_path>.bak.<timestamp>``
    before clobbering.
    """
    if time_axis not in {"created", "valid", "learned"}:
        err_console.print(f"[red]unknown --time-axis[/red] {time_axis!r}")
        raise typer.Exit(2)
    if not file.exists():
        err_console.print(f"[red]file not found[/red]: {file}")
        raise typer.Exit(1)
    settings = load_settings(_resolve_config(config))
    if not yes:
        action = "merge into" if merge else "REPLACE"
        confirm = Prompt.ask(
            f"[yellow]{action} {settings.db_path} from {file}? [y/N][/yellow]",
            default="N",
        )
        if confirm.strip().lower() not in {"y", "yes"}:
            console.print("[dim]aborted[/dim]")
            return
    mem = _open_memory(settings)
    try:
        try:
            result = mem.load_from(
                file, merge=merge,
                before=_parse_date(before),
                since=_parse_date(since),
                time_axis=time_axis,  # type: ignore[arg-type]
            )
        except (FileNotFoundError, ValueError) as e:
            err_console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
    finally:
        mem.close()

    if result["action"] == "merge":
        table = Table(title="Merge summary", show_header=False, border_style="cyan")
        table.add_column(style="bold")
        table.add_column(justify="right")
        for k, v in result.items():
            if k in {"action", "source"}:
                continue
            table.add_row(k, str(v))
        console.print(table)
        return

    # Replace mode: do the actual file swap here (Memory is closed).
    import shutil
    db_path = Path(settings.db_path)
    bak = db_path.with_suffix(
        db_path.suffix + f".bak.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )
    if db_path.exists():
        shutil.move(str(db_path), str(bak))
        # Also stash any WAL/SHM sidecars so the swap is clean.
        for ext in ("-wal", "-shm"):
            sidecar = Path(str(db_path) + ext)
            if sidecar.exists():
                sidecar.unlink()
    shutil.copy2(str(file), str(db_path))
    # Re-open + run migrations so older snapshots upgrade cleanly.
    mem2 = _open_memory(settings)
    mem2.close()
    console.print(
        f"  [green][ok][/green] loaded {file} → {db_path}\n"
        f"  [dim]previous DB preserved at {bak}[/dim]"
    )


@db_app.command("inspect")
def db_inspect_cmd(
    file: Path = typer.Argument(..., help="Snapshot file to inspect."),
    schema: bool = typer.Option(
        False, "--schema", help="Also show entity-type + relation-type counts.",
    ),
    json_out: bool = typer.Option(False, "--json"),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Show counts (and optional schema breakdown) of a backup file.

    Doesn't touch the active DB — useful for *"is this snapshot worth loading?"*
    before running ``thought db load``.
    """
    if not file.exists():
        err_console.print(f"[red]file not found[/red]: {file}")
        raise typer.Exit(1)
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        try:
            data = mem.inspect_file(file, include_schema=schema)
        except (FileNotFoundError, ValueError) as e:
            err_console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
    finally:
        mem.close()
    if json_out:
        console.print_json(data=data)
        return
    table = Table(title=f"Snapshot: {file}", show_header=False, border_style="cyan")
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("size", _human_bytes(int(data["size_bytes"])))
    table.add_row("schema_version", str(data["schema_version"]))
    table.add_row("entities (current/total)",
                  f"{data['entities_current']} / {data['entities_total']}")
    table.add_row("edges", str(data["edges"]))
    table.add_row("contradictions", str(data["contradictions"]))
    table.add_row("sources", str(data["sources"]))
    console.print(table)
    if schema:
        et = data.get("entity_types") or {}
        if et:
            t = Table(title="entity types", show_header=False, border_style="dim")
            t.add_column(style="bold")
            t.add_column(justify="right")
            for k, v in et.items():
                t.add_row(k, str(v))
            console.print(t)
        rt = data.get("relation_types") or {}
        if rt:
            t = Table(title="relation types", show_header=False, border_style="dim")
            t.add_column(style="bold")
            t.add_column(justify="right")
            for k, v in rt.items():
                t.add_row(k, str(v))
            console.print(t)


@hook_app.command("context")
def hook_context_cmd(
    view_name: str = typer.Option(
        "__startup__", "--view-name",
        help="Saved view to evaluate and inject at session start.",
    ),
    limit: int = typer.Option(20, help="Max rows from the view to inject."),
    config: Path = typer.Option(Path("thought.toml")),
) -> None:
    """Auto-context hook for Claude Code's SessionStart event.

    Loads the named saved view (default: ``__startup__``) and emits its
    result rows as additionalContext. Lets users designate "always know
    this" facts that surface on every new session.

    v0.4 ships the hook entrypoint and CLI install plumbing. Saved-views
    integration arrives with the Cypher query layer; until then this hook
    runs cleanly and emits a no-op skip when the named view doesn't exist.
    """
    import json as _json
    import sys as _sys

    try:
        raw = _sys.stdin.read()
        _ = _json.loads(raw) if raw.strip() else {}
    except _json.JSONDecodeError:
        # Don't surface as a hook error.
        return
    settings = load_settings(_resolve_config(config))
    mem = _open_memory(settings)
    try:
        # If the view doesn't exist yet, silently skip — the hook MUST NOT
        # error out at session-start time.
        try:
            row = mem._backend._conn.execute(  # type: ignore[attr-defined]
                "SELECT cypher_source FROM saved_views WHERE name = ?",
                (view_name,),
            ).fetchone()
        except Exception:
            row = None  # saved_views table doesn't exist yet (pre-v0.5).
        if row is None:
            err_console.print(
                f"[dim]thought hook context: no saved view {view_name!r}; skipping[/dim]"
            )
            return
        # Cypher execution lands in v0.5; until then echo the view's text so
        # users can verify the hook is wired up.
        msg = f"--- thought context view: {view_name} ---\n{row['cypher_source'][:1000]}"
        console.print(_json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": msg[: limit * 200],
            }
        }))
    finally:
        mem.close()


@hook_app.command("install")
def hook_install_cmd(
    recall: bool = typer.Option(
        False, "--recall", help="Install the UserPromptSubmit auto-recall hook.",
    ),
    write: bool = typer.Option(
        False, "--write", help="Install the Stop auto-write hook.",
    ),
    context: bool = typer.Option(
        False, "--context", help="Install the SessionStart auto-context hook.",
    ),
    both: bool = typer.Option(
        False, "--both", help="Install --recall and --write (shorthand).",
    ),
    scope: str = typer.Option(
        "project", help="'project' writes ./.claude/settings.json; 'user' writes ~/.claude/settings.json.",
    ),
) -> None:
    """Register thought hooks in Claude Code's settings.json.

    Idempotent — running twice is a no-op. Backs up the original to
    settings.json.thought.bak before write.

    Examples:
        thought hook install --recall              # auto-recall only
        thought hook install --both                # auto-recall + auto-write
        thought hook install --both --context      # also SessionStart
        thought hook install --both --scope user   # global, all projects
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
    if context:
        kinds.append("context")
    if not kinds:
        err_console.print(
            "[red]specify at least one of --recall / --write / --both / --context[/red]"
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
