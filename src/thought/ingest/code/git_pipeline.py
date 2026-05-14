"""Git-history-aware ingest pipeline.

Two modes:

- ``mode="snapshot"`` (fast) — ingest the tree at HEAD only. Every entity
  is stamped with the HEAD SHA so later bi-temporal queries can pin to
  that point. This is the right default for "I want to ingest my current
  codebase."

- ``mode="full"`` (slow but expressive) — walk every commit in chronological
  order. For each commit, ingest the file tree fresh, stamping each entity
  with that commit's SHA. The resulting database supports the killer demo
  query: *"what did `auth.middleware` look like at commit X?"* via
  ``CodeLayer.diff(from_sha, to_sha)``.

Full-history ingest is bounded by file count × commits. For mid-size
repos (Python's stdlib-equivalent: 500 files × 1000 commits) this is
~500k file-parses, ~5 minutes with the deterministic embedder.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from ...embeddings.base import Embedder
from ...models import ScopeName
from ...storage.sqlite.backend import SQLiteBackend
from .ast_extractor import detect_language
from .call_graph import build_call_graph
from .git_walker import GitWalker
from .pipeline import CodeIngestPipeline

GitMode = Literal["snapshot", "full"]


@dataclass(frozen=True)
class GitIngestReport:
    head_sha: str
    mode: GitMode
    commits_visited: int
    files_ingested: int
    call_edges: int


class GitIngestPipeline:
    def __init__(
        self,
        *,
        backend: SQLiteBackend,
        embedder: Embedder,
        scope: ScopeName = "shared",
        owner_id: str | None = None,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._scope = scope
        self._owner_id = owner_id

    def ingest_history(
        self,
        repo_path: Path | str,
        *,
        mode: GitMode = "snapshot",
        paths: tuple[str, ...] = ("*.py",),
        since: str | None = None,
        until: str | None = None,
        now: datetime,
        skip_call_graph: bool = False,
    ) -> GitIngestReport:
        walker = GitWalker(Path(repo_path))
        head = walker.head_sha()

        if mode == "snapshot":
            commits = [next(walker.iter_commits(limit=1))]
        else:
            # Full history — oldest first so supersession chains build the right way.
            commits = list(reversed(list(walker.iter_commits())))

        code_pipe = CodeIngestPipeline(
            backend=self._backend, embedder=self._embedder,
            scope=self._scope, owner_id=self._owner_id,
        )

        n_files = 0
        n_call_edges = 0
        for commit in commits:
            files = []
            for pattern in paths:
                files.extend(walker.list_files_at(commit.sha, pattern=pattern))
            files = sorted(set(files))
            for fpath in files:
                lang = detect_language(fpath)
                if lang is None:
                    continue
                content = walker.read_file_at(commit.sha, fpath)
                # We materialise the file in a temp location so the
                # CodeIngestPipeline (which is path-based) can read it. The
                # ``code_file`` column stores the repo-relative path, not the
                # temp path, via ``repo_root=``.
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpd = Path(tmpdir)
                    real_path = tmpd / fpath
                    real_path.parent.mkdir(parents=True, exist_ok=True)
                    real_path.write_text(content, encoding="utf-8")
                    r = code_pipe.ingest_code_file(
                        real_path, commit_sha=commit.sha, now=now,
                        language=lang, repo_root=tmpd,
                    )
                if not skip_call_graph:
                    n_call_edges += build_call_graph(
                        backend=self._backend, file_path=fpath, source=content,
                        language=lang, commit_sha=commit.sha,
                        scope=self._scope, owner_id=self._owner_id,
                        source_ref=r.source_id, now=now,
                    )
                n_files += 1

        return GitIngestReport(
            head_sha=head,
            mode=mode,
            commits_visited=len(commits),
            files_ingested=n_files,
            call_edges=n_call_edges,
        )
