"""Git history walker — shells out to ``git`` (always available).

A thin wrapper over a handful of git subcommands:
- ``rev-parse HEAD`` for the current SHA
- ``log --format=...`` for commit metadata
- ``ls-tree -r <sha>`` for files at a commit
- ``show <sha>:<path>`` for file contents at a commit

We deliberately avoid ``pygit2`` as a hard dep — the C extension is a
real install footprint and the subprocess path is fast enough for the
v0.2 workload (<1k commits / per-commit cost dominated by tree-sitter
parse, not git). Users who want pygit2 can install ``thought-mcp[git]``;
v0.3 wires it as an optional acceleration.
"""
from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Commit:
    sha: str
    author: str
    author_email: str
    author_date: datetime
    subject: str


class GitWalker:
    """Read-only view of a git repository."""

    def __init__(self, repo_path: Path | str) -> None:
        self.repo = Path(repo_path)
        if shutil.which("git") is None:
            raise RuntimeError("git executable not on PATH")
        if not (self.repo / ".git").exists():
            raise ValueError(f"not a git repository: {self.repo}")

    # ---------------------------------------------------- internals

    def _run(self, *args: str) -> str:
        env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
        r = subprocess.run(
            ["git", *args],
            cwd=self.repo, env=env, check=True,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return r.stdout

    def _run_bytes(self, *args: str) -> bytes:
        env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
        r = subprocess.run(
            ["git", *args],
            cwd=self.repo, env=env, check=True,
            capture_output=True,
        )
        return r.stdout

    # ---------------------------------------------------- public

    def head_sha(self) -> str:
        return self._run("rev-parse", "HEAD").strip()

    def iter_commits(self, limit: int | None = None):
        """Yield commits newest → oldest, optionally bounded by ``limit``.

        Uses an ASCII record separator so commit subjects with newlines
        don't break parsing.
        """
        rs = chr(0x1e)  # record separator
        us = chr(0x1f)  # unit separator
        fmt = us.join(["%H", "%an", "%ae", "%aI", "%s"]) + rs
        args = ["log", f"--format={fmt}"]
        if limit is not None:
            args.append(f"-n{limit}")
        out = self._run(*args)
        for record in out.split(rs):
            record = record.strip("\n")
            if not record:
                continue
            parts = record.split(us)
            if len(parts) < 5:
                continue
            sha, author, email, iso, subj = parts[:5]
            yield Commit(
                sha=sha, author=author, author_email=email,
                author_date=datetime.fromisoformat(iso),
                subject=subj,
            )

    def list_files_at(
        self, sha: str, *, pattern: str | None = None,
    ) -> list[str]:
        """Files present in the tree at ``sha``, optionally glob-filtered."""
        out = self._run("ls-tree", "-r", "--name-only", sha)
        files = [line.strip() for line in out.splitlines() if line.strip()]
        if pattern is not None:
            files = [f for f in files if fnmatch.fnmatch(f, pattern)]
        return files

    def read_file_at(self, sha: str, path: str) -> str:
        """Read the contents of ``path`` at commit ``sha``."""
        return self._run("show", f"{sha}:{path}")
