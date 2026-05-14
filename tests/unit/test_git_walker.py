"""Tests for the git walker — Phase 3.

Uses a synthetic temporary git repo so we don't depend on the THOUGHT
repo's actual history (which would make the test non-hermetic). The
walker shells out to ``git`` (always available on dev machines + CI runners).
"""
from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from thought.embeddings.deterministic import DeterministicEmbedder
from thought.ingest.code.git_pipeline import GitIngestPipeline
from thought.ingest.code.git_walker import GitWalker
from thought.storage.sqlite.backend import SQLiteBackend


def _git(repo: Path, *args: str, env: dict | None = None) -> str:
    """Run a git command in ``repo`` and return stdout. Raises on failure."""
    full_env = {
        "GIT_AUTHOR_NAME": "Test User",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test User",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        # Force English output so we can parse it.
        "LC_ALL": "C", "LANG": "C",
        "GIT_AUTHOR_DATE": "2026-01-01T12:00:00+0000",
        "GIT_COMMITTER_DATE": "2026-01-01T12:00:00+0000",
    }
    if env:
        full_env.update(env)
    # Inherit PATH so the system git is findable.
    import os
    full_env = {**os.environ, **full_env}
    r = subprocess.run(
        ["git", *args], cwd=repo, env=full_env,
        check=True, capture_output=True, text=True,
    )
    return r.stdout


@pytest.fixture()
def tiny_repo(tmp_path):
    """Three-commit synthetic Python repo for walker tests."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = tmp_path / "tinyrepo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "commit.gpgsign", "false")

    auth = repo / "auth.py"

    auth.write_text(
        'def login(user):\n    return True\n',
        encoding="utf-8",
    )
    _git(repo, "add", "auth.py")
    _git(repo, "commit", "-q", "-m", "initial",
         env={"GIT_AUTHOR_DATE": "2026-01-01T12:00:00+0000",
              "GIT_COMMITTER_DATE": "2026-01-01T12:00:00+0000"})
    sha1 = _git(repo, "rev-parse", "HEAD").strip()

    auth.write_text(
        'def login(user, password):\n    return True\n\n'
        'def logout(user):\n    return None\n',
        encoding="utf-8",
    )
    _git(repo, "add", "auth.py")
    _git(repo, "commit", "-q", "-m", "add logout + password",
         env={"GIT_AUTHOR_DATE": "2026-02-01T12:00:00+0000",
              "GIT_COMMITTER_DATE": "2026-02-01T12:00:00+0000"})
    sha2 = _git(repo, "rev-parse", "HEAD").strip()

    (repo / "users.py").write_text(
        'def lookup(uid):\n    return uid\n',
        encoding="utf-8",
    )
    _git(repo, "add", "users.py")
    _git(repo, "commit", "-q", "-m", "add users module",
         env={"GIT_AUTHOR_DATE": "2026-03-01T12:00:00+0000",
              "GIT_COMMITTER_DATE": "2026-03-01T12:00:00+0000"})
    sha3 = _git(repo, "rev-parse", "HEAD").strip()

    return {"repo": repo, "shas": [sha1, sha2, sha3]}


# ----------------------------------------------------------- GitWalker

def test_git_walker_head_sha(tiny_repo):
    w = GitWalker(tiny_repo["repo"])
    assert w.head_sha() == tiny_repo["shas"][-1]


def test_git_walker_iter_commits_returns_newest_first(tiny_repo):
    w = GitWalker(tiny_repo["repo"])
    commits = list(w.iter_commits(limit=10))
    assert len(commits) == 3
    assert commits[0].sha == tiny_repo["shas"][-1]
    assert commits[-1].sha == tiny_repo["shas"][0]


def test_git_walker_files_at_commit(tiny_repo):
    w = GitWalker(tiny_repo["repo"])
    files_at_first = w.list_files_at(tiny_repo["shas"][0], pattern="*.py")
    assert files_at_first == ["auth.py"]
    files_at_third = w.list_files_at(tiny_repo["shas"][2], pattern="*.py")
    assert set(files_at_third) == {"auth.py", "users.py"}


def test_git_walker_read_file_at_commit(tiny_repo):
    w = GitWalker(tiny_repo["repo"])
    first_auth = w.read_file_at(tiny_repo["shas"][0], "auth.py")
    assert "password" not in first_auth  # password param added in commit 2
    second_auth = w.read_file_at(tiny_repo["shas"][1], "auth.py")
    assert "password" in second_auth


# ----------------------------------------------------------- GitIngestPipeline

def test_git_pipeline_snapshot_mode_ingests_head_only(tiny_repo, tmp_path):
    backend = SQLiteBackend(tmp_path / "g.db")
    backend.migrate()
    pipe = GitIngestPipeline(
        backend=backend, embedder=DeterministicEmbedder(dim=64),
    )
    r = pipe.ingest_history(
        tiny_repo["repo"], mode="snapshot", paths=("*.py",),
        now=datetime.now(UTC),
    )
    assert r.commits_visited == 1
    assert r.files_ingested == 2  # auth.py + users.py at HEAD

    # Every code entity should be stamped with the HEAD SHA.
    rows = backend._conn.execute(
        "SELECT DISTINCT code_commit_sha FROM entities "
        "WHERE code_commit_sha IS NOT NULL"
    ).fetchall()
    shas = {r["code_commit_sha"] for r in rows}
    assert shas == {tiny_repo["shas"][-1]}
    backend.close()


def test_git_pipeline_full_history_walks_every_commit(tiny_repo, tmp_path):
    backend = SQLiteBackend(tmp_path / "g.db")
    backend.migrate()
    pipe = GitIngestPipeline(
        backend=backend, embedder=DeterministicEmbedder(dim=64),
    )
    r = pipe.ingest_history(
        tiny_repo["repo"], mode="full", paths=("*.py",),
        now=datetime.now(UTC),
    )
    assert r.commits_visited == 3
    # Every commit's snapshot should leave a code_commit_sha trace.
    shas = {
        row["code_commit_sha"] for row in backend._conn.execute(
            "SELECT DISTINCT code_commit_sha FROM entities "
            "WHERE code_commit_sha IS NOT NULL"
        ).fetchall()
    }
    assert shas == set(tiny_repo["shas"])
    backend.close()


def test_git_pipeline_full_history_lets_diff_work(tiny_repo, tmp_path):
    """End-to-end: ``CodeLayer.diff(from=sha1, to=sha2)`` returns the
    function that was added between commit 1 and commit 2.
    """
    from thought.layers.code import CodeLayer
    backend = SQLiteBackend(tmp_path / "g.db")
    backend.migrate()
    pipe = GitIngestPipeline(
        backend=backend, embedder=DeterministicEmbedder(dim=64),
    )
    pipe.ingest_history(
        tiny_repo["repo"], mode="full", paths=("*.py",),
        now=datetime.now(UTC),
    )
    code = CodeLayer(backend)
    diff = code.diff(
        from_sha=tiny_repo["shas"][0], to_sha=tiny_repo["shas"][1],
        code_file="auth.py",
    )
    added_names = {e.name for e in diff["added"]}
    assert "logout" in added_names  # logout was added in commit 2
    backend.close()
