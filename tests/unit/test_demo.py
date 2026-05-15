"""Tests for ``thought demo run`` — every audience must complete cleanly."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from thought.cli import app
from thought.demo import (
    _PREFIX,
    cleanup_demo_workspaces,
    run_demo,
)


@pytest.mark.parametrize("kind", ["code", "writer", "legal", "researcher"])
def test_demo_run_audience_passes(kind: str) -> None:
    report = run_demo(kind=kind)  # type: ignore[arg-type]
    failures = [s for s in report.stages if not s.passed]
    assert not failures, "\n".join(
        f"{s.audience}.{s.name}: {s.error}" for s in failures
    )
    assert report.all_passed
    assert report.cleaned_up
    assert report.stages, "no stages ran"


def test_demo_run_all_runs_every_audience() -> None:
    report = run_demo(kind="all")
    audiences = {s.audience for s in report.stages}
    assert audiences == {"code", "writer", "legal", "researcher"}
    failures = [s for s in report.stages if not s.passed]
    assert not failures, "\n".join(
        f"{s.audience}.{s.name}: {s.error}" for s in failures
    )


def test_demo_run_keep_preserves_workspace() -> None:
    report = run_demo(kind="writer", keep=True)
    workspace = Path(report.workspace)
    try:
        assert workspace.exists()
        assert not report.cleaned_up
    finally:
        # Manual cleanup since --keep was on.
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)


def test_cleanup_removes_leftover_workspaces() -> None:
    # Manufacture a leftover.
    leftover = Path(tempfile.mkdtemp(prefix=_PREFIX))
    (leftover / "something.txt").write_text("hello")
    n = cleanup_demo_workspaces()
    assert n >= 1
    assert not leftover.exists()


# ---------------------------------------------------------------- CLI

def test_demo_run_cli(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r = runner.invoke(app, ["demo", "run", "--kind", "writer", "--json"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(r.stdout)
    assert data["all_passed"] is True
    assert data["cleaned_up"] is True
    assert any(s["audience"] == "writer" for s in data["stages"])


def test_demo_run_cli_unknown_kind(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r = runner.invoke(app, ["demo", "run", "--kind", "nope"])
    assert r.exit_code == 2


def test_demo_cleanup_cli(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Manufacture a leftover so cleanup has something to do.
    Path(tempfile.mkdtemp(prefix=_PREFIX))
    runner = CliRunner()
    r = runner.invoke(app, ["demo", "cleanup"])
    assert r.exit_code == 0
